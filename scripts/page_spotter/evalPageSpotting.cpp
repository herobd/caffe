//g++ -std=c++11 -fopenmp gwdataset.cpp evalPageSpotting.cpp -lcaffe -lglog -l:libopencv_core.so.3.1 -l:libopencv_imgcodecs.so.3.1 -l:libopencv_imgproc.so.3.1 -l:libopencv_highgui.so.3.1 -lprotobuf -lboost_system -I ../include/ -L ../build/lib/ -o evalPageSpotting
#define CPU_ONLY
#include <caffe/caffe.hpp>
#include "caffe/util/io.hpp"
#include <opencv2/core.hpp>
#ifndef OPENCV2
#include <opencv2/imgcodecs.hpp>
#endif
//#else
#include <opencv2/highgui.hpp>
//#endif
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>


#include "cnn_featurizer.h"
#include "cnn_spotter.h"
#include "BBPredictor.h"
using namespace caffe;
using namespace std;


int sort_xxx(const void *x, const void *y) {
    if (*(int*)x > *(int*)y) return 1;
    else if (*(int*)x < *(int*)y) return -1;
    else return 0;
}

void featurizeDivided(vector<CNNFeaturizer>& featurizers, cv::Mat pageIm, int edgeRegion, int windowSize, vector< vector<vector<cv::Mat>*> >& pageQFeatures,  vector< pair<cv::Rect,cv::Rect> >& pagePullDstRects)
{
    for (auto& fv : pageQFeatures)
        for (vector<cv::Mat>* ptr : fv)
            if (ptr != NULL)
                delete ptr;
    //int edgeRegion=100;
    //int windowSize=600;
    int numHorz = ceil(pageIm.cols/(windowSize+0.0));
    int numVert = ceil(pageIm.rows/(windowSize+0.0));
    pageQFeatures.resize(numHorz*numVert);
    pagePullDstRects.resize(numHorz*numVert);
    for (int i=0; i<numHorz*numVert; i++)
        pageQFeatures[i].resize(featurizers.size());


    for (int h=0; h<numHorz; h++)
    {
        for (int v=0; v<numVert; v++)
        {
            int srcX = max(0,h*windowSize-edgeRegion);
            int srcY = max(0,v*windowSize-edgeRegion);
            cv::Rect dst(h*windowSize,v*windowSize,min(windowSize,pageIm.cols-h*windowSize),min(windowSize,pageIm.rows-v*windowSize));
            cv::Rect src(srcX,srcY,min(windowSize+edgeRegion+dst.x-srcX,pageIm.cols-srcX),min(windowSize+edgeRegion+dst.y-srcY,pageIm.rows-srcY));
            cv::Rect pull(dst.x-srcX,dst.y-srcY,dst.width,dst.height);

            pagePullDstRects[h*numVert + v] = make_pair(pull,dst);

            cv::Mat qPage = pageIm(src);

            //pad
            int addCol = (16 - qPage.cols%16)%16;
            int addRow = (16 - qPage.rows%16)%16;
            if (addCol>0)
            {
                cv::Mat blankCols(qPage.rows,addCol,CV_8U);
                cv::hconcat(qPage,blankCols,qPage);
            }
            if (addRow>0)
            {
                cv::Mat blankRows(addRow, qPage.cols ,CV_8U);
                cv::vconcat(qPage,blankRows,qPage);
            }


            for (int f=0; f<featurizers.size(); f++)
            {
                pageQFeatures[h*numVert + v][f] = featurizers[f].featurize(qPage);
            }
        }
    }
}
cv::Mat spotDivided(CNNSpotter& spotter, cv::Size size, const vector<float>& queryF, const vector< vector<vector<cv::Mat>*> >& pageQFeatures,  const vector< pair<cv::Rect,cv::Rect> >& pagePullDstRects)
{

    cv::Mat spottingRes = cv::Mat(size,CV_32F);
    for (int i=0; i<pageQFeatures.size(); i++)
    {
        cv::Mat qRes = spotter.spot(queryF,pageQFeatures[i]);
        qRes(pagePullDstRects[i].first).copyTo(spottingRes(pagePullDstRects[i].second));
    }

    return spottingRes;
    
}


float calcAP(const multimap<float,cv::Rect>& predictions, const vector<cv::Rect>& gt)
{
    vector<bool> rel;
    vector<float> scores;
    vector<bool> hits(gt.size());
    float maxScore=-999999;
    int countHits=0;
    for (auto p : predictions)
    {
        float score = -1*p.first;//reverse score, implemented for lower score is better
        cv::Rect bb = p.second;
        bool relevant = false;
        for (int i=0; i<gt.size(); i++)
        {
            cv::Rect t = gt[i];
            float overlapping = (bb&t).area();
            if (overlapping/max(bb.area(),t.area())>0.5)
            {
                relevant=true;
                hits[i]=true;
                countHits++;
                break;
            }
        }
        rel.push_back(relevant);
        scores.push_back(score);

        if (score > maxScore)
            maxScore=score;
    }
    if (countHits==0)
        return 0;
    for (bool hit : hits)
    {
        if (!hit)
        {
            scores.push_back(maxScore);
            rel.push_back(true);
        }
    }

    //ap calc
    int Nrelevants=0;
    vector<int> rank;
    for (int j=0; j < scores.size(); j++)
    {            
        float s = scores[j];
        
        if (rel[j])
        {
            int better=0;
            int equal = 0;
            
            for (int k=0; k < scores.size(); k++)
            {
                if (k!=j)
                {
                    float s2 = scores[k];
                    if (s2< s) better++;
                    else if (s2==s) equal++;
                }
            }
            
            
            rank.push_back(better+floor(equal/2.0));
            Nrelevants++;
        }
        
    }
    qsort(rank.data(), Nrelevants, sizeof(int), sort_xxx);
    
    float ap=0;
    
    for(int j=0;j<Nrelevants;j++){
        
        float prec_at_k =  ((float)(j+1))/(rank[j]+1);
        ap += prec_at_k;            
    }
    ap/=Nrelevants;
    assert(ap==ap);
    return ap;
}

#define MAX_QUERIES 5
void evalSimple(vector<string> queries, vector<string> classes, vector< string >& pages, vector< vector<cv::Rect> >& gt, CNNFeaturizer* query_featurizer, vector<CNNFeaturizer>& featurizers, CNNSpotter& spotter, string saveDir , string dispDir, vector<float> sizes=vector<float>(), float sizeThresh=-1)
{
    BBPredictor bbPredictor;
    
    int queryCount=0;
    bool testtt=true;
    float maxAP=0;
    int maxIdx;

    float MAP=0;
    int countMAP=0;
    map<string,pair<float,int> > perClassAP;
    float meanRecall =0;
    float meanPrecision =0;    
    float meanPixelRecall =0;
    float meanPixelPrecision =0;

    mkdir(saveDir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir(dispDir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    cv::Mat pageIm;
    string curPage="";
    //vector<vector<cv::Mat>*> pageFeatures(featurizers.size());
    //vector<vector<cv::Mat>*> pageQFeatures[4];
    //for (int i=0; i<4; i++)
    //    pageQFeatures[i].resize(featurizers.size());
    vector< vector<vector<cv::Mat>*> > pageQFeatures;
    vector< pair<cv::Rect,cv::Rect> > pagePullDstRects;
    int edgeRegion=100;
    int windowSize=600;

    //bool tooBig = pageIm.rows*pageIm.cols > 6000*600;


    for (int i=0; i<queries.size(); i++)
    {
        if (i%50==0)
            cout<<(100.0*i)/queries.size()<<"% done. cur MAP: "<<MAP/countMAP<<endl;

        if (gt[i].size()==0)
            continue;
        cv::Mat queryIm = cv::imread(queries[i],0);
        vector<float> queryF = query_featurizer->featurizePool(queryIm);
        if (curPage.compare(pages[i])!=0)
        {
            pageIm = cv::imread(pages[i],0);
            curPage=pages[i];
            featurizeDivided(featurizers, pageIm, edgeRegion, windowSize, pageQFeatures,  pagePullDstRects);
            /*cv::Rect q1(0,0,edgeRegion+pageIm.cols/2,edgeRegion+pageIm.rows/2);
            cv::Rect q2(pageIm.cols/2-edgeRegion,0,edgeRegion+(pageIm.cols%2)+pageIm.cols/2,edgeRegion+pageIm.rows/2);
            cv::Rect q3(0,pageIm.rows/2-edgeRegion,edgeRegion+pageIm.cols/2,edgeRegion+(pageIm.rows%2)+pageIm.rows/2);
            cv::Rect q4(pageIm.cols/2-edgeRegion,pageIm.rows/2-edgeRegion,edgeRegion+(pageIm.cols%2)+pageIm.cols/2,edgeRegion+(pageIm.rows%2)+pageIm.rows/2);
            for (int f=0; f<featurizers.size(); f++)
            {
                if (!tooBig)
                    pageFeatures[f] = featurizers[f].featurize(pageIm);
                else
                {
                    pageQFeatures[0][f] = featurizers[f].featurize(pageIm(q1));
                    pageQFeatures[1][f] = featurizers[f].featurize(pageIm(q2));
                    pageQFeatures[2][f] = featurizers[f].featurize(pageIm(q3));
                    pageQFeatures[3][f] = featurizers[f].featurize(pageIm(q4));
                }
            }*/
        }

        cv::Mat spottingRes = spotDivided(spotter, pageIm.size(), queryF, pageQFeatures, pagePullDstRects);
        /*if (!tooBig)
           spottingRes = spotter.spot(queryF,pageFeatures);
        else
        {
            spottingRes = cv::Mat(pageIm.size(),CV_32F);
            cv::Mat spottingQRes[4];
            for (int i=0; i<4; i++)
                spottingQRes[i] = spotter.spot(queryF,pageQFeatures[i]);
            cv::Rect rq1(0,0,pageIm.cols/2,pageIm.rows/2);
            cv::Rect rq2(pageIm.cols/2,0,(pageIm.cols%2)+pageIm.cols/2,pageIm.rows/2);
            cv::Rect rq3(0,pageIm.rows/2,pageIm.cols/2,(pageIm.rows%2)+pageIm.rows/2);
            cv::Rect rq4(pageIm.cols/2,pageIm.rows/2,(pageIm.cols%2)+pageIm.cols/2,(pageIm.rows%2)+pageIm.rows/2);
            spottingQRes[0](cv::Rect(0,0,rq1.width,rq1.height)).copyTo(spottingRes(rq1));
            spottingQRes[1](cv::Rect(edgeRegion,0,rq2.width,rq2.height)).copyTo(spottingRes(rq2));
            spottingQRes[2](cv::Rect(0,edgeRegion,rq3.width,rq3.height)).copyTo(spottingRes(rq3));
            spottingQRes[3](cv::Rect(edgeRegion,edgeRegion,rq4.width,rq4.height)).copyTo(spottingRes(rq4));
            

        }*/
        multimap<float,cv::Rect> bbs;
        if (sizes.size()>0)
        {
            bbs = bbPredictor.predictHardScale(spottingRes,queryIm.size(),sizes);
        }
        else if (sizeThresh>=0)
           bbs = bbPredictor.predict(spottingRes,queryIm.size(),sizeThresh);
        else
           bbs = bbPredictor.predict(spottingRes);


        float ap = calcAP(bbs,gt[i]);
        MAP += ap;
        countMAP++;
        perClassAP[classes[i]].first+=ap;
        perClassAP[classes[i]].second+=1;

        //vector<bool> hitsGT(instP.at(qP.first).size());
        /*
        int tp=0;
        int totalGTArea=0;
        int totalSpottedArea=0;
        int totalOverlappingArea=0;
        for (cv::Rect t : gt[i])
        {
            totalGTArea+=t.area();
        }
        for (cv::Rect bb : bbs)
        {
            totalSpottedArea+=bb.area();
            bool hit=false;
            for (cv::Rect t : gt[i])
            {
                float overlapping = (bb&t).area();
                totalOverlappingArea+=overlapping;
                if (overlapping/max(bb.area(),t.area())>0.5)
                {
                    //hitsGT[i]=true;
                    if (!hit)
                    {
                        hit=true;
                        tp++;
                        break;
                    }
                }
            }
            //if (!hit)
            //    fp++;
        }
        //for (bool hit : hitsGT)
        //    if (!hit)
        //        fn++;

        
        float recall = (0.0+tp)/gt[i].size();
        float precision = (0.0+tp)/bbs.size();
        float pixelRecall = (0.0+totalOverlappingArea)/totalGTArea;
        float pixelPrecision = (0.0+totalOverlappingArea)/totalSpottedArea;
        meanRecall += recall;
        meanPrecision += precision;
        meanPixelRecall += pixelRecall;
        meanPixelPrecision += pixelPrecision;
        */

        //Visualization
        cv::Mat disp;
        cv::cvtColor(pageIm,disp,CV_GRAY2BGR);
        for (cv::Rect t : gt[i])
        {
            //disp(t)*=cv::Scalar(0,1,1);
            //disp(t).channel(0)*=0;
            for (int c=t.x; c<t.x+t.width; c++)
                for (int r=t.y; r<t.y+t.height; r++)
                    if (c==t.x || c==t.x+t.width-1 || r==t.y || r==t.y+t.height-1)
                    {
                        disp.at<cv::Vec3b>(r,c)[0]=0;
                        disp.at<cv::Vec3b>(r,c)[1]=0;
                        disp.at<cv::Vec3b>(r,c)[2]=255;
                    }
                    else
                        disp.at<cv::Vec3b>(r,c)[0] *= 0.5;
        }
        if (bbs.size()>0)
        {
            auto iter = bbs.begin();
            float minScore = iter->first; //best
            iter = bbs.end();
            iter--;
            float maxScore = iter->first; //worst
            for (auto p : bbs)
            {
                float score = p.first;
                float scale = (score-minScore)/(maxScore-minScore);
                cv::Rect bb = p.second;
                //disp(bb)*=cv::Scalar(1,1,0);
                //disp(bb).channel(2)*=0;
                for (int c=bb.x; c<bb.x+bb.width; c++)
                    for (int r=bb.y; r<bb.y+bb.height; r++)
                    {
                        disp.at<cv::Vec3b>(r,c)[2] *= scale; //less red, the better
                        if (c==bb.x || c==bb.x+bb.width-1 || r==bb.y || r==bb.y+bb.height-1)
                        {
                            disp.at<cv::Vec3b>(r,c)[0]=0;
                            disp.at<cv::Vec3b>(r,c)[1]=255;
                            disp.at<cv::Vec3b>(r,c)[2]=255;
                        }
                    }

            }
        }
        //process spottingRes for saving
        double mean = cv::mean(spottingRes)[0];
        double minV, maxV;
        cv::minMaxLoc(spottingRes,&minV,&maxV);
        cv::Mat out(spottingRes.size(),CV_8U);
        for (int r=0; r<out.rows; r++)
            for (int c=0; c<out.cols; c++)
            {
                out.at<unsigned char>(r,c) = (spottingRes.at<float>(r,c)>mean)?(255*(spottingRes.at<float>(r,c)-mean)/(maxV-mean)):0;
                disp.at<cv::Vec3b>(r,c)[1] = out.at<unsigned char>(r,c);
            }
        char buff[12];
        for (cv::Rect t : gt[i])
        {
            //disp(t)*=cv::Scalar(0,1,1);
            //disp(t).channel(0)*=0;
            for (int c=t.x; c<t.x+t.width; c++)
                for (int r=t.y; r<t.y+t.height; r++)
                    if (c==t.x || c==t.x+t.width-1 || r==t.y || r==t.y+t.height-1)
                    {
                        disp.at<cv::Vec3b>(r,c)[0]=0;
                        disp.at<cv::Vec3b>(r,c)[1]=0;
                        disp.at<cv::Vec3b>(r,c)[2]=255;
                    }
        }
        snprintf(buff, sizeof(buff), "%09d", i);
        string key_str = buff; //caffe::format_int(num_items, 8);
        string outName = saveDir+key_str+".png";
        cv::imwrite(outName,out);

        string dispName = dispDir+key_str+".png";
        cv::imwrite(dispName,disp);
#ifdef SHOW
        cv::imshow("spotting",disp);
        cv::waitKey();
#endif

        /*for (int f=0; f<featurizers.size(); f++)
            if (tooBig)
                for (int i=0; i<4; i++)
                    delete pageQFeatures[i][f];
            else
                delete pageFeatures[f];*/
    }
    for (auto& fv : pageQFeatures)
        for (vector<cv::Mat>* ptr : fv)
            delete ptr;
    /*
    meanRecall/=queries.size();
    meanPrecision/=queries.size();
    meanPixelRecall/=queries.size();
    meanPixelPrecision/=queries.size();
    cout<<"Recall:         "<<meanRecall<<endl;
    cout<<"Precision:      "<<meanPrecision<<endl;
    cout<<"Pixel Recall:   "<<meanPixelRecall<<endl;
    cout<<"PixelPrecision: "<<meanPixelPrecision<<endl;
    */
    for (auto p : perClassAP)
    {
        cout<<"class "<<p.first<<" MAP: "<<(p.second.first/p.second.second)<<endl;
    }
    cout<<"FULL MAP: "<<MAP/countMAP<<endl;

}

void eval(map<string,vector<string> > queries, map<string, map<string, vector< cv::Rect > > >& instancesByPage, CNNFeaturizer* query_featurizer, vector<CNNFeaturizer>& featurizers, CNNSpotter& spotter)
{
    BBPredictor bbPredictor;
    
    map<string, vector< vector<float> > > queryFeatures;//class, instances<feature vector>
    for (auto p : queries)
    {
        for (int i=0; i<min(MAX_QUERIES,(int)p.second.size()); i++)
        {
            cv::Mat im = cv::imread(p.second.at(i),0);
            queryFeatures[p.first].push_back(query_featurizer->featurizePool(im));
        }
    }
    float MAP=0;
    int countMAP=0;
    int queryCount=0;
    bool testtt=true;
    float maxAP=0;
    int maxIdx;
    for (auto instP : instancesByPage)
    {
        cv::Mat pageIm = cv::imread(instP.first,0);
        bool tooBig = pageIm.rows*pageIm.cols > 6000*600;
        vector< vector<vector<cv::Mat>*> > pageQFeatures;
        vector< pair<cv::Rect,cv::Rect> > pagePullDstRects;
        int edgeRegion=100;
        int windowSize=600;
        featurizeDivided(featurizers, pageIm, edgeRegion, windowSize, pageQFeatures,  pagePullDstRects);

        for (auto qP : queryFeatures)
        {
            if (instP.second.find(qP.first) == instP.second.end())
                continue;


            for (const vector<float>& queryF : qP.second)
            {
                
                cv::Mat spottingRes = spotDivided(spotter, pageIm.size(), queryF, pageQFeatures, pagePullDstRects);
                multimap<float,cv::Rect> bbs = bbPredictor.predict(spottingRes);
                /* This isn't needed as we've padded inputs to the network
                if (spottingRes.cols != pageIm.cols || spottingRes.rows != pageIm.rows)
                {
                    //assert(spottingRes.cols<=pageIm.cols);
                    int hShift = (pageIm.cols-spottingRes.cols)/2;
                    int vShift = (pageIm.rows-spottingRes.rows)/2;
                    for (auto& p : bbs)
                    {
                        p.second.x=max(0,p.second.x+hShift);
                        p.second.y=max(0,p.second.y+vShift);
                    }
                }*/

                MAP += calcAP(bbs,instP.second.at(qP.first));
                countMAP++;

                //vector<bool> hitsGT(instP.at(qP.first).size());
                /*
                int tp=0;
                int totalGTArea=0;
                int totalSpottedArea=0;
                int totalOverlappingArea=0;
                for (cv::Rect t : instP.second.at(qP.first))
                {
                    totalGTArea+=t.area();
                }
                for (cv::Rect bb : bbs)
                {
                    totalSpottedArea+=bb.area();
                    bool hit=false;
                    for (int i=0; i<instP.second.at(qP.first).size(); i++)
                    {
                        cv::Rect t = instP.second.at(qP.first)[i];
                        int overlapping = (bb&t).area();
                        totalOverlappingArea+=overlapping;
                        if (overlapping/max(bb.area(),t.area())>0.5)
                        {
                            //hitsGT[i]=true;
                            if (!hit)
                            {
                                hit=true;
                                tp++;
                            }
                        }
                    }
                    //if (!hit)
                    //    fp++;
                }
                //for (bool hit : hitsGT)
                //    if (!hit)
                //        fn++;

                float recall = (0.0+tp)/instP.second.at(qP.first).size();
                float precision = (0.0+tp)/bbs.size();
                float pixelRecall = (0.0+totalOverlappingArea)/totalGTArea;
                float pixelPrecision = (0.0+totalOverlappingArea)/totalSpottedArea;
                cout<<"Recall:         "<<recall<<endl;
                cout<<"Precision:      "<<precision<<endl;
                cout<<"Pixel Recall:   "<<pixelRecall<<endl;
                cout<<"PixelPrecision: "<<pixelPrecision<<endl;
                */
#ifdef SHOW
                cv::Mat disp;
                cv::cvtColor(pageIm,disp,CV_GRAY2BGR);
                for (cv::Rect t : instP.second.at(qP.first))
                {
                    //disp(t)*=cv::Scalar(0,1,1);
                    //disp(t).channel(0)*=0;
                    for (int c=t.x; c<t.x+t.width; c++)
                        for (int r=t.y; r<t.y+t.height; r++)
                            disp.at<cv::Vec3b>(r,c)[0] *= 0;
                }
                for (cv::Rect bb : bbs)
                {
                    //disp(bb)*=cv::Scalar(1,1,0);
                    //disp(bb).channel(2)*=0;
                    for (int c=bb.x; c<bb.x+bb.width; c++)
                        for (int r=bb.y; r<bb.y+bb.height; r++)
                            disp.at<cv::Vec3b>(r,c)[2] *= 0;
                }
                cv::imshow("spotting",disp);
                cv::waitKey();
#endif

            }
        }

        for (auto& fv : pageQFeatures)
            for (vector<cv::Mat>* ptr : fv)
                delete ptr;
    }
    cout<<"MAP: "<<MAP/countMAP<<endl;

}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " [caffemodels.mod iteration] OR [deploy_featurizer.prototxt [+deploy_featurizer.prototxt...] deploy_spotter.prototxt network.caffemodel] AND [-gpu #] [queries.txt query_dir/ page.gtp page_dir/] OR [pointer.txt root_dir/ out_dir/ [-scales ...(sizes #s)]]"
              << "" << std::endl;
    cout<<argc<<endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  vector<string> model_featurizer_files;
  string query_featurizer_file;
  string model_spotter_file;
  string trained_file;

  int argI=1;
  string arg1 = argv[argI++];
  if (arg1.length()>4 && arg1.substr(arg1.length()-4).compare(".mod")==0)
  {
      ifstream model(arg1);
      string line;
      getline(model,query_featurizer_file);
      getline(model,line);
      string part;
      stringstream ss(line);
      while (getline(ss,part,' '))
      {
          model_featurizer_files.push_back(part);
      }
      getline(model,model_spotter_file);
      getline(model,trained_file);
      trained_file += "_iter_"+string(argv[argI++])+".caffemodel";
  }
  else
  {

      model_featurizer_files.push_back(arg1);
      query_featurizer_file=arg1;
      while (argv[argI][0]=='+')
      {
          model_featurizer_files.push_back(string(argv[argI++]).substr(1));
      }
      model_spotter_file   = argv[argI++];
      trained_file = argv[argI++];
  }

  int gpu=-1;
  if (argv[argI][0]=='-' && argv[argI][0]=='g')
  {
      gpu=atoi(argv[(++argI)++]);
  }


  CNNFeaturizer* query_featurizer=NULL;
  vector<CNNFeaturizer> page_featurizers;
  for (string file : model_featurizer_files)
    page_featurizers.emplace_back(file, trained_file);
  if (query_featurizer_file.compare(model_featurizer_files[0])!=0)
      query_featurizer = new CNNFeaturizer(query_featurizer_file,trained_file,gpu);
  else
      query_featurizer = &(page_featurizers[0]);
  CNNSpotter spotter(model_spotter_file, trained_file,gpu);

  if (argc-argI == 4 && argv[argI+3][0]!='-')
  {
      string query_pointer_file = argv[argI++];
      string query_image_dir  = argv[argI++];
      string page_label_file = argv[argI++];
      string page_image_dir  = argv[argI++];


      map<string, map<string, vector< cv::Rect > > > instancesByPage;

      map<string,vector<string> > queries;

      ifstream filein(query_pointer_file);
      assert(filein.good());
      string line;
      while (getline(filein,line))
      {
          stringstream ss(line);
          string fileName;
          getline(ss,fileName,' ');
          string c;
          getline(ss,c,' ');
          queries[c].push_back(query_image_dir+fileName);
      }
      filein.close();
      
      
      
      filein.open(page_label_file);
      assert(filein.good());
      string curPathIm="";
      cv::Mat curIm;
      int minDim=999999;
      while (getline(filein,line))
      {
          stringstream ss(line);
          string part;
          getline(ss,part,' ');

          string pathIm=page_image_dir+string(part);
          if (curPathIm.compare(pathIm)!=0)
          {
              curPathIm=pathIm;
              curIm = cv::imread(curPathIm,0);
          }
          
          getline(ss,part,' ');
          int x1=max(1,stoi(part));//;-1;
          getline(ss,part,' ');
          int y1=max(1,stoi(part));//;-1;
          getline(ss,part,' ');
          int x2=min(curIm.cols,stoi(part));//;-1;
          getline(ss,part,' ');
          int y2=min(curIm.rows,stoi(part));//;-1;
          cv::Rect loc(x1,y1,x2-x1+1,y2-y1+1);
          //locs.push_back(loc);
          if (x1<0 || x1>=x2 || x2>=curIm.cols)
          {
              cout<<pathIm<<endl;
              cout<<"line: ["<<line<<"]  loc: "<<x1<<" "<<y1<<" "<<x2<<" "<<y2<<"  im:"<<curIm.rows<<","<<curIm.cols<<endl;
              assert(false);
          }
          getline(ss,part,' ');
          string label=part;
          instancesByPage[pathIm][label].push_back(cv::Rect(x1,y1,x2-x1+1,y2-y1+1));
      }

      eval(queries,instancesByPage,query_featurizer,page_featurizers,spotter);
      /*cv::Mat res = spotter.spot(query,page);
      double minV,maxV;
      cv::minMaxLoc(res,&minV,&maxV);
      cv::Mat disp;
      cvtColor(page,disp,CV_GRAY2BGR);
      for (int r=0; r<page.rows; r++)
          for (int c=0; c<page.cols; c++)
          {
              disp.at<cv::Vec3b>(r,c)[2] = 255 * (res.at<float>(r,c)-minV)/(maxV-minV);
          }
      cv::imshow("res",disp);
      cv::waitKey();
      cv::imwrite("test.png",disp);
    */

      //delete dataset;
  }
  else
  {
    string pointer_file = argv[argI++];
    string root_dir = argv[argI++];
    string out_dir = argv[argI++];
    vector<float> sizes;
    if (argc>argI++)
    {
        while (argI<argc)
        {
            sizes.push_back(atof(argv[argI++]));
        }
    }
    mkdir(out_dir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    

    vector<string> queries,classes,pages;
    vector< vector<cv::Rect> > gts;

    ifstream filein;
    filein.open(pointer_file);
    assert(filein.good());
    string line;
    while (getline(filein,line))
    {
          stringstream ss(line);
          string part;
          getline(ss,part,' ');
          classes.push_back(part);
          getline(ss,part,' ');
          queries.push_back(root_dir+part);
          getline(ss,part,' ');
          pages.push_back(root_dir+part);
          //getline(ss,part,' ');
          //discard gt image

          getline(ss,part,' ');
          int numTrue=stoi(part);
          vector<cv::Rect> trueLocs(numTrue);
          //cout<<numTrue<<" trues for "<<queries.back()<<" "<<pages.back()<<endl;
          for (int i=0; i<numTrue; i++)
          {
              cv::Rect loc;
              getline(ss,part,' ');
              loc.x=stoi(part);
              getline(ss,part,' ');
              loc.y=stoi(part);
              getline(ss,part,' ');
              loc.width=stoi(part);
              getline(ss,part,' ');
              loc.height=stoi(part);
              trueLocs[i]=loc;
              //cout<<loc.x<<" "<<loc.y<<" "<<loc.width<<" "<<loc.height<<endl;
          }
          gts.push_back(trueLocs);
    }
    evalSimple(queries,classes,pages,gts,query_featurizer,page_featurizers,spotter, out_dir+"out/", out_dir+"disp/",sizes);
  }

  if (query_featurizer_file.compare(model_featurizer_files[0])!=0)
          delete query_featurizer;
}
