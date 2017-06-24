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


float calcAP(const map<float,cv::Rect>& predictions, const vector<cv::Rect>& gt)
{
    vector<bool> rel;
    vector<float> scores;
    vector<bool> hits(gt.size());
    float maxScore=-999999;
    for (auto p : predictions)
    {
        float score = p.first;
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
                break;
            }
        }
        rel.push_back(relevant);
        scores.push_back(score);

        if (score > maxScore)
            maxScore=score;
    }
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
void evalSimple(vector<string> queries, vector< string >& pages, vector< vector<cv::Rect> >& gt, CNNFeaturizer& featurizer, CNNSpotter& spotter, string saveDir , string dispDir)
{
    BBPredictor bbPredictor;
    
    int queryCount=0;
    bool testtt=true;
    float maxAP=0;
    int maxIdx;

    float MAP=0;
    int countMAP=0;
    float meanRecall =0;
    float meanPrecision =0;    
    float meanPixelRecall =0;
    float meanPixelPrecision =0;

    mkdir(saveDir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir(dispDir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    for (int i=0; i<queries.size(); i++)
    {
        if (gt[i].size()==0)
            continue;
        cv::Mat queryIm = cv::imread(queries[i],0);
        vector<float> queryFeatures = featurizer.featurizePool(queryIm);
        cv::Mat pageIm = cv::imread(pages[i],0);
        vector<cv::Mat>* pageFeatures = featurizer.featurize(pageIm);
        assert(queryFeatures.size()==pageFeatures->size());

                
        //vector<cv::Mat> queryTiled(queryFeatures.size());
        //tile out to page size
        //for (int i=0; i<queryFeatures.size(); i++)
        //{
        //    queryTiled[i] = cv::Mat(pageFeatures->at(i).size(), CV_32F, cv::Scalar(queryFeatures[i]));
        //}
        cv::Mat spottingRes = spotter.spot(queryFeatures,pageFeatures);
                assert(spottingRes.cols == pageIm.cols && spottingRes.rows == pageIm.rows);
        map<float,cv::Rect> bbs = bbPredictor.predict(spottingRes);

        MAP += calcAP(bbs,gt[i]);
        countMAP++;

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
                    disp.at<cv::Vec3b>(r,c)[0] *= 0;
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
                            disp.at<cv::Vec3b>(r,c)[2]=0;
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

        delete pageFeatures;
    }
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
    cout<<"MAP: "<<MAP/countMAP<<endl;

}

void eval(map<string,vector<string> > queries, map<string, map<string, vector< cv::Rect > > >& instancesByPage, CNNFeaturizer& featurizer, CNNSpotter& spotter)
{
    BBPredictor bbPredictor;
    
    map<string, vector< vector<float> > > queryFeatures;//class, instances<feature vector>
    for (auto p : queries)
    {
        for (int i=0; i<min(MAX_QUERIES,(int)p.second.size()); i++)
        {
            cv::Mat im = cv::imread(p.second.at(i),0);
            queryFeatures[p.first].push_back(featurizer.featurizePool(im));
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
        vector<cv::Mat>* pageFeatures = featurizer.featurize(pageIm);

        for (auto qP : queryFeatures)
        {
            if (instP.second.find(qP.first) == instP.second.end())
                continue;


            for (const vector<float>& queryF : qP.second)
            {
                
                //vector<cv::Mat> queryTiled(queryF.size());
                //tile out to page size
                //for (int i=0; i<queryF.size(); i++)
                //{
                //    queryTiled[i] = cv::Mat(pageFeatures->at(i).size(), CV_32F, cv::Scalar(queryF[i]));
                //}
                cv::Mat spottingRes = spotter.spot(queryF,pageFeatures);
                map<float,cv::Rect> bbs = bbPredictor.predict(spottingRes);
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
                }

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
        delete pageFeatures;
    }
    cout<<"MAP: "<<MAP/countMAP<<endl;

}

int main(int argc, char** argv) {
  if (argc != 8 && argc != 7) {
    std::cerr << "Usage: " << argv[0]
              << " deploy_featurizer.prototxt deploy_spotter.prototxt network.caffemodel [queries.txt query_dir/ page.gtp page_dir/] OR [pointer.txt root_dir/ out_dir/]"
              << "" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_featurizer_file   = argv[1];
  string model_spotter_file   = argv[2];
  string trained_file = argv[3];
  CNNFeaturizer featurizer(model_featurizer_file, trained_file);
  CNNSpotter spotter(model_spotter_file, trained_file);

  if (argc == 8)
  {
      string query_pointer_file = argv[4];
      string query_image_dir  = argv[5];
      string page_label_file = argv[6];
      string page_image_dir  = argv[7];


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

      eval(queries,instancesByPage,featurizer,spotter);
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
    string pointer_file = argv[4];
    string root_dir = argv[5];
    string out_dir = argv[6];
    mkdir(out_dir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    

    vector<string> queries,pages;
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
          queries.push_back(root_dir+part);
          getline(ss,part,' ');
          pages.push_back(root_dir+part);
          getline(ss,part,' ');
          //discard gt image

          getline(ss,part,' ');
          int numTrue=stoi(part);
          vector<cv::Rect> trueLocs(numTrue);
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
          }
          gts.push_back(trueLocs);
    }
    evalSimple(queries,pages,gts,featurizer,spotter, out_dir+"out/", out_dir+"disp/");
  }
}
