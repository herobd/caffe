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

#define MAX_QUERIES 5

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
    float map=0;
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
                
                vector<cv::Mat> queryTiled(queryF.size());
                //tile out to page size
                for (int i=0; i<queryF.size(); i++)
                {
                    queryTiled[i] = cv::Mat(pageFeatures->at(i).size(), CV_32F, cv::Scalar(queryF[i]));
                }
                cv::Mat spottingRes = spotter.spot(&queryTiled,pageFeatures);
                vector<cv::Rect> bbs = bbPredictor.predict(spottingRes);

                //vector<bool> hitsGT(instP.at(qP.first).size());
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

}

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "Usage: " << argv[0]
              << " deploy_featurizer.prototxt deploy_spotter.prototxt network.caffemodel queries.txt query_dir/ page.gtp page_dir/"
              << "" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_featurizer_file   = argv[1];
  string model_spotter_file   = argv[2];
  string trained_file = argv[3];
  string query_pointer_file = argv[4];
  string query_image_dir  = argv[5];
  string page_label_file = argv[6];
  string page_image_dir  = argv[7];

  CNNFeaturizer featurizer(model_featurizer_file, trained_file);
  CNNSpotter spotter(model_spotter_file, trained_file);

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
