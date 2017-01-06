//g++ -std=c++11 -fopenmp gwdataset.cpp evalSpotting_fixed.cpp -lcaffe -lglog -l:libopencv_core.so.3.0 -l:libopencv_imgcodecs.so.3.0 -l:libopencv_imgproc.so.3.0 -lprotobuf -lboost_system -I ../include/ -L ../build/lib/ -o evalSpotting_fixed
//g++ -std=c++11 -fopenmp gwdataset.cpp evalSpotting_fixed.cpp -lcaffe -lglog -lopencv_core -lopencv_highgui -lopencv_imgproc -lprotobuf -lboost_system -I /home/brianld/include -I ../include/ -L ../build/lib/ -o evalSpotting_fixed
#define CPU_ONLY
#include <caffe/caffe.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "gwdataset.h"
using namespace caffe;
using namespace std;

class CNNEmbedder {
 public:
  CNNEmbedder(const string& model_file,
             const string& trained_file
             //const string& mean_file,
             //const string& label_file
             );

  //std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);
  cv::Mat embed(const cv::Mat& img);

 private:
  //void SetMean(const string& mean_file);


  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  std::shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

