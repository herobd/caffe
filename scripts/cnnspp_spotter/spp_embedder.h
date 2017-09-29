#ifndef SPPEMBEDDER_H
#define SPPEMBEDDER_H

//#define CPU_ONLY
#include <caffe/caffe.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#ifndef OPENCV2
#include <opencv2/imgcodecs.hpp>
#else
#include <opencv2/highgui/highgui.hpp>
#endif
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

class SPPEmbedder {
 public:
  SPPEmbedder(const string& model_file,
             const string& trained_file,
             bool normalize=true,
             int gpu=-1
             //const string& mean_file,
             //const string& label_file
             );

  //std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);
  cv::Mat embed(const std::vector<cv::Mat>* features);
  cv::Mat embed(const std::vector< std::vector<cv::Mat> >& batchFeatures);

 private:
  //void SetMean(const string& mean_file);


  void WrapInputLayer(std::vector<cv::Mat>* input_channels);
  void WrapInputLayer(std::vector< std::vector<cv::Mat> >* input_channels);

  void Preprocess(const std::vector<cv::Mat>* features,
                  std::vector<cv::Mat>* input_channels);

 private:
  std::shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  bool normalize;
  cv::Mat mean_;
  std::vector<string> labels_;
};
#endif
