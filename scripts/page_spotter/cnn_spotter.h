#ifndef SPPEMBEDDER_H
#define SPPEMBEDDER_H

#define CPU_ONLY
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
using namespace caffe;
using namespace std;

class CNNSpotter {
 public:
  CNNSpotter(const string& model_file,
             const string& trained_file);

  //std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);
  cv::Mat spot(const vector<float>& features_query, vector< std::vector<cv::Mat>* > features_pages);

 private:
  //void SetMean(const string& mean_file);


  void WrapInputLayer(Blob<float>* input_layer,std::vector<cv::Mat>* input_channels);
  void WrapInputLayer(Blob<float>* input_layer,std::vector< std::vector<cv::Mat> >* input_channels);

  void Preprocess(const std::vector<cv::Mat>* features,
                  std::vector<cv::Mat>* input_channels);
void Preprocess(const vector<float>& features,
                  std::vector<cv::Mat>* input_channels);

 private:
  std::shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
};
#endif
