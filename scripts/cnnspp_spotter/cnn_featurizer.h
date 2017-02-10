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

class CNNFeaturizer {
 public:
  CNNFeaturizer(const string& model_file,
             const string& trained_file
             //const string& mean_file,
             //const string& label_file
             );

  //std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);
  std::vector<cv::Mat>* featurize(const cv::Mat& img);

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

