//g++ --std=c++11 -fopenmp gwdataset.cpp showPageSpotting.cpp -lcaffe -lglog -l:libopencv_core.so.3.1 -l:libopencv_imgcodecs.so.3.1 -l:libopencv_imgproc.so.3.1 -l:libopencv_highgui.so.3.1 -lprotobuf -lboost_system -I ../include/ -L ../build/lib/ -o showPageSpotting
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
#include <memory>


using namespace caffe;
using namespace std;


class Spotter {
 public:
  Spotter(const string& model_file,
             const string& trained_file);

    cv::Mat spot(const cv::Mat& query, const cv::Mat& page, cv::Mat* low);

 private:
  //void SetMean(const string& mean_file);


  void WrapInputLayer(Blob<float>* input_layer,std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  std::shared_ptr<Net<float> > net_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
  bool normalizeOut;
};

Spotter::Spotter(const string& model_file,
                       const string& trained_file)
{
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 2) << "Network should have exactly two inputs.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";

}



cv::Mat Spotter::spot(const cv::Mat& query, const cv::Mat& page, cv::Mat* low) {
  Blob<float>* input_query = net_->input_blobs()[0];
  Blob<float>* input_page = net_->input_blobs()[1];
  //Blob<float>* input_layer = net_->blobs["data"];
  input_query->Reshape(1, num_channels_,
                       query.rows, query.cols);
  input_page->Reshape(1, num_channels_,
                       page.rows, page.cols);
//net.blobs['data'].reshape(1, *in_.shape)
//net.blobs['data'].data[...] = in_
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_query_channels;
  WrapInputLayer(input_query,&input_query_channels);
  std::vector<cv::Mat> input_page_channels;
  WrapInputLayer(input_page,&input_page_channels);

  Preprocess(query, &input_query_channels);
  Preprocess(page, &input_page_channels);

  CHECK(reinterpret_cast<float*>(input_query_channels.at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";

  net_->Forward();

///////////////////////



  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = output_layer->cpu_data()+(output_layer->width()*output_layer->height());
  assert(output_layer->channels()==1);
  cv::Mat ret(output_layer->height(),output_layer->width(),CV_32F);
  //copy(begin,end,ret.data);
  int ii=0;
  for (int r=0; r<ret.rows; r++)
      for (int c=0; c<ret.cols; c++)
      {
        ret.at<float>(r,c) = begin[ii++];
      }

  if (low != NULL)
  {
      const boost::shared_ptr< Blob< float > > low_layer = net_->blob_by_name("comb2_concat");
      const float* begin = low_layer->cpu_data();
      *low = cv::Mat(low_layer->height(),low_layer->width(),CV_32F);
      ii=0;
      for (int r=0; r<low->rows; r++)
          for (int c=0; c<low->cols; c++)
          {
            low->at<float>(r,c) = begin[ii++];
          }
  }
  
  return ret;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Spotter::WrapInputLayer(Blob<float>* input_layer, std::vector<cv::Mat>* input_channels) {

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Spotter::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  //if (sample.size() != input_geometry_)
  //  cv::resize(sample, sample_resized, input_geometry_);
  //else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  //cv::Mat sample_normalized;
  //cv::subtract(sample_float, mean_, sample_normalized);
  sample_float*=0.00390625;
  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_float, *input_channels);

}


int sort_xxx(const void *x, const void *y) {
    if (*(int*)x > *(int*)y) return 1;
    else if (*(int*)x < *(int*)y) return -1;
    else return 0;
}

int main(int argc, char** argv) {
  if (argc != 5 &&argc!=6) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " query page" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string query_file = argv[3];
  string page_file  = argv[4];

  Spotter spotter(model_file, trained_file);
  cv::Mat query = cv::imread(query_file,0);
  cv::Mat page = cv::imread(page_file,0);
  int x2 = min(page.cols/2 + 400, page.cols-1);
  int x1 = x2-399;
  int y2 = min(10 + 400, page.rows-1);
  int y1 = y2-399;
  page=page(cv::Rect(x1,y1,400,400));
  cv::Mat low;
  cv::Mat res = spotter.spot(query,page,NULL);//&low);
  double minV,maxV;
  cv::minMaxLoc(res,&minV,&maxV);
  double thresh = cv::mean(res)[0];//(maxV+minV)/2;
  cv::Mat disp;
  cvtColor(page,disp,CV_GRAY2BGR);
  for (int r=0; r<page.rows; r++)
      for (int c=0; c<page.cols; c++)
      {
          disp.at<cv::Vec3b>(r,c)[2] = 255 * max(0.0,res.at<float>(r,c)-thresh)/(maxV-thresh);
          if (res.at<float>(r,c)<thresh)
              disp.at<cv::Vec3b>(r,c)[1]=0;
      }

  int kernel_size=19;
  cv::Mat kernel = cv::Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);
  cv::Mat conv;
  cv::filter2D(res, conv, -1 , kernel);
  cv::Point loc;
  cv::minMaxLoc(res,&minV,&maxV,NULL,&loc);
  cv::Point pt1(loc.x-9,loc.y-9);
  cv::Point pt2(loc.x+9,loc.y+9);
  cv::rectangle(disp, pt1, pt2, cv::Scalar(0,255,0));
  /*
  cv::Mat threshed(disp.size(),CV_8U);
  float thresh = (maxV+minV)/2;
  for (int r=0; r<page.rows; r++)
      for (int c=0; c<page.cols; c++)
      {
          if (res.at<float>(r,c)>thresh)
            threshed.at<unsigned char>(r,c) = 255;
          else
            threshed.at<unsigned char>(r,c) = 0;
      }
  Mat ccs,stats,cent;
  int count = cv::connectedComponentsWithStats (threshed, ccs, stats, cent, 8, CV_32S);
  */

  /*
  cv::minMaxLoc(low,&minV,&maxV);
  cv::Mat dispLow(low.size(),CV_8U);
  for (int r=0; r<low.rows; r++)
      for (int c=0; c<low.cols; c++)
      {
          dispLow.at<unsigned char>(r,c) = 255 * (low.at<float>(r,c)-minV)/(maxV-minV);
      }
      */
  cv::imshow("res",disp);
  //cv::imshow("low",dispLow);
  cv::waitKey();
  cv::imwrite("test.png",disp);
  //cv::imwrite("testLow.png",dispLow);


  //delete dataset;
}
