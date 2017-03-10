//g++ -std=c++11 -fopenmp gwdataset.cpp evalSpotting.cpp -lcaffe -lglog -l:libopencv_core.so.3.0 -l:libopencv_imgcodecs.so.3.0 -l:libopencv_imgproc.so.3.0 -lprotobuf -lboost_system -I ../include/ -L ../build/lib/ -o evalSpotting
#define CPU_ONLY
#include <caffe/caffe.hpp>
#include "caffe/util/io.hpp"
#include <opencv2/core.hpp>
#ifndef OPENCV2
#include <opencv2/imgcodecs.hpp>
#else
#include <opencv2/highgui.hpp>
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

class Embedder {
 public:
  Embedder(const string& model_file,
             const string& trained_file,
             //const string& mean_file,
             //const string& label_file
             bool normalizeOut=true);

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
  bool normalizeOut;
};

Embedder::Embedder(const string& model_file,
                       const string& trained_file,
                       //const string& mean_file,
                       //const string& label_file
                       bool normalizeOut) : normalizeOut(normalizeOut){
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  //SetMean(mean_file);

  /* Load labels. */
  //std::ifstream labels(label_file.c_str());
  //CHECK(labels) << "Unable to open labels file " << label_file;
  //string line;
  //while (std::getline(labels, line))
  //  labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  //CHECK_EQ(labels_.size(), output_layer->channels())
  //  << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. 
std::vector<Prediction> Embedder::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}*/

/* Load the mean file in binaryproto format. */
/*void Embedder::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  // Convert from BlobProto to Blob<float> 
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";
  // The format of the mean file is planar 32-bit float BGR or grayscale. 
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    // Extract an individual channel. 
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  // Merge the separate channels into a single image. 
  cv::Mat mean;
  cv::merge(channels, mean);

  // Compute the global mean pixel value and create a mean image
    // filled with this value. 
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}*/

cv::Mat Embedder::embed(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  //Blob<float>* input_layer = net_->blobs["data"];
  input_layer->Reshape(1, num_channels_,
                       img.rows, img.cols);
//net.blobs['data'].reshape(1, *in_.shape)
//net.blobs['data'].data[...] = in_
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

///////////////////////
#ifdef SHOW
/*
  vector< string > layers = net_->layer_names();
  const vector< boost::shared_ptr< Blob< float > > >& weights = net_->params();

  const float* begin_;

  int counter;
  int i=0;
  while (layers[i].compare("conv3")!=0)
      i++;
  begin_ = weights[i]->cpu_data();
  cout<<"Weights conv3:"<<endl;
  for (int ii=0; ii<40; ii+=4)
  {
      cout<<begin_[ii]<<", ";
  }
  cout<<endl;
  
  i=0;
  while (layers[i].compare("conv2")!=0)
      i++;
  begin_ = weights[i]->cpu_data();
  cout<<"Weights conv2:"<<endl;
  for (int ii=0; ii<40; ii+=4)
  {
      cout<<begin_[ii]<<", ";
  }
  cout<<endl;

  i=0;
  while (layers[i].compare("conv1")!=0)
      i++;
  begin_ = weights[i]->cpu_data();
  cout<<"Weights conv1:"<<endl;
  for (int ii=0; ii<40; ii+=4)
  {
      cout<<begin_[ii]<<", ";
  }
  cout<<endl;

  const boost::shared_ptr<Blob<float> > ip3_layer = net_->blob_by_name("ip3");
  begin_ = ip3_layer->cpu_data();
  cout<<"ip3:"<<endl;
  //for (int ii=0; ii<100; ii+=4)
  //{
  //    cout<<begin_[ii]<<", ";
  //}
  //cout<<endl;
  counter=10;
  for (int ii=0; ii<ip3_layer->channels()*ip3_layer->height()*ip3_layer->width(); ii++)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<"; ";
  }
  cout<<endl;
  
  const boost::shared_ptr<Blob<float> > ip2_layer = net_->blob_by_name("ip2");
  begin_ = ip2_layer->cpu_data();
  cout<<"ip2:"<<endl;
  //for (int ii=0; ii<100; ii+=4)
  //{
  //    cout<<begin_[ii]<<", ";
  //}
  //cout<<endl;
  counter=10;
  for (int ii=0; ii<ip2_layer->channels()*ip2_layer->height()*ip2_layer->width(); ii++)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<"; ";
  }
  cout<<endl;
  
  const boost::shared_ptr<Blob<float> > ip1_layer = net_->blob_by_name("ip1");
  begin_ = ip1_layer->cpu_data();
  cout<<"ip1:"<<endl;
  //for (int ii=0; ii<100; ii+=4)
  //{
  //    cout<<begin_[ii]<<", ";
  //}
  //cout<<endl;
  counter=10;
  for (int ii=0; ii<ip1_layer->channels()*ip1_layer->height()*ip1_layer->width(); ii++)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<"; ";
  }
  cout<<endl;
  
  const boost::shared_ptr<Blob<float> > conv13_layer = net_->blob_by_name("conv13");
  begin_ = conv13_layer->cpu_data();
  cout<<"conv13:"<<conv13_layer->channels()<<"  "<<conv13_layer->height()<<"  "<<conv13_layer->width()<<endl;
  //for (int ii=0; ii<400; ii+=4)
  //{
  //    cout<<begin_[ii]<<", ";
  //}
  //cout<<endl;
  counter=10;
  for (int ii=0; ii<conv13_layer->channels()*conv13_layer->height()*conv13_layer->width(); ii++)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<"; ";
  }
  cout<<endl;
  counter=10;
  for (int ii=conv13_layer->channels()*conv13_layer->height()*conv13_layer->width()-1; ii>=0; ii--)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<": ";
  }
  cout<<endl;
  
  const boost::shared_ptr<Blob<float> > conv12_layer = net_->blob_by_name("conv12");
  begin_ = conv12_layer->cpu_data();
  cout<<"conv12:"<<conv12_layer->channels()<<"  "<<conv12_layer->height()<<"  "<<conv12_layer->width()<<endl;
  //for (int ii=0; ii<40; ii+=4)
  //{
  //    cout<<begin_[ii]<<", ";
  //}
  //cout<<endl;
  counter=10;
  for (int ii=0; ii<conv12_layer->channels()*conv12_layer->height()*conv12_layer->width(); ii++)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<"; ";
  }
  cout<<endl;
  counter=10;
  for (int ii=conv12_layer->channels()*conv12_layer->height()*conv12_layer->width()-1; ii>=0; ii--)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<": ";
  }
  cout<<endl;

  const boost::shared_ptr<Blob<float> > conv11_layer = net_->blob_by_name("conv11");
  begin_ = conv11_layer->cpu_data();
  cout<<"conv11:"<<conv11_layer->channels()<<"  "<<conv11_layer->height()<<"  "<<conv11_layer->width()<<endl;
  //for (int ii=0; ii<40; ii+=4)
  //{
  //    cout<<begin_[ii]<<", ";
  //}
  //cout<<endl;
  counter=10;
  for (int ii=0; ii<conv11_layer->channels()*conv11_layer->height()*conv11_layer->width(); ii++)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<"; ";
  }
  cout<<endl;
  counter=10;
  for (int ii=conv11_layer->channels()*conv11_layer->height()*conv11_layer->width()-1; ii>=0; ii--)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<": ";
  }
  cout<<endl;
  
  const boost::shared_ptr<Blob<float> > conv10_layer = net_->blob_by_name("conv10");
  begin_ = conv10_layer->cpu_data();
  cout<<"conv10:"<<conv10_layer->channels()<<"  "<<conv10_layer->height()<<"  "<<conv10_layer->width()<<endl;
  //for (int ii=0; ii<40; ii+=4)
  //{
  //    cout<<begin_[ii]<<", ";
  //}
  // cout<<endl;
  counter=10;
  for (int ii=0; ii<conv11_layer->channels()*conv11_layer->height()*conv11_layer->width(); ii++)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<"; ";
  }
  cout<<endl;
  counter=10;
  for (int ii=conv11_layer->channels()*conv11_layer->height()*conv11_layer->width()-1; ii>=0; ii--)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<": ";
  }
  cout<<endl;

  const boost::shared_ptr<Blob<float> > conv9_layer = net_->blob_by_name("conv9");
  begin_ = conv9_layer->cpu_data();
  cout<<"conv9:"<<conv9_layer->channels()<<"  "<<conv9_layer->height()<<"  "<<conv9_layer->width()<<endl;
  counter=10;
  for (int ii=0; ii<conv11_layer->channels()*conv11_layer->height()*conv11_layer->width(); ii++)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<"; ";
  }
  cout<<endl;
  counter=10;
  for (int ii=conv11_layer->channels()*conv11_layer->height()*conv11_layer->width()-1; ii>=0; ii--)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<": ";
  }
  cout<<endl;

  const boost::shared_ptr<Blob<float> > conv8_layer = net_->blob_by_name("conv8");
  begin_ = conv8_layer->cpu_data();
  cout<<"conv8:"<<conv8_layer->channels()<<"  "<<conv8_layer->height()<<"  "<<conv8_layer->width()<<endl;
  counter=10;
  for (int ii=0; ii<conv11_layer->channels()*conv11_layer->height()*conv11_layer->width(); ii++)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<"; ";
  }
  cout<<endl;
  counter=10;
  for (int ii=conv11_layer->channels()*conv11_layer->height()*conv11_layer->width()-1; ii>=0; ii--)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<": ";
  }
  cout<<endl;

  const boost::shared_ptr<Blob<float> > conv7_layer = net_->blob_by_name("conv7");
  begin_ = conv7_layer->cpu_data();
  cout<<"conv7:"<<conv7_layer->channels()<<"  "<<conv7_layer->height()<<"  "<<conv7_layer->width()<<endl;
  counter=10;
  for (int ii=0; ii<conv11_layer->channels()*conv11_layer->height()*conv11_layer->width(); ii++)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<"; ";
  }
  cout<<endl;
  counter=10;
  for (int ii=conv11_layer->channels()*conv11_layer->height()*conv11_layer->width()-1; ii>=0; ii--)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<": ";
  }
  cout<<endl;

  const boost::shared_ptr<Blob<float> > conv6_layer = net_->blob_by_name("conv6");
  begin_ = conv6_layer->cpu_data();
  cout<<"conv6:"<<conv6_layer->channels()<<"  "<<conv6_layer->height()<<"  "<<conv6_layer->width()<<endl;
  counter=10;
  for (int ii=0; ii<conv11_layer->channels()*conv11_layer->height()*conv11_layer->width(); ii++)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<"; ";
  }
  cout<<endl;
  counter=10;
  for (int ii=conv11_layer->channels()*conv11_layer->height()*conv11_layer->width()-1; ii>=0; ii--)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<": ";
  }
  cout<<endl;

  const boost::shared_ptr<Blob<float> > conv5_layer = net_->blob_by_name("conv5");
  begin_ = conv5_layer->cpu_data();
  cout<<"conv5:"<<conv5_layer->channels()<<"  "<<conv5_layer->height()<<"  "<<conv5_layer->width()<<endl;
  counter=10;
  for (int ii=0; ii<conv11_layer->channels()*conv11_layer->height()*conv11_layer->width(); ii++)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<"; ";
  }
  cout<<endl;
  counter=10;
  for (int ii=conv11_layer->channels()*conv11_layer->height()*conv11_layer->width()-1; ii>=0; ii--)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<": ";
  }
  cout<<endl;

  const boost::shared_ptr<Blob<float> > conv4_layer = net_->blob_by_name("conv4");
  begin_ = conv4_layer->cpu_data();
  cout<<"conv4:"<<conv4_layer->channels()<<"  "<<conv4_layer->height()<<"  "<<conv4_layer->width()<<endl;
  counter=10;
  for (int ii=0; ii<conv11_layer->channels()*conv11_layer->height()*conv11_layer->width(); ii++)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<"; ";
  }
  cout<<endl;
  counter=10;
  for (int ii=conv11_layer->channels()*conv11_layer->height()*conv11_layer->width()-1; ii>=0; ii--)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<": ";
  }
  cout<<endl;

  const boost::shared_ptr<Blob<float> > conv3_layer = net_->blob_by_name("conv3");
  begin_ = conv3_layer->cpu_data();
  cout<<"conv3:"<<conv3_layer->channels()<<"  "<<conv3_layer->height()<<"  "<<conv3_layer->width()<<endl;
  counter=10;
  for (int ii=0; ii<conv11_layer->channels()*conv11_layer->height()*conv11_layer->width(); ii++)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<"; ";
  }
  cout<<endl;
  counter=10;
  for (int ii=conv11_layer->channels()*conv11_layer->height()*conv11_layer->width()-1; ii>=0; ii--)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<": ";
  }
  cout<<endl;

  const boost::shared_ptr<Blob<float> > conv2_layer = net_->blob_by_name("conv2");
  begin_ = conv2_layer->cpu_data();
  cout<<"conv2:"<<conv2_layer->channels()<<"  "<<conv2_layer->height()<<"  "<<conv2_layer->width()<<endl;
  counter=10;
  for (int ii=0; ii<conv11_layer->channels()*conv11_layer->height()*conv11_layer->width(); ii++)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<"; ";
  }
  cout<<endl;
  counter=10;
  for (int ii=conv11_layer->channels()*conv11_layer->height()*conv11_layer->width()-1; ii>=0; ii--)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<": ";
  }
  cout<<endl;

  const boost::shared_ptr<Blob<float> > conv1_layer = net_->blob_by_name("conv1");
  begin_ = conv1_layer->cpu_data();
  cout<<"conv1:"<<conv1_layer->channels()<<"  "<<conv1_layer->height()<<"  "<<conv1_layer->width()<<endl;
  for (int ii=0; ii<40; ii+=4)
  {
      cout<<begin_[ii]<<", ";
  }
  cout<<endl;
  counter=10;
  for (int ii=0; ii<conv1_layer->channels()*conv1_layer->height()*conv1_layer->width(); ii++)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<"; ";
  }
  cout<<endl;
  counter=10;
  for (int ii=conv1_layer->channels()*conv1_layer->height()*conv1_layer->width()-1; ii>=0; ii--)
  {
      if (begin_[ii]>0 && counter-- >= 0)
          cout<<begin_[ii]<<": ";
  }
  cout<<endl;
  */
#endif
/////////////////////////



  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  cv::Mat ret(output_layer->channels(),1,CV_32F);
  //copy(begin,end,ret.data);
  float ss=0;
  for (int ii=0; ii<output_layer->channels(); ii++)
  {
      ss += begin[ii]*begin[ii];
      ret.at<float>(ii,0) = begin[ii];
  }
  if (ss!=0 && normalizeOut)
    ret /= sqrt(ss);
  for (int ii=0; ii<output_layer->channels(); ii++)
  {
      assert(ret.at<float>(ii,0) == ret.at<float>(ii,0));
#ifdef SHOW
      if (ii<10)
        cout<<ret.at<float>(ii,0)<<", ";
#endif
  }
#ifdef SHOW
  cout<<endl;
#endif
  
  return ret;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Embedder::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Embedder::Preprocess(const cv::Mat& img,
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

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}


int sort_xxx(const void *x, const void *y) {
    if (*(int*)x > *(int*)y) return 1;
    else if (*(int*)x < *(int*)y) return -1;
    else return 0;
}

void eval(const Dataset* data, Embedder* embedder)
{
    
    vector<cv::Mat> embeddings(data->size());
    //#pragma omp parallel  for
    for (int inst=0; inst<data->size(); inst++)
    {
#ifdef SHOW
        cout<<data->labels()[inst]<<": "<<data->image(inst).rows<<" X "<<data->image(inst).cols<<endl;
#endif
        embeddings[inst] = embedder->embed(data->image(inst));
    }
    float map=0;
    int queryCount=0;
    bool testtt=true;
    float maxAP=0;
    int maxIdx;
    //#pragma omp parallel  for
    for (int inst=0; inst<data->size(); inst++)
    {
        int other=0;
        string text = data->labels()[inst];
        for (int inst2=0; inst2<data->size(); inst2++)
        {
            if (inst!=inst2 && text.compare(data->labels()[inst2])==0)
            {
                other++;
            }
        }
        if (other==0)
            continue;

        int *rank = new int[other];//(int*)malloc(NRelevantsPerQuery[i]*sizeof(int));
        int Nrelevants = 0;
        float ap=0;

        float bestS=-99999;
        vector<float> scores(data->size());// = spot(data->image(inst),text,hy); //scores

        for (int j=0; j < data->size(); j++)
        {
            scores[j] = embeddings[inst].dot(embeddings[j]);
        }
        for (int j=0; j < data->size(); j++)
        {

            
            float s = scores[j];
            //if (testtt)
            //    cout<<j<<": "<<s<<endl;
            /* If it is from the same class and it is not the query idx, it is a relevant one. */
            /* Compute how many on the dataset get a better score and how many get an equal one, excluding itself and the query.*/
            if (text.compare(data->labels()[j])==0 && inst!=j)
            {
                int better=0;
                int equal = 0;

                for (int k=0; k < data->size(); k++)
                {
                    if (k!=j && inst!=k)
                    {
                        float s2 = scores[k];
                        if (s2> s) better++;
                        else if (s2==s) equal++;
                    }
                }


                rank[Nrelevants]=better+floor(equal/2.0);
                //if (testtt)
                 //   cout<<"  rel: "<<rank[Nrelevants]<<endl;
                Nrelevants++;
            }

        }
        qsort(rank, Nrelevants, sizeof(int), sort_xxx);
        //pP1[i] = p1;

        /* Get mAP and store it */
        for(int j=0;j<Nrelevants;j++){
            /* if rank[i] >=k it was not on the topk. Since they are sorted, that means bail out already */

            float prec_at_k =  ((float)(j+1))/(rank[j]+1);
            //if (testtt)
            //    printf("prec_at_k: %f\n", prec_at_k);
            ap += prec_at_k;
        }
        ap/=Nrelevants;

        //#pragma omp critical (storeMAP)
        {
            queryCount++;
            map+=ap;
            if (ap>maxAP)
            {
                maxAP=ap;
                maxIdx=inst;
            }
        }

        delete[] rank;
        testtt=false;
    }

    cout<<"map: "<<(map/queryCount)<<endl;
    cout<<"best was "<<maxIdx<<" at "<<maxAP<<" AP. Word is: "<<data->labels()[maxIdx]<<endl;
}

int main(int argc, char** argv) {
  if (argc != 5 &&argc!=6) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " images-labels.txt imagedir [-r (dont normalize)]" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string queries    = argv[3];
  string imdir   = argv[4];
  bool normalizeOut = argc<=5;
  Dataset* dataset = new GWDataset(queries,imdir);
  Embedder embedder(model_file, trained_file,normalizeOut);
  eval(dataset,&embedder);
  delete dataset;
}
