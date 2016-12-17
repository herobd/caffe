//g++ -std=c++11 -fopenmp gwdataset.cpp evalBigramClass.cpp -lcaffe -lglog -lopencv_core -lopencv_highgui -lopencv_imgproc -lprotobuf -lboost_system -I /home/brianld/include -I ../include/ -L ../build/lib/ -o evalBigramClass
#define CPU_ONLY
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <set>
#include <utility>
#include "gwdataset.h"
using namespace caffe;
using namespace std;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             //const string& mean_file,
             const string& label_file
             );

  std::vector<pair<string,float> > Classify(const cv::Mat& img, int N = 5);
  std::vector<float> Predict(const cv::Mat& img);

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

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       //const string& mean_file,
                       const string& label_file
                       ) {
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
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
  {
    for (int i=0; i<line.size(); i++)
        line[i]=tolower(line[i]);
    labels_.push_back(string(line));
  }

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

// Return the top N predictions. 
std::vector<pair<string,float> > Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<pair<string,float> > predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
/*void Classifier::SetMean(const string& mean_file) {
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

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  //cv::Mat ret(output_layer->channels(),1,CV_32F);
  vector<float> ret(output_layer->channels());
  //copy(begin,end,ret.data);
  for (int ii=0; ii<output_layer->channels(); ii++)
      ret.at(ii) = begin[ii];
  for (int ii=0; ii<output_layer->channels(); ii++)
      assert(ret.at(ii) == ret.at(ii));
  return ret;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
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

void Classifier::Preprocess(const cv::Mat& img,
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

void eval(const Dataset* data, Classifier* embedder)
{
    vector<string> classified(data->size());
    set<string> labels;
    //#pragma omp parallel for
    for (int inst=0; inst<data->size(); inst++)
    {
        labels.insert(data->labels()[inst]);
        cv::Mat imgFix;
        cv::resize(data->image(inst),imgFix,cv::Size(65,59));//size of training images
        imgFix = imgFix(cv::Rect(3,6,52,52));//(center) cropping of training images
        classified[inst] = embedder->Classify(imgFix,1)[0].first;
    }
    float acc=0;
    map<string, map<string,int> > confusionMat;
    int maxC=0;
    //#pragma omp parallel  for
    for (int inst=0; inst<data->size(); inst++)
    {
        string text = data->labels()[inst];
        if (text.compare(classified[inst])==0)
            acc++;
        if(++confusionMat[text][classified[inst]] > maxC)
            maxC=confusionMat[text][classified[inst]];
    }

    cout<<"acc: "<<(acc/data->size())<<endl;
    int bs=20;
    cv::Mat confImg(bs*labels.size(),bs*labels.size(),CV_8UC3);
    int rs=0;
    for (string label : labels)
    {
        int cs=0;
        for (string classified : labels)
        {
            int r=255.0*confusionMat[label][classified]/(0.0+maxC);
            int g=0;
            int b=0;
            cv::Vec3b color(b,g,r);
            for (int r=rs; r<rs+bs; r++)
                for (int c=cs; c<cs+bs; c++)
                    confImg.at<cv::Vec3b>(r,c)=color;
            cv::putText(confImg,label+","+classified,cv::Point(cs,rs),cv::FONT_HERSHEY_SIMPLEX,4.0,cv::Scalar(50,225,0));
            cs+=bs;
        }
        rs+=bs;
    }
    cv::imwrite("confusionMatrix.png",confImg);

}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " images-labels.txt imagedir labelsfile\n(h w are hardcoded)" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string queries    = argv[3];
  string imdir   = argv[4];
  string labelsfile   = argv[5];
  //int height = stoi(argv[5]);
  //int width = stoi(argv[6]);
  Dataset* dataset = new GWDataset(queries,imdir);
  Classifier embedder(model_file, trained_file, labelsfile);
  eval(dataset,&embedder);
  delete dataset;
}
