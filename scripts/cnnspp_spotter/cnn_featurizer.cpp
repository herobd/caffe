#include "cnn_featurizer.h"


CNNFeaturizer::CNNFeaturizer(const string& model_file,
                       const string& trained_file,
                       int gpu
                       ) {
  if (gpu<0)
      Caffe::set_mode(Caffe::CPU);
  else
  {
      Caffe::set_mode(Caffe::GPU);
      Caffe::SetDevice(gpu);
  }

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


std::vector<cv::Mat>* CNNFeaturizer::featurize(const cv::Mat& img) {
  assert(img.cols*img.rows>1);
  //cout<<img.rows<<" , "<<img.cols<<endl;
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       img.rows, img.cols);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  //cv::Mat ret(output_layer->channels(),output_layer->width(),CV_32F);
  vector<cv::Mat>* ret = new vector<cv::Mat>(output_layer->channels());
  //copy(begin,end,ret.data);
  int ii=0;
  for (int c=0; c<output_layer->channels(); c++)
  {
      ret->at(c) = cv::Mat(output_layer->height(),output_layer->width(),CV_32F);
      for (int y=0; y<output_layer->height(); y++)
      for (int x=0; x<output_layer->width(); x++)
      {
          assert(begin[ii]==begin[ii]);
          ret->at(c).at<float>(y,x) = begin[ii];
          ii++;
      }
  }
  
  return ret;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void CNNFeaturizer::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  //assert(input_channels->front().cols==width);
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void CNNFeaturizer::Preprocess(const cv::Mat& img,
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
    << "Input channels are not wrapping the input layer of the network. Image["<<img.rows<<","<<img.cols<<"]["<<img.channels()<<"]";
}


