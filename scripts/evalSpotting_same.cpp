//g++ -std=c++11 -fopenmp gwdataset.cpp evalBigramClass.cpp -lcaffe -lglog -lopencv_core -lopencv_highgui -lopencv_imgproc -lprotobuf -lboost_system -I /home/brianld/include -I ../include/ -L ../build/lib/ -o evalBigramClass

//This is to evaluate an anomalas network which was given two images as channels and learned to classify if they were the same bigram or not.

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
  num_channels_ = input_layer->channels()*2;
  CHECK(num_channels_%2==0)
    << "Input layer should have 2 images as channels.";
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

std::vector<float> Classifier::Predict(const cv::Mat& img1, const cv::Mat& img2) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img1, &input_channels,0);
  Preprocess(img2, &input_channels,1);

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
                            std::vector<cv::Mat>* input_channels,
                            int imgNum) {
  /* Convert the input image to the input image format of the network. */
    //This adds the image to the caffe channel specified by imgNum

  CHECK(img.channels()==1) << "Input images should have only 1 channel. I see " << img.channels();
  cv::Mat sample;
  /*
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
  */
    sample = img;

  cv::Mat sample_resized;
  //if (sample.size() != input_geometry_)
  //  cv::resize(sample, sample_resized, input_geometry_);
  //else
    sample_resized = sample;

  cv::Mat sample_float;
  //if (num_channels_ == 3)
  //  sample_resized.convertTo(sample_float, CV_32FC3);
  //else
    sample_resized.convertTo(sample_float, CV_32FC1);

  //cv::Mat sample_normalized;
  //cv::subtract(sample_float, mean_, sample_normalized);
  sample_float*=0.00390625;
  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  //cv::split(sample_float, *input_channels);
  input_channels->at(imgNum)=sample_float;

  //CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
  //      == net_->input_blobs()[0]->cpu_data())
  //  << "Input channels are not wrapping the input layer of the network.";
}


int sort_xxx(const void *x, const void *y) {
    if (*(int*)x > *(int*)y) return 1;
    else if (*(int*)x < *(int*)y) return -1;
    else return 0;
}

void eval(const Dataset* data, Classifier* embedder)
{
    
    vector<cv::Mat> imgFix(data->size());
    //#pragma omp parallel  for
    for (int inst=0; inst<data->size(); inst++)
    {
        cv::resize(data->image(inst),imgFix[inst],cv::Size(52,52));//size of training images
        //imgFix = imgFix(cv::Rect(3,6,52,52));//(center) cropping of training images
        //classified[inst] = embedder->Classify(imgFix,1)[0].first;
    }
    float mapSame=0;
    float mapDif=0;
    float mapComb=0;
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

        int *rankSame = new int[other];
        int *rankDif = new int[other];
        int *rankComb = new int[other];
        int Nrelevants = 0;
        float apSame=0;
        float apDif=0;
        float apComb=0;

        float bestS=-99999;
        vector<float> scoresSame(data->size());
        vector<float> scoresDif(data->size());
        vector<float> scoresComb(data->size());

        for (int j=0; j < data->size(); j++)
        {
            vector<float> acts = embedder->Predict(imgFix[inst],imgFix[inst2]);
            scoresSame[j] = acts[1];
            scoresDif[j] = -1*acts[0];
            scoresComb[j] = acts[1]-acts[0];

        }
        for (int j=0; j < data->size(); j++)
        {

            
            float sSame = scoresSame[j];
            float sDif = scoresDif[j];
            float sComb = scoresComb[j];
            //if (testtt)
            //    cout<<j<<": "<<s<<endl;
            /* If it is from the same class and it is not the query idx, it is a relevant one. */
            /* Compute how many on the dataset get a better score and how many get an equal one, excluding itself and the query.*/
            if (text.compare(data->labels()[j])==0 && inst!=j)
            {
                int betterSame=0;
                int equalSame = 0;

                int betterDif=0;
                int equalDif = 0;

                int betterComb=0;
                int equalComb = 0;

                for (int k=0; k < data->size(); k++)
                {
                    if (k!=j && inst!=k)
                    {
                        float s2 = scoresSame[k];
                        if (s2> sSame) betterSame++;
                        else if (s2==sSame) equalSame++;

                        s2 = scores[k];
                        if (s2> sDif) betterDif++;
                        else if (s2==sDif) equalDif++;

                        s2 = scores[k];
                        if (s2> sComb) betterComb++;
                        else if (s2==sComb) equalComb++;
                    }
                }


                rankSame[Nrelevants]=betterSame+floor(equalSame/2.0);
                rankDif[Nrelevants]=betterDif+floor(equalDif/2.0);
                rankComb[Nrelevants]=betterComb+floor(equalComb/2.0);
                //if (testtt)
                 //   cout<<"  rel: "<<rank[Nrelevants]<<endl;
                Nrelevants++;
            }

        }
        qsort(rankSame, Nrelevants, sizeof(int), sort_xxx);
        qsort(rankDif, Nrelevants, sizeof(int), sort_xxx);
        qsort(rankComb, Nrelevants, sizeof(int), sort_xxx);
        //pP1[i] = p1;

        /* Get mAP and store it */
        for(int j=0;j<Nrelevants;j++){
            /* if rank[i] >=k it was not on the topk. Since they are sorted, that means bail out already */

            float prec_at_k =  ((float)(j+1))/(rankSame[j]+1);
            apSame += prec_at_k;

            prec_at_k =  ((float)(j+1))/(rankDif[j]+1);
            apDif += prec_at_k;

            prec_at_k =  ((float)(j+1))/(rankComb[j]+1);
            apComb += prec_at_k;
        }
        apSame/=Nrelevants;
        apDif/=Nrelevants;
        apComb/=Nrelevants;

        //#pragma omp critical (storeMAP)
        {
            queryCount++;
            mapSame+=apSame;
            mapDif+=apDif;
            mapComb+=apComb;
            //if (ap>maxAP)
            //{
            //    maxAP=ap;
            //    maxIdx=inst;
            //}
        }

        delete[] rank;
        testtt=false;
    }

    cout<<"mapSame: "<<(mapSame/queryCount)<<endl;
    cout<<"mapDif: "<<(mapDif/queryCount)<<endl;
    cout<<"mapComb: "<<(mapComb/queryCount)<<endl;
    //cout<<"best was "<<maxIdx<<" at "<<maxAP<<" AP. Word is: "<<data->labels()[maxIdx]<<endl;
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