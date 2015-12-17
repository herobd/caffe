#include <functional>
#include <utility>
#include <vector>

#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
namespace caffe {

#define NUM_CLASSES 201
template <typename Dtype>
AccuracyLayer<Dtype>::~AccuracyLayer() {

  //Brian
  cv::Mat out(NUM_CLASSES,NUM_CLASSES,CV_8U);
  float maxC=0;
  float minC=9999999;
  for (int r=0; r<NUM_CLASSES; r++)
    for (int c=0; c<NUM_CLASSES; c++) {
      float v = confusion[r][c];
      if (v<minC) minC=v;
      if (v>maxC) maxC=v;
    }
  for (int r=0; r<NUM_CLASSES; r++)
    for (int c=0; c<NUM_CLASSES; c++) {
      float v = confusion[r][c];
      out.at<unsigned char>(r,c)=255*(v-minC)/maxC;
    }
  cv::imwrite("confusion_standard.png",out);
  std::cout << "conf image written" << std::endl;
  for (int i=0; i<NUM_CLASSES; i++) {
    delete[] confusion[i];
    delete[] conf_mutex[i];
  }
  delete[] confusion;
  delete[] conf_mutex;
  //Layer<Dtype>::~Layer();
}

template <typename Dtype>
void AccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }


  confusion = new int*[NUM_CLASSES];
  conf_mutex = new pthread_mutex_t*[NUM_CLASSES];
  for (int i=0; i<NUM_CLASSES; i++) {
    confusion[i]=new int[NUM_CLASSES];
    conf_mutex[i]=new pthread_mutex_t[NUM_CLASSES];
    for (int ii=0; ii<NUM_CLASSES; ii++) {
      confusion[i][ii]=0;
      if(pthread_mutex_init(&(conf_mutex[i][ii]), NULL))
      {
        std::cout<<"Could not create a mutex"<<std::endl;
        return;
      } 
    }
  }
  std::cout << "conf image created" << std::endl;
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    // Per-class accuracy is a vector; 1 axes.
    vector<int> top_shape_per_class(1);
    top_shape_per_class[0] = bottom[0]->shape(label_axis_);
    top[1]->Reshape(top_shape_per_class);
    nums_buffer_.Reshape(top_shape_per_class);
  }
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);
  if (top.size() > 1) {
    caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  }
  int count = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      if (top.size() > 1) ++nums_buffer_.mutable_cpu_data()[label_value];
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);
      // Top-k accuracy
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < num_labels; ++k) {
        bottom_data_vector.push_back(std::make_pair(
            bottom_data[i * dim + k * inner_num_ + j], k));
      }
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      // check if true label is in top k predictions
      for (int k = 0; k < top_k_; k++) {
        if (label_value<0||label_value>(NUM_CLASSES-1)||bottom_data_vector[k].second<0||bottom_data_vector[k].second>(NUM_CLASSES-1)) std::cout<<"bad classes "<<bottom_data_vector[k].second<<" and "<<label_value<<std::endl;
        //pthread_mutex_lock(&(conf_mutex[label_value][bottom_data_vector[k].second]));
        pthread_mutex_lock(&(conf_mutex[0][0]));
          confusion[label_value][bottom_data_vector[k].second]+=1;
        //pthread_mutex_unlock(&(conf_mutex[label_value][bottom_data_vector[k].second]));
        pthread_mutex_unlock(&(conf_mutex[0][0]));
        if (bottom_data_vector[k].second == label_value) {
          ++accuracy;
          if (top.size() > 1) ++top[1]->mutable_cpu_data()[label_value];
          break;
        }
      }
      ++count;
    }
  }

  // Accuracy layer should not be used as a loss function.
  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / count;
  if (top.size() > 1) {
    for (int i = 0; i < top[1]->count(); ++i) {
      top[1]->mutable_cpu_data()[i] =
          nums_buffer_.cpu_data()[i] == 0 ? 0
          : top[1]->cpu_data()[i] / nums_buffer_.cpu_data()[i];
    }
  }
}

INSTANTIATE_CLASS(AccuracyLayer);
REGISTER_LAYER_CLASS(Accuracy);

}  // namespace caffe
