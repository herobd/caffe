#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>
//
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/multisizedata_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
MultiSizeDataLayer<Dtype>::MultiSizeDataLayer(const LayerParameter& param)
  : DataLayer<Dtype>(param) {
}


// This function is called on prefetch thread
template<typename Dtype>
void MultiSizeDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  //const int max_size = this->layer_param_.data_param().max_size();
  vector<Datum*> datums(batch_size);
  vector< vector<int> > top_shapes(batch_size);
  float avgH=0;
  float avgW=0;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    datums[item_id] = (this->reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    top_shapes[item_id] = this->data_transformer_->InferBlobShape(*datums[item_id]);
    CHECK(top_shapes[item_id][2]==datums[item_id]->height() && top_shapes[item_id][3]==datums[item_id]->width()) << "MultiSizeDataLayer assumes data transformations do not effect size.";
    avgH+=top_shapes[item_id][2];
    avgW+=top_shapes[item_id][3];
  }
  avgH/=batch_size;
  avgW/=batch_size;
  float sdH=0;
  float sdW=0;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
      sdH += pow(avgH-top_shapes[item_id][2],2);
      sdW += pow(avgW-top_shapes[item_id][3],2);
  }
  sdH = sqrt(sdH/batch_size)/1.5;//tighten distribution
  sdW = sqrt(sdW/batch_size)/1.5;
  //normal_distribution<float> distributionH(avgH,sdH);
  //normal_distribution<float> distributionW(avgW,sdW);
  vector<int> top_shape = top_shapes[0];
  //test
  //top_shape[2]=std::max((double)(rng.gaussian(sdH)+avgH), (double)10.0); //max(distriburionH(randgen),10.0f);
  //top_shape[3]=std::max((double)(rng.gaussian(sdW)+avgW), (double)10.0); //max(distriburionW(randgen),10.0f);

  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    //First we must transform the datum in our batch to the same size
    
    int datum_channels = datums[item_id]->channels();
    CHECK_EQ(datum_channels,1) <<"MultiSizeDataLayer only implemented from single channel images.";
      const string& data = datums[item_id]->data();
      const int datum_height = datums[item_id]->height();
      const int datum_width = datums[item_id]->width();
      const int height = datum_height;
      const int width = datum_width;
      int h_off = 0;
      int w_off = 0;
      const bool has_uint8 = data.size() > 0;
    cv::Mat_<Dtype> fitted(datum_height,datum_width);
    //cv::Mat_<Dtype> datum_mat(datum_height,datum_width,datums[item_id]->mutable_cpu_data());
      Dtype datum_element;
      int top_index, data_index;
      for (int c = 0; c < datum_channels; ++c) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
            top_index = (c * height + h) * width + w;
            if (has_uint8) {
              datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            } else {
              datum_element = datums[item_id]->float_data(data_index);
            }
            fitted(h,w) = datum_element;
          }
        }
      }
    //test
    cv::resize(fitted,fitted,cv::Size(top_shape[3],top_shape[2]));

    cv::Mat toTrans;
    //cv::resize(datum_mat,toTrans,cv::Size(top_shape[3],top_shape[2]));

    fitted.convertTo(toTrans, CV_8U);    
    //toTrans.convertTo(toTrans, CV_8U);   
    //CVMatToDatum(toTrans, datums[item_id]);
    
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    //test
    this->data_transformer_->Transform(toTrans, &(this->transformed_data_));
    //this->data_transformer_->Transform(*datums[item_id], &(this->transformed_data_));
    // Copy label.
    if (this->output_labels_) {
      top_label[item_id] = datums[item_id]->label();
    }
    trans_time += timer.MicroSeconds();

    this->reader_.free().push(const_cast<Datum*>(datums[item_id]));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(MultiSizeDataLayer);
REGISTER_LAYER_CLASS(MultiSizeData);

}  // namespace caffe
