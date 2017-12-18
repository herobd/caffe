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
  : DataLayer<Dtype>(param), lastHeight(-1), lastWidth(-1) {
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
  int max_height = this->layer_param_.multisizedata_param().max_height();
  int min_height = this->layer_param_.multisizedata_param().min_height();
  int max_width = this->layer_param_.multisizedata_param().max_width();
  int min_width = this->layer_param_.multisizedata_param().min_width();
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
      //if (top_shapes[item_id][2]>max_height)
      //    max_height=top_shapes[item_id][2];
      //if (top_shapes[item_id][2]<min_height)
      //    min_height=top_shapes[item_id][2];
      //if (top_shapes[item_id][3]>max_width)
      //    max_width=top_shapes[item_id][3];
      //if (top_shapes[item_id][3]<min_width)
      //    min_width=top_shapes[item_id][3];
  }
  sdH = sqrt(sdH/batch_size)/1.5;//tighten distribution
  sdW = sqrt(sdW/batch_size)/1.5;
  //normal_distribution<float> distributionH(avgH,sdH);
  //normal_distribution<float> distributionW(avgW,sdW);
  vector<int> top_shape = top_shapes[0];
  //test
  //top_shape[2]=std::min(std::max((double)(rng.gaussian(sdH)+avgH), (double)min_size),(double)max_size); //max(distriburionH(randgen),10.0f);
  //top_shape[3]=std::min(std::max((double)(rng.gaussian(sdW)+avgW), (double)min_size),(double)max_size); //max(distriburionW(randgen),10.0f);
  float newH, newW;
  caffe_rng_gaussian(1,avgH,std::max(0.0001f,sdH),&newH);
  caffe_rng_gaussian(1,avgW,std::max(0.0001f,sdW),&newW);
  top_shape[2]=std::min(std::max(newH, (float)min_height),(float)max_height); //max(distriburionH(randgen),10.0f);
  top_shape[3]=std::min(std::max(newW, (float)min_width),(float)max_width); //max(distriburionW(randgen),10.0f);
    //if (top_shape[3]>500)
    //    std::cout<<"batch width: "<<top_shape[3]<<" max is "<<max_width<<std::endl;
    //if (top_shape[2]>500)
    //    std::cout<<"batch height: "<<top_shape[2]<<" max is "<<max_height<<std::endl;
  //std::cout<<"h:"<<top_shape[2]<<" w:"<<top_shape[3]<<std::endl;

  //assert(top_shape[2]>15 && top_shape[3]>15);//for debugging, not a necessary condition
  //assert(top_shape[2]<=64 && top_shape[3]<=64);//for debugging, not a necessary condition
  //std::cout<<"multisize: "<<top_shape[2]<<", "<<top_shape[3]<<std::endl;

  //This prevents reshaping when the change is little
  int min_height_change = this->layer_param_.multisizedata_param().min_height_change();
  int min_width_change = this->layer_param_.multisizedata_param().min_width_change();
  if (lastHeight>0 && lastWidth>0 && 
          std::abs(lastHeight-top_shape[2])<min_height_change &&
          std::abs(lastWidth-top_shape[3])<min_width_change ) {
      top_shape[2]=lastHeight;
      top_shape[3]=lastWidth;
  }
  else {
      lastHeight=top_shape[2];
      lastWidth=top_shape[3];
  }

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
      //int top_index
      int data_index;
      for (int c = 0; c < datum_channels; ++c) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
            //top_index = (c * height + h) * width + w;
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
    CHECK(top_shape[3]>0 && top_shape[2]>0);
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
