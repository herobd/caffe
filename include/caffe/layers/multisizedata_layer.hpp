#ifndef CAFFE_MULTISIZEDATA_LAYER_HPP_
#define CAFFE_MULTISIZEDATA_LAYER_HPP_

#include <random>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"


namespace caffe {

template <typename Dtype>
class MultiSizeDataLayer : public DataLayer<Dtype> {
 public:
  explicit MultiSizeDataLayer(const LayerParameter& param);
  //virtual ~MultiSizeDataLayer();
  //virtual void MultiSizeDataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
  //    const vector<Blob<Dtype>*>& top);
  // MultiSizeDataLayer uses DataReader instead for sharing for parallelism
  //virtual inline bool ShareInParallel() const { return false; }
  //virtual inline const char* type() const { return "Data"; }
  //virtual inline int ExactNumBottomBlobs() const { return 0; }
  //virtual inline int MinTopBlobs() const { return 1; }
  //virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  //DataReader reader_;
  default_random_engine randgen;
};

}  // namespace caffe

#endif  // CAFFE_MULTISIZEDATA_LAYER_HPP_
