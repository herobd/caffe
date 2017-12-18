#ifndef CAFFE_MULTISIZEDATA_LAYER_HPP_
#define CAFFE_MULTISIZEDATA_LAYER_HPP_

#ifdef USE_OPENCV
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

#include "caffe/util/math_functions.hpp"
//#include "caffe/util/rng.hpp"
//#include <random>

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
  int lastWidth, lastHeight;

  //DataReader reader_;
  //default_random_engine randgen;
  //cv::RNG rng;
};
#endif

}  // namespace caffe

#endif  // CAFFE_MULTISIZEDATA_LAYER_HPP_
