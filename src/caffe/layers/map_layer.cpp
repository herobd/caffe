#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/map_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MAPLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //top_k_ = this->layer_param_.accuracy_param().top_k();

  //has_ignore_label_ =
  //  this->layer_param_.accuracy_param().has_ignore_label();
  //if (has_ignore_label_) {
  //  ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  //}
}

template <typename Dtype>
void MAPLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
  //    << "top_k must be less than or equal to the number of classes.";
  //label_axis_ =
  //    bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  //outer_num_ = bottom[0]->count(0, label_axis_);
  //inner_num_ = bottom[0]->count(label_axis_ + 1);
  //CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
  //    << "Number of labels must match number of predictions; "
  //    << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
  //    << "label count (number of labels) must be N*H*W, "
  //    << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // MAP is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    // Per-class ap is a vector; 1 axes.
    vector<int> top_shape_per_class(1);
    top_shape_per_class[0] = ??;//bottom[0]->shape(label_axis_);
    top[1]->Reshape(top_shape_per_class);
    nums_buffer_.Reshape(top_shape_per_class);
  }
}

template <typename Dtype>
void MAPLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_data2 = bottom[1]->cpu_data();
  const Dtype* bottom_label = bottom[2]->cpu_data();
  //const int dim = bottom[0]->count() / outer_num_;
  //const int num_labels = ??://bottom[0]->shape(label_axis_);
  const int emb_size = bottom[0]->shape(1);
  DCHECK_EQ(bottom[1]->shape(1),emb_size);
  const int batch_size = bottom[0]->shape(0);
  DCHECK_EQ(bottom[2]->shape[0],batch_size);
  if (top.size() > 1) {
    caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  }
  std::map<int,std::vector<std::vector<float>>> embeddingsByLabel;
  int count = 0;
  for (int i = 0; i < batch_size; ++i) {
    //for (int j = 0; j < inner_num_; ++j) {
      const int label_value =
          static_cast<int>(bottom_label[i]);
      //if (has_ignore_label_ && label_value == ignore_label_) {
      //  continue;
      //}
      if (top.size() > 1) nums_buffer_.mutable_cpu_data()[label_value]+=2;
      //assert (top.size() == 1);
      DCHECK_GE(label_value, 0);
      //DCHECK_LT(label_value, num_labels);
      // Top-k accuracy
      //std::vector<std::pair<Dtype, int> > bottom_data_vector;
      std::vector<float>emb1(emb_size);
      float norm=0;
      for (int k = 0; k < emb_size; ++k) { //num labels
        //bottom_data_vector.push_back(std::make_pair(
        //    bottom_data[i * dim + k * inner_num_ + j], k));
          emb1.at(k) = bottom_data[(i*emb_size) +k];
          norm+= emb1.at(k)*emb1.at(k);
      }
      norm=std::sqrt(norm);
      for (int k = 0; k < emb_size; ++k) {
          emb1.at(k) /= norm;
      }
      std::vector<float>emb2(emb_size);
      norm=0;
      for (int k = 0; k < emb_size; ++k) { 
          emb2.at(k) = bottom_data2[(i*emb_size) +k];
          norm+= emb2.at(k)*emb2.at(k);
      }
      norm=std::sqrt(norm);
      for (int k = 0; k < emb_size; ++k) {
          emb2.at(k) /= norm;
      }
      embeddingsByLabel[label_value].push_back(emb1);
      embeddingsByLabel[label_value].push_back(emb2);
      /*
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      // check if true label is in top k predictions
      for (int k = 0; k < top_k_; k++) {
        if (bottom_data_vector[k].second == label_value) {
          ++accuracy;
          if (top.size() > 1) ++top[1]->mutable_cpu_data()[label_value];
          break;
        }
      }
      ++count;*/
    //}
  }

  float map=0;
  int mapCount=0;
  int pos=0;
  for (auto p : embeddingsByLabel)
  {
      if (p.second.size()>10)
      {
          std::vector<float> scores;
          std::vector<bool> rel;
          for (int inst=0; inst<p.second.size(); inst++)
          {
              const std::vector<float>& exem = p.second.at(inst);
              int pos2=0;
              for (auto p2 : embeddingsByLabel)
              {
                  for (int inst2=0; inst2<p2.second.size(); inst2++)
                  {
                      if (pos!=pos2 || inst!=inst2)
                      {
                        const std::vector<float>& em = p2.second.at(inst2);
                        float score=0;
                        for (int i=0; i<emb_size; i++)
                            score+=exem.at(i)*em.at(i);
                        scores.push_back(score);
                        rel.push_back(pos2==pos);
                      }
                  }
                  pos2++;
              }
          }

        vector<int> rank;
        int Nrelevants=0;
        for (int j=0; j < scores.size(); j++)
        {
            float s = scores[j];

            if (rel[j])
            {
                int better=0;
                int equal = 0;

                for (int k=0; k < scores.size(); k++)
                {
                    if (k!=j)
                    {
                        float s2 = scores[k];
                        if (s2> s) better++;
                        else if (s2==s) equal++;
                    }
                }


                rank.push_back(better+floor(equal/2.0));
                Nrelevants++;

            }

        }
        sort(rank.begin(), rank.end());


        float ap=0;
        /* Get mAP and store it */
        for(int j=0;j<Nrelevants;j++){

            float prec_at_k =  ((float)(j+1))/(rank[j]+1);
            ap += prec_at_k;
        }
        ap/=Nrelevants;
        map+=ap;
        mapCount++;
      }
      pos++;
  }

  // LOG(INFO) << "MAP: " << accuracy;
  top[0]->mutable_cpu_data()[0] = map / mapCount;
  /*if (top.size() > 1) {
    for (int i = 0; i < top[1]->count(); ++i) {
      top[1]->mutable_cpu_data()[i] =
          nums_buffer_.cpu_data()[i] == 0 ? 0
          : top[1]->cpu_data()[i] / nums_buffer_.cpu_data()[i];
    }
  }*/
  // MAP layer should not be used as a loss function.
}

INSTANTIATE_CLASS(MAPLayer);
REGISTER_LAYER_CLASS(MAP);

}  // namespace caffe
