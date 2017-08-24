#include "cnn_spotter.h"


CNNSpotter::CNNSpotter(const string& model_file,
                       const string& trained_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

}



cv::Mat CNNSpotter::spot(const vector<float>& features_query, vector< std::vector<cv::Mat>*> features_pages)  {
  CHECK_EQ(features_query.size(), num_channels_) << "Input has incorrect number of channels.";
  CHECK_EQ(features_query.size(),features_pages[0]->size()) << "Inputs have unequal features.";
  CHECK_EQ(net_->num_inputs(), features_pages.size()+1) << "Unequal net inputs and supplied feature maps.";
  //assert(img.cols>=input_geometry_.width); appearently this isn't important?
  //cout<<img.rows<<" , "<<img.cols<<endl;
  Blob<float>* input_layer_query = net_->input_blobs()[0];
  //input_layer_query->Reshape(1, num_channels_,
  //                     features_query->front().rows, features_query->front().cols);
  vector<Blob<float>*> input_layer_pages(features_pages.size());
  for (int f=0; f<features_pages.size(); f++)
  {
     input_layer_pages[f] = net_->input_blobs()[1+f];
     input_layer_pages[f]->Reshape(1, num_channels_,
                           features_pages[f]->front().rows, features_pages[f]->front().cols);
  }
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels_query;
  WrapInputLayer(input_layer_query,&input_channels_query);
  Preprocess(features_query, &input_channels_query);

  vector<std::vector<cv::Mat> > input_channels_pages(features_pages.size());
  for (int f=0; f<features_pages.size(); f++)
  {
       WrapInputLayer(input_layer_pages[f],&(input_channels_pages[f]));
       Preprocess(features_pages[f], &(input_channels_pages[f]));
  }


  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  
  const float* begin = output_layer->cpu_data();
  //const float* end = begin + output_layer->channels();
  cv::Mat ret(output_layer->height(), output_layer->width(), CV_32F);
  int ii=0;
  for (int y=0; y<output_layer->height(); y++)
      for (int x=0; x<output_layer->width(); x++)
      {
          assert(begin[ii]==begin[ii]);
          ret.at<float>(y,x) = begin[ii];
          ii++;
      }
  return ret;
}

/*
cv::Mat CNNSpotter::spot(const std::vector< std::vector<cv::Mat> >& batchFeatures)  {
  //assert(img.cols>=input_geometry_.width); appearently this isn't important?
  //cout<<img.rows<<" , "<<img.cols<<endl;
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(batchFeatures.size(), num_channels_,
                       batchFeatures.front().front().rows, batchFeatures.front().front().cols);
  // Forward dimension change to all layers. 
  net_->Reshape();

  std::vector< std::vector<cv::Mat> > input_channels(batchFeatures.size());
  WrapInputLayer(&input_channels);

  for (int bi=0; bi<batchFeatures.size(); bi++) {
      CHECK_EQ(batchFeatures[bi].size(), num_channels_) << "Input has incorrect number of channels.";
      assert(batchFeatures[bi].front().cols*batchFeatures[bi].front().rows>1);

    Preprocess(&batchFeatures[bi], &(input_channels[bi]));
  }

  net_->Forward();

  // Copy the output layer to a std::vector
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  //const float* end = begin + output_layer->channels();
  CHECK_EQ(output_layer->width(),1) << "CNNSpotter assumes output of vector";
  CHECK_EQ(output_layer->height(),1) << "CNNSpotter assumes output of vector";
  //std::vector<cv::Mat> ret(batchFeatures.size());
  cv::Mat ret (output_layer->channels(),batchFeatures.size(),CV_32F);
  for (int bi=0; bi<batchFeatures.size(); bi++) {
      //ret[bi] = cv::Mat(output_layer->channels(),1,CV_32F);
      //assert(output_layer->channels()==52);
      //copy(begin,end,ret.data);
      float ss=0;
      int ii=0;
      for (int r=0; r<output_layer->channels(); r++)
      {
          assert(begin[ii]==begin[ii]);
          ret.at<float>(r,bi) = begin[ii];
          ss+=begin[ii]*begin[ii];
          ii++;
      }
      //for (int ii=0; ii<output_layer->channels(); ii++)
      //    assert(ret.at<float>(ii,0) == ret.at<float>(ii,0));
      if (normalize) {
          ss = sqrt(ss);
          if (ss!=0)
            ret.col(bi) /= ss;

      }
  }
  return ret;
}*/

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void CNNSpotter::WrapInputLayer(Blob<float>* input_layer,std::vector< std::vector<cv::Mat> >* input_channels) {

  int width = input_layer->width();
  int height = input_layer->height();
  //assert(input_channels->front().cols==width);
  float* input_data = input_layer->mutable_cpu_data();
  for (int bi=0; bi<input_channels->size(); bi++)
  {
      //int offset = input_layer->offset(bi);
      for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->at(bi).push_back(channel);
        input_data += width * height;
      }
  }
}

void CNNSpotter::WrapInputLayer(Blob<float>* input_layer,std::vector<cv::Mat>* input_channels) {

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

void CNNSpotter::Preprocess(const std::vector<cv::Mat>* features,
                            std::vector<cv::Mat>* input_channels) {
  for (int i=0; i<num_channels_; i++)
  {
      features->at(i).copyTo(input_channels->at(i));
  }

  //CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
  //      == net_->input_blobs()[0]->cpu_data() + offset)
  //  << "Input channels are not wrapping the input layer of the network. Image["<<features->front().rows<<","<<features->front().cols<<"]["<<features->front().channels()<<"]";
}


void CNNSpotter::Preprocess(const vector<float>& features,
                            std::vector<cv::Mat>* input_channels) {
  for (int i=0; i<num_channels_; i++)
  {
      input_channels->at(i)=features.at(i);
  }

}
