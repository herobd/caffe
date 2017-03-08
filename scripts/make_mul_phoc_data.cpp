//g++ -std=c++11 make_mul_phoc_data.cpp -lcaffe -lglog -l:libopencv_core.so.3.0 -l:libopencv_highgui.so.3.0 -l:libopencv_imgproc.so.3.0 -l:libopencv_imgcodecs.so.3.0 -lprotobuf -lleveldb -I ../include/ -L ../build/lib/ -o make_mul_phoc_data
// This script converts the dataset to the leveldb format used
// by caffe to train siamese network.
#define CPU_ONLY
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <list>
#include <set>
#include <vector>
#include <map>
#include <assert.h>
#include <tuple>
#include <regex>
#include <iostream>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
//#include "caffe/util/format.hpp"
#include "caffe/util/math_functions.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#if 1
#include "leveldb/db.h"

#define PER 10

using namespace std;

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

string serialize_image(cv::Mat& im) {
#ifdef DEBUG
        cv::imshow("image",im);
        cv::waitKey();
#endif
        assert(im.rows*im.cols>1);
        
        caffe::Datum datum;
        datum.set_channels(1);  
        datum.set_height(im.rows);
        datum.set_width(im.cols);
        //datum.set_label(label);
	//copy(((char*)im.data),((char*)im.data)+(rows*cols),pixels);	
        datum.set_data(im.data, im.rows*im.cols);
        string ret;

        datum.SerializeToString(&ret);
        return ret;
}
string read_image(string image_file) {
	cv::Mat im = cv::imread(image_file,CV_LOAD_IMAGE_GRAYSCALE);
#ifdef DEBUG
        cv::imshow("image",im);
        cv::waitKey();
#endif
        return serialize_image(im);
}
string prep_vec(vector<float> phoc) {
        
#ifdef DEBUG
        cout<<"phoc: "<<endl;
        for (float f : phoc)
            cout<<f<<", ";
        cout<<endl;
#endif
        caffe::Datum datum;
        datum.set_channels(phoc.size());  
        datum.set_height(1);
        datum.set_width(1);
        google::protobuf::RepeatedField<float>* datumFloatData = datum.mutable_float_data();
        for (float f : phoc)
            datumFloatData->Add(f);

        string ret;

        datum.SerializeToString(&ret);
        return ret;
}

void convert_dataset(vector<string>& image_filenames, vector<cv::Mat>& images,  vector<vector<float> >& phocs, vector<string>& labels,
	vector<string>& image_filenames2, vector<cv::Mat>& images2,  vector<vector<float> >& phocs2, vector<string>& labels2,
        const char* images_db_filename, const char* labels_db_filename, bool test) {
  // Open files
  // Read the magic and the meta data
  uint32_t num_items=0;
  int num_true=0;
  int num_false=0;
  uint32_t num_labels;


  // Open leveldb
  leveldb::DB* images_db;
  leveldb::DB* labels_db;
  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists = true;
  leveldb::Status status = leveldb::DB::Open(
      options, images_db_filename, &images_db);
  CHECK(status.ok()) << "Failed to open leveldb " << images_db_filename
      << ". Is it already existing?";
  leveldb::Status status2 = leveldb::DB::Open(
      options, labels_db_filename, &labels_db);
  CHECK(status2.ok()) << "Failed to open leveldb " << labels_db_filename
      << ". Is it already existing?";

  if (!test)
  {
      //Even word distribution, following after PHOCNet paper
      map<string,vector<int> > wordMap;
      for (int i=0; i<labels.size(); i++)
          wordMap[labels[i]].push_back(i);
      map<string,vector<int> > wordMap2;
      for (int i=0; i<labels2.size(); i++)
          wordMap2[labels2[i]].push_back(i);

      int maxCount=0;
      int averageCount=0;
      for (auto p : wordMap)
      {
          averageCount+=p.second.size();
          if (p.second.size() > maxCount)
              maxCount = p.second.size();
      }
      for (auto p : wordMap2)
      {
          averageCount+=p.second.size();
          if (p.second.size() > maxCount)
              maxCount = p.second.size();
      }
      averageCount /= wordMap.size() + wordMap2.size();
      int goalCount = 0.6*maxCount + 0.4*averageCount;

      for (auto p : wordMap)
      {
          int im=0;
          for (int i=p.second.size(); i<goalCount; i++)
          {
              if (image_filenames.size()>0)
                image_filenames.push_back(image_filenames[p.second[im]]);
              else
                images.push_back(images[p.second[im]]);
              phocs.push_back(phocs[p.second[im]]);
              labels.push_back(labels[p.second[im]]);
              im = (im+1)%p.second.size();
          }
      }
      for (auto p : wordMap2)
      {
          int im=0;
          for (int i=p.second.size(); i<goalCount; i++)
          {
              if (image_filenames2.size()>0)
                image_filenames2.push_back(image_filenames2[p.second[im]]);
              else
                images2.push_back(images2[p.second[im]]);
              phocs2.push_back(phocs2[p.second[im]]);
              labels2.push_back(labels2[p.second[im]]);
              im = (im+1)%p.second.size();
          }
      }
  }
    


  //char label_i;
  //char label_j;
  //char* pixels = new char[2 * rows * cols];
  //std::string value;

  //caffe::Datum datum;
  //datum.set_channels(2);  // one channel for each image in the pair
  //datum.set_height(rows);
  //datum.set_width(cols);
  //vector<bool> used(labels.size());
  list<int>toWrite;
  LOG(INFO) << "from " << labels.size() << " items.";
  for (int i=0; i<labels.size(); i++) {
      //int inst = caffe::caffe_rng_rand() % image_filenames.size();  // pick a random  
      //int start=inst;
      //while (used[inst])
      //    inst=(inst+1)%image_filenames.size();
      toWrite.push_back(i);
  }
  for (int i=0; i<labels2.size(); i++) {
      toWrite.push_back(-1*(i+1));
  }
  
  //write them in random order
  while (toWrite.size()>0) {
        int i = caffe::caffe_rng_rand() % toWrite.size();
        auto iter = toWrite.begin();
        for (int ii=0; ii<i; ii++) iter++;
        int im=(*iter);
        string label = prep_vec(im>=0?phocs[im]:phocs2[-1-im]);
        string value;
        if ((im>=0 && image_filenames.size()>0) || (im<0 && image_filenames2.size()>0))
            value = read_image(im>=0?image_filenames[im]:image_filenames2[-1-im]);
        else
            value = serialize_image(im>=0?images[im]:images2[-1-im]);
        char buff[10];
        snprintf(buff, sizeof(buff), "%08d", num_items);
        std::string key_str = buff; //caffe::format_int(num_items, 8);
        images_db->Put(leveldb::WriteOptions(), key_str, value);
        labels_db->Put(leveldb::WriteOptions(), key_str, label);
        num_items++;

        toWrite.erase(iter);
    
  }
  cout << "A total of    " << num_items << " items written."<<endl;

  delete images_db;
  delete labels_db;
}

#include "cnnspp_spotter/phocer.cpp"

void readin(string labelfile, string image_dir, vector<string>& image_paths, vector<cv::Mat>& images, vector<vector<float> >& phocs, vector<string>& labels)
{

    string extension = labelfile.substr(labelfile.find_last_of('.')+1);
    bool gtp=extension.compare("gtp")==0;

    ifstream filein(labelfile);
    assert(filein.good());
    //2700270.tif 519 166 771 246 orders
    string line;
    //regex qExtGtp("(\\S+\\.\\S+) (\\d+) (\\d+) (\\d+) (\\d+) (\\w+)");
    //regex qExt("(\\S+\\.\\S+) (\\w+)");
    PHOCer phocer;
    
    
    string curPathIm="";
    cv::Mat curIm;
    while (getline(filein,line))
    {
        if (gtp)
        {
            stringstream ss(line);
            string part;
            getline(ss,part,' ');

            string pathIm=image_dir+string(part);
            //pathIms.push_back(pathIm);
            
            if (curPathIm.compare(pathIm)!=0)
            {
                curPathIm=pathIm;
                curIm = cv::imread(curPathIm,CV_LOAD_IMAGE_GRAYSCALE);
                if (curIm.rows<1)
                {
                    cout<<"Error reading: "<<curPathIm<<endl;
                    assert(curIm.rows>0);
                }
            }
            getline(ss,part,' ');
            int x1=max(1,stoi(part));//;-1;
            getline(ss,part,' ');
            int y1=max(1,stoi(part));//;-1;
            getline(ss,part,' ');
            int x2=min(curIm.cols,stoi(part));//;-1;
            getline(ss,part,' ');
            int y2=min(curIm.rows,stoi(part));//;-1;
            cv::Rect loc(x1,y1,x2-x1+1,y2-y1+1);
            //locs.push_back(loc);
            if (x1<0 || x1>=x2 || x2>=curIm.cols)
                cout<<"line: ["<<line<<"]  loc: "<<x1<<" "<<y1<<" "<<x2<<" "<<y2<<"  im:"<<curIm.rows<<","<<curIm.cols<<endl;
            assert(x1>=0 && x1<x2);
            assert(x2<curIm.cols);
            assert(y1>=0 && y1<y2);
            assert(y2<curIm.rows);
            images.push_back(curIm(loc));
            getline(ss,part,' ');
            string label=part;
	    vector<float> phoc = phocer.makePHOC(label);
            phocs.push_back(phoc);
            labels.push_back(label);

        }
        else
        {
            smatch sm;
            //cout <<line<<endl;
            //regex_search(line,sm,qExt);
            stringstream ss(line);
            string part;
            getline(ss,part,' ');
            string pathIm=image_dir+part;
            getline(ss,part,' ');
            string label=part;
	    vector<float> phoc = phocer.makePHOC(label);
            image_paths.push_back(pathIm);
            phocs.push_back(phoc);
            labels.push_back(label);
#ifdef DEBUG
            //cout<<label<<": "<<endl;
            //for (float f : phoc)
            //    cout<<f<<", ";
            //cout<<endl;
            //int iii;
            //cin>>iii;
#endif
        }
    }

    filein.close();
}

int main(int argc, char** argv) {
  if (argc != 8 && argc!=9 ) {
    printf("This script converts the dataset to 2 leveldbs, one of images, one of phoc vectors.\n"
           "Usage:\n"
           " make_mul_phoc_data image_dir image_label_file image_dir2 image_label_file2 bigramfile"
           " out_image_db_file out_label_db_file [-t]\n"
           );
  } else {
    google::InitGoogleLogging(argv[0]);
    string image_dir=argv[1];
    string labelfile = argv[2];
    string image_dir2=argv[3];
    string labelfile2 = argv[4];
    string bigramfile = argv[5];

    ////
    vector<int> phoc_levels = {2, 3, 4, 5};
    vector<char> unigrams = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};
    if (true)
    {
        for (char n : "0123456789") {
            //cout << "added numeral "<<n<<endl;
            if (n!=0x00)
                unigrams.push_back(n);
        }
    }
    ///
    vector<string> image_paths;
    vector<cv::Mat> images;
    vector<vector<float> > phocs;
    vector<string> labels;
    readin(labelfile,image_dir,image_paths,images,phocs,labels);

    vector<string> image_paths2;
    vector<cv::Mat> images2;
    vector<vector<float> > phocs2;
    vector<string> labels2;
    readin(labelfile2,image_dir2,image_paths2,images2,phocs2,labels2);

    convert_dataset(image_paths,images,phocs,labels,
		    image_paths2,images2,phocs2,labels2,
                    argv[6], argv[7],argc>8);
  }
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires LevelDB; compile with USE_LEVELDB.";
}
#endif  // USE_LEVELDB
