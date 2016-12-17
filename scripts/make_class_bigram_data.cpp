//g++ -std=c++11 make_class_bigram_data.cpp -lcaffe -lglog -lopencv_core -lopencv_highgui -lopencv_imgproc -lprotobuf -lleveldb -I /home/brianld/include -I ../include/ -L ../build/lib/ -o make_class_bigram_data
// This script converts the dataset to the leveldb format used
// by caffe to train classification network.
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

#if 1
#include "leveldb/db.h"

#define PER 10

using namespace std;

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

void read_image(string image_file,
        uint32_t rows, uint32_t cols,
        char* pixels) {
	cv::Mat im = cv::imread(image_file,CV_LOAD_IMAGE_GRAYSCALE);
        assert(im.rows*im.cols>1);
	//resize
	cv::resize(im,im,cv::Size(rows,cols));

	copy(((char*)im.data),((char*)im.data)+(rows*cols),pixels);	
}

void convert_dataset(const vector<string>& image_filenames, const vector<string>& labels, const vector<string>& bigramClasses,
        const char* db_filename, uint32_t rows, uint32_t cols) {
  // Open files
  // Read the magic and the meta data
  uint32_t num_items=0;
  int num_true=0;
  int num_false=0;
  uint32_t num_labels;


  // Open leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists = true;
  leveldb::Status status = leveldb::DB::Open(
      options, db_filename, &db);
  CHECK(status.ok()) << "Failed to open leveldb " << db_filename
      << ". Is it already existing?";

  char* pixels = new char[rows * cols];
  std::string value;

  caffe::Datum datum;
  datum.set_channels(1);  // one channel
  datum.set_height(rows);
  datum.set_width(cols);
  list<tuple<int,int> >toWrite;
  LOG(INFO) << "from " << image_filenames.size() << " items.";
  LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
  for (int im1=0; im1<image_filenames.size(); im1++) {
      for (int i=0; i<bigramClasses.size(); i++)
      {
          if (labels[im1].compare(bigramClasses[i])==0)
          {
              toWrite.push_back(make_tuple(im1,i));
              break;
          }
      }
  }
  //write them in random order
  while (toWrite.size()>0) {
        int i = caffe::caffe_rng_rand() % toWrite.size();
        auto iter = toWrite.begin();
        for (int ii=0; ii<i; ii++) iter++;
        int im1=get<0>(*iter);
        int bigramIndex=get<1>(*iter);
        read_image(image_filenames[im1], rows, cols,
            pixels);
        datum.set_data(pixels, rows*cols);
        datum.set_label(bigramIndex);

        datum.SerializeToString(&value);
        char buff[10];
        snprintf(buff, sizeof(buff), "%08d", num_items);
        std::string key_str = buff; //caffe::format_int(num_items, 8);
        db->Put(leveldb::WriteOptions(), key_str, value);
        num_items++;

        toWrite.erase(iter);
    
  }
  cout << "A total of    " << num_items << " items written."<<endl;

  delete db;
  delete [] pixels;
}

int main(int argc, char** argv) {
  if (argc != 7) {
      cout<<argc<<endl;
    printf("This script converts the dataset to the leveldb format used\n"
           "by caffe to train a classification network.\n"
           "Usage:\n"
           " make_class_bigram_data image_dir image_label_file bigramListFile rows cols "
           "output_db_file\n"
           );
  } else {
    google::InitGoogleLogging(argv[0]);
    string image_dir=argv[1];
    string labelfile = argv[2];
    string bigramfile = argv[3];
    int rows=atoi(argv[4]);
    int cols=atoi(argv[5]);
    ifstream fileBigrams(bigramfile);
    ifstream filein(labelfile);
    assert(filein.good());
    //2700270.tif 519 166 771 246 orders
    string line;
    //regex qExtGtp("(\\S+\\.\\S+) (\\d+) (\\d+) (\\d+) (\\d+) (\\w+)");
    //regex qExt("(\\S+\\.\\S+) (\\w+)");
    
    
    vector<string> bigramClasses;
    while (getline(fileBigrams,line))
    {
        if (line.size()>0)
        {
            for (int i=0; i<line.size(); i++)
                line[i]=tolower(line[i]);
            bigramClasses.push_back(line);
        }
    }
    fileBigrams.close();
    
    string curPathIm="";
    
    vector<string> images,labels;
    while (getline(filein,line))
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
        for (int i=0; i<label.size(); i++)
            label[i]=tolower(label[i]);
        images.push_back(pathIm);
        labels.push_back(label);
    }
    filein.close();
    convert_dataset(images,labels,bigramClasses, argv[6],rows,cols);
  }
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires LevelDB; compile with USE_LEVELDB.";
}
#endif  // USE_LEVELDB
