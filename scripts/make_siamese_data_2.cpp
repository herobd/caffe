//g++ -std=c++11 make_siamese_data_2.cpp -lcaffe -lglog -lopencv_core -lopencv_highgui -lopencv_imgproc -lprotobuf -lleveldb -I ../include/ -L ../build/lib/ -o make_siamese_data_2
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

#if 1
#include "leveldb/db.h"

#define PER 10

using namespace std;

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

string read_image_with_label(string image_file, int label) {
	cv::Mat im = cv::imread(image_file,CV_LOAD_IMAGE_GRAYSCALE);
        assert(im.rows*im.cols>1);
        
        caffe::Datum datum;
        datum.set_channels(1);  
        datum.set_height(im.rows);
        datum.set_width(im.cols);
        datum.set_label(label);
	//copy(((char*)im.data),((char*)im.data)+(rows*cols),pixels);	
        datum.set_data(im.data, im.rows*im.cols);
        string ret;

        datum.SerializeToString(&ret);
        return ret;
}

void convert_dataset(const vector<string>& image_filenames, const vector<string>& labels,
        const char* db_filename, const char* db_filename2) {
  // Open files
  // Read the magic and the meta data
  uint32_t num_items=0;
  int num_true=0;
  int num_false=0;
  uint32_t num_labels;


  // Open leveldb
  leveldb::DB* db;
  leveldb::DB* db2;
  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists = true;
  leveldb::Status status = leveldb::DB::Open(
      options, db_filename, &db);
  CHECK(status.ok()) << "Failed to open leveldb " << db_filename
      << ". Is it already existing?";
  leveldb::Status status2 = leveldb::DB::Open(
      options, db_filename2, &db2);
  CHECK(status2.ok()) << "Failed to open leveldb " << db_filename2
      << ". Is it already existing?";

  map<string,vector<int> > wordMap;
  for (int i=0; i<labels.size(); i++)
      wordMap[labels[i]].push_back(i);


  char label_i;
  char label_j;
  //char* pixels = new char[2 * rows * cols];
  //std::string value;

  //caffe::Datum datum;
  //datum.set_channels(2);  // one channel for each image in the pair
  //datum.set_height(rows);
  //datum.set_width(cols);
  map<int, set<int> > used;
  list<tuple<int,int,bool> >toWrite;
  LOG(INFO) << "from " << image_filenames.size() << " items.";
  for (int im1=0; im1<image_filenames.size(); im1++) {
    //false matches
    vector<int> thisMap(wordMap[labels[im1]]);
    for (int count=0; count<PER; count++) {
        int im2;
        while (1) {
            im2 = caffe::caffe_rng_rand() % image_filenames.size();  // pick a random  pair
            if (used[im1].find(im2)==used[im1].end() && labels[im1].compare(labels[im2])!=0)
                    break;
        }
        used[im1].insert(im2);
        used[im2].insert(im1);
        toWrite.push_back(make_tuple(im1,im2,false));
    }
    //true matches
    //cout <<"on word: "<<labels[im1]<<endl;
    for (int count=0; count<PER; count++) {
        int im2=-1;
        
        while (thisMap.size()>0) {
            int i = caffe::caffe_rng_rand() % thisMap.size();  // pick a random  pair
            int tmp=thisMap[i];
            thisMap.erase(thisMap.begin()+i);
            if (tmp!=im1 && used[im1].find(tmp)==used[im1].end()) {
                im2 = tmp;
                break;
            }
        }
        if (im2!=-1) {
            used[im1].insert(im2);
            used[im2].insert(im1);
            //cout <<"pair: "<<im1<<", "<<im2<<endl;
            toWrite.push_back(make_tuple(im1,im2,true));
        } else {
            //cout<<"no pair for :"<<im1<<endl;
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
        int im2=get<1>(*iter);
        bool match=get<2>(*iter);
        string value = read_image_with_label(image_filenames[im1],match);
        string value2 = read_image_with_label(image_filenames[im2],match);
        //datum.set_data(pixels, rows*cols);
        //datum2.set_data(pixels2, rows2*cols2);
        if (match) {
          //datum.set_label(1);
          //datum2.set_label(1);
          num_true++;
        } else {
          //datum.set_label(0);
          //datum2.set_label(0);
          num_false++;
        }
        //datum.SerializeToString(&value);
        //datum2.SerializeToString(&value2);
        char buff[10];
        snprintf(buff, sizeof(buff), "%08d", num_items);
        std::string key_str = buff; //caffe::format_int(num_items, 8);
        db->Put(leveldb::WriteOptions(), key_str, value);
        db2->Put(leveldb::WriteOptions(), key_str, value2);
        num_items++;

        toWrite.erase(iter);
    
  }
  cout << "A total of    " << num_items << " items written."<<endl;
  cout << " true pairs:  " << num_true<<endl;
  cout << " false pairs: " << num_false<<endl;

  delete db;
  delete db2;
}

void convert_dataset_labels(const vector<string>& image_filenames, const vector<string>& labels,
        string db_filename, string db_filename2, uint32_t rows=-1, uint32_t cols=-1) {
    // Open files
    // Read the magic and the meta data
    uint32_t num_items=0;
    int num_true=0;
    int num_false=0;
    uint32_t num_labels;


    // Open leveldb
    leveldb::DB* db;
    leveldb::DB* db2=NULL;
    leveldb::Options options;
    options.create_if_missing = true;
    options.error_if_exists = true;
    leveldb::Status status = leveldb::DB::Open(
      options, db_filename, &db);
    CHECK(status.ok()) << "Failed to open leveldb " << db_filename
      << ". Is it already existing?";

    if (db_filename2.length()>0)
    {
         leveldb::Status status2 = leveldb::DB::Open(
          options, db_filename2, &db2);
        CHECK(status2.ok()) << "Failed to open leveldb " << db_filename2
          << ". Is it already existing?";
    }

    map<string,vector<int> > wordMap;
    for (int i=0; i<labels.size(); i++)
      wordMap[labels[i]].push_back(i);


    char* pixels = new char[2 * rows * cols];
    std::string value;

    caffe::Datum datum;
    datum.set_channels(2);  // one channel for each image in the pair
    datum.set_height(rows);
    datum.set_width(cols);
    list<tuple<int,int,int> >toWrite;
    LOG(INFO) << "from " << images.size() << " items.";
    LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
    int labelInd=0;
    for (auto p : wordMap)
    {
        for (int i=1; i<p.second.size(); i+=2)
        {
            toWrite.push_back(make_tuple(p.second[i-1],p.second[i],labelInd));
        }
        labelInd++;
    }
    //write them in random order
    while (toWrite.size()>0) {
        int i = caffe::caffe_rng_rand() % toWrite.size();
        auto iter = toWrite.begin();
        for (int ii=0; ii<i; ii++) iter++;
        int im1=get<0>(*iter);
        int im2=get<1>(*iter);
        int label=get<2>(*iter);

        char buff[10];
        snprintf(buff, sizeof(buff), "%08d", num_items);
        std::string key_str = buff; //caffe::format_int(num_items, 8);
        if (db2==NULL)
        {
            /*read_image_fixed(images[im1], rows, cols,
                pixels);
            read_image_fixed(images[im2], rows, cols,
                pixels + (rows * cols));
            datum.set_data(pixels, 2*rows*cols);
            datum.set_label(label);
            datum.SerializeToString(&value);
            db->Put(leveldb::WriteOptions(), key_str, value);
            */
            assert(false);
        }
        else
        {
            string value = read_image_with_label(image_filenames[im1],label);
            string value2 = read_image_with_label(image_filenames[im2],label);
            db->Put(leveldb::WriteOptions(), key_str, value);
            db2->Put(leveldb::WriteOptions(), key_str, value2);
        }
        num_items++;

        toWrite.erase(iter);

    }
    cout << "A total of    " << num_items << " X 2 items written."<<endl;

    delete db;
    if (db2!=NULL)
        delete db2;
    else
        delete [] pixels;
}


int main(int argc, char** argv) {
  if (argc != 5 && argc!=6) {
    printf("This script converts the dataset to the leveldb format used\n"
           "by caffe to train a siamese network.\n"
           "Usage:\n"
           " make_siamese_data image_dir image_label_file "
           "output1_db_file output2_db_file [-l]\n"
           );
  } else {
    google::InitGoogleLogging(argv[0]);
    string image_dir=argv[1];
    string labelfile = argv[2];

    bool labelSet= (argv[argc-1][0]=='-' && argv[argc-1][1]=='l');

    ifstream filein(labelfile);
    assert(filein.good());
    //2700270.tif 519 166 771 246 orders
    string line;
    //regex qExtGtp("(\\S+\\.\\S+) (\\d+) (\\d+) (\\d+) (\\d+) (\\w+)");
    //regex qExt("(\\S+\\.\\S+) (\\w+)");
    
    
    
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
        images.push_back(pathIm);
        labels.push_back(label);
    }
    filein.close();
    if (labelSet)
    {
        cout<<"Labeling"<<endl;
        convert_dataset(images,labels, argv[3], argv[4]);
    }
    else
        convert_dataset(images,labels, argv[3], argv[4]);
  }
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires LevelDB; compile with USE_LEVELDB.";
}
#endif  // USE_LEVELDB
