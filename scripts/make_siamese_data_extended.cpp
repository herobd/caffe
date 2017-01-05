//g++ -g -std=c++11 make_siamese_data_extended.cpp -lcaffe -lglog -l:libopencv_core.so.3.0 -l:libopencv_imgproc.so.3.0 -l:libopencv_imgcodecs.so.3.0 -lprotobuf -lleveldb -I ../include/ -L ../build/lib/ -o make_siamese_data_extended
//g++ -std=c++11 make_siamese_data_extended.cpp -lcaffe -lglog -lopencv_core -lopencv_highgui -lopencv_imgproc -lprotobuf -lleveldb -I /home/brianld/include -I ../include/ -L ../build/lib/ -o make_siamese_data_extended
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

//#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/core.hpp"
//#include "opencv2/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

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
	cv::Mat im = cv::imread(image_file,0);
        if (im.rows*im.cols<2)
            cout<<"could not open: "<<image_file<<endl;
        assert(im.rows*im.cols>1);
	//resize
	cv::resize(im,im,cv::Size(rows,cols));

	copy(((char*)im.data),((char*)im.data)+(rows*cols),pixels);	
}

void convert_dataset(
        const vector<string>& image_filenames, const vector<string>& labels,
        const vector<string>& tri_image_filenames, const vector<string>& tri_labels,
        const vector<string>& uni_image_filenames, const vector<string>& uni_labels,
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

  map<string,vector<int> > wordMap;
  for (int i=0; i<labels.size(); i++)
      wordMap[labels[i]].push_back(i);


  char label_i;
  char label_j;
  char* pixels = new char[2 * rows * cols];
  std::string value;

  caffe::Datum datum;
  datum.set_channels(2);  // one channel for each image in the pair
  datum.set_height(rows);
  datum.set_width(cols);
  map<int, set<int> > used;
  list<tuple<int,int,bool> >toWrite;
  LOG(INFO) << "from " << image_filenames.size() << " items.";
  LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
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
    //bad seg matches
    int tri=caffe::caffe_rng_rand() % tri_image_filenames.size();
    int triInit=tri;
    do
    {
        tri=tri%tri_image_filenames.size();
        if (labels.at(im1).compare( tri_labels.at(tri).substr(0,2) ) == 0)
        {
            toWrite.push_back(make_tuple(im1,tri+image_filenames.size(),false));
            break;
        }
    } while (++tri!=triInit);
    tri=caffe::caffe_rng_rand() % tri_image_filenames.size();
    triInit=tri;
    do
    {
        tri=tri%tri_image_filenames.size();
        if (labels.at(im1).compare( tri_labels.at(tri).substr(1,2) ) == 0)
        {
            toWrite.push_back(make_tuple(im1,tri+image_filenames.size(),false));
            break;
        }
    } while (++tri!=triInit);
    int uni=caffe::caffe_rng_rand() % uni_image_filenames.size();
    int uniInit=uni;
    do
    {
        uni=uni%uni_image_filenames.size();
        if (labels.at(im1).substr(0,1).compare( uni_labels.at(uni) ) == 0)
        {
            toWrite.push_back(make_tuple(im1,uni+image_filenames.size()+tri_image_filenames.size(),false));
            break;
        }
    } while (++uni!=uniInit);
    uni=caffe::caffe_rng_rand() % uni_image_filenames.size();
    uniInit=uni;
    do
    {
        uni=uni%uni_image_filenames.size();
        if (labels.at(im1).substr(1,1).compare( uni_labels.at(uni) ) == 0)
        {
            toWrite.push_back(make_tuple(im1,uni+image_filenames.size()+tri_image_filenames.size(),false));
            break;
        }
    } while (++uni!=uniInit);
        

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
        string file1, file2;
        if (im1<image_filenames.size())
            file1=image_filenames[im1];
        else if (im1<image_filenames.size()+tri_image_filenames.size())
            file1=tri_image_filenames.at(im1-image_filenames.size());
        else
            file1=uni_image_filenames.at(im1-image_filenames.size()-tri_image_filenames.size());
        if (im2<image_filenames.size())
            file2=image_filenames[im2];
        else if (im2<image_filenames.size()+tri_image_filenames.size())
            file2=tri_image_filenames.at(im2-image_filenames.size());
        else
            file2=uni_image_filenames.at(im2-image_filenames.size()-tri_image_filenames.size());
        read_image(file1, rows, cols,
            pixels);
        read_image(file2, rows, cols,
            pixels + (rows * cols));
        datum.set_data(pixels, 2*rows*cols);
        if (match) {
          datum.set_label(1);
          num_true++;
        } else {
          datum.set_label(0);
          num_false++;
        }
        datum.SerializeToString(&value);
        char buff[10];
        snprintf(buff, sizeof(buff), "%08d", num_items);
        std::string key_str = buff; //caffe::format_int(num_items, 8);
        db->Put(leveldb::WriteOptions(), key_str, value);
        num_items++;

        toWrite.erase(iter);
    
  }
  cout << "A total of    " << num_items << " items written."<<endl;
  cout << " true pairs:  " << num_true<<endl;
  cout << " false pairs: " << num_false<<endl;

  delete db;
  delete [] pixels;
}

int main(int argc, char** argv) {
  if (argc != 8) {
    printf("This script converts the dataset to the leveldb format used\n"
           "by caffe to train a siamese network.\n"
           "Usage:\n"
           " make_siamese_data_extended image_dir bi_label_file tri_label_file uni_label_file rows cols "
           "output_db_file\n"
           );
  } else {
    google::InitGoogleLogging(argv[0]);
    string image_dir=argv[1];
    string labelfile = argv[2];
    string trilabelfile = argv[3];
    string unilabelfile = argv[4];
    int rows=atoi(argv[5]);
    int cols=atoi(argv[6]);
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

    filein.open(trilabelfile);
    assert(filein.good());
    vector<string> tri_images,tri_labels;
    while (getline(filein,line))
    {
        smatch sm;
        stringstream ss(line);
        string part;
        getline(ss,part,' ');
        string pathIm=image_dir+part;
        getline(ss,part,' ');
        string label=part;
        tri_images.push_back(pathIm);
        tri_labels.push_back(label);
    }
    filein.close();

    filein.open(unilabelfile);
    assert(filein.good());
    vector<string> uni_images,uni_labels;
    while (getline(filein,line))
    {
        smatch sm;
        stringstream ss(line);
        string part;
        getline(ss,part,' ');
        string pathIm=image_dir+part;
        getline(ss,part,' ');
        string label=part;
        uni_images.push_back(pathIm);
        uni_labels.push_back(label);
    }
    filein.close();
    cout<<"read in files"<<endl;

    convert_dataset(images,labels,
                    tri_images, tri_labels,
                    uni_images, uni_labels,
                    argv[7],rows,cols);
  }
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires LevelDB; compile with USE_LEVELDB.";
}
#endif  // USE_LEVELDB
