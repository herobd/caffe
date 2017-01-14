//g++ -std=c++11 make_class_data_from_seg_csv.cpp -lcaffe -lglog -l:libopencv_core.so.3.0 -l:libopencv_imgcodecs.so.3.0 -l:libopencv_imgproc.so.3.0 -l:libopencv_highgui.so.3.0 -lprotobuf -lleveldb -I ../include/ -L ../build/lib/ -o make_class_data_from_seg_csv

// This script converts the dataset (described by a char segmentation csv) to the leveldb format used
// by caffe to train a classification network (bigrams).

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

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
/*
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
*/
/*
#include "/usr/local/include/opencv2/core.hpp"
#include "/usr/local/include/opencv2/highgui.hpp"
#include </usr/local/include/opencv2/imgproc.hpp>
#include </usr/local/include/opencv2/imgcodecs.hpp>
*/

#if 1
#include "leveldb/db.h"

#define PER 10
#define PAD 9
#define END_PAD 3

using namespace std;

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

map<string,cv::Mat> loaded;
struct Location
{
    string imagePath;
    int x1,y1,x2,y2;
    Location() {}
    Location(string imagePath, int x1, int y1, int x2, int y2) : imagePath(imagePath),
                                                            x1(x1),
                                                            y1(y1),
                                                            x2(x2),
                                                            y2(y2) {}
    //string toString() {return imagePath+to_string(x1)+"_"+to_string(y1)+"_"+to_string(x2)+"_"+to_string(y2);}
};

void read_image_fixed(Location image,
        uint32_t rows, uint32_t cols,
        char* pixels) {
        if (loaded.find(image.imagePath) == loaded.end())
	    loaded[image.imagePath] = cv::imread(image.imagePath,CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat im = loaded[image.imagePath](cv::Rect(image.x1,image.y1,image.x2-image.x1+1,image.y2-image.y1+1));
        assert(im.rows*im.cols>1);
	//resize
	cv::resize(im,im,cv::Size(rows,cols));

	copy(((char*)im.data),((char*)im.data)+(rows*cols),pixels);	
}


void convert_dataset(const vector<Location>& images, const vector<string>& labels, const vector<string>& bigramClasses,
        string db_filename, uint32_t rows, uint32_t cols) {
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




    char* pixels = new char[ rows * cols];
    std::string value;

    caffe::Datum datum;
    datum.set_channels(2);  // one channel for each image in the pair
    datum.set_height(rows);
    datum.set_width(cols);
    list<tuple<int,int> >toWrite;
    LOG(INFO) << "from " << images.size() << " items.";
    LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
    for (int im1=0; im1<images.size(); im1++) {
        string bigram = labels[im1];
        //if (bigramClasses.find(bigram) != bigramClasses.end())
        for (int j=0; j<bigramClasses.size(); j++)
        {
            if (bigramClasses[j].compare(bigram)==0)
            {
                toWrite.push_back(make_tuple(im1,j));
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

        char buff[10];
        snprintf(buff, sizeof(buff), "%08d", num_items);
        std::string key_str = buff; //caffe::format_int(num_items, 8);
        read_image_fixed(images[im1], rows, cols,
            pixels);
        datum.set_data(pixels, rows*cols);
        datum.set_label(bigramIndex);
        datum.SerializeToString(&value);
        db->Put(leveldb::WriteOptions(), key_str, value);
        num_items++;

        toWrite.erase(iter);

    }
    cout << "A total of    " << num_items << " items written."<<endl;

    delete db;
    delete [] pixels;
}

int main(int argc, char** argv) {
  if (argc!=7) {
    printf("This script converts the dataset (char segmented) to the leveldb format used\n"
           "by caffe to train a classification network. Assumes gtp is full (has all pages in order).\n"
           "bigramListFile is a list of the bigrams wanted as classes. Order used for class numbers \n"
           "Usage:\n"
           " make_class_data_from_seg_csv image_dir bigramListFile gtpfile segcsvfile "
           "output_db_file fixed_rows fixed_cols\n"
           );
  } else {
    google::InitGoogleLogging(argv[0]);
    string image_dir=argv[1];
    string bigramFile=argv[2];
    string gtpfile = argv[3];
    string segfile = argv[4];
    int rows=0;
    int cols=0;
    string outdbname = argv[5];
    rows=atoi(argv[6]);
    cols=atoi(argv[7]);
    
    
    ifstream fileBigrams(bigramFile);
    ifstream fileGTP(gtpfile);
    ifstream fileSeg(segfile);
    assert(fileGTP.good());
    assert(fileSeg.good());
    //2700270.tif 519 166 771 246 orders
    string line;
    //regex qExtGtp("(\\S+\\.\\S+) (\\d+) (\\d+) (\\d+) (\\d+) (\\w+)");
    //regex qExt("(\\S+\\.\\S+) (\\w+)");
    
    vector<string> bigramClasses;
    while (getline(fileBigrams,line))
    {
        if (line.size()>0)
            bigramClasses.push_back(line);
    }
    fileBigrams.close();

    string curPage="";
    vector<string> pages;
    while (getline(fileGTP,line))
    {
        stringstream ss(line);
        string part;
        getline(ss,part,' ');
        if (part.compare(curPage)!=0)
        {
            curPage=part;
            pages.push_back(part);
        }
    }
    fileGTP.close();
    
    
    vector<Location> images;
    vector<string> labels;
    while (getline(fileSeg,line))
    {
        //smatch sm;
        //cout <<line<<endl;
        //regex_search(line,sm,qExt);
        stringstream ss(line);
        string part;
        getline(ss,part,',');
        string word=part;
        getline(ss,part,',');
        string imagePath=image_dir+pages.at(stoi(part));
        getline(ss,part,',');//x1
        int x1w=stoi(part);
        getline(ss,part,',');
        int y1=stoi(part);
        getline(ss,part,',');//x2
        int x2w=stoi(part);
        getline(ss,part,',');
        int y2=stoi(part);
        vector<int> lettersStart, lettersEnd;
        while (getline(ss,part,','))
        {
            lettersStart.push_back(stoi(part));
            getline(ss,part,',');
            lettersEnd.push_back(stoi(part));
            //getline(ss,s,',');//conf
        }
        assert(lettersStart.size()==word.length());
        for (int i=0; i<word.length()-1; i++)
        {
            string bigram = word[i]+""+word[i+1];
            int x1 = max(x1w,lettersStart[i] - (i==0?END_PAD:PAD));
            int x2 = min(x2w,lettersEnd[i] + (i==word.length()-2?END_PAD:PAD));
            images.push_back(Location(imagePath,x1,y1,x2,y2));
            labels.push_back(bigram);
        }

    }
    fileSeg.close();
    convert_dataset(images,labels, bigramClasses, outdbname,rows,cols);
  }
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires LevelDB; compile with USE_LEVELDB.";
}
#endif  // USE_LEVELDB
