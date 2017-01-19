//g++ -std=c++11 make_siamese_data_from_seg_csv.cpp -lcaffe -lglog -lleveldb -l:libopencv_core.so.3.0 -l:libopencv_imgcodecs.so.3.0 -l:libopencv_imgproc.so.3.0 -l:libopencv_highgui.so.3.0 -lprotobuf -lboost_system -I ../include/ -L ../build/lib/ -o make_siamese_data_from_seg_csv
// This script converts the dataset (described by a char segmentation csv) to the leveldb format used
// by caffe to train siamese network.
//It can create a fixed size database (where each pair is on image on different channels),
//or parallel lmdbs for varying sizes
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
#define JITTER 3

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
        int origH = image.y2-image.y1+1;
        //int origW = image.x2-image.x1+1;
        double scale = (rows-2*JITTER+0.0)/origH;
        double widthOrig = (cols+0.0)/scale;
        widthOrig/=2.0;
        double x = (image.x1+image.x2)/2.0;
        int newY1 = max((int)(image.y1-(JITTER/scale)), 0);
        int newY2 = min((int)(image.y2+(JITTER/scale)), loaded[image.imagePath].rows-1);
        int newX1 = max((int)(x-widthOrig/2.0), 0);
        int newX2 = min((int)(x+widthOrig/2.0), loaded[image.imagePath].cols-1);

        cv::Mat im = loaded[image.imagePath](cv::Rect(newX1,newY1,newX2-newX1+1,newY2-newY1+1));
        //cout<<"["<<image.x1<<" "<<image.y1<<" "<<image.x2<<" "<<image.y2<<"] "<<newX1<<" "<<newY1<<" "<<newX2<<" "<<newY2<<endl;
        assert(im.rows*im.cols>1);

	//resize
	cv::resize(im,im,cv::Size(cols,rows));
	/*cv::resize(im,im,cv::Size(),scale,scale);
        if (im.rows!=rows || im.cols!=cols)
        {
            cout<<"Origin ["<<image.y2-image.y1+1<<","<<image.x2-image.x1+1<<"]"<<endl;
            cout<<"Error, resize of image["<<im.rows<<","<<im.cols<<"] with scale "<<scale<<" did not yield ["<<rows<<","<<cols<<"]"<<endl;
            assert(im.rows==rows && im.cols==cols);
        }*/

	copy(((char*)im.data),((char*)im.data)+(rows*cols),pixels);	
}

string read_image_with_label(Location image, int match) {
        if (loaded.find(image.imagePath) == loaded.end())
	    loaded[image.imagePath] = cv::imread(image.imagePath,CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat im = loaded[image.imagePath](cv::Rect(image.x1,image.y1,image.x2-image.x1+1,image.y2-image.y1+1));
        assert(im.rows*im.cols>1);

        caffe::Datum datum;
        datum.set_channels(1);
        datum.set_height(im.rows);
        datum.set_width(im.cols);
        datum.set_label(match);
        //if (match) {
        //  datum.set_label(1);
        //}
        //else
        //{
        //  datum.set_label(0);
        //}
        datum.set_data(im.data, im.rows*im.cols);
        string ret;

        datum.SerializeToString(&ret);
        return ret;
}

void convert_dataset(const vector<Location>& images, const vector<string>& labels,
        string db_filename, string db_filename2, uint32_t rows, uint32_t cols) {
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
    LOG(INFO) << "from " << images.size() << " items.";
    LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
    for (int im1=0; im1<images.size(); im1++) {
        //false matches
        vector<int> thisMap(wordMap[labels[im1]]);
        for (int count=0; count<PER; count++) {
            int im2;
            while (1) {
                im2 = caffe::caffe_rng_rand() % images.size();  // pick a random  pair
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

        char buff[10];
        snprintf(buff, sizeof(buff), "%08d", num_items);
        std::string key_str = buff; //caffe::format_int(num_items, 8);
        if (db2==NULL)
        {
            read_image_fixed(images[im1], rows, cols,
                pixels);
            read_image_fixed(images[im2], rows, cols,
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
            db->Put(leveldb::WriteOptions(), key_str, value);
        }
        else
        {
            string value = read_image_with_label(images[im1],match);
            string value2 = read_image_with_label(images[im2],match);
            if (match) {
              num_true++;
            } else {
              num_false++;
            }
            db->Put(leveldb::WriteOptions(), key_str, value);
            db2->Put(leveldb::WriteOptions(), key_str, value2);
        }
        num_items++;

        toWrite.erase(iter);

    }
    cout << "A total of    " << num_items << " items written."<<endl;
    cout << " true pairs:  " << num_true<<endl;
    cout << " false pairs: " << num_false<<endl;

    delete db;
    if (db2!=NULL)
        delete db2;
    else
        delete [] pixels;
}

void convert_dataset_labels(const vector<Location>& images, const vector<string>& labels,
        string db_filename, string db_filename2, uint32_t rows, uint32_t cols) {
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
            read_image_fixed(images[im1], rows, cols,
                pixels);
            read_image_fixed(images[im2], rows, cols,
                pixels + (rows * cols));
            datum.set_data(pixels, 2*rows*cols);
            datum.set_label(label);
            datum.SerializeToString(&value);
            db->Put(leveldb::WriteOptions(), key_str, value);
        }
        else
        {
            string value = read_image_with_label(images[im1],label);
            string value2 = read_image_with_label(images[im2],label);
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
  if (argc != 6 && argc!=7 && argc !=8) {
    printf("This script converts the dataset (char segmented) to the leveldb format used\n"
           "by caffe to train a siamese network. Assumes gtp is full (has all pages in order).\n"
           "Either varying sized parallel dbs or \n"
           "fixed size, with paired images as channels.\n"
           "Usage:\n"
           " make_siamese_data_from_seg_csv image_dir gtpfile segcsvfile "
           "output_db_file [output_db_file2 OR fixed_rows fixed_cols] [-l]\n"
           "The -l flag causes a validation set to be made with pairs sharing the same ngram label\n"
           );
  } else {
    google::InitGoogleLogging(argv[0]);
    string image_dir=argv[1];
    string gtpfile = argv[2];
    string segfile = argv[3];
    int rows=0;
    int cols=0;
    string outdbname = argv[4];
    string outdbname2="";
    if (argc==6 || (argc==7 && argv[6][0]=='-'))
    {
        outdbname2=argv[5];
    }
    else
    {
        rows=atoi(argv[5])+2*JITTER;
        cols=atoi(argv[6])+2*JITTER;
    }

    bool labelSet= (argv[argc-1][0]=='-' && argv[argc-1][1]=='l');
    
    ifstream fileGTP(gtpfile);
    ifstream fileSeg(segfile);
    assert(fileGTP.good());
    assert(fileSeg.good());
    //2700270.tif 519 166 771 246 orders
    string line;
    //regex qExtGtp("(\\S+\\.\\S+) (\\d+) (\\d+) (\\d+) (\\d+) (\\w+)");
    //regex qExt("(\\S+\\.\\S+) (\\w+)");
    
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
    if (labelSet)
    {
        cout<<"labeling"<<endl;
        convert_dataset_labels(images,labels, outdbname, outdbname2,rows,cols);
    }
    else
        convert_dataset(images,labels, outdbname, outdbname2,rows,cols);
  }
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires LevelDB; compile with USE_LEVELDB.";
}
#endif  // USE_LEVELDB
