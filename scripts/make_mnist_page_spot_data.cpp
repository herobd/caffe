//g++ -std=c++11 make_mnist_page_spot_data.cpp -lcaffe -lglog -l:libopencv_core.so.3.1 -l:libopencv_highgui.so.3.1 -l:libopencv_imgproc.so.3.1 -l:libopencv_imgcodecs.so.3.1 -lprotobuf -lleveldb -I ../include/ -L ../build/lib/ -o make_mnist_page_spot_data
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
#include <sys/stat.h>
#include <sys/types.h>

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

#define NUM_PER 3

using namespace std;

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

string serialize_image(cv::Mat im) {
        assert(im.rows*im.cols>1);
        
        caffe::Datum datum;
        datum.set_channels(1);  
        datum.set_height(im.rows);
        datum.set_width(im.cols);
        //datum.set_label(label);
	//copy(((char*)im.data),((char*)im.data)+(rows*cols),pixels);	
        assert(im.isContinuous());
        datum.set_data(im.data, im.rows*im.cols);
        string ret;

        datum.SerializeToString(&ret);
        return ret;
}
string read_image(string image_file) {
	cv::Mat im = cv::imread(image_file,CV_LOAD_IMAGE_GRAYSCALE);
        if (im.rows==0)
        {
            cout<<"Failed to open: "<<image_file<<endl;
            //return "";
        }
        //else
        //    cout<<"did open: "<<image_file<<endl;
        assert(im.rows*im.cols>1);
#ifdef DEBUG
        //cv::imshow("image",im);
        //cv::waitKey();
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

int getNextIndex(vector<bool>& used)
{
    int index = rand()%used.size();
    int start=index;
    while (used.at(index))
    {
        index = (1+index)%used.size();
        if (index==start)
        {
            used.assign(used.size(),false);
        }
    }
    used.at(index)=true;
    return index;
}
struct setcomp {
    bool operator() (const set<int>& lhs, const set<int>& rhs) const
    {
        if (lhs.size() == rhs.size())
        {
            auto iterL=lhs.begin();
            auto iterR=rhs.begin();
            while(iterL!=lhs.end())
            {
                if (*iterL!=*iterR)
                    return *iterL<*iterR;
                iterL++;
                iterR++;
            }
            return false;
        }
        return lhs.size()<rhs.size();
    }
};


void generateDataset(string out_query_name, string out_page_name, string out_label_name, vector<string> imagePaths, vector<int> imageLabels, int size, int numPerClass, int numNeg, int maxNumTrue, vector<int> classes, string saveDir="")
{
    // Open files
    // Read the magic and the meta data
    uint32_t num_items=0;
    int num_true=0;
    int num_false=0;
    uint32_t num_labels;

    float querySplit = 0.7;

    //vector<string> classes;
    //map<string,int> classMap;
    //int classCounter=1;
    vector<bool> usedClass(10);
    map<int, vector<string> > queries;
    map<int, vector<bool> > usedQuery;
    map<int, vector<string> > forPages;
    map<int, vector<bool> > usedForPages;


    for (int i=0; i<imagePaths.size()*querySplit; i++)
    {
        queries[imageLabels[i]].push_back(imagePaths[i]);
    }
    for (auto p : queries)
    {
        usedQuery[p.first].resize(p.second.size());
    }

    //for (int i=imagePaths.size()*querySplit; i<imagePaths.size(); i+=NUM_PER)
    //{
    //    cv::Mat first = cv::imread(
    //}
    for (int i=imagePaths.size()*querySplit; i<imagePaths.size(); i++)
    {
        forPages[imageLabels[i]].push_back(imagePaths[i]);
    }
    for (auto p : forPages)
    {
        usedForPages[p.first].resize(p.second.size());
    }





    char label_i;
    char label_j;
    //char* pixels = new char[2 * rows * cols];
    //std::string value;

    //caffe::Datum datum;
    //datum.set_channels(2);  // one channel for each image in the pair
    //datum.set_height(rows);
    //datum.set_width(cols);
    vector< tuple<string,int,vector<string>,vector<string> > >toWrite; //query, classID, pos_im, neg_im, neg_im
    //for (auto cAndId : queries)
    for (int cls : classes)
    {
        //int cls = cAndId.first;
        cout<<"on class: "<<cls<<endl;
        for (int i=0; i<numPerClass; i++)
        {
            int query = getNextIndex(usedQuery.at(cls));
            vector<string> trues;
            int numTrue= rand()%(maxNumTrue+1);
            for (int ii=0; ii<numTrue; ii++)
            {
                int instanceIndex = getNextIndex(usedForPages.at(cls));
                trues.push_back(forPages.at(cls).at(instanceIndex));
            }
            vector<string> negs;
            for (int ii=0; ii<numNeg-numTrue; ii++)
            {
                int randClass;
                if (classes.size()>1)
                {
                    randClass = classes[rand()%(classes.size()-1)];
                    if (randClass==cls)
                        randClass=classes.back();
                }
                else
                {
                    randClass = rand()%9;
                    if (randClass==cls)
                        randClass=9;
                }
                string neg = forPages.at(randClass).at(rand()%forPages.at(randClass).size());
                negs.push_back(neg);
            }

            toWrite.emplace_back(queries.at(cls).at(query),cls,trues,negs);
        }

    }


    //write them in random order
    shuffle(toWrite.begin(), toWrite.end(), default_random_engine(11));
    cout<<"Writing..."<<endl;
    /*
    while (toWrite.size()>0) {
        int i = caffe::caffe_rng_rand() % toWrite.size();
        auto iter = toWrite.begin();
        for (int ii=0; ii<i; ii++) iter++;
        toWrite[toWrite.size()-1] = *iter;
        toWrite.erase(iter);
    }*/

    ofstream saveFile;
    if (saveDir.length()>0)
    {
        mkdir(saveDir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        mkdir((saveDir+"/queries").c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        mkdir((saveDir+"/pages").c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        mkdir((saveDir+"/gt").c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        saveFile.open(saveDir+"/index.txt");
    }

    // Open leveldb
    leveldb::DB* queries_db;
    leveldb::Options options;
    options.create_if_missing = true;
    options.error_if_exists = true;
    leveldb::Status status = leveldb::DB::Open(
        options, out_query_name, &queries_db);
    CHECK(status.ok()) << "Failed to open leveldb " << out_query_name
        << ". Is it already existing?";

    for (int index=0; index<toWrite.size(); index++)
    {
        string query=get<0>(toWrite[index]);
        int classId=get<1>(toWrite[index]);
        string ser_query= read_image(query);
        if (ser_query.length()==0)
            continue;

        char buff[12];
        snprintf(buff, sizeof(buff), "%09d", index);
        std::string key_str = buff; //caffe::format_int(num_items, 8);
        leveldb::Status s = queries_db->Put(leveldb::WriteOptions(), key_str, ser_query);
        assert(s.ok());
        num_items++;
        
        if (saveDir.length()>0) {
            cv::imwrite(saveDir+"/queries/"+string(buff)+".png",cv::imread(query,0));
        }
        //system ("cp "+query+" ");

    }
    delete queries_db;


    leveldb::DB* pages_db;
    leveldb::Status status1 = leveldb::DB::Open(
        options, out_page_name, &pages_db);
    CHECK(status1.ok()) << "Failed to open leveldb " << out_page_name
        << ". Is it already existing?";

    vector< vector<cv::Rect> > labels(toWrite.size());
    for (int index=0; index<toWrite.size(); index++)
    {
        string query=get<0>(toWrite[index]);
        int classId=get<1>(toWrite[index]);
        vector<string> trues=get<2>(toWrite[index]);
        //cv::Mat posIm = cv::imread(pos,0);
        vector<string> negs = get<3>(toWrite[index]);
        
        cv::Mat page (size,size,CV_8U);
        page=255;
        vector<cv::Rect> locs;//prevent overlaps, track all digit locations
        vector<cv::Rect> posLocs;//for label creation

        for (string posFile : trues)
        {
            cv::Mat posIm = cv::imread(posFile,0);
            cv::Rect posLoc;
            for (int i=0; i<1000; i++) //max tries
            {
                posLoc.width=posIm.cols;
                posLoc.height=posIm.rows;
                posLoc.x=rand()%(size-posIm.cols);
                posLoc.y=rand()%(size-posIm.rows);
                bool good=true;
                for (const cv::Rect& loc : locs)
                {
                    if ( ((posLoc.x>=loc.x&&posLoc.x<loc.x+loc.width) || (posLoc.x<=loc.x&&posLoc.x+posLoc.width>loc.x)) &&
                     ((posLoc.y>=loc.y&&posLoc.y<loc.y+loc.height) || (posLoc.y<=loc.y&&posLoc.y+posLoc.height>loc.y)) )
                    {
                        good=false;
                        break;
                    }
                }
                if (good)
                {
                    posIm.copyTo(page(posLoc));
                    locs.push_back(posLoc);
                    posLocs.push_back(posLoc);
                    break;
                }
            }
        }

        for (string negFile : negs)
        {
            cv::Mat negIm = cv::imread(negFile,0);
            cv::Rect negLoc;
            for (int i=0; i<1000; i++) //max tries
            {
                negLoc.width=negIm.cols;
                negLoc.height=negIm.rows;
                negLoc.x=rand()%(size-negIm.cols);
                negLoc.y=rand()%(size-negIm.rows);
                bool good=true;
                for (const cv::Rect& loc : locs)
                {
                    if ( ((negLoc.x>=loc.x&&negLoc.x<loc.x+loc.width) || (negLoc.x<=loc.x&&negLoc.x+negLoc.width>loc.x)) &&
                     ((negLoc.y>=loc.y&&negLoc.y<loc.y+loc.height) || (negLoc.y<=loc.y&&negLoc.y+negLoc.height>loc.y)) )
                    {
                        good=false;
                        break;
                    }
                }
                if (good)
                {
                    negIm.copyTo(page(negLoc));
                    locs.push_back(negLoc);
                    break;
                }
            }
        }
            
        string ser_page= serialize_image(page);
        //Create binary label image
#ifdef DEBUG
        cout<<"pos class: "<<classId<<endl;
        cv::imshow("page",page);
        cv::waitKey();
#endif

        char buff[12];
        snprintf(buff, sizeof(buff), "%09d", index);
        std::string key_str = buff; //caffe::format_int(num_items, 8);
        leveldb::Status s = pages_db->Put(leveldb::WriteOptions(), key_str, ser_page);
        assert(s.ok());

        labels[index]=posLocs;
        if (saveDir.length()>0)
        {
            cv::imwrite(saveDir+"/pages/"+string(buff)+".png",page);
            saveFile << "queries/"<<string(buff)<<".png pages/"<<string(buff)<<".png gt/"<<string(buff)<<".png "<<posLocs.size();//<<posLoc.x<<" "<<posLoc.y<<" "<<posLoc.width<<" "<<posLoc.height<<endl;
            for (auto r : posLocs)
                saveFile<<" "<<r.x<<" "<<r.y<<" "<<r.width<<" "<<r.height;
            saveFile<<endl;
        }
    }
    delete pages_db;

    leveldb::DB* labels_db;
    leveldb::Status status2 = leveldb::DB::Open(
        options, out_label_name, &labels_db);
    CHECK(status2.ok()) << "Failed to open leveldb " << out_label_name
        << ". Is it already existing?";

    for (int index=0; index<toWrite.size(); index++)
    {
        //Create binary label image
        cv::Mat labelIm = cv::Mat::zeros(size,size,CV_8U);
        for (auto rect : labels[index])
            labelIm(rect)=255;
        string ser_label= serialize_image(labelIm);
#ifdef DEBUG
        cv::imshow("label",labelIm);
        cv::waitKey();
#endif

        char buff[12];
        snprintf(buff, sizeof(buff), "%09d", index);
        std::string key_str = buff; //caffe::format_int(num_items, 8);
        leveldb::Status s = labels_db->Put(leveldb::WriteOptions(), key_str, ser_label);
        assert(s.ok());

        if (saveDir.length()>0)
            cv::imwrite(saveDir+"/gt/"+string(buff)+".png",labelIm);

    }

    cout << "A total of    " << num_items << " items written."<<endl;

    delete labels_db;
}

int main(int argc, char** argv) {
  if (argc < 14 || argc > 16) {
    printf("This script converts the dataset to 3 leveldbs, one of query images, one of page images, and one of page GTs. These are aligned and must not to randomized.\n"
           "Usage:\n"
           " make_mnist_page_spot_data query_image_dir query_pointer.txt"
           "  windowSize train_numPerClass test_numPerClass"
           " train_query_db_file trian_page_db_file trian_label_db_file"
           " test_query_db_file test_page_db_file test_label_db_file"
           " numPerPage maxNumPosPerPage [save img dir]\n"
           );
  } else {
    google::InitGoogleLogging(argv[0]);
    string query_image_dir=argv[1];
    string query_pointer_file=argv[2];

    int size = atoi(argv[3]);
    int train_numPerClass = atoi(argv[4]);
    int test_numPerClass= atoi(argv[5]);
    float split = train_numPerClass/(0.0+train_numPerClass+test_numPerClass);

    string train_query_name = argv[6];
    string train_page_name = argv[7];
    string train_label_name = argv[8];
    string test_query_name = argv[9];
    string test_page_name = argv[10];
    string test_label_name = argv[11];
    int numNeg = atoi(argv[12]);
    int numTrue = atoi(argv[13]);

    int holdOut=-1;
    if (argc>14)
        holdOut = atoi(argv[14]);

    string saveDir;
    if (argc==16)
        saveDir = argv[15];

    vector<int> train_classes, test_classes;
    if (holdOut<0)
    {
        for (int i=0; i<10; i++)
        {
            train_classes.push_back(i);
            test_classes.push_back(i);
        }
    }
    else
    {
        test_classes.push_back(holdOut);

        for (int i=0; i<10; i++)
        {
            if (i!=holdOut)
                train_classes.push_back(i);
        }
    }


    vector<string> train_imagePaths;
    vector<int> train_imageLabels;


    ifstream filein(query_pointer_file);
    assert(filein.good());
    string line;
    while (getline(filein,line))
    {
        stringstream ss(line);
        string fileName;
        getline(ss,fileName,' ');
        string c;
        getline(ss,c,' ');
        train_imagePaths.push_back(query_image_dir+fileName);
        train_imageLabels.push_back(stoi(c));
    }
    filein.close();
    
    vector<string> test_imagePaths(train_imagePaths.begin()+(int)(train_imagePaths.size()*split),train_imagePaths.end());
    train_imagePaths.erase(train_imagePaths.begin()+(int)(train_imagePaths.size()*split),train_imagePaths.end());
    vector<int> test_imageLabels(train_imageLabels.begin()+(int)(train_imageLabels.size()*split),train_imageLabels.end());
    train_imageLabels.erase(train_imageLabels.begin()+(int)(train_imageLabels.size()*split),train_imageLabels.end());

    ///////////////////////////////////////////////////
    if (test_imagePaths.size()>0)
        generateDataset(test_query_name, test_page_name, test_label_name, test_imagePaths, test_imageLabels, size, test_numPerClass, numNeg, numTrue, test_classes, saveDir);
    //if (saveDir.length()==0)
    if (train_imagePaths.size()>0)
        generateDataset(train_query_name, train_page_name, train_label_name, train_imagePaths, train_imageLabels, size, train_numPerClass, numNeg, numTrue, train_classes);

  }
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires LevelDB; compile with USE_LEVELDB.";
}
#endif  // USE_LEVELDB