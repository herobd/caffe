//g++ -std=c++11 make_page_spot_data.cpp -lcaffe -lglog -l:libopencv_core.so.3.1 -l:libopencv_highgui.so.3.1 -l:libopencv_imgproc.so.3.1 -l:libopencv_imgcodecs.so.3.1 -lprotobuf -lleveldb -I ../include/ -L ../build/lib/ -o make_page_spot_data
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

int main(int argc, char** argv) {
  if (argc != 11) {
    printf("This script converts the dataset to 3 leveldbs, one of query images, one of page images, and one of page GTs. These are aligned and must not to randomized.\n"
           "Usage:\n"
           " make_page_spot_data query_image_dir query_pointer.txt"
           " page_image_dir page_label.gtp windowSize numTruePerClass numFalsePerClass"
           " out_query_db_file out_page_db_file out_label_db_file\n"
           );
  } else {
    google::InitGoogleLogging(argv[0]);
    string query_image_dir=argv[1];
    string query_pointer_file=argv[2];
    string page_image_dir = argv[3];
    string page_label_file = argv[4];

    int wSize = atoi(argv[5]);
    int numTruePerClass = atoi(argv[6]);
    int numFalsePerClass= atoi(argv[7]);

    string out_query_name = argv[8];
    string out_page_name = argv[9];
    string out_label_name = argv[10];

    vector<string> classes;
    map<string,int> classMap;
    int classCounter=1;
    vector<bool> usedClass;
    map<string, vector<string> > queries;
    map<string, vector<bool> > usedQuery;

    set<string> imagePaths;
    map<string,cv::Mat> images, labels;
    map<string, vector< tuple<string,int,int,int,int> > > instances;
    map<string, vector<bool> > usedInstance;

    int overlapIndex=0;
    map< set<int>, int, setcomp > overlapC_I;
    map< int, set<int> > overlapI_C;


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
        queries[c].push_back(query_image_dir+fileName);
    }
    filein.close();
    for (auto p : queries)
    {
        classes.push_back(p.first);
        classMap[p.first]=classCounter++;
        usedQuery[p.first].resize(p.second.size());
    }
    usedClass.resize(classes.size());
    
    
    
    filein.open(page_label_file);
    assert(filein.good());
    string curPathIm="";
    cv::Mat curIm;
    int minDim=999999;
    while (getline(filein,line))
    {
        stringstream ss(line);
        string part;
        getline(ss,part,' ');

        string pathIm=page_image_dir+string(part);
        //pathIms.push_back(pathIm);
        if (images.find(pathIm)==images.end())
        {
            images[pathIm] = cv::imread(pathIm,CV_LOAD_IMAGE_GRAYSCALE);
            if (images[pathIm].rows==0)
                cout<<"Failed to open: "<<pathIm<<endl;

            curIm=images[pathIm];
            labels[pathIm] = cv::Mat::zeros(images.at(pathIm).size(),CV_32S);
        }
        
        /*if (curPathIm.compare(pathIm)!=0)
        {
            curPathIm=pathIm;
            curIm = cv::imread(curPathIm,CV_LOAD_IMAGE_GRAYSCALE);
            if (curIm.rows<1)
            {
                cout<<"Error reading: "<<curPathIm<<endl;
                assert(curIm.rows>0);
            }
        }*/
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
        {
            cout<<pathIm<<endl;
            cout<<"line: ["<<line<<"]  loc: "<<x1<<" "<<y1<<" "<<x2<<" "<<y2<<"  im:"<<curIm.rows<<","<<curIm.cols<<endl;
            assert(false);
        }
        getline(ss,part,' ');
        string label=part;
        instances[label].push_back(make_tuple(pathIm,x1,y1,x2,y2));
        imagePaths.insert(pathIm);
        if (classMap.find(label)==classMap.end())
        {
            continue;
            classMap[label]=classCounter++;
        }
        //labels.at(pathIm)(loc)=classMap.at(label);
        for (int r=y1; r<=y2; r++)
            for (int c=x1; c<=x2; c++)
            {
                int v = labels.at(pathIm).at<int>(r,c);
                if (v==0 || v==classMap.at(label))
                    labels.at(pathIm).at<int>(r,c)=classMap.at(label);
                else if (v<0)
                {
                    if (overlapI_C.at(v).find(classMap.at(label)) == overlapI_C.at(v).end())//am I not in the overlapping class
                    {
                        set<int> o = overlapI_C.at(v);
                        o.insert(classMap.at(label));
                        if (overlapC_I.find(o) == overlapC_I.end())//does the desired overlap class not exist?
                        {
                            //Make it!
                            overlapC_I[o]=--overlapIndex;
                            overlapI_C[overlapIndex]=o;
                        }
                        labels.at(pathIm).at<int>(r,c)=overlapC_I[o];
                    }
                }
                else//We need to overlap
                {
                    set<int> o;
                    o.insert(v);
                    o.insert(classMap.at(label));
                    if (overlapC_I.find(o) == overlapC_I.end())//does the desired overlap class not exist?
                    {
                        //Make it!
                        overlapC_I[o]=--overlapIndex;
                        overlapI_C[overlapIndex]=o;
                    }
                    labels.at(pathIm).at<int>(r,c)=overlapC_I[o];
                }
            }

        if (x2-x1+1 < minDim)
            minDim = x2-x1+1;
        if (y2-y1+1 < minDim)
            minDim = y2-y1+1;

    }
    filein.close();
    for (auto p : instances)
        usedInstance[p.first].resize(p.second.size());


    ///////////////////////////////////////////////////

    // Open files
    // Read the magic and the meta data
    uint32_t num_items=0;
    int num_true=0;
    int num_false=0;
    uint32_t num_labels;


    // Open leveldb
    leveldb::DB* queries_db;
    leveldb::Options options;
    options.create_if_missing = true;
    options.error_if_exists = true;
    leveldb::Status status = leveldb::DB::Open(
        options, out_query_name, &queries_db);
    CHECK(status.ok()) << "Failed to open leveldb " << out_query_name
        << ". Is it already existing?";






    char label_i;
    char label_j;
    //char* pixels = new char[2 * rows * cols];
    //std::string value;

    //caffe::Datum datum;
    //datum.set_channels(2);  // one channel for each image in the pair
    //datum.set_height(rows);
    //datum.set_width(cols);
    list< tuple<string,int,string,cv::Rect> >toWrite; //query, classID, page, bounding box
    for (auto cAndId : classMap)
    {
        string cls = cAndId.first;
        int cId = cAndId.second;
        if (instances.find(cls) == instances.end())
            continue;
        for (int i=0; i<numTruePerClass; i++)
        {
            int query = getNextIndex(usedQuery.at(cls));
            int instanceIndex = getNextIndex(usedInstance.at(cls));

            tuple<string,int,int,int,int> instance = instances.at(cls).at(instanceIndex);
            string pathIm = get<0>(instance);
            int x0 = get<1>(instance);
            int y0 = get<2>(instance);
            int x1 = get<3>(instance);
            int y1 = get<4>(instance);
            //Find bounds for valid top-right points for the window that fully include the bb
            int wX0Min = max(0,x0-(wSize-(x1-x0+1)));
            int wX0Max = min(images.at(pathIm).cols-wSize,x0);
            int wY0Min = max(0,y0-(wSize-(y1-y0+1)));
            int wY0Max = min(images.at(pathIm).rows-wSize,y0);
            //Randomly select the window given those bounds
            int wX0 = wX0Min + rand()%(wX0Max-wX0Min);
            int wY0 = wY0Min + rand()%(wY0Max-wY0Min);
            cv::Rect window(wX0,wY0,wSize,wSize);
            assert(wX0>=0 && wX0+wSize<images.at(pathIm).cols);
            assert(wY0>=0 && wY0+wSize<images.at(pathIm).rows);
            toWrite.emplace(toWrite.end(),queries.at(cls).at(query),cId,pathIm,window);
        }

        for (int i=0; i<numFalsePerClass; i++)
        {
            int query = getNextIndex(usedQuery.at(cls));
            int page = rand()%imagePaths.size();
            auto iter = imagePaths.begin();
            for (int ii=0; ii<page; ii++)
                ++iter;
            string pathIm = *iter;
            
            cv::Rect window;
            bool match=true;
            int count=0;
            while(match)//Until we find a window without class c
            {
                if (count++>8000)
                {
                    if (imagePaths.size()==1)
                        break;
                    //cout<<"Couldn't find window on "<<pathIm<<" for class "<<c<<endl;
                    page = rand()%imagePaths.size();
                    iter = imagePaths.begin();
                    for (int ii=0; ii<page; ii++)
                        ++iter;
                    pathIm = *iter;
                    count=0;
                }
                cv::Mat& labelIm = labels.at(pathIm);
                match=false;
                //Randomly select a window
                int wX0 = rand()%(images.at(pathIm).cols-wSize);
                int wY0 = rand()%(images.at(pathIm).rows-wSize);
                window = cv::Rect(wX0,wY0,wSize,wSize);
                assert(window.x>=0 && window.x+wSize<images.at(pathIm).cols);
                assert(window.y>=0 && window.y+wSize<images.at(pathIm).rows);
                //Now, we check to see if the target class is present in the hypothesis window
                //to speed this up, we stide along the minimum bb size
                for (int r=wY0; r<wY0+wSize; r+=minDim)
                    for (int c=wX0; c<wX0+wSize; c+=minDim)
                        if (labelIm.at<int>(r,c)==cId)
                        {
                            match=true;
                            r=wY0+wSize;
                            break;
                        }
                        else if (labelIm.at<int>(r,c)<0)
                        {//check overlapping classes
                            if (overlapI_C.at(labelIm.at<int>(r,c)).find(cId)!=overlapI_C.at(labelIm.at<int>(r,c)).end())
                            {
                                match=true;
                                r=wY0+wSize;
                                break;
                            }
                        }
                //Then we check the far edges that are likely missed
                for (int r=wY0; r<wY0+wSize-1; r+=1)
                {
                    int c = wX0+wSize-1;
                    if (labelIm.at<int>(r,c)==cId)
                    {
                        match=true;
                        r=wY0+wSize;
                        break;
                    }
                    else if (labelIm.at<int>(r,c)<0)
                    {
                        if (overlapI_C.at(labelIm.at<int>(r,c)).find(cId)!=overlapI_C.at(labelIm.at<int>(r,c)).end())
                        {
                            match=true;
                            r=wY0+wSize;
                            break;
                        }
                    }
                }
                for (int c=wX0; c<wX0+wSize-1; c+=minDim)
                {
                    int r = wY0+wSize-1;
                    if (labelIm.at<int>(r,c)==cId)
                    {
                        match=true;
                        r=wY0+wSize;
                        break;
                    }
                    else if (labelIm.at<int>(r,c)<0)
                    {
                        if (overlapI_C.at(labelIm.at<int>(r,c)).find(cId)!=overlapI_C.at(labelIm.at<int>(r,c)).end())
                        {
                            match=true;
                            r=wY0+wSize;
                            break;
                        }
                    }
                }
            }
            if (imagePaths.size()==1 && count>8000)
                break;
            assert(window.x>=0 && window.x+wSize<images.at(pathIm).cols);
            assert(window.y>=0 && window.y+wSize<images.at(pathIm).rows);
            //Great, a good window!
            toWrite.emplace(toWrite.end(),queries.at(cls).at(query),cId,pathIm,window);
        }
    }

    //write them in random order
    vector< tuple<string,int,string,cv::Rect> > randWrite(toWrite.size()); //query, classID, page, bounding box
    while (toWrite.size()>0) {
        int i = caffe::caffe_rng_rand() % toWrite.size();
        auto iter = toWrite.begin();
        for (int ii=0; ii<i; ii++) iter++;
        randWrite[toWrite.size()-1] = *iter;
        toWrite.erase(iter);
    }

    for (int index=0; index<randWrite.size(); index++)
    {
        string query=get<0>(randWrite[index]);
        int classId=get<1>(randWrite[index]);
        string pathIm=get<2>(randWrite[index]);
        cv::Rect bb=get<3>(randWrite[index]);
        string ser_query= read_image(query);
        if (ser_query.length()==0)
            continue;
#ifdef DEBUG
        cv::imshow("page",images.at(pathIm)(bb));
        cv::imshow("label",labelIm);
        cv::waitKey();
#endif

        char buff[12];
        snprintf(buff, sizeof(buff), "%09d", index);
        std::string key_str = buff; //caffe::format_int(num_items, 8);
        leveldb::Status s = queries_db->Put(leveldb::WriteOptions(), key_str, ser_query);
        assert(s.ok());
        num_items++;


    }
    delete queries_db;

    leveldb::DB* pages_db;
    leveldb::Status status1 = leveldb::DB::Open(
        options, out_page_name, &pages_db);
    CHECK(status1.ok()) << "Failed to open leveldb " << out_page_name
        << ". Is it already existing?";

    for (int index=0; index<randWrite.size(); index++)
    {
        string query=get<0>(randWrite[index]);
        int classId=get<1>(randWrite[index]);
        string pathIm=get<2>(randWrite[index]);
        cv::Rect bb=get<3>(randWrite[index]);
        string ser_page= serialize_image(images.at(pathIm)(bb).clone());
        //Create binary label image
#ifdef DEBUG
        cv::imshow("page",images.at(pathIm)(bb));
        cv::waitKey();
#endif

        char buff[12];
        snprintf(buff, sizeof(buff), "%09d", index);
        std::string key_str = buff; //caffe::format_int(num_items, 8);
        leveldb::Status s = pages_db->Put(leveldb::WriteOptions(), key_str, ser_page);
        assert(s.ok());


    }
    delete pages_db;

    leveldb::DB* labels_db;
    leveldb::Status status2 = leveldb::DB::Open(
        options, out_label_name, &labels_db);
    CHECK(status2.ok()) << "Failed to open leveldb " << out_label_name
        << ". Is it already existing?";

    for (int index=0; index<randWrite.size(); index++)
    {
        string query=get<0>(randWrite[index]);
        int classId=get<1>(randWrite[index]);
        string pathIm=get<2>(randWrite[index]);
        cv::Rect bb=get<3>(randWrite[index]);
        //Create binary label image
        cv::Mat label = labels.at(pathIm)(bb);
        cv::Mat labelIm = cv::Mat::zeros(bb.height,bb.width,CV_8U);
        for (int r=0; r<labelIm.rows; r++)
            for (int c=0; c<labelIm.cols; c++)
            {
                if (label.at<int>(r,c)==classId || (label.at<int>(r,c)<0 && overlapI_C.at(label.at<int>(r,c)).find(classId) != overlapI_C.at(label.at<int>(r,c)).end()))
                    labelIm.at<unsigned char>(r,c)=255;
            }
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


    }

    cout << "A total of    " << num_items << " items written."<<endl;

    delete labels_db;
  }
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires LevelDB; compile with USE_LEVELDB.";
}
#endif  // USE_LEVELDB
