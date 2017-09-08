//g++ -std=c++11 make_phoc_data.cpp -lcaffe -lglog -l:libopencv_core.so.3.0 -l:libopencv_highgui.so.3.0 -l:libopencv_imgproc.so.3.0 -l:libopencv_imgcodecs.so.3.0 -lprotobuf -lleveldb -I ../include/ -L ../build/lib/ -o make_phoc_data
//g++ -std=c++11 make_phoc_data.cpp -lcaffe -lglog -l:libopencv_core.so.3.1 -l:libopencv_highgui.so.3.1 -l:libopencv_imgproc.so.3.1 -l:libopencv_imgcodecs.so.3.1 -lprotobuf -lleveldb -I ../include/ -L ../build/lib/ -o make_phoc_data
//g++ -std=c++11 make_phoc_data.cpp -lcaffe -lglog -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lprotobuf -lleveldb -I ../include/ -L ../build/lib/ -o make_phoc_data
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
        if (im.rows*im.cols<2)
            return "";
        assert(im.rows*im.cols>1);
        
        caffe::Datum datum;
        datum.set_channels(1);  
        datum.set_height(im.rows);
        datum.set_width(im.cols);
        //datum.set_label(label);
	//copy(((char*)im.data),((char*)im.data)+(rows*cols),pixels);
        if (im.isContinuous())
        {
            datum.set_data(im.data, im.rows*im.cols);
        }
        else
        {
            cv::Mat newIm;
            im.copyTo(newIm);
            assert(newIm.isContinuous());
            datum.set_data(newIm.data, newIm.rows*newIm.cols);
        }
        string ret;

        datum.SerializeToString(&ret);
        return ret;
}
string read_image(string image_file) {
	cv::Mat im = cv::imread(image_file,CV_LOAD_IMAGE_GRAYSCALE);
//#ifdef DEBUG
//        cv::imshow("image",im);
//        cv::waitKey();
//#endif
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
    
  int counter=0;
  if (!test)
  {
      //Even word distribution, following after PHOCNet paper
      map<string,vector<int> > wordMap;
      for (int i=0; i<labels.size(); i++)
          wordMap[labels[i]].push_back(i);

      int maxCount=0;
      int averageCount=0;
      for (auto p : wordMap)
      {
          averageCount+=p.second.size();
          if (p.second.size() > maxCount)
              maxCount = p.second.size();
      }
      averageCount /= wordMap.size();
      int goalCount = 0.6*maxCount + 0.4*averageCount;
      cout<<"goal count: "<<goalCount<<endl;
      goalCount = min(goalCount,1000);
      cout<<"goal count: "<<goalCount<<endl;


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
          counter++;
          if (counter%500==0)
              cout<<((100.0*counter)/wordMap.size())<<"% preped"<<endl;
      }
  }
    


  char label_i;
  char label_j;
  //char* pixels = new char[2 * rows * cols];
  //std::string value;

  //caffe::Datum datum;
  //datum.set_channels(2);  // one channel for each image in the pair
  //datum.set_height(rows);
  //datum.set_width(cols);
  vector<bool> used(labels.size());
  list<int>toWrite;
  LOG(INFO) << "from " << labels.size() << " items.";
  map<string,int> bigramCounts;
  for (int i=0; i<labels.size(); i++) {
      //int inst = caffe::caffe_rng_rand() % image_filenames.size();  // pick a random  
      //int start=inst;
      //while (used[inst])
      //    inst=(inst+1)%image_filenames.size();
      toWrite.push_back(i);

      //estimate bigram counts
      for (int a=0; a<labels[i].size()-1; a++)
      {
          bigramCounts[labels[i].substr(a,2)]++;
      }
  }
  
  cout<<"writing."<<endl;
  //write them in random order
  counter=0;
  int initSize = toWrite.size();
  while (toWrite.size()>0) {
        int i = caffe::caffe_rng_rand() % toWrite.size();
        auto iter = toWrite.begin();
        for (int ii=0; ii<i; ii++) iter++;
        int im=(*iter);
#ifdef DEBUG
        cout<<labels[im]<<endl;
#endif
        string label = prep_vec(phocs[im]);
        string value;
        if (image_filenames.size()>0)
        {
            value = read_image(image_filenames[im]);
            if (value.length()==0)
                cout<<"Failed to open "<<image_filenames[im]<<endl;
        }
        else
            value = serialize_image(images[im]);
        if (value.length()>0)
        {
            char buff[10];
            snprintf(buff, sizeof(buff), "%08d", num_items);
            std::string key_str = buff; //caffe::format_int(num_items, 8);
            images_db->Put(leveldb::WriteOptions(), key_str, value);
            labels_db->Put(leveldb::WriteOptions(), key_str, label);
            num_items++;
        }

        toWrite.erase(iter);
    
      counter++;
      if (counter%500==0)
          cout<<((100.0*counter)/initSize)<<"% written"<<endl;
  }
  
  cout << "A total of    " << num_items << " items written."<<endl;
  /*
  multimap<int,string> flipped;
  for (auto p : bigramCounts)
      flipped.emplace(-1*p.second,p.first);
  auto iter = flipped.begin();
  for (int i=0; i<500; i++)
  {
      cout<<iter->second<<": "<<iter->first<<endl;
      iter++;
  }*/

  delete images_db;
  delete labels_db;
}


//copied from EmbAttSpotter
void computePhoc(string str, map<char,int> vocUni2pos, map<string,int> vocBi2pos, int Nvoc, vector<int> levels, int descSize, vector<float>* out)
{
    int strl = str.length();

    int doUnigrams = vocUni2pos.size()!=0;
    int doBigrams = vocBi2pos.size()!=0;

    /* For each block */
    //float *p = out;
    int p=0;
    for (int level : levels)
    {
        /* For each split in that level */
        for (int ns=0; ns < level; ns++)
        {
            float starts = ns/(float)level;
            float ends = (ns+1)/(float)level;

            /* For each character */
            if (doUnigrams)
            {
                for (int c=0; c < strl; c++)
                {
                    if (vocUni2pos.count(str[c])==0)
                    {
                        /* Character not included in dictionary. Skipping.*/
                        continue;
                    }
                    int posOff = vocUni2pos[str[c]]+p;
                    float startc = c/(float)strl;
                    float endc = (c+1)/(float)strl;

                    /* Compute overlap over character size (1/strl)*/
                    if (endc < starts || ends < startc) continue;
                    float start = (starts > startc)?starts:startc;
                    float end = (ends < endc)?ends:endc;
                    float ov = (end-start)*strl;
                    #if HARD
                    if (ov >=0.48)
                    {
                        //p[posOff]+=1;
                        //out.at<float>(posOff,instance)+=1;
                        out->at(posOff)+=1;
                    }
                    #else
                    //p[posOff] = max(ov, p[posOff]);
                    //out.at<float>(posOff,instance)=max(ov, out.at<float>(posOff,instance));
                    out->at(posOff) = max(ov, out->at(posOff));
                    #endif
                }
            }
            if (doBigrams)
            {
                for (int c=0; c < strl-1; c++)
                {
                    string sstr=str.substr(c,2);
                    if (vocBi2pos.count(sstr)==0)
                    {
                        /* Character not included in dictionary. Skipping.*/
                        continue;
                    }
                    int posOff = vocBi2pos[sstr]+p;
                    float startc = c/(float)strl;
                    float endc = (c+2)/(float)strl;

                    /* Compute overlap over bigram size (2/strl)*/
                    if (endc < starts || ends < startc){ continue;}
                    float start = (starts > startc)?starts:startc;
                    float end = (ends < endc)?ends:endc;
                    float ov = (end-start)*strl/2.0;
                    if (ov >=0.48)
                    {
                        //p[posOff]+=1;
                        //out.at<float>((out.rows-descSize)+posOff,instance)+=1;
                        out->at((out->size()-descSize)+posOff)+=1;
                    }
                }
            }
            p+=Nvoc;
        }
    }
    return;
}

int main(int argc, char** argv) {
  if (argc != 6 && argc!=7 ) {
    printf("This script converts the dataset to 2 leveldbs, one of images, one of phoc vectors.\n"
           "Usage:\n"
           " make_phoc_data image_dir image_label_file bigramfile"
           " out_image_db_file out_label_db_file [-t]\n"
           );
  } else {
    google::InitGoogleLogging(argv[0]);
    string image_dir=argv[1];
    string labelfile = argv[2];
    string bigramfile = argv[3];

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
    vector<int> phoc_levels_bi = {2};
    ifstream bFile(bigramfile);
    assert(bFile.is_open());
    string bigram;
    vector<string> bigrams;
    while (getline(bFile,bigram)  
#ifndef FULL
            && bigrams.size()<50
#endif
          )
    {
        bigrams.push_back(bigram);
    }
    bFile.close();
    assert(bigrams.size()>0);
    /* Prepare dict */


    map<char,int> vocUni2pos;
    for (int i=0; i<unigrams.size(); i++)
    {
        vocUni2pos[unigrams[i]] = i;
    }

    map<std::string,int> vocBi2pos;
    for (int i=0; i<bigrams.size(); i++)
    {
        vocBi2pos[bigrams[i]] = i;
    }


    int totalLevels = 0;
    for (int level : phoc_levels)
    {
        totalLevels+=level;
    }

    int phocSize = totalLevels*unigrams.size();

    int phocSize_bi=0;
    for (int level : phoc_levels_bi)
    {
        phocSize_bi+=level*bigrams.size();
    }
    cout<<"vector size: "<<phocSize+phocSize_bi<<endl;
    //assert(phocSize+phocSize_bi==604);
    ///
    vector<string> image_paths;
    vector<cv::Mat> images;
    vector<vector<float> > phocs;
    vector<string> labels;

    string extension = labelfile.substr(labelfile.find_last_of('.')+1);
    bool gtp=extension.compare("gtp")==0;

    ifstream filein(labelfile);
    assert(filein.good());
    //2700270.tif 519 166 771 246 orders
    string line;
    //regex qExtGtp("(\\S+\\.\\S+) (\\d+) (\\d+) (\\d+) (\\d+) (\\w+)");
    //regex qExt("(\\S+\\.\\S+) (\\w+)");
    
    
    
#ifdef DEBUG
    float avgH=0;
    int minH=0;
#endif
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
            int x1=stoi(part);
            getline(ss,part,' ');
            int y1=stoi(part);
            getline(ss,part,' ');
            int x2=stoi(part);
            getline(ss,part,' ');
            int y2=stoi(part);
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
            vector<float> phoc(phocSize+phocSize_bi);
            computePhoc(label, vocUni2pos, map<string,int>(),unigrams.size(), phoc_levels, phocSize, &phoc);
            computePhoc(label, map<char,int>(), vocBi2pos,bigrams.size(), phoc_levels_bi, phocSize_bi, &phoc);
            phocs.push_back(phoc);
            labels.push_back(label);

#ifdef DEBUG
            avgH+=loc.height;
            if (loc.height < avgH)
                minH = loc.height;
            /*cout<<label<<": "<<endl;
            for (float f : phoc)
                cout<<f<<", ";
            cout<<endl;
            int iii;
            cin>>iii;*/
#endif
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
            vector<float> phoc(phocSize+phocSize_bi);
            computePhoc(label, vocUni2pos, map<string,int>(),unigrams.size(), phoc_levels, phocSize, &phoc);
            computePhoc(label, map<char,int>(), vocBi2pos,bigrams.size(), phoc_levels_bi, phocSize_bi, &phoc);
            image_paths.push_back(pathIm);
            phocs.push_back(phoc);
            labels.push_back(label);
#ifdef DEBUG
            /*
            cout<<label<<": "<<endl;
            for (float f : phoc)
                cout<<f<<", ";
            cout<<endl;
            int iii;
            cin>>iii;
            */
#endif
        }
    }

#ifdef DEBUG
    if (gtp)
    {
        cout<<"avgH: "<<avgH/labels.size()<<endl;
        cout<<"minH: "<<minH<<endl;
    }
#endif
    filein.close();
    cout<<"finished reading. Creating dataset"<<endl;
    convert_dataset(image_paths,images,phocs,labels, argv[4], argv[5],argc>6);
  }
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires LevelDB; compile with USE_LEVELDB.";
}
#endif  // USE_LEVELDB
