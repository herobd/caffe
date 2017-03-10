#ifndef DATASET_H
#define DATASET_H


#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <fstream>
#include <iostream>
#include <dirent.h>



using namespace std;
using namespace cv;

class Dataset
{

public:
    
    virtual const vector<string>& labels() const =0;
    virtual int size() const =0;
    virtual const Mat image(unsigned int) const =0;
};
#endif
