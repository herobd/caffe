#ifndef GWDATASET_H
#define GWDATASET_H


#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <fstream>
#include <iostream>
#include <dirent.h>
#include <assert.h>
#include <regex>

#include "dataset.h"


using namespace std;
using namespace cv;

class GWDataset : public Dataset
{
private:
    vector<string> pathIms, _labels;
    vector<Rect> locs;
    vector<Mat> wordImages;
    string name;
    
public:
    GWDataset(const string& queries, const string& imDir, int minH=-1, int maxH=-1, int margin=0);
    virtual const vector<string>& labels() const;
    virtual int size() const;
    virtual const Mat image(unsigned int i) const;
    virtual string getName() const;
    Rect getLoc(int i) {return locs.at(i);}
};
#endif
