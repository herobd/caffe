#ifndef CNNSPOTTER_H
#define CNNSPOTTER_H

#define TEST_MODE 1


#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <vector>
#include <deque>
#include <fstream>
#include <iostream>
#include <dirent.h>
#include <iomanip>
#include <functional>
#include <limits>
#include "dataset.h"
#include "SubwordSpottingResult.h"
#include "cnnembedder.h"

using namespace cv;
using namespace std;

//#define NET_IN_SIZE 52
//#define NET_PIX_STRIDE 8
#define HALF_STRIDE 0

class CNNSpotter
{

public:
    CNNSpotter(string netModel, string netWeights, int netInputSize=52, int netPixelStride=8, string saveName="cnnspotter");
    ~CNNSpotter();

    vector< SubwordSpottingResult > subwordSpot(const Mat& exemplar, float refinePortion=0.25) const;
    vector< SubwordSpottingResult > subwordSpot_eval(const Mat& exemplar, float refinePortion, vector< SubwordSpottingResult >* accumRes, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, float* ap, float* accumAP) const;

    float evalSubwordSpotting_singleScore(string ngram, const vector<SubwordSpottingResult>& res, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, int skip=-1) const;

    void evalSubwordSpotting(const Dataset* exemplars, const Dataset* data);
    void evalSubwordSpottingWithCharBounds(const Dataset* data, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds);
   // void evalSubwordSpottingCombine(const Dataset* exemplars, const Dataset* data);


private:
    string saveName;
    const Dataset* corpus_dataset;

    vector<Mat> corpus_embedded;
    vector<float> corpus_scalars;

    CNNEmbedder* embedder;
    int NET_IN_SIZE;
    int NET_PIX_STRIDE;

    SubwordSpottingResult refine(float score, int imIdx, int windIdx, const Mat& exemplarEmbedding) const;
    void setCorpus_dataset(const Dataset* dataset);

    void writeFloatMat(ofstream& dst, const Mat& m);
    Mat readFloatMat(ifstream& src);
};


#endif
