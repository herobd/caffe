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
#include "dataset.h"
#include "SubwordSpottingResult.h"
#include "CNNEmbedder.h"

using namespace cv;
using namespace std;

#define NET_IN_SIZE 52
#define NET_PIX_STRIDE 8

class CNNSpotter
{

public:
    vector< SubwordSpottingResult > subwordSpot(const Mat& exemplar, float refinePortion=0.25) const;
    vector< SubwordSpottingResult > subwordSpot_eval(const Mat& exemplar, float refinePortion, vector< SubwordSpottingResult >* accumRes, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, float* ap, float* accumAP) const;

private:
    const Dataset* corpus_dataset;

    int window_stride;
    vector<Mat> embedded_corpus;

    CNNEmbedder* embedder;
    SubwordSpottingResult refine(float score, int imIdx, int windIdx, int s_windowWidth, int s_stride, const Mat& embedding) const;
};


#endif
