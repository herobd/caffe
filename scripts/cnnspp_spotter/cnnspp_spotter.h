#ifndef CNNSPPSPOTTER_H
#define CNNSPPSPOTTER_H

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
#include "cnn_featurizer.h"
#include "spp_embedder.h"

using namespace cv;
using namespace std;


class CNNSPPSpotter
{

public:
    CNNSPPSpotter(string featurizerModel, string embedderModel, string netWeights, bool normalizeEmbedding, float featurizeScale=.25, int windowWidth=65, int stride=3, string saveName="cnnspp_spotter");
    ~CNNSPPSpotter();

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
    vector< vector<Mat>* > corpus_featurized;

    CNNFeaturizer* featurizer;
    SPPEmbedder* embedder;
    int windowWidth, stride;
    float featurizeScale;

    SubwordSpottingResult refine(float score, int imIdx, int windIdx, const Mat& exemplarEmbedding) const;
    void setCorpus_dataset(const Dataset* dataset);

    void writeFloatMat(ofstream& dst, const Mat& m);
    Mat readFloatMat(ifstream& src);
};


#endif
