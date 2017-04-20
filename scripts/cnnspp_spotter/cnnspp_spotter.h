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
#include "phocer.h"
#include "Transcriber.h"
#include <mutex>

using namespace cv;
using namespace std;

#define TRANSCRIBE_KEEP_PORTION 0.25


class CNNSPPSpotter : public Transcriber
{

public:
    CNNSPPSpotter(string featurizerModel, string embedderModel, string netWeights, bool normalizeEmbedding=true, float featurizeScale=.25, int charWidth=33, int stride=4, string saveName="cnnspp_spotter");
    ~CNNSPPSpotter();

    void setCorpus_dataset(const Dataset* dataset, bool fullWordEmbed);

    vector< SubwordSpottingResult > subwordSpot(const Mat& exemplar, float refinePortion=0.25);
    vector< SubwordSpottingResult > subwordSpot(const string& exemplar, float refinePortion=0.25);
    vector< SubwordSpottingResult > subwordSpot_eval(const Mat& exemplar, string word, float refinePortion, vector< SubwordSpottingResult >* accumRes, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, float* ap, float* accumAP, mutex* resLock);
    vector< SubwordSpottingResult > subwordSpot_eval(const string& exemplar, float refinePortion, vector< SubwordSpottingResult >* accumRes, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, float* ap, float* accumAP, mutex* resLock);

    float compare(string text, const Mat& image);
    float compare(string text, int wordIndex);
    float compare(int wordIndex, int wordIndex2);

    float evalSubwordSpotting_singleScore(string ngram, const vector<SubwordSpottingResult>& res, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, int skip=-1, multimap<float,int>* trues=NULL, multimap<float,int>* alls=NULL);

    void evalSubwordSpotting(const Dataset* exemplars, const Dataset* data);
    void evalSubwordSpotting(const vector<string>& exemplars, const Dataset* data);
    void evalSubwordSpottingWithCharBounds(const Dataset* data, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds);
   // void evalSubwordSpottingCombine(const Dataset* exemplars, const Dataset* data);
    void evalSubwordSpottingRespot(const Dataset* data, vector<string> toSpot, int numSteps, int numRepeat, int repeatSteps, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds);


    void evalRecognition(const Dataset* data, const vector<string>& lexicon);
    multimap<float,string> transcribe(const Mat& image);
    multimap<float,string> transcribeCorpus(int i);
    vector< multimap<float,string> > transcribeCorpus();
    vector< multimap<float,string> > transcribe(Dataset* words);

    void addLexicon(const vector<string>& lexicon);

    static string lowercaseAndStrip(string s);

private:
    string saveName;
    string featurizerFile, embedderFile;
    const Dataset* corpus_dataset;


    vector<Mat> corpus_embedded;
    vector< vector<Mat>* > corpus_featurized;

    CNNFeaturizer* featurizer;
    SPPEmbedder* embedder;
    int windowWidth, stride;
    float featurizeScale;

    PHOCer phocer;

    vector<string> lexicon;
    Mat lexicon_phocs;

    float compare_(string text, vector<Mat>* im_featurized);
    SubwordSpottingResult refine(float score, int imIdx, int windIdx, const Mat& exemplarEmbedding);

    void writeFloatMat(ofstream& dst, const Mat& m);
    Mat readFloatMat(ifstream& src);

    void refineStep(int imIdx, float* bestScore, int* bestX0, int* bestX1, float scale, const Mat& exemplarEmbedding);
    void refineStepFast(int imIdx, float* bestScore, int* bestX0, int* bestX1, float scale, const Mat& exemplarEmbedding);
    Mat embedFromCorpusFeatures(int imIdx, Rect window);

    float calcAP(const vector<SubwordSpottingResult>& res, string ngram);
    void _eval(string word, vector< SubwordSpottingResult >& ret, vector< SubwordSpottingResult >* accumRes, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, float* ap, float* accumAP, multimap<float,int>* truesAccum=NULL, multimap<float,int>* allsAccum=NULL, multimap<float,int>* truesN=NULL, multimap<float,int>* allN=NULL);
    float getRankChangeRatio(const vector<SubwordSpottingResult>& prevRes, const vector<SubwordSpottingResult>& res, const multimap<float,int>& prevTrues, const multimap<float,int>& trues, const multimap<float,int>& prevAlls, const multimap<float,int>& allsN, float* rankDrop, float* rankRise, float* rankDropFull, float* rankRiseFull, float* mean, float* std, float* meanTop, float* stdTop);
};


#endif
