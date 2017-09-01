#ifndef CNNSPPSPOTTER_H
#define CNNSPPSPOTTER_H

#define TEST_MODE_CNNSPP 0


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
#define DEFAULT_REFINE_PORTION 0.25
#define BRAY_CURTIS 0


class CNNSPPSpotter : public Transcriber
{

public:
    CNNSPPSpotter(string featurizerModel, string embedderModel, string netWeights, set<int> ngrams, bool normalizeEmbedding=true, float featurizeScale=.25, int charWidth=33, int stride=4, string saveName="cnnspp_spotter", bool ideal_comb=false);
    ~CNNSPPSpotter();

    void setCorpus_dataset(const Dataset* dataset, bool fullWordEmbed_only=false);

    vector< SubwordSpottingResult > subwordSpot(int numChar, const Mat& exemplar, float refinePortion=DEFAULT_REFINE_PORTION);
    vector< SubwordSpottingResult > subwordSpot(const string& exemplar, float refinePortion=DEFAULT_REFINE_PORTION);
    vector< SubwordSpottingResult > subwordSpot(int numChar, int exemplarId, int x0, float refinePortion=DEFAULT_REFINE_PORTION);
    vector< SubwordSpottingResult > subwordSpot(int numChar, int exemplarId, int x0, int x1, int focus0, int focus1, float refinePortion=DEFAULT_REFINE_PORTION);
    vector< SubwordSpottingResult > subwordSpot_eval(const Mat& exemplar, string word, float refinePortion, vector< SubwordSpottingResult >* accumRes, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, float* ap, float* accumAP, mutex* resLock, float help=-1);
    vector< SubwordSpottingResult > subwordSpot_eval(int exemplarId, int x0, string word, float refinePortion, vector< SubwordSpottingResult >* accumRes, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, float* ap, float* accumAP, mutex* resLock, float help=-1);
    vector< SubwordSpottingResult > subwordSpot_eval(const string& exemplar, float refinePortion, vector< SubwordSpottingResult >* accumRes, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, float* ap, float* accumAP, mutex* resLock, float help=-1);

    multimap<float,int> wordSpot(const Mat& exemplar);
    multimap<float,int> wordSpot(const string& exemplar);
    multimap<float,int> wordSpot(int exemplarIndex);
    float compare(string text, const Mat& image);
    float compare(string text, int wordIndex);
    float compare(int wordIndex, int wordIndex2);

    void helpAP(vector<SubwordSpottingResult>& res, string ngram, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, float goalAP);

    float evalSubwordSpotting_singleScore(string ngram, vector<SubwordSpottingResult>& res, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, int skip=-1, multimap<float,int>* trues=NULL, multimap<float,int>* alls=NULL,vector<int>* notSpottedIn=NULL);

    float evalWordSpotting_singleScore(string word, const multimap<float,int>& res, int skip=-1, multimap<float,int>* trues=NULL);

    void evalSubwordSpotting(const Dataset* exemplars, const Dataset* data);
    void evalSubwordSpotting(const vector<string>& exemplars, const Dataset* data);
    void evalSubwordSpottingWithCharBounds(const Dataset* data, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds);
   // void evalSubwordSpottingCombine(const Dataset* exemplars, const Dataset* data);
    void evalSubwordSpottingRespot(const Dataset* data, vector<string> toSpot, int numSteps, int numRepeat, int repeatSteps, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds);
    void evalFullWordSpottingRespot(const Dataset* data, vector<string> toSpot, int numSteps, int numRepeat, int repeatSteps);
    void evalFullWordSpotting(const Dataset* data);

    void demonstrateClustering(string destDir, string ngram, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds);


    void evalRecognition(const Dataset* data, const vector<string>& lexicon);
    multimap<float,string> transcribe(const Mat& image);
    multimap<float,string> transcribeCorpus(int i);
    vector< multimap<float,string> > transcribeCorpus();
    vector< multimap<float,string> > transcribe(Dataset* words);

    void addLexicon(const vector<string>& lexicon);

    static string lowercaseAndStrip(string s);

    Mat cpv(int i);
    Mat npv(int i);
    void npvPrep(const vector<string>& ngrams);
    void cpvPrep(const vector<string>& ngrams) {npvPrep(ngrams);}

    //This is a function used for the graph-transcription paradigm
    //It returns the combined results of QbS spotting the given ngrams (each result is also spotted for other ngrams)
    //and additionally has scores for densely comparing all results to eachother (QbE score) in crossScores
    vector<SpottingLoc> massSpot(const vector<string>& ngrams, Mat& crossScores);

private:
    string saveName;
    string featurizerFile, embedderFile;
    const Dataset* corpus_dataset;


    map<int, vector<Mat> > corpus_embedded;
    vector<Mat> corpus_full_embedded;
    vector< vector<Mat>* > corpus_featurized;

    CNNFeaturizer* featurizer;
    SPPEmbedder* embedder;
    int windowWidth, stride;
    float featurizeScale;
    int charWidth;
    set<int> ngrams;

    PHOCer phocer;

    vector<string> lexicon;
    Mat lexicon_phocs;

    Mat normalizedPHOC(string s);
    Mat distFunc(const Mat& a, const Mat& b);

    
    default_random_engine generator;
    bool IDEAL_COMB;

    vector<Mat> npvectors;
    vector<int> npvNs;
    vector<string> npvNgrams;

    float compare_(string text, vector<Mat>* im_featurized);
    vector< SubwordSpottingResult > _subwordSpot(const Mat& exemplarEmbedding, int numChar, float refinePortion, int skip=-1);
    SubwordSpottingResult refine(int windowWidth, float score, int imIdx, int windIdx, const Mat& exemplarEmbedding);

    multimap<float,int>  _wordSpot(const Mat& exemplarEmbedding);

    void writeFloatMat(ofstream& dst, const Mat& m);
    Mat readFloatMat(ifstream& src);

    void refineStep(int imIdx, float* bestScore, int* bestX0, int* bestX1, float scale, const Mat& exemplarEmbedding);
    void refineStepFast(int imIdx, float* bestScore, int* bestX0, int* bestX1, float scale, const Mat& exemplarEmbedding);
    Mat embedFromCorpusFeatures(int imIdx, Rect window);

    float calcAP(const vector<SubwordSpottingResult>& res, string ngram);

    void getEmbedding(int numChar);
    void getCorpusFeaturization();

    void _eval(string word, vector< SubwordSpottingResult >& ret, vector< SubwordSpottingResult >* accumRes, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, float* ap, float* accumAP, multimap<float,int>* truesAccum=NULL, multimap<float,int>* allsAccum=NULL, multimap<float,int>* truesN=NULL, multimap<float,int>* allN=NULL);
    void _eval(string word, multimap<float,int>& ret, multimap<float,int>* accumRes, float* ap, float* accumAP, multimap<float,int>* truesAccum, multimap<float,int>* truesN);
    float getRankChangeRatio(const vector<SubwordSpottingResult>& prevRes, const vector<SubwordSpottingResult>& res, const multimap<float,int>& prevTrues, const multimap<float,int>& trues, const multimap<float,int>& prevAlls, const multimap<float,int>& allsN, float* rankDrop, float* rankRise, float* rankDropFull, float* rankRiseFull, float* mean, float* std, float* meanTop, float* stdTop);

    float getRankChangeRatioFull(const multimap<float,int>& prevRes, const multimap<float,int>& res, const multimap<float,int>& prevTrues, const multimap<float,int>& trues, float* rankDrop, float* rankRise, float* rankDropFull, float* rankRiseFull, float* mean, float* std, float* meanTop, float* stdTop);

    void softMax(Mat colVec,set<int> skip);

    void CL_cluster(vector< list<int> >& clusters, Mat& minSimilarity, int numClusters, const vector<bool>& gt, vector<float>& meanCPurity, vector<float>& medianCPurity, vector<float>& meanIPurity, vector<float>& medianIPurity, vector<float>& maxPurity, vector< vector< list<int> > >& clusterLevels);

};

#endif
