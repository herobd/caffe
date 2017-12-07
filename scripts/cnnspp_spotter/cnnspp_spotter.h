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
#define BRAY_CURTIS  0
#define PRECOMP_QBE 1 //overrides below, does QbE using precomputed features
#define SQUARE_QBE 1 //1=old, 2=force full capture, 0=none

#define CHEAT_WINDOW 0


class CNNSPPSpotter : public Transcriber
{

public:
    CNNSPPSpotter(string featurizerModel, string embedderModel, string netWeights, string ngramWWFile, int gpu=-1, bool normalizeEmbedding=true, float featurizeScale=.25, int stride=4, string saveName="cnnspp_spotter", bool ideal_comb=false);
    ~CNNSPPSpotter();

    void setCorpus_dataset(const Dataset* dataset, bool fullWordEmbed_only=false);

    vector< SubwordSpottingResult > subwordSpot(string ngram, const Mat& exemplar, float refinePortion, int windowWidth=-1);
    vector< SubwordSpottingResult > subwordSpot(const Mat& exemplar, float refinePortion, int windowWidth, int returnWindowWidth=-1);
    vector< SubwordSpottingResult > subwordSpot(const string& exemplar, float refinePortion=DEFAULT_REFINE_PORTION, int windowWidth=-1);
    vector< SubwordSpottingResult > subwordSpot(int exemplarId, int x0, float refinePortion, int windowWidth, int returnWindowWidth=-1);
    vector< SubwordSpottingResult > subwordSpot(string ngram, int exemplarId, int x0, float refinePortion, int windowWidth=-1);
    vector< SubwordSpottingResult > subwordSpotAbout(int exemplarId, float xCenter, float refinePortion, int windowWidth, int returnWindowWidth=-1);
    vector< SubwordSpottingResult > subwordSpotAbout(string ngram, int exemplarId, float xCenter, float refinePortion=DEFAULT_REFINE_PORTION, int windowWidth=-1);
    vector< SubwordSpottingResult > subwordSpot(int exemplarId, int x0, int x1, int focus0, int focus1, float refinePortion, int windowWidth, int returnWindowWidth=-1);
    vector< SubwordSpottingResult > subwordSpot(string ngram, int exemplarId, int x0, int x1, int focus0, int focus1, float refinePortion=DEFAULT_REFINE_PORTION, int windowWidth=-1);

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
    void evalSubwordSpottingWithCharBounds(int N, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, vector<string> queries=vector<string>(), string outDir="");
   // void evalSubwordSpottingCombine(const Dataset* exemplars, const Dataset* data);
    void evalSubwordSpottingRespot(const Dataset* data, vector<string> toSpot, int numSteps, int numRepeat, int repeatSteps, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds);
    void evalFullWordSpottingRespot(const Dataset* data, vector<string> toSpot, int numSteps, int numRepeat, int repeatSteps);
    void evalFullWordSpotting(const Dataset* data, set<string> print=set<string>(), int doTheseFlags=3);

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
    vector<SpottingLoc> massSpot(string ngram, Mat& crossScores)
    {
        vector<string> n = {ngram};
        return massSpot(n,crossScores);
    }

    //For creating new, not thorough, window widths
    void refineWindowSubwordSpottingWithCharBounds(int N, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, vector<string> queries, int charWidth, string outFile);

    //For testing embedding time
    void timeEmbedding();
    void getEmbedding(int windowWidth);
    void getCorpusFeaturization();

    vector< SubwordSpottingResult > suffixSpot(string suffix, float refinePortion);
    void evalSuffixSpotting(const vector<string>& suffixes, const Dataset* data, string saveDir="");

private:
    string saveName;
    string featurizerFile, embedderFile, weightFile;
    const Dataset* corpus_dataset;


    map<int, vector<Mat> > corpus_embedded;
    vector<Mat> corpus_full_embedded;
    vector< vector<Mat>* > corpus_featurized;

    CNNFeaturizer* featurizer;
    SPPEmbedder* embedder;
    int stride;
    map<string,int> ngramWW, ngramRW;
    float featurizeScale;
    set<string> ngrams;

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

    int minSPPSize;

    float compare_(string text, vector<Mat>* im_featurized);
    vector< SubwordSpottingResult > _subwordSpot(const Mat& exemplarEmbedding, int windowWidth, int returnWindowWidth, float refinePortion, int skip=-1);
    SubwordSpottingResult refine(int windowWidth, int returnWindowWidth, float score, int imIdx, int windIdx, const Mat& exemplarEmbedding, bool suffix=false);

    multimap<float,int>  _wordSpot(const Mat& exemplarEmbedding);

    void writeFloatMat(ofstream& dst, const Mat& m);
    Mat readFloatMat(ifstream& src);

    void refineStep(int imIdx, float* bestScore, int* bestX0, int* bestX1, float scale, const Mat& exemplarEmbedding);
    void refineStepFast(int imIdx, float* bestScore, int* bestX0, int* bestX1, float scale, const Mat& exemplarEmbedding);
    void refineSuffixStepFast(int imIdx, float* bestScore, int* bestX0, int* bestX1, float scale, const Mat& exemplarEmbedding);
    Mat embedFromCorpusFeatures(int imIdx, Rect window);

    float calcAP(const vector<SubwordSpottingResult>& res, string ngram);


    void _eval(string word, vector< SubwordSpottingResult >& ret, vector< SubwordSpottingResult >* accumRes, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, float* ap, float* accumAP, multimap<float,int>* truesAccum=NULL, multimap<float,int>* allsAccum=NULL, multimap<float,int>* truesN=NULL, multimap<float,int>* allN=NULL);
    void _eval(string word, multimap<float,int>& ret, multimap<float,int>* accumRes, float* ap, float* accumAP, multimap<float,int>* truesAccum, multimap<float,int>* truesN);
    float getRankChangeRatio(const vector<SubwordSpottingResult>& prevRes, const vector<SubwordSpottingResult>& res, const multimap<float,int>& prevTrues, const multimap<float,int>& trues, const multimap<float,int>& prevAlls, const multimap<float,int>& allsN, float* rankDrop, float* rankRise, float* rankDropFull, float* rankRiseFull, float* mean, float* std, float* meanTop, float* stdTop);

    float getRankChangeRatioFull(const multimap<float,int>& prevRes, const multimap<float,int>& res, const multimap<float,int>& prevTrues, const multimap<float,int>& trues, float* rankDrop, float* rankRise, float* rankDropFull, float* rankRiseFull, float* mean, float* std, float* meanTop, float* stdTop);

    void softMax(Mat colVec,set<int> skip);

    void CL_cluster(vector< list<int> >& clusters, Mat& minSimilarity, int numClusters, const vector<bool>& gt, vector<float>& meanCPurity, vector<float>& medianCPurity, vector<float>& meanIPurity, vector<float>& medianIPurity, vector<float>& maxPurity, vector< vector< list<int> > >& clusterLevels);

#if CHEAT_WINDOW
    int getBestWindowWidth(int i, string searchNgram);
    const vector< vector<int> >* corpusXLetterStartBounds, *corpusXLetterEndBounds;
    string searchNgram;
#endif
    int stichWW(const map<string,int>& ww, string word);
    float calcSuffixAP(const vector<SubwordSpottingResult>& res, string suffix, int* trueCount=NULL, int* wholeCount=NULL);
};

#endif
