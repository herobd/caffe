#include "cnnspp_spotter.h"
#include "cnnspp_spotter_eval.cpp"

CNNSPPSpotter::CNNSPPSpotter(string featurizerModel, string embedderModel, string netWeights, set<int> ngrams, bool normalizeEmbedding, float featurizeScale, int charWidth, int stride, string saveName, bool ideal_comb) : stride(stride), featurizeScale(featurizeScale), ngrams(ngrams), charWidth(charWidth), IDEAL_COMB(ideal_comb)
{
    assert(charWidth>0);
    //windowWidth = 2*charWidth;
    this->saveName = saveName;
    featurizer = new CNNFeaturizer(featurizerModel,netWeights);
    embedder = new SPPEmbedder(embedderModel,netWeights,normalizeEmbedding);
    //cout<<"Window width:"<<windowWidth<<endl;
    cout<<"Char width: "<<charWidth<<endl;

    windowWidths[1] = charWidth;
    windowWidths[2] = 2*charWidth;
    windowWidths[3] = 3*charWidth;


    if (IDEAL_COMB)
        cout<<"CNNSPPSpotter is using ideal combination scoring."<<endl;

#if PRECOMP_QBE
    cout<<"CNNSPPSpotter is using precomputed features for QbE testing"<<endl;
#endif

    corpus_dataset=NULL;
    //corpus_featurized=NULL;
    int lastSlash = featurizerModel.find_last_of('/');
    if (lastSlash==string::npos)
        lastSlash=-1;
    this->featurizerFile = featurizerModel.substr(lastSlash+1);

    lastSlash = embedderModel.find_last_of('/');
    if (lastSlash==string::npos)
        lastSlash=-1;
    this->embedderFile = embedderModel.substr(lastSlash+1);

    lastSlash = netWeights.find_last_of('/');
    if (lastSlash==string::npos)
        lastSlash=-1;
    this->weightFile = netWeights.substr(lastSlash+1);
}

CNNSPPSpotter::~CNNSPPSpotter()
{
    if (corpus_featurized.size()>1)
    {
        for (auto i : corpus_featurized)
            delete i;
    }
    delete featurizer;
    delete embedder;
}

float CNNSPPSpotter::compare(string text, const Mat& image)
{
    vector<Mat>* im_featurized = featurizer->featurize(image);
    return compare_(text,im_featurized);
}

float CNNSPPSpotter::compare(string text, int wordIndex)
{
    Mat textEmbedding = normalizedPHOC(text);
    Mat imEmbedding = corpus_full_embedded.at(wordIndex);
    return distFunc(imEmbedding,textEmbedding).at<float>(0,0);

    //vector<Mat>* im_featurized = corpus_featurized.at(wordIndex);
    //return compare_(text,im_featurized);
}
float CNNSPPSpotter::compare(int wordIndex, int wordIndex2)
{
    Mat phoc = corpus_full_embedded.at(wordIndex);
    Mat phoc2 = corpus_full_embedded.at(wordIndex2);
    return distFunc(phoc,phoc2).at<float>(0,0);
}

float CNNSPPSpotter::compare_(string text, vector<Mat>* im_featurized)
{
    Mat imEmbedding = embedder->embed(im_featurized);
    delete im_featurized;

    Mat textEmbedding = normalizedPHOC(text);
    
    return distFunc(imEmbedding, textEmbedding).at<float>(0,0);
}

multimap<float,int> CNNSPPSpotter::wordSpot(const Mat& exemplar)
{
    vector<Mat>* ex_featurized = featurizer->featurize(exemplar);
    Mat exemplarEmbedding = embedder->embed(ex_featurized);
    delete ex_featurized;
    
    return _wordSpot(exemplarEmbedding); 
}
multimap<float,int> CNNSPPSpotter::wordSpot(const string& exemplar)
{
    //vector<Mat>* ex_featurized = featurizer->featurize(exemplar);
    //Mat exemplarEmbedding = embedder->embed(ex_featurized);
    //delete ex_featurized;
    Mat exemplarEmbedding = normalizedPHOC(exemplar);
    
    return _wordSpot(exemplarEmbedding);    
 
}

multimap<float,int> CNNSPPSpotter::wordSpot(int exemplarIndex)
{
    return _wordSpot(corpus_full_embedded.at(exemplarIndex));    
}

Mat CNNSPPSpotter::normalizedPHOC(string s)
{
    vector<float> phoc = phocer.makePHOC(s);
    float ss=0;
    for (float v : phoc)
        ss+=v*v;
    ss=sqrt(ss);
    Mat ret(phoc.size(),1,CV_32F);
    //normalize(ret,ret);
    //ret/=ss;
    if (ss==0)
        ss=1;
    for (int i=0; i<ret.rows; i++)
#ifdef NO_NORM_PHOC
        ret.at<float>(i,0)=phoc[i];
#else
        ret.at<float>(i,0)=phoc[i]/ss;
#endif
    return ret;
}

//With a GPU, we could efficiently batch multiple exemplars together. Currently the refineStepFast function does this, but it uses a small batch
//Cannot be const because of network objects
vector< SubwordSpottingResult > CNNSPPSpotter::subwordSpot(int numChar, const Mat& exemplar, float refinePortion)
{
    vector<Mat>* ex_featurized = featurizer->featurize(exemplar);
    Mat exemplarEmbedding = embedder->embed(ex_featurized);
    delete ex_featurized;
    
    return _subwordSpot(exemplarEmbedding,numChar,refinePortion); 
}


vector< SubwordSpottingResult > CNNSPPSpotter::subwordSpot(const string& exemplar, float refinePortion)
{
    //vector<Mat>* ex_featurized = featurizer->featurize(exemplar);
    //Mat exemplarEmbedding = embedder->embed(ex_featurized);
    //delete ex_featurized;
    Mat exemplarEmbedding = normalizedPHOC(exemplar);
    
    return _subwordSpot(exemplarEmbedding,exemplar.length(),refinePortion);    
 
}

vector< SubwordSpottingResult > CNNSPPSpotter::subwordSpot(int numChar, int exemplarId, int x0, float refinePortion)
{
    //assert(abs(x1-x0 -min(windowWidth,corpus_dataset->image(exemplarId).cols))<stride);
    int windIdx = x0/stride;
    if (corpus_embedded.at(numChar).at(exemplarId).cols<=windIdx)
        windIdx = corpus_embedded.at(numChar).at(exemplarId).cols-1;
    return _subwordSpot(corpus_embedded.at(numChar).at(exemplarId).col(windIdx),numChar,refinePortion,exemplarId);
}
vector< SubwordSpottingResult > CNNSPPSpotter::subwordSpotAbout(int numChar, int exemplarId, float xCenter, float refinePortion)
{
    //assert(abs(x1-x0 -min(windowWidth,corpus_dataset->image(exemplarId).cols))<stride);
    float x0 = std::max(0.0f,xCenter-charWidth*(numChar/2.0f));
    int windIdx = round(x0/stride);
    if (corpus_embedded.at(numChar).at(exemplarId).cols<=windIdx)
        windIdx = corpus_embedded.at(numChar).at(exemplarId).cols-1;
    assert(windIdx>=0);
    return _subwordSpot(corpus_embedded.at(numChar).at(exemplarId).col(windIdx),numChar,refinePortion,exemplarId);
}

vector< SubwordSpottingResult > CNNSPPSpotter::subwordSpot(int numChar, int exemplarId, int x0, int x1, int focus0, int focus1, float refinePortion)
{
    int x0f=featurizeScale*x0;
    int x1f=featurizeScale*x1;
    int focus0f=featurizeScale*focus0;
    int focus1f=featurizeScale*focus1;
    getCorpusFeaturization();
    vector<Mat>* features= corpus_featurized.at(exemplarId);
    vector<Mat> ex_featurized(features->size());
    for (int i=0; i<features->size(); i++)
    {
        ex_featurized.at(i) = features->at(i).colRange(x0f,x1f+1).clone();
        //feather
        int left = focus0f-x0f;
        for (int c=0; c<left; c++)
        {
            ex_featurized.at(i).col(c) *= c/(0.0+left);
        }
        int right = x1f-focus1f;
        int colsC = x1f-x0f;
        for (int c=0; c<right; c++)
        {
            ex_featurized.at(i).col(colsC-c) *= c/(0.0+right);
        }
    }
    
    Mat exemplarEmbedding = embedder->embed(&ex_featurized);
    return _subwordSpot(exemplarEmbedding,numChar,refinePortion,exemplarId);
}



vector< SubwordSpottingResult > CNNSPPSpotter::_subwordSpot(const Mat& exemplarEmbedding, int numChar, float refinePortion, int skip)
{

    //int windowWidth=numChar*charWidth;
    multimap<float,pair<int,int> > scores;

    #pragma omp parallel for
    for (int i=0; i<corpus_dataset->size(); i++)
    {
        if (i==skip)
            continue;
        Mat s_batch = distFunc(exemplarEmbedding, corpus_embedded.at(numChar).at(i));

        assert(s_batch.rows==1);
        float topScoreInd=-1;
        float topScore=numeric_limits<float>::max();
        float top2ScoreInd=-1;
        float top2Score=numeric_limits<float>::max();
        for (int c=0; c<s_batch.cols; c++) {
            float s = s_batch.at<float>(0,c);
            if (s<topScore)
            {
                topScore=s;
                topScoreInd=c;
            }
        }
        int diff = ((windowWidths[numChar]/2.0) *.8)/stride;
        for (int c=0; c<s_batch.cols; c++) {
            float s = s_batch.at<float>(0,c);
            if (s<top2Score && abs(c-topScoreInd)>diff)
            {
                top2Score=s;
                top2ScoreInd=c;
            }
        }

        #pragma omp critical (_subword_spot)
        {
        assert(topScoreInd!=-1);
        scores.emplace(topScore, make_pair(i,topScoreInd));
        if (top2ScoreInd!=-1)
            scores.emplace(top2Score, make_pair(i,top2ScoreInd));
        }
    }

    //Now, we will refine only the top X% of the results
    auto iter = scores.begin();
    int finalSize = scores.size()*refinePortion;
    vector< SubwordSpottingResult > finalScores(finalSize);
    for (int i=0; i<finalSize; i++, iter++)
    {
        finalScores.at(i) = refine(windowWidths[numChar], iter->first,iter->second.first,iter->second.second,exemplarEmbedding);
    }

    return finalScores;
}


vector< SubwordSpottingResult > CNNSPPSpotter::subwordSpot_eval(const Mat& exemplar, string word, float refinePortion, vector< SubwordSpottingResult >* accumRes, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, float* ap, float* accumAP, mutex* resLock, float help)
{
    vector< SubwordSpottingResult > ret = subwordSpot(word.length(),exemplar,refinePortion);
    if (help>=0)
        helpAP(ret,word,corpusXLetterStartBounds,corpusXLetterEndBounds,help);
    resLock->lock();
    _eval(word,ret,accumRes,corpusXLetterStartBounds,corpusXLetterEndBounds,ap,accumAP);
    resLock->unlock();

    return ret;
 
}
vector< SubwordSpottingResult > CNNSPPSpotter::subwordSpot_eval(int exemplarId, int x0, string word, float refinePortion, vector< SubwordSpottingResult >* accumRes, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, float* ap, float* accumAP, mutex* resLock, float help)
{
    vector< SubwordSpottingResult > ret = subwordSpot(word.length(),exemplarId,x0,refinePortion);
    if (help>=0)
        helpAP(ret,word,corpusXLetterStartBounds,corpusXLetterEndBounds,help);
    resLock->lock();
    _eval(word,ret,accumRes,corpusXLetterStartBounds,corpusXLetterEndBounds,ap,accumAP);
    resLock->unlock();

    return ret;
 
}
vector< SubwordSpottingResult > CNNSPPSpotter::subwordSpot_eval(const string& exemplar, float refinePortion, vector< SubwordSpottingResult >* accumRes, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, float* ap, float* accumAP, mutex* resLock, float help)
{
    vector< SubwordSpottingResult > ret = subwordSpot(exemplar,refinePortion);
    if (help>=0)
        helpAP(ret,exemplar,corpusXLetterStartBounds,corpusXLetterEndBounds,help);
    resLock->lock();
    _eval(exemplar,ret,accumRes,corpusXLetterStartBounds,corpusXLetterEndBounds,ap,accumAP);
    resLock->unlock();

    return ret;
 
}

void CNNSPPSpotter::_eval(string word, vector< SubwordSpottingResult >& ret, vector< SubwordSpottingResult >* accumRes, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, float* ap, float* accumAP, multimap<float,int>* truesAccum, multimap<float,int>* allsAccum, multimap<float,int>* truesN, multimap<float,int>* allsN)
{
#ifdef TEST_MODE
    cout<<"Start CNNSPPSpotter::_eval"<<endl;
#endif
    *ap = evalSubwordSpotting_singleScore(word, ret, corpusXLetterStartBounds, corpusXLetterEndBounds,-1, truesN, allsN);

    //vector< SubwordSpottingResult > accumRes2(*accumRes);
    //vector< SubwordSpottingResult > accumRes3(*accumRes);
    //vector< SubwordSpottingResult > accumRes4(*accumRes);
    vector< SubwordSpottingResult > newAccum;
    for (auto r : ret)
    {
        bool matchFound=false;
        for (int i=0; i<accumRes->size(); i++)
        {
            if (accumRes->at(i).imIdx == r.imIdx)
            {
                double ratio = ( min(accumRes->at(i).endX,r.endX) - max(accumRes->at(i).startX,r.startX) ) /
                               ( max(accumRes->at(i).endX,r.endX) - min(accumRes->at(i).startX,r.startX) +0.0);
                if (ratio > LIVE_SCORE_OVERLAP_THRESH)
                {
                    //double ratioOff = 1.0 - (ratio-LIVE_SCORE_OVERLAP_THRESH)/(1.0-LIVE_SCORE_OVERLAP_THRESH);
                    float worseScore = max(r.score,accumRes->at(i).score);
                    float bestScore = min(r.score,accumRes->at(i).score);

                    //float combScore = (1.0f-ratioOff)*worseScore + (ratioOff)*bestScore;
                    //float combScore = (worseScore + bestScore)/2.0f;
                    float combScore = min(worseScore, bestScore);//take best
                    if (IDEAL_COMB)
                        if (r.gt!=-10 || accumRes->at(i).gt!=-10)
                        {
                           if (r.gt!=1 && accumRes->at(i).gt!=1)
                               combScore = worseScore;
                        }
                    if (r.score < accumRes->at(i).score)
                        accumRes->at(i)=r;
                    accumRes->at(i).score = combScore;
                    matchFound=true;
                    ///////////////////////
                    /*
                    if (r.score < accumRes2.at(i).score)
                        accumRes2.at(i)=r;
                    accumRes2.at(i).score = worseScore;

                    combScore = combScore*0.5 + worseScore*0.5;//skew towards worse
                    if (r.score < accumRes3.at(i).score)
                        accumRes3.at(i)=r;
                    accumRes3.at(i).score = combScore;

                    combScore = (worseScore + bestScore)/2.0f;
                    if (r.score < accumRes4.at(i).score)
                        accumRes4.at(i)=r;
                    accumRes4.at(i).score = combScore;
                    */
                    ////////////////////////
                    break;
                }
            }

        }
        if (!matchFound)
        {
            newAccum.push_back(r);

            //accumRes2.push_back(r);
            //accumRes3.push_back(r);
            //accumRes4.push_back(r);
        }

    }
    accumRes->insert(accumRes->end(),newAccum.begin(),newAccum.end());
    *accumAP = evalSubwordSpotting_singleScore(word, *accumRes, corpusXLetterStartBounds, corpusXLetterEndBounds,-1, truesAccum, allsAccum);
    /*float aap2 = evalSubwordSpotting_singleScore(word, accumRes2, corpusXLetterStartBounds, corpusXLetterEndBounds);
    float aap3 = evalSubwordSpotting_singleScore(word, accumRes3, corpusXLetterStartBounds, corpusXLetterEndBounds);
    float aap4 = evalSubwordSpotting_singleScore(word, accumRes4, corpusXLetterStartBounds, corpusXLetterEndBounds);
    cerr <<"accumAP for ["<<word<<"]; blend: "<<*accumAP<<", worse: "<<aap2<<", bias: "<<aap3<<", avg: "<<aap4<<endl;
    if (aap2>*accumAP)
    {
        *accumAP=aap2;
        *accumRes=accumRes2;
    }
    if (aap3>*accumAP)
    {
        *accumAP=aap3;
        *accumRes=accumRes3;
    }
    if (aap4>*accumAP)
    {
        *accumAP=aap4;
        *accumRes=accumRes4;
    }*/
#ifdef TEST_MODE
    cout<<"End CNNSPPSpotter::_eval"<<endl;
#endif
}

SubwordSpottingResult CNNSPPSpotter::refine(int windowWidth, float score, int imIdx, int windIdx, const Mat& exemplarEmbedding)
{
    //before 0.503722 refine, s:0.630617
    float bestScore=score;
    int newX0 = windIdx*stride;
    int newX1 = std::min(newX0+windowWidth-1, corpus_dataset->image(imIdx).cols-1);
    ////

    //float scale = 1.0;
    int bestX0=newX0;
    int bestX1=newX1;

    //refineStep(imIdx, &bestScore, &bestX0, &bestX1, 2.0, exemplarEmbedding);//wita 1.0:h 0.490994
    //refineStep(imIdx, &bestScore, &bestX0, &bestX1, 1.0, exemplarEmbedding);//0.504115
    
    //refineStepFast(imIdx, &bestScore, &bestX0, &bestX1, 5.0, exemplarEmbedding);//1.0i: 0.509349, 5.0i:0.503195,   5.0s:0.633528

    assert(bestX0>=0 && bestX1>=0);
    assert(bestX0<corpus_dataset->image(imIdx).cols && bestX0<corpus_dataset->image(imIdx).cols);
    assert(bestX1>=1 && bestX1>=1);
    assert(bestX1<corpus_dataset->image(imIdx).cols && bestX1<corpus_dataset->image(imIdx).cols);
    return SubwordSpottingResult(imIdx,bestScore,bestX0,bestX1);
}

void CNNSPPSpotter::refineStep(int imIdx, float* bestScore, int* bestX0, int* bestX1, float scale, const Mat& exemplarEmbedding)
{
    getCorpusFeaturization();
    int newX0out = max(0,(int)((*bestX0)-scale*stride));
    int newX0in = ((*bestX0)+scale*stride);
    int newX1in = ((*bestX1)-scale*stride);
    int newX1out = min((int)(((*bestX1)+scale*stride)), corpus_dataset->image(imIdx).cols-1);

    int newX0outF =newX0out*featurizeScale;
    int newX0inF = newX0in *featurizeScale;
    int newX1inF = newX1in *featurizeScale;
    int newX1outF =newX1out*featurizeScale;
    int bestX0F = (*bestX0)*featurizeScale;
    int bestX1F = (*bestX1)*featurizeScale;
    //This could be made even faster by changing the spp_embedder to accept a batch of features to embed.
    //Create a batch from all the windows you want to try and embed all at once.
    Rect windowX0out(newX0outF,0,(bestX1F)-newX0outF+1,corpus_featurized.at(imIdx)->front().rows);
    Mat wEmbedding = embedFromCorpusFeatures(imIdx,windowX0out);
    Rect windowX0in(newX0inF,0,(bestX1F)-newX0inF+1,corpus_featurized.at(imIdx)->front().rows);
    Mat ee = embedFromCorpusFeatures(imIdx,windowX0in);
    hconcat(wEmbedding,ee);
    Rect windowX1in((bestX0F),0,newX1inF-(bestX0F)+1,corpus_featurized.at(imIdx)->front().rows);
    ee = embedFromCorpusFeatures(imIdx,windowX1in);
    hconcat(wEmbedding,ee);
    Rect windowX1out((bestX0F),0,newX1outF-(bestX0F)+1,corpus_featurized.at(imIdx)->front().rows);
    ee = embedFromCorpusFeatures(imIdx,windowX1out);
    hconcat(wEmbedding,ee);
    Mat wScores = -1*(exemplarEmbedding.t() * wEmbedding);
    int oldX0=*bestX0;
    int oldX1=*bestX1;
    float best0=9999;
    float best1=9999;
    //if (wScores.at<float>(0,0)< *bestScore)
    {
        best0= wScores.at<float>(0,0);
        *bestX0=newX0out;
    }
    if (wScores.at<float>(0,1)< best0)
    {
        best0 = wScores.at<float>(0,1);
        *bestX0=newX0in;
    }
    //if (wScores.at<float>(0,2)< *bestScore)
    {
        best1 = wScores.at<float>(0,2);
        *bestX1=newX1in;
    }
    if (wScores.at<float>(0,3)< best1)
    {
        best1 = wScores.at<float>(0,3);
        *bestX1=newX1out;
    }

    if (best0 < *bestScore)
        *bestScore = best0;
    if (best1 < *bestScore)
        *bestScore = best1;

    Rect windowBests((*bestX0)*featurizeScale,0,((*bestX1)-(*bestX0))*featurizeScale,corpus_featurized.at(imIdx)->front().rows);
    wEmbedding = embedFromCorpusFeatures(imIdx,windowBests);
    Mat bestsScore = distFunc(exemplarEmbedding, wEmbedding);
    if (bestsScore.at<float>(0,0) <= *bestScore)
    {
        *bestScore = bestsScore.at<float>(0,0);
    }
    else if (best0==*bestScore)
    {
        *bestX1=oldX1;
    }
    else if (best1==*bestScore)
    {
        *bestX0=oldX0;
    }
    else
    {
        *bestX1=oldX1;
        *bestX0=oldX0;
    }
}

void CNNSPPSpotter::refineStepFast(int imIdx, float* bestScore, int* bestX0, int* bestX1, float scale, const Mat& exemplarEmbedding)
{
    getCorpusFeaturization();
    int batchSize=6; 
    int newX0out = max(0,(int)((*bestX0)-scale*stride));
    int newX0in = ((*bestX0)+scale*stride);
    int newX1in = ((*bestX1)-scale*stride);
    int newX1out = min((int)(((*bestX1)+scale*stride)), corpus_dataset->image(imIdx).cols-1);

    int newX0outF =newX0out*featurizeScale;
    int newX0inF = newX0in *featurizeScale;
    int newX1inF = newX1in *featurizeScale;
    int newX1outF =newX1out*featurizeScale;
    int bestX0F = (*bestX0)*featurizeScale;
    int bestX1F = (*bestX1)*featurizeScale;

    //Rect extraX1(
    Rect windows[batchSize];
    Rect windowsTo[batchSize];
    windows[0] = Rect (newX0outF,0,(bestX1F)-newX0outF+1,corpus_featurized.at(imIdx)->front().rows);//windowX0out;
    windowsTo[0]=Rect(0,0,windows[0].width,windows[0].height);
    windows[1] = Rect (newX0inF,0,(bestX1F)-newX0inF+1,corpus_featurized.at(imIdx)->front().rows);//windowX0in;
    windowsTo[1]=Rect(newX0inF-newX0outF,0,windows[1].width,windows[1].height);
    windows[2] = Rect ((bestX0F),0,newX1inF-(bestX0F)+1,corpus_featurized.at(imIdx)->front().rows);//windowX1in
    windowsTo[2]=Rect(bestX0F-newX0outF,0,windows[2].width,windows[2].height);
    windows[3] = Rect ((bestX0F),0,newX1outF-(bestX0F)+1,corpus_featurized.at(imIdx)->front().rows);//windowX1out
    windowsTo[3]=Rect(bestX0F-newX0outF,0,windows[3].width,windows[3].height);
    windows[4] = Rect (newX0outF,0,newX1outF-newX0outF+1,corpus_featurized.at(imIdx)->front().rows);//both out
    windowsTo[4]=Rect(0,0,windows[4].width,windows[4].height);
    windows[5] = Rect (newX0inF,0,max(newX1inF-newX0inF+1,1),corpus_featurized.at(imIdx)->front().rows);//both in
    windowsTo[5]=Rect(bestX0F-newX0outF,0,windows[5].width,windows[5].height);

    Rect window(newX0outF,0,newX1outF-newX0outF+1,corpus_featurized.at(imIdx)->front().rows);
    vector< vector<Mat> > batch(batchSize);
    for (int bi=0; bi<batchSize; bi++)
    {
        batch.at(bi).resize(corpus_featurized.at(imIdx)->size());
        for (int c=0; c<corpus_featurized.at(imIdx)->size(); c++)
        {
            batch.at(bi).at(c) = Mat::zeros(corpus_featurized.at(imIdx)->front().rows,newX1outF-newX0outF+1,CV_32F);
            corpus_featurized.at(imIdx)->at(c)(windows[bi]).copyTo(batch[bi][c](windowsTo[bi]));


        }
    }
    Mat embeddings = embedder->embed(batch);
    Mat wScores = distFunc(exemplarEmbedding, embeddings);

    int oldX0=*bestX0;
    int oldX1=*bestX1;

    if (wScores.at<float>(0,0)< *bestScore)
    {
        *bestScore= wScores.at<float>(0,0);
        *bestX0=newX0out;
        *bestX1=oldX1;
    }
    if (wScores.at<float>(0,1)< *bestScore)
    {
        *bestScore = wScores.at<float>(0,1);
        *bestX0=newX0in;
        *bestX1=oldX1;
    }
    if (wScores.at<float>(0,2)< *bestScore)
    {
        *bestScore = wScores.at<float>(0,2);
        *bestX0=oldX0;
        *bestX1=newX1in;
    }
    if (wScores.at<float>(0,3)< *bestScore)
    {
        *bestScore = wScores.at<float>(0,3);
        *bestX0=oldX0;
        *bestX1=newX1out;
    }
    if (wScores.at<float>(0,4)< *bestScore)
    {
        *bestScore = wScores.at<float>(0,4);
        *bestX0=newX0out;
        *bestX1=newX1out;
    }
    if (wScores.at<float>(0,5)< *bestScore)
    {
        *bestScore = wScores.at<float>(0,5);
        *bestX0=newX0in;
        *bestX1=newX1in;
    }

}



multimap<float,int> CNNSPPSpotter::_wordSpot(const Mat& exemplarEmbedding)
{
    multimap<float,int> scores;

    //#pragma omp parallel for
    for (int i=0; i<corpus_dataset->size(); i++)
    {
        Mat cal = distFunc(exemplarEmbedding,corpus_full_embedded.at(i));//exemplarEmbedding.t() * corpus_full_embedded.at(i);

        float s = cal.at<float>(0,0);
        //#pragma omp critical (_wordSpot)
        scores.emplace(s,i);
    }

    return scores;
}

void CNNSPPSpotter::_eval(string word, multimap<float,int>& ret, multimap<float,int>* accumRes, float* ap, float* accumAP, multimap<float,int>* truesAccum, multimap<float,int>* truesN)
{
    *ap = evalWordSpotting_singleScore(word, ret, -1, truesN);

    multimap<float,int> newAccum;
    map<int,float> tempAccum;
    for (auto ar : *accumRes)
        tempAccum[ar.second]=ar.first;
    for (auto r : ret)
    {
        bool matchFound=false;
        if (tempAccum.find(r.second)!=tempAccum.end())
        {
            float oldScore = tempAccum.at(r.second);
            //double ratioOff = 1.0 - (ratio-LIVE_SCORE_OVERLAP_THRESH)/(1.0-LIVE_SCORE_OVERLAP_THRESH);
            float worseScore = max(r.first,oldScore);
            float bestScore = min(r.first,oldScore);
            //float combScore = (1.0f-ratioOff)*worseScore + (ratioOff)*bestScore;
            //float combScore = (worseScore + bestScore)/2.0f;
            float combScore = min(worseScore, bestScore);//take best
            //if (r.first < ar.first)
            //    ar=r;
            tempAccum.at(r.second) = combScore;
            matchFound=true;
            break;
        }
        if (!matchFound)
        {
            newAccum.insert(r);
        }

    }
    *accumRes = newAccum;
    for (auto fr : tempAccum)
        accumRes->emplace(fr.second,fr.first);
    *accumAP = evalWordSpotting_singleScore(word, *accumRes, -1, truesAccum);
}

Mat CNNSPPSpotter::distFunc(const Mat& A, const Mat& B)
{
#if BRAY_CURTIS
    Mat a = max(A,0);
    Mat b = max(B,0);
    /*for (int r=0; r<a.rows; r++)
        if (a.at<float>(r,0) < 0.5)
            a.at<float>(r,0)=0;
        else
            a.at<float>(r,0) = (a.at<float>(r,0)-0.5)*2;
    for (int c=0; c<b.cols; c++)
    for (int r=0; r<b.rows; r++)
        if (b.at<float>(r,c) < 0.5)
            b.at<float>(r,c)=0;
        else
            b.at<float>(r,c) = (b.at<float>(r,c)-0.5)*2;*/
    assert(a.cols==1);
    assert(b.rows==a.rows);
    Mat top(b.size(),b.type());
    for (int c=0; c<b.cols; c++)
        absdiff(a,b.col(c),top.col(c));
    reduce(top, top, 0, CV_REDUCE_SUM);
    Mat bot;
    Scalar aSum = sum(a);
    reduce(b,bot,0, CV_REDUCE_SUM);
    bot += aSum;
    Mat res;
    divide(top,bot,res);
    //cout<<res.at<float>(0,0)<<endl;
    assert(res.at<float>(0,0)>=0 && res.at<float>(0,0)<=1.0001);
    return res;
#else
    return -1*A.t()*B;
#endif
}



Mat CNNSPPSpotter::embedFromCorpusFeatures(int imIdx, Rect window)
{
    getCorpusFeaturization();
    vector<Mat> windowed_features(corpus_featurized.at(imIdx)->size());
    for (int c=0; c<corpus_featurized.at(imIdx)->size(); c++)
    {
        windowed_features.at(c) = corpus_featurized.at(imIdx)->at(c)(window);
    }
    return embedder->embed(&windowed_features);
}

void CNNSPPSpotter::getCorpusFeaturization()
{
    if (corpus_featurized.size()>0)
        return;

    string nameFeaturization = saveName+"_corpus_cnnFeatures_"+featurizerFile+"_"+weightFile+"_"+corpus_dataset->getName()+".dat";
    ifstream in(nameFeaturization);
    if (in)
    {
        cout<<"Reading in features: "<<nameFeaturization<<endl;
        int numWordsRead;
        in >> numWordsRead;
        assert(numWordsRead == corpus_dataset->size());
        corpus_featurized.resize(numWordsRead);
        for (int i=0; i<numWordsRead; i++)
        {
            int numChannels;
            in >> numChannels;
            if (i>0)
                assert(corpus_featurized.at(0)->size() == numChannels);
            corpus_featurized.at(i) = new vector<Mat>(numChannels);
            for (int j=0; j<numChannels; j++)
            {
                corpus_featurized.at(i)->at(j) = readFloatMat(in);
            }
        }
        in.close();
        cout <<"done"<<endl;

    }
    else
    {
        assert(stride>0);
        cout<<"Featurizing "<<corpus_dataset->getName()<<endl;
        cout<<"writing to: "<<nameFeaturization<<endl;

        corpus_featurized.resize(corpus_dataset->size());
        for (int i=0; i<corpus_dataset->size(); i++)
        {
            Mat im = corpus_dataset->image(i);
            
            corpus_featurized.at(i) = featurizer->featurize(im);

            assert(stride>0);
        }
        ofstream out(nameFeaturization);
        cout<<"writing..."<<endl;
        out << corpus_dataset->size() << " ";
        for (int i=0; i<corpus_dataset->size(); i++)
        {
            out << corpus_featurized.at(i)->size() << " ";
            for (int j=0; j<corpus_featurized.at(i)->size(); j++)
            {
                writeFloatMat(out,corpus_featurized.at(i)->at(j));
            }

        }
        out.close();
    }
}


void CNNSPPSpotter::getEmbedding(int numChar, int windowWidth)
{
    assert(numChar!=0);
    const Dataset* dataset = corpus_dataset;
    if (windowWidth<=0)
        windowWidth = numChar*charWidth;
    assert(windowWidth>0);
    ifstream in;
    string nameEmbedding = saveName+"_corpus_sppEmbedding_"+embedderFile+"_"+weightFile+"_"+dataset->getName()+"_w"+to_string(windowWidth)+"_s"+to_string(stride)+".dat";
    in.open(nameEmbedding);
    if (in)
    {
        cout<<"Reading in embedding: "<<nameEmbedding<<endl;
        int numWordsRead;
        in >> numWordsRead;
        assert(numWordsRead == dataset->size());
        corpus_embedded[numChar].resize(numWordsRead);
        for (int i=0; i<numWordsRead; i++)
        {
            corpus_embedded[numChar].at(i) = readFloatMat(in);
        }
        in.close();
        cout <<"done"<<endl;

    }
    else
    {
        getCorpusFeaturization();
        cout<<"Creating embedding for "<<corpus_dataset->getName()<<" (w:"<<windowWidth<<" s:"<<stride<<")"<<endl;
        cout<<"will write to: "<<nameEmbedding<<endl;
        corpus_embedded[numChar].resize(corpus_dataset->size());
        for (int i=0; i<corpus_dataset->size(); i++)
        {
            if (corpus_featurized.at(i)->front().cols != ceil(corpus_dataset->image(i).cols*featurizeScale))
            {
                cout<<"SIZE FAIL, featurized: "<<corpus_featurized.at(i)->front().cols<<", image("<<corpus_dataset->image(i).cols<<") scaled: "<<ceil(corpus_dataset->image(i).cols*featurizeScale)<<endl;
            }
            assert(corpus_featurized.at(i)->front().cols == ceil(corpus_dataset->image(i).cols*featurizeScale));
            vector<Mat> windowed_features(corpus_featurized.at(i)->size());
            
            for (int ws=0; ws+windowWidth < corpus_dataset->image(i).cols; ws+=stride)
            {
                Rect window(ws*featurizeScale,0,windowWidth*featurizeScale,corpus_featurized.at(i)->front().rows);
                for (int c=0; c<corpus_featurized.at(i)->size(); c++)
                {
                    windowed_features.at(c) = corpus_featurized.at(i)->at(c)(window);
                }
                Mat a = embedder->embed(&windowed_features);
                assert(a.rows>0);
                if (corpus_embedded[numChar].at(i).rows==0)
                    corpus_embedded[numChar].at(i)=a;
                else
                    hconcat(corpus_embedded[numChar].at(i), a, corpus_embedded[numChar].at(i));
            }
            if (corpus_embedded[numChar].at(i).rows==0 && corpus_featurized.at(i)->front().cols<=windowWidth)
            {
                for (int c=0; c<corpus_featurized.at(i)->size(); c++)
                {
                    windowed_features.at(c) = corpus_featurized.at(i)->at(c);
                }

                corpus_embedded[numChar].at(i)=embedder->embed(&windowed_features);
            }
            else if (corpus_embedded[numChar].at(i).rows==0)
            {
                cout<<"["<<i<<"] window: "<<windowWidth<<", image width: "<< corpus_featurized.at(i)->front().cols<<endl;
                assert(corpus_embedded[numChar].at(i).rows>0);
            }

        }
        ofstream out(nameEmbedding);
        out << corpus_dataset->size() << " ";
        for (int i=0; i<corpus_dataset->size(); i++)
        {
            assert(corpus_embedded[numChar].at(i).rows>0);
            writeFloatMat(out,corpus_embedded[numChar].at(i));
        }
        out.close();
    }
}


void CNNSPPSpotter::setCorpus_dataset(const Dataset* dataset, bool fullWordEmbed_only)
{
    corpus_dataset = dataset;
    //loadCorpusEmbedding(NET_IN_SIZE,NET_PIX_STRIDE);
    string nameFullWordEmbedding=saveName+"_corpus_sppEmbedding_"+embedderFile+"_"+weightFile+"_"+dataset->getName()+"_full.dat";
    
    ifstream in;

    if (!fullWordEmbed_only)
    {
        for (int numChar : ngrams)
            if (numChar>0)
                getEmbedding(numChar);
    }

    in.open(nameFullWordEmbedding);
    if (in)
    {
        cout<<"Reading in full word embedding: "<<nameFullWordEmbedding<<endl;
        int numWordsRead;
        in >> numWordsRead;
        assert(numWordsRead == dataset->size());
        corpus_full_embedded.resize(numWordsRead);
        for (int i=0; i<numWordsRead; i++)
        {
            corpus_full_embedded.at(i) = readFloatMat(in);
        }
        in.close();
        cout <<"done"<<endl;

    }
    else
    {
        getCorpusFeaturization();
        cout<<"Creating full word embedding for "<<corpus_dataset->getName()<<endl;
        cout<<"writing to: "<<nameFullWordEmbedding<<endl;
        corpus_full_embedded.resize(corpus_dataset->size());
        for (int i=0; i<corpus_dataset->size(); i++)
        {
            corpus_full_embedded.at(i)=embedder->embed(corpus_featurized.at(i));

        }
        ofstream out(nameFullWordEmbedding);
        out << corpus_dataset->size() << " ";
        for (int i=0; i<corpus_dataset->size(); i++)
        {
            assert(corpus_full_embedded.at(i).rows>0);
            writeFloatMat(out,corpus_full_embedded.at(i));
        }
        out.close();
    }
    
}

void CNNSPPSpotter::writeFloatMat(ofstream& dst, const Mat& m)
{
    assert(m.type()==CV_32F);
    dst << "[ "<< m.rows<<" "<<m.cols<<" ] ";
    dst << setprecision(9);
    for (int r=0; r<m.rows; r++)
        for (int c=0; c<m.cols; c++)
        {
            assert(m.at<float>(r,c)==m.at<float>(r,c));
            dst << m.at<float>(r,c) << " ";
        }
}

Mat CNNSPPSpotter::readFloatMat(ifstream& src)
{
    int rows, cols;
    string rS ="";
    string cS ="";
    //src >> rows;
    //src >> cols;
    char c=' ';
    while (c!='[')
    {
        c=src.get();
    }
    src.get();

    while (c!=' ')
    {
        c=src.get();
        rS+=c;
    }
    c='.';
    while (c!=' ')
    {
        c=src.get();
        cS+=c;
    }
    while (c!=']')
        c=src.get();
    c=src.get();
    rows = stoi(rS);
    cols = stoi(cS);
    Mat ret(rows,cols,CV_32F);
    for (int r=0; r<rows; r++)
        for (int c=0; c<cols; c++)
        {
            src >> ret.at<float>(r,c);
            assert(ret.at<float>(r,c)==ret.at<float>(r,c));
        }
    return ret;
}

void CNNSPPSpotter::addLexicon(const vector<string>& lexicon)
{

    this->lexicon.resize(lexicon.size());
    Mat phocs(lexicon.size(),phocer.length(),CV_32F);
    for (int i=0; i<lexicon.size(); i++)
    {
        this->lexicon.at(i) = lowercaseAndStrip(lexicon[i]);
        vector<float> phoc = phocer.makePHOC(this->lexicon[i]);
        float n=0;
        for (int c=0; c<phoc.size(); c++)
        {
            n+=phoc[c]*phoc[c];
            phocs.at<float>(i,c) = phoc[c];
        }
        n = sqrt(n);
        if (n!=0)
            phocs.row(i) /= n;
    }
    lexicon_phocs = phocs;
}

 multimap<float,string> CNNSPPSpotter::transcribe(const Mat& image)
{
    assert(lexicon.size()>0);
    vector<Mat>* featurized = featurizer->featurize(image);
    Mat embedding = embedder->embed(featurized);
    delete featurized;
    Mat scores = lexicon_phocs*embedding;///now column vector
    multimap<float,string> ret;
    for (int j=0; j<lexicon.size(); j++)
    {
        ret.emplace(-1*scores.at<float>(j,0),lexicon.at(j));
    }
}

vector< multimap<float,string> > CNNSPPSpotter::transcribeCorpus()
{
    int KEEP=50;
    assert(lexicon.size()>0);
    vector< multimap<float,string> > ret(corpus_dataset->size());
    for (int i=0; i<corpus_dataset->size(); i++)
    {
        Mat phoc = corpus_full_embedded.at(i);
        Mat scores = lexicon_phocs*phoc;///now column vector
        assert(scores.rows == lexicon.size());
        //map<float,int> orderedScores;
        for (int j=0; j<lexicon.size(); j++)
        {
            ret.at(i).emplace(-1*scores.at<float>(j,0),lexicon.at(j));
        }
        auto iter=ret.at(i).begin();
        for (int j=0; j<KEEP; j++)
            iter++;
        ret.at(i).erase(iter,ret.at(i).end());
    }
    return ret;
}
multimap<float,string> CNNSPPSpotter::transcribeCorpus(int i)
{
    assert(lexicon.size()>0);
    multimap<float,string> ret;
    Mat phoc = corpus_full_embedded.at(i);
    Mat scores = lexicon_phocs*phoc;///now column vector
    assert(scores.rows == lexicon.size());
    //map<float,int> orderedScores;
    for (int j=0; j<lexicon.size(); j++)
    {
        ret.emplace(-1*scores.at<float>(j,0),lexicon.at(j));
    }
    return ret;
}

vector< multimap<float,string> > CNNSPPSpotter::transcribe(Dataset* words)
{
    assert(lexicon.size()>0);
    vector< multimap<float,string> > ret(words->size());
    for (int i=0; i<words->size(); i++)
    {
        //cout<<i<<" / "<<words->size()<<endl;
        vector<Mat>* featurized = featurizer->featurize(words->image(i));
        Mat phoc = embedder->embed(featurized);
        delete featurized;
        Mat scores = lexicon_phocs*phoc;///now column vector
        assert(scores.rows == lexicon.size());
        //map<float,int> orderedScores;
        for (int j=0; j<lexicon.size(); j++)
        {
            ret.at(i).emplace(-1*scores.at<float>(j,0),lexicon.at(j));
        }

        assert(ret.at(i).size()==lexicon.size());
        //takes too much memory
        auto iter = ret.at(i).begin();
        for (int i=0; i<lexicon.size()*TRANSCRIBE_KEEP_PORTION; i++)
        {
            iter++;
        }
        iter++;
        ret.at(i).erase(iter,ret.at(i).end());
    }
    return ret;
}


/*vector< multimap<float,string> > CNNSPPSpotter::transcribeCorpusCPV()
{
    int KEEP=50;
    assert(lexicon.size()>0);
    vector< multimap<float,string> > ret(corpus_dataset->size());
    for (int i=0; i<corpus_dataset->size(); i++)
    {
        Mat vec = npv(i);
        Mat phoc;// = corpus_full_embedded.at(i);
        Mat scores = lexicon_phocs*phoc;///now column vector
        assert(scores.rows == lexicon.size());
        //map<float,int> orderedScores;
        for (int j=0; j<lexicon.size(); j++)
        {
            ret.at(i).emplace(-1*scores.at<float>(j,0),lexicon.at(j));
        }
        auto iter=ret.at(i).begin();
        for (int j=0; j<KEEP; j++)
            iter++;
        ret.at(i).erase(iter,ret.at(i).end());
    }
    return ret;
}
*/


void CNNSPPSpotter::npvPrep(const vector<string>& ngrams)
{
    npvNgrams=ngrams;
    npvectors.resize(ngrams.size());
    auto vectorIter = npvectors.begin();
    npvNs.resize(ngrams.size());
    auto nIter = npvNs.begin();
    for (string ngram : ngrams)
    {
        *(vectorIter++) = normalizedPHOC(ngram);
        *(nIter++) = ngram.length();
    }
}

Mat CNNSPPSpotter::cpv(int i)
{
    int maxLen=0;
    int minLen=999999;
    for (auto& n : corpus_embedded)
    {
        if (n.second.at(i).cols>maxLen)
            maxLen=n.second.at(i).cols;
        if (n.second.at(i).cols<minLen)
            minLen=n.second.at(i).cols;
    }
    Mat ret = Mat::zeros(26, maxLen, CV_32F);
    map<int,Mat> nRet;
    map<int,vector<int> > nCounts;
    //Mat counts = Mat::zeros(26, maxLen, CV_32F);
    //vector<int> letterCounts(26);
    //float minVAll=0.2;
    for (int nIdx=0; nIdx<npvectors.size(); nIdx++)
    {
        int n = npvNs[nIdx];
        //if (n!=2)
        //    continue;
        Mat npvNgram = npvectors[nIdx];
        Mat wordEmbedding = corpus_embedded.at(n).at(i);
        //int lenDiff = maxLen-wordEmbedding.cols;
        /////
        //double minV,maxV;
        //Point minP,maxP;
        //minMaxLoc(wordEmbedding,&minV,&maxV,&minP,&maxP);
        //assert(minV>=-10 && maxV<=10);
        //minMaxLoc(npvNgram,&minV,&maxV,&minP,&maxP);
        //assert(minV>=-10 && maxV<=10);
        /////

        Mat scores = -1*distFunc(npvNgram, wordEmbedding);

        if (nRet.find(n) == nRet.end())
        {
            nRet[n] = Mat::zeros(26, maxLen, CV_32F);
            nCounts[n].resize(26);
        }


        for (int charIdx=0; charIdx<n; charIdx++)
        {
            int letterIdx = npvNgrams[nIdx][charIdx]-'a';
            nCounts[n][letterIdx]++;
            float offset = (maxLen-scores.cols)/2.0;
            if (n==2 && charIdx==0)
                offset-=0.5*charWidth*featurizeScale;
            else if (n==2 && charIdx==1)
                offset+=0.5*charWidth*featurizeScale;
            else if (n==3 && charIdx==0)
                offset-=charWidth*featurizeScale;
            else if (n==3 && charIdx==2)
                offset+=charWidth*featurizeScale;

            offset = std::min(maxLen-1.0f,std::max(0.0f,round(offset)));
            int width = scores.cols + std::min(maxLen-(scores.cols+(int)offset),0);//clip if going to go off ret vector
            nRet[n].row(letterIdx)(Rect(offset,0,width,1)) += scores(Rect(0,0,width,1));
        }

        /*

        ///// 
        //minMaxLoc(scores,&minV,&maxV,&minP,&maxP);
        //if (minV < minVAll)
        //    minVAll = minV;
        //assert(minV>=-10 && maxV<=10);
        //scores = (scores-minV)/(maxV-minV);//normalize
        scores = (scores+1)/(2);//normalize
        /////
        int windowWidth = n*charWidth*featurizeScale;
        //Mat filter(1,1.5*charWidth*featurizeScale,CV_32F);
        float stddev = 0.1*charWidth*featurizeScale;
        //for (int c=0; c<(int)(1.5*charWidth*featurizeScale); c++)
        //{
        //    filter.at<float>(0,c) = (1/(sqrt(2*CV_PI)*stddev))*exp(-0.5*pow((c-1.5*charWidth*featurizeScale/2)/stddev,2));
        //}
        for (int c=lenDiff/2; c<scores.cols+lenDiff/2; c++)
        {
            for (int charIdx=0; charIdx<n; charIdx++)
            {
                int center=c;
                
                if (n==2 && charIdx==0)
                    center-=0.5*charWidth*featurizeScale;
                else if (n==2 && charIdx==1)
                    center+=0.5*charWidth*featurizeScale;
                else if (n==3 && charIdx==0)
                    center-=charWidth*featurizeScale;
                else if (n==3 && charIdx==2)
                    center+=charWidth*featurizeScale;



                int letterIdx = npvNgrams[nIdx][charIdx]-'a';
                if (c==lenDiff/2)
                    letterCounts[letterIdx]+=1;
                //ret.row(npvNgrams[nIdx][charIdx]-'a') += scores.at<float>(1,c)*toAdd;
                //for (int c2=(maxLen-minLen)/2; c2<minLen; c2++)

                //assert(center>=0 && center<maxLen);
                //if (center>=0 && center<maxLen)
                center = std::max(0,std::min(maxLen-1,center));
                   ret.at<float>(letterIdx,center) += scores.at<float>(0,c-lenDiff/2)/n;
                   counts.at<float>(letterIdx,center) += 1;
                /*for (int c2=0; c2<maxLen; c2++)
                {
                    //assert(scores.at<float>(0,c)>=-2 && scores.at<float>(0,c)<=2);
                    //assert((1/(sqrt(2*CV_PI)*stddev))*exp(-0.5*pow((c2-center)/stddev,2))>=0 && (1/(sqrt(2*CV_PI)*stddev))*exp(-0.5*pow((c2-center)/stddev,2))<=1);
                    ret.at<float>(letterIdx,c2) += scores.at<float>(0,c-lenDiff/2) * (1/(sqrt(2*CV_PI)*stddev))*exp(-0.5*pow((c2-center)/stddev,2));
                    //ret.at<float>(letterIdx,c2) += std::max(0.0f,(c2<center)?(scores.at<float>(0,c-lenDiff/2)-center + c2) : (scores.at<float>(0,c-lenDiff/2)+center - c2));
                }bet_size,/
            }
        }
        */

    }
    /*
    for (int letterIdx=0; letterIdx<26; letterIdx++)
    {
        if (letterCounts[letterIdx]>0)
        {
            ret.row(letterIdx) /= letterCounts[letterIdx];
            //double minV;
            //minMaxLoc(ret.row(letterIdx),&minV);
            //if (minV < minVAll && minV>0.00001)
            //    minVAll=minV;
        }
    }
    /*
    for (int letterIdx=0; letterIdx<26; letterIdx++)
    {
        if (letterCounts[letterIdx]==0)
            ret.row(letterIdx) = minVAll;
    }*/
    //divide(ret,counts,ret);
    //float m = mean(ret.row('z'-'a'))[0];
    //ret.row('z'-'a') = ret.row('z'-'a')*0 + m;
    map<int,int> skipped;

    for (auto& p : nRet)
    {
        set<int> skip; //These have no spottings, so don't include in softmax
        for (int letterIdx=0; letterIdx<26; letterIdx++)
        {
            if (nCounts[p.first][letterIdx]>0)
                p.second.row(letterIdx) /= nCounts[p.first][letterIdx];
            else
            {
                skip.insert(letterIdx);
                skipped[letterIdx]++;
            }
        }
        for (int c=0; c<p.second.cols; c++)
        {
            softMax(p.second.col(c),skip);
        }
        add(ret,p.second*((26.0-skip.size())/26),ret);
    }
    for (int letterIdx=0; letterIdx<26; letterIdx++)
        ret.row(letterIdx) /= nRet.size()-skipped[letterIdx];
    for (int c=0; c<maxLen; c++)
        softMax(ret.col(c),set<int>());
    return ret;   
    
}

void CNNSPPSpotter::softMax(Mat colVec,set<int> skip)
{
    exp(colVec,colVec);
    for (int i : skip)
        colVec.at<float>(i,0)=0;
    colVec /= sum(colVec)[0];
}

Mat CNNSPPSpotter::npv(int wordI)
{
    //Mat phoc = corpus_full_embedded.at(i);
    //Mat scores = ngram_phocs*phoc;
    int maxLen=0;
    for (auto& n : corpus_embedded)
    {
        if (n.second.at(wordI).cols>maxLen)
            maxLen=n.second.at(wordI).cols;
    }
    Mat ret(npvectors.size(), maxLen, CV_32F);
    for (int i=0; i<npvectors.size(); i++)
    {
        Mat npvNgram = npvectors[i];
        Mat wordEmbedding = corpus_embedded.at(npvNs[i]).at(wordI);
        int lenDiff = maxLen-wordEmbedding.cols;
        if (lenDiff>0)
        {
            //pad the embedding to make them all the same length
            int frontL = lenDiff/2;
            int backL = ceil(lenDiff/2.0);
            if (frontL>0)
            {
                Mat front = Mat::zeros(wordEmbedding.rows,frontL,CV_32F);
                hconcat(front,wordEmbedding,wordEmbedding);
            }
            if (backL>0)
            {
                Mat back = Mat::zeros(wordEmbedding.rows,backL,CV_32F);
                hconcat(wordEmbedding,back,wordEmbedding);
            }
        }
        ret.row(i) = distFunc(npvNgram, wordEmbedding);
    }
    return ret;   
}

vector<SpottingLoc> CNNSPPSpotter::massSpot(const vector<string>& ngrams, Mat& crossScores)
{
    vector<SpottingLoc> ret;
    float minScoreQbS=9999999;
    float maxScoreQbS=-9999999;
    float refinePortion=0.1;
    for (string ngram : ngrams)
    {
        vector< SubwordSpottingResult > res = subwordSpot(ngram, refinePortion);
        for (SubwordSpottingResult r : res)
        {
            //check if any results are duplicates
            bool nodup=true;
            for (SpottingLoc& l : ret)
            {
                if (r.imIdx==l.imIdx && r.startX==l.startX && r.endX==l.endX)
                {
                    l.scores[ngram]=-1*r.score;
                    nodup=false;
                    break;
                }
            }
            if (nodup)
            {
                int id = ret.size();
                ret.emplace_back(r,ngram,id);
            }
            if (r.score<minScoreQbS)
                minScoreQbS=r.score;
            if (r.score>maxScoreQbS)
                maxScoreQbS=r.score;
        }
    }

    Mat allInstanceVectors(ret.size(),phocer.length(),CV_32F);//each row is instance
    for (SpottingLoc& l : ret)
    {
        //get vec
        int numChar = l.numChar;
        int windIdx = l.startX/stride;
        if (corpus_embedded.at(numChar).at(l.imIdx).cols<=windIdx)
            windIdx = corpus_embedded.at(numChar).at(l.imIdx).cols-1;
        allInstanceVectors(Rect(0,l.id,phocer.length(),1)) = corpus_embedded.at(numChar).at(l.imIdx).col(windIdx).t();
        /////
        for (int x=0; x<phocer.length(); x++)
            assert(allInstanceVectors.at<float>(l.id,x) == allInstanceVectors.at<float>(l.id,x));
        /////
    }

    //spot missing QbS
    for (string ngram : ngrams)
    {
        Mat ngramEmbedding = normalizedPHOC(ngram);
        for (SpottingLoc& l : ret)
        {
            if (l.scores.find(ngram) == l.scores.end())
            {
                float newScore=ngramEmbedding.t().dot(allInstanceVectors(Rect(0,l.id,phocer.length(),1)));
                l.scores[ngram]=newScore;

                if (newScore<minScoreQbS)
                    minScoreQbS=newScore;
                if (newScore>maxScoreQbS)
                    maxScoreQbS=newScore;
            }
        }
    }
    /////
    for (int r=0; r<allInstanceVectors.rows; r++)
        for (int c=0; c<allInstanceVectors.cols; c++)
            assert(allInstanceVectors.at<float>(r,c) == allInstanceVectors.at<float>(r,c));
    ////

    mulTransposed(allInstanceVectors,crossScores,false);
    /////
    for (int r=0; r<crossScores.rows; r++)
        for (int c=0; c<crossScores.cols; c++)
            assert(crossScores.at<float>(r,c) == crossScores.at<float>(r,c));
    ////

    //normalize QbS scores
    //We'll do this in a way that slightly biases the QbS to have higher scores?? nah...
    double minScoreQbE, maxScoreQbE;
    minMaxLoc(crossScores,&minScoreQbE,&maxScoreQbE);
    for (SpottingLoc& l : ret)
    {
        for (auto& n_s : l.scores)
        {
            n_s.second = (n_s.second-minScoreQbS)/(maxScoreQbS-minScoreQbS) * (maxScoreQbE-minScoreQbE) + minScoreQbE;
        }
    }

    return ret;

}
