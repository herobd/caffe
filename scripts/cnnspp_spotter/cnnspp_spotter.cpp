#include "cnnspp_spotter.h"
#include "cnnspp_spotter_eval.cpp"

CNNSPPSpotter::CNNSPPSpotter(string featurizerModel, string embedderModel, string netWeights, bool normalizeEmbedding, float featurizeScale, int charWidth, int stride, string saveName) : stride(stride), featurizeScale(featurizeScale)
{
    windowWidth = 2*charWidth;
    this->saveName = saveName;
    featurizer = new CNNFeaturizer(featurizerModel,netWeights);
    embedder = new SPPEmbedder(embedderModel,netWeights,normalizeEmbedding);
    cout<<"Window width:"<<windowWidth<<endl;
    cout<<charWidth<<endl;

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

    vector<Mat>* im_featurized = corpus_featurized.at(wordIndex);
    return compare_(text,im_featurized);
}

float CNNSPPSpotter::compare_(string text, vector<Mat>* im_featurized)
{
    Mat imEmbedding = embedder->embed(im_featurized);
    delete im_featurized;

    vector<float> phoc = phocer.makePHOC(text);
    Mat textEmbedding(phoc.size(),1,CV_32F,phoc.data());
    
    return imEmbedding.dot(textEmbedding);
}

//With a GPU, we could efficiently batch multiple exemplars together. Currently the refineStepFast function does this, but it uses a small batch
//Cannot be const because of network objects
vector< SubwordSpottingResult > CNNSPPSpotter::subwordSpot(const Mat& exemplar, float refinePortion)
{
    vector<Mat>* ex_featurized = featurizer->featurize(exemplar);
    Mat exemplarEmbedding = embedder->embed(ex_featurized);
    delete ex_featurized;
    
    multimap<float,pair<int,int> > scores;



    //ttt#pragma omp parallel for
    for (int i=0; i<corpus_dataset->size(); i++)
    {
        Mat s_batch;
        Mat cal = exemplarEmbedding.t() * corpus_embedded.at(i);
        /*Mat calS = exemplarEmbedding.t() * corpus_embeddedS[i];
        Mat tmp(1,cal.cols+calS.cols,CV_32F);
        for (int i=0; i<cal.cols; i++)
        {
            tmp.at<float>(0,i*2)=cal.at<float>(0,i);
            if (i<calS.cols)
                tmp.at<float>(0,1+ i*2)=calS.at<float>(0,i);
        }
        cal=tmp;
        */
        //Mat cal2 = exemplarEmbedding2.t() * corpus_embedded[i];
        //Mat cal3 = exemplarEmbedding3.t() * corpus_embedded[i];
        
        //for (int c=0; c<cal.cols; c++)
        //    cal.at<float>(0,c) = max(cal.at<float>(0,c), max(cal2.at<float>(0,c),cal3.at<float>(0,c)));

        s_batch=-1*cal;//flip, so lower scores are better

        //Mat cal = corpus_embedded[i].clone();
        //for (int c=0; c<corpus_embedded[i].cols; c++)
        //{
            //cal.col(c) -= exemplarEmbedding;
            //cv::pow(cal.col(c),2,cal.col(c));
        //}
        //cv::reduce(cal,s_batch,0,CV_REDUCE_SUM);
        //cv::sqrt(s_batch, s_batch);

        //assert(s_batch.cols == 

        assert(s_batch.rows==1);
        float topScoreInd=-1;
        float topScore=numeric_limits<float>::max();
        float top2ScoreInd=-1;
        float top2Score=numeric_limits<float>::max();
        for (int c=0; c<s_batch.cols; c++) {
            //if ((r)*NET_PIX_STRIDE >= im_width) {
            //    cout<<"ERROR: sliding window moving out of bounds for iamge "<<i<<". Window starts at "<<(r)*NET_PIX_STRIDE<<", but image is only "<<im_width<<" wide"<<endl;
            //}
            //assert((r)*NET_PIX_STRIDE<im_width);
            float s = s_batch.at<float>(0,c);
            //cout <<"im["<<i<<"]: "<<corpus_dataset->labels()[i]<<" x: "<<c*NET_PIX_STRIDE<<" score: "<<s<<endl;
            if (s<topScore)
            {
                topScore=s;
                topScoreInd=c;
            }
        }
        int diff = ((windowWidth/2.0) *.8)/stride;
        for (int c=0; c<s_batch.cols; c++) {
            float s = s_batch.at<float>(0,c);
            if (s<top2Score && abs(c-topScoreInd)>diff)
            {
                top2Score=s;
                top2ScoreInd=c;
            }
        }

        //ttt#pragma omp critical (subword_spot)
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
        //if (refineThresh!=0 && iter->first > -1*fabs(refineThresh))
        //    break;
        finalScores.at(i) = refine(iter->first,iter->second.first,iter->second.second,exemplarEmbedding);
    }

    return finalScores;
 
}


vector< SubwordSpottingResult > CNNSPPSpotter::subwordSpot(const string& exemplar, float refinePortion)
{
    //vector<Mat>* ex_featurized = featurizer->featurize(exemplar);
    //Mat exemplarEmbedding = embedder->embed(ex_featurized);
    //delete ex_featurized;
    vector<float> phoc = phocer.makePHOC(exemplar);
    Mat exemplarEmbedding(phoc.size(),1,CV_32F,phoc.data());
    
    multimap<float,pair<int,int> > scores;



    //ttt#pragma omp parallel for
    for (int i=0; i<corpus_dataset->size(); i++)
    {
        Mat s_batch;
        Mat cal = exemplarEmbedding.t() * corpus_embedded.at(i);

        s_batch=-1*cal;//flip, so lower scores are better


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
        int diff = ((windowWidth/2.0) *.8)/stride;
        for (int c=0; c<s_batch.cols; c++) {
            float s = s_batch.at<float>(0,c);
            if (s<top2Score && abs(c-topScoreInd)>diff)
            {
                top2Score=s;
                top2ScoreInd=c;
            }
        }

        //ttt#pragma omp critical (subword_spot)
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
        finalScores.at(i) = refine(iter->first,iter->second.first,iter->second.second,exemplarEmbedding);
    }

    return finalScores;
 
}

SubwordSpottingResult CNNSPPSpotter::refine(float score, int imIdx, int windIdx, const Mat& exemplarEmbedding)
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


    return SubwordSpottingResult(imIdx,bestScore,bestX0,bestX1);
}

void CNNSPPSpotter::refineStep(int imIdx, float* bestScore, int* bestX0, int* bestX1, float scale, const Mat& exemplarEmbedding)
{
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
    Mat bestsScore = -1* (exemplarEmbedding.t() * wEmbedding);
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
    windows[5] = Rect (newX0inF,0,newX1inF-newX0inF+1,corpus_featurized.at(imIdx)->front().rows);//both in
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
    Mat wScores = -1*(exemplarEmbedding.t() * embeddings);

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

Mat CNNSPPSpotter::embedFromCorpusFeatures(int imIdx, Rect window)
{
    vector<Mat> windowed_features(corpus_featurized.at(imIdx)->size());
    for (int c=0; c<corpus_featurized.at(imIdx)->size(); c++)
    {
        windowed_features.at(c) = corpus_featurized.at(imIdx)->at(c)(window);
    }
    return embedder->embed(&windowed_features);
}


void CNNSPPSpotter::setCorpus_dataset(const Dataset* dataset, bool fullWordEmbed)
{
    corpus_dataset = dataset;
    //loadCorpusEmbedding(NET_IN_SIZE,NET_PIX_STRIDE);
    string nameFeaturization = saveName+"_corpus_cnnFeatures_"+featurizerFile+"_"+dataset->getName()+".dat";
    string nameEmbedding = saveName+"_corpus_sppEmbedding_"+embedderFile+"_"+dataset->getName()+"_w"+to_string(windowWidth)+"_s"+to_string(stride)+".dat";
    if (fullWordEmbed)
        nameEmbedding=saveName+"_corpus_sppEmbedding_"+embedderFile+"_"+dataset->getName()+"_full.dat";
#ifndef NO_FEAT
    ifstream in(nameFeaturization);
    if (in)
    {
        cout<<"Reading in features: "<<nameFeaturization<<endl;
        int numWordsRead;
        in >> numWordsRead;
        assert(numWordsRead == dataset->size());
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
            //corpus_embedded.at(i) = embedder->embed(resized);
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
#else
    
        ifstream in;
#endif

    in.open(nameEmbedding);
    if (in)
    {
        cout<<"Reading in embedding: "<<nameEmbedding<<endl;
        int numWordsRead;
        in >> numWordsRead;
        assert(numWordsRead == dataset->size());
        corpus_embedded.resize(numWordsRead);
        for (int i=0; i<numWordsRead; i++)
        {
            corpus_embedded.at(i) = readFloatMat(in);
        }
        in.close();
        cout <<"done"<<endl;

    }
    else
    {
        cout<<"Creating embedding for "<<corpus_dataset->getName()<<" (w:"<<windowWidth<<" s:"<<stride<<")"<<endl;
        cout<<"writing to: "<<nameEmbedding<<endl;
        corpus_embedded.resize(corpus_dataset->size());
        for (int i=0; i<corpus_dataset->size(); i++)
        {
            if (fullWordEmbed)
            {
                corpus_embedded.at(i)=embedder->embed(corpus_featurized.at(i));
            }
            else
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
                    if (corpus_embedded.at(i).rows==0)
                        corpus_embedded.at(i)=a;
                    else
                        hconcat(corpus_embedded.at(i), a, corpus_embedded.at(i));
                }
                if (corpus_embedded.at(i).rows==0 && corpus_featurized.at(i)->front().cols<=windowWidth)
                {
                    for (int c=0; c<corpus_featurized.at(i)->size(); c++)
                    {
                        windowed_features.at(c) = corpus_featurized.at(i)->at(c);
                    }

                    corpus_embedded.at(i)=embedder->embed(&windowed_features);
                }
                else if (corpus_embedded.at(i).rows==0)
                {
                    cout<<"["<<i<<"] window: "<<windowWidth<<", image width: "<< corpus_featurized.at(i)->front().cols<<endl;
                    assert(corpus_embedded.at(i).rows>0);
                }
            }

        }
        ofstream out(nameEmbedding);
        out << corpus_dataset->size() << " ";
        for (int i=0; i<corpus_dataset->size(); i++)
        {
            assert(corpus_embedded.at(i).rows>0);
            writeFloatMat(out,corpus_embedded.at(i));
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
            src >> ret.at<float>(r,c);
    return ret;
}

void CNNSPPSpotter::addLexicon(const vector<string>& lexicon)
{

    this->lexicon.resize(lexicon.size());
    Mat phocs(lexicon.size(),phocer.length(),CV_32F);
    for (int i=0; i<lexicon.size(); i++)
    {
        this->lexicon.at(i) = lowercase(lexicon[i]);
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
    assert(lexicon.size()>0);
    vector< multimap<float,string> > ret(corpus_dataset->size());
    for (int i=0; i<corpus_dataset->size(); i++)
    {
        Mat phoc = corpus_embedded.at(i);
        Mat scores = lexicon_phocs*phoc;///now column vector
        assert(scores.rows == lexicon.size());
        //map<float,int> orderedScores;
        for (int j=0; j<lexicon.size(); j++)
        {
            ret.at(i).emplace(-1*scores.at<float>(j,0),lexicon.at(j));
        }
    }
    return ret;
}
multimap<float,string> CNNSPPSpotter::transcribeCorpus(int i)
{
    assert(lexicon.size()>0);
    multimap<float,string> ret;
    Mat phoc = corpus_embedded.at(i);
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
    }
    return ret;
}

