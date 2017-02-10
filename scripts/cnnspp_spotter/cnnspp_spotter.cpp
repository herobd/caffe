#include "cnnspotter.h"
#include "cnnspotter_eval.cpp"

CNNSPPSpotter::CNNSPPSpotter(string featurizerModel, string embedderModel, string netWeights, int windowWidth, int stride, float featurizeScale, string saveName) : windowWidth(windowWidth), stride(stride), featurizerScale(featurizerScale)
{
    this->saveName = saveName;
    featurizer = new CNNFeaturizer(featurizerModel,netWeights);
    embedder = new SPPEmbedder(embedderModel,netWeights);

    corpus_dataset=NULL;
    //corpus_featurized=NULL;
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

//With a GPU, we could efficiently batch multiple exemplars together
vector< SubwordSpottingResult > CNNSPPSpotter::subwordSpot(const Mat& exemplar, float refinePortion) const
{
    vector<Mat>* ex_featurized = featurizer->featurize(exemplar);
    Mat exemplarEmbedding = embedder->embed(ex_featurized);
    delete ex_featurized;
    
    multimap<float,pair<int,int> > scores;



    //ttt#pragma omp parallel for
    for (int i=0; i<corpus_dataset->size(); i++)
    {
        Mat s_batch;
        Mat cal = exemplarEmbedding.t() * corpus_embedded[i];
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
        int diff = ((NET_IN_SIZE/2) *.8)/NET_PIX_STRIDE;
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

SubwordSpottingResult CNNSPPSpotter::refine(float score, int imIdx, int windIdx, const Mat& exemplarEmbedding) const
{
    //TODO
    float bestScore=score;
    int newX0 = windIdx*stride;
    int newX1 = std::min(newX0+windowWidth-1, corpus_dataset->image(imIdx).cols-1);
    ////
    /*
    Mat disp;
    cvtColor(corpus_dataset->image(imIdx),disp,CV_GRAY2BGR);
    for (int r=0; r<disp.rows; r++)
    {
        disp.at<Vec3b>(r,newX0)[2]=255;
        disp.at<Vec3b>(r,newX1)[2]=255;
        disp.at<Vec3b>(r,newX0)[1]=5;
        disp.at<Vec3b>(r,newX1)[1]=5;
    }
    cout<<"score: "<<score<<endl;
    imshow("spotting",disp);
    waitKey();
    */
    ////
    /*if (corpus_dataset->image(imIdx).cols*corpus_scalars[imIdx] > NET_PIX_STRIDE+NET_IN_SIZE)
    {
        Mat im = corpus_dataset->image(imIdx);
        Mat resized;
        resize(im,resized,Size(max((int)(corpus_scalars[imIdx]*im.cols),(int)NET_IN_SIZE),NET_IN_SIZE));
        int length=NET_IN_SIZE+NET_PIX_STRIDE;
        int from = windIdx*NET_PIX_STRIDE - NET_PIX_STRIDE/2.0;
        int firstIndex=0;
        int to = from+length-1;//newX1+ceil(NET_PIX_STRIDE/2.0);
        int secondIndex=1;

        if (from<0)
        {
            firstIndex=-1;
            secondIndex=0;
            from = windIdx*NET_PIX_STRIDE + NET_PIX_STRIDE/2.0;
            length=NET_IN_SIZE;
        }
        if (to>=resized.cols)
        {
            secondIndex=-1;
            //to = newX1-ceil(NET_PIX_STRIDE/2.0);
            length=NET_IN_SIZE;
        }
        assert(firstIndex>=0 || secondIndex >=0);

        Mat shiftedEmbedding = embedder->embed( resized(Rect( from, 0, length, NET_IN_SIZE)) );
        assert(shiftedEmbedding.cols==2 || firstIndex<0 || secondIndex<0);//The shifting should have gotten 2 embeddings each half a stride away from the starting max.

        if (firstIndex>=0 && shiftedEmbedding.at<float>(0,firstIndex) > bestScore)
        {
            bestScore = shiftedEmbedding.at<float>(0,firstIndex);
            newX0 = (windIdx*NET_PIX_STRIDE - NET_PIX_STRIDE/2.0)/corpus_scalars[imIdx];
            newX1 = (windIdx*NET_PIX_STRIDE+NET_IN_SIZE - NET_PIX_STRIDE/2.0)/corpus_scalars[imIdx] - 1;
        }
        if (secondIndex>=0 && shiftedEmbedding.at<float>(0,secondIndex) > bestScore)
        {
            bestScore = shiftedEmbedding.at<float>(0,secondIndex);
            newX0 = (windIdx*NET_PIX_STRIDE + NET_PIX_STRIDE/2.0)/corpus_scalars[imIdx];
            newX1 = (windIdx*NET_PIX_STRIDE+NET_IN_SIZE + NET_PIX_STRIDE/2.0)/corpus_scalars[imIdx] - 1;
        }
    }*/

    //hide
    /*
    Mat im = corpus_dataset->image(imIdx);
    int avg=0;
    for (int r=0; r<im.rows; r++)
        for (int c=0; c<im.cols; c++)
            avg += im.at<unsigned char>(r,c);
    avg /= im.rows*im.cols;
    im = im(Rect( newX0, 0, newX1-newX0 +1,im.rows));
    Mat resized;
    //resize(im,resized,Size(max((int)(corpus_scalars[imIdx]*im.cols*2),(int)NET_IN_SIZE),NET_IN_SIZE));
    //resized = resized(Rect( newX0, 0, newX1-newX0 +1, NET_IN_SIZE));
    resize(im,resized,Size(NET_IN_SIZE,NET_IN_SIZE));

    int stride=4;
    vector<float> left, right;
    float min= 99999;
    float max=-99999;
    for (int x=stride; x<=resized.cols/2; x+=stride)
    {
        Mat hid = resized.clone();
        hid(Rect(0,0,x,resized.rows))=avg;
        imshow("left",hid);
        waitKey(300);
        Mat hidEmb = embedder->embed(hid);
        Mat s = exemplarEmbedding.t() * hidEmb;
        assert(s.cols==1);
        assert(s.rows==1);
        left.push_back(s.at<float>(0,0)+score);
        if (left.back()>max)
            max=left.back();
        if (left.back()<min)
            min=left.back();

        hid = resized.clone();
        hid(Rect(resized.cols-x-1,0,x,resized.rows))=avg;
        imshow("right",hid);
        waitKey(300);
        hidEmb = embedder->embed(hid);
        s = exemplarEmbedding.t() * hidEmb;
        assert(s.cols==1);
        assert(s.rows==1);
        right.push_back(s.at<float>(0,0)+score);
        if (right.back()>max)
            max=right.back();
        if (right.back()<min)
            min=right.back();

        cout<<"l: "<<left.back()<<"  r: "<<right.back()<<endl;
    }

    Mat disp;
    cvtColor(resized,disp,CV_GRAY2BGR);
    for (int i=0; i<left.size(); i++)
    {
        int r=0;
        int b=0;
        if (left[i]<0)
            b=255* left[i]/(min+0.0);
        else
            r=255* left[i]/(max+0.0);
        for (int c=i*stride; c<(i+1)*stride; c++)
        {
            for (int r=0; r<disp.rows; r++)
            {
                disp.at<Vec3b>(r,c)[0]=b;
                disp.at<Vec3b>(r,c)[2]=r;
            }
        }

        r=0;
        b=0;
        if (right[i]<0)
            b=255* right[i]/(min+0.0);
        else
            r=255* right[i]/(max+0.0);
        for (int c=disp.cols-1-i*stride; c>disp.cols-1-(i+1)*stride; c--)
        {
            for (int r=0; r<disp.rows; r++)
            {
                disp.at<Vec3b>(r,c)[0]=b;
                disp.at<Vec3b>(r,c)[2]=r;
            }
        }
    }
    imshow("hiding",disp);
    waitKey();
    */


    return SubwordSpottingResult(imIdx,bestScore,newX0,newX1);
}


void CNNSPPSpotter::setCorpus_dataset(const Dataset* dataset)
{
    corpus_dataset = dataset;
    //loadCorpusEmbedding(NET_IN_SIZE,NET_PIX_STRIDE);
    string nameEmbedding = saveName+"_corpus_sppEmbedding_"+dataset->getName()+"_w"+to_string(windowWidth)+"_s"+to_string(stride)+".dat";
    string nameFeaturization = saveName+"_corpus_cnnFeatures_"+dataset->getName()+".dat";
    ifstream in(nameEmbedding);
    if (in)
    {
        int numWordsRead;
        in >> numWordsRead;
        assert(numWordsRead == dataset->size());
        corpus_featurized.resize(numWordsRead);
        for (int i=0; i<numWordsRead; i++)
        {
            int numChannels;
            in >> numChannels;
            if (i>0)
                assert(corpus_featurized[0]->size() == numChannels);
            corpus_featurized[i] = new vector<Mat>(numChannels);
            for (int j=0; j<numChannels; j++)
            {
                corpus_featurized.at(i)->at(j) = readFloatMat(in);
            }
        }
        in.close();

    }
    else
    {
        cout<<"Featurizing "<<corpus_dataset->getName()<<endl;

        corpus_featurized.resize(corpus_dataset->size());
        for (int i=0; i<corpus_dataset->size(); i++)
        {
            Mat im = corpus_dataset->image(i);
            //corpus_embedded.at(i) = embedder->embed(resized);
            corpus_featurized.at(i) = featurizer->featurize(im);

        }
        ofstream out(name);
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


    ifstream in(name);
    if (in)
    {
        int numWordsRead;
        in >> numWordsRead;
        assert(numWordsRead == dataset->size());
        corpus_embedded.resize(numWordsRead);
        for (int i=0; i<numWordsRead; i++)
        {
            corpus_embedded.at(i) = readFloatMat(in);
        }
        in.close();

    }
    else
    {
        cout<<"Creating embedding for "<<corpus_dataset->getName()<<" (w:"<<windowWidth<<" s:"<<stride<<")"<<endl;
        corpus_embedded.resize(corpus_dataset->size());
        for (int i=0; i<corpus_dataset->size(); i++)
        {
            vector<Mat> windowed_features(corpus_featurized.at(i)->size());
            for (int ws=0; ws+windowWidth < corpus_featurized.at(i)->front().cols; ws+=stride)
            {
                Rect window(ws*featurizeScale,0,windowWidth*featurizeScale,corpus_featurized.at(i)->front().rows);
                for (int c=0; c<corpus_featurized.at(i)->size(); c++)
                {
                    windowed_features[c] = corpus_featurized.at(i)->at(c)(window);
                }
                Mat a = embedder->embed(&windowed_features);
                if (corpus_embedded.at(i).rows==0)
                    corpus_embedded.at(i)=a;
                else
                    hconcat(corpus_embedded.at(i), a, corpus_embedded.at(i));
            }

        }
        ofstream out(name);
        out << corpus_dataset->size() << " ";
        for (int i=0; i<corpus_dataset->size(); i++)
        {
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

