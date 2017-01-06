#include "cnnspotter.h"
#include "cnnspotter_eval.cpp"

CNNSpotter::CNNSpotter(string netModel, string netWeights, int netInputSize, int netPixelStride, string saveName)
{
    this->saveName = saveName;
    embedder = new CNNEmbedder(netModel,netWeights);
    NET_IN_SIZE=netInputSize;
    NET_PIX_STRIDE=netPixelStride;
    corpus_dataset=NULL;
}

CNNSpotter::~CNNSpotter()
{
    delete embedder;
}

//With a GPU, we could efficiently batch multiple exemplars together
vector< SubwordSpottingResult > CNNSpotter::subwordSpot(const Mat& exemplar, float refinePortion) const
{
    Mat fitted;
    resize(exemplar, fitted, Size(NET_IN_SIZE, NET_IN_SIZE));
    Mat exemplarEmbedding = embedder->embed(fitted);
    
    multimap<float,pair<int,int> > scores;



    //ttt#pragma omp parallel for
    for (int i=0; i<corpus_dataset->size(); i++)
    {
        Mat s_batch;
        cout<<"Word ["<<corpus_embedded[i].channels()<<" , "<<corpus_embedded[i].rows<<" , "<<corpus_embedded[i].cols<<"]"<<endl;
        cout<<"Exem ["<<exemplarEmbedding.channels()<<" , "<<exemplarEmbedding.rows<<" , "<<exemplarEmbedding.cols<<"]"<<endl;
        Mat cal = corpus_embedded[i].clone();
        for (int c=0; c<corpus_embedded[i].cols; c++)
        {
            cal.col(c) -= exemplarEmbedding;
            cal.col(c) *= cal.col(c);
        }
        cv::reduce(cal,s_batch,along row,SUM);
        cv::sqrt(s_batch, s_batch);
        //int im_width = corpus_dataset->image(i).cols;
        //assert(s_batch.rows<=im_width);
        float topScoreInd=-1;
        float topScore=-999999;
        float top2ScoreInd=-1;
        float top2Score=-999999;
        for (int c=0; c<s_batch.cols; c++) {
            //if ((r)*NET_PIX_STRIDE >= im_width) {
            //    cout<<"ERROR: sliding window moving out of bounds for iamge "<<i<<". Window starts at "<<(r)*NET_PIX_STRIDE<<", but image is only "<<im_width<<" wide"<<endl;
            //}
            //assert((r)*NET_PIX_STRIDE<im_width);
            float s = s_batch.at<float>(0,c);
            //cout <<"im "<<i<<" x: "<<r*stride<<" score: "<<s<<endl;
            if (s>topScore)
            {
                topScore=s;
                topScoreInd=c;
            }
        }
        int diff = (NET_IN_SIZE*.8)/NET_PIX_STRIDE;
        for (int c=0; c<s_batch.cols; c++) {
            float s = s_batch.at<float>(0,c);
            if (s>top2Score && abs(c-topScoreInd)>diff)
            {
                top2Score=s;
                top2ScoreInd=c;
            }
        }

        //ttt#pragma omp critical (subword_spot)
        {
        assert(topScoreInd!=-1);
        scores.emplace(-1*topScore, make_pair(i,topScoreInd));
        if (top2ScoreInd!=-1)
            scores.emplace(-1*top2Score, make_pair(i,top2ScoreInd));
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

SubwordSpottingResult CNNSpotter::refine(float score, int imIdx, int windIdx, const Mat& exemplarEmbedding) const
{
    //TODO
    float bestScore=score;
    int newX0 = windIdx*NET_PIX_STRIDE/corpus_scalars[imIdx];
    int newX1 = (windIdx*NET_PIX_STRIDE+NET_IN_SIZE)/corpus_scalars[imIdx] - 1;
    /*if (corpus_dataset->image(imIdx).cols*corpus_scalars[i] > NET_PIX_STRIDE+NET_IN_SIZE)
    {
        Mat im = corpus_dataset->image(imIdx);
        Mat resized;
        resize(im,resized,Size(max((int)(corpus_scalars[i]*im.cols),(int)NET_IN_SIZE),NET_IN_SIZE));
        Mat shiftedEmbedding = embedder->embed( resized(Rect( newX0-ceil(NET_PIX_STRIDE/2.0), 0, NET_IN_SIZE+NET_PIX_STRIDE, NET_PIX_STRIDE)) );
        assert(shiftedEmbedding.cols==2);//The shifting should have gotten 2 embeddings each half a stride away from the starting max.

        if (shiftedEmbedding.at<float>(0,0) > bestScore)
        {
            bestScore = shiftedEmbedding.at<float>(0,0);
            newX0 = windIdx*NET_PIX_STRIDE/corpus_scalars[imIdx] - NET_PIX_STRIDE/2.0;
            newX1 = (windIdx*NET_PIX_STRIDE+NET_IN_SIZE)/corpus_scalars[imIdx] - 1 - NET_PIX_STRIDE/2.0;
        }
        if (shiftedEmbedding.at<float>(0,1) > bestScore)
        {
            bestScore = shiftedEmbedding.at<float>(0,1);
            newX0 = windIdx*NET_PIX_STRIDE/corpus_scalars[imIdx] + NET_PIX_STRIDE/2.0;
            newX1 = (windIdx*NET_PIX_STRIDE+NET_IN_SIZE)/corpus_scalars[imIdx] - 1 + NET_PIX_STRIDE/2.0;
        }
    }*/
    return SubwordSpottingResult(imIdx,bestScore,newX0,newX1);
}


void CNNSpotter::setCorpus_dataset(const Dataset* dataset)
{
    corpus_dataset = dataset;
    //loadCorpusEmbedding(NET_IN_SIZE,NET_PIX_STRIDE);
    string name = saveName+"_corpus_embedding_"+dataset->getName()+"_w"+to_string(NET_IN_SIZE)+"_s"+to_string(NET_PIX_STRIDE)+".dat";

    ifstream in(name);
    if (in)
    {
        int numWordsRead;
        in >> numWordsRead;
        corpus_embedded.resize(numWordsRead);
        corpus_scalars.resize(numWordsRead);
        for (int i=0; i<numWordsRead; i++)
        {
            corpus_embedded.at(i) = readFloatMat(in);
            in >> corpus_scalars[i];
        }
        in.close();
        assert(corpus_dataset->size() == numWordsRead);

    }
    else
    {
        cout<<"Creating embedding for "<<corpus_dataset->getName()<<" (w:"<<NET_IN_SIZE<<" s:"<<NET_PIX_STRIDE<<")"<<endl;

        corpus_embedded.resize(corpus_dataset->size());
        corpus_scalars.resize(corpus_dataset->size());
        for (int i=0; i<corpus_dataset->size(); i++)
        {
            Mat im = corpus_dataset->image(i);
            corpus_scalars[i] = (0.0+NET_IN_SIZE)/im.rows;
            Mat resized;
            resize(im,resized,Size(max((int)(corpus_scalars[i]*im.cols),(int)NET_IN_SIZE),NET_IN_SIZE));
            corpus_embedded.at(i) = embedder->embed(resized);
        }
        ofstream out(name);
        out << corpus_dataset->size() << " ";
        for (int i=0; i<corpus_dataset->size(); i++)
        {
            writeFloatMat(out,corpus_embedded.at(i));
            out << corpus_scalars[i] << " ";
        }
        out.close();
    }
}

void CNNSpotter::writeFloatMat(ofstream& dst, const Mat& m)
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

Mat CNNSpotter::readFloatMat(ifstream& src)
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

