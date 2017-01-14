#include "cnnspotter.h"
#include "cnnspotter_eval.cpp"

CNNSpotter::CNNSpotter(string netModel, string netWeights, int netInputSize, int netPixelStride, string saveName)
{
    this->saveName = saveName;
    embedder = new CNNEmbedder(netModel,netWeights);
    NET_IN_SIZE=netInputSize;
#if HALF_STRIDE
    NET_PIX_STRIDE=netPixelStride/4;
#else
    NET_PIX_STRIDE=netPixelStride/2;//we strech the images horz
#endif
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
    //resize(exemplar(Rect(2,0,exemplar.cols-2,exemplar.rows)), fitted, Size(NET_IN_SIZE, NET_IN_SIZE));
    //Mat exemplarEmbedding2 = embedder->embed(fitted);
    //resize(exemplar(Rect(4,0,exemplar.cols-4,exemplar.rows)), fitted, Size(NET_IN_SIZE, NET_IN_SIZE));
    //Mat exemplarEmbedding3 = embedder->embed(fitted);
    
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
        int diff = (NET_IN_SIZE*.8)/NET_PIX_STRIDE;
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

SubwordSpottingResult CNNSpotter::refine(float score, int imIdx, int windIdx, const Mat& exemplarEmbedding) const
{
    //TODO
    float bestScore=score;
    int newX0 = windIdx*NET_PIX_STRIDE/corpus_scalars[imIdx];
    int newX1 = (windIdx*NET_PIX_STRIDE+NET_IN_SIZE)/corpus_scalars[imIdx] - 1;
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
    return SubwordSpottingResult(imIdx,bestScore,newX0,newX1);
}


void CNNSpotter::setCorpus_dataset(const Dataset* dataset)
{
    corpus_dataset = dataset;
    //loadCorpusEmbedding(NET_IN_SIZE,NET_PIX_STRIDE);
#if FIXED_CORPUS
    string name = saveName+"_corpus_embedding_FIXED_"+dataset->getName()+"_w"+to_string(NET_IN_SIZE)+"_s"+to_string(NET_PIX_STRIDE)+".dat";
#else
    string name = saveName+"_corpus_embedding_"+dataset->getName()+"_w"+to_string(NET_IN_SIZE)+"_s"+to_string(NET_PIX_STRIDE)+".dat";
#endif
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
#if FIXED_CORPUS
            resize(im,resized,Size(NET_IN_SIZE,NET_IN_SIZE));
#else
            //We make it twice as wide horizontally
            resize(im,resized,Size(max((int)(corpus_scalars[i]*im.cols*2.0),(int)NET_IN_SIZE),NET_IN_SIZE));
#endif
            //corpus_embedded.at(i) = embedder->embed(resized);
            Mat a = embedder->embed(resized);
            ////
            /*
            float maxV=0;
            float minV=0;
            for (int r=0; r<a.rows; r++)
                for (int c=0; c<a.cols; c++)
                {
                    float v = a.at<float>(r,c);
                    if (v>maxV)
                        maxV=v;
                    if (v<minV)
                        minV=v;
                }
            Mat disp(a.rows,a.cols,CV_8UC3);
            for (int r=0; r<a.rows; r++)
                for (int c=0; c<a.cols; c++)
                {
                    float v = a.at<float>(r,c);
                    Vec3b color(0,0,0);
                    if (v>0)
                    {
                        color[0] = 255*v/maxV;
                    }
                    else if (v<0)
                    {
                        color[1] = 255*v/minV;
                    }
                    disp.at<Vec3b>(r,c)=color;
                    //disp.at<Vec3b>(1+2*r,2*c)=color;
                    //disp.at<Vec3b>(2*r,1+2*c)=color;
                    //disp.at<Vec3b>(1+2*r,1+2*c)=color;
                }
            imshow("embedded",disp);
            waitKey();
            */
            ////
#if HALF_STRIDE
            //cout<<"in: "<<resized.cols<<"  out: "<<a.cols<<endl;
            if (resized.cols>=NET_PIX_STRIDE+NET_IN_SIZE)
            {
                //This does an off-stride computation, maing our stride half of the networks
                resized = resized(Rect(NET_PIX_STRIDE,0,resized.cols-NET_PIX_STRIDE,NET_IN_SIZE));
                Mat b = embedder->embed(resized);
                assert(a.cols==b.cols || a.cols==b.cols+1);
                assert(a.rows==b.rows);
                corpus_embedded.at(i) = Mat(a.rows,a.cols+b.cols,CV_32F);
                for (int c=0; c<a.cols; c++)
                {
                    //corpus_embedded.at(i).col(c*2)=a.col(c);
                    a.col(c).copyTo(corpus_embedded.at(i).col(c*2));
                    if (c<b.cols)
                    {
                        //corpus_embedded.at(i).col(1+ c*2)=b.col(c);
                        b.col(c).copyTo(corpus_embedded.at(i).col(1+ c*2));
                    }
                }
                ///
                for (int r=0; r<corpus_embedded.at(i).rows; r++)
                    for (int c=0; c<corpus_embedded.at(i).cols; c++)
                        assert(corpus_embedded.at(i).at<float>(r,c)==corpus_embedded.at(i).at<float>(r,c));
                ///
            }
            else
            {
                //assert(a.cols==1);
                corpus_embedded.at(i)=a.col(0);
            }
#else
            corpus_embedded.at(i)=a;
#endif

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

