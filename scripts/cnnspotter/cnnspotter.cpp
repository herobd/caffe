#include "cnnspotter.h"

//With a GPU, we could efficiently batch multiple exemplars together
vector< SubwordSpottingResult > CNNSpotter::subwordSpot(const Mat& exemplar, float refinePortion=0.25) const
{
    Mat fitted;
    resize(exemplar, fitted, Size(NET_IN_SIZE, NET_IN_SIZE));
    Mat embedding = embedder->embed(fitted);
    
    multimap<float,pair<int,int> > scores;



    //ttt#pragma omp parallel for
    for (int i=0; i<corpus_dataset->size(); i++)
    {
        Mat s_batch;
        cv::sqrt(subwordWindows_saved(i,NET_IN_SIZE,NET_PIX_STRIDE) - embedding, s_batch);
        //int im_width = corpus_dataset->image(i).cols;
        //assert(s_batch.rows<=im_width);
        float topScoreInd=-1;
        float topScore=-999999;
        float top2ScoreInd=-1;
        float top2Score=-999999;
        for (int r=0; r<s_batch.rows; r++) {
            //if ((r)*NET_PIX_STRIDE >= im_width) {
            //    cout<<"ERROR: sliding window moving out of bounds for iamge "<<i<<". Window starts at "<<(r)*NET_PIX_STRIDE<<", but image is only "<<im_width<<" wide"<<endl;
            //}
            //assert((r)*NET_PIX_STRIDE<im_width);
            float s = s_batch.at<float>(r,0);
            //cout <<"im "<<i<<" x: "<<r*stride<<" score: "<<s<<endl;
            if (s>topScore)
            {
                topScore=s;
                topScoreInd=r;
            }
        }
        int diff = (NET_IN_SIZE*.8)/NET_PIX_STRIDE;
        for (int r=0; r<s_batch.rows; r++) {
            float s = s_batch.at<float>(r,0);
            if (s>top2Score && abs(r-topScoreInd)>diff)
            {
                top2Score=s;
                top2ScoreInd=r;
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
        finalScores.at(i) = refine(iter->first,iter->second.first,iter->second.second,NET_IN_SIZE,NET_PIX_STRIDE,query_cca_hy);
    }

    return finalScores;
 
}

SubwordSpottingResult CNNSpotter::refine(float score, int imIdx, int windIdx, int s_windowWidth, int NET_PIX_STRIDE, const Mat& embedding) const
{
    //TODO
    float bestScore=score;
    int newX0 = windIdx*NET_PIX_STRIDE/corpus_scalers[imIdx];
    int newX1 = (windIdx+s_windowWidth)/corpus_scalers[imIdx] - 1;
    return SubwordSpottingResult(imIdx,bestScore,newX0,newX1);
}
