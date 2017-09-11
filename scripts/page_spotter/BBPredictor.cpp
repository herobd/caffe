#include "BBPredictor.h"

#include <iostream>

multimap<float,cv::Rect> BBPredictor::predict(const cv::Mat& spottingRes, cv::Size querySize, float sizeThresh)
{
    int minH=querySize.height * (1-sizeThresh);
    int maxH=querySize.height * (1+sizeThresh);
    int minW=querySize.width * (1-sizeThresh);
    int maxW=querySize.width * (1+sizeThresh);

    int kernel_size=5; 
    cv::Mat kernel = cv::Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);
    cv::Mat conv;
    cv::filter2D(spottingRes, conv, -1 , kernel);
    cv::Point loc;
    double minPeak, maxPeak;
    cv::minMaxLoc(conv,&minPeak,&maxPeak,NULL,&loc);

    double mean = cv::mean(conv)[0];
    //double minV, maxV;
    //cv::minMaxLoc(spottingRes,&minV,&maxV);
    multimap<float,cv::Rect> ret;
    cv::Mat threshed, ccs,stats,cent;
    double thresh = (mean+maxPeak)/2;
    cv::threshold(conv,threshed,max(0.0,thresh),255,cv::THRESH_BINARY);
    threshed.convertTo(threshed, CV_8U);
    int count = cv::connectedComponentsWithStats(threshed,ccs,stats,cent,8,CV_32S);
    for (int i=1; i<count; i++)
    {
    //int i = ccs.at<int>(loc);
        int x1 = stats.at<int>(i,cv::CC_STAT_LEFT);
        int y1 = stats.at<int>(i,cv::CC_STAT_TOP);
        int w = stats.at<int>(i,cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i,cv::CC_STAT_HEIGHT);
        if (x1+w>spottingRes.cols || y1+h>spottingRes.rows)
        {
            cout<<"Error: bad CC returned by OpenCV, "<<x1<<" "<<y1<<" "<<w<<" "<<h<<endl;
            return ret;//continue;
        }
        if (w>MIN_SIZE && h>MIN_SIZE && 
                (maxH==0 || (w>minW && w<maxW && h>minH && h<maxH)))
        {
            cv::Rect bb(x1,y1,w,h);
            float score = cv::sum(spottingRes(bb))[0]/bb.area();
            ret.emplace(score,bb);
        }
    }
    return ret;
}


multimap<float,cv::Rect> BBPredictor::predictHardScale(const cv::Mat& spottingRes, cv::Size querySize, vector<float> scales)
{
    

    multimap<float,cv::Rect> ret;
    for (float scale : scales)
    {
        cv::Size kernel_size(querySize.width*scale, querySize.height*scale);
        cv::Mat kernel = cv::Mat::ones( kernel_size, CV_32F )/ (float)(kernel_size.height*kernel_size.width);
        cv::Mat conv;
        //sums the window of the kernel size
        cv::filter2D(spottingRes, conv, -1 , kernel);


        //Find local maximum values for the local of the kernel size
        double mean = cv::mean(conv)[0];
        double maxV;
        for (int r=0; r<conv.rows; r++)
        {
            int windowR=max(0,r-(kernel_size.height/2 + kernel_size.height%2));
            int windowH=kernel_size.height + min(0,r-(kernel_size.height/2 + kernel_size.height%2)); //add a negative
            if (windowH+windowR>conv.rows)
                windowH+=conv.rows-(windowH+windowR); //add a negative
            float curMax=-9999;
            //float curMaxC=-1;
            deque<float> maxes;
            for (int c=0; c<kernel_size.width/2 + kernel_size.width%2; c++)
            {
                cv::minMaxLoc(conv(cv::Rect(c,windowR,1,windowH)),NULL,&maxV);
                maxes.push_back(maxV);
                if (maxV>curMax)
                {
                    curMax=maxV;
                    //curMaxC=c;
                }
            }
            for (int c=0; c<conv.cols; c++)
            {
                int windowAhead = kernel_size.width/2 + kernel_size.width%2;
                if (c<conv.cols-windowAhead)
                {
                    cv::minMaxLoc(conv(cv::Rect(c+windowAhead,windowR,1,windowH)),NULL,&maxV);
                    maxes.push_back(maxV);
                    if (maxV>curMax)
                    {
                        curMax=maxV;
                        //curMaxC=c+windowAhead;
                    }
                }
                if (maxes.size()>kernel_size.width)
                {
                    if (maxes.front()==curMax)
                    {//get new max
                        curMax=maxes[1];
                        for (int i=2; i<maxes.size(); i++)
                        {
                            if (maxes[i]>curMax)
                                curMax=maxes[i];
                        }
                    }
                    maxes.pop_front();
                }

                if (conv.at<float>(r,c)==curMax && curMax>mean)
                {
                    int windowC = max(0,c-(kernel_size.width/2+1));
                    int windowW = kernel_size.width + min(0,c-(kernel_size.width/2+1));
                    if (windowC+windowW>conv.cols)
                        windowW+=conv.cols-(windowC+windowW);
                    ret.emplace(curMax,cv::Rect(windowC,windowR,windowW,windowH));
                }
            }
        }

    }
    //TODO something should be done to rectify different sizes. Definitely take care of overlap and indeally make it so smaller regions aren't favored.
    return ret;
}
//*/
