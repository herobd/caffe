#include "BBPredictor.h"

#include <iostream>

map<float,cv::Rect> BBPredictor::predict(const cv::Mat& spottingRes)
{
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
    map<float,cv::Rect> ret;
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
        if (w>MIN_SIZE && h>MIN_SIZE)
        {
            cv::Rect bb(x1,y1,w,h);
            float score = cv::sum(spottingRes(bb))[0]/bb.area();
            ret[score]= bb;
        }
    }
    return ret;
}
