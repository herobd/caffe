#include "BBPredictor.h"


vector<cv::Rect> BBPredictor::predict(const cv::Mat& spottingRes)
{
    vector<cv::Rect> ret;
    cv::Mat threshed, ccs,stats,cent;
    cv::threshold(spottingRes,threshed,0,255,CV_8U);
    int count = cv::connectedComponentsWithStats(threshed,ccs,stats,cent,8,CV_32S);
    for (int i=1; i<=count; i++)
    {
        int x1 = stats.at<int>(i,cv::CC_STAT_LEFT);
        int y1 = stats.at<int>(i,cv::CC_STAT_TOP);
        int w = stats.at<int>(i,cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i,cv::CC_STAT_HEIGHT);
        if (w>MIN_SIZE && h>MIN_SIZE)
            ret.emplace_back(x1,y1,w,h);
    }
    return ret;
}
