#ifndef BBPREDICTOR_H
#define BBPREDICTOR_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <map>
#include <deque>

using namespace std;

class BBPredictor
{
    public:
    BBPredictor() : MIN_SIZE(5)
    {
    }

    multimap<float,cv::Rect> predict(const cv::Mat& spottingRes, cv::Size querySize=cv::Size(0,0), float sizeThresh=0.25);
    multimap<float,cv::Rect> predictHardScale(const cv::Mat& spottingRes, cv::Size querySize, vector<float> scales);

    private:
    int MIN_SIZE;
};

#endif
