#ifndef BBPREDICTOR_H
#define BBPREDICTOR_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <map>

using namespace std;

class BBPredictor
{
    public:
    BBPredictor() : MIN_SIZE(5)
    {
    }

    map<float,cv::Rect> predict(const cv::Mat& spottingRes);

    private:
    int MIN_SIZE;
};

#endif
