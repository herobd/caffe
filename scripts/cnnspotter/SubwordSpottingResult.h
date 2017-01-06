#ifndef SUB_SPOT_RES_H
#define SUB_SPOT_RES_H
struct SubwordSpottingResult {
    int imIdx;
    float score;
    int startX;
    int endX;
    SubwordSpottingResult(int imIdx, float score, int startX, int endX) : 
        imIdx(imIdx), score(score), startX(startX), endX(endX)
    {
    }
    SubwordSpottingResult() : 
        imIdx(-1), score(0), startX(-1), endX(-1)
    {
    }
};
#endif
