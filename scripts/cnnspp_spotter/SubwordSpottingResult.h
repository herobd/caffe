#ifndef SUB_SPOT_RES_H
#define SUB_SPOT_RES_H
struct SubwordSpottingResult {
    int imIdx;
    float score;
    int startX;
    int endX;
    int gt;
    SubwordSpottingResult(int imIdx, float score, int startX, int endX) : 
        imIdx(imIdx), score(score), startX(startX), endX(endX), gt(-10)
    {
    }
    SubwordSpottingResult() : 
        imIdx(-1), score(0), startX(-1), endX(-1), gt(-10)
    {
    }
};

struct SpottingLoc {
    int imIdx;
    int startX;
    int endX;
    unsigned long id;
    map<string,float> scores;
    SpottingLoc(const SubwordSpottingResult& r, string ngram, unsigned long id) : imIdx(r.imIdx), startX(r.startX), endX(r.endX), id(id)
    {
        scores[ngram]=r.score;
    }
};
#endif
