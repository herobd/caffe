#include "cnnspp_spotter.h"
#include <set>
#include <stdlib.h>

#ifdef SAVE_IMAGES
#include <sys/stat.h>
#include <sys/types.h>
#endif

#define PAD_EXE 9
#define END_PAD_EXE 3

int sort_xxx(const void *x, const void *y) {
    if (*(int*)x > *(int*)y) return 1;
    else if (*(int*)x < *(int*)y) return -1;
    else return 0;
}

/*void CNNSPPSpotter::eval(const Dataset* data)
{
    setCorpus_dataset(data);
    for (double hy=0.0; hy<=1.0; hy+=0.1)
    {
        
        float map=0;
        int queryCount=0;
        #pragma omp parallel  for
        for (int inst=0; inst<data->size(); inst++)
        {
            int other=0;
            string text = data->labels()[inst];
            for (int inst2=0; inst2<data->size(); inst2++)
            {
                if (inst!=inst2 && text.compare(data->labels()[inst2])==0)
                {
                    other++;
                }
            }
            if (other==0)
                continue;
            
            int *rank = new int[other];//(int*)malloc(NRelevantsPerQuery[i]*sizeof(int));
            int Nrelevants = 0;
            float ap=0;
            
            float bestS=-99999;
            vector<float> scores = spot(data->image(inst),text,hy); //scores
            for (int j=0; j < data->size(); j++)
            {            
                float s = scores[j];
                //cout <<"score for "<<j<<" is "<<s<<". It is ["<<data->labels()[j]<<"], we are looking for ["<<text<<"]"<<endl;
                // Precision at 1 part 
                //if (inst!=j && s > bestS)
               //
                //  bestS = s;
                //  p1 = text==data->labels()[j];
                    //bestIdx[inst] = j;
                //}
                // If it is from the same class and it is not the query idx, it is a relevant one. 
                // Compute how many on the dataset get a better score and how many get an equal one, excluding itself and the query.
                if (text.compare(data->labels()[j])==0 && inst!=j)
                {
                    int better=0;
                    int equal = 0;
                    
                    for (int k=0; k < data->size(); k++)
                    {
                        if (k!=j && inst!=k)
                        {
                            float s2 = scores[k];
                            if (s2> s) better++;
                            else if (s2==s) equal++;
                        }
                    }
                    
                    
                    rank[Nrelevants]=better+floor(equal/2.0);
                    Nrelevants++;
                }
                
            }
            qsort(rank, Nrelevants, sizeof(int), sort_xxx);
            
            //pP1[i] = p1;
            
            Y// Get mAP and store it 

                
                float prec_at_k =  ((float)(j+1))/(rank[j]+1);
                //mexPrintf("prec_at_k: %f\n", prec_at_k);
                ap += prec_at_k;            
            }
            ap/=Nrelevants;
            
            #pragma omp critical (storeMAP)
            {
                queryCount++;
                map+=ap;
            }
            
            delete[] rank;
        }
        
        cout<<"map: "<<(map/queryCount)<<" for "<<hy<<endl;
    }
}*/

#define RAND_PROB (static_cast <float> (rand()) / static_cast <float> (RAND_MAX))
void CNNSPPSpotter::helpAP(vector<SubwordSpottingResult>& res, string ngram, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, float goalAP)
{
    vector<int> notSpottedIn;
    float currentAP = evalSubwordSpotting_singleScore(ngram, res, corpusXLetterStartBounds,corpusXLetterEndBounds,-1,NULL,NULL,&notSpottedIn);
    cout<<"help["<<ngram<<"], init AP: "<<currentAP<<endl;
    while (currentAP < goalAP)
    {
        //swap lowest score false and highest score true
        float minScore=999999;
        float maxScore=-999999;
        int minFalse;
        float minFalseScore=999999;
        int maxTrue;
        float maxTrueScore=-999999;
        for (int i=0; i<res.size(); i++)
        {
            if (res[i].gt!=-10)
            {
                if (res[i].gt==1 && res[i].score>maxTrueScore)
                {
                    maxTrue=i;
                    maxTrueScore=res[i].score;
                }
                if (res[i].score<minScore)
                {
                    minScore=res[i].score;
                }
                if (res[i].gt!=1 && res[i].score < minFalseScore)
                {
                    minFalse=i;
                    minFalseScore=res[i].score;
                }
                if (res[i].score > maxScore)
                {
                    maxScore=res[i].score;
                }
            }
        }
        //res[minFalse].score=maxTrueScore;
        //res[maxTrue].score=minFalseScore;
        uniform_real_distribution<float> newTrueDist(minScore,maxTrueScore);
        uniform_real_distribution<float> newFalseDist(minFalseScore,maxScore);
        res[minFalse].score = newFalseDist(generator);
        if (RAND_PROB < 0.5 && notSpottedIn.size()>0)
        {
            //add missed spotting
            int wordId = notSpottedIn[rand()%notSpottedIn.size()];
            size_t loc = corpus_dataset->labels()[wordId].find(ngram);
            SubwordSpottingResult newResult(wordId, newTrueDist(generator), corpusXLetterStartBounds->at(wordId)[loc], corpusXLetterEndBounds->at(wordId)[loc+ngram.length()-1]);
            res.push_back(newResult);
        }
        else
        {
            res[maxTrue].score = newTrueDist(generator);
        }


        currentAP = evalSubwordSpotting_singleScore(ngram, res, corpusXLetterStartBounds,corpusXLetterEndBounds,-1,NULL,NULL,&notSpottedIn);
        cout<<"help["<<ngram<<"], new  AP: "<<currentAP<<endl;
    }
            
}

//This is a testing function for the simulator
#define LIVE_SCORE_OVERLAP_THRESH .2//0.65
float CNNSPPSpotter::evalSubwordSpotting_singleScore(string ngram, vector<SubwordSpottingResult>& res, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, int skip, multimap<float,int>* trues, multimap<float,int>* alls, vector<int>* notSpottedIn)
{

    if (trues!=NULL)
        trues->clear();
    if (alls!=NULL)
        alls->clear();
    if (notSpottedIn!=NULL)
        notSpottedIn->clear();
    //string ngram = exemplars->labels()[inst];
    int Nrelevants = 0;
    float ap=0;
    
    float bestS=-99999;
    //vector<SubwordSpottingResult> res = subwordSpot(exemplars->image(inst),ngram,hy); //scores
    float maxScore=-9999;
    for (auto r : res)
        if (r.score>maxScore)
            maxScore=r.score;
    vector<float> scores;
    vector<bool> rel;
    vector<int> indexes;
    vector<bool> checked(corpus_dataset->size());
    int l=ngram.length()-1;
    for (int j=0; j<res.size(); j++)
    {
        SubwordSpottingResult& r = res[j];
        if (alls!=NULL)
            alls->emplace(r.score,j);
        checked.at(r.imIdx)=true;
        if (skip == r.imIdx)
        {
            //cout <<"skipped "<<j<<endl;
            continue;
        }
        if (r.gt==1)
        {
            scores.push_back(r.score);
            rel.push_back(true);
            indexes.push_back(j);
        }
        else if (r.gt==0)
        {
            scores.push_back(r.score);
            rel.push_back(false);
            indexes.push_back(j);
        }
        else if (r.gt==-1)
        {
            scores.push_back(r.score);
            rel.push_back(false);
            indexes.push_back(j);

            scores.push_back(maxScore);
            rel.push_back(true);
            indexes.push_back(-1);
            if (notSpottedIn!=NULL)
                notSpottedIn->push_back(j);
        }
        else
        {

            size_t loc = corpus_dataset->labels()[r.imIdx].find(ngram);
            if (loc==string::npos)
            {
                scores.push_back(r.score);
                rel.push_back(false);
                indexes.push_back(j);
                r.gt=0;
            }
            else
            {
                vector<int> matching;
                for (int jj=0; jj < res.size(); jj++)
                {
                    if (res[jj].imIdx == r.imIdx && j!=jj && res[jj].imIdx!=skip)
                        matching.push_back(jj);
                }
                float myOverlap = ( min(corpusXLetterEndBounds->at(r.imIdx)[loc+l], r.endX) 
                                    - max(corpusXLetterStartBounds->at(r.imIdx)[loc], r.startX) ) 
                                  /
                                  ( max(corpusXLetterEndBounds->at(r.imIdx)[loc+l], r.endX) 
                                    - min(corpusXLetterStartBounds->at(r.imIdx)[loc], r.startX) +0.0);
                
                if (matching.size()>0)
                {
                    //float relPos = (loc+(ngram.length()/2.0))/corpus_dataset->labels()[r.imIdx].length();
                    //float myDif = fabs(relPos - (r.startX + (r.endX-r.startX)/2.0)/corpus_dataset->image(r.imIdx).cols);
                    bool other=false;
                    for (int oi : matching)
                    {
                        float otherOverlap = ( min(corpusXLetterEndBounds->at(res[oi].imIdx)[loc+l], res[oi].endX) 
                                                - max(corpusXLetterStartBounds->at(res[oi].imIdx)[loc], res[oi].startX) ) 
                                              /
                                              ( max(corpusXLetterEndBounds->at(res[oi].imIdx)[loc+l], res[oi].endX) 
                                                - min(corpusXLetterStartBounds->at(res[oi].imIdx)[loc], res[oi].startX) +0.0);
                        if (otherOverlap > myOverlap) {
                            other=true;
                            break;
                        }
                    }
                    if (other)
                    {
                        scores.push_back(r.score);
                        rel.push_back(false);
                        indexes.push_back(j);
                        r.gt=0;
                    }
                    else if (myOverlap > LIVE_SCORE_OVERLAP_THRESH)
                    {
                        scores.push_back(r.score);
                        rel.push_back(true);
                        indexes.push_back(j);
                        r.gt=1;
                    }
                }
                else
                {
                    /*bool ngram1H = loc+(ngram.length()/2.0) < 0.8*corpus_dataset->labels()[r.imIdx].length()/2.0;
                    bool ngram2H = loc+(ngram.length()/2.0) > 1.2*corpus_dataset->labels()[r.imIdx].length()/2.0;
                    bool ngramM = loc+(ngram.length()/2.0) > corpus_dataset->labels()[r.imIdx].length()/3.0 &&
                        loc+(ngram.length()/2.0) < 2.0*corpus_dataset->labels()[r.imIdx].length()/3.0;
                    float sLoc = r.startX + (r.endX-r.startX)/2.0;
                    bool spot1H = sLoc < 0.8*corpus_dataset->image(r.imIdx).cols/2.0;
                    bool spot2H = sLoc > 1.2*corpus_dataset->image(r.imIdx).cols/2.0;
                    bool spotM = sLoc > corpus_dataset->image(r.imIdx).cols/3.0 &&
                        sLoc < 2.0*corpus_dataset->image(r.imIdx).cols/3.0;
                        */

                    if (myOverlap > LIVE_SCORE_OVERLAP_THRESH)
                    //if ( (ngram1H&&spot1H) || (ngram2H&&spot2H) || (ngramM&&spotM) )
                    {
                        scores.push_back(r.score);
                        rel.push_back(true);
                        indexes.push_back(j);
                        r.gt=1;
                    }
                    else
                    {
                        ////
                        //cout<<"bad overlap["<<j<<"]: "<<myOverlap<<endl;
                        ////
                        scores.push_back(r.score);
                        rel.push_back(false);
                        indexes.push_back(j);
                        r.gt=-1;
                        //Insert a dummy result for the correct spotting to keep MAP accurate
                        scores.push_back(maxScore);
                        rel.push_back(true);
                        indexes.push_back(-1);
                    }

                }
            }
        }
    }
    for (int j=0; j<corpus_dataset->size(); j++)
    {
        if (!checked[j] &&  corpus_dataset->labels()[j].find(ngram)!=string::npos)
        {
            scores.push_back(maxScore);
            rel.push_back(true);
            indexes.push_back(-1);
            if (notSpottedIn!=NULL)
                notSpottedIn->push_back(j);
        }
    }
    ////
    //cout<<"r:"<<rel[0]<<":"<<scores[0]<<" "<<rel[1]<<":"<<scores[1]<<" "<<rel[2]<<":"<<scores[2]<<" "<<rel[3]<<":"<<scores[3]<<endl;
    ////
    vector<int> rank;
    for (int j=0; j < scores.size(); j++)
    {            
        float s = scores[j];
        //cout <<"score for "<<j<<" is "<<s<<". It is ["<<data->labels()[j]<<"], we are looking for ["<<text<<"]"<<endl;
        
        if (rel[j])
        {
            int better=0;
            int equal = 0;
            
            for (int k=0; k < scores.size(); k++)
            {
                if (k!=j)
                {
                    float s2 = scores[k];
                    if (s2< s) better++;
                    else if (s2==s) equal++;
                }
            }
            
            
            rank.push_back(better+floor(equal/2.0));
            Nrelevants++;

            ////
            //if (j<5)
            //    cout<<rank.back()<<"  ";
            ////
            if (trues!=NULL && indexes.at(j)!=-1)
                trues->emplace(s,indexes[j]);
        }
        
    }

    ////
    //cout<<endl;
    ////
    qsort(rank.data(), Nrelevants, sizeof(int), sort_xxx);
    
    //pP1[i] = p1;
    
    /* Get mAP and store it */
    for(int j=0;j<Nrelevants;j++){
        /* if rank[i] >=k it was not on the topk. Since they are sorted, that means bail out already */
        
        float prec_at_k =  ((float)(j+1))/(rank[j]+1);
        ///
        //cout<<"prec at "<<j<<": "<<prec_at_k<<endl;
        ///
        ap += prec_at_k;            
    }
    ap/=Nrelevants;
    
   return ap;
}


float CNNSPPSpotter::evalWordSpotting_singleScore(string word, const multimap<float,int>& res, int skip, multimap<float,int>* trues)
{
    if (trues!=NULL)
        trues->clear();
    //string word = exemplars->labels()[inst];
    int Nrelevants = 0;
    float ap=0;
    
    float bestS=-99999;
    //vector<SubwordSpottingResult> res = subwordSpot(exemplars->image(inst),word,hy); //scores
    vector<float> scores;
    vector<bool> rel;
    for (auto r : res)
    {
        bool same = corpus_dataset->labels()[r.second].compare(word)==0;
        scores.push_back(r.first);
        rel.push_back(same);
        if (trues!=NULL && same)
            trues->insert(r);
    }
    ////
    //cout<<"r:"<<rel[0]<<":"<<scores[0]<<" "<<rel[1]<<":"<<scores[1]<<" "<<rel[2]<<":"<<scores[2]<<" "<<rel[3]<<":"<<scores[3]<<endl;
    ////
    vector<int> rank;
    for (int j=0; j < scores.size(); j++)
    {            
        float s = scores[j];
        //cout <<"score for "<<j<<" is "<<s<<". It is ["<<data->labels()[j]<<"], we are looking for ["<<text<<"]"<<endl;
        
        if (rel[j])
        {
            int better=0;
            int equal = 0;
            
            for (int k=0; k < scores.size(); k++)
            {
                if (k!=j)
                {
                    float s2 = scores[k];
                    if (s2< s) better++;
                    else if (s2==s) equal++;
                }
            }
            
            
            rank.push_back(better+floor(equal/2.0));
            Nrelevants++;

            ////
            //if (j<5)
            //    cout<<rank.back()<<"  ";
            ////
        }
        
    }

    ////
    //cout<<endl;
    ////
    qsort(rank.data(), Nrelevants, sizeof(int), sort_xxx);
    
    //pP1[i] = p1;
    
    /* Get mAP and store it */
    for(int j=0;j<Nrelevants;j++){
        /* if rank[i] >=k it was not on the topk. Since they are sorted, that means bail out already */
        
        float prec_at_k =  ((float)(j+1))/(rank[j]+1);
        ///
        //cout<<"prec at "<<j<<": "<<prec_at_k<<endl;
        ///
        ap += prec_at_k;            
    }
    ap/=Nrelevants;
    
   return ap;
}

void CNNSPPSpotter::evalSubwordSpottingWithCharBounds(const Dataset* data, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds)
{
    setCorpus_dataset(data,false);


    set<string> done;
    float mAP=0;
    int queryCount=0;
    map<string,int> ngramCounter;
    map<string,float> ngramAPs;

    cout<<"---QbE---"<<endl;
    //#pragma omp parallel for
    for (int inst=0; inst<data->size(); inst++)
    {
        string label = data->labels()[inst];
        if (label.length()<2)
            continue;
        int bigram= inst%(label.length()-1);
        string ngram="";
        do {
            ngram = label.substr(bigram,2);
            int nRel=0;
            for (int inst2=0; inst2<data->size(); inst2++)
            {
                if (data->labels()[inst2].find(ngram) != string::npos)
                    nRel++;
            }
            if (nRel<30)
            {
                bigram= (bigram+1)%(label.length()-1);
                ngram="";
            }
        } while(ngram.length()==0 && bigram!=inst%(label.length()-1));
        if (ngram.length()==0)
            continue;

        Mat exemplar;
        Mat wordIm = data->image(inst);
        int x1 = max(0,corpusXLetterStartBounds->at(inst)[bigram] - (bigram==0?END_PAD_EXE:PAD_EXE));
        int x2 = min(wordIm.cols-1,corpusXLetterEndBounds->at(inst)[bigram+1] + (bigram==label.length()-2?END_PAD_EXE:PAD_EXE));
        //double scalar = NET_IN_SIZE / (0.0+wordIm.rows);
        //we use half the height for our width
        int newX1 = max(0.0,(x2+x1)/2.0 - wordIm.rows/4.0);
        int leftOnRight = wordIm.cols-(newX1 + wordIm.rows/2);
        int newX2;
        if (leftOnRight<0)
        {
            newX1 += leftOnRight;
            newX2 = wordIm.cols-1;
        }
        else
        {
            newX2=newX1+wordIm.rows/2 -1;
        }
        if (newX1<0)
            newX1=0;
           
        if (newX1<0 || newX2>=wordIm.cols || newX2<newX1)
           cout<<"Error wordIm w:"<< wordIm.cols<<"  x1:"<<x1<<" x2:"<<x2<<"  newx1:"<<newX1<<" newx2:"<<newX2<<endl;
        //This crops a square region so no distortion happens.
        exemplar = wordIm(Rect(newX1,0,newX2-newX1+1,wordIm.rows));
        float ap=0;

        vector<SubwordSpottingResult> res = subwordSpot(ngram.length(),exemplar); //scores
        ////
        /*
        imshow("exe", exemplar);
        cout<<"exemplar: "<<ngram<<endl;
        cout<<data->labels()[res[0].imIdx]<<":"<<res[0].score<<"  "<<data->labels()[res[1].imIdx]<<":"<<res[1].score<<"  "<<data->labels()[res[2].imIdx]<<":"<<res[2].score<<"  "<<data->labels()[res[3].imIdx]<<":"<<res[3].score<<endl;
        if (res[0].startX<0 || res[0].endX>=data->image(res[0].imIdx).cols || res[0].endX<=res[0].startX)
            cout<<"ERROR[0]  image w:"<<data->image(res[0].imIdx).cols<<"  s:"<<res[0].startX<<" e:"<<res[0].endX<<endl;
        if (res[1].startX<0 || res[1].endX>=data->image(res[1].imIdx).cols || res[1].endX<=res[1].startX)
            cout<<"ERROR[1]  image w:"<<data->image(res[1].imIdx).cols<<"  s:"<<res[1].startX<<" e:"<<res[1].endX<<endl;
        if (res[2].startX<0 || res[2].endX>=data->image(res[2].imIdx).cols || res[2].endX<=res[2].startX)
            cout<<"ERROR[2]  image w:"<<data->image(res[2].imIdx).cols<<"  s:"<<res[2].startX<<" e:"<<res[2].endX<<endl;
        //cout<<"["<<data->image(res[0].imIdx).cols<<","<<data->image(res[0].imIdx).rows<<"] R "<<res[0].startX<<" 0 "<<
        Mat top1 = data->image(res[0].imIdx)(Rect(res[0].startX,0,res[0].endX-res[0].startX+1,data->image(res[0].imIdx).rows));
        imshow("top1",top1);
        Mat top2 = data->image(res[1].imIdx)(Rect(res[1].startX,0,res[1].endX-res[1].startX+1,data->image(res[1].imIdx).rows));
        imshow("top2",top2);
        Mat top3 = data->image(res[2].imIdx)(Rect(res[2].startX,0,res[2].endX-res[2].startX+1,data->image(res[2].imIdx).rows));
        imshow("top3",top3);
        Mat top4 = data->image(res[3].imIdx)(Rect(res[3].startX,0,res[3].endX-res[3].startX+1,data->image(res[3].imIdx).rows));
        imshow("top4",top4);
        waitKey();
        */
        ////

        ap = evalSubwordSpotting_singleScore(ngram, res, corpusXLetterStartBounds, corpusXLetterEndBounds, inst);
        //#pragma omp critical (storeMAP)
        {
            queryCount++;
            mAP+=ap;
            //cout<<"on spotting inst:"<<inst<<", "<<ngram<<"   ap: "<<ap<<endl;
            ngramCounter[ngram]++;
            ngramAPs[ngram]+=ap;
        }

    }
    cout<<"ngram, num inst, AP";
    vector<string>exemplars;
    for (auto p : ngramCounter)
    {
        cout<<p.first<<", "<<p.second<<",\t"<<ngramAPs[p.first]/p.second<<endl;
        exemplars.push_back(p.first);
    }
    cout<<endl;
    cout<<"FULL QbE map: "<<(mAP/queryCount)<<endl;

    cout<<"\n---QbS---\nngram, AP"<<endl;
    mAP=0;
    for (int inst=0; inst<exemplars.size(); inst++)
    {
        string ngram = exemplars[inst];
        int Nrelevants = 0;
        float ap=0;
        
        //imshow("exe", exemplars->image(inst));
        //waitKey();
        vector<SubwordSpottingResult> res = subwordSpot(exemplars[inst]); //scores
        ap = evalSubwordSpotting_singleScore(ngram, res, corpusXLetterStartBounds, corpusXLetterEndBounds,-1);
        assert(ap==ap);
        if (ap<0)
            continue;
        
        queryCount++;
        mAP+=ap;
        cout<<ngram<<", "<<ap<<endl;
    }
    cout<<endl;
    cout<<"FULL QbS map: "<<(mAP/exemplars.size())<<endl;
}

void meanAndStd(const vector<SubwordSpottingResult>& data, float *meanR, float* stdR)
{
    float mean=0;
    for (auto s : data)
        mean+=s.score;
    mean/=data.size();
    float std=0;
    for (auto s : data)
        std+=pow(mean-s.score,2);
    std = sqrt(std/data.size());

    *meanR=mean;
    *stdR=std;
}
void meanAndStd(const multimap<float,int>& data, float *meanR, float* stdR)
{
    float mean=0;
    for (auto s : data)
        mean+=s.first;
    mean/=data.size();
    float std=0;
    for (auto s : data)
        std+=pow(mean-s.first,2);
    std = sqrt(std/data.size());

    *meanR=mean;
    *stdR=std;
}
void meanAndStd(const vector<int>& data, float *meanR, float* stdR)
{
    float mean=0;
    for (int s : data)
        mean+=s;
    mean/=data.size();
    float std=0;
    for (int s : data)
        std+=pow(mean-s,2);
    std = sqrt(std/data.size());

    *meanR=mean;
    *stdR=std;
}
void meanAndStd(const vector<float>& data, float *meanR, float* stdR)
{
    float mean=0;
    for (float s : data)
        mean+=s;
    mean/=data.size();
    float std=0;
    for (float s : data)
        std+=pow(mean-s,2);
    std = sqrt(std/data.size());

    *meanR=mean;
    *stdR=std;
}


double median( cv::Mat channel )
{
    double m = (channel.rows*channel.cols) / 2;
    int bin = 0;
    double med = -1.0;

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;
    cv::Mat hist;
    cv::calcHist( &channel, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

    for ( int i = 0; i < histSize && med < 0.0; ++i )
    {
        bin += cvRound( hist.at< float >( i ) );
        if ( bin > m && med < 0.0 )
            med = i;
    }

    return med;
}


void CNNSPPSpotter::evalSubwordSpottingRespot(const Dataset* data, vector<string> toSpot, int numSteps, int numRepeat, int repeatSteps, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds)
{
    cout<<"Not accumulating QbS result."<<endl;
    setCorpus_dataset(data,false);


    set<string> done;
    float MAP_QbS=0;
    float MAP_comb=0;
    float MAP_QbE=0;
    int queryCount=0;

    map<string,int> ngramCounter;
    map<string,float> ngramAPs;

    //vector<float> stepAPSum(numSteps);
    vector< vector<float> > stepAps(numSteps);

    //unsigned char med = median(corpus_dataset->image(0));


    //cout<<"ngram,\titer,\tAP,  \tcombAP,\tshiftRatio(- good),\tbetterRatio,\tworseRatio,\tbetterRatio(inc miss),\tworseRatio(inc miss)"<<endl;
    cout<<"ngram,\titer,\tAP,  \tcombAP,\tshift mean(- good),\tshift STD,\tshiftTop mean(- good),\tshiftTop STD"<<endl;
    for (int inst=0; inst<toSpot.size(); inst++)
    {
        string ngram = toSpot[inst];
        //verify ngram presence is significant (double comb steps)
        int Nrelevants = 0;
        for (const string& word : corpus_dataset->labels())
        {
            if (word.find(ngram)!=string::npos)
            {
                if (++Nrelevants>=2*numSteps)
                    break;
            }
        }
        if (Nrelevants<2*numSteps)
        {
            cout<<ngram<<", skipping"<<endl;
            continue;
        }

        float ap=0;
        
        //imshow("exe", exemplars->image(inst));
        //waitKey();

        multimap<float,int> truesAccum;
        multimap<float,int> allsAccum;
        vector<SubwordSpottingResult> resAccum = subwordSpot(ngram); //scores
        ap = evalSubwordSpotting_singleScore(ngram, resAccum, corpusXLetterStartBounds, corpusXLetterEndBounds,-1, &truesAccum, &allsAccum);
        auto midTrue = truesAccum.begin();
        //for (int iii=0; iii<trues.size()/2; iii++)
        //    iter++;
        assert(ap==ap);
        if (ap<0)
            continue;
        
        queryCount++;
        MAP_QbS+=ap;
        cout<<ngram<<",\t[QbS],\t"<<ap<<endl;
        //stats
        float mean, std;
        //meanAndStd(res,&mean, &std);
        //cout<<mean<<", "<<std<<endl;

        
        //vector<SubwordSpottingResult> prevRes=resAccum;
        set<int> alreadySpotted;
        for (int i=1; i<=numSteps; i++)
        {
            //vector<SubwordSpottingResult> tmpRes=resAccum;
            multimap<float,int> truesN, allsN;
            vector<SubwordSpottingResult> resN;
            //for (int ii=0; ii<numRepeat&&(ii==0||i<repeatSteps+1); ii++)
            int ii=0;
            {

                midTrue = truesAccum.begin();
                for (int iii=0; iii<(truesAccum.size()/2)-ii; iii++)
                   ++midTrue;
                while (alreadySpotted.find(midTrue->second) != alreadySpotted.end() && midTrue!=truesAccum.begin())
                    --midTrue;
                if (midTrue==truesAccum.begin())
                {
                    for (int iii=0; iii<(truesAccum.size()/2)-ii; iii++)
                       ++midTrue;
                    while (alreadySpotted.find(midTrue->second) != alreadySpotted.end() && midTrue!=truesAccum.end())
                       ++midTrue;
                }


                SubwordSpottingResult next = resAccum[midTrue->second];//imIdx(-1), score(0), startX(-1), endX(-1)
                alreadySpotted.insert(midTrue->second);
                Mat wordIm = corpus_dataset->image(next.imIdx);

#ifdef FEATHER_QUERIES           
                //Crop to be square
                int newX1 = max(0.0,(next.endX+next.startX)/2.0 - wordIm.rows/2.0);
                int leftOnRight = wordIm.cols-(newX1 + wordIm.rows);
                int newX2;
                if (leftOnRight<0)
                {
                    newX1 += leftOnRight;
                    newX2 = wordIm.cols-1;
                }
                else
                {
                    newX2=newX1+wordIm.rows -1;
                }
                if (newX1<0)
                    newX1=0;
                if (newX2>=wordIm.cols)
                    newX2=wordIm.cols-1;
                   
                if (newX1<0 || newX2>=wordIm.cols || newX2<newX1)
                   cout<<"Error wordIm w:"<< wordIm.cols<<"  x1:"<<next.startX<<" x2:"<<next.endX<<"  newx1:"<<newX1<<" newx2:"<<newX2<<endl;

                //This crops a square region so no distortion happens.
                //Mat exemplar = wordIm(Rect(newX1,0,newX2-newX1+1,wordIm.rows));
                resN = subwordSpot(ngram.length(),next.imIdx,newX1,newX2,next.startX,next.endX);
#else          
                //Leave rectangular using preembedded (assumes sliding window size)
                resN = subwordSpot(ngram.length(),next.imIdx,next.startX);
#endif
                /*
                //Pad to be square
                int exWidth = next.endX-next.startX+1;
                int newDim = max((int)wordIm.rows,exWidth);
                Mat exemplar(newDim,newDim,CV_8U);
                exemplar=med;
                if (newDim==wordIm.rows)
                {
                    int offset = (newDim-(exWidth))/2;
                    wordIm(Rect(next.startX,0,exWidth,wordIm.rows)).copyTo(exemplar(Rect(offset,0,exWidth,wordIm.rows)));
                }
                else
                {
                    int offset = (newDim-wordIm.rows)/2;
                    wordIm(Rect(next.startX,0,exWidth,wordIm.rows)).copyTo(exemplar(Rect(0,offset,exWidth,wordIm.rows)));
                }
                xxx
                */

                //float apN = evalSubwordSpotting_singleScore(ngram, resN, corpusXLetterStartBounds, corpusXLetterEndBounds,-1, &truesN);
                float apN, combAP, rankRise, rankDrop, rankRiseFull, rankDropFull;
                float moveRatio;
                if (i==1)//QbS doesn't combine
                {
                    resAccum.clear();
                    truesAccum.clear();
                    allsAccum.clear();
                }
                vector<SubwordSpottingResult> prevResAccum=resAccum;
                multimap<float,int> prevTruesAccum=truesAccum;
                multimap<float,int> prevAllsAccum=allsAccum;
                _eval(ngram,resN,&resAccum,corpusXLetterStartBounds,corpusXLetterEndBounds,&apN,&combAP,&truesAccum,&allsAccum,&truesN,&allsN);
#ifdef SAVE_IMAGES
                string savePre = "./saveEx/"+ngram+"/";
                mkdir(savePre.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                savePre+=to_string(i)+"/";
                mkdir(savePre.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                //imwrite(savePre+"exemplar.png",exemplar);
#ifdef FEATHER_QUERIES
                Mat draw = wordIm(Rect(newX1,0,newX2-newX1+1,wordIm.rows));
                
                int left = next.startX-newX1;
                for (int c=0; c<left; c++)
                {
                    draw.col(c) *= c/(0.0+left);
                }
                int right = newX2-next.endX;
                int colsC = newX2-newX1;
                for (int c=0; c<right; c++)
                {
                    draw.col(colsC-c) *= c/(0.0+right);
                }
                imwrite(savePre+"exemplar.png",draw);
#else
                Mat draw;
                cvtColor(wordIm,draw,CV_GRAY2BGR);
                for(int c=next.startX; c<=next.endX; c++)
                    for (int r=0; r<draw.rows; r++)
                        draw.at<Vec3b>(r,c)[0]*=0;
                imwrite(savePre+"exemplar.png",draw);
#endif
                auto iter = allsAccum.begin();
                for (int img=0; img<20; img++)
                {
                    SubwordSpottingResult r = resAccum.at((iter++)->second);
                    cvtColor(corpus_dataset->image(r.imIdx),draw,CV_GRAY2BGR);
                    for(int c=r.startX; c<=r.endX; c++)
                        for (int r=0; r<draw.rows; r++)
                            draw.at<Vec3b>(r,c)[2]*=0;
                    imwrite(savePre+to_string(img)+".png",draw);
                }
#endif
                float moveMean, moveStd, moveTopMean, moveTopStd;
                moveRatio = getRankChangeRatio(prevResAccum,resAccum,prevTruesAccum,truesAccum,prevAllsAccum,allsAccum,&rankDrop,&rankRise,&rankDropFull,&rankRiseFull, &moveMean, &moveStd, &moveTopMean, &moveTopStd);

                //cout<<ngram<<",\t["<<i<<"],\t"<<apN<<",\t"<<combAP<<",\t"<<moveRatio<<",\t"<<rankDrop<<",\t"<<rankRise<<",\t"<<rankDropFull<<",\t"<<rankRiseFull<<endl;
                cout<<ngram<<",\t["<<i<<"],\t"<<apN<<",\t"<<combAP<<",\t"<<moveMean<<",\t"<<moveStd<<",\t"<<moveTopMean<<",\t"<<moveTopStd<<endl;
                MAP_QbE+=apN;
                if (i==numSteps)
                    MAP_comb+=combAP;
                //stepAPSum[i-1]+=combAP;
                stepAps[i-1].push_back(combAP);
                //stats
                //meanAndStd(resN,&mean, &std);
                //cout<<"  "<<mean<<", "<<std<<endl;
                
            }

            //prevRes=resN;
        }

    }
    cout<<endl;
    cout<<"MAP QbS:  "<<(MAP_QbS/queryCount)<<endl;
    cout<<"MAP QbE:  "<<(MAP_QbE/(queryCount*numSteps))<<endl;
    cout<<"MAP comb: "<<(MAP_comb/queryCount)<<endl;
    for (int i=0; i<numSteps; i++)
    {
        float meanA, stdA;
        meanAndStd(stepAps.at(i),&meanA,&stdA);
        cout<<"MAP comb at "<<(i+1)<<": "<<meanA<<"\tstd: "<<stdA<<endl;
    }
}

float CNNSPPSpotter::getRankChangeRatio(const vector<SubwordSpottingResult>& prevRes, const vector<SubwordSpottingResult>& res, const multimap<float,int>& prevTrues, const multimap<float,int>& trues, const multimap<float,int>& prevAlls, const multimap<float,int>& alls, float* rankDrop, float* rankRise, float* rankDropFull, float* rankRiseFull, float* mean, float* std, float* meanTop, float* stdTop)//, float* meanFull, float* stdFull)
{
    *rankDrop=0;
    *rankRise=0;
    int rankUpdates=0;
    *rankDropFull=0;
    *rankRiseFull=0;
    int rankUpdatesFull=0;
    vector<int> difs;
    vector<int> difsTop;
    vector<int> difsFull;
    for (auto p : prevTrues)
    {

        int oldRank=0;
        for (auto iter=prevAlls.begin(); iter!=prevAlls.end(); iter++)
        {
            if (iter->second == p.second)
                break;
            oldRank++;
        }
        SubwordSpottingResult r = prevRes.at(p.second);
        bool matchFound=false;
        for (int i=0; i<res.size(); i++)
        {
            if (res.at(i).imIdx == r.imIdx)
            {
                double ratio = ( min(res.at(i).endX,r.endX) - max(res.at(i).startX,r.startX) ) /
                               ( max(res.at(i).endX,r.endX) - min(res.at(i).startX,r.startX) +0.0);
                if (ratio > LIVE_SCORE_OVERLAP_THRESH)
                {
                    matchFound=true;
                    int rank=0;
                    for (auto iter=alls.begin(); iter!=alls.end(); iter++)
                    {
                        if (iter->second == i)
                            break;
                        rank++;
                    }
                    int dif=rank-oldRank;
                    difsFull.push_back(dif);
                    difs.push_back(dif);
                    if (oldRank<100)
                        difsTop.push_back(dif);
                    if (dif<0)
                    {
                        *rankDrop-=dif;
                        *rankDropFull-=dif;
                    }
                    if (dif>0)
                    {
                        *rankRise+=dif;
                        *rankRiseFull+=dif;
                    }
                    rankUpdates++;
                    rankUpdatesFull++;
                    break;
                }
            }
            
        }
        if (!matchFound)
        {
            int rank=0;
            for (auto iter=alls.begin(); iter!=alls.end(); iter++)
            {
                if (iter->first >= r.score)
                    break;
                rank++;
            }
            int dif=rank-oldRank;
            difsFull.push_back(dif);
            if (dif<0)
                *rankDropFull-=dif;
            if (dif>0)
                *rankRiseFull+=dif;
            rankUpdatesFull++;
        }
    }
    *rankRise/=rankUpdates+0.0;
    *rankDrop/=rankUpdates+0.0;
    *rankRiseFull/=rankUpdatesFull+0.0;
    *rankDropFull/=rankUpdatesFull+0.0;
    assert (prevTrues.size()==0 || difs.size()>0);
    meanAndStd(difs,mean,std);
    meanAndStd(difsTop,meanTop,stdTop);
    //meanAndStd(difsFull,meanFull,stdFull);
    return *rankRise-*rankDrop;
}


void CNNSPPSpotter::evalFullWordSpottingRespot(const Dataset* data, vector<string> toSpot, int numSteps, int numRepeat, int repeatSteps)
{
    cout<<"Not accumulating QbS result."<<endl;
    setCorpus_dataset(data,true);


    set<string> done;
    float mAP=0;
    int queryCount=0;
    map<string,int> ngramCounter;
    map<string,float> ngramAPs;



    //cout<<"ngram,\titer,\tAP,  \tcombAP,\tshiftRatio(- good),\tbetterRatio,\tworseRatio,\tbetterRatio(inc miss),\tworseRatio(inc miss)"<<endl;
    cout<<"word,\titer,\tAP,  \tcombAP,\tshift mean(- good),\tshift STD,\tshiftTop mean(- good),\tshiftTop STD"<<endl;
    for (int inst=0; inst<toSpot.size(); inst++)
    {
        string word = toSpot[inst];
        //cout<<word<<endl;
        int Nrelevants = 0;
        float ap=0;
        
        //imshow("exe", exemplars->image(inst));
        //waitKey();

        multimap<float,int> truesAccum;
        multimap<float,int> resAccum = wordSpot(word); //scores
        ap = evalWordSpotting_singleScore(word, resAccum, -1, &truesAccum);
        auto midTrue = truesAccum.begin();
        //for (int iii=0; iii<trues.size()/2; iii++)
        //    iter++;
        assert(ap==ap);
        if (ap<0)
            continue;
        
        queryCount++;
        mAP+=ap;
        cout<<word<<",\t[QbS],\t"<<ap<<endl;
        //stats
        //float mean, std;
        //meanAndStd(resAccum,&mean, &std);
        //cout<<"Stats: "<<mean<<", "<<std<<endl;

        
        //vector<SubwordSpottingResult> prevRes=resAccum;
        for (int i=1; i<numSteps; i++)
        {
            //vector<SubwordSpottingResult> tmpRes=resAccum;
            multimap<float,int> truesN, resN;
            //for (int ii=0; ii<numRepeat&&(ii==0||i<repeatSteps+1); ii++)
            int ii=0;
            {

                midTrue = truesAccum.begin();
                for (int iii=0; iii<(truesAccum.size()/2)-ii; iii++)
                   ++midTrue;
                int next = midTrue->second;
                Mat exemplar = corpus_dataset->image(next);
                resN = wordSpot(exemplar);
                //cout<<"!! "<<resN.size()<<endl;

                //float apN = evalSubwordSpotting_singleScore(word, resN, corpusXLetterStartBounds, corpusXLetterEndBounds,-1, &truesN);
                float apN, combAP, rankRise, rankDrop, rankRiseFull, rankDropFull;
                float moveRatio;
                if (i==1)//QbS doesn't combine
                {
                    resAccum.clear();
                    truesAccum.clear();
                }
                multimap<float,int> prevTruesAccum=truesAccum;
                multimap<float,int> prevResAccum=resAccum;
                _eval(word,resN,&resAccum,&apN,&combAP,&truesAccum,&truesN);
                float moveMean, moveStd, moveTopMean, moveTopStd;
                moveRatio = getRankChangeRatioFull(prevResAccum,resAccum,prevTruesAccum,truesAccum,&rankDrop,&rankRise,&rankDropFull,&rankRiseFull, &moveMean, &moveStd, &moveTopMean, &moveTopStd);

                //cout<<word<<",\t["<<i<<"],\t"<<apN<<",\t"<<combAP<<",\t"<<moveRatio<<",\t"<<rankDrop<<",\t"<<rankRise<<",\t"<<rankDropFull<<",\t"<<rankRiseFull<<endl;
                cout<<word<<",\t["<<i<<"],\t"<<apN<<",\t"<<combAP<<",\t"<<moveMean<<",\t"<<moveStd<<",\t"<<moveTopMean<<",\t"<<moveTopStd<<endl;
                //stats
                //meanAndStd(resN,&mean, &std);
                //cout<<"Stats: "<<"  "<<mean<<", "<<std<<endl;
                
            }

            //prevRes=resN;
        }

    }
    cout<<endl;
}


float CNNSPPSpotter::getRankChangeRatioFull(const multimap<float,int>& prevRes, const multimap<float,int>& res, const multimap<float,int>& prevTrues, const multimap<float,int>& trues, float* rankDrop, float* rankRise, float* rankDropFull, float* rankRiseFull, float* mean, float* std, float* meanTop, float* stdTop)//, float* meanFull, float* stdFull)
{
    *rankDrop=0;
    *rankRise=0;
    int rankUpdates=0;
    *rankDropFull=0;
    *rankRiseFull=0;
    int rankUpdatesFull=0;
    vector<int> difs;
    vector<int> difsTop;
    vector<int> difsFull;
    for (auto p : prevTrues)
    {

        int oldRank=0;
        for (auto iter=prevRes.begin(); iter!=prevRes.end(); iter++)
        {
            if (iter->second == p.second)
                break;
            oldRank++;
        }
        bool matchFound=false;
        for (auto r : trues)
        {
            if (r.second == p.second)
            {
                matchFound=true;
                int rank=0;
                for (auto iter=res.begin(); iter!=res.end(); iter++)
                {
                    if (iter->second == r.second)
                        break;
                    rank++;
                }
                int dif=rank-oldRank;
                difsFull.push_back(dif);
                difs.push_back(dif);
                if (oldRank<100)
                    difsTop.push_back(dif);
                if (dif<0)
                {
                    *rankDrop-=dif;
                    *rankDropFull-=dif;
                }
                if (dif>0)
                {
                    *rankRise+=dif;
                    *rankRiseFull+=dif;
                }
                rankUpdates++;
                rankUpdatesFull++;
                break;
            }
            
        }
        if (!matchFound)
        {
            int rank=0;
            for (auto iter=res.begin(); iter!=res.end(); iter++)
            {
                if (iter->first >= p.first)
                    break;
                rank++;
            }
            int dif=rank-oldRank;
            difsFull.push_back(dif);
            if (dif<0)
                *rankDropFull-=dif;
            if (dif>0)
                *rankRiseFull+=dif;
            rankUpdatesFull++;
        }
    }
    *rankRise/=rankUpdates+0.0;
    *rankDrop/=rankUpdates+0.0;
    *rankRiseFull/=rankUpdatesFull+0.0;
    *rankDropFull/=rankUpdatesFull+0.0;
    assert (prevTrues.size()==0 || difs.size()>0);
    meanAndStd(difs,mean,std);
    meanAndStd(difsTop,meanTop,stdTop);
    //meanAndStd(difsFull,meanFull,stdFull);
    return *rankRise-*rankDrop;
}


float CNNSPPSpotter::calcAP(const vector<SubwordSpottingResult>& res, string ngram)
{
    int Nrelevants = 0;
    float ap=0;
    float maxScore=-9999;
    for (auto r : res)
        if (r.score>maxScore)
            maxScore=r.score;
    vector<float> scores;
    vector<bool> rel;
    int numTrumped=0;
    int numOff=0;
    int num_relevant=0;
    for (int j=0; j<corpus_dataset->size(); j++)
    {
        int loc = corpus_dataset->labels()[j].find(ngram);
        if (loc !=string::npos)
        {
            num_relevant++;
            if (corpus_dataset->labels()[j].find(ngram,loc+1) != string::npos) {
                num_relevant++; //allow at most 2 of an ngram in a word
            }
        }
    }
    if (num_relevant<11)
    {
        cout <<" too few"<<endl;
        return -1;
    }
    vector<int> checked(corpus_dataset->size());
    for (int j=0; j<res.size(); j++)
    {
        SubwordSpottingResult r = res[j];
        size_t loc = corpus_dataset->labels()[r.imIdx].find(ngram);
        if (loc==string::npos)
        {
            scores.push_back(r.score);
            rel.push_back(false);
            checked[r.imIdx]++;
        }
        else
        {
            int loc2 = corpus_dataset->labels()[r.imIdx].find(ngram,loc+1);
            vector<int> matching;
            for (int jj=0; jj < res.size(); jj++)
            {
                if (res[jj].imIdx == r.imIdx && j!=jj)
                    matching.push_back(jj);
            }
            if (matching.size()>0)
            {
                float relPos = (loc+(ngram.length()/2.0))/corpus_dataset->labels()[r.imIdx].length();
                float myDif = fabs(relPos - (r.startX + (r.endX-r.startX)/2.0)/(corpus_dataset->image(r.imIdx).cols));
                if (loc2 != string::npos)
                {
                    float relPos2 = (loc2+(ngram.length()/2.0))/corpus_dataset->labels()[r.imIdx].length();
                    float myDif2 = fabs(relPos2 - (r.startX + (r.endX-r.startX)/2.0)/(corpus_dataset->image(r.imIdx).cols));
                    if (myDif2<myDif)
                    {
                        relPos=relPos2;
                        myDif=myDif2;
                    }
                }
                bool other=false;
                for (int oi : matching)
                {
                    float oDif = fabs(relPos - (res[oi].startX + (res[oi].endX-res[oi].startX)/2.0)/(corpus_dataset->image(res[oi].imIdx).cols));

                    if ((oDif < myDif && checked[r.imIdx]==0) || oDif<=myDif) {
                        other=true;
                        break;
                    }
                }
                if (other)
                {
                    scores.push_back(r.score);
                    rel.push_back(false);
                    numTrumped++;
                }
                else
                {
                    scores.push_back(r.score);
                    rel.push_back(true);
                    checked[r.imIdx]++;
                }
            }
            else
            {
                bool ngram1H = loc+(ngram.length()/2.0) < 0.4*corpus_dataset->labels()[r.imIdx].length();
                bool ngram2H = loc+(ngram.length()/2.0) > 0.6*corpus_dataset->labels()[r.imIdx].length();
                bool ngramM = loc+(ngram.length()/2.0) > 0.25*corpus_dataset->labels()[r.imIdx].length() &&
                    loc+(ngram.length()/2.0) < 0.75*corpus_dataset->labels()[r.imIdx].length();

                bool ngram1H2 = loc2!=string::npos && loc2+(ngram.length()/2.0) < 0.4*corpus_dataset->labels()[r.imIdx].length();
                bool ngram2H2 = loc2!=string::npos && loc2+(ngram.length()/2.0) > 0.6*corpus_dataset->labels()[r.imIdx].length();
                bool ngramM2 = loc2!=string::npos && loc2+(ngram.length()/2.0) > 0.25*corpus_dataset->labels()[r.imIdx].length() &&
                    loc2+(ngram.length()/2.0) < 0.75*corpus_dataset->labels()[r.imIdx].length();

                float sLoc = r.startX + (r.endX-r.startX)/2.0;
                bool spot1H = sLoc < 0.4*(corpus_dataset->image(r.imIdx).cols);
                bool spot2H = sLoc > 0.6*(corpus_dataset->image(r.imIdx).cols);
                bool spotM = sLoc > 0.25*(corpus_dataset->image(r.imIdx).cols) &&
                    sLoc < 0.75*(corpus_dataset->image(r.imIdx).cols);

                if ( (ngram1H&&spot1H) || (ngram2H&&spot2H) || (ngramM&&spotM) ||
                     (ngram1H2&&spot1H) || (ngram2H2&&spot2H) || (ngramM2&&spotM) )
                {
                    scores.push_back(r.score);
                    rel.push_back(true);
                }
                else
                {
                    scores.push_back(r.score);
                    rel.push_back(false);
                    //Insert a dummy result for the correct spotting to keep MAP accurate
                    scores.push_back(maxScore);
                    rel.push_back(true);
                    //cout<<r.imIdx<<", ";
                    numOff++;

                    ////
                    /*
                    cv::Mat disp;
                    cv::cvtColor(data->image(r.imIdx),disp,CV_GRAY2BGR);
                    for (int x=r.startX; x<=r.endX; x++)
                        for (int y=0; y<data->image(r.imIdx).rows; y++)
                            disp.at<cv::Vec3b>(y,x)[0]=0;
                    cout<<"OFF: ["<<ngram<<"] in  "<<data->labels()[r.imIdx]<<endl;
                    cout<<ngram1H<<":"<<spot1H<<"  "<<ngram2H<<":"<<spot2H<<"  "<<ngramM<<":"<<spotM<<endl;
                    //cv::imshow("spotting",disp);
                    //cv::waitKey();
                    cv::imwrite("spotting_"+ngram+to_string(r.imIdx)+"_"+to_string(ngram1H)+"_"+to_string(spot1H)+"_"+to_string(ngram2H)+"_"+to_string(spot2H)+"_"+to_string(ngramM)+"_"+to_string(spotM)+".png",disp);
                    */
                    ////
                }

                checked.at(r.imIdx)++;
            }
        }
    }
    for (int j=0; j<corpus_dataset->size(); j++)
    {
        int loc = corpus_dataset->labels().at(j).find(ngram);
        if (checked.at(j)==0 &&  loc !=string::npos)
        {
            scores.push_back(maxScore);
            rel.push_back(true);
            checked.at(j)++;
        }
        if (loc !=string::npos && checked[j]<2 && corpus_dataset->labels()[j].find(ngram,loc+1) != string::npos)
        {
            scores.push_back(maxScore);
            rel.push_back(true);
            checked.at(j)++;
        }
    }
    vector<int> rank;
    for (int j=0; j < scores.size(); j++)
    {            
        float s = scores[j];
        //cout <<"score for "<<j<<" is "<<s<<". It is ["<<data->labels()[j]<<"], we are looking for ["<<text<<"]"<<endl;
        
        if (rel[j])
        {
            int better=0;
            int equal = 0;
            
            for (int k=0; k < scores.size(); k++)
            {
                if (k!=j)
                {
                    float s2 = scores[k];
                    if (s2< s) better++;
                    else if (s2==s) equal++;
                }
            }
            
            
            rank.push_back(better+floor(equal/2.0));
            Nrelevants++;
        }
        
    }
    if (Nrelevants != num_relevant)
        cout<<"Nrelevants: "<<Nrelevants<<" != num_relevant: "<<num_relevant<<endl;
    assert(Nrelevants == num_relevant);
    qsort(rank.data(), Nrelevants, sizeof(int), sort_xxx);
    
    //pP1[i] = p1;
    
    /* Get mAP and store it */
    for(int j=0;j<Nrelevants;j++){
        /* if rank[i] >=k it was not on the topk. Since they are sorted, that means bail out already */
        
        float prec_at_k =  ((float)(j+1))/(rank[j]+1);
        //mexPrintf("prec_at_k: %f\n", prec_at_k);
        ap += prec_at_k;            
        assert(ap==ap);
    }
    ap/=Nrelevants;
    cout<<" num relv: "<<Nrelevants<<"  numTrumped: "<<numTrumped<<" numOff: "<<numOff<<"  ";
    return ap;
}

void CNNSPPSpotter::evalSubwordSpotting(const Dataset* exemplars, /*string exemplars_locations,*/ const Dataset* data)
{
    setCorpus_dataset(data,false);

    map<string,set<int> > widths;
    for (int i=0; i<exemplars->size(); i++)
    {
        widths[exemplars->labels()[i]].insert(exemplars->image(i).cols);
    }

    cout<<"Average exemplar widths"<<endl;
    for (auto p : widths)
    {
        double avg=0;
        for (int w : p.second)
            avg+=w;
        avg/=p.second.size();
        cout <<p.first<<": "<<avg<<endl;
    }

    float map=0;
    int queryCount=0;
    float gramMap=0;
    string gram="";
    int gramCount=0;
    #pragma omp parallel for
    for (int inst=0; inst<exemplars->size(); inst++)
    {
        string ngram = exemplars->labels()[inst];
        cout <<"on spotting inst:"<<inst<<", "<<ngram<<" ";
        cout << flush;
        //int *rank = new int[other];//(int*)malloc(NRelevantsPerQuery[i]*sizeof(int));
        float ap=0;
        
        //imshow("exe", exemplars->image(inst));
        //waitKey();
        vector<SubwordSpottingResult> res = subwordSpot(ngram.length(),exemplars->image(inst)); //scores
        ap = calcAP(res, ngram);
        assert(ap==ap);
        if (ap<0)
            continue;
        
        #pragma omp critical (storeMAP)
        {
            queryCount++;
            map+=ap;
            cout<<" ap: "<<ap;
            //cout<<"on spotting inst:"<<inst<<", "<<ngram<<"   ap: "<<ap<<endl;
            /*if (gram.compare(ngram)!=0)
            {
                if (gramCount>0)
                {
                    cout <<"ap for ["<<gram<<"]: "<<(gramMap/gramCount)<<endl;
                    gramCount=0;
                    gramMap=0;
                }
                gram=ngram;
            }
            gramMap+=ap;
            gramCount++;*/
        }
        cout <<endl;
    }
        //cout <<"ap for ["<<gram<<"]: "<<(gramMap/gramCount)<<endl;
        
    cout<<"FULL map: "<<(map/queryCount)<<endl;
}
void CNNSPPSpotter::evalSubwordSpotting(const vector<string>& exemplars, const Dataset* data)
{
    setCorpus_dataset(data,false);


    float map=0;
    int queryCount=0;
    float gramMap=0;
    string gram="";
    int gramCount=0;
    #pragma omp parallel for
    for (int inst=0; inst<exemplars.size(); inst++)
    {
        string ngram = exemplars[inst];
        cout <<"on spotting inst:"<<inst<<", "<<ngram<<" ";
        cout << flush;
        //int *rank = new int[other];//(int*)malloc(NRelevantsPerQuery[i]*sizeof(int));
        int Nrelevants = 0;
        float ap=0;
        
        //imshow("exe", exemplars->image(inst));
        //waitKey();
        vector<SubwordSpottingResult> res = subwordSpot(exemplars[inst]); //scores
        ap = calcAP(res,ngram);
        assert(ap==ap);
        if (ap<0)
            continue;
        
        #pragma omp critical (storeMAP)
        {
            queryCount++;
            map+=ap;
            cout<<" ap: "<<ap;
            //cout<<"on spotting inst:"<<inst<<", "<<ngram<<"   ap: "<<ap<<endl;
            /*if (gram.compare(ngram)!=0)
            {
                if (gramCount>0)
                {
                    cout <<"ap for ["<<gram<<"]: "<<(gramMap/gramCount)<<endl;
                    gramCount=0;
                    gramMap=0;
                }
                gram=ngram;
            }
            gramMap+=ap;
            gramCount++;*/
        }
        cout <<endl;
    }
        //cout <<"ap for ["<<gram<<"]: "<<(gramMap/gramCount)<<endl;
        
    cout<<"FULL map: "<<(map/queryCount)<<endl;
}
/*void CNNSPPSpotter::evalSubwordSpottingCombine(const Dataset* exemplars, const Dataset* data)
{
    setCorpus_dataset(data);

    map<string,set<int> > widths;
    for (int i=0; i<exemplars->size(); i++)
    {
        widths[exemplars->labels()[i]].insert(exemplars->image(i).cols);
    }

    cout<<"Average exemplar widths"<<endl;
    for (auto p : widths)
    {
        double avg=0;
        for (int w : p.second)
            avg+=w;
        avg/=p.second.size();
        cout <<p.first<<": "<<avg<<endl;
    }
    map<string,vector<Mat> > combExemplars;
    for (int inst=0; inst<exemplars->size(); inst++)
    {
        string ngram = exemplars->labels()[inst];
        combExemplars[ngram].push_back(exemplars->image(inst));
    }


        float map=0;
        int queryCount=0;
        float gramMap=0;
        string gram="";
        int gramCount=0;
        #pragma omp parallel  for
        for (int inst=0; inst<combExemplars.size(); inst++)
        {
            auto iter = combExemplars.begin();
            for (int i=0; i<inst; i++)
                iter++;
            string ngram = iter->first;
           //cout <<"on spotting inst:"<<inst<<", "<<ngram;
            //cout << flush;
            //int *rank = new int[other];//(int*)malloc(NRelevantsPerQuery[i]*sizeof(int));
            int Nrelevants = 0;
            float ap=0;
            
            float bestS=-99999;
            //imshow("exe", exemplars->image(inst));
            //waitKey();
            vector<SubwordSpottingResult> res = subwordSpot(iter->second); //scores
            float maxScore=-9999;
            for (auto r : res)
                if (r.score>maxScore)
                    maxScore=r.score;
            vector<float> scores;
            vector<bool> rel;
            for (int j=0; j<res.size(); j++)
            {
                SubwordSpottingResult r = res[j];
                size_t loc = data->labels()[r.imIdx].find(ngram);
                if (loc==string::npos)
                {
                    scores.push_back(r.score);
                    rel.push_back(false);
                }
                else
                {
                    vector<int> matching;
                    for (int jj=0; jj < res.size(); jj++)
                    {
                        if (res[jj].imIdx == r.imIdx && j!=jj)
                            matching.push_back(jj);
                    }
                    if (matching.size()>0)
                    {
                        float relPos = (loc+(ngram.length()/2.0))/data->labels()[r.imIdx].length();
                        float myDif = fabs(relPos - (r.startX + (r.endX-r.startX)/2.0)/data->image(r.imIdx).cols);
                        bool other=false;
                        for (int oi : matching)
                        {
                            float oDif = fabs(relPos - (res[oi].startX + (res[oi].endX-res[oi].startX)/2.0)/data->image(res[oi].imIdx).cols);
                            if (oDif < myDif) {
                                other=true;
                                break;
                            }
                        }
                        if (other)
                        {
                            scores.push_back(r.score);
                            rel.push_back(false);
                        }
                        else
                        {
                            scores.push_back(r.score);
                            rel.push_back(true);
                        }
                    }
                    else
                    {
                        bool ngram1H = loc+(ngram.length()/2.0) < 0.8*data->labels()[r.imIdx].length()/2.0;
                        bool ngram2H = loc+(ngram.length()/2.0) > 1.2*data->labels()[r.imIdx].length()/2.0;
                        bool ngramM = loc+(ngram.length()/2.0) > data->labels()[r.imIdx].length()/3.0 &&
                            loc+(ngram.length()/2.0) < 2.0*data->labels()[r.imIdx].length()/3.0;
                        float sLoc = r.startX + (r.endX-r.startX)/2.0;
                        bool spot1H = sLoc < 0.8*data->image(r.imIdx).cols/2.0;
                        bool spot2H = sLoc > 1.2*data->image(r.imIdx).cols/2.0;
                        bool spotM = sLoc > data->image(r.imIdx).cols/3.0 &&
                            sLoc < 2.0*data->image(r.imIdx).cols/3.0;

                        if ( (ngram1H&&spot1H) || (ngram2H&&spot2H) || (ngramM&&spotM) )
                        {
                            scores.push_back(r.score);
                            rel.push_back(true);
                        }
                        else
                        {
                            scores.push_back(r.score);
                            rel.push_back(false);
                            //Insert a dummy result for the correct spotting to keep MAP accurate
                            scores.push_back(maxScore);
                            rel.push_back(true);
                        }

                    }
                }
            }
            vector<int> rank;
            for (int j=0; j < scores.size(); j++)
            {            
                float s = scores[j];
                //cout <<"score for "<<j<<" is "<<s<<". It is ["<<data->labels()[j]<<"], we are looking for ["<<text<<"]"<<endl;
                
                if (rel[j])
                {
                    int better=0;
                    int equal = 0;
                    
                    for (int k=0; k < scores.size(); k++)
                    {
                        if (k!=j)
                        {
                            float s2 = scores[k];
                            if (s2> s) better++;
                            else if (s2==s) equal++;
                        }
                    }
                    
                    
                    rank.push_back(better+floor(equal/2.0));
                    Nrelevants++;
                }
                
            }
            qsort(rank.data(), Nrelevants, sizeof(int), sort_xxx);
            
            //pP1[i] = p1;
            
            // Get mAP and store it 
            for(int j=0;j<Nrelevants;j++){
                // if rank[i] >=k it was not on the topk. Since they are sorted, that means bail out already
                
                float prec_at_k =  ((float)(j+1))/(rank[j]+1);
                //mexPrintf("prec_at_k: %f\n", prec_at_k);
                ap += prec_at_k;            
            }
            ap/=Nrelevants;
            
            #pragma omp critical (storeMAP)
            {
                queryCount++;
                map+=ap;
                cout<<"on spotting inst:"<<inst<<", "<<ngram<<"   ap: "<<ap<<endl;
            }
            
        }
        //cout <<"ap for ["<<gram<<"]: "<<(gramMap/gramCount)<<endl;
        
        cout<<"FULL map: "<<(map/queryCount)<<endl;
}*/

string CNNSPPSpotter::lowercaseAndStrip(string s)
{
    string ret="";
    for (int i=0; i<s.length(); i++)
    {
        if (s[i]!=' ' && s[i]!='\n' &&  s[i]!='\t' &&  s[i]!='\r')
            ret+=tolower(s[i]);
    }
    return ret;
}

void CNNSPPSpotter::evalRecognition(const Dataset* data, const vector<string>& lexicon)
{
    addLexicon(lexicon);

    float precision=0;
    //float precisionOoV=0;
    int numIV=0;
    float recall=0;
    //vector<float> diffT, diffF;
    setCorpus_dataset(data,true);
    //vector< multimap<float,string> > corpusScores = transcribeCorpus();
    int pruningOn=10;
    //net arch: one hidden layer size 15
    set<string> trues, falses;

    for (int i=0; i<corpus_dataset->size(); i++)
    {
        //pruning
        /*
        float diff=orderedScores.begin()->first;
        auto iter = orderedScores.begin();
        for (int j=0; j<5; j++, iter++)
            if (j==4)
                diff-=iter->first;
        if (diff>-0.00282267 -0.0145615)
            continue;
        */

        //create pruning data
        multimap<float,string> scores = transcribeCorpus(i);
        auto iter = scores.begin();
        string pruningInstance="";
        for (int j=0; j<pruningOn; j++, iter++)
        {
            pruningInstance+=to_string(iter->first)+" ";
        }
        pruningInstance+="\n";

        recall+=1;
        string gt = lowercaseAndStrip(corpus_dataset->labels()[i]);
        //for (string w : lexicon)
        //    if (gt.compare(w)==0)
        //    {
        //        numIV++;
        //        break;
        //    }
        //cout<<gt<<": ";
        bool t=false;
        iter = scores.begin();
        for (int j=0; j<5; j++, iter++)
        {
            string word = iter->second;
            //cout<<word<<", ";
            if (word.compare(gt)==0)
            {
                precision+=1;
                t=true;
                numIV++;
                break;
            }
        }
        if (!t)
        for (int j=5; j<scores.size(); j++, iter++)
        {
            string word = iter->second;
            //cout<<word<<", ";
            if (word.compare(gt)==0)
            {
                numIV++;
                break;
            }
        }
        if (t)
        {
            //diffT.push_back(diff);
            pruningInstance+="1\n";
            trues.insert(pruningInstance);
        }
        else
        {
            //diffF.push_back(diff);
            pruningInstance+="-1\n";
            falses.insert(pruningInstance);
        }
        //cout<<endl;

    }
    float ivp = precision/numIV;
    precision/=recall;
    recall/=corpus_dataset->size();
    cout<<"precision: "<<precision<<"    recal: "<<recall<<endl;
    cout<<"IV precision: "<<ivp<<endl;

    set<string> falsesCopy(falses);

    ofstream pruningData(saveName+"_pruningData.spec");
    pruningData<<"# number of inputs"<<endl;
    pruningData<<pruningOn<<endl;
    pruningData<<"# number of outputs"<<endl<<1<<endl;
    pruningData<<"# input & output"<<endl;

    while (1)
    {
        int tf = rand()%2;
        if (tf)
        {
            if (trues.size()==0)
                break;
            int write = rand()%trues.size();
            auto iter = trues.begin();
            for (int i=0; i<write; i++)
                iter++;
            pruningData<<*iter;
            trues.erase(iter);
        }
        else
        {
            int write = rand()%falses.size();
            auto iter = falses.begin();
            for (int i=0; i<write; i++)
                iter++;
            pruningData<<*iter;
            falses.erase(iter);

            if (falses.size()==0)
                falses=falsesCopy;
        }
    }


    pruningData.close();
    /*
    float avg=0;
    for (float f : diffT)
        avg+=f;
    avg/=diffT.size();
    float std=0;
    for (float f : diffT)
        std+=pow(avg - f,2);
    std = sqrt(std/diffT.size());
    cout<<"True diff mean: "<<avg<<", std dev: "<<std<<endl;

    avg=0;
    for (float f : diffF)
        avg+=f;
    avg/=diffT.size();
    std=0;
    for (float f : diffF)
        std+=pow(avg - f,2);
    std = sqrt(std/diffF.size());
    cout<<"false diff mean: "<<avg<<", std dev: "<<std<<endl;
    */
}
