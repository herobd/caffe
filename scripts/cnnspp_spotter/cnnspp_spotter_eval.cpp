#include "cnnspp_spotter.h"
#include <set>
#include <stdlib.h>

#include <sys/stat.h>
#include <sys/types.h>

#ifdef SHOW_CLUST
#include "dimage.h"
#include "ddynamicprogramming.h"
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
    float currentAP = evalSubwordSpotting_singleScore(ngram, res, corpusXLetterStartBounds,corpusXLetterEndBounds,-1,NULL,NULL,NULL,&notSpottedIn);
    while (currentAP < goalAP)
    {
        //swap lowest score false and highest score true
        float minScore=999999;
        float maxScore=-999999;
        int minFalse=-1;
        float minFalseScore=999999;
        int maxTrue=-1;
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
        if (minFalse>=0)
            res[minFalse].score = newFalseDist(generator);
        if (maxTrue>=0)
        {
            if (RAND_PROB < 0.5 && notSpottedIn.size()>0)
            {
                //add missed spotting
                int randIdx = rand()%notSpottedIn.size();
                int wordId = notSpottedIn[randIdx];
                notSpottedIn.erase(notSpottedIn.begin()+randIdx);
                size_t loc = corpus_dataset->labels()[wordId].find(ngram);
                assert(loc!=string::npos);
                SubwordSpottingResult newResult(wordId, newTrueDist(generator), corpusXLetterStartBounds->at(wordId)[loc], corpusXLetterEndBounds->at(wordId)[loc+ngram.length()-1]);
                newResult.gt=1;
                res.push_back(newResult);
            }
            else
            {
                res[maxTrue].score = newTrueDist(generator);
            }
        }


        currentAP = evalSubwordSpotting_singleScore(ngram, res, corpusXLetterStartBounds,corpusXLetterEndBounds,-1,NULL,NULL,NULL,&notSpottedIn);
    }
            
}

float overlapPortion(int s1, int e1, int s2, int e2)
{
    return ( min(e1,e2) - max(s1,s2) +0.0) / min(e1-s1,e2-s2);
}


bool swapSkip(string key, string ngram, string word)
{
   if (key[0]=='$')
   {
      if (word.length()>=ngram.length() && word.find(ngram)==string::npos)
       {
            //does the word have any permutations of ngram?

            // Sort the string in lexicographically
            // ascennding order
            sort(ngram.begin(), ngram.end());
         
            // Keep checking next permutation while there
            // is next permutation
            do {
                if (word.find(ngram)!=string::npos)
                {
                    //cout<<"skipping word '"<<word<<"' for (perm)ngram '"<<ngram<<"'"<<endl;
                    return true;
                }
            } while (next_permutation(ngram.begin(), ngram.end()));

       }
   }
   else if (key[0]=='#')
   {
       if (word.length()>=ngram.length() && word.find(ngram)==string::npos)
       {
           for (char c:ngram)
           {
               if (word.find(c)!=string::npos)
                   return true;
           }
       }

   }
  return false;
} 

//This is a testing function for the simulator
#define LIVE_SCORE_OVERLAP_THRESH .5
float CNNSPPSpotter::evalSubwordSpotting_singleScore(string ngram, vector<SubwordSpottingResult>& res, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, int skip, int* trueCount, multimap<float,int>* trues, multimap<float,int>* alls, vector<int>* notSpottedIn, string skipIfNgram, bool notPresent)
{

    if (trues!=NULL)
        trues->clear();
    if (alls!=NULL)
        alls->clear();
    //string ngram = exemplars->labels()[inst];
    int Nrelevants = 0;
    float ap=0;
    
    float bestS=-99999;
    float maxScore=-9999;
    for (auto r : res)
        if (r.score>maxScore)
            maxScore=r.score;
    vector<float> scores;
    vector<bool> rel;
    vector<int> indexes;
    if (trueCount!=NULL)
        *trueCount=0;
    //vector<bool> checked(corpus_dataset->size());
    vector< map<int,bool> > checked(corpus_dataset->size());//whether each positive instance is accounted for
    for (int j=0; j<corpus_dataset->size(); j++)
    {
        if (j==skip || (skipIfNgram.length()>0&&(corpus_dataset->labels()[j].find(skipIfNgram)!=string::npos)!=notPresent) || swapSkip(skipIfNgram,ngram,corpus_dataset->labels()[j]))
        {
            //cout <<"skipped1 "<<j<<endl;
            continue;
        }
        int lastLoc=-1;
        while(true)
        {
            size_t loc = corpus_dataset->labels()[j].find(ngram,lastLoc+1);
            if (loc==string::npos)
                break;
            else
            {
                checked[j][loc]=false;
                if (trueCount!=NULL)
                    (*trueCount)++;
            }
            lastLoc=loc;
        }
    }
    int l=ngram.length()-1;
    set<int> overlapSkip;
    for (int j=0; j<res.size(); j++)
    {
        SubwordSpottingResult& r = res[j];
        /*if (r.imIdx==3848)
        {
            cout<<"3848 overlapSkip:";
            for (int jj : overlapSkip)
                cout<<" "<<jj;
            cout<<endl;
        }*/
        if (skip == r.imIdx || overlapSkip.find(j)!=overlapSkip.end() || (skipIfNgram.length()>0&&(corpus_dataset->labels()[r.imIdx].find(skipIfNgram)!=string::npos)!=notPresent) || swapSkip(skipIfNgram,ngram,corpus_dataset->labels()[r.imIdx]))
        {
            //cout <<"skipped2 "<<r.imIdx<<endl;
            continue;
        }
        if (alls!=NULL)
            alls->emplace(r.score,j);
        vector<size_t> locs;
        for (auto p : checked.at(r.imIdx))
            locs.push_back(p.first);
        float myOverlap=0;
        int myLoc=-1;
        if (locs.size()>0)
        {
            for (int locIdx=0; locIdx<locs.size(); locIdx++)
            {
                float o = overlapPortion(corpusXLetterStartBounds->at(r.imIdx)[locs[locIdx]],corpusXLetterEndBounds->at(r.imIdx)[locs[locIdx]+l],r.startX,r.endX);
                if (o > myOverlap)
                {
                    myOverlap=o;
                    myLoc=locs[locIdx];
                }
            }
            if (myLoc==-1) //There was no overlap
            {
                //Use distance
                float bestDist=999999;
                float myCenter = (r.startX+r.endX)/2.0;
                for (int locIdx=0; locIdx<locs.size(); locIdx++)
                {
                    float locCenter = (corpusXLetterStartBounds->at(r.imIdx)[locs[locIdx]]+corpusXLetterEndBounds->at(r.imIdx)[locs[locIdx]+l])/2.0;
                    float dist = fabs(myCenter-locCenter);
                    if (dist<bestDist)
                    {
                        bestDist=dist;
                        myLoc=locs[locIdx];
                    }
                }
            }
        }

        /*if (r.gt==1) something is broken here
        {
            scores.push_back(r.score);
            rel.push_back(true);
            indexes.push_back(j);
            checked.at(r.imIdx)[myLoc]=true;
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
            checked.at(r.imIdx)[myLoc]=true;
            //if (notSpottedIn!=NULL)
            //    notSpottedIn->push_back(j);
        }
        else*/
        {
            //find gt label

            //size_t loc = corpus_dataset->labels()[r.imIdx].find(ngram);
            //if (loc==string::npos)
            if (locs.size()==0)
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
                    if (res[jj].imIdx == r.imIdx && j!=jj)// && res[jj].imIdx!=skip && !((skipIfNgram.length()>0&&(corpus_dataset->labels()[res[jj].imIdx].find(skipIfNgram)!=string::npos)!=notPresent) || swapSkip(skipIfNgram,ngram,corpus_dataset->labels()[res[jj].imIdx])))
                        matching.push_back(jj);
                }
                //if (r.imIdx==3848)
                //    cout<<"3848 matching "<<matching.size()<<endl;
                //float myOverlap = ( min(corpusXLetterEndBounds->at(r.imIdx)[loc+l], r.endX) 
                //                    - max(corpusXLetterStartBounds->at(r.imIdx)[loc], r.startX) ) 
                //                  /
                //                  ( max(corpusXLetterEndBounds->at(r.imIdx)[loc+l], r.endX) 
                //                    - min(corpusXLetterStartBounds->at(r.imIdx)[loc], r.startX) +0.0);
                //vector<float> myOverlaps(locs.size());

                
                
                if (matching.size()>0)
                {
                    //float relPos = (loc+(ngram.length()/2.0))/corpus_dataset->labels()[r.imIdx].length();
                    //float myDif = fabs(relPos - (r.startX + (r.endX-r.startX)/2.0)/corpus_dataset->image(r.imIdx).cols);
                    if (myOverlap > LIVE_SCORE_OVERLAP_THRESH)
                    {
                        bool other=false;
                        int otherJ=-1;
                        float otherO=-1;
                        for (int oi : matching)
                        {
                            float otherOverlap=0;
                            int otherLoc=-1;
                            //float otherOverlap = ( min(corpusXLetterEndBounds->at(res[oi].imIdx)[loc+l], res[oi].endX) 
                            //                        - max(corpusXLetterStartBounds->at(res[oi].imIdx)[loc], res[oi].startX) ) 
                            //                      /
                            //                      ( max(corpusXLetterEndBounds->at(res[oi].imIdx)[loc+l], res[oi].endX) 
                            //                        - min(corpusXLetterStartBounds->at(res[oi].imIdx)[loc], res[oi].startX) +0.0);
                            for (int locIdx=0; locIdx<locs.size(); locIdx++)
                            {
                                float o = overlapPortion(corpusXLetterStartBounds->at(res[oi].imIdx)[locs[locIdx]],corpusXLetterEndBounds->at(res[oi].imIdx)[locs[locIdx]+l],res[oi].startX,res[oi].endX);
                                if (o>otherOverlap)
                                {
                                    otherOverlap=o;
                                    otherLoc=locs[locIdx];
                                }
                            }
                            if (otherLoc == myLoc && otherOverlap > myOverlap) {
                                other=true;
                            }
                            if (otherLoc == myLoc && otherOverlap > LIVE_SCORE_OVERLAP_THRESH)
                            {
                                otherJ=oi;
                                otherO=otherOverlap;
                            }
                        }
                //if (r.imIdx==3848)
                //    cout<<"3848 other "<<other<<", otherJ "<<otherJ<<", otherO "<<otherO<<endl;
                        if (other)
                        {
                            //scores.push_back(r.score);
                            //rel.push_back(false);
                            //indexes.push_back(j);

                            //skip this instance
                            r.gt=1;
                        }
                        else 
                        {
                            scores.push_back(r.score);
                            rel.push_back(true);
                            indexes.push_back(j);
                            checked.at(r.imIdx)[myLoc]=true;
                            r.gt=1;
                            if (otherO > LIVE_SCORE_OVERLAP_THRESH)
                            {
                                //skip the other one so we dont count two positives for one instance
                                overlapSkip.insert(otherJ);
                                res.at(otherJ).gt=1;//TODO ??
                            }
                        }
                    }
                    else
                    {
                        scores.push_back(r.score);
                        rel.push_back(false);
                        indexes.push_back(j);
                        r.gt=0;
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
                        checked.at(r.imIdx)[myLoc]=true;
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
                        checked.at(r.imIdx)[myLoc]=true;
                    }

                }
            }
        }
    }
    for (int j=0; j<corpus_dataset->size(); j++)
    {
        for (auto p : checked[j])
        {
            //if (!checked[j] &&  corpus_dataset->labels()[j].find(ngram)!=string::npos)
            if (!p.second) 
            {
                scores.push_back(maxScore);
                rel.push_back(true);
                indexes.push_back(-1);
                if (notSpottedIn!=NULL)
                    notSpottedIn->push_back(j);
            }
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
            {
                //assert(indexes.at(j)!=-1);
                trues->emplace(s,indexes[j]);
            }
        }
        
    }
    if (Nrelevants==0)
        return -1;

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

#ifdef SHOW_CLUST
DImage toDImage(Mat src)
{
    DImage img;
    img.setLogicalSize(src.cols,src.rows);
    unsigned char* data1 = img.dataPointer_u8();
    unsigned char* dataO = src.data;
    for (int i=0; i< src.cols * src.rows; i++)
    {
        data1[i]=dataO[i];
    }

    return img;
}
Mat toMat(DImage& src)
{
    Mat img(src.height(),src.width(),CV_8U);
    unsigned char* dataO = src.dataPointer_u8();
    unsigned char* data1 = img.data;
    for (int i=0; i< img.cols * img.rows; i++)
    {
        data1[i]=dataO[i];
    }

    return img;
}

Mat warpIm(Mat orig, int pathLen, int* path, bool horz)
{
    int meanV = mean(orig)[0];
    Mat ret;
    if (horz)
        ret=Mat(orig.rows,pathLen,CV_8U);
    else
        ret=Mat(pathLen,orig.cols,CV_8U);
    int oc=-1;
    int rc=-1;
    for (int i=0; i<pathLen; i++)
    {
        if (path[i]==0)
        {
            oc++;
            rc++;
        }
        else if(path[i]==1)
            oc++;
        else if (path[i]==2)
            rc++;
        Rect oSlice;
        Rect rSlice;
        if (horz)
        {
            oSlice=Rect(oc,0,1,orig.rows);
            rSlice=Rect(rc,0,1,ret.rows);
        }
        else
        {
            oSlice=Rect(0,oc,orig.cols,1);
            rSlice=Rect(0,rc,ret.cols,1);
        }
        
        if (rc<0)
            continue;
        else if (rc>=0 && oc<0)
            if (horz)
                ret(rSlice) = meanV*Mat::ones(ret.rows,1,CV_8U);
            else
                ret(rSlice) = meanV*Mat::ones(1,ret.cols,CV_8U);
        else if (path[i]==1)
            ret(rSlice) = min(orig(oSlice),ret(rSlice));
        else
            orig(oSlice).copyTo(ret(rSlice));
    }
    if (horz)
        return ret(Rect(0,0,rc+1,ret.rows));
    else
        return ret(Rect(0,0,ret.cols,rc+1));
}

void transposeDI(DImage& src, DImage& dst)
{
    if (dst.height() != src.width() || dst.width() != src.height())
    {
        dst.setLogicalSize(src.height(),src.width());
    }
    unsigned char* dataSrc = src.dataPointer_u8();
    unsigned char* dataDst = dst.dataPointer_u8();
    
    for (int i=0; i<src.width(); i++)
    {
        for (int j=0; j<src.height(); j++)
        {
            int idxSrc = src.width() * j + i;
            int idxDst = dst.width() * i + j;
            dataDst[idxDst]=dataSrc[idxSrc];
        }
    }
}

#endif
void CNNSPPSpotter::demonstrateClustering(string destDir, string ngram, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds)
{
    if (destDir[destDir.length()-1]!='/')
        destDir+="/";
    vector<string> ngrams={ngram};
    Mat crossScores;
    vector<SpottingLoc> res = massSpot(ngrams,crossScores);

    float maxScore=-9999;
    for (auto r : res)
        if (r.scores[ngram]>maxScore)
            maxScore=r.scores[ngram];
    vector<float> scores;
    vector<bool> gt;
    vector<bool> checked(corpus_dataset->size());
    int l=ngram.length()-1;
    for (int j=0; j<res.size(); j++)
    {
        SpottingLoc& r = res[j];

        size_t loc = corpus_dataset->labels()[r.imIdx].find(ngram);
        if (loc==string::npos)
        {
            scores.push_back(r.scores[ngram]);
            gt.push_back(false);
            //r.gt=0;
        }
        else
        {
            vector<int> matching;
            for (int jj=0; jj < res.size(); jj++)
            {
                if (res[jj].imIdx == r.imIdx && j!=jj)
                    matching.push_back(jj);
            }
            float myOverlap = overlapPortion(corpusXLetterStartBounds->at(r.imIdx)[loc],corpusXLetterEndBounds->at(r.imIdx)[loc+l],r.startX,r.endX);
            
            if (matching.size()>0)
            {
                //float relPos = (loc+(ngram.length()/2.0))/corpus_dataset->labels()[r.imIdx].length();
                //float myDif = fabs(relPos - (r.startX + (r.endX-r.startX)/2.0)/corpus_dataset->image(r.imIdx).cols);
                bool other=false;
                for (int oi : matching)
                {
                    float otherOverlap = overlapPortion(corpusXLetterStartBounds->at(res[oi].imIdx)[loc],corpusXLetterEndBounds->at(res[oi].imIdx)[loc+l],res[oi].startX,res[oi].endX);
                    if (otherOverlap > myOverlap) {
                        other=true;
                        break;
                    }
                }
                if (other)
                {
                    scores.push_back(r.scores[ngram]);
                    gt.push_back(false);
                    //r.gt=0;
                }
                else if (myOverlap > LIVE_SCORE_OVERLAP_THRESH)
                {
                    scores.push_back(r.scores[ngram]);
                    gt.push_back(true);
                    //r.gt=1;
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
                    scores.push_back(r.scores[ngram]);
                    gt.push_back(true);
                    //r.gt=1;
                }
                else
                {
                    ////
                    //cout<<"bad overlap["<<j<<"]: "<<myOverlap<<endl;
                    ////
                    scores.push_back(r.scores[ngram]);
                    gt.push_back(false);
                    //r.gt=-1;
                }

            }
        }
    }

    //now cluster!
    //using hierarchical clustering
    //vector< vector< list<int> > > clustersAtLevels;
    //level,clusters,instances
    vector< list<int> > clusters(res.size());
    Mat minSimilarity = crossScores.clone();//minimum pair-wise link for each cluster
    for (int r=0; r<minSimilarity.rows; r++)
        minSimilarity.at<float>(r,r)=-99999;
    for (int i=0; i<res.size(); i++)
        clusters[i].push_back(i);
    vector<float> meanCPurity,  medianCPurity,  meanIPurity,  medianIPurity,  maxPurity;
    vector< vector< list<int> > > clusterLevels;
    CL_cluster(clusters,minSimilarity,10, gt, meanCPurity,  medianCPurity,  meanIPurity,  medianIPurity,  maxPurity, clusterLevels);


    cout<<"clustering statistics:"<<endl;
    cout<<"i:\tmeanCPurity,\tmedianCPurity,\tmeanIPurity,\tmedianIPurity,\tmaxPurity"<<endl;
    for (int i=0; i<meanCPurity.size(); i++)
    {
        cout<<i<<":\t"<<meanCPurity[i]<<",\t"<<medianCPurity[i]<<",\t"<<meanIPurity[i]<<",\t"<<medianIPurity[i]<<",\t"<<maxPurity[i]<<endl;
    }

#ifndef SHOW_CLUST
    cout<<"saving..."<<endl;
    for (int i=0; i<clusters.size(); i++)
    {
        system(("mkdir -p "+destDir+to_string(i)).c_str());
        int count=0;
        for (int inst : clusters[i])
        {
            Mat spot = corpus_dataset->image( res[inst].imIdx );
            spot = spot(Rect(res[inst].startX,0,res[inst].endX-res[inst].startX+1,spot.rows));
            imwrite(destDir+to_string(i)+"/"+to_string(count++)+".png",spot);
        }   
    }
#else
    int level;//=clusterLevels.size();
    while (true)
    {
        cout<<"Enter cluster level (-1 end): "<<flush;
        cin >> level;
        if (level<0)
            break;

        Mat collage = Mat::zeros(1200,100,CV_8U);
        int curY=0;
        int curX=0;
        int nextX=0;
        int clusterId=0;
        for (list<int>& cluster : clusterLevels[level])
        {
            //get cluster center
            int maxSimSum=-99999;
            int center;
            for (int i1 : cluster)
            {
                int simSum=0;
                for (int i2 : cluster)
                {
                    if (i1!=i2)
                    {
                        simSum += crossScores.at<float>(i1,i2);
                    }
                }
                if (simSum>maxSimSum)
                {
                    maxSimSum=simSum;
                    center=i1;
                }
            }

            Mat centerIm = corpus_dataset->image(res[center].imIdx);
            centerIm = centerIm(Rect(res[center].startX,0,res[center].endX-res[center].startX+1,centerIm.rows));
            
            Mat hCProfile;
            reduce(centerIm,hCProfile,0,CV_REDUCE_AVG,CV_64F);
            Mat vCProfile;
            reduce(centerIm,vCProfile,1,CV_REDUCE_AVG,CV_64F);
            hCProfile.convertTo(hCProfile,CV_64F);
            vCProfile.convertTo(vCProfile,CV_64F);
            DFeatureVector hCVector;
            hCVector.setData_dbl(((double*)hCProfile.data),hCProfile.cols,1,true,false);
            DFeatureVector vCVector;
            vCVector.setData_dbl(((double*)vCProfile.data),vCProfile.rows,1,true,false);
            
            //DFeatureVector fvHC = DWordFeatures::extractWordFeatures(centerDIm,true,false,true,true,127);
            //vector<Mat> adjustedImages = {centerIm};
            Mat sumIm;
            centerIm.convertTo(sumIm,CV_32S);
            Mat minIm = centerIm.clone();
            for (int i : cluster)
            {
                if (i!=center)
                {
                    Mat im = corpus_dataset->image(res[i].imIdx);
                    im = im(Rect(res[i].startX,0,res[i].endX-res[i].startX+1,im.rows));
                    
                    Mat hProfile;
                    reduce(im,hProfile,0,CV_REDUCE_AVG,CV_64F);
                    Mat vProfile;
                    reduce(im,vProfile,1,CV_REDUCE_AVG,CV_64F);
                    hProfile.convertTo(hProfile,CV_64F);
                    vProfile.convertTo(vProfile,CV_64F);
                    DFeatureVector hVector;
                    hVector.setData_dbl(((double*)hProfile.data),hProfile.cols,1,true,false);
                    DFeatureVector vVector;
                    vVector.setData_dbl(((double*)vProfile.data),vProfile.rows,1,true,false);
                    
                    int pathLenH;
                    int *rgPathH=new int[(1+hCVector.vectLen)*(1+hVector.vectLen)];
                    DDynamicProgramming::findDPAlignment(hCVector,hVector,30,20000,1000,&pathLenH,rgPathH);
                    int pathLenV;
                    int *rgPathV=new int[(1+vCVector.vectLen)*(1+vVector.vectLen)];
                    DDynamicProgramming::findDPAlignment(vCVector,vVector,30,20000,1000,&pathLenV,rgPathV);
                    /*
                    DImage dIm = toDImage(im);
                    dIm = DDynamicProgramming::piecewiseLinearWarpDImage(dIm,centerIm.cols,pathLenH,rgPathH,false);
                    DImage dIm_t;
                    transposeDI(dIm,dIm_t);
                    dIm_t = DDynamicProgramming::piecewiseLinearWarpDImage(dIm_t,centerIm.rows,pathLenV,rgPathV,false);
                    transposeDI(dIm_t,dIm);
                    Mat warp = toMat(dIm);
                    */
                    Mat warp = warpIm(im,pathLenH,rgPathH,true);
                    warp = warpIm(warp,pathLenV,rgPathV,false);
                    //Mat warp = warpIm(im,pathLenV,rgPathV,false);
                    delete [] rgPathH;
                    delete [] rgPathV;
                    //resize(warp,warp,centerIm.size());
                    
                    //Mat warp;
                    //resize(im,warp,centerIm.size());
                    
                    assert(warp.rows==centerIm.rows && warp.cols==centerIm.cols);
                    //adjustedImages.push_back(warp);
                    //sumIm += warp;
                    add(sumIm,warp,sumIm,noArray(),CV_32S);
                    minIm=cv::min(minIm,warp);


                }
            }
            sumIm/=cluster.size();

            sumIm.convertTo(sumIm,CV_8U);
            //imshow(to_string(clusterId)+" mean",sumIm);
            //imshow(to_string(clusterId)+" min",minIm);
            //imshow(to_string(clusterId)+" center",centerIm);
            if (centerIm.rows+curY>=collage.rows)
            {
                //new column
                curY=0;
                curX=nextX;
                nextX=-1;
            }
            if (curX+centerIm.cols>=collage.cols)
            {
                Mat append = Mat::zeros(collage.rows,2*(curX+centerIm.cols-collage.cols),CV_8U);
                hconcat(collage,append,collage);
            }
            centerIm.copyTo(collage(Rect(curX,curY,centerIm.cols,centerIm.rows)));
            putText(collage,to_string(clusterId),Point(curX,curY+centerIm.rows/2),FONT_HERSHEY_PLAIN,1.5,Scalar(0),2);
            curY+=centerIm.rows;
            if (curX+centerIm.cols+2>nextX)
                nextX=curX+centerIm.cols+2;
            if (centerIm.rows+curY>=collage.rows)
            {
                //new column
                curY=0;
                curX=nextX;
                nextX=-1;
            }
            if (curX+centerIm.cols>=collage.cols)
            {
                Mat append = Mat::zeros(collage.rows,2*(curX+centerIm.cols-collage.cols),CV_8U);
                hconcat(collage,append,collage);
            }
            sumIm.copyTo(collage(Rect(curX,curY,sumIm.cols,sumIm.rows)));
            curY+=sumIm.rows;
            if (curX+sumIm.cols+2>nextX)
                nextX=curX+sumIm.cols+2;
            if (centerIm.rows+curY>=collage.rows)
            {
                //new column
                curY=0;
                curX=nextX;
                nextX=-1;
            }
            if (curX+centerIm.cols>=collage.cols)
            {
                Mat append = Mat::zeros(collage.rows,2*(curX+centerIm.cols-collage.cols),CV_8U);
                hconcat(collage,append,collage);
            }
            minIm.copyTo(collage(Rect(curX,curY,minIm.cols,minIm.rows)));
            curY+=minIm.rows+2;
            if (curX+minIm.cols+2>nextX)
                nextX=curX+minIm.cols+2;
            clusterId++;
        }
        imshow("collage",collage);
        cout<<"all clusters for level "<<level<<" ("<<clusterLevels[level].size()<<" clusters)"<<endl;
        cout<<"saving..."<<endl;
        //system(("rm "+destDir+"/*");
        for (int i=0; i<clusterLevels[level].size(); i++)
        {
            system(("mkdir -p "+destDir+to_string(i)).c_str());
            system(("rm "+destDir+to_string(i)+"/*.png").c_str());
            int count=0;
            for (int inst : clusterLevels[level][i])
            {
                Mat spot = corpus_dataset->image( res[inst].imIdx );
                spot = spot(Rect(res[inst].startX,0,res[inst].endX-res[inst].startX+1,spot.rows));
                imwrite(destDir+to_string(i)+"/"+to_string(count++)+".png",spot);
            }   
        }
        waitKey();
    }
#endif
}



void CNNSPPSpotter::CL_cluster(vector< list<int> >& clusters, Mat& minSimilarity, int numClusters, const vector<bool>& gt, vector<float>& meanCPurity, vector<float>& medianCPurity, vector<float>& meanIPurity, vector<float>& medianIPurity, vector<float>& maxPurity, vector< vector< list<int> > >& clusterLevels)
{
    while (clusters.size()>numClusters)
    {
        /////
        for (int r=0; r<minSimilarity.rows; r++)
            for (int c=0; c<minSimilarity.cols; c++)
                assert(minSimilarity.at<float>(r,c) == minSimilarity.at<float>(r,c));

        Point maxLink;
        minMaxLoc(minSimilarity,NULL,NULL,NULL,&maxLink);//get cluster with strongest link
        int clust1 = std::min(maxLink.x,maxLink.y);
        int clust2 = std::max(maxLink.x,maxLink.y);
        clusters[clust1].insert(clusters[clust1].end(), clusters[clust2].begin(), clusters[clust2].end());
        clusters.erase(clusters.begin()+clust2);
        for (int r=0; r<minSimilarity.rows; r++)
        {
            if (r!=clust1 && r!=clust2)
            {
                minSimilarity.at<float>(clust1,r) = minSimilarity.at<float>(r,clust1) = min(minSimilarity.at<float>(clust1,r),minSimilarity.at<float>(clust2,r));
            }
        }
        Mat newMinS(minSimilarity.rows-1,minSimilarity.cols-1,CV_32F);
        minSimilarity(Rect(0,0,clust2,clust2)).copyTo(newMinS(Rect(0,0,clust2,clust2)));
        if (minSimilarity.cols-(clust2+1)>0)
        {
            minSimilarity(Rect(clust2+1,0,minSimilarity.cols-(clust2+1),clust2)).copyTo(newMinS(Rect(clust2,0,minSimilarity.cols-(clust2+1),clust2)));
            minSimilarity(Rect(0,clust2+1,clust2,minSimilarity.rows-(clust2+1))).copyTo(newMinS(Rect(0,clust2,clust2,minSimilarity.rows-(clust2+1))));
            minSimilarity(Rect(clust2+1,clust2+1,minSimilarity.cols-(clust2+1),minSimilarity.rows-(clust2+1))).copyTo(newMinS(Rect(clust2,clust2,minSimilarity.cols-(clust2+1),minSimilarity.rows-(clust2+1))));
        }
        minSimilarity=newMinS;
        clusterLevels.push_back(clusters);

        //stats tracking
        float sumCPurity=0;
        float sumIPurity=0;
        float mPurity=0;
        set<float> sortCPurity, sortIPurity;
        for (auto clust : clusters)
        {
            float numF = 0;
            float numT = 0;
            for (int i : clust)
            {
                if (gt[i])
                    numT++;
                else
                    numF++;
            }
            float purity = 2* ((max(numT,numF)/(numT+numF))-0.5);
            sumCPurity += purity;
            sumIPurity += clust.size()*purity;
            sortCPurity.insert(purity);
            for (int i=0; i<clust.size(); i++)
                sortIPurity.insert(purity);
            if (purity > mPurity)
                mPurity=purity;
        }
        meanCPurity.push_back(sumCPurity/clusters.size());
        meanIPurity.push_back(sumIPurity/gt.size());
        maxPurity.push_back(mPurity);
        auto iter = sortCPurity.begin();
        for (int i=0; i<sortCPurity.size()/2; i++)
            iter++;
        medianCPurity.push_back(*iter);
        iter = sortIPurity.begin();
        for (int i=0; i<sortIPurity.size()/2; i++)
            iter++;
        medianIPurity.push_back(*iter);
    }
}


float CNNSPPSpotter::evalWordSpotting_singleScore(string word, const multimap<float,int>& res, int skip, multimap<float,int>* trues)
{
    if (trues!=NULL)
        trues->clear();
    //string word = exemplars->labels()[inst];
    int Nrelevants = 0;
    float ap=0;
    
    float bestS=-99999;
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

#if CHEAT_WINDOW
int CNNSPPSpotter::getBestWindowWidth(int i, string searchNgram)
{
    int loc = corpus_dataset->labels()[i].find(searchNgram);
    int startX = corpusXLetterStartBounds->at(i).at(loc);
    int endX = corpusXLetterEndBounds->at(i).at(loc+searchNgram.size()-1);
    int width = endX-startX+10;
    int bestClustDist=9999999;
    int bestClustWidth=-1;
    int maxClustWidth=0;
    for (auto p : ngramWW)
    {
        int clustWidth = p.second;
        int d = abs(clustWidth-width);
        if (clustWidth>width && d<bestClustDist)
        {
            bestClustDist=d;
            bestClustWidth=clustWidth;
        }
        if (clustWidth>maxClustWidth)
            maxClustWidth=clustWidth;
    }
    if (bestClustWidth==-1)
        bestClustDist=bestClustWidth;
    return bestClustWidth;
}
#endif

void CNNSPPSpotter::evalSubwordSpottingWithCharBounds(int N, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, vector<string> queries, string outDir, int windowWidth, map<string,float>* aps, map<string,vector<SubwordSpottingResult> >* allResults)
{
#if CHEAT_WINDOW
    this->corpusXLetterStartBounds = corpusXLetterStartBounds;
    this->corpusXLetterEndBounds = corpusXLetterEndBounds;
#endif
    if (outDir.length()>0 && outDir[0]!='!')
        mkdir(outDir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    map<string,int> exDrawn;


    set<string> done;
    float mAP=0;
    int queryCount=0;
    map<string,int> ngramCounter;
    map<string,float> ngramAPs;
    vector<string>exemplars;
    vector<string>same_exemplars;

    if ((outDir.length()==0 || queries.size()==0) && windowWidth==-1 && aps==NULL)
    {
    cout<<"---QbE---"<<N<<endl;
    //#pragma omp parallel for
    for (int inst=0; inst<corpus_dataset->size(); inst++)
    {
        string label = corpus_dataset->labels()[inst];
        if (label.length()<N)
            continue;

        map<int,string> instanceQueries;

        if (queries.size()==0)
        {
            //Take a single random ngram from each word

            int ngramLoc= inst%(label.length()-(N-1)); //inst is used to randomize where inthe string the ngram is extracted
            string ngram="";
            do {
                ngram = label.substr(ngramLoc,N);
                int nRel=0;
                for (int inst2=0; inst2<corpus_dataset->size(); inst2++)
                {
                    if (corpus_dataset->labels()[inst2].find(ngram) != string::npos)
                        nRel++;
                }
                if (nRel<30)
                {
                    ngramLoc= (ngramLoc+1)%(label.length()-(N-1));
                    ngram="";
                }
            } while(ngram.length()==0 && ngramLoc!=inst%(label.length()-(N-1)));
            if (ngram.length()==0)
                continue;
            instanceQueries[ngramLoc]=ngram;
        }
        else
        {
            ///take all instances of the set queries from each word
            for (int ngramLoc=0; ngramLoc<label.length()-(N-1); ngramLoc++)
            {
                string ngram = label.substr(ngramLoc,N);
                if (find(queries.begin(), queries.end(), ngram) != queries.end())
                {
                    //int nRel=0;
                    //for (int inst2=0; inst2<corpus_dataset->size() && nRel<2; inst2++)
                    //{
                    //    if (corpus_dataset->labels()[inst2].find(ngram) != string::npos)
                    //        nRel++;
                    //}
                    //if (nRel>=2)
                        instanceQueries[ngramLoc]=ngram;
                }
            }
        }

        for (auto p : instanceQueries)
        {
            string ngram = p.second;
            int ngramLoc = p.first;
#if CHEAT_WINDOW
            searchNgram=ngram;
#endif

            Mat exemplar;
            Mat wordIm = corpus_dataset->image(inst);
            int x1 = max(0,corpusXLetterStartBounds->at(inst)[ngramLoc] - (ngramLoc==0?END_PAD_EXE:PAD_EXE));
            int x2 = min(wordIm.cols-1,corpusXLetterEndBounds->at(inst)[ngramLoc+(N-1)] + (ngramLoc==label.length()-N?END_PAD_EXE:PAD_EXE));
#if PRECOMP_QBE

            vector<SubwordSpottingResult> res = subwordSpotAbout(ngram,inst,(x2+x1)/2.0,1.0); //scores
#else
#if SQUARE_QBE==1
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
#elif SQUARE_QBE==2
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
            if (newX1>x1)
                newX1=x1;
            if (newX2<x2)
                newX2=x2;
#else
            int newX1=x1;
            int newX2=x2;
#endif
               
            if (newX1<0 || newX2>=wordIm.cols || newX2<newX1)
               cout<<"Error wordIm w:"<< wordIm.cols<<"  x1:"<<x1<<" x2:"<<x2<<"  newx1:"<<newX1<<" newx2:"<<newX2<<endl;
            //This crops a square region so no distortion happens.
            exemplar = wordIm(Rect(newX1,0,newX2-newX1+1,wordIm.rows));

            vector<SubwordSpottingResult> res = subwordSpot(ngram,exemplar,1.0); //scores
#endif
            ////
            /*
            imshow("exe", exemplar);
            cout<<"exemplar: "<<ngram<<endl;
            cout<<corpus_dataset->labels()[res[0].imIdx]<<":"<<res[0].score<<"  "<<corpus_dataset->labels()[res[1].imIdx]<<":"<<res[1].score<<"  "<<corpus_dataset->labels()[res[2].imIdx]<<":"<<res[2].score<<"  "<<corpus_dataset->labels()[res[3].imIdx]<<":"<<res[3].score<<endl;
            if (res[0].startX<0 || res[0].endX>=corpus_dataset->image(res[0].imIdx).cols || res[0].endX<=res[0].startX)
                cout<<"ERROR[0]  image w:"<<corpus_dataset->image(res[0].imIdx).cols<<"  s:"<<res[0].startX<<" e:"<<res[0].endX<<endl;
            if (res[1].startX<0 || res[1].endX>=corpus_dataset->image(res[1].imIdx).cols || res[1].endX<=res[1].startX)
                cout<<"ERROR[1]  image w:"<<corpus_dataset->image(res[1].imIdx).cols<<"  s:"<<res[1].startX<<" e:"<<res[1].endX<<endl;
            if (res[2].startX<0 || res[2].endX>=corpus_dataset->image(res[2].imIdx).cols || res[2].endX<=res[2].startX)
                cout<<"ERROR[2]  image w:"<<corpus_dataset->image(res[2].imIdx).cols<<"  s:"<<res[2].startX<<" e:"<<res[2].endX<<endl;
            //cout<<"["<<corpus_dataset->image(res[0].imIdx).cols<<","<<corpus_dataset->image(res[0].imIdx).rows<<"] R "<<res[0].startX<<" 0 "<<
            Mat top1 = corpus_dataset->image(res[0].imIdx)(Rect(res[0].startX,0,res[0].endX-res[0].startX+1,corpus_dataset->image(res[0].imIdx).rows));
            imshow("top1",top1);
            Mat top2 = corpus_dataset->image(res[1].imIdx)(Rect(res[1].startX,0,res[1].endX-res[1].startX+1,corpus_dataset->image(res[1].imIdx).rows));
            imshow("top2",top2);
            Mat top3 = corpus_dataset->image(res[2].imIdx)(Rect(res[2].startX,0,res[2].endX-res[2].startX+1,corpus_dataset->image(res[2].imIdx).rows));
            imshow("top3",top3);
            Mat top4 = corpus_dataset->image(res[3].imIdx)(Rect(res[3].startX,0,res[3].endX-res[3].startX+1,corpus_dataset->image(res[3].imIdx).rows));
            imshow("top4",top4);
            waitKey();
            */
            ////

            float ap = evalSubwordSpotting_singleScore(ngram, res, corpusXLetterStartBounds, corpusXLetterEndBounds, inst);

            if (outDir.length()>0 && exDrawn[ngram]++<5)
            {
                /*
                string dirName = outDir+"/QbE_"+ngram+"_"+to_string(exDrawn[ngram])+"/";
                mkdir(dirName.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                imwrite(dirName+"exemplar.png",exemplar);
                for (int ii=0; ii<15; ii++)
                {
                    Mat top = corpus_dataset->image(res[ii].imIdx)(Rect(res[ii].startX,0,res[ii].endX-res[ii].startX+1,corpus_dataset->image(res[ii].imIdx).rows));
                    imwrite(dirName+"top"+to_string(ii)+".png",top);
                }*/
                
            }
            //#pragma omp critical (storeMAP)
            if (ap>=0)
            {
                queryCount++;
                mAP+=ap;
                //cout<<"on spotting inst:"<<inst<<", "<<ngram<<"   ap: "<<ap<<endl;
                ngramCounter[ngram]++;
                ngramAPs[ngram]+=ap;
            }
        }

    }
    cout<<"ngram\tnum_inst\tAP";
    float byNgramMAP=0;
    for (auto p : ngramCounter)
    {
        cout<<p.first<<"\t"<<p.second<<"\t"<<ngramAPs[p.first]/p.second<<endl;
        same_exemplars.push_back(p.first);
        byNgramMAP+=ngramAPs[p.first]/p.second;
    }
    cout<<endl;
    cout<<"by ngram QbE "<<N<<" map: "<<(byNgramMAP/same_exemplars.size())<<endl;
    cout<<"individual QbE "<<N<<" map: "<<(mAP/queryCount)<<endl;
    }//testing

    multimap<float,pair<string,vector<SubwordSpottingResult> >,greater<float> > bestNgrams;
    multimap<float,pair<string,vector<SubwordSpottingResult> > > worstNgrams;
    cout<<"\n---QbS---"<<N<<"\nngram\ttr_cnt\tAP"<<endl;
    map<string,float> scores;
    map<string,int> tpCount;
    if (queries.size()>0)
    {
        exemplars.insert(exemplars.end(),queries.begin(),queries.end());
        mAP=0;
        queryCount=0;
        float weightedMAP=0;
        int totalCount=0;
        for (int inst=0; inst<exemplars.size(); inst++)
        {
            string ngram = exemplars[inst];
#if CHEAT_WINDOW
            searchNgram=ngram;
#endif
            vector<SubwordSpottingResult> res = subwordSpot(exemplars[inst],1.0,windowWidth); //scores
            if (outDir.length()>0 && outDir[0]=='!' && ngram.compare(outDir.substr(1))==0)
                printAllVectors(ngram,res);
            if (allResults!=NULL)
                (*allResults)[ngram]=res;

            int truesCount;
            float ap = evalSubwordSpotting_singleScore(ngram, res, corpusXLetterStartBounds, corpusXLetterEndBounds,-1, &truesCount);
            scores[ngram]=ap;
            tpCount[ngram]=truesCount;
            assert(ap==ap);
            if (ap<0)
                continue;
            if (aps != NULL)
                (*aps)[ngram]=ap;
            /*
        int cc=0;
        float meanMax=0;
        for (const SubwordSpottingResult& r : res)
            if (corpus_dataset.labels()[r.imIdx].find(ngram)!=string::npos)
            {
                cc++;
                double minV,maxV;
                minMaxLoc(
                meanMax+=maxV;
                */

            if (outDir.length()>0 && outDir[0]!='!')
            {
                bestNgrams.emplace(ap,make_pair(ngram,res));
                worstNgrams.emplace(ap,make_pair(ngram,res));
                if (bestNgrams.size()>20)
                {
                    bestNgrams.erase(prev(bestNgrams.end()));
                    worstNgrams.erase(prev(worstNgrams.end()));
                }
                    /*string dirName = outDir+"/QbS_"+ngram+"/";
                    mkdir(dirName.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                    for (int ii=0; ii<20; ii++)
                    {
                        Mat top = corpus_dataset->image(res[ii].imIdx)(Rect(res[ii].startX,0,res[ii].endX-res[ii].startX+1,corpus_dataset->image(res[ii].imIdx).rows));
                        imwrite(dirName+"top"+to_string(ii)+".png",top);
                    }*/
                    
                /*
                    vector<int> falseNegatives;
                    auto iter=trues.rbegin();
                    for (int i=0; i<10 && iter!=trues.rend(); i++, iter++)
                    {
                        falseNegatives.push_back(iter->second);
                    }
                    string badDir = outDir+"/falseNegatives_"+ngram+"/";
                    mkdir(badDir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                    for (int ii : falseNegatives)
                    {
                        Mat top = corpus_dataset->image(res[ii].imIdx)(Rect(res[ii].startX,0,res[ii].endX-res[ii].startX+1,corpus_dataset->image(res[ii].imIdx).rows));
                        imwrite(badDir+"fn"+to_string(ii)+".png",top);
                    }/**/
            }
            //if (ngram.compare("abo")==0)
            //    cout<<"!!! abo: "<<ap<<"  !!!"<<endl;
            
            queryCount++;
            mAP+=ap;
            weightedMAP+=truesCount*ap;
            totalCount+=truesCount;
            cout<<ngram<<"\t"<<truesCount<<"\t"<<ap<<endl;
        }
        cout<<endl;
        if (windowWidth==-1)
        {
            cout<<"FULL QbS "<<N<<" map: "<<(mAP/queryCount)<<endl;
            cout<<"weighted QbS "<<N<<" map: "<<(weightedMAP/totalCount)<<endl;
        }

        if (outDir.length()>0 && outDir[0]!='!')
        {
            for (auto p : bestNgrams)
            {
                float ap = p.first;
                string ngram = p.second.first;
                vector<SubwordSpottingResult> res = p.second.second;
                string dir = outDir+"/best_"+ngram+"_"+to_string(ap)+"/";
                mkdir(dir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                ofstream info(dir+"info.csv");
                info<<"score, image, label, startX, endX"<<endl;
                int n=0;
                for (auto sr : res)
                {
                    info<<sr.score<<", "<<sr.imIdx<<", "<<corpus_dataset->labels()[sr.imIdx]<<", "<<sr.startX<<", "<<sr.endX<<endl;
                    string fileName = dir+to_string(n++)+"_"+to_string(sr.score)+"_"+corpus_dataset->labels()[sr.imIdx]+"_";
                    if (sr.gt==1)
                        fileName+="TRUE.png";
                    else 
                        fileName+="FALSE.png";
                    Mat img = 255-corpus_dataset->image(sr.imIdx);
                    cvtColor(img,img,CV_GRAY2BGR);
                    for (int r=0; r<corpus_dataset->image(sr.imIdx).rows; r++)
                        for (int c=sr.startX; c<=sr.endX; c++)
                            img.at<Vec3b>(r,c)[0]=0;
                    //img(Rect(sr.startX,0,sr.endX-sr.startX+1,corpus_dataset->image(sr.imIdx).rows)) *= Scalar(0,1,1);
                    imwrite(fileName,img);
                    if ((ngram.length() < 3 && n==10) || n==20)
                        break;
                }
                info.close();
            }
            for (auto p : worstNgrams)
            {
                float ap = p.first;
                string ngram = p.second.first;
                vector<SubwordSpottingResult> res = p.second.second;
                string dir = outDir+"/worst_"+ngram+"_"+to_string(ap)+"/";
                mkdir(dir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                ofstream info(dir+"info.csv");
                info<<"score, image, label, startX, endX"<<endl;
                int n=0;
                for (auto sr : res)
                {
                    info<<sr.score<<", "<<sr.imIdx<<", "<<corpus_dataset->labels()[sr.imIdx]<<", "<<sr.startX<<", "<<sr.endX<<endl;
                    string fileName = dir+to_string(n++)+"_"+to_string(sr.score)+"_";
                    if (sr.gt==1)
                        fileName+="TRUE.png";
                    else 
                        fileName+="FALSE.png";
                    Mat img = 255-corpus_dataset->image(sr.imIdx);
                    cvtColor(img,img,CV_GRAY2BGR);
                    for (int r=0; r<corpus_dataset->image(sr.imIdx).rows; r++)
                        for (int c=sr.startX; c<=sr.endX; c++)
                            img.at<Vec3b>(r,c)[0]=0;
                    //img(Rect(sr.startX,0,sr.endX-sr.startX+1,corpus_dataset->image(sr.imIdx).rows)) *= Scalar(0,1,1);
                    imwrite(fileName,img);
                    if (n==10)
                        break;
                }
                info.close();
            }
        }
    }


    if (windowWidth==-1)
    {
        mAP=0;
        queryCount=0;
        for (int inst=0; inst<same_exemplars.size(); inst++)
        {
            string ngram = same_exemplars[inst];
            int Nrelevants = 0;
            float ap;
            if (scores.find(ngram) != scores.end())
                ap = scores[ngram];
            else
            {
                vector<SubwordSpottingResult> res = subwordSpot(same_exemplars[inst],1.0); //scores


                ap = evalSubwordSpotting_singleScore(ngram, res, corpusXLetterStartBounds, corpusXLetterEndBounds,-1);
            }
            assert(ap==ap);
            if (ap<0)
                continue;
            
            queryCount++;
            mAP+=ap;
            //cout<<ngram<<", "<<ap<<endl;
        }
        cout<<endl;
        cout<<"Same QbS "<<N<<" map: "<<(mAP/queryCount)<<endl;
    }

    if (queries.size()>0 && windowWidth==-1)
    {
        cout<<"Removing infrequent ngrams"<<endl;
        for (int rem=0; rem<=10; rem++)
        {
            mAP=0;
            queryCount=0;
            float weightedMAP=0;
            int totalCount=0;
            //map<string,int> numN;
            for (string ngram : exemplars)
            {
                if (tpCount[ngram]>rem)
                {
                    mAP+=scores[ngram];
                    queryCount+=1;
                    weightedMAP+=scores[ngram]*tpCount[ngram];
                    totalCount+=tpCount[ngram];
                    //numN[ngram.length()]++;
                }
            }
            cout<<endl;
            cout<<"above "<<rem<<",    ngram QbS "<<N<<" map: "<<(mAP/queryCount)<<endl;
            cout<<"above "<<rem<<", weighted QbS "<<N<<" map: "<<(weightedMAP/totalCount)<<endl;
            cout<<"  count: "<<queryCount<<endl;
            //cout<<"above "<<rem<<", num uni: "<<numN[1]<<", num bi: "<<numN[2]<<", num tri: "<<numN[3]<<endl;
        }
    }

}


void CNNSPPSpotter::printAllVectors(string ngram, const vector<SubwordSpottingResult>& res)
{
    string outDir = "allVectors_"+ngram;
    mkdir(outDir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    int windowWidth=ngramWW.at(ngram);
    Mat phoc = normalizedPHOC(ngram);
    //savePHOC(outDir+"/query.png",phoc);
    int maxSave=150;
    for (const SubwordSpottingResult& r : res)
    {
        Mat v = corpus_embedded.at(windowWidth).at(r.imIdx).col(r.windowIdx);
        savePHOC(outDir+"/w_"+to_string(r.score)+"_"+corpus_dataset->labels()[r.imIdx]+".png",v,phoc);
        if (maxSave--<=0)
            break;
    }

}

void CNNSPPSpotter::savePHOC(string dest, Mat v, Mat q)
{
    Mat out = Mat::zeros(201,v.rows+13,CV_8UC3);
    int offset=0;
    int color=0;
    Vec3b colors[6];
    colors[0]=Vec3b(255,0,0);
    colors[1]=Vec3b(0,255,0);
    colors[2]=Vec3b(0,0,255);
    colors[3]=Vec3b(222,222,0);
    colors[4]=Vec3b(222,0,222);
    colors[5]=Vec3b(0,222,222);
    int alphaSize=36;
    for (int r=0; r<v.rows; r++)
    {
        if (r%alphaSize==0 && r>0)
        {
            for (int h=0; h<201; h++)
                out.at<Vec3b>(h,r+offset)=Vec3b(255,255,255);
            offset+=1;
        }
        for (int h=0; h<q.at<float>(r,0)*100; h++)
            out.at<Vec3b>(100-h,r+offset)=colors[color];
        for (int h=0; h<v.at<float>(r,0)*100; h++)
            out.at<Vec3b>(100+100-h,r+offset)=colors[color];
        color = (color+1)%6;
    }
    imwrite(dest,out);
}

void CNNSPPSpotter::swapTest(int N, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, vector<string> queries)
{
    cout<<"\n---QbS---"<<N<<"\nngram\tAP\tno swap AP"<<endl;
    map<string,float> scores;
    map<string,int> tpCount;
    float  mAP=0;
    float  mAPNoSwap=0;
    int queryCount=0;
    int totalCount=0;
    for (int inst=0; inst<queries.size(); inst++)
    {
        string ngram = queries[inst];
#if CHEAT_WINDOW
        searchNgram=ngram;
#endif
        vector<SubwordSpottingResult> res = subwordSpot(queries[inst],1.0); //scores

        float ap = evalSubwordSpotting_singleScore(ngram, res, corpusXLetterStartBounds, corpusXLetterEndBounds,-1);
        float apNoSwaps = evalSubwordSpotting_singleScore(ngram, res, corpusXLetterStartBounds, corpusXLetterEndBounds,-1, NULL,NULL,NULL,NULL,"$wap$$$$$$$$$$$$$$",false);
        //scores[ngram]=ap;
        assert(ap==ap);
        if (ap<0)
            continue;

        
        queryCount++;
        mAP+=ap;
        mAPNoSwap+=apNoSwaps;
        cout<<ngram<<"\t"<<ap<<"\t"<<apNoSwaps<<endl;
    }
    cout<<endl;
    cout<<"FULL QbS "<<N<<" map:    "<<(mAP/queryCount)<<endl;
    cout<<"No swap QbS "<<N<<" map: "<<(mAPNoSwap/queryCount)<<endl;
    
}

void CNNSPPSpotter::conflationTest(const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, map<int,vector<string> > queries)
{
    map<string,float> aps;
    map<string,vector<SubwordSpottingResult> > res;
    for (int N=1; N<=3; N++)
        evalSubwordSpottingWithCharBounds(N, corpusXLetterStartBounds, corpusXLetterEndBounds, queries[N], "", -1, &aps, &res);

    vector<string> ngrams;
    ngrams.insert(ngrams.end(),queries[2].begin(),queries[2].end());
    ngrams.insert(ngrams.end(),queries[3].begin(),queries[3].end());
    for (string ngram : ngrams)
    {
        if (ngram.length()==1)
            continue;
        float apNoSingle = evalSubwordSpotting_singleScore(ngram, res[ngram], corpusXLetterStartBounds, corpusXLetterEndBounds,-1,NULL,NULL,NULL,NULL,"#single###");
        cout<<"\n"<<ngram<<":"<<aps[ngram]<<", without single char words:"<<apNoSingle<<"\n  normal ";
        for (char c : ngram)
            cout<<" "<<c<<":"<<aps[string(1,c)];
        cout<<"\n  in     ";
        for (char c : ngram)
        {
            float apInNgram = evalSubwordSpotting_singleScore(string(1,c), res[string(1,c)], corpusXLetterStartBounds, corpusXLetterEndBounds,-1,NULL,NULL,NULL,NULL,ngram,true);
            cout<<" "<<c<<":"<<apInNgram;
        }
        cout<<"\n  not in ";
        for (char c : ngram)
        {
            float apNoNgram = evalSubwordSpotting_singleScore(string(1,c), res[string(1,c)], corpusXLetterStartBounds, corpusXLetterEndBounds,-1,NULL,NULL,NULL,NULL,ngram,false);
            cout<<" "<<c<<":"<<apNoNgram;
        }
        cout<<endl;
    }
}


void CNNSPPSpotter::refineWindowSubwordSpottingWithCharBounds(int N, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds, vector<string> queries, int charWidth, string outFile)
{
    string newOutFile=outFile;
    int dot = newOutFile.rfind('.');
    newOutFile = newOutFile.substr(0,dot)+to_string(N)+newOutFile.substr(dot);
    ofstream out(newOutFile);


    //map<string,pair<float,int> > bestWidthForNgrams;
    map<string,map<int,float> > widthAPsForNgrams;

    cout<<"\n---find width---"<<N<<endl;
    
    /*int minWidth = max(8*stride,charWidth*(N-1));
    int maxWidth = charWidth*(N+1);
    minWidth += (charWidth*N - minWidth)%stride;
    maxWidth -= (maxWidth-charWidth*N)%stride;*/
    int minWidth = max(4*stride,charWidth*(N-1));
    int maxWidth = charWidth*(N+2);
    minWidth -= (minWidth%stride) ;
    maxWidth += ((stride-(maxWidth%stride))%stride) ;
    for (int inst=0; inst<queries.size(); inst++)
        out<<","<<queries[inst];
    out<<endl;
    for (int windowWidth=minWidth; windowWidth<=maxWidth; windowWidth+=stride)
    {
        out<<windowWidth;
        //windowWidths[N]=windowWidth;
        //corpus_embedded.erase(windowWidth);
        getEmbedding(windowWidth);
        for (int inst=0; inst<queries.size(); inst++)
        {
            string ngram = queries[inst];
            vector<SubwordSpottingResult> res = subwordSpot(queries[inst],1.0,windowWidth); //scores


            float ap = evalSubwordSpotting_singleScore(ngram, res, corpusXLetterStartBounds, corpusXLetterEndBounds,-1);
            out<<","<<ap;
            if (ap<0)
                continue;
            widthAPsForNgrams[ngram][windowWidth]=ap;
            //cout<<ngram<<" "<<windowWidth<<" : "<<ap<<endl;

        }
        out<<endl;
    }

    int sum=0;
    int count=0;
    map<int,list<float> > together;
    out<<"Ngram, best width:"<<endl;
    for (string ngram : queries)
    {
        const map<int,float>& maps = widthAPsForNgrams[ngram];
        float bestAP=-99999;
        int bestWindow=-1;
        for (auto q : maps)
        {
            if (q.second>bestAP)
            {
                bestAP = q.second;
                bestWindow = q.first;
            }
            together[q.first].push_back(q.second);
        }
        //assert(bestWindow>0);
        out<<ngram<<","<<bestWindow<<endl;
        if (bestWindow>0)
        {
            count++;
            sum+=bestWindow;
        }
    }
    out<<"Mean(class) best: "<<sum/(count+0.0)<<endl;
    float bestAP=-99999;
    int bestWindow=-1;
    for (auto p : together)
    {
        float sumA=0;
        for (float s : p.second)
            sum+=s;
        float ms = sumA/p.second.size();
        if (ms>bestAP)
        {
            bestAP=ms;
            bestWindow=p.first;
        }
    }
    out<<"Overall best: "<<bestWindow<<endl;
    out.close();
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
        vector<SubwordSpottingResult> resAccum = subwordSpot(ngram,1.0); //scores
        ap = evalSubwordSpotting_singleScore(ngram, resAccum, corpusXLetterStartBounds, corpusXLetterEndBounds,-1, NULL, &truesAccum, &allsAccum);
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
                resN = subwordSpot(ngram,next.imIdx,newX1,newX2,next.startX,next.endX,1.0);
#else          
                //Leave rectangular using preembedded (assumes sliding window size)
                resN = subwordSpot(ngram,next.imIdx,next.startX,1.0);
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

//Intended to mimic PHOCNet paper
void CNNSPPSpotter::evalFullWordSpotting(const Dataset* data, set<string> print, int doTheseFlags)
{
    setCorpus_dataset(data,true);

    bool doQbS = doTheseFlags&1;
    bool doQbE = doTheseFlags&2;

    float mAP=0;
    int mAPCount=0;

    map<string,list<int> > wordCounts;
    for (int i=0; i<corpus_dataset->size(); i++)
    {
        string word = corpus_dataset->labels()[i];
        wordCounts[word].push_back(i);
    }
    list<string> toSpotQbS;
    map<int,string> toSpotQbE;
    for (auto p : wordCounts)
    {
        toSpotQbS.push_back(p.first);
        if (p.second.size()>1)
        {
            for (int i : p.second)
                toSpotQbE[i]=p.first;
        }
    }


    //QbS
    if (doQbS)
    {
        for (string word : toSpotQbS)
        {
            float ap=0;
            
            multimap<float,int> resAccum = wordSpot(word); //scores
            ap = evalWordSpotting_singleScore(word, resAccum);
            assert(ap==ap);
            
            mAP+=ap;
            mAPCount++;

            if (print.size()>0 && print.find(word)!=print.end())
                cout<<word<<": "<<ap<<endl;
        }
        cout<<"QbS mAP: "<<(mAP/mAPCount)<<endl;
        mAP=0;
        mAPCount=0;
    }

    //QbE
    if (doQbE)
    {
        for (auto p : toSpotQbE)
        {
            string word = p.second;
            int inst = p.first;
            float ap=0;
            
            multimap<float,int> resAccum = wordSpot(inst); //scores
            ap = evalWordSpotting_singleScore(word, resAccum, inst);
            //for (int iii=0; iii<trues.size()/2; iii++)
            //    iter++;
            assert(ap==ap);
            
            mAP+=ap;
            mAPCount++;
        }
        cout<<"QbE mAP: "<<(mAP/mAPCount)<<endl;
    }
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
        vector<SubwordSpottingResult> res = subwordSpot(ngram,exemplars->image(inst),1.0); //scores
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
        vector<SubwordSpottingResult> res = subwordSpot(exemplars[inst],1.0); //scores
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
            total 
        }
        //cout <<"ap for ["<<gram<<"]: "<<(gramMap/gramCount)<<endl;
        
        cout<<"FULL map: "<<(map/queryCount)<<endl;
}*/
float CNNSPPSpotter::calcSuffixAP(const vector<SubwordSpottingResult>& res, string suffix, int* trueCount, int* wholeCount)
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
    /*
    for (int j=0; j<corpus_dataset->size(); j++)
    {
        int loc = corpus_dataset->labels()[j].rfind(suffix);
        if (loc == corpus_dataset->labels()[j].length()-suffix.length())
        {
            num_relevant++;
        }
    }*/
    vector<int> checked(corpus_dataset->size());
    for (int j=0; j<res.size(); j++)
    {
        SubwordSpottingResult r = res[j];
        size_t loc = corpus_dataset->labels()[r.imIdx].rfind(suffix);
        if (loc!=string::npos && loc==corpus_dataset->labels()[r.imIdx].length()-suffix.length())
        {

            scores.push_back(r.score);
            rel.push_back(true);
            num_relevant++;
            checked[r.imIdx]++;
        }
        else
        {
            scores.push_back(r.score);
            rel.push_back(false);
            checked[r.imIdx]++;
        }
    }
    //get words we didn't spot in
    for (int j=0; j<corpus_dataset->size(); j++)
    {
        int loc = corpus_dataset->labels().at(j).rfind(suffix);
        if (checked.at(j)==0 && loc!=string::npos &&  loc == corpus_dataset->labels().at(j).length()-suffix.length())
        {
            scores.push_back(maxScore);
            rel.push_back(true);
            num_relevant++;
            checked.at(j)++;
        }
        if (wholeCount!=NULL && suffix.compare(corpus_dataset->labels().at(j))==0)
            (*wholeCount) +=1;
    }
    if (num_relevant<1)
    {
        //cout <<" too few"<<endl;
        return -1;
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
    if (trueCount!=NULL)
        *trueCount=Nrelevants;
    //cout<<" num relv: "<<Nrelevants<<"  numTrumped: "<<numTrumped<<" numOff: "<<numOff<<"  ";
    return ap;
}
void CNNSPPSpotter::evalSuffixSpotting(const vector<string>& suffixes, const Dataset* data, string saveDir)
{
    setCorpus_dataset(data,true);

    if (saveDir.length()>0)
    {
        mkdir(saveDir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        cout<<"saving result images to "<<saveDir<<endl;
    }

    float map=0;
    int queryCount=0;
    float weightedMap=0;
    int totalCount=0;
    cout<<"suffix\tap\t#whole\t#total"<<endl;
    //#pragma omp parallel for
    for (int inst=0; inst<suffixes.size(); inst++)
    {
        string suffix = suffixes[inst];
        cout << flush;
        //int *rank = new int[other];//(int*)malloc(NRelevantsPerQuery[i]*sizeof(int));
        int Nrelevants = 0;
        float ap=0;
        
        //imshow("exe", suffixes->image(inst));
        //waitKey();
        vector<SubwordSpottingResult> res = suffixSpot(suffixes[inst],1.0); //scores
        int trueCount=0;
        int wholeCount=0;
        ap = calcSuffixAP(res,suffix,&trueCount,&wholeCount);
        assert(ap==ap);
        if (ap<0)
            continue;
        
        //#pragma omp critical (storeMAP)
        {
            queryCount++;
            map+=ap;
            totalCount+=trueCount;
            weightedMap+=ap*trueCount;
            cout <<suffix<<":\t"<<ap<<"\t"<<wholeCount<<"\t"<<trueCount<<endl;
            //cout<<"on spotting inst:"<<inst<<", "<<suffix<<"   ap: "<<ap<<endl;
            /*if (gram.compare(suffix)!=0)
            {
                if (gramCount>0)
                {
                    cout <<"ap for ["<<gram<<"]: "<<(gramMap/gramCount)<<endl;
                    gramCount=0;
                    gramMap=0;
                }
                gram=suffix;
            }
            gramMap+=ap;
            gramCount++;*/
        }
        cout <<endl;

        if (saveDir.length()>0)
        {
            string sufDir = saveDir+"/"+suffix+"/";
            mkdir(sufDir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            for (int i=0; i<20; i++)
            {
                Mat word;
                cvtColor(corpus_dataset->image(res[i].imIdx),word,CV_GRAY2BGR);
                for (int c=res[i].startX; c<=res[i].endX; c++)
                    for (int r=0; r<word.rows; r++)
                        word.at<Vec3b>(r,c)[0]*=0.5;
                imwrite(sufDir+to_string(i)+".png", word);
            }
        }
    }
        //cout <<"ap for ["<<gram<<"]: "<<(gramMap/gramCount)<<endl;
        
    cout<<"  suffix map: "<<(map/queryCount)<<endl;
    cout<<"weighted map: "<<(weightedMap/totalCount)<<endl;
}

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

void CNNSPPSpotter::timeEmbedding()
{
    for (int windowWidth=40; windowWidth<=140; windowWidth+=stride)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        getEmbedding(windowWidth);
        auto t2 = std::chrono::high_resolution_clock::now();
        int time = chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
        cout<<windowWidth<<","<<time<<endl;
    }
}
