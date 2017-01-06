#include "cnnspotter.h"
#include <set>

int sort_xxx(const void *x, const void *y) {
    if (*(int*)x > *(int*)y) return 1;
    else if (*(int*)x < *(int*)y) return -1;
    else return 0;
}

/*void CNNSpotter::eval(const Dataset* data)
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

//This is a testing function for the simulator
#define LIVE_SCORE_OVERLAP_THRESH 0.65
float CNNSpotter::evalSubwordSpotting_singleScore(string ngram, const vector<SubwordSpottingResult>& res, const vector< vector<int> >* corpusXLetterStartBounds, const vector< vector<int> >* corpusXLetterEndBounds) const
{
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
    int l=ngram.length()-1;
    for (int j=0; j<res.size(); j++)
    {
        SubwordSpottingResult r = res[j];
        size_t loc = corpus_dataset->labels()[r.imIdx].find(ngram);
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
            float myOverlap = ( min(corpusXLetterEndBounds->at(r.imIdx)[loc+l], r.endX) 
                                - max(corpusXLetterStartBounds->at(r.imIdx)[loc], r.startX) ) 
                              /
                              ( max(corpusXLetterEndBounds->at(r.imIdx)[loc+l], r.endX) 
                                - min(corpusXLetterStartBounds->at(r.imIdx)[loc], r.startX) );
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
                                            - min(corpusXLetterStartBounds->at(res[oi].imIdx)[loc], res[oi].startX) );
                    if (otherOverlap > myOverlap) {
                        other=true;
                        break;
                    }
                }
                if (other)
                {
                    scores.push_back(r.score);
                    rel.push_back(false);
                }
                else if (myOverlap > LIVE_SCORE_OVERLAP_THRESH)
                {
                    scores.push_back(r.score);
                    rel.push_back(true);
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
    
    /* Get mAP and store it */
    for(int j=0;j<Nrelevants;j++){
        /* if rank[i] >=k it was not on the topk. Since they are sorted, that means bail out already */
        
        float prec_at_k =  ((float)(j+1))/(rank[j]+1);
        //mexPrintf("prec_at_k: %f\n", prec_at_k);
        ap += prec_at_k;            
    }
    ap/=Nrelevants;
    
   return ap;
}


void CNNSpotter::evalSubwordSpotting(const Dataset* exemplars, /*string exemplars_locations,*/ const Dataset* data)
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

    float map=0;
    int queryCount=0;
    float gramMap=0;
    string gram="";
    int gramCount=0;
    #pragma omp parallel for
    for (int inst=0; inst<exemplars->size(); inst++)
    {
        string ngram = exemplars->labels()[inst];
        //cout <<"on spotting inst:"<<inst<<", "<<ngram;
        //cout << flush;
        //int *rank = new int[other];//(int*)malloc(NRelevantsPerQuery[i]*sizeof(int));
        int Nrelevants = 0;
        float ap=0;
        
        float bestS=-99999;
        //imshow("exe", exemplars->image(inst));
        //waitKey();
        vector<SubwordSpottingResult> res = subwordSpot(exemplars->image(inst)); //scores
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
        
        /* Get mAP and store it */
        for(int j=0;j<Nrelevants;j++){
            /* if rank[i] >=k it was not on the topk. Since they are sorted, that means bail out already */
            
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
        
    }
        //cout <<"ap for ["<<gram<<"]: "<<(gramMap/gramCount)<<endl;
        
    cout<<"FULL map: "<<(map/queryCount)<<endl;
}
/*void CNNSpotter::evalSubwordSpottingCombine(const Dataset* exemplars, const Dataset* data)
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
