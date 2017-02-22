#include "phocer.h"

//copied from EmbAttSpotter
void PHOCer::computePhoc(string str, map<char,int> vocUni2pos, map<string,int> vocBi2pos, int Nvoc, vector<int> levels, int descSize, vector<float>* out)
{
    int strl = str.length();

    int doUnigrams = vocUni2pos.size()!=0;
    int doBigrams = vocBi2pos.size()!=0;

    /* For each block */
    //float *p = out;
    int p=0;
    for (int level : levels)
    {
        /* For each split in that level */
        for (int ns=0; ns < level; ns++)
        {
            float starts = ns/(float)level;
            float ends = (ns+1)/(float)level;

            /* For each character */
            if (doUnigrams)
            {
                for (int c=0; c < strl; c++)
                {
                    if (vocUni2pos.count(str[c])==0)
                    {
                        /* Character not included in dictionary. Skipping.*/
                        continue;
                    }
                    int posOff = vocUni2pos[str[c]]+p;
                    float startc = c/(float)strl;
                    float endc = (c+1)/(float)strl;

                    /* Compute overlap over character size (1/strl)*/
                    if (endc < starts || ends < startc) continue;
                    float start = (starts > startc)?starts:startc;
                    float end = (ends < endc)?ends:endc;
                    float ov = (end-start)*strl;
                    #if HARD
                    if (ov >=0.48)
                    {
                        //p[posOff]+=1;
                        //out.at<float>(posOff,instance)+=1;
                        out->at(posOff)+=1;
                    }
                    #else
                    //p[posOff] = max(ov, p[posOff]);
                    //out.at<float>(posOff,instance)=max(ov, out.at<float>(posOff,instance));
                    out->at(posOff) = max(ov, out->at(posOff));
                    #endif
                }
            }
            if (doBigrams)
            {
                for (int c=0; c < strl-1; c++)
                {
                    string sstr=str.substr(c,2);
                    if (vocBi2pos.count(sstr)==0)
                    {
                        /* Character not included in dictionary. Skipping.*/
                        continue;
                    }
                    int posOff = vocBi2pos[sstr]+p;
                    float startc = c/(float)strl;
                    float endc = (c+2)/(float)strl;

                    /* Compute overlap over bigram size (2/strl)*/
                    if (endc < starts || ends < startc){ continue;}
                    float start = (starts > startc)?starts:startc;
                    float end = (ends < endc)?ends:endc;
                    float ov = (end-start)*strl/2.0;
                    if (ov >=0.48)
                    {
                        //p[posOff]+=1;
                        //out.at<float>((out.rows-descSize)+posOff,instance)+=1;
                        out->at((out->size()-descSize)+posOff)+=1;
                    }
                }
            }
            p+=Nvoc;
        }
    }
    return;
}

PHOCer::PHOCer()
{
    //string bigramfile = argv[3];

    phoc_levels = {2, 3, 4, 5};
    unigrams = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};
    if (true)
    {
        for (char n : "0123456789") {
            //cout << "added numeral "<<n<<endl;
            if (n!=0x00)
                unigrams.push_back(n);
        }
    }
    phoc_levels_bi = {2};
    string bigram;
    bigrams={
        "er",
        "in",
        "es",
        "ti",
        "te",
        "at",
        "on",
        "an",
        "en",
        "st",
        "al",
        "re",
        "is",
        "ed",
        "le",
        "ra",
        "ri",
        "li",
        "ar",
        "ng",
        "ne",
        "ic",
        "or",
        "nt",
        "ss",
        "ro",
        "la",
        "se",
        "de",
        "co",
        "ca",
        "ta",
        "io",
        "it",
        "si",
        "us",
        "ea",
        "ac",
        "el",
        "ma",
        "na",
        "ni",
        "tr",
        "ch",
        "di",
        "ia",
        "et",
        "to",
        "un",
        "ns"/*,
        "ll",
        "ec",
        "me",
        "lo",
        "sc",
        "ol",
        "as",
        "he",
        "ly",
        "ce",
        "nd",
        "il",
        "pe",
        "sa",
        "mi",
        "rs",
        "ve",
        "ou",
        "th",
        "sp",
        "ur",
        "om",
        "ha",
        "sh",
        "nc"*/
    };
    /* Prepare dict */


    vocUni2pos;
    for (int i=0; i<unigrams.size(); i++)
    {
        vocUni2pos[unigrams[i]] = i;
    }

    vocBi2pos;
    for (int i=0; i<bigrams.size(); i++)
    {
        vocBi2pos[bigrams[i]] = i;
    }


    int totalLevels = 0;
    for (int level : phoc_levels)
    {
        totalLevels+=level;
    }

    phocSize = totalLevels*unigrams.size();

    phocSize_bi=0;
    for (int level : phoc_levels_bi)
    {
        phocSize_bi+=level*bigrams.size();
    }
    assert(phocSize+phocSize_bi==604);
}
    
    
   
vector<float> PHOCer::makePHOC(string word)
{
    vector<float> phoc(phocSize+phocSize_bi);
    computePhoc(word, vocUni2pos, map<string,int>(),unigrams.size(), phoc_levels, phocSize, &phoc);
    computePhoc(word, map<char,int>(), vocBi2pos,bigrams.size(), phoc_levels_bi, phocSize_bi, &phoc);
    return phoc;
}

