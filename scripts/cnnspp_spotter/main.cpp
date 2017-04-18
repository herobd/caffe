#include "cnnspp_spotter.h"
#include "gwdataset.h"


int main(int argc, char** argv)
{

    if (argc<9)
    {
        cout<<"usage: \n"<<argv[0]<<" featurizerModel.prototxt embedderModel.prototxt netWeights.caffemodel [normalize/dont] netScale testCorpus imageDir [segs.csv] OR [segs.csv ! toSpot.txt (respotting) depth repeat repeatDepth] OR [toSpot.txt (QbS)] OR [exemplars exemplarsDir [combine]] OR [lexicon.txt +(recognize)]"<<endl;
        exit(1);
    }
    string featurizerModel = argv[1];
    string embedderModel = argv[2];
    string netWeights = argv[3];
    bool normalizeEmbedding = argv[4][0]=='n';
    float netScale = atof(argv[5]);
    string testCorpus = argv[6];
    string imageDir = argv[7];
    CNNSPPSpotter spotter(featurizerModel, embedderModel,netWeights,normalizeEmbedding,netScale);
    GWDataset test(testCorpus,imageDir);
    if (argc==9 || argv[9][0]=='+' || argv[9][0]=='!')
    {
        string queryFile=argv[8];
        if (queryFile.substr(queryFile.length()-4).compare(".csv") ==0)
        {

            vector< vector<int> > corpusXLetterStartBoundsRel;
            vector< vector<int> > corpusXLetterEndBoundsRel;
            ifstream in (argv[8]);
            string line;
            //getline(in,line);//header
            while (getline(in,line))
            {
                string s;
                std::stringstream ss(line);
                getline(ss,s,',');
                if (s.compare(test.labels()[corpusXLetterStartBoundsRel.size()])!=0)
                    cout<<"["<<corpusXLetterStartBoundsRel.size()<<"]: "<<s<<" != "<<test.labels()[corpusXLetterStartBoundsRel.size()]<<endl;
                assert(s.compare(test.labels()[corpusXLetterStartBoundsRel.size()])==0);
                getline(ss,s,',');
                getline(ss,s,',');//x1
                int x1=stoi(s);
                getline(ss,s,',');
                getline(ss,s,',');//x2
                getline(ss,s,',');
                vector<int> lettersStartRel, lettersEndRel;

                //The x1/x2 may have been shifted is the image was tall-rectangular
                x1 = test.getLoc(corpusXLetterStartBoundsRel.size()).x;
                //x2 = x1 + test.getLoc(corpusXLetterStartBoundsRel.size()).width-1;
                while (getline(ss,s,','))
                {
                    lettersStartRel.push_back(stoi(s)-x1);
                    getline(ss,s,',');
                    lettersEndRel.push_back(stoi(s)-x1);
                    //getline(ss,s,',');//conf
                }
                corpusXLetterStartBoundsRel.push_back(lettersStartRel);
                corpusXLetterEndBoundsRel.push_back(lettersEndRel);
            }
            in.close();

            
            if (argv[9][0]!='!')
            {
                spotter.evalSubwordSpottingWithCharBounds(&test, &corpusXLetterStartBoundsRel, &corpusXLetterEndBoundsRel);
            }
            else
            {
                assert(argc>13);
                ifstream in (argv[10]);
                string line;
                vector<string> toSpot;
                while (getline(in,line))
                    toSpot.push_back(CNNSPPSpotter::lowercaseAndStrip(line));
                in.close();
                int numSteps=stoi(argv[11]);
                int numRepeat=stoi(argv[12]);
                int repeatSteps=stoi(argv[13]);
                spotter.evalSubwordSpottingRespot(&test, toSpot, numSteps, numRepeat, repeatSteps, &corpusXLetterStartBoundsRel, &corpusXLetterEndBoundsRel);
            }
        }
        else
        {
            ifstream in (queryFile);
            string line;
            vector<string> queries;
            while (getline(in,line))
                queries.push_back(line);
            in.close();
            if (argc==9 || argv[9][0]!='+')
                spotter.evalSubwordSpotting(queries, &test);
            else
                spotter.evalRecognition(&test, queries);
        }
            return 0;
    }


    string exemplarsFile = argv[8];
    string exemplarsDir = argv[9];
    GWDataset exemplars(exemplarsFile,exemplarsDir);
    
    if ( argc==10 )
        spotter.evalSubwordSpotting(&exemplars, &test);
    else
    {
        cout<<"Combine scoing not implemented"<<endl;
        //spotter.evalSubwordSpottingCombine(&exemplars, &test);
    }
}

