#include "cnnspotter.h"
#include "gwdataset.h"


int main(int argc, char** argv)
{

    if (argc!=6 && argc!=7 && argc!=8)
    {
        cout<<"usage: \n"<<argv[0]<<" netModel.prototxt netWeights.caffemodel testCorpus imageDir ( segs.csv OR exemplars exemplarsDir [combine] )"<<endl;
        exit(0);
    }
    string netModel = argv[1];
    string netWeights = argv[2];
    string testCorpus = argv[3];
    string imageDir = argv[4];
    CNNSpotter spotter(netModel,netWeights);
    GWDataset test(testCorpus,imageDir);
    if (argc==6)
    {

        vector< vector<int> > corpusXLetterStartBoundsRel;
        vector< vector<int> > corpusXLetterEndBoundsRel;
        ifstream in (argv[5]);
        string line;
        //getline(in,line);//header
        while (getline(in,line))
        {
            string s;
            std::stringstream ss(line);
            getline(ss,s,',');
            if (s.compare(test.labels()[corpusXLetterStartBoundsRel.size()])!=0)
                cout<<s<<" != "<<test.labels()[corpusXLetterStartBoundsRel.size()]<<endl;
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

        spotter.evalSubwordSpottingWithCharBounds(&test, &corpusXLetterStartBoundsRel, &corpusXLetterEndBoundsRel);
        return 0;
    }


    string exemplarsFile = argv[5];
    string exemplarsDir = argv[6];
    GWDataset exemplars(exemplarsFile,exemplarsDir);
    
    if ( argc==7 )
        spotter.evalSubwordSpotting(&exemplars, &test);
    else
    {
        cout<<"Combine scoing not implemented"<<endl;
        //spotter.evalSubwordSpottingCombine(&exemplars, &test);
    }
}

