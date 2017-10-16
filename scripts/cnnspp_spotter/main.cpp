#include "cnnspp_spotter.h"
#include "gwdataset.h"

//22 pix for name
int main(int argc, char** argv)
{

    if (argc<9)
    {
        cout<<"Tests various tasks using the spotter."<<endl;
        cout<<"usage: \n"<<argv[0]<<" featurizerModel.prototxt embedderModel.prototxt netWeights.caffemodel gpu(none or #) netScale(0.25) testCorpus imageDir ngramWindowWidths [- (full word spotting)] OR [time (timing embedding)] OR [segs.csv (subword spotting) [= ngramlist ngramlist ... [-out outDir] OR [-width charWidth outFile.csv (calc window widths)]]] OR [segs.csv ! toSpot.txt (subword respotting) depth repeat repeatDepth] OR [segs.csv ? ngram destDir (subword clustering)] OR [!  toSpot.txt (respotting full word) depth] OR [toSpot.txt (QbS subword)] OR [exemplars exemplarsDir [combine]] OR [lexicon.txt +(recognize)] OR [suffix suffixes.txt]"<<endl;
        exit(1);
    }
    string featurizerModel = argv[1];
    string embedderModel = argv[2];
    string netWeights = argv[3];
    bool normalizeEmbedding = true;// argv[4][0]!='d';
    int gpu = -1;
    if (argv[4][0]>='0' && argv[4][0]<='9')
        gpu = atoi(argv[4]);
    float netScale = atof(argv[5]);
    string testCorpus = argv[6];
    string imageDir = argv[7];
    string ngramWW = argv[8];
    GWDataset test(testCorpus,imageDir);
    if (argv[8+1][0]=='-')
    {
        cout<<"full word spotting"<<endl;
        CNNSPPSpotter spotter(featurizerModel, embedderModel,netWeights,"",gpu,normalizeEmbedding,netScale);
        spotter.evalFullWordSpotting(&test);
    }
    else if (string(argv[8+1]).compare("time")==0)
    {
        cout<<"time test"<<endl;
        CNNSPPSpotter spotter(featurizerModel, embedderModel,netWeights,ngramWW,gpu,normalizeEmbedding,netScale);
        spotter.setCorpus_dataset(&test,false);
        spotter.timeEmbedding();
    }
    else if (string(argv[8+1]).compare("suffix")==0)
    {
        cout<<"suffix spotting"<<endl;
        string suffixFile = argv[10];
        vector<string> suffixes;
        ifstream in(suffixFile);
        string line;
        while (getline(in,line))
        {
            suffixes.push_back(CNNSPPSpotter::lowercaseAndStrip(line));
        }
        CNNSPPSpotter spotter(featurizerModel, embedderModel,netWeights,ngramWW,gpu,normalizeEmbedding,netScale);
        spotter.evalSuffixSpotting(suffixes, &test);
    }
    else if (argc==9 || argv[9+1][0]=='+' || argv[9+1][0]=='!' || argv[9+1][0]=='=' || argv[9+1][0]=='?')
    {
        string queryFile=argv[8+1];
        if (queryFile.substr(queryFile.length()-4).compare(".csv") ==0)
        {

            vector< vector<int> > corpusXLetterStartBoundsRel;
            vector< vector<int> > corpusXLetterEndBoundsRel;
            ifstream in (argv[8+1]);
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

            if (argc==9)
            {
                cout<<"OLD subword spotting"<<endl;
                assert(false);
                /*
                set<int> ngrams={1,2,3};
                //ngrams.insert(2);
                CNNSPPSpotter spotter(featurizerModel, embedderModel,netWeights,ngramWW,gpu,normalizeEmbedding,netScale);
                spotter.setCorpus_dataset(&test,false);
                for (int N : ngrams)
                {
                    cout<<"Spotting "<<N<<"-grams."<<endl;
                    spotter.evalSubwordSpottingWithCharBounds(N, &corpusXLetterStartBoundsRel, &corpusXLetterEndBoundsRel);
                }*/
            }
            else if (argv[9+1][0]=='=')
            {
                cout<<"Subword spotting (for thesis)"<<endl;
                map<int,vector<string> > queries;
                set<string> ngrams;
                set<int> Ns;
                bool doWidths=false;
                int charWidth=-1;
                string outDir="";
                for (int i=0; i<10; i++)
                    cout<<argv[i]<<" ";
                for (int i=10; i<argc; i++)
                {
                    if (argv[i][0]=='-' && argv[i][1]=='o')
                    {
                        i++;
                        outDir=argv[i];
                        cout<<"set out dir: "<<outDir<<endl;
                        continue;
                    }
                    else if (argv[i][0]=='-' && argv[i][1]=='w')
                    {
                        i++;
                        charWidth=atoi(argv[i]);
                        i++;
                        outDir=argv[i];
                        doWidths=true;
                        cout<<"Calculating best window widths."<<endl;
                        continue;
                    }
                    cout<<argv[i]<<" ";
                    ifstream in (argv[i]);
                    while (getline(in,line))
                    {
                        string ngram = (CNNSPPSpotter::lowercaseAndStrip(line));
                        int N = ngram.length();
                        queries[N].push_back(ngram);
                        ngrams.insert(ngram);
                        Ns.insert(N);
                    }
                    in.close();
                }
                cout<<endl;


                CNNSPPSpotter spotter(featurizerModel, embedderModel,netWeights,ngramWW,gpu,normalizeEmbedding,netScale);
                spotter.setCorpus_dataset(&test,false);
                cout<<"--------------------------------"<<endl;
                if (doWidths)
                {
                    for (int N : Ns)
                    {
                        spotter.refineWindowSubwordSpottingWithCharBounds(N, &corpusXLetterStartBoundsRel, &corpusXLetterEndBoundsRel,queries[N], charWidth, outDir);
                        cout<<"--------------------------------"<<endl;
                    }
                }
                else
                {
                    for (int N : Ns)
                    {
                        cout<<"Spotting "<<N<<"-grams."<<endl;
                        spotter.evalSubwordSpottingWithCharBounds(N, &corpusXLetterStartBoundsRel, &corpusXLetterEndBoundsRel,queries[N], outDir);
                        cout<<"--------------------------------"<<endl;
                    }
                }
            }
            else if (argv[9+1][0]=='!')
            {
                cout<<"Respot subword"<<endl;
                assert(argc>13);
                ifstream in (argv[10+1]);
                string line;
                vector<string> toSpot;
                set<string> ngrams;
                while (getline(in,line))
                {
                    toSpot.push_back(CNNSPPSpotter::lowercaseAndStrip(line));
                    ngrams.insert(toSpot.back());
                }
                in.close();
                int numSteps=stoi(argv[11+1]);
                int numRepeat=stoi(argv[12+1]);
                int repeatSteps=stoi(argv[13+1]);
                CNNSPPSpotter spotter(featurizerModel, embedderModel,netWeights,ngramWW,gpu,normalizeEmbedding,netScale);
                spotter.evalSubwordSpottingRespot(&test, toSpot, numSteps, numRepeat, repeatSteps, &corpusXLetterStartBoundsRel, &corpusXLetterEndBoundsRel);
            }
            else if (argv[9+1][0]=='?')
            {
                cout<<"Cluster demo"<<endl;
                string ngram=argv[10+1];
                string destDir=argv[11+1];
                set<string> ngrams;
                ngrams.insert(ngram);
                CNNSPPSpotter spotter(featurizerModel, embedderModel,netWeights,ngramWW,gpu,normalizeEmbedding,netScale);
                spotter.setCorpus_dataset(&test);
                spotter.demonstrateClustering(destDir,ngram,&corpusXLetterStartBoundsRel, &corpusXLetterEndBoundsRel);
            }   
            else
                cout<<"Unrecognized command: "<<argv[9+1]<<endl;
        }
        else
        {
            ifstream in (queryFile);
            string line;
            vector<string> queries;
            set<string> ngrams;
            while (getline(in,line))
            {
                queries.push_back(line);
                ngrams.insert(queries.back());
            }
            in.close();
            if (argc==9 || argv[9+1][0]!='+')
            {
                CNNSPPSpotter spotter(featurizerModel, embedderModel,netWeights,ngramWW,gpu,normalizeEmbedding,netScale);
                spotter.evalSubwordSpotting(queries, &test);
            }
            else
            {
                CNNSPPSpotter spotter(featurizerModel, embedderModel,netWeights,ngramWW,gpu,normalizeEmbedding,netScale);
                spotter.evalRecognition(&test, queries);
            }
        }
            return 0;
    }
    else if (argv[8+1][0]=='!')
    {
        ifstream in (argv[9+1]);
        string line;
        vector<string> queries;
        while (getline(in,line))
            queries.push_back(CNNSPPSpotter::lowercaseAndStrip(line));
        in.close();
        int numSteps=stoi(argv[10+1]);
        CNNSPPSpotter spotter(featurizerModel, embedderModel,netWeights,ngramWW,gpu,normalizeEmbedding,netScale);
        spotter.evalFullWordSpottingRespot(&test, queries, numSteps,1,1);

    }
    else
    {


        string exemplarsFile = argv[8+1];
        string exemplarsDir = argv[9+1];
        GWDataset exemplars(exemplarsFile,exemplarsDir);
        set<string> ngrams;
        for (string l : exemplars.labels())
            ngrams.insert(l);
        if ( argc==10 )
        {
            CNNSPPSpotter spotter(featurizerModel, embedderModel,netWeights,ngramWW,gpu,normalizeEmbedding,netScale);
            spotter.evalSubwordSpotting(&exemplars, &test);
        }
        else
        {
            cout<<"Combine scoing not implemented"<<endl;
            //spotter.evalSubwordSpottingCombine(&exemplars, &test);
        }
    }
}

