#include "cnnspotter.h"
#include "gwdataset.h"


int main(int argc, char** argv)
{

    if (argc<7 || argc>8)
    {
        cout<<"usage: \n"<<argv[0]<<" netModel.prototxt netWeights.caffemodel testCorpus imageDir exemplars exemplarsDir [combine]"<<endl;
        exit(0);
    }
    string netModel = argv[1];
    string netWeights = argv[2];
    string testCorpus = argv[3];
    string imageDir = argv[4];
    string exemplarsFile = argv[5];
    string exemplarsDir = argv[6];
    CNNSpotter spotter(netModel,netWeights);
    GWDataset test(testCorpus,imageDir);
    GWDataset exemplars(exemplarsFile,exemplarsDir);
    
    if ( argc==7 )
        spotter.evalSubwordSpotting(&exemplars, &test);
    else
    {
        cout<<"Combine scoing not implemented"<<endl;
        //spotter.evalSubwordSpottingCombine(&exemplars, &test);
    }
}

