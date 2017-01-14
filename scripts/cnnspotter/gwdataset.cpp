#include "gwdataset.h"

GWDataset::GWDataset(const string& queries, const string& imDir, int minH, int maxH, int margin) 
{
    if (queries.find_last_of('/') != string::npos)
        name=queries.substr(queries.find_last_of('/')+1);
    else
        name=queries;
    string extension = queries.substr(queries.find_last_of('.')+1);
    bool gtp=extension.compare("gtp")==0;

    ifstream fileQueries(queries);
    assert(fileQueries.good());
    //2700270.tif 519 166 771 246 orders
    string line;
    //regex qExtGtp("(\\S+\\.\\S+) (\\d+) (\\d+) (\\d+) (\\d+) (\\w+)");
    //regex qExt("(\\S+\\.\\S+) (\\w+)");
    
    
    
    string curPathIm="";
    Mat curIm;
    
    while (getline(fileQueries,line))
    {
        //smatch sm;
        Mat patch;
        string label;
        if (gtp)
        {
            //regex_search(line,sm,qExtGtp);
            stringstream ss(line);
            string part;
            getline(ss,part,' ');

            string pathIm=imDir+string(part);
            pathIms.push_back(pathIm);
            
            if (curPathIm.compare(pathIm)!=0)
            {
                curPathIm=pathIm;
                curIm = imread(curPathIm,CV_LOAD_IMAGE_GRAYSCALE);
            }
            getline(ss,part,' ');
            int x1=max(1,stoi(part)-margin);//;-1;
            getline(ss,part,' ');
            int y1=max(1,stoi(part)-margin);//;-1;
            getline(ss,part,' ');
            int x2=min(curIm.cols,stoi(part)+margin);//;-1;
            getline(ss,part,' ');
            int y2=min(curIm.rows,stoi(part)+margin);//;-1;
            if ((y2-y1)/2 > x2-x1) //This is to ensure we don't warp inputs to the net
            {
                float dif = ((y2-y1)/2.0-(x2-x1))/2.0;
                x1 = x1-floor(dif);
                x2 = x2+ceil(dif);
                if (x1<0)
                {
                    x2-=x1;
                    x1=0;
                }
                if (x2>=curIm.cols)
                {
                    x1-=x2-(curIm.cols+1);
                    x2 = curIm.cols-1;
                }
                if (x1<0)
                    x1=0;
            }
            Rect loc(x1,y1,x2-x1+1,y2-y1+1);
            locs.push_back(loc);
            if (x1<0 || x1>=x2 || x2>=curIm.cols)
                cout<<"line: "<<line<<"  loc "<<x1<<" "<<y1<<" "<<x2<<" "<<y2<<endl;
            assert(x1>=0 && x1<x2);
            assert(x2<curIm.cols);
            assert(y1>=0 && y1<y2);
            assert(y2<curIm.rows);
            patch = curIm(loc);
            getline(ss,part,' ');
            label=part;
            ///
            cout <<x1<<" "<<y1<<" "<<x2<<" "<<y2<<" ["<<label<<"]"<<endl;
            imshow("readin",patch);
            waitKey();
        }
        else
        {
            stringstream ss(line);
            string part;
            getline(ss,part,' ');
            //regex_search(line,sm,qExt);
            patch=imread(imDir+part,CV_LOAD_IMAGE_GRAYSCALE);
            if (patch.rows*patch.cols <= 1)
                cout<<imDir+part<<"  line["<<wordImages.size()<<"]: "<<line<<endl;
            getline(ss,part,' ');
            label=part;
        }
        assert(patch.rows*patch.cols>1);
        /*patch.convertTo(patch,CV_32F);
        patch/=255;
        #if TEST_MODE
        if (wordImages.size()==0)
            cout << "pre canary "<<patch.at<float>(0,0)<<endl;
        #endif
        
        double m;
        minMaxIdx(patch,NULL,&m);
        if (m<0.2)
            patch*=0.2/m;
        
        if (patch.rows>maxH)
        {
            double ratio = (maxH+0.0)/patch.rows;
            resize(patch,patch,Size(),ratio,ratio,INTER_CUBIC);
        }
        else if (patch.rows<minH)
        {
            double ratio = (maxH+0.0)/patch.rows;
            resize(patch,patch,Size(),ratio,ratio,INTER_CUBIC);
        }
        
        #if TEST_MODE
        if (wordImages.size()==0)
            cout << "pre canary "<<patch.at<float>(0,0)<<endl;
        #endif
        
        patch*=255;
        patch.convertTo(patch,CV_8U);
        */
        
        wordImages.push_back(patch);
        _labels.push_back(label);
    }

}

const vector<string>& GWDataset::labels() const
{
    return _labels;
}
int GWDataset::size() const
{
    return _labels.size();
}
const Mat GWDataset::image(unsigned int i) const
{
    return wordImages.at(i);
}
/*
int xa=min(stoi(sm[2]),stoi(sm[3]));
        int xb=max(stoi(sm[2]),stoi(sm[3]));
        int ya=min(stoi(sm[4]),stoi(sm[5]));
        int yb=max(stoi(sm[4]),stoi(sm[5]));
        
        int x1=max(0,xa);
        int x2=min(curIm.cols-1,xb);
        int y1=max(0,ya);
        int y2=min(curIm.rows-1,yb);
*/

string GWDataset::getName() const {return name;}
