#include "gwdataset.h"

GWDataset::GWDataset(const string& queries, const string& imDir_, bool inv, int minH, int maxH, int margin) : inv(inv)
{
    string imDir = imDir_;
    if (imDir[imDir.length()-1] != '/')
        imDir+="/";
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
    
    if (gtp)
    {
	//cout<<"WARNING: max image height set 200"<<endl;
	cout<<"WARNING: min image width/h set 20"<<endl;
    }
    else
    {
	cout<<"WARNING: min image height 32 and width set 20"<<endl;
    }
    
    
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
                if (curIm.cols<1)
                {
                    cout<<"Counld not open "<<curPathIm<<endl;
                    assert(curIm.cols>0);
                }
                if (inv)
                    curIm = 255-curIm;
            }
            getline(ss,part,' ');
            int x1=max(1,stoi(part)-margin);//;-1;
            getline(ss,part,' ');
            int y1=max(1,stoi(part)-margin);//;-1;
            getline(ss,part,' ');
            int x2=min(curIm.cols,stoi(part)+margin);//;-1;
            getline(ss,part,' ');
            int y2=min(curIm.rows,stoi(part)+margin);//;-1;

            if (y2-y1+1 < 20)
            {
                int dif = 20 -(y2-y1+1);
                y1 = max(0,y1-dif/2);
                y2 = min(curIm.rows-1,y2+dif/2 + (dif%2));
            }
            /*
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
            }*/
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
            //cout <<x1<<" "<<y1<<" "<<x2<<" "<<y2<<" ["<<label<<"]"<<endl;
            //imshow("readin",patch);
            //waitKey();
	    //////
	    /*if (patch.rows>200)
	    {
                double scale = 200.0/patch.rows;
                resize(patch,patch,Size(),1,scale);//preserve length for char seg gt
            }*/
	    if (patch.rows<20)
	    {
                double scale = 20.0/patch.rows;
                resize(patch,patch,Size(),scale,1);//preserve length for char seg gt
            }
	    /*if (patch.cols<32)
	    {
                double scale = 32.0/patch.cols;
                resize(patch,patch,Size(),scale,scale);
            }*/

            //////
        }
        else
        {
            stringstream ss(line);
            string part;
            getline(ss,part,' ');
            //regex_search(line,sm,qExt);
            patch=imread(imDir+part,CV_LOAD_IMAGE_GRAYSCALE);
            if (inv)
                patch = 255-patch;
            if (patch.rows*patch.cols <= 1)
                cout<<imDir+part<<"  line["<<wordImages.size()<<"]: "<<line<<endl;
            //
            if (patch.rows<32)
            {
                float scale = 32.0/patch.rows;
                resize(patch,patch,Size(),scale,scale);
            }
            if (patch.cols<20)
            {
                float scale = 20.0/patch.cols;
                resize(patch,patch,Size(),scale,scale);
            }
            //
            getline(ss,part,' ');
            label=part;
        }
        assert(patch.rows*patch.cols>1);
        /*patch.convertTo(patch,CV_32F);
        patch/=255;
        #if TEST_MODE_CNNSPP
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
        
        #if TEST_MODE_CNNSPP
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

GWDataset::GWDataset(const vector<tuple<int,int,int,int,string> >& words, const string& imDir_, bool inv, int minH, int maxH, int margin) : inv(inv)
{
    string imDir = imDir_;
    if (imDir[imDir.length()-1] != '/')
        imDir+="/";
    Mat curIm;
    string curPathIm;
    for (auto tup : words)
    {
        string pathIm = imDir+get<4>(tup);
        if (curPathIm.compare(pathIm)!=0)
        {
            curPathIm=pathIm;
            curIm = imread(curPathIm,CV_LOAD_IMAGE_GRAYSCALE);
            if (inv)
                curIm = 255-curIm;
        }
        int x1 = get<0>(tup);
        int y1 = get<1>(tup);
        int x2 = get<2>(tup);
        int y2 = get<3>(tup);
        if (y2-y1+1 < 30)
        {
            int dif = 30 -(y2-y1+1);
            y1 = max(0,y1-dif/2);
            y2 = min(curIm.rows-1,y2+dif/2 + (dif%2));
        }
        Rect loc(x1,y1,x2-x1+1,y2-y1+1);
        locs.push_back(loc);
        if (x1<0 || x1>=x2 || x2>=curIm.cols)
            cout<<"line: "<<line<<"  loc "<<x1<<" "<<y1<<" "<<x2<<" "<<y2<<endl;
        assert(x1>=0 && x1<x2);
        assert(x2<curIm.cols);
        assert(y1>=0 && y1<y2);
        assert(y2<curIm.rows);
        Mat patch = curIm(loc);
        if (patch.rows>200)
        {
            double scale = 200.0/patch.rows;
            resize(patch,patch,Size(),1,scale);//preserve length for char seg gt
        }
        if (patch.rows<20)
        {
            double scale = 20.0/patch.rows;
            resize(patch,patch,Size(),1,scale);//preserve length for char seg gt
        }
        wordImages.push_back(patch);
        _labels.push_back("UNKNOWN");
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
