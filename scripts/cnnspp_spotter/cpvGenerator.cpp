#include "cnnspp_spotter.h"
#include "gwdataset.h"


int main(int argc, char** argv)
{
    vector<string>* ngramsP= new vector<string>({"e","t","a","o","i","n","s","h","r","d","l","c","u","m","w","f","g","y","p","b","v","k","j","x","q","z","th","he","in","er","an","re","on","at","en","nd","ti","es","or","te","of","ed","is","it","al","ar","st","to","nt","ng","se","ha","as","ou","io","le","ve","co","me","de","hi","ri","ro","ic","ne","ea","ra","ce","li","ch","ll","be","ma","si","om","ur","ca","el","ta","la","ns","di","fo","ho","pe","ec","pr","no","ct","us","ac","ot","il","tr","ly","nc","et","ut","ss","so","rs","un","lo","wa","ge","ie","wh","ee","wi","em","ad","ol","rt","po","we","na","ul","ni","ts","mo","ow","pa","im","mi","ai","sh","the","and","ing","ion","tio","ent","ati","for","her","ter","hat","tha","ere","ate","his","con","res","ver","all","ons","nce","men","ith","ted","ers","pro","thi","wit","are","ess","not","ive","was","ect","rea","com","eve","per","int","est","sta","cti","ica","ist","ear","ain","one","our","iti","rat","nte","tin","ine","der","ome","man","pre","rom","tra","whi","ave","str","act","ill","ure","ide","ove","cal","ble","out","sti","tic","oun","enc","ore","ant","ity","fro","art","tur","par","red","oth","eri","hic","ies","ste","ght","ich","igh","und","you","ort","era","wer","nti","oul","nde","ind","tho","hou","nal","but","hav","uld","use","han","hin","een","ces","cou","lat","tor","ese","age","ame","rin","anc","ten","hen","min","eas","can","lit","cha","ous","eat","end","ssi","ial","les","ren","tiv","nts","whe","tat","abl","dis","ran","wor","rou","lin","had","sed","ont","ple","ugh","inc","sio","din","ral","ust","tan","nat","ins","ass","pla","ven","ell","she","ose","ite","lly","rec","lan","ard","hey","rie","pos","eme","mor","den","oug","tte","ned","rit","ime","sin","ast","any","orm","ndi","ona","spe","ene","hei","ric","ice","ord","omp","nes","sen","tim","tri","ern","tes","por","app","lar","ntr","eir","sho","son","cat","lle","ner","hes","who","mat","ase","kin","ost","ber","its","nin","lea","ina","mpl","sto","ari","pri","own","ali","ree","ish","des","ead","nst","sit","ses","ans","has","gre","ong","als","fic","ual","ien","gen","ser","unt","eco","nta","ace","chi","fer","tal","low","ach","ire","ang","sse","gra","mon","ffe","rac","sel","uni","ake","ary","wil","led","ded","som","owe","har","ini","ope","nge","uch","rel","che","ade","att","cia","exp","mer","lic","hem","ery","nsi","ond","rti","duc","how","ert","see","now","imp","abo","pec","cen","ris","mar","ens","tai","ely","omm","sur","hea"});
    vector<string>& ngrams = *ngramsP;
    for (string s : ngrams)
        cout<<s<<" ";
    cout<<endl;

    if (argc<9)
    {
        cout<<"Create training set for string-to-CPV network. Or just visualize"<<endl;
        cout<<"usage: \n"<<argv[0]<<" featurizerModel.prototxt embedderModel.prototxt netWeights.caffemodel [normalize/dont] netScale testCorpus imageDir [vecLen out.spec] OR [cpvVizDir]"<<endl;
        exit(1);
    }
    string featurizerModel = argv[1];
    string embedderModel = argv[2];
    string netWeights = argv[3];
    bool normalizeEmbedding = argv[4][0]=='n';
    float netScale = atof(argv[5]);
    string testCorpus = argv[6];
    string imageDir = argv[7];
    int fixedSize = -1;
    string outFileName = "";
    if (argc>9)
    {
        fixedSize = atoi(argv[8]); 
        outFileName = argv[9];
    }
    else
        outFileName = argv[8];
    GWDataset test(testCorpus,imageDir);
    set<int> ns={1,2,3};
    //ngrams.insert(2);
    CNNSPPSpotter spotter(featurizerModel, embedderModel,netWeights,ns,normalizeEmbedding,netScale);
    spotter.setCorpus_dataset(&test);


//    vector<string> ngrams;
//    ngrams.push_back("e");  ngrams.push_back("t");  ngrams.push_back("a");  ngrams.push_back("o");  ngrams.push_back("i");  ngrams.push_back("n");  ngrams.push_back("s");  ngrams.push_back("h");  ngrams.push_back("r");  ngrams.push_back("d");  ngrams.push_back("l");  ngrams.push_back("c");  ngrams.push_back("u");  ngrams.push_back("m");  ngrams.push_back("w");  ngrams.push_back("f");  ngrams.push_back("g");  ngrams.push_back("y");  ngrams.push_back("p");  ngrams.push_back("b");  ngrams.push_back("v");  ngrams.push_back("k");  ngrams.push_back("j");  ngrams.push_back("x");  ngrams.push_back("q");  ngrams.push_back("z");  
//    ngrams.push_back("th");  ngrams.push_back("he");  ngrams.push_back("in");  ngrams.push_back("er");  ngrams.push_back("an");  ngrams.push_back("re");  ngrams.push_back("on");  ngrams.push_back("at");  ngrams.push_back("en");  ngrams.push_back("nd");  ngrams.push_back("ti");  ngrams.push_back("es");  ngrams.push_back("or");  ngrams.push_back("te");  ngrams.push_back("of");  ngrams.push_back("ed");  ngrams.push_back("is");  ngrams.push_back("it");  ngrams.push_back("al");  ngrams.push_back("ar");  ngrams.push_back("st");  ngrams.push_back("to");  ngrams.push_back("nt");  ngrams.push_back("ng");  ngrams.push_back("se");  ngrams.push_back("ha");  ngrams.push_back("as");  ngrams.push_back("ou");  ngrams.push_back("io");  ngrams.push_back("le");  ngrams.push_back("ve");  ngrams.push_back("co");  ngrams.push_back("me");  ngrams.push_back("de");  ngrams.push_back("hi");  ngrams.push_back("ri");  ngrams.push_back("ro");  ngrams.push_back("ic");  ngrams.push_back("ne");  ngrams.push_back("ea");  ngrams.push_back("ra");  ngrams.push_back("ce");  ngrams.push_back("li");  ngrams.push_back("ch");  ngrams.push_back("ll");  ngrams.push_back("be");  ngrams.push_back("ma");  ngrams.push_back("si");  ngrams.push_back("om");  ngrams.push_back("ur");  ngrams.push_back("ca");  ngrams.push_back("el");  ngrams.push_back("ta");  ngrams.push_back("la");  ngrams.push_back("ns");  ngrams.push_back("di");  ngrams.push_back("fo");  ngrams.push_back("ho");  ngrams.push_back("pe");  ngrams.push_back("ec");  ngrams.push_back("pr");  ngrams.push_back("no");  ngrams.push_back("ct");  ngrams.push_back("us");  ngrams.push_back("ac");  ngrams.push_back("ot");  ngrams.push_back("il");  ngrams.push_back("tr");  ngrams.push_back("ly");  ngrams.push_back("nc");  ngrams.push_back("et");  ngrams.push_back("ut");  ngrams.push_back("ss");  ngrams.push_back("so");  ngrams.push_back("rs");  ngrams.push_back("un");  ngrams.push_back("lo");  ngrams.push_back("wa");  ngrams.push_back("ge");  ngrams.push_back("ie");  ngrams.push_back("wh");  ngrams.push_back("ee");  ngrams.push_back("wi");  ngrams.push_back("em");  ngrams.push_back("ad");  ngrams.push_back("ol");  ngrams.push_back("rt");  ngrams.push_back("po");  ngrams.push_back("we");  ngrams.push_back("na");  ngrams.push_back("ul");  ngrams.push_back("ni");  ngrams.push_back("ts");  ngrams.push_back("mo");  ngrams.push_back("ow");  ngrams.push_back("pa");  ngrams.push_back("im");  ngrams.push_back("mi");  ngrams.push_back("ai");  ngrams.push_back("sh");  
//    ngrams.push_back("the");  ngrams.push_back("and");  ngrams.push_back("ing");  ngrams.push_back("ion");  ngrams.push_back("tio");  ngrams.push_back("ent");  ngrams.push_back("ati");  ngrams.push_back("for");  ngrams.push_back("her");  ngrams.push_back("ter");  ngrams.push_back("hat");  ngrams.push_back("tha");  ngrams.push_back("ere");  ngrams.push_back("ate");  ngrams.push_back("his");  ngrams.push_back("con");  ngrams.push_back("res");  ngrams.push_back("ver");  ngrams.push_back("all");  ngrams.push_back("ons");  ngrams.push_back("nce");  ngrams.push_back("men");  ngrams.push_back("ith");  ngrams.push_back("ted");  ngrams.push_back("ers");  ngrams.push_back("pro");  ngrams.push_back("thi");  ngrams.push_back("wit");  ngrams.push_back("are");  ngrams.push_back("ess");  ngrams.push_back("not");  ngrams.push_back("ive");  ngrams.push_back("was");  ngrams.push_back("ect");  ngrams.push_back("rea");  ngrams.push_back("com");  ngrams.push_back("eve");  ngrams.push_back("per");  ngrams.push_back("int");  ngrams.push_back("est");  ngrams.push_back("sta");  ngrams.push_back("cti");  ngrams.push_back("ica");  ngrams.push_back("ist");  ngrams.push_back("ear");  ngrams.push_back("ain");  ngrams.push_back("one");  ngrams.push_back("our");  ngrams.push_back("iti");  ngrams.push_back("rat");  ngrams.push_back("nte");  ngrams.push_back("tin");  ngrams.push_back("ine");  ngrams.push_back("der");  ngrams.push_back("ome");  ngrams.push_back("man");  ngrams.push_back("pre");  ngrams.push_back("rom");  ngrams.push_back("tra");  ngrams.push_back("whi");  ngrams.push_back("ave");  ngrams.push_back("str");  ngrams.push_back("act");  ngrams.push_back("ill");  ngrams.push_back("ure");  ngrams.push_back("ide");  ngrams.push_back("ove");  ngrams.push_back("cal");  ngrams.push_back("ble");  ngrams.push_back("out");  ngrams.push_back("sti");  ngrams.push_back("tic");  ngrams.push_back("oun");  ngrams.push_back("enc");  ngrams.push_back("ore");  ngrams.push_back("ant");  ngrams.push_back("ity");  ngrams.push_back("fro");  ngrams.push_back("art");  ngrams.push_back("tur");  ngrams.push_back("par");  ngrams.push_back("red");  ngrams.push_back("oth");  ngrams.push_back("eri");  ngrams.push_back("hic");  ngrams.push_back("ies");  ngrams.push_back("ste");  ngrams.push_back("ght");  ngrams.push_back("ich");  ngrams.push_back("igh");  ngrams.push_back("und");  ngrams.push_back("you");  ngrams.push_back("ort");  ngrams.push_back("era");  ngrams.push_back("wer");  ngrams.push_back("nti");  ngrams.push_back("oul");  ngrams.push_back("nde");  ngrams.push_back("ind");  ngrams.push_back("tho");  ngrams.push_back("hou");  ngrams.push_back("nal");  ngrams.push_back("but");  ngrams.push_back("hav");  ngrams.push_back("uld");  ngrams.push_back("use");  ngrams.push_back("han");  ngrams.push_back("hin");  ngrams.push_back("een");  ngrams.push_back("ces");  ngrams.push_back("cou");  ngrams.push_back("lat");  ngrams.push_back("tor");  ngrams.push_back("ese");  ngrams.push_back("age");  ngrams.push_back("ame");  ngrams.push_back("rin");  ngrams.push_back("anc");  ngrams.push_back("ten");  ngrams.push_back("hen");  ngrams.push_back("min");  ngrams.push_back("eas");  ngrams.push_back("can");  ngrams.push_back("lit");  ngrams.push_back("cha");  ngrams.push_back("ous");  ngrams.push_back("eat");  ngrams.push_back("end");  ngrams.push_back("ssi");  ngrams.push_back("ial");  ngrams.push_back("les");  ngrams.push_back("ren");  ngrams.push_back("tiv");  ngrams.push_back("nts");  ngrams.push_back("whe");  ngrams.push_back("tat");  ngrams.push_back("abl");  ngrams.push_back("dis");  ngrams.push_back("ran");  ngrams.push_back("wor");  ngrams.push_back("rou");  ngrams.push_back("lin");  ngrams.push_back("had");  ngrams.push_back("sed");  ngrams.push_back("ont");  ngrams.push_back("ple");  ngrams.push_back("ugh");  ngrams.push_back("inc");  ngrams.push_back("sio");  ngrams.push_back("din");  ngrams.push_back("ral");  ngrams.push_back("ust");  ngrams.push_back("tan");  ngrams.push_back("nat");  ngrams.push_back("ins");  ngrams.push_back("ass");  ngrams.push_back("pla");  ngrams.push_back("ven");  ngrams.push_back("ell");  ngrams.push_back("she");  ngrams.push_back("ose");  ngrams.push_back("ite");  ngrams.push_back("lly");  ngrams.push_back("rec");  ngrams.push_back("lan");  ngrams.push_back("ard");  ngrams.push_back("hey");  ngrams.push_back("rie");  ngrams.push_back("pos");  ngrams.push_back("eme");  ngrams.push_back("mor");  ngrams.push_back("den");  ngrams.push_back("oug");  ngrams.push_back("tte");  ngrams.push_back("ned");  ngrams.push_back("rit");  ngrams.push_back("ime");  ngrams.push_back("sin");  ngrams.push_back("ast");  ngrams.push_back("any");  ngrams.push_back("orm");  ngrams.push_back("ndi");  ngrams.push_back("ona");  ngrams.push_back("spe");  ngrams.push_back("ene");  ngrams.push_back("hei");  ngrams.push_back("ric");  ngrams.push_back("ice");  ngrams.push_back("ord");  ngrams.push_back("omp");  ngrams.push_back("nes");  ngrams.push_back("sen");  ngrams.push_back("tim");  ngrams.push_back("tri");  ngrams.push_back("ern");  ngrams.push_back("tes");  ngrams.push_back("por");  ngrams.push_back("app");  ngrams.push_back("lar");  ngrams.push_back("ntr");  ngrams.push_back("eir");  ngrams.push_back("sho");  ngrams.push_back("son");  ngrams.push_back("cat");  ngrams.push_back("lle");  ngrams.push_back("ner");  ngrams.push_back("hes");  ngrams.push_back("who");  ngrams.push_back("mat");  ngrams.push_back("ase");  ngrams.push_back("kin");  ngrams.push_back("ost");  ngrams.push_back("ber");  ngrams.push_back("its");  ngrams.push_back("nin");  ngrams.push_back("lea");  ngrams.push_back("ina");  ngrams.push_back("mpl");  ngrams.push_back("sto");  ngrams.push_back("ari");  ngrams.push_back("pri");  ngrams.push_back("own");  ngrams.push_back("ali");  ngrams.push_back("ree");  ngrams.push_back("ish");  ngrams.push_back("des");  ngrams.push_back("ead");  ngrams.push_back("nst");  ngrams.push_back("sit");  ngrams.push_back("ses");  ngrams.push_back("ans");  ngrams.push_back("has");  ngrams.push_back("gre");  ngrams.push_back("ong");  ngrams.push_back("als");  ngrams.push_back("fic");  ngrams.push_back("ual");  ngrams.push_back("ien");  ngrams.push_back("gen");  ngrams.push_back("ser");  ngrams.push_back("unt");  ngrams.push_back("eco");  ngrams.push_back("nta");  ngrams.push_back("ace");  ngrams.push_back("chi");  ngrams.push_back("fer");  ngrams.push_back("tal");  ngrams.push_back("low");  ngrams.push_back("ach");  ngrams.push_back("ire");  ngrams.push_back("ang");  ngrams.push_back("sse");  ngrams.push_back("gra");  ngrams.push_back("mon");  ngrams.push_back("ffe");  ngrams.push_back("rac");  ngrams.push_back("sel");  ngrams.push_back("uni");  ngrams.push_back("ake");  ngrams.push_back("ary");  ngrams.push_back("wil");  ngrams.push_back("led");  ngrams.push_back("ded");  ngrams.push_back("som");  ngrams.push_back("owe");  ngrams.push_back("har");  ngrams.push_back("ini");  ngrams.push_back("ope");  ngrams.push_back("nge");  ngrams.push_back("uch");  ngrams.push_back("rel");  ngrams.push_back("che");  ngrams.push_back("ade");  ngrams.push_back("att");  ngrams.push_back("cia");  ngrams.push_back("exp");  ngrams.push_back("mer");  ngrams.push_back("lic");  ngrams.push_back("hem");  ngrams.push_back("ery");  ngrams.push_back("nsi");  ngrams.push_back("ond");  ngrams.push_back("rti");  ngrams.push_back("duc");  ngrams.push_back("how");  ngrams.push_back("ert");  ngrams.push_back("see");  ngrams.push_back("now");  ngrams.push_back("imp");  ngrams.push_back("abo");  ngrams.push_back("pec");  ngrams.push_back("cen");  ngrams.push_back("ris");  ngrams.push_back("mar");  ngrams.push_back("ens");  ngrams.push_back("tai");  ngrams.push_back("ely");  ngrams.push_back("omm");  ngrams.push_back("sur");  ngrams.push_back("hea");

    spotter.npvPrep(ngrams);
    //spotter.createNPVTrainingSet(ngrams,outFile);
    /*
    ofstream outFile(outFileName);
    Mat inputs(0,28*2 + 27*3 + 1 + 1 + 1,CV_32F);
    Mat outputs(0,fixedSize,CV_32F);
    for (int wordIndex=0; wordIndex<test.size(); wordIndex++)
    {
        //cout<<"on word "<<wordIndex<<" / "<<test.size()<<endl;
        string word = test.labels()[wordIndex];
        Mat cpv = spotter.cpv(wordIndex);
        resize(cpv,cpv,Size(fixedSize,cpv.rows));
        //before char, after char, ngram1, ngram2, ngram3, n-2, relPos, (wordLen-5)/5.0
        for (int letterIdx=0; letterIdx<26; letterIdx++)
        {
            //cout<<"."<<flush;
            string ngram = ngrams[ngramIndex];
            size_t loc = word.find(ngram);
            if (loc != string::npos)
            {
                size_t loc2 = word.find(ngram,loc+1);
                if (loc2 == string::npos)//we dont handel dup ngrams
                {
                    char before = 123;//{ is after z
                    if (loc>0)
                        before = word[loc-1];
                    char after = 123;
                    if (loc<word.length()-ngram.length())
                        after = word[loc+ngram.length()];
                    vector<int> beforeVec(28);
                    if (before>=97 && before<=123)
                        beforeVec[before-97]=1;
                    else
                        beforeVec[27]=1;
                    vector<int> afterVec(28);
                    if (after>=97 && after<=123)
                        afterVec[after-97]=1;
                    else
                        afterVec[27]=1;

                    vector<int> ngramVec1(27);
                    if (ngram[0]>=97 && ngram[0]<=122)
                        ngramVec1[ngram[0]-97]+=0.5;
                    else
                        ngramVec1[26]+=0.5;
                    vector<int> ngramVec2(27);
                    if (ngram.length()>1)
                    {
                        if (ngram[1]>=97 && ngram[1]<=122)
                            ngramVec2[ngram[1]-97]+=1.5;
                        else
                            ngramVec2[26]+=1.5;
                    }
                    vector<int> ngramVec3(27);
                    if (ngram.length()>2)
                    {
                        if (ngram[2]>=97 && ngram[2]<=222)
                            ngramVec3[ngram[2]-97]+=2.5;
                        else
                            ngramVec3[26]+=2.5;
                    }

                    int inIndex=0;
                    Mat in(1,28*2 + 27*3 + 1 + 1 + 1,CV_32F);
                    for (int v : beforeVec)
                        in.at<float>(0,inIndex++)=v;
                        //out<<v<<" ";
                    for (int v : afterVec)
                        in.at<float>(0,inIndex++)=v;
                        //out<<v<<" ";
                    for (int v : ngramVec1)
                        in.at<float>(0,inIndex++)=v;
                        //out<<v<<" ";
                    for (int v : ngramVec2)
                        in.at<float>(0,inIndex++)=v;
                        //out<<v<<" ";
                    for (int v : ngramVec3)
                        in.at<float>(0,inIndex++)=v;
                        //out<<v<<" ";
                    in.at<float>(0,inIndex++)=((float)ngram.length())-2;
                    in.at<float>(0,inIndex++)=loc/(0.0+word.length());
                    in.at<float>(0,inIndex++)=((float)word.length()-5)/5.0;
                    //assert((word.length()-5)/5.0 < 10);
                    //out<<ngram.length()-2<<" "<<loc/(0.0+word.length())<<" "<<(word.length()-5)/0.5<<endl;

                    Mat out = cpv.row(ngramIndex);
                    //for (int c=0; c<fixedSize; c++)
                    //{
                    //    out<<cpv.at<float>(ngramIndex,c)<<" ";
                    //}
                    //out<<endl;
                    if (inputs.rows==0)
                        inputs=in;
                    else
                        inputs.push_back(in);
                    if (outputs.rows==0)
                        outputs=out;
                    else
                        outputs.push_back(out);
                    
                }
            }
        }
        //cout<<endl;
    }
    double minV, maxV;
    minMaxIdx(outputs,&minV,&maxV);
    //outputs = 2*(outputs-minV)/(maxV-minV) - 1;
    //out<<"# data:"<<endl;
    outFile<<"# NPV generator, num instances"<<endl;
    outFile<<outputs.rows<<endl;
    outFile<<"# len of inputs"<<endl;
    outFile<<(28*2 + 27*3 + 1 + 1 + 1)<<endl;
    outFile<<"# len of outputs"<<endl;
    outFile<<fixedSize<<endl;
    //outFile<<"# output min="<<minV<<", max="<<maxV<<" converted to range -1,1"<<endl;

    for (int r=0; r<outputs.rows; r++)
    {
        for (int c=0; c<inputs.cols; c++)
            outFile<<inputs.at<float>(r,c)<<" ";
        outFile<<endl;
        for (int c=0; c<outputs.cols; c++)
            outFile<<outputs.at<float>(r,c)<<" ";
        outFile<<endl;
    }


    outFile.close();
    */

    if (fixedSize==-1)
    {
        for (int wordIndex=0; wordIndex<test.size(); wordIndex++)
        {
            //cout<<"on word "<<wordIndex<<" / "<<test.size()<<endl;
            Mat image = test.image(wordIndex);
            if (image.channels()!=3)
                cvtColor(image,image,CV_GRAY2BGR);
            string word = test.labels()[wordIndex];
            Mat cpv = spotter.cpv(wordIndex);
            double minV, maxV;
            minMaxLoc(cpv,&minV,&maxV);

            int letterSize=40;

            Mat draw = Mat::zeros(image.rows+26*letterSize,letterSize+std::max((int)image.cols,(int)(cpv.cols/netScale)),CV_8UC3);
            draw(Rect(letterSize,0,image.cols,image.rows))=image;
            for (int i=0; i<cpv.rows; i++)
            {
                string letter=" ";
                letter[0]=i+'a';

                cv::putText(draw,letter,cv::Point(1,image.rows+i*letterSize+15),cv::FONT_HERSHEY_COMPLEX_SMALL,0.75,cv::Scalar(250,250,255));
                for (int c=0; c<cpv.cols; c++)
                {
                    float v = cpv.at<float>(i,c);
                    float height = (letterSize-1)*(1-(v-minV)/(maxV-minV));
                    float color = 510-510*(v+1)/2;
                    Vec3b colorP(0,(color<255?color:255),(color<255?255:510-color));
                    for (int h=0; h<height; h++)
                        for (int cc=c/netScale; cc<(c+1)/netScale; cc++)
                            draw.at<Vec3b>(image.rows+letterSize*i+(letterSize-1)-h,letterSize+cc)=colorP;
                }
            }
            imwrite(outFileName+"/"+to_string(wordIndex)+"_"+word+".png",draw);
        }

    }

    delete ngramsP;
}

