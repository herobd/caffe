#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <assert.h>

using namespace std;

#define USE_PHOCNET 1

class PHOCer
{
public:
    PHOCer();
    vector<float> makePHOC(string word);
    int length() {return phocSize+phocSize_bi;}

private:
    vector<int> phoc_levels;
    vector<char> unigrams;
    vector<int> phoc_levels_bi;
    vector<string> bigrams;


    map<char,int> vocUni2pos;

    map<std::string,int> vocBi2pos;



    int phocSize;

    int phocSize_bi;
    void computePhoc(string str, map<char,int> vocUni2pos, map<string,int> vocBi2pos, int Nvoc, vector<int> levels, int descSize, vector<float>* out);
};
