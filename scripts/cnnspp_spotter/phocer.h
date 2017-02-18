#include <vector>
#include <map>
#include <iostream>
#include <ifstream>

using namespace std;

class PHOCer
{
    PHOCer(string bigramfile);
    vector<float> makePHOC(string word);

private:
    vector<int> phoc_levels;
    vector<char> unigrams;
    vector<int> phoc_levels_bi;
    vector<string> bigrams;


    map<char,int> vocUni2pos;

    map<std::string,int> vocBi2pos;



    int phocSize;

    int phocSize_bi;
};
