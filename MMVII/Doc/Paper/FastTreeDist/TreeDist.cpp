//  g++ TreeDist.cpp  -std=c++11
#include "TreeDist.h"

int main(int argc,char ** argv)
{
    NS_MMVII_FastTreeDist::AllBenchFastTreeDist(true);

    std::cout << "############################################\n";

    NS_MMVII_FastTreeDist::StatTimeBenchFastTreeDist();
}

