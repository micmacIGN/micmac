#include "api/api_mm3d.h"
#include <cstdio>

//g++ -c -std=c++11 -fPIC -Wall -Werror  -O2  -march=native -I. -I../include/ -DFORSWIG test_er.cpp -o test_er.o
//g++ ./api/api_mm3d.o mm3d_wrap.o test_er.o ../lib/libelise.a -lX11 -lXext -lm -ldl -lpthread ../lib/libANN.a `pkg-config --libs python3-embed` `pkg-config --libs python3` -Wl,--warn-unresolved-symbols -o test_er
int main()
{
	std::string aImPat = ".*png";
	std::string aSH = "Ratafia";
	std::string aDir = "./";
	std::string InCal = "Calib";
	
	//std::vector<RelMotion> aRMVec = GetRelMotionSet(aImPat,aSH,aDir,InCal);
	
	std::cout << "ewelina tests: " << aImPat << " " << aSH << " " << aDir << " " << InCal << "\n";

	return 0;
}
