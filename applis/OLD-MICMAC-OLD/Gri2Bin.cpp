#include <iostream>
#include "cOrientationGrille.h"


int main(int argc, char **argv)
{
	if (argc!=3)
	{
		std::cout << "usage : toto.gri toto.bin"<<std::endl;
		return 1;
	}
	std::string nomGri(argv[1]);
	std::string nomBin(argv[2]);
	OrientationGrille gri(nomGri);
	gri.WriteBinary(nomBin);
	return 0;
}
