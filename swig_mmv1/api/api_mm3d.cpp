#include "api_mm3d.h"
#include <iostream>

CamStenope *  CamOrientFromFile(std::string filename)
{
	cElemAppliSetFile anEASF(filename);
	return CamOrientGenFromFile(filename,anEASF.mICNM);
}

