#include "GpGpu/GpGpuTools.h"

void GpGpuTools::Memcpy2Dto1D( float** dataImage2D, float* dataImage1D, uint2 dimDest, uint2 dimSource )
{

	for (uint j = 0; j < dimSource.y ; j++)
		memcpy(  dataImage1D + dimDest.x * j , dataImage2D[j],  dimSource.x * sizeof(float));			

}

std::string GpGpuTools::GetImagesFolder()
{

#ifdef _WIN32

	TCHAR name [ UNLEN + 1 ];
	DWORD size = UNLEN + 1;
	GetUserName( (TCHAR*)name, &size );

	std::string suname = name;
	std::string ImagesFolder = "C:\\Users\\" + suname + "\\Pictures\\";
#else
	struct passwd *pw = getpwuid(getuid());

	const char *homedir = pw->pw_dir;


	std::string ImagesFolder = std::string(homedir) + "/Images/";

#endif

	return ImagesFolder;
}


void GpGpuTools::OutputReturn( char * out )
{
#ifndef DISPLAYOUTPUT
	return;
#endif

	std::cout << std::string(out) << "\n";

}

