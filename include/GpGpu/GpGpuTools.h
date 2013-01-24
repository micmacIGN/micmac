#pragma once

#include "GpGpu/helper_math_extented.cuh"

#ifdef _WIN32
	#include <Lmcons.h>
#endif

class GpGpuTools
{

public:
	
	GpGpuTools(){};
	
	~GpGpuTools(){};
	
	//					Save to file image PGM
	static bool			Array1DtoImageFile(float* dataImage, char* pathFile, uint2 dimImage, float factor = 1.0f);
	
	//					Convert array 2D to linear array
	static void			memcpy2Dto1D(float** dataImage2D, float* dataImage1D, uint2 dimDest, uint2 dimSource);

	//					renvoi la dossier image de l'utilisateur
	static std::string	getImagesFolder();

	//					divise toutes les valeurs du tableau par un facteur
	//static void			divideArray(float* data, float factor);

};



