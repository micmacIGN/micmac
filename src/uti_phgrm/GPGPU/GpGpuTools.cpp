#include "GpGpu/GpGpuTools.h"


bool GpGpuTools::Array1DtoImageFile(float* dataImage, char* fileName, uint2 dimImage, float factor)
{
	float* image	= new float[size(dimImage)];
	
	if (factor != 1.0f)
		for (uint j = 0; j < dimImage.y ; j++)	
			for (uint i = 0; i < dimImage.x ; i++)
			{
				int id = j * dimImage.x + i;
				image[id] = dataImage[id] / factor;

			}
	
	std::string pathfileImage = std::string(getImagesFolder()) + std::string(fileName);

	std::cout <<  pathfileImage << "\n";
	bool r = sdkSavePGM<float>(pathfileImage.c_str(), image, dimImage.x,dimImage.y);

	delete[] image;

	return r;
}

void GpGpuTools::memcpy2Dto1D( float** dataImage2D, float* dataImage1D, uint2 dimDest, uint2 dimSource )
{

	for (uint j = 0; j < dimSource.y ; j++)
		memcpy(  dataImage1D + dimDest.x * j , dataImage2D[j],  dimSource.x * sizeof(float));			

}

std::string GpGpuTools::getImagesFolder()
{

	TCHAR name [ UNLEN + 1 ];
	DWORD size = UNLEN + 1;
	GetUserName( (TCHAR*)name, &size );

	std::string suname = name;
	std::string ImagesFolder = "C:\\Users\\" + suname + "\\Pictures\\";

	return ImagesFolder;
}
/*

float* GpGpuTools::divideArray( float* data, uint2 dimImage, float factor )
{

	float* image	= new float[size(dimImage)];

	if (factor != 1.0f)
		for (uint j = 0; j < dimImage.y ; j++)	
			for (uint i = 0; i < dimImage.x ; i++)
			{
				int id = j * dimImage.x + i;
				image[id] = dataImage[id] / factor;

			}

	return image;

}

*/

