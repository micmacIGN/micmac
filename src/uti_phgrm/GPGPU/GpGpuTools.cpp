#include "GpGpu/GpGpuTools.h"

bool DISPLAYOUTPUT;

void GpGpuTools::Memcpy2Dto1D( float** dataImage2D, float* dataImage1D, uint2 dimDest, uint2 dimSource )
{

	for (uint j = 0; j < dimSource.y ; j++)
		memcpy(  dataImage1D + dimDest.x * j , dataImage2D[j],  dimSource.x * sizeof(float));			

}

bool GpGpuTools::Array1DtoImageFile( float* dataImage,const char* fileName, uint2 dimImage )
{
	std::string pathfileImage = std::string(GetImagesFolder()) + std::string(fileName);

	std::cout << pathfileImage << "\n";
	return sdkSavePGM<float>(pathfileImage.c_str(), dataImage, dimImage.x,dimImage.y);
}

bool GpGpuTools::Array1DtoImageFile(float* dataImage,const char* fileName, uint2 dimImage, float factor)
{
	float* image = DivideArray(dataImage, dimImage, factor);
	
	bool r = Array1DtoImageFile( image, fileName, dimImage );

	delete[] image;

	return r;
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

float* GpGpuTools::DivideArray( float* data, uint2 dim, float factor )
{

	int sizeData = size(dim);

	float* image = new float[sizeData];

	for (int i = 0; i < sizeData ; i++)
		image[i] = data[i] / factor;

	return image;

}

void GpGpuTools::OutputArray( float* data, uint2 dim, float offset, float defaut, float sample, float factor )
{
	if (!DISPLAYOUTPUT) return;

	uint2 p;

	for (p.y = 0 ; p.y < dim.y; p.y+= (int)sample)
	{
		for (p.x = 0; p.x < dim.x ; p.x+= (int)sample)
		
			OutputValue(data[to1D(p,dim)],offset,defaut,factor);
		
		std::cout << "\n";	
	}
	std::cout << "------------------------------------------\n";
}	

void GpGpuTools::OutputValue( float value, float offset, float defaut, float factor)
{
	if (!DISPLAYOUTPUT) return;
	
	std::string S2	= "    ";
	std::string ES	= "";
	std::string S1	= " ";

	float outO	= value/factor;
	float out	= floor(outO*offset)/offset ;

	std::string valS;
	stringstream sValS (stringstream::in | stringstream::out);

	sValS << abs(out);
	long sizeV = (long)sValS.str().length();

	if (sizeV == 5) ES = ES + "";
	else if (sizeV == 4) ES = ES + " ";
	else if (sizeV == 3) ES = ES + "  ";
	else if (sizeV == 2) ES = ES + "   ";
	else if (sizeV == 1) ES = ES + "    ";

	if (outO == 0.0f)
		std::cout << S1 << "0" << S2;
	else if (outO == defaut)
		std::cout << S1 << "!" + S2;
	else if (outO == -1000.0f)
		std::cout << S1 << "." << S2;
	else if (outO == 2*defaut)
		std::cout << S1 << "s" << S2;
	else if (outO == 3*defaut)
		std::cout << S1 << "z" << S2;
	else if (outO == 4*defaut)
		std::cout << S1 << "s" << S2;
	else if (outO == 5*defaut)
		std::cout << S1 << "v" << S2;
	else if (outO == 6*defaut)
		std::cout << S1 << "e" << S2;
	else if (outO == 7*defaut)
		std::cout << S1 << "c" << S2;
	else if (outO == 8*defaut)
		std::cout << S1 << "?" << S2;
	else if (outO == 9*defaut)
		std::cout << S1 << "¤" << S2;
	else if ( outO < 0.0f)
		std::cout << out << ES;				
	else 
		std::cout << S1 << out << ES;
	
}

void GpGpuTools::DisplayOutput( bool diplay )
{
	DISPLAYOUTPUT = true;
}

void GpGpuTools::OutputReturn( char * out )
{
	if (!DISPLAYOUTPUT) return;
	
	std::cout << std::string(out) << "\n";

}


