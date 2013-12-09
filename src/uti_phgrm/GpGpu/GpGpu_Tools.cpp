#include "GpGpu/GpGpu_Tools.h"

void GpGpuTools::SetParamterTexture(textureReference &textRef)
{
    textRef.addressMode[0]	= cudaAddressModeBorder;
    textRef.addressMode[1]	= cudaAddressModeBorder;
    textRef.filterMode		= cudaFilterModeLinear; //cudaFilterModePoint cudaFilterModeLinear
    textRef.normalized		= false;
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
#else
	
	std::cout << std::string(out) << "\n";

#endif

}

void GpGpuTools::OutputGpu()
{
#if (ELISE_windows)
	Sleep(10);
	std::cout << "\b\\" << std::flush;
	Sleep(10);
	std::cout << "\b|" << std::flush;
	Sleep(10);
	std::cout << "\b/" << std::flush;
	Sleep(10);
	std::cout << "\b-" << std::flush;
#endif
}
#ifdef NVTOOLS
void GpGpuTools::NvtxR_Push(const char* message, int32_t color)
{

    nvtxEventAttributes_t initAttrib = {0};

    initAttrib.version = NVTX_VERSION;
    initAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    initAttrib.color = color;
    initAttrib.colorType = NVTX_COLOR_ARGB;
    initAttrib.message.ascii = message;
    initAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;

    nvtxRangePushEx(&initAttrib);

}
#endif
float GpGpuTools::fValue( float value,float factor )
{
	return value * factor;
}

float GpGpuTools::fValue( float2 value,float factor )
{
	return (float)value.x * factor;
}

std::string GpGpuTools::toStr( uint2 tt )
{
	stringstream sValS (stringstream::in | stringstream::out);

	sValS << "(" << tt.x << "," << tt.y << ")";

    return sValS.str();
}

const char *GpGpuTools::conca(const char *texte, int t)
{
    stringstream sValS (stringstream::in | stringstream::out);

    sValS << texte << t;

    return sValS.str().c_str();
}

void GpGpuTools::OutputInfoGpuMemory()
{
	size_t free;  
	size_t total;  
    checkCudaErrors( cudaMemGetInfo(&free, &total));
    cout << "Memoire video       : " << (float)free / pow(2.0f,20) << " / " << (float)total / pow(2.0f,20) << "Mo" << endl;
}
