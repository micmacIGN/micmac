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
void GpGpuTools::Nvtx_RangePop()
{
#ifdef NVTOOLS
	nvtxRangePop();
#endif
}

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

//void GpGpuTools::OutputInfoGpuMemory()
//{
//	size_t free;
//	size_t total;
//    checkCudaErrors( cudaMemGetInfo(&free, &total));
//    cout << "Memoire video       : " << (float)free / pow(2.0f,20) << " / " << (float)total / pow(2.0f,20) << "Mo" << endl;
//}
/*
void GpGpuTools::check_Cuda()
{
    cout << "CUDA build enabled\n";

//    int apiVersion = 0;

//    cudaRuntimeGetVersion(&apiVersion);

            //DUMP_INT(apiVersion)

//	switch (__CUDA_API_VERSION)
//	{
//	case 0x3000:
//		cout << "3.0";
//		break;
//	case 0x3020:
//		cout << "3.2";
//		break;
//	case 0x4000:
//		cout << "4.0";
//		break;
//	case 0x5000:
//		cout << "5.0";
//		break;
//	case 0x5050:
//		cout << "5.5";
//		break;
//	case 0x6000:
//		cout << "6.0";
//		break;
//	}
//	cout << endl;

	int device_count = 0;
	 
	checkCudaErrors(cudaGetDeviceCount(&device_count));

	if(device_count == 0)
        printf("NO NVIDIA GRAPHIC CARD FOR USE CUDA");
	else
	{

		// Creation du contexte GPGPU
		cudaDeviceProp deviceProp;
		// Obtention de l'identifiant de la carte la plus puissante
		int devID = gpuGetMaxGflopsDeviceId();

		// Initialisation du contexte
		checkCudaErrors(cudaSetDevice(devID));
		// Obtention des proprietes de la carte
		checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
		// Affichage des proprietes de la carte
		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

	}
}*/
