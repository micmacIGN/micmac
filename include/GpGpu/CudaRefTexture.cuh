#ifndef CUDAREFTEXTURE
#define CUDAREFTEXTURE

// ATTENTION : erreur de compilation avec l'option cudaReadModeNormalizedFloat et l'utilisation de la fonction tex2DLayered
texture< pixel,	cudaTextureType2D >			TexS_MaskTer;
texture< float,	cudaTextureType2DLayered >	TexL_Images;
TexFloat2Layered							TexL_Proj_00;
TexFloat2Layered							TexL_Proj_01;
TexFloat2Layered							TexL_Proj_02;
TexFloat2Layered							TexL_Proj_03;

template<int TexSel> __device__ __host__ TexFloat2Layered TexFloat2L();

template<> __device__ __host__ TexFloat2Layered TexFloat2L<0>() { return TexL_Proj_00; };
template<> __device__ __host__ TexFloat2Layered TexFloat2L<1>() { return TexL_Proj_01; };
template<> __device__ __host__ TexFloat2Layered TexFloat2L<2>() { return TexL_Proj_02; };
template<> __device__ __host__ TexFloat2Layered TexFloat2L<3>() { return TexL_Proj_03; };


//------------------------------------------------------------------------------------------

extern "C" textureReference& getMask(){	return TexS_MaskTer;}
extern "C" textureReference& getImage(){ return TexL_Images;}
extern "C" textureReference& getProjection(int TexSel)
{
	switch (TexSel)
	{
	case 0:
		return TexL_Proj_00;
	case 1:
		return TexL_Proj_01;
	case 2:
		return TexL_Proj_02;
	case 3:
		return TexL_Proj_03;
	default:
		return TexL_Proj_00;
	}								
}

#endif /*CUDAREFTEXTURE*/ 