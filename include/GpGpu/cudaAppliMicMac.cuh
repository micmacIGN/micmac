#pragma once

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_math.h>

#include "GpGpu/helper_math_extented.cuh"
#include "GpGpu/GpGpuTools.h"

#define INTDEFAULT	-64
#define SAMPLETERR	4
#define INTERZ		8
#define LOCINTERZ	1
#define NEAREST		0
#define LINEARINTER	1
#define BICUBIC		2
#define INTERPOLA	NEAREST
#define FLOATMATH


#ifdef _DEBUG
	#define   BLOCKDIM	16
	#define   SBLOCKDIM 10
#else
	#define   BLOCKDIM	32
	#define   SBLOCKDIM 15
#endif

struct paramMicMacGpGpu
{

	 uint	ZInter;
	 uint	ZLocInter;
	 uint2	rDiTer;		// Dimension du bloque terrain
	 uint2	dimTer;		// Dimension du bloque terrain + halo
	 uint2	dimSTer;	// Dimension du bloque terrain + halo sous echantilloné
	 uint2	restTer;
	 uint2	dimVig;		// Dimension de la vignette
	 uint2	dimImg;		// Dimension des images
	 uint2	rVig;		// Rayon de la vignette
	 uint	sizeVig;	// Taille de la vignette en pixel
	 uint	sizeTer;	// Taille du bloque terrain + halo
	 uint	rSiTer;		// taille reel du terrain
	 uint	sizeSTer;	// Taille du bloque terrain + halo sous echantilloné
	 uint	sampTer;	// Pas echantillonage du terrain
	 float	DefaultVal;	// UV Terrain incorrect
	 int	IntDefault;	// INT UV Terrain incorrect
	 uint2	dimCach;	// Dimension cache
	 uint	sizeCach;	// Taille du cache
	 uint	nbImages;		// Nombre d'images
	 Rect	rUTer;
	 Rect	rMask;
	 float	badVig;		//
	 float	mAhEpsilon;

	 Rect GetRUTer()
	 {
		 return rUTer;
	 };

	 Rect GetRMask()
	 {
		 return rMask;
	 };

	 void SetDimension(Rect Ter, uint Zinter)
	 {

		 ZInter		= Zinter;
		 rMask		= Ter;
		 rUTer		= Rect(Ter.pt0 - rVig,Ter.pt1 + rVig);
		 rDiTer		= rMask.dimension();
		 dimTer		= rUTer.dimension();
		 dimSTer	= iDivUp(dimTer,sampTer);	// Dimension du bloque terrain sous echantilloné
		 sizeTer	= size(dimTer);				// Taille du bloque terrain
		 sizeSTer	= size(dimSTer);				// Taille du bloque terrain sous echantilloné
		 rSiTer		= size(rDiTer);
		 dimCach	= rDiTer * dimVig;
		 sizeCach	= size(dimCach);
		 restTer	= dimSTer * sampTer - dimTer;
		 //ZLocInter	= LOCINTERZ;
		 ZLocInter	= ZInter;
	 
	 };

	 void SetParamInva(uint2 dV,uint2 dRV, uint2 dI, float mAhEpsilon, uint samplingZ, int uvINTDef, uint nLayer)
	 {
		 float uvDef;
		 memset(&uvDef,uvINTDef,sizeof(float));

		 nbImages	= nLayer;
		 dimVig		= dV;							// Dimension de la vignette
		 dimImg		= dI;							// Dimension des images
		 rVig		= dRV;							// Rayon de la vignette
		 sizeVig	= size(dV);						// Taille de la vignette en pixel 
		 sampTer	= samplingZ;					// Pas echantillonage du terrain
		 DefaultVal	= uvDef;						// UV Terrain incorrect
		 IntDefault	= uvINTDef;
		 badVig		= -4.0f;
		 mAhEpsilon	= mAhEpsilon;

	 };

	 bool MaskNoNULL()
	 {
		 return (GetRMask().pt0.x != -1);
	 }

};

