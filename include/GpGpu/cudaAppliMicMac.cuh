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
#define INTERPOLA	LINEARINTER
#define FLOATMATH
//#define USEDILATEMASK
#define MAX_THREADS_PER_BLOCK 1024 // A definir en fonction du device!!!

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
		 dimSTer	= iDivUp(dimTer,sampTer)+1;	// Dimension du bloque terrain sous echantilloné
		 sizeTer	= size(dimTer);				// Taille du bloque terrain
		 sizeSTer	= size(dimSTer);			// Taille du bloque terrain sous echantilloné
		 rSiTer		= size(rDiTer);
		 dimCach	= rDiTer * dimVig;
		 sizeCach	= size(dimCach);
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
		 mAhEpsilon	= mAhEpsilon;

	 };

	 bool MaskNoNULL()
	 {
		 return (GetRMask().pt0.x != -1);
	 }

	 void outConsole()
	 {
		std::cout << "Parametre de calcul GPU pour la correlation symetrique\n";
		std::cout << "\n";
		std::cout << "----------------------------------------------------------\n";
		std::cout << "ZInter                : " << ZInter << "\n";
		std::cout << "ZLocInter             : " << ZLocInter << "\n";
		std::cout << "Dim Reel Terrain      : " << GpGpuTools::toStr(rDiTer) << "\n";
		std::cout << "Dim calcul Terrain    : " << GpGpuTools::toStr(dimTer) << "\n";
		std::cout << "Dim calcul Ter Samp   : " << GpGpuTools::toStr(dimSTer) << "\n";
		std::cout << "Dim vignette          : " << GpGpuTools::toStr(dimVig) << "\n";
		std::cout << "Rayon vignette        : " << GpGpuTools::toStr(rVig) << "\n";
		std::cout << "Dim Image             : " << GpGpuTools::toStr(dimImg) << "\n";
		std::cout << "Dim Cache             : " << GpGpuTools::toStr(dimCach) << "\n";
		std::cout << "Taille vignette       : " << sizeVig << "\n";
		std::cout << "Taille terrain + halo : " << sizeTer << "\n";
		std::cout << "Taille Reel Terrain   : " << rSiTer << "\n";
		std::cout << "Taille Samp Terrain   : " << sizeSTer << "\n";
		std::cout << "Taille cache          : " << sizeCach << "\n";
		std::cout << "Sample                : " << sampTer << "\n";
		std::cout << "Default Val float     : " << DefaultVal << "\n";
		std::cout << "Default Val int       : " << IntDefault << "\n";
		std::cout << "Nombre Images         : " << nbImages << "\n";
		std::cout << "mAhEpsilon            : " << mAhEpsilon << "\n";
		std::cout << "Rectangle terrain     : ";rUTer.out();std::cout << "\n";
		std::cout << "Rectangle masque      : ";rMask.out();std::cout << "\n";
		std::cout << "\n";
		std::cout << "----------------------------------------------------------\n";
	 }
};

