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
#define USEATOMICFUNCT	false
#define NSTREAM		1

//#define USEDILATEMASK
#define MAX_THREADS_PER_BLOCK 1024 // A definir en fonction du device!!!

#ifdef _DEBUG
	#define   BLOCKDIM	16
	#define   SBLOCKDIM 10
#else
	#define   BLOCKDIM	32
	#define   SBLOCKDIM 15
#endif

struct pCorGpu
{

	 uint	ZLocInter;
	 
	 uint2	dimTer;			// Dimension du bloque terrain
	 uint2	dimDTer;		// Dimension du bloque terrain + halo
	 uint2	dimSTer;		// Dimension du bloque terrain + halo sous echantilloné
	 uint2	dimImg;			// Dimension de l'image la plus grande
	 uint2	dimCach;		// Dimension cache des calculs intermédiaires
	 uint2	dimVig;			// Dimension de la vignette
	 uint2	rayVig;			// Rayon de la vignette
	 
	 uint	sizeVig;		// Taille de la vignette en pixel
	 uint	sizeDTer;		// Taille du bloque terrain + halo
	 uint	sizeTer;		// taille reel du terrain
	 uint	sizeSTer;		// Taille du bloque terrain + halo sous echantilloné
	 uint	sizeCach;		// Taille du cache
	 uint	sizeCachAll;		// Taille du cache

	 uint	sampProj;		// Pas echantillonage du terrain
	 float	floatDefault;	// UV Terrain incorrect
	 int	IntDefault;		// INT UV Terrain incorrect
	 uint	nbImages;		// Nombre d'images
	 Rect	rDTer;			// Rectangle du terrain dilaté du rayon de la vignette
	 Rect	rTer;			// Rectangle du terrain 
	 float	mAhEpsilon;		// 

	 Rect	RDTer() { return rDTer; }
	 Rect	RTer() { return rTer; }

	 void SetDimension(Rect Ter, uint Zinter)
	 {

		 rTer		= Ter;
		 rDTer		= Rect(Ter.pt0 - rayVig,Ter.pt1 + rayVig);
		 dimTer		= rTer.dimension();
		 dimDTer	= rDTer.dimension();
		 dimSTer	= iDivUp(dimDTer,sampProj)+1;	// Dimension du bloque terrain sous echantilloné
		 dimCach	= dimTer * dimVig;
		 
		 sizeDTer	= size(dimDTer);				// Taille du bloque terrain
		 sizeSTer	= size(dimSTer);			// Taille du bloque terrain sous echantilloné
		 sizeTer	= size(dimTer);
		 sizeCach	= size(dimCach);
		 sizeCachAll	= sizeCach * nbImages;
		 //ZLocInter	= LOCINTERZ;
		 ZLocInter	= Zinter;
	 
	 };

	 void SetParamInva(uint2 dV,uint2 dRV, uint2 dI, float mAhEpsilon, uint samplingZ, int uvINTDef, uint nLayer)
	 {
		 float uvDef;
		 memset(&uvDef,uvINTDef,sizeof(float));

		 nbImages		= nLayer;
		 dimVig			= dV;							// Dimension de la vignette
		 dimImg			= dI;							// Dimension des images
		 rayVig			= dRV;							// Rayon de la vignette
		 sizeVig		= size(dV);						// Taille de la vignette en pixel 
		 sampProj		= samplingZ;					// Pas echantillonage du terrain
		 floatDefault	= uvDef;						// UV Terrain incorrect
		 IntDefault		= uvINTDef;
		 mAhEpsilon		= mAhEpsilon;

	 };

	 bool MaskNoNULL()
	 {
		 return (rTer.pt0.x != -1);
	 }

	 void outConsole()
	 {
		std::cout << "Parametre de calcul GPU pour la correlation symetrique\n";
		std::cout << "\n";
		std::cout << "----------------------------------------------------------\n";
		std::cout << "ZLocInter             : " << ZLocInter << "\n";
		std::cout << "Dim Reel Terrain      : " << GpGpuTools::toStr(dimTer) << "\n";
		std::cout << "Dim calcul Terrain    : " << GpGpuTools::toStr(dimDTer) << "\n";
		std::cout << "Dim calcul Ter Samp   : " << GpGpuTools::toStr(dimSTer) << "\n";
		std::cout << "Dim vignette          : " << GpGpuTools::toStr(dimVig) << "\n";
		std::cout << "Rayon vignette        : " << GpGpuTools::toStr(rayVig) << "\n";
		std::cout << "Dim Image             : " << GpGpuTools::toStr(dimImg) << "\n";
		std::cout << "Dim Cache             : " << GpGpuTools::toStr(dimCach) << "\n";
		std::cout << "Taille vignette       : " << sizeVig << "\n";
		std::cout << "Taille terrain + halo : " << sizeDTer << "\n";
		std::cout << "Taille Reel Terrain   : " << sizeTer << "\n";
		std::cout << "Taille Samp Terrain   : " << sizeSTer << "\n";
		std::cout << "Taille cache          : " << sizeCach << "\n";
		std::cout << "Sample                : " << sampProj << "\n";
		std::cout << "Default Val float     : " << floatDefault << "\n";
		std::cout << "Default Val int       : " << IntDefault << "\n";
		std::cout << "Nombre Images         : " << nbImages << "\n";
		std::cout << "mAhEpsilon            : " << mAhEpsilon << "\n";
		std::cout << "Rectangle terrain     : ";rDTer.out();std::cout << "\n";
		std::cout << "Rectangle masque      : ";rTer.out();std::cout << "\n";
		std::cout << "\n";
		std::cout << "----------------------------------------------------------\n";
	 }
};

