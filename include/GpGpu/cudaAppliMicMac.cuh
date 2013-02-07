#pragma once
#include <cuda_runtime.h>
#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "GpGpu/helper_math_extented.cuh"
#include "GpGpu/GpGpuTools.h"


#ifdef _WIN32
	#include <windows.h>
	#include <Lmcons.h>
#endif

#include <iostream>
#include <string>
using namespace std;


#define INTDEFAULT	-64
#define SAMPLETERR	4
#define INTERZ		4

#ifdef _DEBUG
	#define   BLOCKDIM	16
	#define   SBLOCKDIM 10
#else
	#define   BLOCKDIM	32
	#define   SBLOCKDIM 16
#endif

struct paramGPU
{

	 int2	pUTer0;
	 int2	pUTer1;

	 uint	ZInter;
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
	 int2	ptMask0;	// point debut du masque
	 int2	ptMask1;	// point fin du masque
	 float	badVig;		//
	 float	mAhEpsilon;
};

