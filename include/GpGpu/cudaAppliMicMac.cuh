#pragma once
#include <cuda_runtime.h>
#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "GpGpuTools.h"

#define INTDEFAULT -64
#define SAMPLETERR 1

struct paramGPU
{

	 int2	pUTer0;
	 int2	pUTer1;

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
	 float	UVDefValue;	// UV Terrain incorrect
	 int	UVIntDef;	// INT UV Terrain incorrect
	 uint2	dimCach;	// Dimension cache
	 uint	sizeCach;	// Taille du cache
	 uint	nLayer;		// Nombre d'images
	 int2	ptMask0;	// point debut du masque
	 int2	ptMask1;	// point fin du masque
	 float	badVig;		//
	 float	mAhEpsilon;
};

static __constant__ paramGPU cH;