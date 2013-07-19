#pragma once

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
#define   BLOCKDIM	16
//#define   BLOCKDIM	32 moins rapide !!!!
#define   SBLOCKDIM 15
#endif


/// \struct pCorGpu
/// \param  La structure contenant tous les parametres necessaires a la correlation
struct pCorGpu
{

    /// \brief  Le nombre de Z calculer en parrallele
    uint        ZLocInter;

    /// \brief  Dimension du bloque terrain
    uint2       dimTer;
    /// \brief  Dimension du bloque terrain + halo
    uint2       dimDTer;
    /// \brief  Dimension du bloque terrain + halo sous echantilloné
    uint2       dimSTer;
    /// \brief  Dimension de l'image la plus grande
    uint2       dimImg;
    /// \brief  Dimension cache des calculs intermédiaires
    uint2       dimCach;
    /// \brief  Dimension de la vignette
    uint2       dimVig;
    /// \brief  Rayon de la vignette
    uint2       rayVig;
    /// \brief  Taille de la vignette en pixel
    uint        sizeVig;
    /// \brief  Taille du bloque terrain + halo
    uint        sizeDTer;
    /// \brief  taille reel du terrain
    uint        sizeTer;
    /// \brief  Taille du bloque terrain + halo sous echantilloné
    uint        sizeSTer;
    /// \brief  Taille du cache
    uint        sizeCach;
    /// \brief  Taille du cache des tous les Z
    uint        sizeCachAll;
    /// \brief  Pas echantillonage du terrain
    uint        sampProj;
    /// \brief  Valeur incorrect
    float       floatDefault;
    /// \brief  Valeur entiere incorrect
    int         IntDefault;
    /// \brief  Nombre d'images
    uint        nbImages;
    /// \brief  Rectangle du terrain dilaté du rayon de la vignette
    Rect        rDTer;
    /// \brief  Rectangle du terrain
    Rect        rTer;
    /// \brief  Epsilon
    float       mAhEpsilon;

    /// \brief  Renvoie le rectangle du terrain dilaté du rayon de la vignette
    Rect        RDTer() { return rDTer; }
    /// \brief  Renvoie le rectangle du terrain
    Rect        RTer() { return rTer; }

    /// \brief  Initialise le rectangle du terrain et le nombre de Z a calculer
    void        SetDimension(Rect Ter, uint Zinter = INTERZ)
    {

        rTer		= Ter;
        rDTer		= Rect(Ter.pt0 - rayVig,Ter.pt1 + rayVig);
        dimTer		= rTer.dimension();
        dimDTer     = rDTer.dimension();
        dimSTer     = iDivUp(dimDTer,sampProj)+1;	// Dimension du bloque terrain sous echantilloné
        dimCach 	= dimTer * dimVig;

        sizeDTer	= size(dimDTer);            // Taille du bloque terrain
        sizeSTer	= size(dimSTer);			// Taille du bloque terrain sous echantilloné
        sizeTer     = size(dimTer);
        sizeCach	= size(dimCach);
        sizeCachAll	= sizeCach * nbImages;
        //ZLocInter	= LOCINTERZ;
        ZLocInter	= Zinter;
    }

    void SetZInter(uint Zinter = INTERZ)
    {
        ZLocInter = Zinter;
    }

    /// \brief  Initialise les param?tres invariants pendant le calcul
    void SetParamInva(uint2 dV,uint2 dRV, uint2 dI, float tmAhEpsilon, uint samplingZ, int uvINTDef, uint nLayer)
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
        mAhEpsilon		= tmAhEpsilon;

    }

    /// \brief Renvoie vraie si le masque existe
    bool MaskNoNULL()
    {
        return (rTer.pt0.x != -1);
    }

    /// \brief Affiche tous les parametres dans la console
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


