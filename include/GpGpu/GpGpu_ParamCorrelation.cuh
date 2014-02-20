#pragma once

#include "GpGpu/GpGpu_Data.h"

struct invParamCorrel
{
    /// \brief  Nombre d'images
    uint        nbImages;

    /// \brief  Valeur entiere incorrect
    int         IntDefault;

    /// \brief  Valeur incorrect
    float       floatDefault;

    /// \brief  Pas echantillonage du terrain
    uint        sampProj;

    /// \brief  Taille de la vignette en pixel
    ushort      sizeVig;

    /// \brief  Rayon de la vignette
    ushort2     rayVig;

    /// \brief  Dimension de la vignette
    ushort2     dimVig;

    /// \brief  Epsilon
    float       mAhEpsilon;

    /// \brief  Initialise les param?tres invariants pendant le calcul
    void SetParamInva(ushort2 dV,ushort2 dRV, uint2 dI, float tmAhEpsilon, uint samplingZ, int uvINTDef, uint nLayer)
    {
        float uvDef;
        memset(&uvDef,uvINTDef,sizeof(float));

        nbImages		= nLayer;

        dimVig			= dV;							// Dimension de la vignette

        rayVig			= dRV;							// Rayon de la vignette

        sizeVig         = size(dV);						// Taille de la vignette en pixel

        sampProj		= samplingZ;					// Pas echantillonage du terrain

        floatDefault	= uvDef;						// UV Terrain incorrect

        IntDefault		= uvINTDef;

        mAhEpsilon		= tmAhEpsilon;

    }
};

struct HDParamCorrel
{
    /// \brief  Dimension du bloque terrain
    uint2       dimTer;

    /// \brief  Dimension du bloque terrain + halo
    uint2       dimDTer;

    /// \brief  Dimension cache des calculs intermédiaires
    uint2       dimCach;

    /// \brief  taille reel du terrain
    uint        sizeTer;

    /// \brief  Taille du cache
    uint        sizeCach;

    /// \brief  Taille du cache des tous les Z
    uint        sizeCachAll;

    /// \brief  Rectangle du terrain
    Rect        rTer;
};

/// \struct pCorGpu
/// \param  La structure contenant tous les parametres necessaires a la correlation
struct pCorGpu
{    

    invParamCorrel invPC;

    HDParamCorrel  HdPc;

    /// \brief  Le nombre de Z calculer en parallele
    uint        ZCInter;

    /// \brief  Dimension du bloque terrain + halo sous echantilloné
    uint2       dimSTer;

    /// \brief  Rectangle du terrain dilaté du rayon de la vignette
    Rect        rDTer;

    /// \brief  Renvoie le rectangle du terrain dilaté du rayon de la vignette
    Rect        RDTer() { return rDTer; }

    /// \brief  Renvoie le rectangle du terrain
    Rect        RTer() { return HdPc.rTer; }

    /// \brief  Initialise le rectangle du terrain et le nombre de Z a calculer
    void        SetDimension(Rect Ter, uint Zinter = INTERZ)
    {

        HdPc.rTer		= Ter;

        rDTer           = Rect(Ter.pt0 - make_uint2(invPC.rayVig),Ter.pt1 + make_uint2(invPC.rayVig));

        HdPc.dimTer		= HdPc.rTer.dimension();

        HdPc.dimDTer    = rDTer.dimension();

        dimSTer         = iDivUp(HdPc.dimDTer,invPC.sampProj)+1;	// Dimension du bloque terrain sous echantilloné

        HdPc.dimCach    = HdPc.dimTer * make_uint2(invPC.dimVig);

        HdPc.sizeTer    = size(HdPc.dimTer);

        HdPc.sizeCach	= size(HdPc.dimCach);

        HdPc.sizeCachAll	= HdPc.sizeCach * invPC.nbImages;

        ZCInter     = Zinter;

    }

    /// \brief Affiche tous les parametres dans la console
    void outConsole()
    {
        std::cout << "Parametre de calcul GPU pour la correlation symetrique\n";
        std::cout << "\n";
        std::cout << "----------------------------------------------------------\n";
        std::cout << "ZLocInter             : " << ZCInter << "\n";
        std::cout << "Dim Reel Terrain      : " << GpGpuTools::toStr(HdPc.dimTer) << "\n";
        std::cout << "Dim calcul Terrain    : " << GpGpuTools::toStr(HdPc.dimDTer) << "\n";
        std::cout << "Dim calcul Ter Samp   : " << GpGpuTools::toStr(dimSTer) << "\n";
        std::cout << "Dim vignette          : " << GpGpuTools::toStr(make_uint2(invPC.dimVig)) << "\n";
        std::cout << "Rayon vignette        : " << GpGpuTools::toStr(make_uint2(invPC.rayVig)) << "\n";
        std::cout << "Dim Cache             : " << GpGpuTools::toStr(HdPc.dimCach) << "\n";
        std::cout << "Taille vignette       : " << invPC.sizeVig << "\n";
        std::cout << "Taille Reel Terrain   : " << HdPc.sizeTer << "\n";
        std::cout << "Taille cache          : " << HdPc.sizeCach << "\n";
        std::cout << "Sample                : " << invPC.sampProj << "\n";
        std::cout << "Default Val float     : " << invPC.floatDefault << "\n";
        std::cout << "Default Val int       : " << invPC.IntDefault << "\n";
        std::cout << "Nombre Images         : " << invPC.nbImages << "\n";
        std::cout << "mAhEpsilon            : " << invPC.mAhEpsilon << "\n";
        std::cout << "Rectangle terrain     : ";rDTer.out();std::cout << "\n";
        std::cout << "Rectangle masque      : ";HdPc.rTer.out();std::cout << "\n";
        std::cout << "\n";
        std::cout << "----------------------------------------------------------\n";
    }
};
