/*Header-MicMac-eLiSe-25/06/2007
 
 MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
 eLiSe  : ELements of an Image Software Environnement
 
 www.micmac.ign.fr
 
 
 Copyright : Institut Geographique National
 Author : Marc Pierrot Deseilligny
 Contributors : Gregoire Maillet, Didier Boldo.
 
 [1] M. Pierrot-Deseilligny, N. Paparoditis.
 "A multiresolution and optimization-based image matching approach:
 An application to surface reconstruction from SPOT5-HRS stereo imagery."
 In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
 (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.
 
 [2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
 d'images, adapte au contexte geograhique" to appears in 
 Bulletin d'information de l'Institut Geographique National, 2007.
 
 Francais :
 
 MicMac est un logiciel de mise en correspondance d'image adapte 
 au contexte de recherche en information geographique. Il s'appuie sur
 la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
 licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.
 
 
 English :
 
 MicMac is an open source software specialized in image matching
 for research in geographic information. MicMac is built on the
 eLiSe image library. MicMac is governed by the  "Cecill-B licence".
 See below and http://www.cecill.info.
 
 Header-MicMac-eLiSe-25/06/2007*/
#ifndef __CCAMERAMODULEORIENATION_H__
#define __CCAMERAMODULEORIENATION_H__

#include "cModuleOrientation.h"
#include "StdAfx.h"
#include "general/ptxd.h"


class cCameraModuleOrientation : public ElCamera
{
private:
    ModuleOrientation * mOri;
    // La notion d'origine n'a pas reellement de sens pour un projection ortho (au mieux elle
    // situee n'importe ou sur le rayon partant du centre de l'image), pourtant il en faut bien une
    // meme completement arbitraire  pour  des fonctions telle que image et profondeur 2 Terrains
    // quand on correle en faisceau
    Pt3dr mCentre;
    
    ElAffin2D mOrIntImaM2C;
    
public:
    
    cCameraModuleOrientation(ModuleOrientation * aOri,const Pt2di & aSz, ElAffin2D const &aOrIntImaM2C);
    double ResolutionSol()const;
    double ResolutionSol(const Pt3dr &) const;
    // R3 : terrain
    // L3 : local
    Pt3dr R3toL3(Pt3dr aP) const;
    Pt3dr L3toR3(Pt3dr aP) const;
    ElProj32   &  Proj();
    const ElProj32   &  Proj() const;
    ElDistortion22_Gen & Dist();
    const ElDistortion22_Gen & Dist() const;
    void InstanceModifParam(cCalibrationInternConique & aParam)  const;
    Pt3dr ImEtProf2Terrain(const Pt2dr & aP,double aZ) const;
    Pt3dr NoDistImEtProf2Terrain(const Pt2dr & aP,double aZ) const;
    Pt3dr  OrigineProf() const;
    bool  HasOrigineProf() const;
    double SzDiffFinie() const;
};



#endif


/*Footer-MicMac-eLiSe-25/06/2007
 
 Ce logiciel est un programme informatique servant à la mise en
 correspondances d'images pour la reconstruction du relief.
 
 Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
 respectant les principes de diffusion des logiciels libres. Vous pouvez
 utiliser, modifier et/ou redistribuer ce programme sous les conditions
 de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
 sur le site "http://www.cecill.info".
 
 En contrepartie de l'accessibilité au code source et des droits de copie,
 de modification et de redistribution accordés par cette licence, il n'est
 offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
 seule une responsabilité restreinte pèse sur l'auteur du programme,  le
 titulaire des droits patrimoniaux et les concédants successifs.
 
 A cet égard  l'attention de l'utilisateur est attirée sur les risques
 associés au chargement,  à l'utilisation,  à la modification et/ou au
 développement et à la reproduction du logiciel par l'utilisateur étant 
 donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
 manipuler et qui le réserve donc à des développeurs et des professionnels
 avertis possédant  des  connaissances  informatiques approfondies.  Les
 utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
 logiciel à leurs besoins dans des conditions permettant d'assurer la
 sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
 à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 
 
 Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
 pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
 termes.
 Footer-MicMac-eLiSe-25/06/2007*/
