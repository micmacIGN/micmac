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


#include "SatPhysMod.h"




/******************************************************/
/*                                                    */
/*           cRPC_PushB_PhysMod                       */
/*                                                    */
/******************************************************/

const double cRPC_PushB_PhysMod::ThePdsRay = 0.1;
const double cRPC_PushB_PhysMod::TheMinDeltaZ = 50;

cRPC_PushB_PhysMod::cRPC_PushB_PhysMod(const cRPC & aRPC,eModeRefinePB aModeRefine,const Pt2di & aSzGeoL) :
   cPushB_PhysMod  (Pt2di(aRPC.GetImCol2(),aRPC.GetImRow2()),aModeRefine,aSzGeoL),
   mRPC            (aRPC),
   mWGS84Degr      (cSysCoord::WGS84Degre()),
   mZ0Ray          (barry(0.5+ThePdsRay,mRPC.GetGrC31(),mRPC.GetGrC32())),
   mZ1Ray          (barry(0.5-ThePdsRay,mRPC.GetGrC31(),mRPC.GetGrC32()))
   //mZ0Ray          (barry(0.5+ThePdsRay,mRPC.height_scale,mRPC.height_off)),
   //mZ1Ray          (barry(0.5-ThePdsRay,mRPC.height_scale,mRPC.height_off))
{

   // MPD 
   // Probably something go wrong in altitude interval
   // ERupnik changed; see initialization
   if (ElAbs(mZ0Ray-mZ1Ray) < TheMinDeltaZ)
   {
        double aZMean = (mZ0Ray+mZ1Ray) /2.0;
        mZ0Ray = aZMean  - TheMinDeltaZ/2.0;
        mZ1Ray = aZMean  + TheMinDeltaZ/2.0;
   }
}

cRPC_PushB_PhysMod * cRPC_PushB_PhysMod::NewRPC_PBP(const cRPC & aRPC,eModeRefinePB aModeRefine,const Pt2di &aSzGeoL)
{
   cRPC_PushB_PhysMod * aRes = new cRPC_PushB_PhysMod(aRPC,aModeRefine,aSzGeoL);
   aRes->PostInit();
   return aRes;
}


Pt2dr  cRPC_PushB_PhysMod::RPC_LlZ2Im(const Pt3dr & aLlZ) const
{
    Pt2dr aPInv = mRPC.InverseRPC(aLlZ);
    return aPInv;
}

Pt3dr cRPC_PushB_PhysMod::RPC_ImAndZ2LlZ(const Pt2dr & aPIm,const double & aZ) const
{
    // std::cout << aPIm << " " << aZ << "\n";
    Pt3dr aRes =  mRPC.DirectRPC(aPIm, aZ);
    // std::cout << "GGGGGGgggg \n";

    return aRes;
}

ElSeg3D cRPC_PushB_PhysMod::Im2GeoC_Init(const Pt2dr & aPIm) const
{


   Pt3dr aPT0 = RPC_ImAndZ2LlZ(aPIm,mZ0Ray);
   Pt3dr aPT1 = RPC_ImAndZ2LlZ(aPIm,mZ1Ray);
   //std::cout << " cRPC_PushB_PhysMod::Im2GeoC_Init " << aPT0 << " " << aPT1 << "\n";
   return ElSeg3D(mWGS84Degr->ToGeoC(aPT0),mWGS84Degr->ToGeoC(aPT1));
}


Pt2dr   cRPC_PushB_PhysMod::GeoC2Im_Init(const Pt3dr & aPTer)  const
{
     Pt3dr aLlZ = mWGS84Degr->FromGeoC(aPTer);
     return RPC_LlZ2Im(aLlZ);
}




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
