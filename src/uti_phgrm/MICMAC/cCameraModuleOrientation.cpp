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

#include "cCameraModuleOrientation.h"
#include "general/ptxd.h"



cCameraModuleOrientation::cCameraModuleOrientation(ModuleOrientation * aOri,const Pt2di & aSz,ElAffin2D const &aOrIntImaM2C):
    ElCamera(false,eProjectionStenope)
{
    mOri = aOri;
    ElCamera::SetSz(aSz);
    SetIntrImaM2C(aOrIntImaM2C);
    mCentre = ImEtProf2Terrain(Pt2dr(aSz.x/2,aSz.y/2),0.);
}

double cCameraModuleOrientation::ResolutionSol()const
{
    return mOri->GetResolMoyenne();
}
double cCameraModuleOrientation::ResolutionSol(const Pt3dr &) const 
{
    return  ResolutionSol();
}
Pt3dr cCameraModuleOrientation::L3toR3(Pt3dr aP) const
{
    //std::cout << "L3toR3"<<std::endl;
    // TODO verifier le sens et verifier s'il faut iterer?
    Pt3dr ptOut;
    // Prise en compte de l'affinite
    Pt2dr ptIm(aP.x,aP.y);
    Pt2dr ptIm2 = ptIm;
    //mOri->ImageAndPx2Obj(aP.x,aP.y,&aP.z,ptOut.x,ptOut.y);
    mOri->ImageAndPx2Obj(ptIm2.x ,ptIm2.y ,&aP.z,ptOut.x,ptOut.y);
    ptOut.z = aP.z;
    //std::cout << "L3toR3 : "<<aP.x<<" "<<aP.y<<" "<<aP.z<<" -> "<<ptOut.x<<" "<<ptOut.y<<" "<<ptOut.z<<std::endl;
    // Verification
    //Pt3dr verif=R3toL3(ptOut);
    //std::cout << "verif : "<<verif.x<<" "<<verif.y<<" "<<verif.z<<std::endl;
    //std::cout << "diff : "<<verif.x-aP.x<<" "<<verif.y-aP.y<<" "<<verif.z-aP.z<<std::endl;
    return ptOut;
}
Pt3dr cCameraModuleOrientation::R3toL3(Pt3dr aP) const
{
    //std::cout << "R3toL3"<<std::endl;
    // TODO verifier le sens et verifier s'il faut iterer?
    Pt3dr ptOut;
    Pt2dr ptIm;
    //mOri->Objet2ImageInit(aP.x,aP.y,&aP.z,ptOut.x,ptOut.y);
    mOri->Objet2ImageInit(aP.x,aP.y,&aP.z,ptIm.x ,ptIm.y);
    Pt2dr ptIm2 =ptIm;
    ptOut.x = ptIm2.x   ;
    ptOut.y = ptIm2.y  ;
    ptOut.z = aP.z;
    //std::cout << "R3toL3 : "<<aP.x<<" "<<aP.y<<" "<<aP.z<<" -> "<<ptOut.x<<" "<<ptOut.y<<" "<<ptOut.z<<std::endl;
    // verification
    //Pt3dr verif=L3toR3(ptOut);
    //std::cout << "verif : "<<verif.x<<" "<<verif.y<<" "<<verif.z<<std::endl;
    //std::cout << "diff : "<<verif.x-aP.x<<" "<<verif.y-aP.y<<" "<<verif.z-aP.z<<std::endl;
    return ptOut;
}
ElProj32   &  cCameraModuleOrientation::Proj()
{
    return ElProjIdentite::TheOne;
}
const ElProj32   &  cCameraModuleOrientation::Proj() const
{
    return ElProjIdentite::TheOne;
}
ElDistortion22_Gen & cCameraModuleOrientation::Dist()
{
    return ElDistortion22_Triviale::TheOne;
}
const ElDistortion22_Gen & cCameraModuleOrientation::Dist() const
{
    return ElDistortion22_Triviale::TheOne;
}
void cCameraModuleOrientation::InstanceModifParam(cCalibrationInternConique & aParam)  const
{
    aParam.PP() = Pt2dr(12345678,87654321);
    aParam.F()  = 0;
}    

Pt3dr cCameraModuleOrientation::NoDistImEtProf2Terrain(const Pt2dr & aP,double aZ) const
{
   return ImEtProf2Terrain(aP,aZ);
}

Pt3dr cCameraModuleOrientation::ImEtProf2Terrain(const Pt2dr & aP,double aZ) const
{
    // TODO verifier le sens et verifier s'il faut iterer?
    Pt3dr ptOut;
    // Prise en compte de l'affinite
    Pt2dr ptIm(aP.x,aP.y);
    Pt2dr ptIm2 = ptIm;
    //mOri->ImageAndPx2Obj(aP.x,aP.y,&aP.z,ptOut.x,ptOut.y);
    mOri->ImageAndPx2Obj(ptIm2.x ,ptIm2.y ,&aZ,ptOut.x,ptOut.y);
    ptOut.z = aZ;
    //std::cout << "ImEtProf2Terrain : "<<aP.x<<" "<<aP.y<<" "<<aZ<<" -> "<<ptOut.x<<" "<<ptOut.y<<" "<<ptOut.z<<std::endl;
    return ptOut;
}
Pt3dr  cCameraModuleOrientation::OrigineProf() const
{
    return mCentre;
}
bool  cCameraModuleOrientation::HasOrigineProf() const
{
    return true;
}
double cCameraModuleOrientation::SzDiffFinie() const
{
    return 1.0;
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
