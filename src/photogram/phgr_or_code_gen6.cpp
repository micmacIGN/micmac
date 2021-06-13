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


#include "StdAfx.h"

#if  (ELISE_INSERT_CODE_GEN)

#include "../../CodeGenere/photogram/cEqCalibCroisee_NoDist_CodGenC2M.cpp"
#include "../../CodeGenere/photogram/cEqCalibCroisee_DRad5_CodGenC2M.cpp"
#include "../../CodeGenere/photogram/cEqCalibCroisee_DRad5APFraser_CodGenC2M.cpp"

#include "../../CodeGenere/photogram/cEqCalibCroisee_NoDist_CodGenM2C.cpp"
#include "../../CodeGenere/photogram/cEqCalibCroisee_DRad5_CodGenM2C.cpp"
#include "../../CodeGenere/photogram/cEqCalibCroisee_DRad5APFraser_CodGenM2C.cpp"



#include "../../CodeGenere/photogram/cEqCorLI_Multi_1_DRPts0_NonNorm.cpp" //
// #include "../../CodeGenere/photogram/cEqCorLI_Multi_1_DRPts1_NonNorm.cpp" //
// #include "../../CodeGenere/photogram/cEqCorLI_Multi_1_DRPts2_NonNorm.cpp" //
// #include "../../CodeGenere/photogram/cEqCorLI_Multi_1_DRPts3_NonNorm.cpp" //
#include "../../CodeGenere/photogram/cEqObsRotVect_CodGen.cpp"
#include "../../CodeGenere/photogram/EqObsBaclt.cpp"



#include "../../CodeGenere/photogram/cEqDirectDistDRad5Reformat_CodGen.cpp"
#include "../../CodeGenere/photogram/cEqDirectDistDRad5Interp_CodGen.cpp"
#include "../../CodeGenere/photogram/cEqDirectDistDRad5Bayer_CodGen.cpp"


#include "../../CodeGenere/photogram/cEqAppui_AFocal_TerFix_M2CDRad5.cpp"
#include "../../CodeGenere/photogram/cEqAppui_AFocal_GL__TerFix_M2CDRad5.cpp"
#include "../../CodeGenere/photogram/cEqAppui_AFocal_GL__PProjInc_M2CDRad5.cpp"
#include "../../CodeGenere/photogram/cEqAppui_AFocal_PTInc_M2CDRad5.cpp"
#include "../../CodeGenere/photogram/cEqAppui_AFocal_GL__PTInc_M2CDRad5.cpp"
#include "../../CodeGenere/photogram/cEqAppui_AFocal_PProjInc_M2CDRad5.cpp"


//=======================  cEqObsBaseGPS ========================

#include "../../CodeGenere/photogram/cEqObsBaseGPS.cpp"
#include "../../CodeGenere/photogram/cEqObsBaseGPS_GL.cpp"
#include "../../CodeGenere/photogram/cImplEqRelativeGPS.cpp"
#include "../../CodeGenere/photogram/cCodeBlockCam.cpp"
#include "../../CodeGenere/photogram/cCodeDistBlockCam.cpp"
#include "../../CodeGenere/photogram/cEqLinariseAngle.cpp"
#include "../../CodeGenere/photogram/cEqLinariseAngle_AccelCsteCoord.cpp"
#include "../../CodeGenere/photogram/cEqBBCamFirst.cpp"
#include "../../CodeGenere/photogram/cEqBBCamSecond.cpp"
#include "../../CodeGenere/photogram/cEqBBCamThird.cpp"
#include "../../CodeGenere/photogram/cEqBBCamFirst_AccelCsteCoord.cpp"
#include "../../CodeGenere/photogram/cEqBBCamSecond_AccelCsteCoord.cpp"
#include "../../CodeGenere/photogram/cEqBBCamThird_AccelCsteCoord.cpp"


#include "../../CodeGenere/photogram/cGen2DBundleEgProj_Deg0.cpp"
#include "../../CodeGenere/photogram/cGen2DBundleEgProj_Deg1.cpp"
#include "../../CodeGenere/photogram/cGen2DBundleEgProj_Deg2.cpp"
#include "../../CodeGenere/photogram/cGen2DBundleEgProj_Deg3.cpp"
#include "../../CodeGenere/photogram/cGen2DBundleEgProj_Deg4.cpp"
#include "../../CodeGenere/photogram/cGen2DBundleEgProj_Deg5.cpp"
#include "../../CodeGenere/photogram/cGen2DBundleEgProj_Deg6.cpp"
#include "../../CodeGenere/photogram/cGen2DBundleEgProj_Deg7.cpp"

#include "../../CodeGenere/photogram/cGen2DBundleAttach_Deg0.cpp"
#include "../../CodeGenere/photogram/cGen2DBundleAttach_Deg1.cpp"
#include "../../CodeGenere/photogram/cGen2DBundleAttach_Deg2.cpp"
#include "../../CodeGenere/photogram/cGen2DBundleAttach_Deg3.cpp"
#include "../../CodeGenere/photogram/cGen2DBundleAttach_Deg4.cpp"
#include "../../CodeGenere/photogram/cGen2DBundleAttach_Deg5.cpp"
#include "../../CodeGenere/photogram/cGen2DBundleAttach_Deg6.cpp"
#include "../../CodeGenere/photogram/cGen2DBundleAttach_Deg7.cpp"


#include "../../CodeGenere/photogram/cGen2DBundleAtRot_Deg0.cpp"
#include "../../CodeGenere/photogram/cGen2DBundleAtRot_Deg1.cpp"
#include "../../CodeGenere/photogram/cGen2DBundleAtRot_Deg2.cpp"
#include "../../CodeGenere/photogram/cGen2DBundleAtRot_Deg3.cpp"
#include "../../CodeGenere/photogram/cGen2DBundleAtRot_Deg4.cpp"
#include "../../CodeGenere/photogram/cGen2DBundleAtRot_Deg5.cpp"
#include "../../CodeGenere/photogram/cGen2DBundleAtRot_Deg6.cpp"
#include "../../CodeGenere/photogram/cGen2DBundleAtRot_Deg7.cpp"

//=======================  Droite ========================




#include "../../CodeGenere/photogram/cEqAppui_Droite_GL__PTInc_M2CNoVar.cpp"
#include "../../CodeGenere/photogram/cEqAppui_Droite_GL__PProjInc_M2CNoVar.cpp"
#include "../../CodeGenere/photogram/cEqAppui_Droite_NoDist__GL__PTInc_M2CNoVar.cpp"
#include "../../CodeGenere/photogram/cEqAppui_Droite_NoDist__GL__PProjInc_M2CNoVar.cpp"
#include "../../CodeGenere/photogram/cEqAppui_Droite_PTInc_M2CNoVar.cpp"
#include "../../CodeGenere/photogram/cEqAppui_Droite_PProjInc_M2CNoVar.cpp"
#include "../../CodeGenere/photogram/cEqAppui_Droite_NoDist__PTInc_M2CNoVar.cpp"
#include "../../CodeGenere/photogram/cEqAppui_Droite_NoDist__PProjInc_M2CNoVar.cpp"

#include "../../CodeGenere/photogram/cEqAppui_Droite_GL__PTInc_M2CDRad5.cpp"
#include "../../CodeGenere/photogram/cEqAppui_Droite_GL__PProjInc_M2CDRad5.cpp"
#include "../../CodeGenere/photogram/cEqAppui_Droite_NoDist__GL__PTInc_M2CDRad5.cpp"
#include "../../CodeGenere/photogram/cEqAppui_Droite_NoDist__GL__PProjInc_M2CDRad5.cpp"
#include "../../CodeGenere/photogram/cEqAppui_Droite_PTInc_M2CDRad5.cpp"
#include "../../CodeGenere/photogram/cEqAppui_Droite_PProjInc_M2CDRad5.cpp"
#include "../../CodeGenere/photogram/cEqAppui_Droite_NoDist__PTInc_M2CDRad5.cpp"
#include "../../CodeGenere/photogram/cEqAppui_Droite_NoDist__PProjInc_M2CDRad5.cpp"


#include "../../CodeGenere/photogram/cEqAppui_Droite_GL__PTInc_M2CDRad_PPaEqPPs.cpp"
#include "../../CodeGenere/photogram/cEqAppui_Droite_GL__PProjInc_M2CDRad_PPaEqPPs.cpp"
#include "../../CodeGenere/photogram/cEqAppui_Droite_NoDist__GL__PTInc_M2CDRad_PPaEqPPs.cpp"
#include "../../CodeGenere/photogram/cEqAppui_Droite_NoDist__GL__PProjInc_M2CDRad_PPaEqPPs.cpp"
#include "../../CodeGenere/photogram/cEqAppui_Droite_PTInc_M2CDRad_PPaEqPPs.cpp"
#include "../../CodeGenere/photogram/cEqAppui_Droite_PProjInc_M2CDRad_PPaEqPPs.cpp"
#include "../../CodeGenere/photogram/cEqAppui_Droite_NoDist__PTInc_M2CDRad_PPaEqPPs.cpp"
#include "../../CodeGenere/photogram/cEqAppui_Droite_NoDist__PProjInc_M2CDRad_PPaEqPPs.cpp"


#endif // ELISE_INSERT_CODE_GEN





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
