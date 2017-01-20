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


#include "../../CodeGenere/photogram/cREgDistDxDy_CamBilin.cpp"
#include "../../CodeGenere/photogram/cREgDistDxx_CamBilin.cpp"
#include "../../CodeGenere/photogram/cREgDistDx_CamBilin.cpp"
#include "../../CodeGenere/photogram/cREgDistVal_CamBilin.cpp"

#include "../../CodeGenere/photogram/cREgDistDxDy_DCBrown.cpp"
#include "../../CodeGenere/photogram/cREgDistDxx_DCBrown.cpp"
#include "../../CodeGenere/photogram/cREgDistDx_DCBrown.cpp"
#include "../../CodeGenere/photogram/cREgDistVal_DCBrown.cpp"


#include "../../CodeGenere/photogram/cREgDistDxDy_DRad5APFraser.cpp"
#include "../../CodeGenere/photogram/cREgDistDxx_DRad5APFraser.cpp"
#include "../../CodeGenere/photogram/cREgDistDx_DRad5APFraser.cpp"
#include "../../CodeGenere/photogram/cREgDistVal_DRad5APFraser.cpp"

#include "../../CodeGenere/photogram/cREgDistDxDy_DRad5.cpp"
#include "../../CodeGenere/photogram/cREgDistDxx_DRad5.cpp"
#include "../../CodeGenere/photogram/cREgDistDx_DRad5.cpp"
#include "../../CodeGenere/photogram/cREgDistVal_DRad5.cpp"

#include "../../CodeGenere/photogram/cREgDistDxDy_DRad_PPaEqPPs.cpp"
#include "../../CodeGenere/photogram/cREgDistDxx_DRad_PPaEqPPs.cpp"
#include "../../CodeGenere/photogram/cREgDistDx_DRad_PPaEqPPs.cpp"
#include "../../CodeGenere/photogram/cREgDistVal_DRad_PPaEqPPs.cpp"

#include "../../CodeGenere/photogram/cREgDistDxDy_Ebner.cpp"
#include "../../CodeGenere/photogram/cREgDistDxx_Ebner.cpp"
#include "../../CodeGenere/photogram/cREgDistDx_Ebner.cpp"
#include "../../CodeGenere/photogram/cREgDistVal_Ebner.cpp"

#include "../../CodeGenere/photogram/cREgDistDxDy_EquiSolid_FishEye_10_5_5.cpp"
#include "../../CodeGenere/photogram/cREgDistDxx_EquiSolid_FishEye_10_5_5.cpp"
#include "../../CodeGenere/photogram/cREgDistDx_EquiSolid_FishEye_10_5_5.cpp"
#include "../../CodeGenere/photogram/cREgDistVal_EquiSolid_FishEye_10_5_5.cpp"

#include "../../CodeGenere/photogram/cREgDistDxDy_FishEye_10_5_5.cpp"
#include "../../CodeGenere/photogram/cREgDistDxx_FishEye_10_5_5.cpp"
#include "../../CodeGenere/photogram/cREgDistDx_FishEye_10_5_5.cpp"
#include "../../CodeGenere/photogram/cREgDistVal_FishEye_10_5_5.cpp"


  // Four

#include "../../CodeGenere/photogram/cREgDistDxDy_Four11x2.cpp"
#include "../../CodeGenere/photogram/cREgDistDxx_Four11x2.cpp"
#include "../../CodeGenere/photogram/cREgDistDx_Four11x2.cpp"
#include "../../CodeGenere/photogram/cREgDistVal_Four11x2.cpp"

#include "../../CodeGenere/photogram/cREgDistDxDy_Four15x2.cpp"
#include "../../CodeGenere/photogram/cREgDistDxx_Four15x2.cpp"
#include "../../CodeGenere/photogram/cREgDistDx_Four15x2.cpp"
#include "../../CodeGenere/photogram/cREgDistVal_Four15x2.cpp"

#include "../../CodeGenere/photogram/cREgDistDxDy_Four19x2.cpp"
#include "../../CodeGenere/photogram/cREgDistDxx_Four19x2.cpp"
#include "../../CodeGenere/photogram/cREgDistDx_Four19x2.cpp"
#include "../../CodeGenere/photogram/cREgDistVal_Four19x2.cpp"

#include "../../CodeGenere/photogram/cREgDistDxDy_Four7x2.cpp"
#include "../../CodeGenere/photogram/cREgDistDxx_Four7x2.cpp"
#include "../../CodeGenere/photogram/cREgDistDx_Four7x2.cpp"
#include "../../CodeGenere/photogram/cREgDistVal_Four7x2.cpp"


    // ===============

#include "../../CodeGenere/photogram/cREgDistDxDy_Fraser_PPaEqPPs.cpp"
#include "../../CodeGenere/photogram/cREgDistDxx_Fraser_PPaEqPPs.cpp"
#include "../../CodeGenere/photogram/cREgDistDx_Fraser_PPaEqPPs.cpp"
#include "../../CodeGenere/photogram/cREgDistVal_Fraser_PPaEqPPs.cpp"

#include "../../CodeGenere/photogram/cREgDistDxDy_NoDist.cpp"
#include "../../CodeGenere/photogram/cREgDistDxx_NoDist.cpp"
#include "../../CodeGenere/photogram/cREgDistDx_NoDist.cpp"
#include "../../CodeGenere/photogram/cREgDistVal_NoDist.cpp"

  // Polyn
#include "../../CodeGenere/photogram/cREgDistDxDy_Polyn0.cpp"
#include "../../CodeGenere/photogram/cREgDistDxx_Polyn0.cpp"
#include "../../CodeGenere/photogram/cREgDistDx_Polyn0.cpp"
#include "../../CodeGenere/photogram/cREgDistVal_Polyn0.cpp"

#include "../../CodeGenere/photogram/cREgDistDxDy_Polyn1.cpp"
#include "../../CodeGenere/photogram/cREgDistDxx_Polyn1.cpp"
#include "../../CodeGenere/photogram/cREgDistDx_Polyn1.cpp"
#include "../../CodeGenere/photogram/cREgDistVal_Polyn1.cpp"

#include "../../CodeGenere/photogram/cREgDistDxDy_Polyn2.cpp"
#include "../../CodeGenere/photogram/cREgDistDxx_Polyn2.cpp"
#include "../../CodeGenere/photogram/cREgDistDx_Polyn2.cpp"
#include "../../CodeGenere/photogram/cREgDistVal_Polyn2.cpp"

#include "../../CodeGenere/photogram/cREgDistDxDy_Polyn3.cpp"
#include "../../CodeGenere/photogram/cREgDistDxx_Polyn3.cpp"
#include "../../CodeGenere/photogram/cREgDistDx_Polyn3.cpp"
#include "../../CodeGenere/photogram/cREgDistVal_Polyn3.cpp"

#include "../../CodeGenere/photogram/cREgDistDxDy_Polyn4.cpp"
#include "../../CodeGenere/photogram/cREgDistDxx_Polyn4.cpp"
#include "../../CodeGenere/photogram/cREgDistDx_Polyn4.cpp"
#include "../../CodeGenere/photogram/cREgDistVal_Polyn4.cpp"

#include "../../CodeGenere/photogram/cREgDistDxDy_Polyn5.cpp"
#include "../../CodeGenere/photogram/cREgDistDxx_Polyn5.cpp"
#include "../../CodeGenere/photogram/cREgDistDx_Polyn5.cpp"
#include "../../CodeGenere/photogram/cREgDistVal_Polyn5.cpp"

#include "../../CodeGenere/photogram/cREgDistDxDy_Polyn6.cpp"
#include "../../CodeGenere/photogram/cREgDistDxx_Polyn6.cpp"
#include "../../CodeGenere/photogram/cREgDistDx_Polyn6.cpp"
#include "../../CodeGenere/photogram/cREgDistVal_Polyn6.cpp"

#include "../../CodeGenere/photogram/cREgDistDxDy_Polyn7.cpp"
#include "../../CodeGenere/photogram/cREgDistDxx_Polyn7.cpp"
#include "../../CodeGenere/photogram/cREgDistDx_Polyn7.cpp"
#include "../../CodeGenere/photogram/cREgDistVal_Polyn7.cpp"




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
