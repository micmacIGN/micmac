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

#include "../../CodeGenere/photogram/cEqCoplanDPol3Id.cpp" // 
#include "../../CodeGenere/photogram/cEqCoplanDPol5Id.cpp" // 
#include "../../CodeGenere/photogram/cEqCoplanDPol7Id.cpp" // 
#include "../../CodeGenere/photogram/cEqHomogrXDeg5.cpp"   //
#include "../../CodeGenere/photogram/cEqHomogrYDeg5.cpp"   // 
#include "../../CodeGenere/photogram/cEqHomogrSpaceInitX.cpp"   //
#include "../../CodeGenere/photogram/cEqHomogrSpaceInitY.cpp"   //
#include "../../CodeGenere/photogram/cEqHomogrX.cpp"   //
#include "../../CodeGenere/photogram/cEqHomogrY.cpp"   // 
#include "../../CodeGenere/photogram/cEqOneHomogrX.cpp"   // 
#include "../../CodeGenere/photogram/cEqOneHomogrY.cpp"   // 


   // GRILLE
#include "../../CodeGenere/photogram/cEqCoplanGrid.cpp"    //
#include "../../CodeGenere/photogram/cEqResiduIm1Grid.cpp" //
#include "../../CodeGenere/photogram/cEqResiduIm2Grid.cpp" //
#include "../../CodeGenere/photogram/cEqAppuiGrid.cpp" //
    // ELLIPSE
#include "../../CodeGenere/photogram/cEqCibleEllipse1.cpp" // 
#include "../../CodeGenere/photogram/cEqCibleEllipse5.cpp" // 
#include "../../CodeGenere/photogram/cEqCibleEllipse7.cpp" // 
#include "../../CodeGenere/photogram/cEqCibleEllipse6.cpp" // 

    // EBNER-BROWN
    //
//#include "../../CodeGenere/photogram/cEqAppuiXEbner.cpp"
//#include "../../CodeGenere/photogram/cEqAppuiYEbner.cpp"
//#include "../../CodeGenere/photogram/cEqAppuiXDCBrown.cpp"
//#include "../../CodeGenere/photogram/cEqAppuiYDCBrown.cpp"
#include "../../CodeGenere/photogram/cEqCoplanEbnerId.cpp"
#include "../../CodeGenere/photogram/cEqCoplanDCBrownId.cpp"
#include "../../CodeGenere/photogram/cEqResiduIm1EbnerId.cpp"
#include "../../CodeGenere/photogram/cEqResiduIm1DCBrownId.cpp"
#include "../../CodeGenere/photogram/cEqResiduIm2EbnerId.cpp"
#include "../../CodeGenere/photogram/cEqResiduIm2DCBrownId.cpp"


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
