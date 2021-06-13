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

#include "cp_header.h"


int main(int,char **)
{
   // Pt2di aSz(50,50);
   Pt2di aSz(200,200);
   Pt2di aDec(400,200);
   double aStep = 0.125;
   std::string aNameDir = "/DATA2/JeuxReferences/TestHRS/PbIGN-ESP-Epipolaires/";
   std::string aNameI1  = "H220021026_EXT_EPI_REF.HDR";
   std::string aNameI2  = "H220021026_EXT_EPI_SEC.HDR";

    std::string aNameRes = aNameDir + "ResCorPonct/";
    ELISE_fp::MkDirSvp(aNameRes);

   cAppliCorrPont anAppli(aSz);
   anAppli.InitFiles(aNameDir,aNameI1,aNameI2,aDec);
    
   anAppli.MakeCorrPonct(2,0.125,-2.0,2.0001);

   Im2D_REAL8 aIPax = anAppli.IPax();
   Im2D_REAL8 aIm1 = anAppli.Im1();
   Im2D_REAL8 aIm2 = anAppli.Im2();

//======================================= 
   Tiff_Im::Create8BFromFonc
   (
        aNameRes + "NormI1.tif",
	aIm1.sz(),
	FoncNormalisee_S1S2(aIm1.all_pts(),aIm1.in())
   );
   Tiff_Im::Create8BFromFonc(aNameRes+"I1.tif", aIm1.sz(),aIm1.in());
//======================================= 
   Tiff_Im::Create8BFromFonc
   (
        aNameRes + "NormI2.tif",
	aIm2.sz(),
	FoncNormalisee_S1S2(aIm2.all_pts(),aIm2.in())
   );
   Tiff_Im::Create8BFromFonc(aNameRes+"I2.tif", aIm2.sz(),aIm2.in());
//======================================= 
   Tiff_Im::Create8BFromFonc
   (
        aNameRes + "Pax.tif",
	aIPax.sz(),
        Max(0,Min(255,round_ni(128+aIPax.in()/aStep)))
   );


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
