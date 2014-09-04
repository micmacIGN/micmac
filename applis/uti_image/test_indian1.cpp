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
// #include "anag_all.h"

#include "general/all.h"
#include "private/all.h"



void Generate()
{
   L_Arg_Opt_Tiff aList;

   aList = aList + Arg_Tiff(Tiff_Im::ATiles(Pt2di(32,16)));

   Tiff_Im aFile
           (
	       "../TMP/test_indian_r4.tif",
	       Pt2di(300,500),
	       GenIm::real4,
	       Tiff_Im::No_Compr,
	       Tiff_Im::BlackIsZero,
	       aList
	   );

    ELISE_COPY
    (
        aFile.all_pts(),
	sin(FX) * cos(FY),
	aFile.out()
    );


    aFile = Tiff_Im 
           (
	       "../TMP/test_indian_i2.tif",
	       Pt2di(400,250),
	       GenIm::int2,
	       Tiff_Im::No_Compr,
	       Tiff_Im::BlackIsZero,
	       aList
	   );

    ELISE_COPY
    (
        aFile.all_pts(),
	FX-FY,
	aFile.out()
    );
}

void VerifDiff( Tiff_Im aF1,Tiff_Im aF2)
{
   double aVMax;

   ELISE_COPY(aF1.all_pts(), Rconv(aF1.in()-aF2.in()), VMax(aVMax));

   std::cout << "DIF = " << aVMax << "\n";

   ELISE_ASSERT(aVMax<1e-5,"VerifDiff");
}

void Test
     (
         const std::string& aName1,
         const std::string& aName2
     )
{
   Tiff_Im aF1(aName1.c_str());
   Tiff_Im aF2(aName2.c_str());

   VerifDiff(aF1,aF2);

   ELISE_COPY
   (
       disc(Pt2dr(150,150),145.34),
       2.3*FX+ 7.1*FY,
       aF1.out() | aF2.out()
   );
   VerifDiff(aF1,aF2);
}

int main (int argc,char** argv)
{
     // Generate(); 
     Test
     (
         "../TMP/VTest/Intel_test_indian_i2.tif",
         "../TMP/VTest/Motorola_test_indian_i2.tif"
     );
     Test
     (
         "../TMP/VTest/Intel_test_indian_r4.tif",
         "../TMP/VTest/Motorola_test_indian_r4.tif"
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
