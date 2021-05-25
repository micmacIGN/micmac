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
#include "im_tpl/image.h"
#include "im_tpl/reduc_im.h"

#include "XML_GEN/all.h"

using namespace NS_ParamChantierPhotogram;

void  OneImage
      (
	  const std::string & aName,
          Pt2di aSz,
	  Fonc_Num aFinit,
          bool  DoCercle = true
      )
{
   Tiff_Im::Create8BFromFonc(aName,aSz,aFinit);
   Tiff_Im aTif(aName.c_str());

   if (DoCercle)
   {
      double aRayon=10;
      int aNbx=5;
      int aNby=4;

      for (int aKx =0 ; aKx<aNbx ; aKx++)
      {
          for (int aKy =0 ; aKy<aNby ; aKy++)
          {
             // Pt2dr aP = Pt2dr(aRayon,aRayon);
	     double aCx = aRayon + (aSz.x-2*aRayon) * (aKx/double(aNbx-1));
	     double aCy = aRayon + (aSz.y-2*aRayon) * (aKy/double(aNby-1));

	     Fonc_Num aFC = (255/aRayon) * (aRayon -sqrt(Square(FX-aCx)+Square(FY-aCy)));
	     if (aFinit.dimf_out() == 3)
	        aFC = Virgule(aFC,aFC,aFC);
             ELISE_COPY
	     (
	         disc(Pt2dr(aCx,aCy),aRayon),
	         Max(0,Min(255,aFC)),
	         aTif.out()
	     );
          }
      }
   }
}

Fonc_Num Pyram(int aSz)
{
   return  (Abs((FX%aSz) -aSz/2) + Abs((FY%aSz) -aSz/2)) / (aSz-1.0);
}

int main (int argc,char** argv)
{                 

    Pt2di aSz (1000,800);

    if (0)
    {
       // int aVR = 1;

       // Tiff_Im::Create8BFromFonc
       OneImage
       (
            "../TMP/SinRanC.tif",
            aSz,
	    Virgule
	    (
	        Min(255,255*unif_noise_1(1)),
	        1+127*(1+sin(FX/5.0)),
	        1+127*(1+sin(FY/5.0))
            )
       );
    }

    if (1)
    {
       OneImage
       (
            "TexureAleatoire.tif",
            Pt2di(1600,960),
	    Virgule
	    (
	        255* (Pyram(5) *0.3 +  unif_noise_2(3)*0.4 +  unif_noise_4(8)*0.3),
	        255* (((FX+FY)%2)* 0.2  + unif_noise_1(1)*0.6 +  unif_noise_4(3)*0.2),
	        255* (Pyram(9)*0.2 +  unif_noise_2(2)*0.3 +  unif_noise_4(6)*0.5)
           ),
           false
       );
    }

    if (0)
    {
       int aVR = 2,aVV = 5, aVB = 10;
       double  aP1=1;

       OneImage
       (
            "../TMP/RanC.tif",
            aSz,
	    Virgule
	    (
	        Min(255,255*unif_noise_4(&aP1,&aVR,1)) ,
	        Min(255,255*unif_noise_4(&aP1,&aVV,1)) ,
	        Min(255,255*unif_noise_4(&aP1,&aVB,1)) 
            )
       );
    }



    if (0)
    {
       static const int aNbV = 4;
       double  aPds[aNbV] ={1.0,1.0,1.0,1.0};
       int     aV[aNbV] ={1,2,5,10};

       OneImage
       (
            "../TMP/Ran.tif",
            aSz,
	    Min(255,255*unif_noise_4(aPds,aV,aNbV)) 
       );
    }

    if (0)
    {
        OneImage
        (
            "../TMP/SinX_1.tif",
	    aSz,
	    1+127*(1+sin(FX/5.0))
        );

        OneImage
        (
            "../TMP/SinXSinY_1.tif",
	    aSz,
	    1+63*(1+sin(FX/5.0))*(1+sin(FY/20.0))
        );

        OneImage
        (
            "../TMP/SinXCol_1.tif",
	    aSz,
	    Virgule
	    (
	         1+127*(1+sin(FX/2.0)),
	         1+127*(1+sin(FX/5.1)),
	         1+127*(1+sin(FX/12.7))
	    )
         );
    }

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
