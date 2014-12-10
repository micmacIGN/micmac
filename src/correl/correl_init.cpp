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
               
Fonc_Num EliseCorrelation::FoncLissee
      (
            Fonc_Num  aFonc,
            Pt2di     aSz,
            REAL      aFactLissage,
            INT       aNbStep,
            bool      aUseVOut,
            INT       aVOut
      )
{
   if ((aFactLissage > 0) && (aNbStep>0))
   {
       Fonc_Num aP = aUseVOut ? (aFonc != aVOut) : 1.0;
       aFactLissage /= sqrt(double(aNbStep));
       REAL Coeff = 1 - 1/ (1+aFactLissage);

       aFonc =    canny_exp_filt(clip_def(aFonc,0,Pt2di(0,0),aSz)*aP,Coeff,Coeff)
	       /  canny_exp_filt(inside(Pt2di(0,0),aSz)*aP,Coeff,Coeff);
       for (INT k=1; k<aNbStep ; k++)
            aFonc =    canny_exp_filt(aFonc,Coeff,Coeff) 
		     / canny_exp_filt(1.0,Coeff,Coeff) ;
   }

   return aFonc;
}


Im2D_REAL8 EliseCorrelation::ImLissee
      (
            Fonc_Num  aFonc,
            Pt2di     aSz,
            REAL      aFactLissage,
            INT       aNbStep,
            bool      aUseVOut,
            INT       aVOut
      )
{
   Im2D_REAL8 aRes(aSz.x,aSz.y);
   ELISE_COPY
   (
        aRes.all_pts(),
	FoncLissee(aFonc,aSz,aFactLissage,aNbStep,aUseVOut,aVOut),
	aRes.out() 
 
    );

   return aRes;
}

Im2D_REAL8 EliseCorrelation::ImCorrelComplete
      (
             Fonc_Num f1,
             Fonc_Num f2,
             Pt2di aSz,
             REAL  aRatioBord,
             REAL  FactLissage,
             INT   aNbStep,
	     REAL  anEpsilon,
             bool  aUseVOut,
             INT   aVOut
      )
{
   Im2D_REAL8 aR1 = ImLissee(f1,aSz,FactLissage,aNbStep,aUseVOut,aVOut);
   Im2D_REAL8 aR2 = ImLissee(f2,aSz,FactLissage,aNbStep,aUseVOut,aVOut);


   REAL aSMin = aSz.x*aSz.y*aRatioBord;


   if (aUseVOut)
      return ElFFTPonderedCorrelNCPadded
             (
                    aR1.in(0),aR2.in(0),
                    aSz,
                    (f1 != aVOut), (f2 != aVOut),
                    anEpsilon,
                    aSMin
             );
   else
      return ElFFTCorrelNCPadded(aR1,aR2,anEpsilon,aSMin);

}

Pt2di EliseCorrelation::RechTransMaxCorrel
      (
             Fonc_Num f1,
             Fonc_Num f2,
             Pt2di aSz,
             REAL  aRatioBord,
             REAL  FactLissage,
             INT   aNbStep,
	     REAL  anEpsilon,
             bool  aUseVOut,
             INT   aVOut
      )
{
    Im2D_REAL8  aCor = ImCorrelComplete
                       (
                             f1,f2,aSz,aRatioBord,FactLissage,aNbStep,
                             anEpsilon,aUseVOut,aVOut
                       );

    Pt2di aP;
    ELISE_COPY
    (
         aCor.all_pts(),
	 aCor.in(),
	 aP.WhichMax()
    );


    Pt2di aSzC = aCor.sz()/2;

    if (aP.x >= aSzC.x)
        aP.x -=  aCor.tx();
    if (aP.y >= aSzC.y)
        aP.y -=  aCor.ty();


    return aP;

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
