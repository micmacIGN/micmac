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




/*********************************************************************************/
/*                                                                               */
/*                          CalcPtsInteret                                       */
/*                                                                               */
/*********************************************************************************/

Fonc_Num CalcPtsInteret::CritereFonc(Fonc_Num aFonc)
{
    return Abs(courb_tgt(aFonc));
}

Pt2di CalcPtsInteret::GetOnePtsInteret(Flux_Pts aFlux,Fonc_Num aFonc)
{
    Pt2di aP;

    ELISE_COPY
    (
        aFlux,
        CritereFonc(aFonc),
        aP.WhichMax()
    );
    return aP;
}

CalcPtsInteret::tContainerPtsInt CalcPtsInteret::GetEnsPtsInteret_Size
                                 (
                                         Pt2di aP0,
                                         Pt2di aP1,
                                         Fonc_Num aFonc,
                                         REAL aSzRech,
                                         REAL aRatio 
                                 )
{
    Pt2di aSz = aP1 - aP0;
    INT aNbRechX =  round_up(aSz.x/aSzRech);
    INT aNbRechY =  round_up(aSz.y/aSzRech);

    REAL SzRechX = aSz.x/(REAL) aNbRechX;
    REAL SzRechY = aSz.y/(REAL) aNbRechY;

    Pt2dr aDemiSzRect = Pt2dr(SzRechX,SzRechY) * (aRatio/2.0);

    tContainerPtsInt aRes;

    for (INT iX = 0 ; iX < aNbRechX ; iX++)
    {
        for (INT iY = 0 ; iY < aNbRechY ; iY++)
        {

            Pt2dr aPCentre  = Pt2dr(aP0) + Pt2dr((iX+0.5)*SzRechX,(iY+0.5)*SzRechY);

            Pt2di aP0Rect = Pt2di(aPCentre -  aDemiSzRect);
            Pt2di aP1Rect = Pt2di(aPCentre +  aDemiSzRect);

            aRes.push_back
            (
                 Pt2dr
                 (
                    GetOnePtsInteret
                    (
                         rectangle(aP0Rect,aP1Rect),
                         aFonc
                    )
                 )
            );
        }
    }

    return aRes;
}


CalcPtsInteret::tContainerPtsInt CalcPtsInteret::GetEnsPtsInteret_Nb
                                 (
                                         Pt2di aP0,
                                         Pt2di aP1,
                                         Fonc_Num aFonc,
                                         INT  aNb,  // NbTot = NbX * NbY
                                         REAL aRatio 
                                 )
{
   Pt2di aSz = aP1-aP0;
   REAL Surf= aSz.x*aSz.y;

   return GetEnsPtsInteret_Size
          (
              aP0,aP1,aFonc,
              sqrt(Surf/aNb),
              aRatio
          );

}

CalcPtsInteret::tContainerPtsInt CalcPtsInteret::GetEnsPtsInteret_Size
                                 (
                                         Im2D_U_INT1  anIm,
                                         REAL         aSzRech,
                                         REAL         aRatio 
                                 )
{
    return GetEnsPtsInteret_Size(Pt2di(0,0),anIm.sz(),anIm.in(0),aSzRech,aRatio);
}



CalcPtsInteret::tContainerPtsInt CalcPtsInteret::GetEnsPtsInteret_Nb
                                 (
                                         Im2D_U_INT1 anIm,
                                         INT         aNb,
                                         REAL        aRatio
                                 )
{
    return GetEnsPtsInteret_Nb(Pt2di(0,0),anIm.sz(),anIm.in(0),aNb,aRatio);
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
