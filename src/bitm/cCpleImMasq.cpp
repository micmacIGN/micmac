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




int NbDilOfMode(eModeInterp aMode)
{
   if (aMode == eModeBilin) return 1;
   if (aMode == eModeBicub) return 2;
   ELISE_ASSERT(false,"NbDilOfMode");
   return -1;
}

Im2D_Bits<1> MasqForInterpole(Im2D_Bits<1> aMasqInInit,eModeInterp aMode)
{
   Pt2di aSzIn = aMasqInInit.sz();
   Im2D_Bits<1> aMasqInDil(aSzIn.x,aSzIn.y,1);

   TIm2DBits<1>  aTMasqDil(aMasqInDil);
   TIm2DBits<1>  aTMasqInit(aMasqInInit);
   INT aNbDil = NbDilOfMode(aMode);
   for (INT aY=-aNbDil ; aY<aSzIn.y+aNbDil ; aY++)
   {
       for (INT aX=-aNbDil ; aX<aSzIn.x+aNbDil ; aX++)
       {
           if (aTMasqInit.get(Pt2di(aX,aY),0)==0)
           {
               for (INT aDx = -aNbDil ; aDx < aNbDil ; aDx++)
                   for (INT aDy = -aNbDil ; aDy < aNbDil ; aDy++)
                        aTMasqDil.oset_svp(Pt2di(aX+aDx,aY+aDy),0);
           }
       }
   }
   return aMasqInDil;
}



template <class TOut,class TIn>
void cChCoord<TOut,TIn>::DoIt
     (
       const ElDistortion22_Gen & aDist,
       TOut aImOut,Im2D_Bits<1> aMasqOut,
       TIn aImIn,Im2D_Bits<1> aMasqInInit,
       eModeInterp            aMode,
       REAL aFact
     )
{
   Pt2di aSzIn = aImIn.sz();
   Pt2di aSzOut = aImOut.sz();
   ELISE_ASSERT
   (
       (aMasqOut.sz() ==aSzOut) && (aMasqInInit.sz()==aSzIn),
       "SzPb In ChCoord"
   );
   Im2D_Bits<1> aMasqInDil = MasqForInterpole(aMasqInInit,aMode);
   TIm2DBits<1>  aTMasqDil(aMasqInDil);

   cCubicInterpKernel aKer(-0.5);

   TIm2DBits<1>  aTMasqOut(aMasqOut);
   TIm2D<typename TIn::tElem,typename TIn::tBase> aTImIn(aImIn);
   typename TOut::tElem ** aDout = aImOut.data(); 

   for (INT aY=0; aY<aSzOut.y ; aY++)
   {
if ((aY%100)==0) cout << aY << "\n";
       for (INT aX=0; aX<aSzOut.x ; aX++)
       {
           Pt2di aPOut(aX,aY);
           Pt2dr aPInR = aDist.Direct(Pt2dr(aPOut));
           Pt2di aPInI = round_down(aPInR);
           if (aTMasqDil.get(aPInI,0)==0)
           {
               aTMasqOut.oset(aPOut,0);
               // On donne quand meme une valeur
               aDout[aY][aX] = (typename TOut::tElem) aTImIn.getprojR(aPInR);
           }
           else
           {
               aTMasqOut.oset(aPOut,1);
               REAL aV = 0;
               if (aMode==eModeBilin)
                  aV = aTImIn.getr(aPInR) * aFact;
               else
               {
                  aV = aTImIn.getr(aKer,aPInR) * aFact;
               }
               aDout[aY][aX] = El_CTypeTraits<typename TOut::tElem>::Tronque
                               (
                                      (typename TOut::tBase) aV
                               );
           }

       }
   }
}



Im2D_Bits<1> ReducCentered(Im2D_Bits<1> aImIn)
{
   Pt2di aSzIn = aImIn.sz();
   Pt2di aSzR = (aSzIn + Pt2di(1,1)) /2;

   Im2D_Bits<1> aIRes(aSzR.x,aSzR.y);
   TIm2DBits<1> aTIRes(aIRes);
   TIm2DBits<1> aTIn(aImIn);
   
   for (INT aX=0 ; aX < aSzR.x ; aX++)
   {
       for (INT aY =0 ; aY< aSzR.y ; aY++)
       {
               INT X2 = aX * 2;
               INT Y2 = aY * 2;

               aTIRes.oset
               (
                      Pt2di(aX,aY),
                         aTIn.get(Pt2di(X2-1,Y2-1),0)
                      && aTIn.get(Pt2di(X2+1,Y2-1),0)
                      && aTIn.get(Pt2di(X2-1,Y2+1),0)
                      && aTIn.get(Pt2di(X2+1,Y2+1),0)
                      && aTIn.get(Pt2di(X2  ,Y2-1),0)
                      && aTIn.get(Pt2di(X2  ,Y2+1),0)
                      && aTIn.get(Pt2di(X2-1,Y2  ),0)
                      && aTIn.get(Pt2di(X2+1,Y2  ),0)
                      && aTIn.get(Pt2di(X2  ,Y2  ),0)
               );
             
        }
   }

    return aIRes;
}





template  class cChCoord<Im2D_REAL4,Im2D_U_INT2>;
template  class cChCoord<Im2D_U_INT1,Im2D_U_INT2>;
template  class cChCoord<Im2D_U_INT2,Im2D_U_INT2>;
template  class cChCoord<Im2D_U_INT1,Im2D_U_INT1>;


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
