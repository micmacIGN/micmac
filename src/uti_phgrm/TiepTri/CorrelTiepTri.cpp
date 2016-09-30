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

#include "TiepTri.h"

const double TT_DefCorrel = -2;

template <class Type> bool inside_window(const Type & Im1, const Pt2di & aP1,const int &   aSzW)
{
   return Im1.inside(aP1-Pt2di(aSzW,aSzW)) && Im1.inside(aP1+Pt2di(aSzW,aSzW));
}


double TT_CorrelBasique
                             (
                                const tTImTiepTri & Im1,
                                const Pt2di & aP1,
                                const tTImTiepTri & Im2,
                                const Pt2di & aP2,
                                const int   aSzW,
                                const int   aStep
                             )
{
 
     if (! (inside_window(Im1,aP1,aSzW*aStep) && inside_window(Im2,aP2,aSzW*aStep))) return TT_DefCorrel;

     Pt2di aVois;
     RMat_Inertie aMatr;

     for  (aVois.x = -aSzW ; aVois.x<=aSzW  ; aVois.x++)
     {
          for  (aVois.y = -aSzW ; aVois.y<=aSzW  ; aVois.y++)
          {
               aMatr.add_pt_en_place(Im1.get(aP1+aVois*aStep),Im2.get(aP2+aVois*aStep));
          }
     }
     
     return aMatr.correlation();
}

cResulRechCorrel<int> TT_RechMaxCorrelBasique
                      (
                             const tTImTiepTri & Im1,
                             const Pt2di & aP1,
                             const tTImTiepTri & Im2,
                             const Pt2di & aP2,
                             const int   aSzW,
                             const int   aStep,
                             const int   aSzRech
                      )
{
    double aCorrelMax = TT_DefCorrel;
    Pt2di  aDecMax;
    Pt2di  aP;
    for (aP.x=-aSzRech ; aP.x<= aSzRech ; aP.x++)
    {
        for (aP.y=-aSzRech ; aP.y<= aSzRech ; aP.y++)
        {
             double aCorrel = TT_CorrelBasique(Im1,aP1,Im2,aP2+aP,aSzW,aStep);
             if (aCorrel> aCorrelMax)
             {
                 aCorrelMax = aCorrel;
                 aDecMax = aP;
             }
        }
     }

     return cResulRechCorrel<int>(aDecMax,aCorrelMax);

}
 
class cTT_MaxLocCorrelBasique : public Optim2DParam
{
    public :
         REAL Op2DParam_ComputeScore(REAL aDx,REAL aDy) 
         {
              return  TT_CorrelBasique(mIm1,mP1,mIm2,mP2+Pt2di(round_ni(aDx),round_ni(aDy)),mSzW,mStep);
         }
 
         cTT_MaxLocCorrelBasique 
         ( 
              const tTImTiepTri & aIm1,
              const Pt2di &       aP1,
              const tTImTiepTri & aIm2,
              const Pt2di &       aP2,
              const int           aSzW,
              const int           aStep
         )  :
            Optim2DParam ( 0.9, TT_DefCorrel ,1e-5, true),
            mIm1 (aIm1),
            mP1  (aP1),
            mIm2 (aIm2),
            mP2  (aP2),
            mSzW (aSzW),
            mStep (aStep)
         {
         }
            

         const tTImTiepTri & mIm1;
         const Pt2di & mP1;
         const tTImTiepTri & mIm2;
         const Pt2di & mP2;
         const int   mSzW;
         const int   mStep;
};

cResulRechCorrel<int> TT_RechMaxCorrelLocale
                      (
                             const tTImTiepTri & aIm1,
                             const Pt2di & aP1,
                             const tTImTiepTri & aIm2,
                             const Pt2di & aP2,
                             const int   aSzW,
                             const int   aStep,
                             const int   aSzRechMax
                      )

{
   cTT_MaxLocCorrelBasique  anOpt(aIm1,aP1,aIm2,aP2,aSzW,aStep);
   anOpt.optim_step_fixed(Pt2dr(0,0),aSzRechMax);

   return cResulRechCorrel<int>(round_ni(anOpt.param()),anOpt.ScOpt());

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
aooter-MicMac-eLiSe-25/06/2007*/
