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
#include "im_tpl/correl_imget.h"


#define DEF_OUT -2.5


template <class Type> void OptimTranslationCorrelation<Type>::SetP0Im1(Pt2dr  aP0Im1)
{
   mP0Im1 = aP0Im1;
}

template <class Type> OptimTranslationCorrelation<Type>::~OptimTranslationCorrelation()
{
   delete &mQuickCorrel;
}

template <class Type> OptimTranslationCorrelation<Type>::OptimTranslationCorrelation
(
    REAL     aStepLim,
    REAL     aStepInit,
    INT      aSzVoisInit,
    Im2D<Type,INT> aIm1,
    Im2D<Type,INT> aIm2,
    REAL     aSzVignette,
    REAL     aStepIm
)  :
    Optim2DParam(aStepLim,DEF_OUT,1e-6,true),
    mIm1 (aIm1),
    mIm2 (aIm2),
    mSzVignette (aSzVignette),
    mStepIm     (aStepIm),
    mQuickCorrel (* (new TImageFixedCorrelateurSubPix<Type> 
                         (aIm1,aIm2,aSzVignette,aStepIm,DEF_OUT)
                     )
                  ),
    mCanUseICorrel (     (mQuickCorrel.NbPts() < 10000) 
                      && (  nbb_type_num(type_of_ptr((Type *)0)) <= 8)
                   )
{
   set(aStepInit,aSzVoisInit);
}


REAL Epsilon = 1e-7;

template <class Type> REAL OptimTranslationCorrelation<Type>::ScoreFonc(REAL aDx,REAL aDy)
{
     INT  aSzV = round_up(mSzVignette / mStepIm);

     Symb_FNum fx1  ( (FX*mStepIm) + mP0Im1.x);
     Symb_FNum fy1  ( (FY*mStepIm) + mP0Im1.y);

     Symb_FNum fx2 ( fx1 + aDx);
     Symb_FNum fy2 ( fy1 + aDy);

     Symb_FNum iM1 (mIm1.in(0)[Virgule(fx1,fy1)]);
     Symb_FNum iM2 (mIm2.in(0)[Virgule(fx2,fy2)]);

     REAL s,s1,s2,s11,s12,s22;
     ELISE_COPY
     (
           rectangle(Pt2di(-aSzV,-aSzV),Pt2di(aSzV+1,aSzV+1)),
           Virgule(Virgule(1.0,iM1,iM2),Virgule(Square(iM1),iM1*iM2,Square(iM2))),
           Virgule
           (
               sigma(s)  , sigma(s1) , sigma(s2),
               sigma(s11), sigma(s12), sigma(s22)
           )
     );

     s1 /= s;
     s2 /= s;

     s11 /= s;
     s12 /= s;
     s22 /= s;

     s11 -= s1*s1;
     s12 -= s1*s2;
     s22 -= s2*s2;

     mSc =  s12 / sqrt(ElMax(Epsilon,s11*s22));
     return mSc;
}

template <class Type> REAL OptimTranslationCorrelation<Type>::Op2DParam_ComputeScore(REAL aDx,REAL aDy)
{
   if (mCanUseICorrel)
       mSc = mQuickCorrel.icorrel(mP0Im1,mP0Im1+Pt2dr(aDx,aDy));
    else
       mSc =   mQuickCorrel.rcorrel(mP0Im1,mP0Im1+Pt2dr(aDx,aDy));

    return  mSc;
}

template class OptimTranslationCorrelation<U_INT1>;

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
