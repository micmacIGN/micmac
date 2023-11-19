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



/**********************************************/
/*                                            */
/*        cEqTrianguApparImage                */
/*                                            */
/**********************************************/


cEqTrianguApparImage::cEqTrianguApparImage(INT aCapa) :
   mCapa  (aCapa),
   mDerJk (new tV6  [mCapa]),
   mVI    (new REAL [mCapa]),
   mVJ    (new REAL [mCapa])
{
   Reset();
}

cEqTrianguApparImage::~cEqTrianguApparImage()
{
   delete[] mDerJk;
   delete[] mVI;
   delete[] mVJ;
}

void cEqTrianguApparImage::Reset()
{
   mN   = 0;
   mSI1 = 0;
   mSI2 = 0;
   mSJ1 = 0;
   mSJ2 = 0;
 
   for (INT aK6=0 ; aK6<6 ; aK6++)
   {
      mDerSJ1[aK6] = 0;
      mDerSJ2[aK6] = 0;
   }
}

void cEqTrianguApparImage::Close()
{
   mSI1 /= mN;
   mSI2 /= mN;
   mSigmaI = sqrt(mSI2-ElSquare(mSI1));
   mSJ1 /= mN;
   mSJ2 /= mN;

   mSigmaJ2 = mSJ2-ElSquare(mSJ1);
   mSigmaJ =  sqrt(mSigmaJ2);
   mSigmaJ3 = mSigmaJ2 * mSigmaJ;
 
   for (INT aK6=0 ; aK6<6 ; aK6++)
   {
      mDerSJ1[aK6] /= mN;
      mDerSJ2[aK6] *=  2.0/ mN;
      mDerSigmaJ2[aK6] = mDerSJ2[aK6] -2*mDerSJ1[aK6] *mSJ1;
   }
}


void cEqTrianguApparImage::Add
     (
                   REAL aI,
                   REAL aJ,
                   REAL aDxJ,
                   REAL aDyJ,
                   REAL aPdsA,
                   REAL aPdsB,
                   REAL aPdsC
      )
{
    ELISE_ASSERT(mN<mCapa,"Over in cEqTrianguApparImage::Add");

    mVI[mN] = aI;
    mSI1 += aI;
    mSI2 += ElSquare(aI);

    mVJ[mN] = aJ;
    mSJ1 += aJ;
    mSJ2 += ElSquare(aJ);

    REAL * aDerJ = mDerJk[mN];

    aDerJ[0] = aDxJ * aPdsA; 
    aDerJ[1] = aDyJ * aPdsA; 
    aDerJ[2] = aDxJ * aPdsB; 
    aDerJ[3] = aDyJ * aPdsB; 
    aDerJ[4] = aDxJ * aPdsC; 
    aDerJ[5] = aDyJ * aPdsC; 
              
    for (INT aK=0 ; aK<6 ; aK++)
    {
      mDerSJ1[aK] += aDerJ[aK];
      mDerSJ2[aK] += aJ *aDerJ[aK];
    }
    mN++;
}

void  cEqTrianguApparImage::SetDer(REAL & aCste,REAL * aDer,INT aKN)
{
   REAL aVJC = mVJ[aKN]-mSJ1; // Valeur de J centrale
   aCste = aVJC/mSigmaJ - (mVI[aKN]-mSI1)/mSigmaI;

   REAL * aDerJ = mDerJk[aKN];
   for (INT aK6=0;  aK6<6 ; aK6++)
   {
       aDer[aK6] = (
                             (aDerJ[aK6]-mDerSJ1[aK6])*mSigmaJ2
                     - 0.5 * aVJC * mDerSigmaJ2[aK6]
                   ) / mSigmaJ3;
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
