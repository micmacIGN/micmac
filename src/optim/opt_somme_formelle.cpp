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


/****************************************************/
/*                                                  */
/*             cOptimSommeFormelle::cMin            */
/*                                                  */
/****************************************************/

cOptimSommeFormelle::cMin::cMin(cOptimSommeFormelle & anOSF) :
   FoncNVarDer<REAL>(anOSF.Dim()),
   mOSF(anOSF)
{
}

REAL cOptimSommeFormelle::cMin::ValFNV(const REAL *  v)
{
    return mOSF.ValFNV(v);
}

void cOptimSommeFormelle::cMin::GradFNV(REAL *grad,const REAL *   v)
{
    mOSF.GradFNV(grad,v);
}

/****************************************************/
/*                                                  */
/*             cOptimSommeFormelle                  */
/*                                                  */
/****************************************************/

  // static cElMatCreuseGen * StdNewOne(INT aNbCol,INT aNbLign,bool Fixe);

cOptimSommeFormelle::cOptimSommeFormelle(INT aNbVar)  :
  mNbVar  (aNbVar),
  mTabDP  (aNbVar,Fonc_Num(0)),
  mSomme  (Fonc_Num(0)),
  mSetInd (aNbVar),  
  mPts    (0),
  mMatCr     (cElMatCreuseGen::StdNewOne(aNbVar,aNbVar,true)),
  mQuadPart  (aNbVar,mMatCr)
{
	// mQuadPart.SetEpsB(1e-10);
}


cOptimSommeFormelle::~cOptimSommeFormelle() 
{
    delete mPts;
} 

INT cOptimSommeFormelle::Dim() const
{
   return mNbVar;
}

void cOptimSommeFormelle::SetPts(const REAL * Vec)
{
    if ((mPts==0) || (mPts->Dim() < Dim()))
    {
       delete mPts;
       mPts = new PtsKD(Dim());
    }
    for (INT k=0; k< Dim() ; k++)
       (*mPts)(k) = Vec[k];
}

REAL cOptimSommeFormelle::ValFNV(const REAL *  v)
{
   SetPts(v);
   return mSomme.ValFonc(*mPts) + mQuadPart.ValFNV(v);
}

void cOptimSommeFormelle::GradFNV(REAL *grad,const REAL *   v)
{
   mQuadPart.GradFNV(grad,v);

   SetPts(v);
   for (INT k=0; k< Dim() ; k++)
       grad[k] +=  mTabDP[k].ValFonc(*mPts);
}

INT cOptimSommeFormelle:: GradConjMin(REAL * aP0,REAL ftol,INT ITMAX)
{

    cMin aMin(*this);
    return aMin.GradConj(aP0,ftol,ITMAX);
}

INT cOptimSommeFormelle:: GradConjMin(PtsKD & aP0,REAL ftol,INT ITMAX)
{
   return GradConjMin(aP0.AdrX0(),ftol,ITMAX);
}



void  cOptimSommeFormelle::Add(Fonc_Num aFonc,bool CasSpecQuad)
{
   aFonc.VarDerNN(mSetInd);
   INT aDegre = -1 ;
   if (CasSpecQuad)
       aDegre = aFonc.DegrePoly();

   

   if ((aDegre >=0) && (aDegre<=2))
   {
      mQuadPart.AddDiff(aFonc,mSetInd);
   }
   else
   {
       mSomme = aFonc+ mSomme;
       for 
       (
          ElGrowingSetInd::const_iterator anIt =  mSetInd.begin();
          anIt != mSetInd.end();
          anIt ++
       )
       {
           ELISE_ASSERT((*anIt>=0) && (*anIt<mNbVar),"Bad Num Var in cOptimSommeFormelle");
           Fonc_Num aDP = aFonc.deriv(*anIt);
           mTabDP[*anIt] = aDP+mTabDP[*anIt];
       }
   }
   mSetInd.clear();
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
