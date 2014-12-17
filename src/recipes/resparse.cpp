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

//==============================================================

class  cVecByIm1
{
     public :
        typedef Im1D_REAL8  tVec;

        static double  Scal(const tVec & aV1,const tVec & aV2);
        static void  AddAlphaV(tVec & aV1,const double  aAlpha,const tVec & aV2);  // aV1 += aAlpha * aV2
        static void  MulAlphaAndAdd(tVec & aV1,const double  aAlpha,const tVec & aV2);  // aV1 = aV1*aAlpha + aV2
        static void  SetDiff(tVec & aV1,const tVec & aV2,const tVec & aV3);
        static void  Affect(tVec & aV1,const tVec & aV2);

        static  int ParamInit;

        static void Resize(tVec & aV1,const tVec & aV2);
        static int NbInc(tVec & aV1);
};


int  cVecByIm1::ParamInit = 1;

int cVecByIm1::NbInc(tVec & aV1)
{
   return aV1.tx();
}

void cVecByIm1::Resize(tVec & aV1,const tVec & aV2)
{
   aV1.Resize(aV2.tx());
}


double  cVecByIm1::Scal(const tVec & aV1,const tVec & aV2)
{
   int aNb = aV1.tx();
   const double * aD1 = aV1.data();
   const double * aD2 = aV2.data();
   double aRes = 0.0;

   for (int aK=0 ; aK<aNb ; aK++)
       aRes += aD1[aK] * aD2[aK];

   return aRes;
}

void  cVecByIm1::AddAlphaV(tVec & aV1,const double aAlpha,const tVec & aV2)
{
   int aNb = aV1.tx();
   double * aD1 = aV1.data();
   const double * aD2 = aV2.data();

   for (int aK=0 ; aK<aNb ; aK++)
        aD1[aK] += aAlpha * aD2[aK];

}

void  cVecByIm1::MulAlphaAndAdd(tVec & aV1,const double  aAlpha,const tVec & aV2)
{
   int aNb = aV1.tx();
   double * aD1 = aV1.data();
   const double * aD2 = aV2.data();

   for (int aK=0 ; aK<aNb ; aK++)
        aD1[aK] = aAlpha * aD1[aK] + aD2[aK];
}


void  cVecByIm1::SetDiff(tVec & aV1,const tVec & aV2,const tVec & aV3)
{
   int aNb = aV1.tx();
   double * aD1 = aV1.data();
   const double * aD2 = aV2.data();
   const double * aD3 = aV3.data();

   for (int aK=0 ; aK<aNb ; aK++)
        aD1[aK] = aD2[aK] -aD3[aK];
}

void  cVecByIm1::Affect(tVec & aV1,const tVec & aV2)
{
   int aNb = aV1.tx();
   double * aD1 = aV1.data();
   const double * aD2 = aV2.data();

   for (int aK=0 ; aK<aNb ; aK++)
        aD1[aK] = aD2[aK];

}


//==============================================================



cControleGC::cControleGC(int aNbIterMax) :
  mNbIterMax (aNbIterMax)
{
}
                                

//==============================================================

template <class TVect,class TMul, class TCond> class cGradConjSolveur
{
       public  :

            typedef typename TVect::tVec tVec;

            bool MpdGC_SolveComplePrecis(tVec  aImB, tVec  aImXSol);
            cGradConjSolveur(TMul &,TCond &,const cControleGC & aCont);

       protected :
       private :
             void MpdGC_AllocAndInitGlob(tVec aImB,tVec aImX);
             void MpdGC_InitOneEtape();
             bool MpdGC_OneStepDesc();

// Fait N iterations pour arriver a la solution exacte, modulo d'eventuel
// pb d'arrondis


         bool MpdGC_OneItereSolveComplet();
         double MpdGC_Residu() ;
         double MpdGC_ResiduConj() ;
         void VerifResidu();

         TMul &   mVMM;
         TCond &  mVPC;
         const cControleGC & mCont;

         int            mN;
         tVec mImBLin;
         tVec mImXSol;
         tVec mImPk;
         tVec mImAPk;
         tVec mImRk;
         double   mSomRkZk;
         double   mAlphaK;
         double   mBetaK;
         tVec mImZk;
         double * mZk;
         tVec mImM;
         double * mM;
         bool     mPreCond;
};



template <class TVect,class TMul, class TCond> 
  cGradConjSolveur<TVect,TMul,TCond>::cGradConjSolveur(TMul & aVMM,TCond & aVPC,const cControleGC & aCont) :
   mVMM        (aVMM),
   mVPC        (aVPC),
   mCont       (aCont),
   mImBLin     (TVect::ParamInit),
   mImXSol     (TVect::ParamInit),
   mImPk       (TVect::ParamInit),
   mImAPk      (TVect::ParamInit),
   mImRk       (TVect::ParamInit),
   mImZk       (TVect::ParamInit),
   mImM        (TVect::ParamInit)
{
}




//==============================================================



template <class TVect,class TMul, class TCond> 
  void cGradConjSolveur<TVect,TMul,TCond>::MpdGC_AllocAndInitGlob
     (
          tVec  aImB,
          tVec  aImX
     )
{
   mN    = TVect::NbInc(aImB);
   mImBLin = aImB;
   mImXSol  = aImX;
   TVect::Resize(mImPk,aImB);
   TVect::Resize(mImAPk,aImB);
   TVect::Resize(mImRk,aImB);
   TVect::Resize(mImZk,aImB);
}

// #define Sgn -1 


//  Voir http://en.wikipedia.org/wiki/Conjugate_gradient_method




template <class TVect,class TMul, class TCond> 
   void cGradConjSolveur<TVect,TMul,TCond>::MpdGC_InitOneEtape()
{
     mVMM.VMMDo(mImXSol,mImAPk);

     TVect::SetDiff(mImRk,mImBLin,mImAPk);
     mVPC.VPCDo(mImRk,mImZk);
     TVect::Affect(mImPk,mImZk);

     mSomRkZk = TVect::Scal(mImRk,mImZk);


}


template <class TVect,class TMul, class TCond>
   double cGradConjSolveur<TVect,TMul,TCond>::MpdGC_Residu() 
{
   ELISE_ASSERT(false,"cGradConjSolveur::MpdGC_Residu");
   double aRes = 0;
/*
   SMFGC_Asub(mXsol,mAPk,mN);
   for (int anI=0 ;anI<mN ; anI++)
   {
       aRes += ElSquare(mAPk[anI]-mBLin[anI]);
   }
*/
   return aRes;
}


#define aEps       1e-15
#define aEpsResidu 1e-2

template <class TVect,class TMul, class TCond>
   bool  cGradConjSolveur<TVect,TMul,TCond>::MpdGC_OneStepDesc()
{
// std::cout << "mSomRkZk " << mSomRkZk << "\n";
/*
   if (ElAbs(mSomRkZk) < aEps)
      return false;
*/
   // mAPk = A *mPk
   mVMM.VMMDo(mImPk,mImAPk);
   // aSomPAP =  t mPk * A * mPk

   double aSomPAP = TVect::Scal(mImPk,mImAPk);
  
   if (ElAbs(aSomPAP) < aEps)
      return false;
   // aAlpha = ||Rk||^2  / aSomPAP
   double aAlpha = mSomRkZk  / aSomPAP;

   // mXsol = mXsol + aAlpha * mPk
   // mRk = mRk - aAlpha * mPk

   TVect::AddAlphaV(mImXSol,aAlpha,mImPk);
   TVect::AddAlphaV(mImRk,-aAlpha,mImAPk);
   
   if (ElAbs( TVect::Scal(mImRk,mImRk)) < aEps)
      return false;



   mVPC.VPCDo(mImRk,mImZk);
   double aSomNextRkZk = TVect::Scal(mImRk,mImZk);
   TVect::MulAlphaAndAdd(mImPk,aSomNextRkZk / mSomRkZk,mImZk);
   mSomRkZk  = aSomNextRkZk;

/*
   std::cout << TVect::Scal(mImRk,mImRk) 
             << " " <<  TVect::Scal(mImRk,mImZk) 
             << " " <<  TVect::Scal(mImZk,mImZk) 
             << "\n";
*/
   return true;
  
}



template <class TVect,class TMul, class TCond>
   bool cGradConjSolveur<TVect,TMul,TCond>::MpdGC_OneItereSolveComplet()
{
    MpdGC_InitOneEtape();
    for (int aK=0 ; aK<mN ; aK++)
    {
// std::cout << "GGCV--K=" << aK << " N=" << mN << "\n";
       if (!  MpdGC_OneStepDesc()) 
          return false;
    }
    return true;
}

template <class TVect,class TMul, class TCond>
    void cGradConjSolveur<TVect,TMul,TCond>::VerifResidu()
{
    double aResisu = MpdGC_Residu() ;
    if (aResisu > aEpsResidu)
    {
        std::cout  << "RESIDU = " << aResisu << "\n";
        ELISE_ASSERT(false,"cGradConjSolveur::VerifResidu");
    }
}

template <class TVect,class TMul, class TCond>
   bool cGradConjSolveur<TVect,TMul,TCond>::MpdGC_SolveComplePrecis
     (
          tVec    aImB,
          tVec    aImX
      )
{
  MpdGC_AllocAndInitGlob(aImB,aImX);
  for (int aK=0 ; aK<mCont.mNbIterMax ; aK++)
  {
       if (! MpdGC_OneItereSolveComplet())
       {
          return false;
       }
  }
  return true;
}


bool GradConjPrecondSolve
     (
            cVectMatMul& aMM,
            cVectPreCond& aVPC,
            Im1D_REAL8  aImB,
            Im1D_REAL8  aImXSol,
            const cControleGC & aCont
     )
{
   cGradConjSolveur<cVecByIm1,cVectMatMul,cVectPreCond>  aGCS(aMM,aVPC,aCont);
   bool aRes =  aGCS.MpdGC_SolveComplePrecis(aImB,aImXSol);

   // getchar();
   return aRes;
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
