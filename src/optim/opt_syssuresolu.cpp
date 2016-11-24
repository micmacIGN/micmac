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


/********************************************************/
/*                                                      */
/*            SystLinSurResolu                          */
/*                                                      */
/********************************************************/

           // cGenSysSurResol


Im1D_REAL8  SystLinSurResolu::V_GSSR_Solve(bool * aResOk)
{
	// TANT QUE POST TRAITE EST true en dur
  // ELISE_ASSERT(mNbContrainte==0,"L1SYS::Do not Handle Contrainte");
  return   L1Solve();
}

INT  SystLinSurResolu::NbVar() const
{
	return mNbVarCur;
}



void SystLinSurResolu::V_GSSR_Reset()
{
   SetNoEquation();
}


void SystLinSurResolu::V_GSSR_AddNewEquation(REAL aPds,REAL * aCoeff,REAL aB)
{
     PushEquation(aCoeff,aB,aPds);
}


           // SystLinSurResolu

SystLinSurResolu::SystLinSurResolu(INT aNbVar,INT aNbEq) : 
   cGenSysSurResol
   (
        false,
        false,
        true,
        false
   ),
   mNbVarCapa      (-1),
   mNbEqCapa       (-1),
   mA              (1,1),
   mB              (1,1),
   mPds            (1,1),
   mBarodA         (1),
   mBarodB         (1),
   mBarodSOL       (1),
   mBarodRESIDU    (1),
   mL2             (aNbVar)
{
   SetSize(aNbVar,aNbEq);
   SetSize(aNbVar,0);
}

void SystLinSurResolu::AssertIndexEqValide(INT IndEq) const
{
    ELISE_ASSERT
    (
        (IndEq>=0) && (IndEq<mNbEqCur),
	"Bad Indexe in SystLinSurResolu"
    );
}

void SystLinSurResolu::AssertIndexVarValide(INT IndVar) const
{
    ELISE_ASSERT
    (
        (IndVar>=0) && (IndVar<mNbVarCur),
	"Bad Indexe in SystLinSurResolu"
    );
}

void SystLinSurResolu::AssertIndexGoodNbVar(INT aNbVar) const
{
    ELISE_ASSERT
    (
        (aNbVar = mNbVarCur),
	"Bad Indexe in SystLinSurResolu"
    );
}

INT  SystLinSurResolu:: NbEq() const
{
   return mNbEqCur;
}




void SystLinSurResolu::SetSize(INT NbVar,INT NbEq)
{
    mNbVarCur  = NbVar;
    mNbEqCur   = NbEq;
    AdjustSizeCapa();
}

void SystLinSurResolu::AdjustSizeCapa()
{
   if ((mNbVarCur<=mNbVarCapa) && (mNbEqCur<=mNbEqCapa))
      return;

   if (mNbVarCur > mNbVarCapa)
      ElSetMax(mNbVarCapa,ElMax(2*mNbVarCapa,mNbVarCur));

   if (mNbEqCur>mNbEqCapa)
      ElSetMax( mNbEqCapa,ElMax(2*mNbEqCapa, mNbEqCur));

   mA = mA.AugmentSizeTo(Pt2di(mNbVarCapa,mNbEqCapa),0.0);
   mDataA   = mA.data();

   mB = mB.AugmentSizeTo(mNbEqCapa,0.0);
   mDataB   = mB.data();

   mPds = mPds.AugmentSizeTo(mNbEqCapa,1.0);
   mDataPds = mPds.data();
}

void SystLinSurResolu::SetNbEquation(INT aNbEq)
{
    SetSize(mNbVarCur,aNbEq);
}

void SystLinSurResolu::SetNoEquation()
{
    SetNbEquation(0);
}


void SystLinSurResolu::PushDifferentialEquation
     (
          Fonc_Num      aFonc,
          const PtsKD & aPts,
          REAL          aPds
     )
{
    INT anIndEq = mNbEqCur;
    SetNbEquation(mNbEqCur+1);

    mDataB[anIndEq] =  - aFonc.ValFonc(aPts);

    for  (INT iVar=0; iVar<mNbVarCur ; iVar++)
    {
          mDataA[anIndEq][iVar] = aFonc.ValDeriv(aPts,iVar); 
          mDataB[anIndEq] += mDataA[anIndEq][iVar] * aPts(iVar);
    }
    mDataPds[anIndEq] = aPds;
}

void SystLinSurResolu::PopEquation ()
{
    ELISE_ASSERT(mNbEqCur!=0,"No Eq in SystLinSurResolu::PopEquation");
    SetNbEquation(mNbEqCur-1);
}

void SystLinSurResolu::PushEquation 
     ( 
            REAL *        aFormLin,
            REAL          aValue,
            REAL          aPds
     )
{
    INT anIndEq = mNbEqCur;
    SetNbEquation(mNbEqCur+1);

    mDataB[anIndEq] =  aValue;
    mDataPds[anIndEq] = aPds;

    for  (INT iVar=0; iVar<mNbVarCur ; iVar++)
    {
       mDataA[anIndEq][iVar] = aFormLin[iVar];
    }
}

void SystLinSurResolu::PushEquation 
     ( 
            Im1D_REAL8    aFormLin,
            REAL          aValue,
            REAL          aPds
     )
{
   AssertIndexGoodNbVar(aFormLin.tx());
   PushEquation (aFormLin.data(),aValue,aPds);
}



void SystLinSurResolu::BarrodaleSetSize()
{
     mBarodA = mBarodA.AugmentSizeTo((mNbVarCur+2)*(mNbEqCur+2));
     mDataBarodA = mBarodA.data();

     mBarodB = mBarodB.AugmentSizeTo(mNbEqCur);
     mDataBarodB = mBarodB.data();

     mBarodSOL = mBarodSOL.AugmentSizeTo(mNbVarCur);
     mDataBarodSOL = mBarodSOL.data();

     mBarodRESIDU = mBarodRESIDU.AugmentSizeTo(mNbEqCur);
     mDataBarodRESIDU = mBarodRESIDU.data();
}


extern void Elise_Craig_Barrodale_Roberts_l1
       (
          INT m,
          INT  n,
          REAL * a,
          REAL *b,
          REAL toler,
          REAL * x,
          REAL * e
       );

Im1D_REAL8  SystLinSurResolu::L1Solve()
{
   BarrodaleSetSize();

   for (INT iEq=0 ; iEq<mNbEqCur ; iEq++)
   {
       for (INT iVar=0 ; iVar<mNbVarCur ; iVar++)
       {
           mDataBarodA[iVar+iEq*(mNbVarCur+2)] = mDataA[iEq][iVar] * mDataPds[iEq];
       }
       mDataBarodB[iEq] = mDataB[iEq] *  mDataPds[iEq];
   }

   Elise_Craig_Barrodale_Roberts_l1
   (
        mNbEqCur,
        mNbVarCur,
        mDataBarodA,
        mDataBarodB,
        1e-8,
        mDataBarodSOL,
        mDataBarodRESIDU
   );

   return mBarodSOL;
}

REAL SystLinSurResolu::Residu(Im1D_REAL8 anIm,INT iEq) const
{
   AssertIndexEqValide(iEq);
   AssertIndexGoodNbVar(anIm.tx());

   return Residu(anIm.data(),iEq);
}

REAL SystLinSurResolu::Residu(const REAL * aPt,INT iEq) const
{
   REAL aRes = 0;
   REAL * aLine = mDataA[iEq];
   for (INT aVar=0 ; aVar<mNbVarCur ; aVar++)
       aRes += aLine[aVar] * aPt[aVar];

   return aRes - mDataB[iEq];
}

REAL SystLinSurResolu::Pds(INT iEq) const
{
   AssertIndexEqValide(iEq);
   return  mDataPds[iEq];
}


REAL SystLinSurResolu::CoefLin(INT iVar,INT iEq) const
{
   AssertIndexEqValide(iEq);
   AssertIndexVarValide(iVar);
   return mDataA[iEq][iVar];
}

REAL  SystLinSurResolu::CoefCste(INT iEq) const
{
    AssertIndexEqValide(iEq);   
    return mDataB[iEq];
}



REAL  SystLinSurResolu::L2SomResiduPond(Im1D_REAL8 aPt) const
{
   AssertIndexGoodNbVar(aPt.tx());
   REAL aRes = 0.0;
   REAL *aDataP = aPt.data();

   for (INT iEq=0 ; iEq<mNbEqCur ; iEq++)
       aRes += mDataPds[iEq] * ElSquare(Residu(aDataP,iEq));

   return aRes;

}


void SystLinSurResolu::L2SetSize()
{
   mL2.SetSize(mNbVarCur);
}




Im1D_REAL8  SystLinSurResolu::L2Solve(bool * aResOk)
{
    L2SetSize();
    mL2.Reset();

    for (INT iEq=0 ; iEq<mNbEqCur ; iEq++)
        mL2.AddEquation(mDataPds[iEq],mDataA[iEq],mDataB[iEq]);

    return mL2.Solve(aResOk);

}

double  cGenSysSurResol::CoeffNorm() const
{
    return 1.0;
}


ElMatrix<tSysCho>  cGenSysSurResol::MatQuad() const
{
    int aNbV=NbVar();
    ElMatrix<tSysCho> aRes (aNbV,aNbV,0.0);

    for (int anX=0 ; anX<aNbV ; anX++)
    {
        for (int anY=0 ; anY<aNbV ; anY++)
        {
             aRes(anX,anY) = GetElemQuad(anX,anY);
        }
    }
    return aRes;
}



void cGenSysSurResol::ShowGSR(int aMode)
{
   for (int aKy=0 ; aKy<NbVar() ; aKy++)
   {
       for (int aKx=0 ; aKx<NbVar() ; aKx++)
       {
          double aV = GetElemQuad(aKx,aKy);
          if (aMode==0)
          {
             std::cout << ((aV==0) ?  "-" : "*");
          }
          else
          {
               ELISE_ASSERT(false,"cGenSysSurResol::ShowGSR");
          }
       }
       std::cout << " " << aKy <<  " " << GetElemLin(aKy) << "\n";
       // printf(" %2d L=%2e\n",aKy,GetElemLin(aKy));
       //std::cout << " : " << aKy << "\n";
   }
}

tSysCho  cGenSysSurResol::GetElemQuad(int i,int j) const
{
   ELISE_ASSERT(false,"cGenSysSurResol::GetElemQuad");
   //return  * ((double *)0);
   // return  * ((double *)&i); // this is never reached but this silent clang's warning about non-volatile pointer 
   return  (tSysCho) 0; // MPD : Will, hopefully, avoid all the warnings 
}
tSysCho  cGenSysSurResol::GetElemLin(int i) const
{
   ELISE_ASSERT(false,"cGenSysSurResol::GetElemLin");
   //return  * ((double *)0);
   // return  * ((double *)&i); // same as before
   return  (tSysCho) 0;   // same as before
}

void  cGenSysSurResol::SetElemQuad(int i,int j,const tSysCho& )
{
   ELISE_ASSERT(false,"cGenSysSurResol::SetElemQuad");
}

void  cGenSysSurResol::SetElemLin(int i,const tSysCho& ) 
{
   ELISE_ASSERT(false,"cGenSysSurResol::SetElemLin");
}


bool  cGenSysSurResol::OptSym() const
{
   return mOptSym;
}

void cGenSysSurResol::VerifGSS(const std::string & aMes) 
{
   ELISE_ASSERT(false,"cGenSysSurResol::VerifGSS");
}

tSysCho cGenSysSurResol::SomQuad() const
{
   tSysCho aRes = 0;
   for (int aKx=0 ; aKx<NbVar(); aKx++)
   {
       for (int aKy=0 ; aKy<NbVar(); aKy++)
       {
           aRes = ElSquare(GetElemQuad(aKx,aKy));
       }
   }
   return aRes;
}

void cGenSysSurResol::LVM_Mul(const tSysCho& aLambda) 
{
   for (int aKx=0 ; aKx<NbVar(); aKx++)
        SetElemQuad(aKx,aKx,(1+aLambda)*GetElemQuad(aKx,aKx));
}

void cGenSysSurResol::LVM_Mul(const tSysCho& aLambda,int aKx) 
{
     SetElemQuad(aKx,aKx,(1+aLambda)*GetElemQuad(aKx,aKx));
}



  //    In-Out / partie lineaire

/*
void cGenSysSurResol::Indexee_EcrireDansMatrColWithLin
     (
           ElMatrix<double> &aMatr,
	   const std::vector<INT> & aVInd
     ) const
{
    ELISE_ASSERT ( aMatr.tx()==1, "matrice non colonne, Indexee_EcrireMatrColWithLin");

    for (int aK=0 ; aK<int(aVInd.size()) ; aK++)
    {
        aMatr(0,aK) = CGetElemLin(aVInd[aK]);
    }
}
*/

void cGenSysSurResol::Indexee_UpdateLinWithMatrCol
     (
           const ElMatrix<tSysCho> &aMatr,
	   const std::vector<INT> & aVInd
     )
{
    ELISE_ASSERT ( aMatr.tx()==1, "matrice non colonne, Indexee_EcrireMatrColWithLin");

    for (int aK=0 ; aK<int(aVInd.size()) ; aK++)
    {
	SetElemLin(aVInd[aK],aMatr(0,aK));
    }
}

void cGenSysSurResol::Indexee_SoustraitMatrColInLin
     (
           const ElMatrix<tSysCho> &aMatr,
	   const std::vector<cSsBloc> & aVBl
     )
{
    ELISE_ASSERT ( aMatr.tx()==1, "matrice non colonne, Indexee_EcrireMatrColWithLin");

    int anXIn=0;
    for (int aKBl=0 ; aKBl <int(aVBl.size()) ; aKBl++)
    {
        int aI0x = aVBl[aKBl].I0AbsSolve();
        int aI1x = aVBl[aKBl].I1AbsSolve();
        for (int anXOut = aI0x; anXOut < aI1x ; anXOut++)
        {
	     SetElemLin(anXOut,GetElemLin(anXOut)-aMatr(0,anXIn));
             // GetElemLin(anXOut) -= aMatr(0,anXIn);
             anXIn++;
        }
    }
/*
    for (int aK=0 ; aK<int(aVInd.size()) ; aK++)
    {
	GetElemLin(aVInd[aK]) -= aMatr(0,aK);
    }
*/
}

void cGenSysSurResol::Indexee_LinSet0(const std::vector<INT> & aVInd)
{
    for (int aK=0 ; aK<int(aVInd.size()) ; aK++)
    {
	SetElemLin(aVInd[aK],0);
    }
}

double  cGenSysSurResol::ResiduOfSol(const double *)
{
   ELISE_ASSERT(false,"cGenSysSurResol::ResiduOfSol");
   return 0;
}



  //    In-Out / partie quadratique

void cGenSysSurResol::Indexee_EcrireDansMatrWithQuad
     ( 
            ElMatrix<tSysCho> &aMatr,
            const std::vector<INT> & aVIndx,
            const std::vector<INT> & aVIndy
     )  const
{
    for (int aKx=0 ; aKx <int(aVIndx.size()) ; aKx++)
        for (int aKy=0 ; aKy <int(aVIndy.size()) ; aKy++)
	    aMatr(aKx,aKy) = GetElemQuad(aVIndx[aKx],aVIndy[aKy]);
}

void cGenSysSurResol::Indexee_EcrireDansMatrWithQuad
     ( 
            ElMatrix<tSysCho> &aMatr,
            const std::vector<cSsBloc> &  aVx,
            const std::vector<cSsBloc> &  aVy
     )  const
{
    int xout = 0;
    for (int aKx=0 ; aKx <int(aVx.size()) ; aKx++)
    {
        int aI0x = aVx[aKx].I0AbsSolve();
        int aI1x = aVx[aKx].I1AbsSolve();
        for (int xin = aI0x; xin < aI1x ; xin++)
        {
            int yout=0;
            for (int aKy=0 ; aKy <int(aVy.size()) ; aKy++)
            {
                int aI0y = aVy[aKy].I0AbsSolve();
                int aI1y = aVy[aKy].I1AbsSolve();
                for (int yin = aI0y; yin < aI1y ; yin++)
                {
                    aMatr(xout,yout) = GetElemQuad(xin,yin);
                    yout++;
                }
            }
            xout ++;
        }
    }
}

void cGenSysSurResol::Indexee_QuadSet0
     ( 
            const std::vector<cSsBloc> &  aVx,
            const std::vector<cSsBloc> &  aVy
     )  
{
    for (int aKx=0 ; aKx <int(aVx.size()) ; aKx++)
    {
        int aI0x = aVx[aKx].I0AbsSolve();
        int aI1x = aVx[aKx].I1AbsSolve();
        for (int xin = aI0x; xin < aI1x ; xin++)
        {
            for (int aKy=0 ; aKy <int(aVy.size()) ; aKy++)
            {
                int aI0y = aVy[aKy].I0AbsSolve();
                int aI1y = aVy[aKy].I1AbsSolve();
                for (int yin = aI0y; yin < aI1y ; yin++)
                {
                    SetElemQuad(xin,yin,0);
                }
            }
        }
    }
}


void cGenSysSurResol::Indexee_EcrireDansMatrColWithLin
     (
           ElMatrix<tSysCho> &aMatr,
	   const std::vector<cSsBloc> & aVx
     ) const
{
    ELISE_ASSERT ( aMatr.tx()==1, "matrice non colonne, Indexee_EcrireMatrColWithLin");

    int xout = 0;
    for (int aKx=0 ; aKx <int(aVx.size()) ; aKx++)
    {
        int aI0x = aVx[aKx].I0AbsSolve();
        int aI1x = aVx[aKx].I1AbsSolve();
        for (int xin = aI0x; xin < aI1x ; xin++)
        {
            aMatr(0,xout) = GetElemLin(xin);
            xout ++;
        }
    }
}

void cGenSysSurResol::Indexee_LinSet0(const std::vector<cSsBloc> & aVx) 
{
    for (int aKx=0 ; aKx <int(aVx.size()) ; aKx++)
    {
        int aI0x = aVx[aKx].I0AbsSolve();
        int aI1x = aVx[aKx].I1AbsSolve();
        for (int xin = aI0x; xin < aI1x ; xin++)
        {
            SetElemLin(xin,0);
        }
    }
}








void cGenSysSurResol::Indexee_UpdateQuadWithMatr
     ( 
            const ElMatrix<tSysCho> &aMatr,
            const std::vector<INT> & aVIndx,
            const std::vector<INT> & aVIndy
     )
{
    for (int aKx=0 ; aKx <int(aVIndx.size()) ; aKx++)
        for (int aKy=0 ; aKy <int(aVIndy.size()) ; aKy++)
	    SetElemQuad(aVIndx[aKx],aVIndy[aKy],aMatr(aKx,aKy));
}

void cGenSysSurResol::Indexee_SoustraitMatrInQuad
     ( 
            const ElMatrix<tSysCho> &aMatr,
            const std::vector<INT> & aVIndx,
            const std::vector<INT> & aVIndy
     )
{
    for (int aKx=0 ; aKx <int(aVIndx.size()) ; aKx++)
        for (int aKy=0 ; aKy <int(aVIndy.size()) ; aKy++)
        {
	    SetElemQuad(aVIndx[aKx],aVIndy[aKy], GetElemQuad(aVIndx[aKx],aVIndy[aKy])-aMatr(aKx,aKy));
        }
}

void cGenSysSurResol::SoutraitProduc3x3
     ( 
                          bool                   Sym,
                          ElMatrix<tSysCho> &aM1,
                          ElMatrix<tSysCho> &aM2,
                          const std::vector<cSsBloc> * aYVSB

     )
{
// std::cout << "AV 333x3333\n";

    tSysCho ** aData1 = aM1.data();
    tSysCho ** aData2 = aM2.data();

    tSysCho * aL2A = aData2[0];
    tSysCho * aL2B = aData2[1];
    tSysCho * aL2C = aData2[2];


/*
   // std::cout << "AAAAAAAAAAAAAAAAaakkkkkkkkkkkkkk\n";
   if ((!aYVSB) ||   (aYVSB->size()!=1))
   {
       std::cout << "aYVSB: " << aYVSB << "\n";
       std::cout << " size: " << aYVSB->size() << "\n";
       ELISE_ASSERT(aYVSB && (aYVSB->size()==1),"cGenSysSurResol::SoutraitProduc3x3");
   }
*/
       ELISE_ASSERT(aYVSB,"cGenSysSurResol::SoutraitProduc3x3");

       int aY=0;
       for (int aKBy=0 ; aKBy <int(aYVSB->size()) ; aKBy++)
       {
           const cSsBloc &  aBlY = (*aYVSB)[aKBy];
           for (int aKy=aBlY.I0AbsSolve() ; aKy <aBlY.I1AbsSolve() ; aKy++)
           {
              int aDebX = Sym ? aY : 0;
              tSysCho A1 = aData1[aY][0];
              tSysCho B1 = aData1[aY][1];
              tSysCho C1 = aData1[aY][2];

              int aX=0;

              for (int aKBx=0 ; aKBx <int(aYVSB->size()) ; aKBx++)
              {
                  const cSsBloc &  aBlX = (*aYVSB)[aKBx];
                  for (int aKx=aBlX.I0AbsSolve() ; aKx <aBlX.I1AbsSolve() ; aKx++)
                  {
                      if (aX>=aDebX)
                      {
	                 SetElemQuad(aKx,aKy,GetElemQuad(aKx,aKy) -(A1 * aL2A[aX] + B1 * aL2B[aX] + C1 *aL2C[aX]));
                      }
                      aX++;
                  }
              }
/*
              for (int aKx=aDebX ; aKx <int(aVIndy.size()) ; aKx++)
              {
	          GetElemQuad(aVIndy[aKx],aVIndy[aKy]) -= A1 * aL2A[aKx] + B1 * aL2B[aKx] + C1 *aL2C[aKx];
              }
*/
              aY++;
           }
       }

/*
    for (int aKy=0 ; aKy <int(aVIndy.size()) ; aKy++)
    {
        int aDebX = Sym ? aKy : 0;
        double A1 = aData1[aKy][0];
        double B1 = aData1[aKy][1];
        double C1 = aData1[aKy][2];

        for (int aKx=aDebX ; aKx <int(aVIndy.size()) ; aKx++)
        {
	    GetElemQuad(aVIndy[aKx],aVIndy[aKy]) -= A1 * aL2A[aKx] + B1 * aL2B[aKx] + C1 *aL2C[aKx];
        }
    }
*/


// std::cout << "AV 333x3333\n";
}
/*
*/





void cGenSysSurResol::Indexee_QuadSet0
     ( 
            const std::vector<INT> & aVIndx,
            const std::vector<INT> & aVIndy
     )
{
    for (int aKx=0 ; aKx <int(aVIndx.size()) ; aKx++)
        for (int aKy=0 ; aKy <int(aVIndy.size()) ; aKy++)
	    SetElemQuad(aVIndx[aKx],aVIndy[aKy],0);
}

void cGenSysSurResol::Indexee_QuadSetId
     ( 
            const std::vector<INT> & aVIndxy
     )
{
    for (int aKx=0 ; aKx <int(aVIndxy.size()) ; aKx++)
        for (int aKy=0 ; aKy <int(aVIndxy.size()) ; aKy++)
	    SetElemQuad(aVIndxy[aKx],aVIndxy[aKy],((aKx==aKy) ? 1 : 0 )) ;
}



/**************************************************/
/*                                                */
/*        AllocateurDInconnues                    */
/*                                                */
/**************************************************/


    // ==============   cStateAllocI ===================

cStateAllocI::cStateAllocI(const AllocateurDInconnues& anAlloc) :
   mAlloc (anAlloc)
{
   for (int aK=0 ; aK<mAlloc.CurInc() ; aK++)
      mVals.push_back(mAlloc.GetVar(aK));
    
}

void cStateAllocI::ShowDiff(const cStateAllocI & aS)  const
{
   aS.mAlloc.AssertUsable(*this);
   mAlloc. AssertUsable(aS);


   double aSom=0;

   for (int aK=0 ; aK<int(mVals.size()) ; aK++)
   {
      double aDif =  ElAbs(mVals[aK]-aS.mVals[aK]);
      aSom += aDif;
   }
   std::cout << "SomDIF =" << aSom << "\n";
}
    // ===============  AllocateurDInconnues  ======

void AllocateurDInconnues::AssertUsable(const cStateAllocI & aS) const
{
    ELISE_ASSERT(&aS.mAlloc==this,"Diff Allocc size in AllocateurDInconnues::AssertUsable");
    ELISE_ASSERT(int(aS.mVals.size())==mIdInc,"Diff size in AllocateurDInconnues::AssertUsable");
}

void AllocateurDInconnues::RestoreState(const cStateAllocI & aS)
{
    AssertUsable(aS);
    SetVars(&(aS.mVals[0]));
}

AllocateurDInconnues::AllocateurDInconnues() :
   mIdInc(0)
{
}

INT AllocateurDInconnues::CurInc() const
{
   return mIdInc;
}

void AllocateurDInconnues::PushVar(REAL * anAdr)
{
    mAdrVar.push_back(anAdr);
    mValsVar.push_back(*anAdr);
}

Fonc_Num AllocateurDInconnues::NewF(const std::string & aNameBloc,const std::string & aName,REAL * anAdr,bool HasAlwaysInitialValue)
{
    return  kth_coord(NewInc(aNameBloc,aName,anAdr),HasAlwaysInitialValue,*anAdr);
}

INT AllocateurDInconnues::NewInc(const std::string & aNameBloc,const std::string & aName,REAL * anAdr)
{
    mVNamesInc.push_back(aName);
    mVNamesBlocInc.push_back(aNameBloc);
    PushVar(anAdr);
    return  mIdInc++;
}

Pt3d<Fonc_Num>   AllocateurDInconnues::NewPt3
                 (const std::string & aNameBloc,REAL * adrX,REAL * adrY,REAL * adrZ,bool HasAlwaysInitialValue)
{
    PushVar(adrX);
    mVNamesBlocInc.push_back(aNameBloc);
    mVNamesInc.push_back("x");

    PushVar(adrY);
    mVNamesBlocInc.push_back(aNameBloc);
    mVNamesInc.push_back("y");

    PushVar(adrZ);
    mVNamesBlocInc.push_back(aNameBloc);
    mVNamesInc.push_back("z");


    mIdInc += 3;
    return Pt3d<Fonc_Num>
           (
               kth_coord(mIdInc-3,HasAlwaysInitialValue,*adrX),
               kth_coord(mIdInc-2,HasAlwaysInitialValue,*adrY),
               kth_coord(mIdInc-1,HasAlwaysInitialValue,*adrZ)
           );
}

std::vector<Fonc_Num>   AllocateurDInconnues::NewVectInc(const std::string & aNameBloc,const std::vector<std::string> & aVNameInc,std::vector<double> & aV)
{
    std::vector<Fonc_Num> aRes;
    for (int aK=0; aK<int(aV.size()) ; aK++)
    {
       aRes.push_back(NewF(aNameBloc,aVNameInc[aK],&(aV[aK])));
    }
    return aRes;
}


Pt3d<Fonc_Num>   AllocateurDInconnues::NewPt3(const std::string & aNameBloc,Pt3dr & aP,bool HasAlwaysInitialValue)
{
      return   NewPt3(aNameBloc,&aP.x,&aP.y,&aP.z,HasAlwaysInitialValue);
}


Pt2d<Fonc_Num>   AllocateurDInconnues::NewPt2(const std::string & aNameBloc,REAL * adrX,REAL * adrY,bool HasAlwaysInitialValue, const std::string & aNameX,const std::string & aNameY)
{
    PushVar(adrX);
    mVNamesBlocInc.push_back(aNameBloc);
    mVNamesInc.push_back(aNameX);

    PushVar(adrY);
    mVNamesBlocInc.push_back(aNameBloc);
    mVNamesInc.push_back(aNameY);

    mIdInc += 2;
    return Pt2d<Fonc_Num>
           (
               kth_coord(mIdInc-2,HasAlwaysInitialValue,*adrX),
               kth_coord(mIdInc-1,HasAlwaysInitialValue,*adrY)
           );
}

Pt2d<Fonc_Num>   AllocateurDInconnues::NewPt2(const std::string & aNameBloc,REAL * adrX,REAL * adrY,bool HasAlwaysInitialValue)
{
     return NewPt2(aNameBloc,adrX,adrY,HasAlwaysInitialValue,"x","y");
}

Pt2d<Fonc_Num>   AllocateurDInconnues::NewPt2(const std::string & aNameBloc,Pt2dr & aP,bool HasAlwaysInitialValue)
{
      return   NewPt2(aNameBloc,&aP.x,&aP.y,HasAlwaysInitialValue,"x","y");
}

TplElRotation3D<Fonc_Num> AllocateurDInconnues::NewRot
                        (
                            const std::string & aNameBloc,
			    REAL * adrTrX, REAL * adrTrY, REAL * adrTrZ,
			    REAL * adrTeta01,
			    REAL * adrTeta02,
			    REAL * adrTeta12
			 )
{
    mIdInc += 6;
    PushVar(adrTrX);
    mVNamesBlocInc.push_back(aNameBloc);
    mVNamesInc.push_back("Cx");

    PushVar(adrTrY);
    mVNamesBlocInc.push_back(aNameBloc);
    mVNamesInc.push_back("Cy");

    PushVar(adrTrZ);
    mVNamesBlocInc.push_back(aNameBloc);
    mVNamesInc.push_back("Cz");

    PushVar(adrTeta01);
    mVNamesBlocInc.push_back(aNameBloc);
    mVNamesInc.push_back("Tz");

    PushVar(adrTeta02);
    mVNamesBlocInc.push_back(aNameBloc);
    mVNamesInc.push_back("Ty");

    PushVar(adrTeta12);
    mVNamesBlocInc.push_back(aNameBloc);
    mVNamesInc.push_back("Tx");


    return  TplElRotation3D<Fonc_Num>
            (
                Pt3d<Fonc_Num>
                (
                    kth_coord(mIdInc-6),
                    kth_coord(mIdInc-5),
                    kth_coord(mIdInc-4)
                ),
                kth_coord(mIdInc-3),
                kth_coord(mIdInc-2),
                kth_coord(mIdInc-1)
            );
}
 
PtsKD AllocateurDInconnues::PInits()
{
   PtsKD aP((int) mAdrVar.size());
   for (INT aK=0; aK<INT(mAdrVar.size()) ; aK++)
       aP(aK) = *mAdrVar[aK];

   return aP;
}

void  AllocateurDInconnues::SetVars(const REAL * aSol)
{
   for (INT aK=0; aK<INT(mAdrVar.size()) ; aK++)
   {
       mValsVar[aK] = *mAdrVar[aK] = aSol[aK];
   }
}
void  AllocateurDInconnues::Reinit(INT aK)
{
	SetVar(*mAdrVar[aK],aK);
}

REAL * AllocateurDInconnues::ValsVar()
{
  return & (mValsVar[0]);
}

double AllocateurDInconnues::GetVar(INT aK) const
{
   return mValsVar[aK];
}
double * AllocateurDInconnues::GetAdrVar(INT aK)
{
   return mAdrVar[aK];
}


void AllocateurDInconnues::SetVar(double aVal,INT aK)
{
       mValsVar[aK] = *mAdrVar[aK] = aVal;
}

void AllocateurDInconnues::SetVarPt(Pt2dr aP,INT aK)
{
    SetVar(aP.x,aK  );
    SetVar(aP.y,aK+1);
}

const std::string &  AllocateurDInconnues::NamesInc (int aK) const { return mVNamesInc.at(aK); }
const std::string &  AllocateurDInconnues::NamesBlocInc (int aK) const { return mVNamesBlocInc.at(aK); }





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
