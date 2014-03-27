/*eLiSe06/05/99
  
     Copyright (C) 1999 Marc PIERROT DESEILLIGNY

   eLiSe : Elements of a Linux Image Software Environment

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  Author: Marc PIERROT DESEILLIGNY    IGN/MATIS  
Internet: Marc.Pierrot-Deseilligny@ign.fr
   Phone: (33) 01 43 98 81 28
eLiSe06/05/99*/



#include "StdAfx.h"
#include "bench.h"


class cBench_GCF1
{

     public :
         //cBench_GCF1(AllocateurDInconnues & anAlloc,INT aNumF,INT Nb0,INT Nb1);
         cBench_GCF1(cSetEqFormelles & anAlloc,INT aNumF,INT Nb0,INT Nb1); // __NEW
	 void Generate();

	 void ReSet();
	 double Val() const;
	 double Val(const PtsKD &) const;
	 double A() const {return mA;}
	 double B() const {return mB;}
	 PtsKD & Pts() {return mPtsK;}
	 cIncListInterv &  LInt() {return mLInt;}
         Fonc_Num Fonc() {return mF;}

         void VerifDer(
                         INT aD1,cElCompiledFonc & aComp,
                         cElCompiledFonc * aDyn,INT aD2
              );

         void VerifDerSec(
                         INT aI1,INT aI2,cElCompiledFonc * aFcteur,
                         INT aK1,INT aK2
              );

        Fonc_Num             GenF();

	std::string          Name();
     private :



        AllocateurDInconnues & mAlloc;
	INT                  mNumF;
	INT                  mBid0;
        cIncIntervale        mInterv1;
	cIncListInterv       mLInt;

        double               mV0;
        Fonc_Num             mF0;

        double               mV1;
        Fonc_Num             mF1;


       double DerivDiffFinie(INT aD1,double anEps);
       double  ValVois(INT aD1,INT aD2,double anEps,double anEps1);
       double  DerivSecDiffFinie(INT aD1,INT aD2,double anEps,double anEps1);


	INT                  mBid1;
        cIncIntervale        mInterv2;

        double               mV2;
        Fonc_Num             mF2;

        double               mV3;
        Fonc_Num             mF3;

        double               mV4;
        Fonc_Num             mF4;



	double               mA;
	double               mB;
        cVarSpec             mVA;
        cVarSpec             mVB;

        Fonc_Num             mF;
	INT                  mDim;
	PtsKD                mPtsK;
};



double  cBench_GCF1::DerivDiffFinie(INT aD1,double anEps)
{
   PtsKD  aP1 = mPtsK;
   PtsKD  aP2 = mPtsK;
   aP1(aD1) -= anEps;
   aP2(aD1) += anEps;

   return (Val(aP2)-Val(aP1)) / (2*anEps);
   
}

double  cBench_GCF1::ValVois(INT aD1,INT aD2,double anEps1,double anEps2)
{
   PtsKD  aP = mPtsK;
   aP(aD1) += anEps1;
   aP(aD2) += anEps2;
   return Val(aP);
}

double  cBench_GCF1::DerivSecDiffFinie(INT aD1,INT aD2,double anEps1,double anEps2)
{
   return 
           (
                  ValVois(aD1,aD2,anEps1,anEps2)
               +  ValVois(aD1,aD2,-anEps1,-anEps2)
               -  ValVois(aD1,aD2,-anEps1,anEps2)
               -  ValVois(aD1,aD2,anEps1,-anEps2)
           ) / (4*anEps1 * anEps2);
}


void cBench_GCF1::VerifDerSec
     (
                         INT aI1,INT aI2,cElCompiledFonc * aFcteur,
                         INT aK1,INT aK2
     )
{
    double aV2 = aFcteur->DerSec(0,aK1,aK2);
    REAL aDMin = 1e10;

    for (REAL eps1 = 1e-2 ; eps1 > 1e-8; eps1 /= 2)
       for (REAL eps2 = 1e-2 ; eps2 > 1e-8; eps2 /= 2)
       {
          double aV = DerivSecDiffFinie(aI1,aI2,eps1,eps2);
          aDMin = ElMin(aDMin,ElAbs(aV-aV2));
       }
    // cout << "Dif = " <<  aDMin << "\n";
    BENCH_ASSERT(aDMin<epsilon);
}




void cBench_GCF1::VerifDer
     (
          INT aD1,
          cElCompiledFonc & aComp,
          cElCompiledFonc * aDyn,
          INT aD2
     )
{
    REAL aDifMin = 1e10;
    REAL aDer = aComp.Deriv(0,aD2);

    if (aDyn)
    {
        REAL aDD = aDyn->Deriv(0,aD2);
        BENCH_ASSERT(ElAbs(aDD-aDer)<epsilon);
    }

    for (REAL eps = 1e-1 ; eps > 1e-10 ; eps /= 2.0)
    {
        REAL aDif = ElAbs(aDer-DerivDiffFinie(aD1,eps));
        aDifMin = ElMin(aDifMin,aDif);
    }

    BENCH_ASSERT(aDifMin<epsilon);
}



double cBench_GCF1::Val(const PtsKD & aPtsK) const
{
  if (mNumF == 0)
      return  aPtsK(0)*aPtsK(1)*mA + mB;

  if (mNumF == 1)
      return tan(cos(aPtsK(4))/(1+ElSquare(sin(mB*aPtsK(4))))) + mA;

  return  
	    mA*mB*mA*mB + (mA-mB) * (mA+mB) 
	  + mA/mB -mB/mA + (mA-mB)-mB
	  + (mB - (mA-mB))
	  + aPtsK(0)*aPtsK(1) + aPtsK(0)*aPtsK(1) +aPtsK(0)
	  + aPtsK(1) +aPtsK(2)+aPtsK(3) + aPtsK(2)*aPtsK(3) 
	  + aPtsK(2)*aPtsK(3) +cos(aPtsK(4))
          + 1 + 1 + mA * aPtsK(0) 
          + (sin(aPtsK(0)) - sin(aPtsK(1))) + mB* (sin(aPtsK(0)) - sin(aPtsK(1)));
}


double cBench_GCF1::Val() const
{
  return   Val(mPtsK);
}

std::string  CloseInterv 
             (
	          cIncIntervale & anInt,
		  const std::string & anId
             )
{
     anInt.Close(); 
     return anId;
}


Fonc_Num cBench_GCF1::GenF()
{
   if (mNumF == 0)
      return  mF0*mF1*mVA + mVB;

   if (mNumF == 1)
      return tan(cos(mF4)/(1+Square(sin(mVB*mF4)))) + mVA;

   return   mVA*mVB*mVA*mVB + (mVA-mVB) * (mVA+mVB) 
          + mVA/mVB -mVB/mVA + (mVA-mVB)-mVB
          + (mVB - (mVA-mVB))
          + mF0*mF1 + mF0*mF1 +mF0+mF1 
          + mF2+mF3 + mF2*mF3 +mF2*mF3 +cos(mF4)
          + 1 + 1 + mVA * mF0 
          + (sin(mF0) - sin(mF1)) + mVB* (sin(mF0) - sin(mF1));
}

std::string  cBench_GCF1::Name()
{
   if (mNumF == 0)  
      return "cB_GCF0";

   if (mNumF == 1)  
      return "cB_GCF1";

   return "cB_GCF";
}

INT AllocNVar(AllocateurDInconnues & anAlloc,INT aNB)
{
   double aBid;
   for (INT aK=0 ; aK<aNB ; aK++)
        anAlloc.NewF(&aBid);
   return 0;
}

cBench_GCF1::cBench_GCF1
(
   //AllocateurDInconnues & anAlloc,
   cSetEqFormelles & setEquations, // __NEW
   INT aNumF,
   INT Nb0,
   INT Nb1
) :
     //mAlloc   (anAlloc),
     mAlloc   (setEquations.Alloc()), // __NEW
     mNumF    (aNumF),
     //mBid0    (AllocNVar(anAlloc,Nb0)),
     //mInterv1 ( "Interv1",mAlloc) ,
     mBid0    (AllocNVar(setEquations.Alloc(),Nb0)),     // __NEW
     mInterv1 ( false/*isTmp*/,"Interv1",setEquations) , // __NEW
     mF0      ( mAlloc.NewF(&mV0)),
     mF1      ( mAlloc.NewF(&mV1)),
     //mBid1    ((mInterv1.Close(),AllocNVar(anAlloc,Nb1))),
     //mInterv2 ( "Interv2",mAlloc),
     mBid1    ((mInterv1.Close(),AllocNVar(setEquations.Alloc(),Nb1))), // __NEW
     mInterv2 ( false/*isTmp*/,"Interv2",setEquations),                 // __NEW

     mF2      ( mAlloc.NewF(&mV2)),
     mF3      ( mAlloc.NewF(&mV3)),
     mF4      ( mAlloc.NewF(&mV4)),


     mVA      ( 0,"A"),
     mVB      ( 0,"B"),
     mF       (GenF()),
     mDim     (mAlloc.CurInc()),
     mPtsK    ( mDim)
{
     mInterv2.Close();
     mLInt.AddInterv(mInterv1);
     mLInt.AddInterv(mInterv2);
}

void  cBench_GCF1::ReSet()
{
     mA = NRrandC();
     mB = NRrandC();
     for (INT aD=0 ; aD<mDim ; aD++)
	 mPtsK(aD) = NRrandC();
     mVA.Set(mA);
     mVB.Set(mB);
}

void cBench_GCF1::Generate()
{
     cElCompileFN::DoEverything
     (
         "src/bench/",
	 Name(),
	 mF, 
	 mLInt
     );
}

#if (0)
#include "cB_GCF.cpp"
#include "cB_GCF0.cpp"
#include "cB_GCF1.cpp"
template <class tComp> void TplBench_cB_GCF(INT aNumF,tComp & aComp)
{
   //AllocateurDInconnues  anAlloc;
   //cBench_GCF1           anOri(anAlloc,aNumF,0,0);
   //AllocateurDInconnues  anAlloc2;
   //cBench_GCF1           anOri2(anAlloc2,aNumF,0,0);
   cSetEqFormelles equations;
   cBench_GCF1           anOri(equations,aNumF,0,0);
   cSetEqFormelles equations2;
   cBench_GCF1           anOri2(equations2,aNumF,0,0);

   cElCompiledFonc * aFromName = cElCompiledFonc::AllocFromName(anOri2.Name());
   BENCH_ASSERT(aFromName!=0);

   double * pA = aComp.AdrVarLocFromString("A");
   double * pB = aComp.AdrVarLocFromString("B");

   aComp.SetMappingCur(anOri.LInt());

   tComp aCompVal;
   aCompVal.SetMappingCur(anOri.LInt());

   for (INT aK=0 ; aK<10 ; aK++)
   {
      anOri.ReSet();

      // Une fois sur deux, on utilile les SetA, sinon en dyn
      if (NRrandC()>0)
      {
          aComp.SetA(anOri.A());
          aComp.SetB(anOri.B());
      }
      else
      {
          *pA = anOri.A();
          *pB = anOri.B();
      }

      aComp.SetCoordCur(anOri.Pts().AdrX0());
      aComp.SetValDer();

      double aVOri  =  anOri.Val();
      double aVComp =  aComp.Val();

      aCompVal.SetA(anOri.A());
      aCompVal.SetB(anOri.B());
      aCompVal.SetCoordCur(anOri.Pts().AdrX0());
      aCompVal.SetVal();

      double aVV  = aCompVal.Val();
      BENCH_ASSERT(ElAbs(aVOri-aVComp)< epsilon);
      BENCH_ASSERT(ElAbs(aVOri-aVV)< epsilon);
   }

   cElCompiledFonc * aDyn =  cElCompiledFonc::DynamicAlloc(anOri2.LInt(),anOri2.Fonc());


   INT aNb0 = round_ni(100 * NRrandom3());
   INT aNb1 = round_ni(100 * NRrandom3());
   cBench_GCF1  anOri3(anAlloc2,aNumF,aNb0,aNb1);
   cElCompiledFonc * aDVal = cElCompiledFonc::DynamicAlloc(anOri3.LInt(),anOri3.Fonc());


   double * pDA = aDyn->AdrVarLocFromString("A");
   double * pDB = aDyn->AdrVarLocFromString("B");

   double * pFA = aFromName->AdrVarLocFromString("A");
   double * pFB = aFromName->AdrVarLocFromString("B");

   double * p2DA = aDVal->AdrVarLocFromString("A");
   double * p2DB = aDVal->AdrVarLocFromString("B");

   tComp aCompHess;

   for (INT aK=0 ; aK<10 ; aK++)
   {
        anOri.ReSet();

        aComp.SetA(anOri.A());
        aComp.SetB(anOri.B());
        *pDA = anOri.A();
        *pDB = anOri.B();
        *pFA = anOri.A();
        *pFB = anOri.B();

        *p2DA = anOri.A();
        *p2DB = anOri.B();

        INT Ind1 = round_ni(50 +  20 * NRrandC());


        INT delta = round_ni(10 * (1+NRrandom3()));
        if (NRrandC() > 0) 
           delta = - delta;
        INT Ind2 = Ind1 + delta;

        cIncListInterv anInt;
        anInt.AddInterv(cIncIntervale("Interv1",Ind1,Ind1+2));
        anInt.AddInterv(cIncIntervale("Interv2",Ind2,Ind2+3));


        aComp.SetMappingCur(anInt);
        aDyn->SetMappingCur(anInt);
        aDVal->SetMappingCur(anInt);
        aFromName->SetMappingCur(anInt);


        
        double aData[100];
        aData[Ind1+0]  = anOri.Pts()(0);
        aData[Ind1+1]  = anOri.Pts()(1);

        aData[Ind2+0]  = anOri.Pts()(2);
        aData[Ind2+1]  = anOri.Pts()(3);
        aData[Ind2+2]  = anOri.Pts()(4);

        aComp.SetCoordCur(aData);
        aComp.SetValDer();

        aDyn->SetCoordCur(aData);
        aDyn->SetValDer();


        aDVal->SetCoordCur(aData);
        aDVal->SetValDer();

        aFromName->SetCoordCur(aData);
        aFromName->SetValDer();

        double aVOri  =  anOri.Val();
        double aVComp =  aComp.Val();

        double aVDyn  =  aDyn->Val();
        double aVFName  =   aFromName->Val();
        double aVD2  =  aDVal->Val();


        aCompHess.SetMappingCur(anInt);
        aCompHess.SetA(anOri.A());
        aCompHess.SetB(anOri.B());
        aCompHess.SetCoordCur(aData);
        aCompHess.SetValDerHess();
        double aVH =  aCompHess.Val();


        BENCH_ASSERT(ElAbs(aVOri-aVComp)< epsilon);
        BENCH_ASSERT(ElAbs(aVOri-aVDyn)< epsilon);
        BENCH_ASSERT(ElAbs(aVFName-aVDyn)< epsilon);
        BENCH_ASSERT(ElAbs(aVD2-aVDyn)< epsilon);
        BENCH_ASSERT(ElAbs(aVH-aVDyn)< epsilon);


       INT TabInd[5];
       TabInd[0] = Ind1+0;
       TabInd[1] = Ind1+1;
       TabInd[2] = Ind2+0;
       TabInd[3] = Ind2+1;
       TabInd[4] = Ind2+2;

        for (INT aK= 0 ; aK<5 ; aK++)
        {
             anOri.VerifDer(aK,aComp,&aCompHess,TabInd[aK]);
             anOri.VerifDer(aK,aComp,aDVal,TabInd[aK]);
             anOri.VerifDer(aK,aComp,aFromName,TabInd[aK]);
             anOri.VerifDer(aK,aComp,aDyn,TabInd[aK]);
        }
        aDyn->SetValDerHess();

         for (INT aK1= 0 ; aK1<5 ; aK1++)
             for (INT aK2= 0 ; aK2<5 ; aK2++)
             {
                 anOri.VerifDerSec(aK1,aK2,&aCompHess,TabInd[aK1],TabInd[aK2]);
                 anOri.VerifDerSec(aK1,aK2,aDyn,TabInd[aK1],TabInd[aK2]);
             }

   }
   delete aDyn;
   delete aDVal;

}

void Bench_cB_GCF()
{
   cB_GCF  aComp;
   TplBench_cB_GCF(-1,aComp);


   cB_GCF0  aComp0;
   TplBench_cB_GCF(0,aComp0);

   cB_GCF1  aComp1;
   TplBench_cB_GCF(1,aComp1);
}




#else
void Bench_cB_GCF()
{
}
#endif


void bench_GenerationCodeFormelle()
{
   cElCompiledFonc * aNul = cElCompiledFonc::AllocFromName("PP$£%@");
   BENCH_ASSERT(aNul == 0);

/*
   for (INT k=-1 ; k <= 1 ; k++)
   {
      AllocateurDInconnues anAlloc;
      cBench_GCF1 aB1(anAlloc,k,0,0);
      aB1.Generate();
   }
*/

   for (INT k= 0 ; k < 20 ; k++)
   {
      Bench_cB_GCF();
      cout << "AAAAA \n";
   }
}



