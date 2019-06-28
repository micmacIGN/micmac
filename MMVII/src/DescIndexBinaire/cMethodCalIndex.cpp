#include "include/MMVII_all.h"
#include "IndexBinaire.h"
#include "include/MMVII_Tpl_Images.h"

// #include "include/MMVII_2Include_Serial_Tpl.h"
// #include<map>

/** \file cMethodCalIndex.cpp
    \brief Contains class for comuting one bit of info

*/


namespace MMVII
{


/* ==================================== */
/*         cIB_LinearFoncBool           */
/* ==================================== */

cIB_LinearFoncBool::cIB_LinearFoncBool
(
     cAppli_ComputeParamIndexBinaire & anAppli, 
     const cDenseVect<double>&   aVect, 
     double                aThresh 
)  :
   mAppli    (anAppli),
   mVect     (aVect),
   mThresh   (aThresh)
{
}

cIB_LinearFoncBool::cIB_LinearFoncBool
(
     cAppli_ComputeParamIndexBinaire & anAppli, 
     const cDenseVect<double>&   aVect
)  :
    cIB_LinearFoncBool
    (
        anAppli,
        aVect,
        aVect.DotProduct(anAppli.Stat2().Moy())
    )
{
}
   

cIB_LinearFoncBool::cIB_LinearFoncBool(cAppli_ComputeParamIndexBinaire & anAppli,int aK) :
    cIB_LinearFoncBool
    (
        anAppli,
        anAppli.Eigen().EigenVectors().ReadCol(aK)
    )
/*
   mAppli    (anAppli),
   mK        (aK),
   mVect     (mAppli.Eigen().EigenVectors().ReadCol(aK)),
   mThresh   (mVect.DotProduct(mAppli.Stat2().Moy()))
*/
{
}

bool cIB_LinearFoncBool::Calc(const cVecInvRad & aVIR) const
{
    return RCalc(aVIR) > 0;
}

double cIB_LinearFoncBool::RCalc(const cVecInvRad & aVIR) const 
{
    cDenseVect<tREAL8> & aTmp = mAppli.TmpVect();
    CopyIn(aTmp.DIm(),aVIR.mVec.DIm());

    return mVect.DotProduct(aTmp) - mThresh;
}

const cDenseVect<double>&  cIB_LinearFoncBool::Vect() const 
{
   return mVect;
}



/* ==================================== */
/*         cIB_LinearFoncBool           */
/* ==================================== */

void cAppli_ComputeParamIndexBinaire::AddOneEqParamLin(double aPds,const cDenseVect<tREAL4> & aCdg,int aNb)
{
   if (! aNb) return;
   // aCdg  .  aVect - mThr = 0
   cDenseVect<tREAL8> aV(1+mNbValByP);
   for (int aK=0 ; aK<mNbValByP ; aK++)
      aV(aK) = aCdg(aK);
   aV(mNbValByP) = -1;
   mLSQOpt.AddObservation(aPds,aV,0.0);

}



void cAppli_ComputeParamIndexBinaire::TestNewParamLinear(const std::vector<tPtVBool>& aOldVB,int aK0Vec)
{
   double aPdsCloseCur = 1e-1;
   double aPdsEq       = 1;
   mLSQOpt.Reset();

   tPtVBool aVB0 = aOldVB[aK0Vec];
   const cDenseVect<double>& aVK0 = aVB0->FB().Vect() ;
   for (int aKP=0 ; aKP<mNbValByP ; aKP++)
   {
       mLSQOpt.AddObsFixVar(aPdsCloseCur,aKP,aVK0(aKP));
   }
   for (int aKV=0 ; aKV<int(mVVBool.size()) ; aKV++)
   {
      cVecBool & aVB = *(mVVBool[aKV].get());
      double aPds = aPdsEq *  RandUnif_0_1();
      AddOneEqParamLin(aPds,aVB.Cdg0(),aVB.Nb0());
      AddOneEqParamLin(aPds,aVB.Cdg1(),aVB.Nb1());
   }
   cDenseVect<double>  aSol = mLSQOpt.Solve();
   cDenseVect<double>  aSF(mNbValByP);
   for (int aKP=0 ; aKP<mNbValByP ; aKP++)
   {
       aSF(aKP) = aSol(aKP);
   }
   
   tPtVBool aVB(new cVecBool(aVB0->Index(),false,new cIB_LinearFoncBool(*this,aSF,aSol(mNbValByP)),mVIR));

   std::vector<tPtVBool> aNewV = aOldVB;
   aNewV[aK0Vec] = aVB;
   int aK;
   double aSc = ScoreSol(aK,aNewV);

   if (aSc > mBestSc)
   {
      StdOut() << "D2222 " << aSF.L2Dist(aVK0) << mBestSc << " " << aSc << "\n";
      ChangeVB(aSc,aVB,aK0Vec);
   }
   
/*
*/

   // mVVBool = aVSave;
}


const cDenseVect<tREAL4>&  cVecBool::Cdg0() const {return mCdg0;}
int                        cVecBool::Nb0()  const {return mNb0;}
const cDenseVect<tREAL4>&  cVecBool::Cdg1() const {return mCdg1;}
int                        cVecBool::Nb1()  const {return mNb1;}
int                        cVecBool::Index()  const {return mIndex;}



cVecBool::cVecBool(int Index,bool Med,cIB_LinearFoncBool * aFB,const tVPtVIR & aVIR)  :
    mIndex (Index),
    mFB (aFB),
    mCdg0 (aVIR.at(0)->mVec.DIm().Sz(),eModeInitImage::eMIA_Rand),
    mNb0  (0),
    mCdg1 (aVIR.at(0)->mVec.DIm().Sz(),eModeInitImage::eMIA_Rand),
    mNb1  (0)
{
   mVB.reserve(aVIR.size());
   std::vector<double>  aVScore;
   aVScore.reserve(aVIR.size());

   for (const auto & aV : aVIR)
   {
       aVScore.push_back(mFB->RCalc(*aV));
   }
   if (1)
   {
       std::vector<double> aVMed = aVScore; // Copy pour ne pas toucher a l'original
       double aMed = Mediane(aVMed);

       // StdOut() << "Meeeddd  " << aMed << "\n";
       if (!Med)
       {
          aMed=0;
       }

       for (auto & aV : aVScore)
       {
           aV -= aMed;
       }
   }
   for (const auto & aV : aVScore)
   {
       mVB.push_back(aV>0);
   }

       //mVB.push_back(aC>0);

   MMVII_INTERNAL_ASSERT_medium((mVB.size()%2)==0,"cVecBool odd size");

   double aProp1  = 0;
   double aPropEq = 0;

   for (int aK=0 ; aK<int(mVB.size()) ; aK++)
   {
       const auto & aVec =  aVIR.at(aK)->mVec.DIm();
       if (mVB.at(aK))
       {
          AddIn(mCdg1.DIm(),aVec);
          mNb1++;
          aProp1++;
       }
       else
       {
          AddIn(mCdg0.DIm(),aVec);
          mNb0++;
       }
       if ((aK%2==0) && (mVB.at(aK)== mVB.at(aK+1)))
       {
           aPropEq++;
       }

   }
   if (mNb0) mCdg0.DIm() *=  1/double(mNb0);
   if (mNb1) mCdg1.DIm() *=  1/double(mNb1);

   aProp1 /= mVB.size();
   aPropEq /= (mVB.size() / 2.0);

   double aPropThEq =  Square(aProp1) + Square(1-aProp1);
   mScore = aPropEq/aPropThEq;

    // StdOut()  << "Eq=" << aPropEq << " Sc=" << aPropEq/aPropThEq << " P1=" << aProp1 << "\n";
}

cIB_LinearFoncBool&   cVecBool::FB()
{
   return *(mFB.get());
}


/* ==================================== */
/*          cStatDifBits                */
/* ==================================== */

int NbbBitDif(const std::vector<tPtVBool> & aVVB,const cPt2di & aPair)
{
   int aNbBitDif = 0;
   for (const auto & aVB : aVVB)
       if (aVB->KBit(aPair.x()) != aVB->KBit(aPair.y()))
          aNbBitDif++;
   return aNbBitDif;
}

cStatDifBits::cStatDifBits(const std::vector<cPt2di> & aVPair,const std::vector<tPtVBool> & aVVB) :
   mHNbBitDif (aVVB.size()+1,0),
   mStatR     (mHNbBitDif.size(),0.0),
   mStatRCum  (mHNbBitDif.size(),0.0)
{
    for (const auto & aPair : aVPair)
    {
        mHNbBitDif.at(NbbBitDif(aVVB,aPair)) ++;
/*
        int aNbBitDif = 0;
        for (const auto & aVB : aVVB)
            if (aVB->KBit(aP.x()) != aVB->KBit(aP.y()))
                aNbBitDif++;
        mHNbBitDif.at(aNbBitDif) ++;
*/
    }
    for (int aK=0 ; aK<=int(aVVB.size()) ; aK++)
    {
        mStatR.at(aK) = mHNbBitDif[aK] / double(aVPair.size());
    }
    mStatRCum.at(0) = mStatR.at(0);
    for (int aK=1 ; aK<=int(aVVB.size()) ; aK++)
    {
        mStatRCum.at(aK) = mStatR.at(aK) + mStatRCum.at(aK-1);
    }
}

double cStatDifBits::Score(const cStatDifBits & aStatFalse,double aPdsFalse,int &aKMax) const
{
   double aRes = -1e10;
   for (int aK=0 ; aK<int(mStatRCum.size()) ; aK++)
   {
       double aSc = mStatRCum.at(aK) - aPdsFalse * aStatFalse.mStatRCum.at(aK) ;
       if (aSc > aRes)
       {
           aRes = aSc;
           aKMax = aK;
       }
   }
   return aRes;
}

void  cStatDifBits::Show(const cStatDifBits & aStatFalse,int aK1,int aK2,double aVMax) const
{
   aK1 = std::max(aK1,0);
   aK2 = std::min(aK2,int(mStatRCum.size()-1));
   for (int aK=aK1 ; aK<=aK2 ; aK++)
   {
      StdOut() << "Cum[" << aK << "]=" 
                << " " << mStatRCum.at(aK) 
                << " " << aStatFalse.mStatRCum.at(aK) 
                << " " << mStatRCum.at(aK)/ std::max(1e-8,aStatFalse.mStatRCum.at(aK))
                << "\n";
      if (mStatRCum.at(aK)>aVMax)
         return;
   }
}


};

