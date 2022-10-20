
#include "IndexBinaire.h"
#include "MMVII_Tpl_Images.h"

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

template <class Type> void cAppli_ComputeParamIndexBinaire::AddOneEqParamLin(double aPds,const cDataIm1D<Type> & aCdg,double aVal,int aNb)
{
   if (! aNb) return;
   // aCdg  .  aVect - mThr = 0
   cDenseVect<tREAL8> aV(1+mNbValByP);
   for (int aK=0 ; aK<mNbValByP ; aK++)
      aV(aK) = aCdg.GetV(aK);
   aV(mNbValByP) = -1;
   mLSQOpt.AddObservation(aPds,aV,aVal);

}

void cAppli_ComputeParamIndexBinaire::AddVecOneEqParamLin(double aPds,cVecInvRad & aCdg,double aVal)
{
    // const cDataIm1D<tU_INT1> & aDIm = aCdg.mVec.DIm();
    // AddOneEqParamLin(aPds,aDIm,aVal,1);
    AddOneEqParamLin(aPds,aCdg.mVec.DIm(),aVal,1);
}



void cAppli_ComputeParamIndexBinaire::OptimiseScoreAPriori(tPtVBool  aVBOld,const tVPtVIR & aVIR)  
{
   bool Cont = true;
   int aCpt = 0;
   tPtVBool  aV000 = aVBOld;
   while (Cont)
   {
      aCpt++;
      std::vector<double>  aVScore;
      std::vector<double>  aVSAbs;
      std::vector<double>  aVBadMin;
      std::vector<double>  aVBadMax;

      for (const auto & aV : aVIR)
      {
          double aSc = aVBOld->FB().RCalc(*aV);
          aVScore.push_back(aSc);
          aVSAbs.push_back(std::abs(aSc));
      }
      for (int aK=0 ; aK<int(aVIR.size()) ; aK+=2)
      {
         if ((aVScore[aK]>0) != (aVScore[aK+1]>0))
         {
             aVBadMin.push_back(std::min(aVSAbs[aK],aVSAbs[aK+1]));
             aVBadMax.push_back(std::max(aVSAbs[aK],aVSAbs[aK+1]));
         }
      }

      double aMedAbs = ConstMediane(aVSAbs);
      double aMedSigned = ConstMediane(aVScore);
      double aMedBadMin = ConstMediane(aVBadMin);
      double aMedBadMax = ConstMediane(aVBadMax);
      StdOut() << "MedAbs " << aMedAbs  
               << " MedS " << aMedSigned 
               << " MedBad " << aMedBadMin   <<  " " << aMedBadMax
               << " Sc " << aVBOld->Score() << "\n";

      double aSigma = aMedAbs / 2;

   //  Fill least square

      mLSQOpt.Reset();
      double aPdsClose000 = 10;
      double aPdsCloseCur = 1000;
      double aPdsAver0 = 1e6;

      double aSomPds = 0.0;
      std::vector<double> aVPds;

      for (int aK=0 ; aK<int(aVIR.size()) ; aK+=2)
      {
         aVPds.push_back(exp(-Square(aVSAbs[aK]/aSigma)));
         aVPds.push_back(exp(-Square(aVSAbs[aK+1]/aSigma)));
         if ((aVScore[aK]>0) != (aVScore[aK+1]>0))
         {
             int aKToChange =  (aVSAbs[aK] < aVSAbs[aK+1]) ? aK : (aK+1);
             int aKCompl = (2*aK+1) - aKToChange;
             double aPdsCh = aVPds[aKToChange];
             double aPdsCompl = aVPds[aKCompl];
             double aPds = (aPdsCh - aPdsCompl);
             double aVal =  aVScore[aKCompl];
             double aSign = (aVal >0) ? 1 : -1;
             aVal += aSign * aMedBadMax;
             aSomPds += aPds*aVal;
             AddVecOneEqParamLin(aPds,*(aVIR[aKToChange]),aVal);
         }
         else
         {
             for (int aK2=aK; aK2<aK+2 ; aK2++)
             {
                 // Perte de temps, gain pas evident
                 // AddVecOneEqParamLin(aVPds[aK]/10,*(aVIR[aK]),aVScore[aK]);
             }
         }
      }
      AddEqProxCur(aPdsCloseCur*aSomPds,aVBOld);
      AddEqProxCur(aPdsClose000*aSomPds,aV000);
      AddOneEqParamLin(aPdsAver0*aSomPds,mStat2.Moy().DIm(),0.0,1);
      cDenseVect<double>  aSol = mLSQOpt.Solve();

      tPtVBool aVBNew = VecBoolFromSol(aSol,aVBOld->Index());
      double aDelta = aVBOld->Score() - aVBNew->Score() ;

      static std::vector<double> aVDelta ;
      aVDelta.push_back(aDelta);

      StdOut()  << " L200 " << aV000->FB().Vect().L2Dist(aVBNew->FB().Vect())
                << " L2Cur " << aVBOld->FB().Vect().L2Dist(aVBNew->FB().Vect())
                << " SC priori " <<  aDelta 
                << " Med-Delta " << ConstMediane(aVDelta) << "\n";

      if (aDelta < 0)
      {
         aVBOld = aVBNew;
         mVVBool.at(aVBOld->Index()) = aVBOld;
      }
      Cont = (aDelta < 0) && (aCpt<10);
   }
   StdOut() << "CPT=" << aCpt << "\n";
}

void cAppli_ComputeParamIndexBinaire::AddEqProxCur(double aPdsCloseCur,tPtVBool aVB0)
{
   const cDenseVect<double>& aVK0 = aVB0->FB().Vect() ;
   for (int aKP=0 ; aKP<mNbValByP ; aKP++)
   {
       mLSQOpt.AddObsFixVar(aPdsCloseCur,aKP,aVK0(aKP));
   }
}

tPtVBool cAppli_ComputeParamIndexBinaire::VecBoolFromSol(const  cDenseVect<double> & aSol,int aIndex)
{
   cDenseVect<double>  aSF(mNbValByP);
   for (int aKP=0 ; aKP<mNbValByP ; aKP++)
   {
       aSF(aKP) = aSol(aKP);
   }
   
   return tPtVBool (new cVecBool(aIndex,false,new cIB_LinearFoncBool(*this,aSF,aSol(mNbValByP)),mVIR));
}


void cAppli_ComputeParamIndexBinaire::TestNewParamLinear(const std::vector<tPtVBool>& aOldVB,int aK0Vec)
{
   double aPdsCloseCur = 1e-1;
   double aPdsEq       = 1;
   mLSQOpt.Reset();

   AddEqProxCur(aPdsCloseCur,aOldVB[aK0Vec]);
/*
   tPtVBool aVB0 = aOldVB[aK0Vec];
   const cDenseVect<double>& aVK0 = aVB0->FB().Vect() ;
   for (int aKP=0 ; aKP<mNbValByP ; aKP++)
   {
       mLSQOpt.AddObsFixVar(aPdsCloseCur,aKP,aVK0(aKP));
   }
*/
   for (int aKV=0 ; aKV<int(mVVBool.size()) ; aKV++)
   {
      cVecBool & aVB = *(mVVBool[aKV].get());
      double aPds = aPdsEq *  RandUnif_0_1();
      AddOneEqParamLin(aPds,aVB.Cdg0().DIm(),0.0,aVB.Nb0());
      AddOneEqParamLin(aPds,aVB.Cdg1().DIm(),0.0,aVB.Nb1());
   }
   cDenseVect<double>  aSol = mLSQOpt.Solve();

 
   tPtVBool aVB = VecBoolFromSol(aSol,aOldVB[aK0Vec]->Index());

   std::vector<tPtVBool> aNewV = aOldVB;
   aNewV[aK0Vec] = aVB;
   int aK;
   double aSc = ScoreSol(aK,aNewV);

   if (aSc > mBestSc)
   {
      StdOut() << "D2222 " << aVB->FB().Vect().L2Dist(aOldVB[aK0Vec]->FB().Vect()) << mBestSc << " " << aSc << "\n";
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
    mCdg0 (aVIR.at(0)->mVec.DIm().Sz()),
    mNb0  (0),
    mCdg1 (aVIR.at(0)->mVec.DIm().Sz()),
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
       double aMed = ConstMediane(aVScore);

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
   CalcCdgScoreAPriori(aVIR);

   MMVII_INTERNAL_ASSERT_medium((mVB.size()%2)==0,"cVecBool odd size");
}



void cVecBool::CalcCdgScoreAPriori(const tVPtVIR & aVIR)  
{
   mCdg0.DIm().InitNull();
   mCdg1.DIm().InitNull();
   
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

