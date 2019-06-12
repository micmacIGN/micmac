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
/*              cIB_FoncBool            */
/* ==================================== */

cIB_FoncBool::cIB_FoncBool(cAppli_ComputeParamIndexBinaire & anAppli) :
   mAppli (anAppli)
{
}


double cIB_FoncBool::RCalc(const cVecInvRad & aVIR) const
{
   return Calc(aVIR) ? 0.5 : -0.5;
}

cIB_FoncBool::~cIB_FoncBool()
{
}

/* ==================================== */
/*         cIB_LinearFoncBool           */
/* ==================================== */

cIB_LinearFoncBool::cIB_LinearFoncBool(cAppli_ComputeParamIndexBinaire & anAppli,int aK,double aTreshold) :
   cIB_FoncBool (anAppli),
   mK           (aK),
   mThresh      (aTreshold)
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

    return mAppli.Stat2().KthNormalizedCoord(mK,aTmp) - mThresh;
}

/* ==================================== */
/*         cIB_LinearFoncBool           */
/* ==================================== */


cVecBool::cVecBool(cIB_FoncBool * aFB,const tVPtVIR & aVIR)  :
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
   aProp1 /= mVB.size();
   aPropEq /= (mVB.size() / 2.0);

   double aPropThEq =  Square(aProp1) + Square(1-aProp1);
   mScore = aPropEq/aPropThEq;

    StdOut()  << "Eq=" << aPropEq << " Sc=" << aPropEq/aPropThEq << " P1=" << aProp1 << "\n";
}

/* ==================================== */
/*          cStatDifBits                */
/* ==================================== */

int NbbBitDif(const std::vector<const cVecBool*> & aVVB,const cPt2di & aPair)
{
   int aNbBitDif = 0;
   for (const auto & aVB : aVVB)
       if (aVB->KBit(aPair.x()) != aVB->KBit(aPair.y()))
          aNbBitDif++;
   return aNbBitDif;
}

cStatDifBits::cStatDifBits(const std::vector<cPt2di> & aVPair,const std::vector<const cVecBool*> & aVVB) :
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

void  cStatDifBits::Show(const cStatDifBits & aStatFalse,int aK1,int aK2) const
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
   }
}


};

