#include "include/MMVII_all.h"

namespace MMVII
{

// return the variance of  exponential distribution of parameter "a" ( i.e proportiona to  "a^|x|")
double Sigma2FromFactExp(double a)
{
   return (2*a) / Square(1-a);
}

// return the value of "a" such that  exponential distribution of parameter "a" has variance S2
// i.e. this the "inverse" function of Sigma2FromFactExp

double FactExpFromSigma2(double aS2)
{
    return (aS2+1 - sqrt(Square(aS2+1)-Square(aS2))  ) / aS2 ;
}

/* *************************************** */
/*                                         */
/*        cComputeStdDev                   */
/*                                         */
/* *************************************** */

template <class Type> cComputeStdDev<Type>::cComputeStdDev() :
   mSomW   (0.0),
   mSomWV  (0.0),
   mSomWV2 (0.0)
{
}

template <class Type> void cComputeStdDev<Type>::Add(const Type & aW,const Type & aV)
{
    mSomW   += aW;
    mSomWV  += aW *aV;
    mSomWV2 += aW * Square(aV);
}


template <class Type> void cComputeStdDev<Type>::SelfNormalize(const Type&  Epsilon)
{
    MMVII_ASSERT_INVERTIBLE_VALUE(mSomW);
     
    mSomWV  /= mSomW;
    mSomWV2  /= mSomW;
    mSomWV2 -= Square(mSomWV);
    mStdDev = std::sqrt(std::max(Epsilon,mSomWV2));
}

template <class Type> Type cComputeStdDev<Type>::NormalizedVal(const Type & aVal)  const
{
    MMVII_ASSERT_INVERTIBLE_VALUE(mStdDev);
    return (aVal-mSomWV) / mStdDev;
}

template <class Type> cComputeStdDev<Type>  cComputeStdDev<Type>::Normalize(const Type&  Epsilon) const
{
     cComputeStdDev<Type> aRes = *this;
     aRes.SelfNormalize(Epsilon);
     return aRes;
}


/* ============================================= */
/*      cMatIner2Var<Type>                       */
/* ============================================= */

template <class Type> cMatIner2Var<Type>::cMatIner2Var() :
   mS0  (0.0),
   mS1  (0.0),
   mS11 (0.0),
   mS2  (0.0),
   mS12 (0.0),
   mS22 (0.0)
{
}
template <class Type> void cMatIner2Var<Type>::Add(const double & aPds,const Type & aV1,const Type & aV2)
{
    mS0  += aPds;
    mS1  += aPds * aV1;
    mS11 += aPds * aV1 * aV1 ;
    mS2  += aPds * aV2;
    mS12 += aPds * aV1 * aV2 ;
    mS22 += aPds * aV2 * aV2 ;
}

template <class Type> void cMatIner2Var<Type>::Add(const Type & aV1,const Type & aV2)
{
    mS0  += 1.0;
    mS1  += aV1;
    mS11 += aV1 * aV1 ;
    mS2  += aV2;
    mS12 += aV1 * aV2 ;
    mS22 += aV2 * aV2 ;
}

template <class Type> 
       void  cMatIner2Var<Type>::Add(const cMatIner2Var& aM2)
{
    mS0  +=  aM2.mS0;
    mS1  +=  aM2.mS1;
    mS11 +=  aM2.mS11;
    mS2  +=  aM2.mS2;
    mS12 +=  aM2.mS12;
    mS22 +=  aM2.mS22;
}

template <class Type> 
       void  cMatIner2Var<Type>::Add(const cMatIner2Var& aM2,const Type & aPds) 
{
    mS0  += aPds * aM2.mS0;
    mS1  += aPds * aM2.mS1;
    mS11 += aPds * aM2.mS11;
    mS2  += aPds * aM2.mS2;
    mS12 += aPds * aM2.mS12;
    mS22 += aPds * aM2.mS22;
}


template <class Type> void cMatIner2Var<Type>::Normalize()
{
     MMVII_ASSERT_INVERTIBLE_VALUE(mS0);

     mS1 /= mS0;
     mS2 /= mS0;
     mS11 /= mS0;
     mS12 /= mS0;
     mS22 /= mS0;
     mS11 -= Square(mS1);
     mS12 -= mS1 * mS2;
     mS22 -= mS2 * mS2;
}

template <class Type> Type cMatIner2Var<Type>::CorrelNotC(const Type & aEps) const
{
    Type  aS11 = mS11/mS0;
    Type  aS12 = mS12/mS0;
    Type  aS22 = mS22/mS0;

    Type  aSqDenominator = std::max(aEps,aS11*aS22);
    MMVII_ASSERT_STRICT_POS_VALUE(aSqDenominator);

    return aS12  / std::sqrt(aSqDenominator);
}

template <class Type> Type cMatIner2Var<Type>::Correl(const Type & aEps) const
{
   cMatIner2Var<Type> aDup(*this);
   aDup.Normalize();

   Type aSqDenominator = std::max(aEps,aDup.mS11*aDup.mS22);
   MMVII_ASSERT_STRICT_POS_VALUE(aSqDenominator);

   return aDup.mS12  / std::sqrt(aSqDenominator);
}

template <class Type> inline Type StdDev(const Type & aS0,const Type & aS1,const Type & aS11)
{
   MMVII_ASSERT_INVERTIBLE_VALUE(aS0);

   Type aEc2 = aS11/aS0 - Square(aS1/aS0);
   return std::sqrt(std::max(Type(0.0),aEc2));
}

template <class Type> Type cMatIner2Var<Type>::StdDev1() const
{
   return StdDev(mS0,mS1,mS11);
}
template <class Type> Type cMatIner2Var<Type>::StdDev2() const
{
   return StdDev(mS0,mS2,mS22);
}

template <class Type> cMatIner2Var<double> StatFromImageDist(const cDataIm2D<Type> & aIm)
{
    cMatIner2Var<double> aRes;
    for (const auto & aP : aIm)
    {
         aRes.Add(aIm.GetV(aP),aP.x(),aP.y());
    }
    aRes.Normalize();
    return aRes;
}

/* *********************************************** */
/*                                                 */
/*          cWeightAv<Type>                        */
/*                                                 */
/* *********************************************** */

template <class Type> cWeightAv<Type>::cWeightAv() :
   mSW(0),
   mSVW(0)
{
}

template <class Type> void cWeightAv<Type>::Add(const Type & aWeight,const Type & aVal)
{
   mSW += aWeight;
   mSVW += aVal * aWeight;
}

template <class Type> Type cWeightAv<Type>::Average() const
{
    MMVII_ASSERT_INVERTIBLE_VALUE(mSW);
    return mSVW / mSW;
}




/* *********************************************** */
/*                                                 */
/*        cUB_ComputeStdDev<Dim>                   */
/*                                                 */
/* *********************************************** */

template  <const int Dim> cUB_ComputeStdDev<Dim>::cUB_ComputeStdDev() :
    mSomW   (0.0),
    mSomWW  (0.0)
{
    for (int aD=0 ; aD<Dim ; aD++)
    {
        mSomWV[aD] = 0.0;
        mSomWVV[aD] = 0.0;
    }
}

template  <const int Dim> void cUB_ComputeStdDev<Dim>::Add(const  double *  aVal,const double & aPds)
{
    mSomW += aPds;
    mSomWW += Square(aPds);
    for (int aD=0 ; aD<Dim ; aD++)
    {
        mSomWV[aD] += aPds * aVal[aD];
        mSomWVV[aD] += aPds * Square(aVal[aD]);
    }
}
template  <const int Dim>  double cUB_ComputeStdDev<Dim>::DeBiasFactor() const
{
    MMVII_INTERNAL_ASSERT_strong(mSomW>0,"No value in DeBiasFactor");
    return   1 - mSomWW/Square(mSomW);
}

template  <const int Dim> bool    cUB_ComputeStdDev<Dim>::OkForUnBiasedVar() const
{
   return (mSomW>0)  && (DeBiasFactor()!=0);
}

template  <const int Dim> const double *   cUB_ComputeStdDev<Dim>::ComputeUnBiasedVar()
{
    /* At least, this formula is correct :
         - when all weight are equal => 1-1/N
         - when all but one weight are 0 => 0 
         - and finally if all are equal or 0
    */
    double aDebias = DeBiasFactor();
    MMVII_INTERNAL_ASSERT_strong(aDebias>0,"No var can be computed in ComputeVar");

    for (int aD=0 ; aD<Dim ; aD++)
    {
        double aAver = mSomWV[aD] / mSomW;
        double aVar  = mSomWVV[aD] / mSomW;
        aVar = aVar - Square(aAver);
        mVar[aD] = std::max(0.0,aVar) /aDebias;
    }

   return mVar;
}

template  <const int Dim> const double *   cUB_ComputeStdDev<Dim>::ComputeBiasedVar()
{
    for (int aD=0 ; aD<Dim ; aD++)
    {
        double aAver = mSomWV[aD] / mSomW;
        double aVar  = mSomWVV[aD] / mSomW;
        aVar = aVar - Square(aAver);
        mBVar[aD] = std::max(0.0,aVar);
    }

   return mBVar;
}





template class cUB_ComputeStdDev<1>;

void BenchUnbiasedStdDev()
{
    for (int aNbTest = 0 ; aNbTest<1000 ; aNbTest++)
    {
         int aNbVar = 1 + RandUnif_N(3);
         int aNbTir = aNbVar;
         int aNbComb = pow(aNbVar,aNbTir);
         std::vector<double> aVecVals = VRandUnif_0_1(aNbVar);
         std::vector<double> aVecWeight = VRandUnif_0_1(aNbVar);
// aVecWeight =  std::vector<double> (aNbVar,1.0);

         // compute the average of all empirical variance
         double aMoyVar=0;
         for (int aFlag=0 ; aFlag < aNbComb ; aFlag++) // Explore all combinaison
         {
             cUB_ComputeStdDev<1> aUBS;
             for (int aVar=0 ; aVar < aNbVar ; aVar++) // All Variable of this realization
             {
                 int aNumVar = (aFlag / round_ni(pow(aNbVar,aVar))) % aNbVar; // "Majic" formula to exdtrac p-adic decomp
                 aUBS.Add(&(aVecVals.at(aNumVar)),aVecWeight.at(aNumVar));
             }
             aMoyVar += aUBS.ComputeBiasedVar()[0];
         }
         aMoyVar /= aNbComb;

         cUB_ComputeStdDev<1> aUBS;
StdOut() << "WWW=" ;
         for (int aK=0 ; aK<aNbVar ; aK++)
         {
StdOut() << " " << aVecWeight.at(aK) ;
            aUBS.Add(&(aVecVals.at(aK)),aVecWeight.at(aK));
         }
StdOut() << "\n" ;
         StdOut() << "MOYVAR " << aMoyVar <<  " " <<  aUBS.DeBiasFactor() *  aUBS.ComputeBiasedVar()[0] 
                  << " DBF=" <<  aUBS.DeBiasFactor() << "\n";
getchar();
    }
}

/* *********************************************** */
/*                                                 */
/*            Test Exp Filtre                      */
/*                                                 */
/* *********************************************** */

// Test Sigma2FromFactExp
template <class Type> void TestVarFilterExp(cPt2di aP0,cPt2di aP1,const Type & aV0,double aFx,double aFy,int aNbIt)
{
   cPt2di aPMil = (aP0+aP1) / 2;
   cRect2 aRect2(aP0,aP1);

   cIm2D<Type> aIm(aP0,aP1,nullptr,eModeInitImage::eMIA_Null);
   cDataIm2D<Type> & aDIm = aIm.DIm();

   // 1- Make 1 iteration of expon filter on a Direc, check variance and aver
   aDIm.InitDirac(aPMil,aV0);

   ExponentialFilter(true,aDIm,aNbIt,aRect2,aFx,aFy);
   cMatIner2Var<double>  aMat = StatFromImageDist(aDIm);

   MMVII_INTERNAL_ASSERT_bench(std::abs(aMat.S1() - aPMil.x()) < 1e-5  ,"Average Exp")
   MMVII_INTERNAL_ASSERT_bench(std::abs(aMat.S2() - aPMil.y()) < 1e-5  ,"Average Exp")

   MMVII_INTERNAL_ASSERT_bench(std::abs(aMat.S11()-Sigma2FromFactExp(aFx) * aNbIt)<1e-5,"Var Exp");
   MMVII_INTERNAL_ASSERT_bench(std::abs(aMat.S22()-Sigma2FromFactExp(aFy) * aNbIt)<1e-5,"Var Exp");
}



template <class Type> void TestVarFilterExp(cPt2di aSz,double aStdDev,int aNbIter,double aEps)
{
   cIm2D<Type> aIm(cPt2di(0,0),aSz,nullptr,eModeInitImage::eMIA_Null);
   cDataIm2D<Type> & aDIm = aIm.DIm();
   cPt2di aPMil = aSz / 2;
   double aV0 = 2.0;

   // 1- Make 1 iteration of expon filter on a Direc, check variance and aver
   aDIm.InitDirac(aPMil,aV0);

   ExpFilterOfStdDev(aDIm,aNbIter,aStdDev);
   cMatIner2Var<double>  aMat = StatFromImageDist(aDIm);

   MMVII_INTERNAL_ASSERT_bench(std::abs(aMat.S11()-Square(aStdDev))<aEps,"Std dev");
   MMVII_INTERNAL_ASSERT_bench(std::abs(aMat.S22()-Square(aStdDev))<aEps,"Std dev");
}

void BenchStat(cParamExeBench & aParam)
{
   if (! aParam.NewBench("ImageStatFilter")) return;

   TestVarFilterExp<double>(cPt2di(-2,2),cPt2di(400,375),2.0,0.6,0.67,1);
   TestVarFilterExp<double>(cPt2di(-2,2),cPt2di(400,375),2.0,0.6,0.67,3);

   // Test Sigma2FromFactExp
   for (int aK=1 ; aK<100 ; aK++)
   {
        double aS2 = aK / 5.0;
        double aF = FactExpFromSigma2(aS2);
        {
           // Check both formula are inverse of each others
           double aCheckS2 = Sigma2FromFactExp(aF);
           MMVII_INTERNAL_ASSERT_bench(std::abs( aS2 - aCheckS2 ) < 1e-5  ,"Sigma2FromFactExp")
         
        }
        {
           // Check formula Sigma2FromFactExp on exponential filters
           TestVarFilterExp<double>(cPt2di(0,0),cPt2di(4000,3),2.0,aF,0,1);
        }
   }

   TestVarFilterExp<double>(cPt2di(300,300),2.0,2,1e-6);
   TestVarFilterExp<float>(cPt2di(300,300),5.0,3,1e-3);

   aParam.EndBench();
}

/* *********************************************** */
/*                                                 */
/*                INSTANTIATION                    */
/*                                                 */
/* *********************************************** */

#define INSTANTIATE_MAT_INER(TYPE)\
template class cMatIner2Var<TYPE>;\
template  class cComputeStdDev<TYPE>;\
template class cWeightAv<TYPE>;\
template  cMatIner2Var<double> StatFromImageDist(const cDataIm2D<TYPE> & aIm);


INSTANTIATE_MAT_INER(tREAL4)
INSTANTIATE_MAT_INER(tREAL8)
INSTANTIATE_MAT_INER(tREAL16)

};

