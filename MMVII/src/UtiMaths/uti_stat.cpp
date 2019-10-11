#include "include/MMVII_all.h"
#include <boost/math/special_functions/fpclassify.hpp>

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


/* *********************************************** */
/*                                                 */
/*        cComputeStdDev<Dim>                     */
/*                                                 */
/* *********************************************** */

#if (0)
///  Class to compute non biased variance from a statisic 

/** Class to compute non biased variance from a statisic
    This generalise the standard formula
            EstimVar = N/(N-1) EmpirVar
    To the case where there is a weighting.

    It can be uses with N variable to factorize the computation on Weight
 */

template <const int Dim> class cComputeStdDev
{
    public :
        typedef  double tTab[Dim];

        cComputeStdDev();

        void Add(const  double * aVal,const double & aPds);
        const double  *  ComputeUnBiasedVar() ;
        const double  *  ComputeBiasedVar() ;
        double  DeBiasFactor() const;

    private :
        double    mSomW; ///< Sum of Weight
        double    mSomWW; ///< Sum of Weight ^2
        tTab      mSomWV;  ///< Weighted som of vals
        tTab      mSomWVV; ///< Weighted som of vals ^2
        tTab      mVar;   ///< Buffer to compute the unbiased variance
        tTab      mBVar;   ///< Buffer to compute the empirical variance
};
#endif

template  <const int Dim> cComputeStdDev<Dim>::cComputeStdDev() :
    mSomW   (0.0),
    mSomWW  (0.0)
{
    for (int aD=0 ; aD<Dim ; aD++)
    {
        mSomWV[aD] = 0.0;
        mSomWVV[aD] = 0.0;
    }
}

template  <const int Dim> void cComputeStdDev<Dim>::Add(const  double *  aVal,const double & aPds)
{
    mSomW += aPds;
    mSomWW += Square(aPds);
    for (int aD=0 ; aD<Dim ; aD++)
    {
        mSomWV[aD] += aPds * aVal[aD];
        mSomWVV[aD] += aPds * Square(aVal[aD]);
    }
}
template  <const int Dim>  double cComputeStdDev<Dim>::DeBiasFactor() const
{
    MMVII_INTERNAL_ASSERT_strong(mSomW>0,"No value in DeBiasFactor");
    return   1 - mSomWW/Square(mSomW);
}

template  <const int Dim> bool    cComputeStdDev<Dim>::OkForUnBiasedVar() const
{
   return (mSomW>0)  && (DeBiasFactor()!=0);
}

template  <const int Dim> const double *   cComputeStdDev<Dim>::ComputeUnBiasedVar()
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

template  <const int Dim> const double *   cComputeStdDev<Dim>::ComputeBiasedVar()
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





template class cComputeStdDev<1>;

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
             cComputeStdDev<1> aUBS;
             for (int aVar=0 ; aVar < aNbVar ; aVar++) // All Variable of this realization
             {
                 int aNumVar = (aFlag / round_ni(pow(aNbVar,aVar))) % aNbVar; // "Majic" formula to exdtrac p-adic decomp
                 aUBS.Add(&(aVecVals.at(aNumVar)),aVecWeight.at(aNumVar));
             }
             aMoyVar += aUBS.ComputeBiasedVar()[0];
         }
         aMoyVar /= aNbComb;

         cComputeStdDev<1> aUBS;
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

void BenchStat()
{
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
}



};

