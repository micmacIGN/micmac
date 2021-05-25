#include "include/MMVII_all.h"


namespace MMVII
{



template <class Type> void TestFilterExp(cPt2di aP0,cPt2di aP1,const Type & aV0)
{
   // Check on "Dirac" image that we get the expected exponential response
   cPt2di aPMil = (aP0+aP1) / 2;

   cIm2D<Type> aImX(aP0,aP1,nullptr,eModeInitImage::eMIA_Null);
   cDataIm2D<Type> & aDImX = aImX.DIm();
   aDImX.InitDirac(aPMil,aV0);

   cIm2D<Type> aImXY(aP0,aP1,nullptr,eModeInitImage::eMIA_Null);
   cDataIm2D<Type> & aDImXY = aImXY.DIm();
   aDImXY.InitDirac(aPMil,aV0);

 
   cRect2 aRect2(aP0+cPt2di(1,2),aP1-cPt2di(4,5));
   // cRect2 aRect2(cPt2di(0,0),aSz);


   double aFx = 0.6;
   double aFy = 0.7;
   // double aEps = 
   // Type aEps = std::numeric_limits<Type>::epsilon();
   Type aEps =  tNumTrait<Type>::Eps();

   ExponentialFilter(false,aDImX ,1,aRect2,aFx,0.0);
   ExponentialFilter(false,aDImXY,1,aRect2,aFx,aFy);
   Type aDMaxX  = 0;
   Type aDMaxXY = 0;
   for (const auto & aP : aDImX)
   {
        bool IsIn = aRect2.Inside(aP);
        cPt2di aPDif = aP-aPMil;

        bool OnLineY = (aPDif.y()==0);
        tREAL16 aVExpX = pow(aFx,std::abs(aPDif.x()));
        tREAL16 aVExpY = pow(aFy,std::abs(aPDif.y()));

        Type aV1X = aDImX.GetV(aP);
        Type aV2X = (IsIn && OnLineY) ? (aV0 * aVExpX)   : (Type) 0;
        aDMaxX = std::max(aDMaxX,std::abs(aV1X-aV2X));

        Type aV1XY = aDImXY.GetV(aP);
        Type aV2XY =   IsIn ? (aV0 * aVExpX * aVExpY) :  (Type) 0;
        aDMaxXY = std::max(aDMaxXY,std::abs(aV1XY-aV2XY));
   }
   aDMaxX /=tREAL16(aV0) ;
   aDMaxXY /=tREAL16(aV0) ;
   // StdOut() << " HHhhhh "<<  aDMaxX << " " << aDMaxXY << "\n";
   MMVII_INTERNAL_ASSERT_bench(aDMaxX<1e-2*sqrt(aEps)  ,"Test Filtr Exp")
   MMVII_INTERNAL_ASSERT_bench(aDMaxXY<1e-2*sqrt(aEps) ,"Test Filtr Exp")

   // Test iteration , check iteration by hand and integrated give same results
   {
       aDImX.InitRandom(0,aV0);
       aDImX.DupIn(aDImXY);
       int aNbIter=3;
       for (int aKIter=0 ; aKIter<aNbIter ; aKIter++)
           ExponentialFilter(false,aDImX ,1,aRect2,aFx,aFy);
       ExponentialFilter(false,aDImXY,aNbIter,aRect2,aFx,aFy);
   
       double aD =  aDImXY.LInfDist(aDImX) / double(sqrt(aV0));

       MMVII_INTERNAL_ASSERT_bench(aD< (aEps*1e5) ,"Test Filtr Exp")
   }

   // StdOut() << "ONE BENC FEXP " <<  aD / pow(aEps,0.25) << " " << aD/aEps << " " << aEps << "\n";
   
   // Test normalization, const => const
   {
       aDImX.InitCste(aV0); // Cste
       aDImXY.InitCste(aV0);
       ExponentialFilter(true,aDImXY,3,aRect2,aFx,aFy); // Cste filtered

       double aD =  aDImXY.LInfDist(aDImX) / aV0;
       MMVII_INTERNAL_ASSERT_bench(aD< (aEps*1e2) ,"Test Norm ")
   }
   //  Test, far from centre norm is just multiplicative
   {
       double aFx = 0.15; // give low value to decrease border influeence
       double aFy = 0.2;
       aDImX.InitDirac(aPMil,aV0);
       aDImXY.InitDirac(aPMil,aV0);
       ExponentialFilter(false,aDImX,2,aRect2,aFx,aFy); // Cste filtered
       ExponentialFilter(true,aDImXY,2,aRect2,aFx,aFy); // Cste filtered

       double aRatio = aDImX.GetV(aPMil)/ double( aDImXY.GetV(aPMil));
       for (const auto & aP : cRect2(aPMil-cPt2di(1,1),aPMil+cPt2di(2,2)))
       {
           double aR2 = aDImX.GetV(aP)/ double( aDImXY.GetV(aP));
           MMVII_INTERNAL_ASSERT_bench(std::abs(1-aR2 / aRatio)<1e-2 ,"Test Norm ")
       }
  
    }
}


/*
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
*/



void BenchFilterLinear(cParamExeBench & aParam)
{
   if (! aParam.NewBench("ImageFilterLinear")) return;
   TestFilterExp<tREAL4>(cPt2di(-2,-3),cPt2di(20,30),1.0);
   TestFilterExp<tREAL8>(cPt2di(3,2),cPt2di(20,30),1.0);
   TestFilterExp<tREAL16>(cPt2di(-4,-5),cPt2di(20,30),1.0);
   TestFilterExp<int>(cPt2di(0,0),cPt2di(20,30),1000000);
   aParam.EndBench();
}



};
