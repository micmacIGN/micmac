#include "include/MMVII_all.h"
#include "include/SymbDer/SymbolicDerivatives.h"
#include "include/SymbDer/SymbDer_MACRO.h"

using namespace NS_SymbolicDerivative;


namespace MMVII
{

cDescOneFuncDist::cDescOneFuncDist(eTypeFuncDist aType,const cPt2di aDegXY) :
   mType    (aType),
   mDegMon  (-1,-1),
   mNum     (-1)
{
   int aNum = aDegXY.x();
   if (mType==eTypeFuncDist::eRad)
   {
       mName = "K" + ToStr(aNum);  //  K1 K2 K3 ...
       mDegTot = 1 + 2 * (aNum);   //  X (X^2+Y^2) ^N
       mNum = aNum;
   }
   else if ((mType==eTypeFuncDist::eDecX) ||  (mType==eTypeFuncDist::eDecY))
   {
       int aDec = (mType==eTypeFuncDist::eDecX)?-1:0;
       // p1,p2  as usual, and by generalization p3 p4 p5 ...
       mName = "p" + ToStr(2*aNum+aDec);
       mDegTot =  2 * (aNum);  // Derivates of X (X^2+Y^2) ^N
       mNum = aNum;
   }
   else
   {
      mDegMon = aDegXY;
      mDegTot = mDegMon.x() + mDegMon.y();
      if ((mType==eTypeFuncDist::eMonX) && (mDegTot==1))
      {
          mName = ( mDegMon.x() == 1) ? "b1" : "b2";  // Usual convention
      }
      else
      {
          mName =  std::string((mType==eTypeFuncDist::eMonX) ? "x" : "y")
                 + "_" + ToStr(mDegMon.x())
                 + "_" + ToStr(mDegMon.y()) ;
      }
   }
}

double cDescOneFuncDist::MajNormJacOfRho(double aRho) const
{
   switch(mType)
   {
       case eTypeFuncDist::eRad :
           return mDegTot * powI(aRho,mDegTot-1);  // Rho ^degT
       case eTypeFuncDist::eDecX :
       case eTypeFuncDist::eDecY :
            return mDegTot*(mDegTot+1) * powI(aRho,mDegTot-1);  // d/drho (Rho ^(degT+1))

       case eTypeFuncDist::eMonX :
       case eTypeFuncDist::eMonY :
            return Norm2(mDegMon) * powI(aRho,mDegTot-1); // 
       default :
          ;
   }

   MMVII_INTERNAL_ERROR("Bad num in cDescOneFuncDist::MajNormJacOfRho");
   return 0.0;
}

double  MajNormJacOfRho
        (
             const double & aRho,
             const std::vector<cDescOneFuncDist> & aVDesc,
             const std::vector<double> & aVCoef
        )
{
    MMVII_INTERNAL_ASSERT_medium(aVDesc.size()==aVCoef.size(),"Sz in MajNormJacOfRho");
    double aRes =0.0;
    
    for (size_t aK=0 ; aK<aVDesc.size() ; aK++)
        aRes += aVCoef[aK] * aVDesc[aK].MajNormJacOfRho(aRho);

   return aRes;
}


/*
double  MajNormJacOfRho
        (
             const cPtxd<double> & aCenter,
             const cTplBox<Type,2> & aBox,
             const std::vector<cDescOneFuncDist> & aVDesc,
             const std::vector<double> & aVCoef
        )
{
    double aRes =0.0;
    typename cTplBox<double,2>::tCorner  aTabC;
}
*/




};//  namespace MMVII

