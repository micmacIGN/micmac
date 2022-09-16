#include "include/MMVII_all.h"
#include "include/SymbDer/SymbolicDerivatives.h"
#include "include/SymbDer/SymbDer_MACRO.h"
#include "include/MMVII_TplSymbImage.h"

using namespace NS_SymbolicDerivative;


namespace MMVII
{
std::vector<std::string> FormalBilinIm2D_NameObs(const std::string & aPrefix)
{
   return std::vector<std::string> 
          {
              "PtX0_" + aPrefix,
              "PtY0_" + aPrefix,
              "Im00_" + aPrefix,
              "Im10_" + aPrefix,
              "Im01_" + aPrefix,
              "Im11_" + aPrefix
          };
}

/*
void FFF()
{
   std::vector<double> aV;

   FormalBilinIm2D_Formula(aV,0,1.0,1.0);

   std::vector<cFormula<tREAL8> > aVF;
   FormalBilinIm2D_Formula(aVF,0,aVF.back(),aVF.back());

   cIm2D<tU_INT2> aIm(cPt2di(10,10));
   FormalBilinIm2D_SetObs(aV,8,cPt2dr(0.4,1.5),aIm);

}
	

template <class Type> class cFormulaBilinIm2D
{
       public :
          // cGenerateFormulaOnIm2D(const std::string & aPrefix);
	  typedef cFormula<Type> tFormula;

          static std::vector<std::string> NameObs(const std::string & aPrefix);

	  tFormula  BilinVal(const std::vector<tFormula> & aVObs,int aKObs0,const tFormula&  FX,const tFormula& FY);

	  static void InitObs(std::vector<Type> & aVObs,int aK0,cPt2dr aPt,cIm2D<tREAL4> aIm);
	  static void InitObs(std::vector<Type> & aVObs,int aK0,cPt2dr aPt,cIm2D<tU_INT1> aIm);
};



template <class Type>   
   std::vector<std::string> cFormulaBilinIm2D<Type>::NameObs(const std::string & aPrefix)
{
   return std::vector<std::string> 
          {
              "PtX0_" + aPrefix,
              "PtY0_" + aPrefix,
              "Im00_" + aPrefix,
              "Im10_" + aPrefix,
              "Im01_" + aPrefix,
              "Im11_" + aPrefix
          };
}

template <class Type>   
	 cFormula<Type> cFormulaBilinIm2D<Type>::BilinVal
	                (
                            const std::vector<tFormula> & aVObs,
                            int aKObs0,
                            const tFormula&  FX,
                            const tFormula& FY
                         )
{
    tFormula aX0 = aVObs.at(aKObs0  );
    tFormula aY0 = aVObs.at(aKObs0+1);
    tFormula aCst1 = CreateCste(1.0,aX0);  // create a symbolic formula for constant 1

    tFormula aWX1 = FX -aX0;
    tFormula aWX0 = aCst1 - aWX1;
    tFormula aWY1 = FY -aY0;
    tFormula aWY0 = aCst1 - aWY1;


    return 
	    aWX0 * aWY0 * aVObs.at(aKObs0+2)
	  + aWX1 * aWY0 * aVObs.at(aKObs0+3)
	  + aWX0 * aWY1 * aVObs.at(aKObs0+4)
	  + aWX1 * aWY1 * aVObs.at(aKObs0+5) ;

}

template <class Type,class TypeIm>   
   void GlobInitObs(std::vector<Type> & aVObs,int aK0,cPt2dr aPt,cIm2D<TypeIm> aIm)
{
     const cDataIm2D<TypeIm> & aDIm = aIm.DIm();
     cPt2di aP0 = Pt_round_down(aPt);

     aVObs.at(aK0)   = aPt.x();
     aVObs.at(aK0+1) = aPt.y();
     aVObs.at(aK0+2) = aDIm.GetV(aP0);
     aVObs.at(aK0+3) = aDIm.GetV(aP0+cPt2di(1,0));
     aVObs.at(aK0+4) = aDIm.GetV(aP0+cPt2di(0,1));
     aVObs.at(aK0+5) = aDIm.GetV(aP0+cPt2di(1,1));
}


template <class Type>   
	  void cFormulaBilinIm2D<Type>::InitObs(std::vector<Type> & aVObs,int aK0,cPt2dr aPt,cIm2D<tREAL4> aIm)
{
   GlobInitObs(aVObs,aK0,aPt,aIm);
}
template <class Type>   
	  void cFormulaBilinIm2D<Type>::InitObs(std::vector<Type> & aVObs,int aK0,cPt2dr aPt,cIm2D<tU_INT1> aIm)
{
   GlobInitObs(aVObs,aK0,aPt,aIm);
}

template class cFormulaBilinIm2D<tREAL8>;

*/

/* ******************************** */
/*                                  */
/*         cDescOneFuncDist         */
/*                                  */
/* ******************************** */





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

