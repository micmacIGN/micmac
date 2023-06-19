#ifndef _FORMULA_RADIOM_H_
#define _FORMULA_RADIOM_H_

#include "SymbDer/SymbDer_Common.h"
#include "MMVII_Ptxd.h"
#include "MMVII_Stringifier.h"
#include "MMVII_DeclareCste.h"

#include "ComonHeaderSymb.h"

using namespace NS_SymbolicDerivative;


namespace MMVII
{

class cRadiomCalibRadSensor
{
      public :
          cRadiomCalibRadSensor(int aDegRad) :
               mDegRad (aDegRad)
          {
          }

          std::vector<std::string> VNamesUnknowns() const 
          {
	      std::vector<std::string>  aRes ;
              for (int aK=0 ; aK< mDegRad ; aK++)
                  aRes.push_back("K"+ToStr(aK));
              return aRes;
          }
          std::vector<std::string> VNamesObs()  const     { return {"xIm","yIm","Cx","Cy","NormR"}; }


          std::string FormulaName() const { return "RadiomCalibRadSensor_" + ToStr(mDegRad);}

          template <typename tUk,typename tObs> 
             std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  )  const
          {
                const auto & xIm   = aVObs[0];
                const auto & yIm   = aVObs[1];
                const auto & Cx    = aVObs[2];
                const auto & Cy    = aVObs[3];
                const auto & NormR = aVObs[4];

		auto aR2N = (  Square(xIm-Cx)+Square(yIm-Cy) ) /  Square(NormR);

	        tUk aCst1         = CreateCste(1.0,xIm);  // create a symbolic formula for constant 1
	        tUk aPowR2        = aCst1;
		tUk  aCorrec      = aCst1;

	        for  (int aK=0 ; aK<mDegRad ; aK++)
	        {
                    aPowR2 = aPowR2 * aR2N;
                    aCorrec = aCorrec + aPowR2*aVUk[aK];
	        }

		return {aCorrec};
	  }


      private :
          int mDegRad;
};

class cRadiomCalibPolIma
{
      public :
          cRadiomCalibPolIma  (int aDegIm) :
               mDegIm         (aDegIm)
          {
          }

          std::vector<std::string> VNamesUnknowns() const 
          {
	      std::vector<std::string>  aRes ;
              for (int aDx=0 ; aDx<= mDegIm ; aDx++)
                  for (int aDy=0 ; (aDy+aDx)<= mDegIm ; aDx++)
                       aRes.push_back("DIm_"+ToStr(aDx) + "_" + ToStr(aDy));
              return aRes;
          }
          std::vector<std::string> VNamesObs()  const     { return {"xIm","yIm","Cx","Cy","NormR"}; }


          std::string FormulaName() const { return "RadiomCalibRadSensor_" + ToStr(mDegIm);}

          template <typename tUk,typename tObs> 
             std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  )  const
          {
                const auto & xIm   = aVObs[0];
                const auto & yIm   = aVObs[1];
                const auto & Cx    = aVObs[2];
                const auto & Cy    = aVObs[3];
                const auto & NormR = aVObs[4];

		auto aXRed =(xIm-Cx) / NormR;
		auto aYRed =(yIm-Cy) / NormR;
	        tUk aCst0         = CreateCste(0.0,xIm);  // create a symbolic formula for constant 1
		int aKPol = 0;
		tUk  aCorrec      = aCst0;

                for (int aDx=0 ; aDx<= mDegIm ; aDx++)
		{
                    for (int aDy=0 ; (aDy+aDx)<= mDegIm ; aDx++)
		    {
                         aCorrec = aCorrec + aVUk[aKPol] *powI(aXRed,aDx) *  powI(aYRed,aDy) ;
			 aKPol++;
		    }
		}

		return {aCorrec};
	  }


      private :
          int mDegIm;
};








/**  Class for calibration of radiometry 
 *
 *    Rad/F (1+K1 r2 + K2 r2 ^2 ...) = Albedo
 *
 *
 * */

class cRadiomVignettageLinear
{
  public :
    cRadiomVignettageLinear(int aNb)  :
	    mNb (aNb)
    {
    }

    std::vector<std::string> VNamesUnknowns() const 
    {
	   std::vector<std::string>  aRes  { "Albedo","MulIm"};
	   for (int aK=0 ; aK< mNb ; aK++)
               aRes.push_back("K"+ToStr(aK));
	   return aRes;
    }
    std::vector<std::string> VNamesObs()  const     { return {"RadIm","Rho2"}; }

    std::string FormulaName() const { return "RadiomVignetLin_" + ToStr(mNb);}

    template <typename tUk,typename tObs> 
             std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  )  const
    {
          const auto & aAlbedo = aVUk[0];
          const auto & aMulIm  = aVUk[1];
          const auto & aRadIm = aVObs[0];
          const auto & aRho2 = aVObs[1];

	  tUk aCst1         = CreateCste(1.0,aAlbedo);  // create a symbolic formula for constant 1
	  tUk aCorrecRadial = aCst1;
	  tUk aPowR2        = aCst1;

	  for  (int aK=0 ; aK<mNb ; aK++)
	  {
                aPowR2 = aPowR2 * aRho2;
		aCorrecRadial = aCorrecRadial + aPowR2*aVUk[aK+2];
	  }

          // return {aRadIm * aCorrecRadial - aAlbedo * aMulIm};
          return {aRadIm / aCorrecRadial - aAlbedo * aMulIm };
	   
     }

  private :
     int mNb;
};



};//  namespace MMVII

#endif // _FORMULA_RADIOM_H_
