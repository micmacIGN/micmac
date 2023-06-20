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

class cRadiomFormulas
{
      public :
          static std::vector<std::string> VNamesObs()  { return {"xIm","yIm","Cx","Cy","NormR"}; }

          template <typename tObs> std::vector<tObs> CoordNom ( const std::vector<tObs> & aVObs,int aK0Obs)  const
          {
                const auto & xIm   = aVObs[aK0Obs+0];
                const auto & yIm   = aVObs[aK0Obs+1];
                const auto & Cx    = aVObs[aK0Obs+2];
                const auto & Cy    = aVObs[aK0Obs+3];
                const auto & NormR = aVObs[aK0Obs+4];

		auto aXRed =(xIm-Cx) / NormR;
		auto aYRed =(yIm-Cy) / NormR;

                return  {aXRed,aYRed};
          }
};

class cRadiomCalibRadSensor : public cRadiomFormulas
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
          // static std::vector<std::string> VNamesObs()  { return RF_NameObs();}


          std::string FormulaName() const { return "RadiomCalibRadSensor_" + ToStr(mDegRad);}

          template <typename tUk,typename tObs> 
             std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs,
                      int   aK0Uk   = 0,
                      int   aK0Obs  = 0
                  )  const
          {

                auto xyNorm = CoordNom (aVObs,aK0Obs);
		auto aR2N   = Square(xyNorm.at(0))+Square(xyNorm.at(1)) ;

	        tUk aCst1         = CreateCste(1.0,xyNorm.at(0));  // create a symbolic formula for constant 1
	        tUk aPowR2        = aCst1;
		tUk  aCorrec      = aCst1;

	        for  (int aK=0 ; aK<mDegRad ; aK++)
	        {
                    aPowR2 = aPowR2 * aR2N;
                    aCorrec = aCorrec + aPowR2*aVUk[aK+aK0Uk];
	        }

		return {aCorrec};
	  }


          int DegRad() const {return mDegRad;}


      private :
          int mDegRad;
};

class cRadiomCalibPolIma  : public cRadiomFormulas
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

          std::string FormulaName() const { return "RadiomCalibPolIm_" + ToStr(mDegIm);}

          template <typename tUk,typename tObs> 
             std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs,
                      int   aK0Uk  = 0,
                      int   aK0Obs = 0
                  )  const
          {
                auto  aVec = CoordNom (aVObs,aK0Obs);
                const auto & aXRed = aVec.at(0);
                const auto & aYRed = aVec.at(1);
	        tUk aCst0         = CreateCste(0.0,aXRed);  // create a symbolic formula for constant 1
		int aKPol = 0;
		tUk  aCorrec      = aCst0;

                for (int aDx=0 ; aDx<= mDegIm ; aDx++)
		{
                    for (int aDy=0 ; (aDy+aDx)<= mDegIm ; aDx++)
		    {
                         aCorrec = aCorrec + aVUk[aKPol+aK0Obs] *powI(aXRed,aDx) *  powI(aYRed,aDy) ;
			 aKPol++;
		    }
		}

		return {aCorrec};
	  }
          int DegIm () const {return mDegIm;}


      private :
          int mDegIm;
};


class cRadiomEqualisation 
{
      public :
          cRadiomEqualisation(int aDegSens,int aDegIm) :
              mK0Sens  (1),
              mCalSens (aDegSens),
              mK0CalIm (1+mCalSens.VNamesUnknowns().size()),
              mCalIm   (aDegIm),
              mK0Obs   (1)
          {
          }
          std::vector<std::string> VNamesUnknowns() const 
          {
              return Append({"Albedo"},Append(mCalSens.VNamesUnknowns(),mCalIm.VNamesUnknowns()));
          }
          std::string FormulaName() const 
          { 
                 return   "RadiomEqualisation_" 
                        + ToStr(mCalSens.DegRad()) + "_"
                        + ToStr(mCalIm.DegIm());
          }
          static std::vector<std::string> VNamesObs()  
          { 
              return Append({"RadIm"},cRadiomFormulas::VNamesObs());
          }

          template <typename tUk,typename tObs> 
             std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  )  const
          {
               const auto & aAlbedo = aVUk[0];
               const auto & aRadIm = aVObs[0];
               auto aCorrecSens = mCalSens.formula(aVUk,aVObs,mK0Sens ,mK0Obs);
               auto aCorrecIm   = mCalIm.formula (aVUk,aVObs ,mK0CalIm,mK0Obs);

               return {aRadIm / aCorrecSens - aAlbedo * aCorrecIm };
          }
            
      private :

          int                     mK0Sens;
          cRadiomCalibRadSensor   mCalSens;
          int                     mK0CalIm;
          cRadiomCalibPolIma      mCalIm;
          int                     mK0Obs;
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
