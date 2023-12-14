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

class cRadiomCalibPolIma  : public cRadiomFormulas
{
      public :
          cRadiomCalibPolIma  (int aDegIm,int aDegRadElim=-1) :
               mDegIm         (aDegIm),
	       mDegRadElim    (aDegRadElim)
          {
          }
	  std::vector<cDescOneFuncDist> VDesc() const
	  {
	      std::vector<cDescOneFuncDist> aRes;

              for (int aDx=0 ; aDx<= mDegIm ; aDx++)
	      {
                  for (int aDy=0 ; (aDy+aDx)<= mDegIm ; aDy++)
		  {
                       //int aDTot = aDx+aDy;
		       // we avoid  1 X2 X4 ... if they are redundant with radial
		       if (   ((aDx+aDy)>mDegRadElim) || (aDy!=0) || (aDx%2!=0))
                           aRes.push_back(cDescOneFuncDist(eTypeFuncDist::eMonom, cPt2di(aDx,aDy),true));
		  }
	      }
	      return aRes;
	  }

          std::vector<std::string> VNamesUnknowns() const 
          {
	      std::vector<std::string>  aRes ;
              for (const auto & aDesc : VDesc())
                  aRes.push_back( ((mDegRadElim<0) ? std::string("") : std::string("Sens_"))+aDesc.mName);
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
		tUk  aCorrec      = aCst0;

		auto aVDesc = VDesc() ;
                for (size_t aKPol=0 ; aKPol<aVDesc.size() ; aKPol++)
		{
                    auto aDeg = aVDesc[aKPol].mDegMon;
                    aCorrec = aCorrec + aVUk[aKPol+aK0Uk] * powI(aXRed,aDeg.x()) *  powI(aYRed,aDeg.y()) ;
		}

		return {aCorrec};
	  }
          int DegIm () const {return mDegIm;}


      private :
          int mDegIm;
          int mDegRadElim;
};

/*
 *
 */

class cRadiomCalibRadSensor : public cRadiomFormulas
{
      public :
          cRadiomCalibRadSensor(int aDegRad,bool  WithCsteAdd=false,int aDegPol=-1) :
             mDegRad      (aDegRad),
             mWithCsteAdd (WithCsteAdd),
	     mDegPol      (aDegPol)
          {
          }

          std::vector<std::string> VNamesUnknowns() const 
          {
	      std::vector<std::string>  aRes ;
	      if (mWithCsteAdd)
                 aRes.push_back("DarkCurrent");
              for (int aK=0 ; aK< mDegRad ; aK++)
                  aRes.push_back("K"+ToStr(aK));
	      if (mDegPol>0)
	      {
                  AppendIn(aRes,cRadiomCalibPolIma(mDegPol,mDegRad).VNamesUnknowns());
	      }
              return aRes;
          }
          // static std::vector<std::string> VNamesObs()  { return RF_NameObs();}

          std::string FormulaAddName() const 
	  {
              std::string aRes =   std::string("_DRad") + ToStr(mDegRad);
	      if (mWithCsteAdd) aRes = aRes + "_WCste";
	      if (mDegPol>=0) aRes = aRes + "_DPolSens" + ToStr(mDegPol);

              return aRes;
	  }

          std::string FormulaName() const 
	  { 
              return "RadiomCalibRadSensor" +  FormulaAddName();
	  }

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
	        tUk aCst0         = CreateCste(0.0,xyNorm.at(0));  // create a symbolic formula for constant 1
	        tUk aPowR2        = aCst1;
		tUk  aCorrecMul   = aCst1;

	        for  (int aK=0 ; aK<mDegRad ; aK++)
	        {
                    aPowR2 = aPowR2 * aR2N;
                    aCorrecMul = aCorrecMul + aPowR2*aVUk[aK0Uk+mWithCsteAdd+aK];
	        }

		if (mDegPol>0)
		{
                     cRadiomCalibPolIma  aRCPI(mDegPol,mDegRad);
		     aCorrecMul  = aCorrecMul + aRCPI.formula(aVUk,aVObs,aK0Uk+mDegRad,aK0Obs).at(0);
		}

		tUk  aCorrecAdd =  mWithCsteAdd ? aVUk[aK0Uk] : aCst0;

		return  {aCorrecAdd,aCorrecMul};
	  }


          int DegRad() const {return mDegRad;}


      private :
         int  mDegRad;
         int  mWithCsteAdd;
         int  mDegPol;
};


class cRadiomEqualisation 
{
      public :
          typedef std::vector<std::string> tVStr;

          cRadiomEqualisation(bool is4Eq,int aDegSens,int aDegIm,bool WithCste=false,int aDegSensPol=-1) :
              m4Eq        (is4Eq),
              mOwnUK      (m4Eq ? tVStr({"Albedo"}) : tVStr() ),
              mOwnObs     (tVStr({"RadIm"})),
              mK0Sens     (mOwnUK.size()),
              mCalSens    (aDegSens,WithCste,aDegSensPol),
              mK0CalIm    (mK0Sens+mCalSens.VNamesUnknowns().size()),
              mCalIm      (aDegIm),
              mK0Obs      (mOwnObs.size())
          {
          }

          tVStr VNamesUnknowns() const 
          {
              //  Albedo?   K0 K1 K2 ...   D_Im00  D_Im01 ... 
              return Append(mOwnUK,Append(mCalSens.VNamesUnknowns(),mCalIm.VNamesUnknowns()));
          }
          tVStr VNamesObs()  const
          { 
              //  RadIm?    xIm  yIm Cx Cy R
              return Append(mOwnObs,cRadiomFormulas::VNamesObs());
          }

          std::string FormulaName() const 
          { 
                 return   (m4Eq ? "RadiomEqualisation"  : "RadiomStabilized") 
                        + mCalSens.FormulaAddName()
                        + std::string("_DegIm") + ToStr(mCalIm.DegIm())
			
                 ;
          }

          template <typename tUk,typename tObs> 
             std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  )  const
          {
               auto aCorrecIm   = mCalIm.formula  (aVUk,aVObs,mK0CalIm,mK0Obs).at(0);
               const auto & aRadIm = aVObs[0];
               if (m4Eq)
               {
                    auto aVCorrSens = mCalSens.formula(aVUk,aVObs,mK0Sens ,mK0Obs);
		    auto aCorSensAdd = aVCorrSens.at(0);
		    auto aCorSensMul = aVCorrSens.at(1);

                    const auto & aAlbedo = aVUk[0];

                    return {aRadIm -  aCorrecIm * (aAlbedo*aCorSensMul+aCorSensAdd)}; // 3.69312
                    // return {CreateCste(1.0,aAlbedo) - (aAlbedo * aCorrecIm * aCorrecSens)/aRadIm};  NAN
                    //  return {aRadIm/aCorrecSens - aAlbedo * aCorrecIm };     //3.73095
                    //  return {aRadIm/(aCorrecSens*aCorrecIm) - aAlbedo  };   // 3.7714
                    // return {aRadIm/aAlbedo -  aCorrecIm * aCorrecSens}; // 3.94555
               }
               else
               {
                     return {aRadIm*(aVUk[mK0CalIm]-aCorrecIm )};
               }
          }
      private :
          bool                       m4Eq;
          tVStr                      mOwnUK;
          tVStr                      mOwnObs;
          int                        mK0Sens;
          cRadiomCalibRadSensor      mCalSens;
          int                        mK0CalIm;
          cRadiomCalibPolIma         mCalIm;
          int                        mK0Obs;
};


// static std::vector<std::string> VNamesObs()  { return {"xIm","yIm","Cx","Cy","NormR"}; }

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
