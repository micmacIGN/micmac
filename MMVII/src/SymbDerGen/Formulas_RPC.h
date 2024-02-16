#ifndef _FORMULA_RPC_H_
#define _FORMULA_RPC_H_

// PUSHB

#include "SymbDer/SymbDer_Common.h"
#include "MMVII_Ptxd.h"
#include "MMVII_Stringifier.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_PhgrDist.h"

#include "ComonHeaderSymb.h"

using namespace NS_SymbolicDerivative;


namespace MMVII
{

/** "Helper" class for generating the distorsion, it can be used in 3 contexts :
*/

class cRPC_Formulas
{
   public :
      static std::vector<std::string>  NameCoeff(const std::string & aPref)
      {
           std::vector<std::string> aRes;

           for (int aK=0 ; aK<20 ; aK++)
               aRes.push_back(aPref + "_" + ToStr(aK));
           return aRes;
      }

      template <typename tUk,typename tObs>
             static tUk Polyn
                          (
                               const cPtxd<tUk,3> & aP,
                               const std::vector<tObs> & aVCoeffs,
		               int aK0
                           ) 
      {
        return   
              aVCoeffs[aK0+0]
            + aVCoeffs[aK0+5]  * aP.y() * aP.z()
            + aVCoeffs[aK0+10] * aP.x() * aP.y() * aP.z()
            + aVCoeffs[aK0+15] * aP.x() * aP.x() * aP.x()
            + aVCoeffs[aK0+1]  * aP.y()
            + aVCoeffs[aK0+6]  * aP.x() * aP.z()
            + aVCoeffs[aK0+11] * aP.y() * aP.y() * aP.y()
            + aVCoeffs[aK0+16] * aP.x() * aP.z() * aP.z()
            + aVCoeffs[aK0+2]  * aP.x()
            + aVCoeffs[aK0+7]  * aP.y() * aP.y()
            + aVCoeffs[aK0+12] * aP.y() * aP.x() * aP.x()
            + aVCoeffs[aK0+17] * aP.y() * aP.y() * aP.z()
            + aVCoeffs[aK0+3]  * aP.z()
            + aVCoeffs[aK0+8]  * aP.x() * aP.x()
            + aVCoeffs[aK0+13] * aP.y() * aP.z() * aP.z()
            + aVCoeffs[aK0+18] * aP.x() * aP.x() * aP.z()
            + aVCoeffs[aK0+4]  * aP.y() * aP.x()
            + aVCoeffs[aK0+9]  * aP.z() * aP.z()
            + aVCoeffs[aK0+14] * aP.y() * aP.y() * aP.x()
            + aVCoeffs[aK0+19] * aP.z() * aP.z() * aP.z();
      }

      template <typename tUk,typename tObs>
             static tUk   RatioPolyn
                          (
                               const cPtxd<tUk,3> & aP,
                               const std::vector<tObs> & aVCoeffs,
		               int aKNum,
			       int aKDenom
                           )  
              {
		      return  Polyn(aP,aVCoeffs,aKNum) /  Polyn(aP,aVCoeffs,aKDenom);
              }

      template <typename tUk,typename tObs>
             static cPtxd<tUk,3> NormalizeIn
	                  (
			        const cPtxd<tUk,3> & aP,
				const cPtxd<tObs,3> & aOffset,
				const cPtxd<tObs,3> aScale
                          )
	     {
		     return DivCByC(aP-aOffset,aScale);
	     }
      template <typename tUk,typename tObs>
             static cPtxd<tUk,3> NormalizeOut
	                  (
			        const cPtxd<tUk,3> & aP,
				const cPtxd<tObs,3> & aOffset,
				const cPtxd<tObs,3> aScale
                          )
	     {
		     return  aOffset + MulCByC(aP,aScale);
	     }

};

class cFormula_RPC_RatioPolyn : public cRPC_Formulas
{
   public :
      /// constructor 
      cFormula_RPC_RatioPolyn()
      {
      }

      static std::vector<std::string> VNamesUnknowns()
      {
	      return {"X","Y","Z"};
      }

      static std::vector<std::string> VNamesObs()
      {
          return
              Append
	      (
	         {"OffIn_x","OffIn_y","OffIn_z","ScaleIn_x","ScaleIn_y","ScaleIn_z"},
	         {"OffOut_x","OffOut_y","OffOut_z","ScaleOut_x","ScaleOut_y","ScaleOut_z"},
	         Append 
	         (
	           Append ( NameCoeff("INum"), NameCoeff("IDen")),
	           Append ( NameCoeff("JNum"), NameCoeff("JDen"))
                 )
              );
      }

      static  std::string FormulaName() { return "RPC_Transfo";}


      template <typename tUk,typename tObs>
             static std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  ) 
      {
	    cPtxd<tObs,3> aOffsIn   =  VtoP3(aVObs,0);
	    cPtxd<tObs,3> aScaleIn  = VtoP3(aVObs,3);
	    cPtxd<tObs,3> aOffsOut  =  VtoP3(aVObs,6);
	    cPtxd<tObs,3> aScaleOut = VtoP3(aVObs,9);

	    cPtxd<tUk,3> aPtIn = VtoP3(aVUk,0);
	    aPtIn  = NormalizeIn(aPtIn,aOffsIn,aScaleIn);

// return ToVect(aPtIn);
            tUk aI = RatioPolyn(aPtIn,aVObs,12,32);
            tUk aJ = RatioPolyn(aPtIn,aVObs,52,72);

	    cPtxd<tUk,3> aPtOut(aI,aJ,aPtIn.z());
	    aPtOut = NormalizeOut(aPtOut,aOffsOut,aScaleOut);

	    return ToVect(aPtOut);
/*
	    return ToVect(aPtIn);
*/
      }

};

};//  namespace MMVII

#endif // _FORMULA_RPC_H_
