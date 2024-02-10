#ifndef _FORMULA_GEOMED_H_
#define _FORMULA_GEOMED_H_

#include "SymbDer/SymbDer_Common.h"
#include "MMVII_Ptxd.h"
#include "MMVII_Stringifier.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_PhgrDist.h"

#include "ComonHeaderSymb.h"

using namespace NS_SymbolicDerivative;


namespace MMVII
{
/**  Class used for generating a "2D" polynomial deformation.  In fact it's a bit  "take a hammer to crush a fly" because
 * it's purely linear in term of coefficients.  BTW, this allow to use the standard interface.
*/

class cFormulaPolyn2D
{
  public :
    cFormulaSumSquares(int aDegree,bool WithNorm) :
       mDegree      (aDegree),
       mWithNom     (WithNorm)
    {
        for (int aDegX=0 ; aDegX<= mDegree; aDegX++)
	{
            for (int aDegY=0 ; (aDegX+aDegY) <= mDegree; aDegY++)
	    {
		mVDesc.push_back(cDescOneFuncDist(eTypeFuncDist::eMonX,cPt2di(aDegX+aDegY),eModeDistMonom::eModeStd));
		mVDesc.push_back(cDescOneFuncDist(eTypeFuncDist::eMonY,cPt2di(aDegX+aDegY),eModeDistMonom::eModeStd));
	    }
	}
    }


    std::vector<std::string> VNamesUnknowns()  const
    { 
         std::vector<std::string> aRes;
         for (const auto & aDesc : mVDesc) 
	 {
             aRes.push_back(aDesc.mName);
	 }
         return aRes;
    }

    std::vector<std::string> VNamesObs() const      
    { 
        if (mWithNorm)
            return {"XMin","YMin","XMax","YMax"};
         return {};
    }

    std::string FormulaName() const { return "SumSquare_"+ToStr(mNb);}

    template <typename tUk,typename tObs> 
             std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  )  const
    {
	    /*
	  auto aSum = aVObs.at(0);
          for (int aK=0 ; aK<mNb ; aK++)
              aSum = aSum - Square(aVUk.at(aK));
          return { aSum};
	  */
     }

/*
*/
     int                             mDegree;
     bool                            mWithNorm;
     std::vector<cDescOneFuncDist>   mVDesc;
};



};//  namespace MMVII

#endif // _FORMULA_GEOMED_H_
