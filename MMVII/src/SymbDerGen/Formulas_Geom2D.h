#ifndef _FORMULA_GEOMED_H_
#define _FORMULA_GEOMED_H_

#include "include/MMVII_all.h"
#include "include/SymbDer/SymbolicDerivatives.h"

using namespace NS_SymbolicDerivative;


namespace MMVII
{

class cDist2DConservation
{
  public :
    cDist2DConservation() 
    {
    }

    static const std::vector<std::string> VNamesUnknowns() { return {"x1","y1","x2","y2"}; }
    static const std::vector<std::string> VNamesObs()      { return {"D"}; }

    std::string FormulaName() const { return "Dist2DCons";}

    template <typename tUk, typename tObs>
             std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  ) const
    {
          const auto & x1 = aVUk[0];
          const auto & y1 = aVUk[1];
          const auto & x2 = aVUk[2];
          const auto & y2 = aVUk[3];

          const auto & ObsDist  = aVObs[0];  
	  const auto aCst1 = CreateCste(1.0,x1);


          return { sqrt(square(x1-x2) + square(y1-y2))/ObsDist - aCst1 } ;
     }
};

class cRatioDist2DConservation
{
  public :
    cRatioDist2DConservation() 
    {
    }

    static const std::vector<std::string> VNamesUnknowns() { return {"x1","y1","x2","y2","x3","y3"}; }
    static const std::vector<std::string> VNamesObs()      { return {"D12","D13","D23"}; }

    std::string FormulaName() const { return "RatioDist2DCons";}

    template <typename tUk, typename tObs>
             std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  ) const
    {
          const auto & x1 = aVUk[0];
          const auto & y1 = aVUk[1];
          const auto & x2 = aVUk[2];
          const auto & y2 = aVUk[3];
          const auto & x3 = aVUk[4];
          const auto & y3 = aVUk[5];

          const auto & Obs_d12  = aVObs[0];  
          const auto & Obs_d13  = aVObs[1];  
          const auto & Obs_d23  = aVObs[2];  

          const auto r12 =  sqrt(square(x1-x2) + square(y1-y2)) / Obs_d12 ;
          const auto r13 =  sqrt(square(x1-x3) + square(y1-y3)) / Obs_d13 ;
          const auto r23 =  sqrt(square(x2-x3) + square(y2-y3)) / Obs_d23 ;
          return { r12-r13,r12-r23,r13-r23};
     }
};





};//  namespace MMVII

#endif // _FORMULA_GEOMED_H_
