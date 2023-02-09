#ifndef _FORMULA_TARGETS_H_
#define _FORMULA_TARGETS_H_

#include "SymbDer/SymbDer_Common.h"
#include "MMVII_Ptxd.h"
#include "MMVII_Stringifier.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_PhgrDist.h"

#include "ComonHeaderSymb.h"

using namespace NS_SymbolicDerivative;


namespace MMVII
{

/**  Class for generating code relative to butterfly target estimation */

class cTargetShape
{
  public :
    cTargetShape()
    {
    }

    static const std::vector<std::string> VNamesUnknowns() { return Append(NamesP2("c_i"), NamesMatr("M_ic",{2,2})); }
    static const std::vector<std::string> VNamesObs()      { return Append(NamesP2("p_i"),{"v", "s", "min_val", "max_val"}); }

    std::string FormulaName() const { return "TargetShape";}

    template <typename tUk,typename tObs> 
             static std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  ) // const
    {
         assert (aVUk.size() == 2+4) ;
         assert (aVObs.size()== 6) ;

         //Todo: use mapping to change coords, inversible
          cPtxd<tUk,2>  c_i = VtoP2(aVUk,0); //target center in image coords

          cPtxd<tUk,2>  p_i = VtoP2(aVObs,0); //image coord
          const auto & v  = aVObs[2]; //image pix value
          const auto & s  = aVObs[3]; //blur factor
          const auto & min_val  = aVObs[4]; //min value in image
          const auto & max_val  = aVObs[5]; //max value in image

          auto p_t = MulMat2(aVUk,2,(p_i-c_i)); //point in target frame
          auto psin_xx = sin(p_t.x())/sqrt(Square(sin(p_t.x()))+Square(s));
          auto psin_yy = sin(p_t.y())/sqrt(Square(sin(p_t.y()))+Square(s));
          auto dist_xx_yy = sqrt(Square(p_t.x())+Square(p_t.y())+0.001); //+0.001 for stability around 0
          auto pcos_dist = cos(dist_xx_yy)/sqrt(cos(dist_xx_yy)*cos(dist_xx_yy)+s*s);
          auto g = 1.0 - (-(psin_xx*psin_yy)/2.0+0.5) * (pcos_dist/2.0+0.5);
          auto g_scaled = (max_val - min_val) * g + min_val;

          return { g_scaled - v } ;
     }
};

};//  namespace MMVII

#endif // _FORMULA_TARGETS_H_
