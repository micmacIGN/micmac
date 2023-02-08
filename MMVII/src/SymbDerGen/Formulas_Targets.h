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

    static const std::vector<std::string> VNamesUnknowns() { return Append(NamesP2("c"), {"a", "b", "alpha", "beta"}); }
    static const std::vector<std::string> VNamesObs()      { return {"x", "y", "v", "s", "min_val", "max_val"}; }

    std::string FormulaName() const { return "TargetShape";}

    template <typename tUk,typename tObs> 
             static std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  ) // const
    {
         assert (aVUk.size() == 6) ;
         assert (aVObs.size()== 6) ;

         //Todo: use mapping to change coords, inversible
          cPtxd<tUk,2>  c = VtoP2(aVUk,0); //target center in image coords
          auto a     = aVUk[2]; //radius axe 1
          auto b     = aVUk[3]; //radius axe 2
          auto alpha = aVUk[4]; //az axe 1
          auto beta  = aVUk[5]; //az axe 2


          const auto & x  = aVObs[0]; //image coord
          const auto & y  = aVObs[1];
          const auto & v  = aVObs[2]; //image pix value
          const auto & s  = aVObs[3]; //blur factor
          const auto & min_val  = aVObs[4]; //min value in image
          const auto & max_val  = aVObs[5]; //max value in image

          auto xx = (cos(alpha)*(x-c.x()) + sin(alpha)*(y-c.y()))/a*M_PI/2.0;
          auto yy = (cos(beta)*(x-c.x()) + sin(beta)*(y-c.y()))/b*M_PI/2.0;
          auto psin_xx = sin(xx)/sqrt(Square(sin(xx))+Square(s));
          auto psin_yy = sin(yy)/sqrt(Square(sin(yy))+Square(s));
          auto dist_xx_yy = sqrt(Square(xx)+Square(yy)+0.001); //+0.001 for stability around 0
          auto pcos_dist = cos(dist_xx_yy)/sqrt(cos(dist_xx_yy)*cos(dist_xx_yy)+s*s);
          auto g = 1.0 - (-(psin_xx*psin_yy)/2.0+0.5) * (pcos_dist/2.0+0.5);
          auto g_scaled = (max_val - min_val) * g + min_val;

          return { g_scaled - v } ;
     }
};

};//  namespace MMVII

#endif // _FORMULA_TARGETS_H_
