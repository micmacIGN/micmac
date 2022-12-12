#ifndef _FORMULA_GEOM3D_H_
#define _FORMULA_GEOM3D_H_

#include "SymbDer/SymbDer_Common.h"
#include "MMVII_Ptxd.h"
#include "MMVII_Stringifier.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_PhgrDist.h"

#include "ComonHeaderSymb.h"

using namespace NS_SymbolicDerivative;


namespace MMVII
{

/**  Class for generating code relative to 3D-distance observation */
class cDist3D
{
  public :
    cDist3D() {}
    static const std::vector<std::string> VNamesUnknowns() { return Append(NamesP3("p1"), NamesP3("p2")); }
    static const std::vector<std::string> VNamesObs()      { return {"D"}; }
    std::string FormulaName() const { return "Dist3D";}
    template <typename tUk,typename tObs>
             static std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  ) // const
    {
          auto p1 = VtoP3(aVUk,0);
          auto p2 = VtoP3(aVUk,3);
          auto v  = p1-p2;

          const auto & ObsDist  = aVObs[0];
          return { sqrt(square(v.x())+square(v.y())+square(v.z())) - ObsDist } ;
     }
};

/**  Class for generating code relative equal 3D-distance observation */
class cDist3DParam
{
  public :
    cDist3DParam() {}
    static const std::vector<std::string> VNamesUnknowns() { return Append({"d"},NamesP3("p1"), NamesP3("p2")); }
    static const std::vector<std::string> VNamesObs()      { return {}; }
    std::string FormulaName() const { return "Dist3DParam";}
    template <typename tUk,typename tObs>
             static std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      [[maybe_unused]] const std::vector<tObs> & aVObs
                  ) // const
    {
          const auto & d  = aVUk[0];
          auto p1 = VtoP3(aVUk,1);
          auto p2 = VtoP3(aVUk,4);
          auto v  = p1-p2;

          return { sqrt(square(v.x())+square(v.y())+square(v.z())) - d } ;
     }
};

/**  Class for generating code relative topometric subframe observation */
class cTopoSubFrame
{
  public :
    cTopoSubFrame() {}
    static const std::vector<std::string> VNamesUnknowns() { return Append(NamesP3("r"), NamesP3("p1"), NamesP3("p2")); }
    static const std::vector<std::string> VNamesObs()      { return Append(NamesMatr("R",cPt2di(3,3)), NamesP3("v")); }
    std::string FormulaName() const { return "TopoSubFrame";}
    template <typename tUk,typename tObs>
             static std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  ) // const
    {
          assert (aVUk.size() == 3+3+3) ;  // SD::UserSError("Bad size for unknown");
          assert (aVObs.size()== 9+3) ;// SD::UserSError("Bad size for observations");
          auto dr = VtoP3(aVUk,0);
          auto p1 = VtoP3(aVUk,3);
          auto p2 = VtoP3(aVUk,6);

          auto v   = VtoP3(aVObs,(9));

          //M=target
          //S=Station
          //MS=M-S, compensated vector in global frame
          auto  MS = p2-p1;

          //U=R*MS: compensated vector in sub frame (U0: without tiny rotation)
          auto U0 = MulMat(aVObs,0,MS);  // multiply by a priori rotation

           // Now "tiny" rotation
           //  Wx      X      Wy * Z - Wz * Y
           //  Wy  ^   Y  =   Wz * X - Wx * Z
           //  Wz      Z      Wx * Y - Wy * X

            //  P =  P0 + W ^ P0
           auto U = U0 + (dr ^ U0);
           /*auto  Ux = Ux0 + b * Uz0 - c * Uy0;
           auto  Uy = Uy0 + c * Ux0 - a * Uz0;
           auto  Uz = Uz0 + a * Uy0 - b * Ux0;*/

           auto err = U-v;
          return { err.x(), err.y(), err.z() } ;
     }
};




};//  namespace MMVII

#endif // _FORMULA_GEOM3D_H_
