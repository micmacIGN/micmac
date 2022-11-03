#ifndef _FORMULA_GEOM3D_H_
#define _FORMULA_GEOM3D_H_

#include "SymbDer/SymbDer_Common.h"
#include "MMVII_Ptxd.h"
#include "MMVII_Stringifier.h"
#include "MMVII_DeclareCste.h"

#include "ComonHeaderSymb.h"

using namespace NS_SymbolicDerivative;


namespace MMVII
{

/**  Class for generating code relative to 3D-distance observation */
class cDist3D
{
  public :
    cDist3D() {}
    static const std::vector<std::string> VNamesUnknowns() { return {"x1","y1","z1","x2","y2","z2"}; }
    static const std::vector<std::string> VNamesObs()      { return {"D"}; }
    std::string FormulaName() const { return "Dist3D";}
    template <typename tUk,typename tObs>
             static std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  ) // const
    {
          const auto & x1 = aVUk[0];
          const auto & y1 = aVUk[1];
          const auto & z1 = aVUk[2];
          const auto & x2 = aVUk[3];
          const auto & y2 = aVUk[4];
          const auto & z2 = aVUk[5];

          const auto & ObsDist  = aVObs[0];
          return { sqrt(square(x1-x2) + square(y1-y2) + square(z1-z2)) - ObsDist } ;
     }
};

/**  Class for generating code relative equal 3D-distance observation */
class cDist3DParam
{
  public :
    cDist3DParam() {}
    static const std::vector<std::string> VNamesUnknowns() { return {"d","x1","y1","z1","x2","y2","z2"}; }
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
          const auto & x1 = aVUk[1];
          const auto & y1 = aVUk[2];
          const auto & z1 = aVUk[3];
          const auto & x2 = aVUk[4];
          const auto & y2 = aVUk[5];
          const auto & z2 = aVUk[6];

          return { sqrt(square(x1-x2) + square(y1-y2) + square(z1-z2)) - d } ;
     }
};

/**  Class for generating code relative topometric subframe observation */
class cTopoSubFrame
{
  public :
    cTopoSubFrame() {}
    static const std::vector<std::string> VNamesUnknowns() { return {"a","b","c","x1","y1","z1","x2","y2","z2"}; }
    static const std::vector<std::string> VNamesObs()      { return {"r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22", "dx","dy","dz"}; }
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
          const auto & a  = aVUk[0];
          const auto & b  = aVUk[1];
          const auto & c  = aVUk[2];
          const auto & x1 = aVUk[3];
          const auto & y1 = aVUk[4];
          const auto & z1 = aVUk[5];
          const auto & x2 = aVUk[6];
          const auto & y2 = aVUk[7];
          const auto & z2 = aVUk[8];

          const auto & dx  = aVObs[9];
          const auto & dy  = aVObs[10];
          const auto & dz  = aVObs[11];

          //M=target
          //S=Station
          //MS=M-S, compensated vector in global frame
          auto  MSx = x2-x1;
          auto  MSy = y2-y1;
          auto  MSz = z2-z1;
          //U=R*MS: compensated vector in sub frame (without tiny rotation)
          auto  Ux0 = aVObs[0] * MSx +  aVObs[1] * MSy +  aVObs[2] * MSz;
          auto  Uy0 = aVObs[3] * MSx +  aVObs[4] * MSy +  aVObs[5] * MSz;
          auto  Uz0 = aVObs[6] * MSx +  aVObs[7] * MSy +  aVObs[8] * MSz;

           // Now "tiny" rotation
           //  Wx      X      Wy * Z - Wz * Y
           //  Wy  ^   Y  =   Wz * X - Wx * Z
           //  Wz      Z      Wx * Y - Wy * X

            //  P =  P0 + W ^ P0

           auto  Ux = Ux0 + b * Uz0 - c * Uy0;
           auto  Uy = Uy0 + c * Ux0 - a * Uz0;
           auto  Uz = Uz0 + a * Uy0 - b * Ux0;

          return { Ux-dx, Uy-dy, Uz-dz } ;
     }
};




};//  namespace MMVII

#endif // _FORMULA_GEOM3D_H_
