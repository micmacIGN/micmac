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
          typedef cPtxd<tUk,3> tPt;
          tPt p1 = VtoP3(aVUk,0);
          tPt p2 = VtoP3(aVUk,3);
          tPt v  = p1-p2;

          const tUk & ObsDist  = aVObs[0];
          // return { sqrt(square(v.x())+square(v.y())+square(v.z())) - ObsDist } ;
          return {  Norm2(v) - ObsDist } ;
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
                      const std::vector<tObs> & 
                      // [[maybe_unused]] const std::vector<tObs> & aVObs
                  ) // const
    {
          typedef cPtxd<tUk,3> tPt;

          const tUk & d  = aVUk[0];
          tPt p1 = VtoP3(aVUk,1);
          tPt p2 = VtoP3(aVUk,4);
          tPt v  = p1-p2;

          return { Norm2(v) - d } ;
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
          typedef cPtxd<tUk,3> tPt;

          assert (aVUk.size() == 3+3+3) ;  // SD::UserSError("Bad size for unknown");
          assert (aVObs.size()== 9+3) ;// SD::UserSError("Bad size for observations");
          tPt dr = VtoP3(aVUk,0);
          tPt p1 = VtoP3(aVUk,3);
          tPt p2 = VtoP3(aVUk,6);

          tPt v   = VtoP3(aVObs,(9));

          //M=target
          //S=Station
          //MS=M-S, compensated vector in global frame
          tPt  MS = p2-p1;

          //U=R*MS: compensated vector in sub frame (U0: without tiny rotation)
          tPt U0 = MulMat(aVObs,0,MS);  // multiply by a priori rotation

           // Now "tiny" rotation
           //  Wx      X      Wy * Z - Wz * Y
           //  Wy  ^   Y  =   Wz * X - Wx * Z
           //  Wz      Z      Wx * Y - Wy * X

            //  P =  P0 + W ^ P0
           tPt U = U0 + (dr ^ U0);
           /*auto  Ux = Ux0 + b * Uz0 - c * Uy0;
           auto  Uy = Uy0 + c * Ux0 - a * Uz0;
           auto  Uz = Uz0 + a * Uy0 - b * Ux0;*/

           tPt err = U-v;
          return { err.x(), err.y(), err.z() } ;
     }
};




};//  namespace MMVII

#endif // _FORMULA_GEOM3D_H_
