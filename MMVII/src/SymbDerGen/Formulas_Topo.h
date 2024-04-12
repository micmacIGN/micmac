#ifndef _FORMULA_TOPO_H_
#define _FORMULA_TOPO_H_

#include "SymbDer/SymbolicDerivatives.h"
#include "SymbDer/SymbDer_MACRO.h"
#include "ComonHeaderSymb.h"
#include "MMVII_PhgrDist.h"

/**  Class for generating topometric equations:
 *
 *  The instrument frame is representated as a pose:
 *    Pose_instr = {Ci ; Ri}
 *
 *  The observations are are expressed in instrument cartesian frame.
 *  If the instrument is verticalized, the tilt part of the rotation will be
 *  fixed to a value corresponding to local vertical.
 *
 *  In instument frame:
 *   - pt_from_instr = [0,0,0]
 *   - pt_to_instr = -tRi Ci + tRi*pt_to_instr
 *
 */
using namespace NS_SymbolicDerivative;

namespace MMVII
{

class cFormulaTopoHz
{
      public :

           std::string FormulaName() const { return "TopoHz";}

           std::vector<std::string>  VNamesUnknowns()  const
           {
                //  Instrument pose with 6 unknowns : 3 for centers, 3 for axiator
               // target pose with 3 unknowns : 3 for center
                return  Append(NamesPose("Ci","Wi"),NamesP3("P_to"));
           }

           std::vector<std::string>    VNamesObs() const
           {
                // for the instrument pose, the 3x3 current rotation matrix as "observation/context"
                // and the measure value
                return  Append(NamesMatr("mi",cPt2di(3,3)), {"val"} );
           }

           template <typename tUk>
                       std::vector<tUk> formula
                       (
                          const std::vector<tUk> & aVUk,
                          const std::vector<tUk> & aVObs
                       ) const
           {
               cPoseF<tUk>  aPoseInstr(aVUk,0,aVObs,0,true);
               cPtxd<tUk,3> aP_to = VtoP3(aVUk,6);
               auto       val = aVObs[9];
               cPtxd<tUk,3>  aP_to_instr = aPoseInstr.Inverse().Value(aP_to);

               auto az = ATan2( aP_to_instr.x(), aP_to_instr.y() );

               return {  az - val };
           }
};


class cFormulaTopoZen
{
      public :

           std::string FormulaName() const { return "TopoZen";}

           std::vector<std::string>  VNamesUnknowns()  const
           {
                //  Instrument pose with 6 unknowns : 3 for centers, 3 for axiator
               // target pose with 3 unknowns : 3 for center
                return  Append(NamesPose("Ci","Wi"),NamesP3("P_to"));
           }

           std::vector<std::string>    VNamesObs() const
           {
                // for the instrument pose, the 3x3 current rotation matrix as "observation/context"
                // and the measure value
                return  Append(NamesMatr("mi",cPt2di(3,3)), {"val"} );
           }

           template <typename tUk>
                       std::vector<tUk> formula
                       (
                          const std::vector<tUk> & aVUk,
                          const std::vector<tUk> & aVObs
                       ) const
           {
               cPoseF<tUk>  aPoseInstr(aVUk,0,aVObs,0,true);
               cPtxd<tUk,3> aP_to = VtoP3(aVUk,6);
               auto       val = aVObs[9];
               cPtxd<tUk,3>  aP_to_instr = aPoseInstr.Inverse().Value(aP_to);

               auto   dist_hz =  Norm2(  cPtxd<tUk,2>(aP_to_instr.x(), aP_to_instr.y() ) );

               auto   zen = ATan2( aP_to_instr.z(), dist_hz );

               return {  zen - val };
           }
};


class cFormulaTopoDist
{
      public :

           std::string FormulaName() const { return "TopoDist";}

           std::vector<std::string>  VNamesUnknowns()  const
           {
                //  Instrument pose with 6 unknowns : 3 for centers, 3 for axiator
               // target pose with 3 unknowns : 3 for center
                return  Append(NamesPose("Ci","Wi"),NamesP3("P_to"));
           }

           std::vector<std::string>    VNamesObs() const
           {
                // for the instrument pose, the 3x3 current rotation matrix as "observation/context"
                // and the measure value
                return  Append(NamesMatr("mi",cPt2di(3,3)), {"val"} );
           }

           template <typename tUk>
                       std::vector<tUk> formula
                       (
                          const std::vector<tUk> & aVUk,
                          const std::vector<tUk> & aVObs
                       ) const
           {
               cPoseF<tUk>  aPoseInstr(aVUk,0,aVObs,0,true);
               cPtxd<tUk,3> aP_to = VtoP3(aVUk,6);
               auto       val = aVObs[9];
               cPtxd<tUk,3>  aP_to_instr = aPoseInstr.Inverse().Value(aP_to);

               auto dist = Norm2(aP_to_instr);
               return {  dist - val };
           }
};



class cFormulaTopoDX
{
      public :

           std::string FormulaName() const { return "TopoDX";}

           std::vector<std::string>  VNamesUnknowns()  const
           {
                //  Instrument pose with 6 unknowns : 3 for centers, 3 for axiator
               // target pose with 3 unknowns : 3 for center
                return  Append(NamesPose("Ci","Wi"),NamesP3("P_to"));
           }

           std::vector<std::string>    VNamesObs() const
           {
                // for the instrument pose, the 3x3 current rotation matrix as "observation/context"
                // and the measure value
                return  Append(NamesMatr("mi",cPt2di(3,3)), {"val"} );
           }

           template <typename tUk>
                       std::vector<tUk> formula
                       (
                          const std::vector<tUk> & aVUk,
                          const std::vector<tUk> & aVObs
                       ) const
           {
               cPoseF<tUk>  aPoseInstr(aVUk,0,aVObs,0,true);
               cPtxd<tUk,3> aP_to = VtoP3(aVUk,6);
               auto       val = aVObs[9];
               cPtxd<tUk,3>  aP_to_instr = aPoseInstr.Inverse().Value(aP_to);

               return {  aP_to_instr.x() - val };
           }
};


class cFormulaTopoDY
{
      public :

           std::string FormulaName() const { return "TopoDY";}

           std::vector<std::string>  VNamesUnknowns()  const
           {
                //  Instrument pose with 6 unknowns : 3 for centers, 3 for axiator
               // target pose with 3 unknowns : 3 for center
                return  Append(NamesPose("Ci","Wi"),NamesP3("P_to"));
           }

           std::vector<std::string>    VNamesObs() const
           {
                // for the instrument pose, the 3x3 current rotation matrix as "observation/context"
                // and the measure value
                return  Append(NamesMatr("mi",cPt2di(3,3)), {"val"} );
           }

           template <typename tUk>
                       std::vector<tUk> formula
                       (
                          const std::vector<tUk> & aVUk,
                          const std::vector<tUk> & aVObs
                       ) const
           {
               cPoseF<tUk>  aPoseInstr(aVUk,0,aVObs,0,true);
               cPtxd<tUk,3> aP_to = VtoP3(aVUk,6);
               auto       val = aVObs[9];
               cPtxd<tUk,3>  aP_to_instr = aPoseInstr.Inverse().Value(aP_to);

               return {  aP_to_instr.y() - val };
           }
};


class cFormulaTopoDZ
{
      public :

           std::string FormulaName() const { return "TopoDZ";}

           std::vector<std::string>  VNamesUnknowns()  const
           {
                //  Instrument pose with 6 unknowns : 3 for centers, 3 for axiator
               // target pose with 3 unknowns : 3 for center
                return  Append(NamesPose("Ci","Wi"),NamesP3("P_to"));
           }

           std::vector<std::string>    VNamesObs() const
           {
                // for the instrument pose, the 3x3 current rotation matrix as "observation/context"
                // and the measure value
                return  Append(NamesMatr("mi",cPt2di(3,3)), {"val"} );
           }

           template <typename tUk>
                       std::vector<tUk> formula
                       (
                          const std::vector<tUk> & aVUk,
                          const std::vector<tUk> & aVObs
                       ) const
           {
               cPoseF<tUk>  aPoseInstr(aVUk,0,aVObs,0,true);
               cPtxd<tUk,3> aP_to = VtoP3(aVUk,6);
               auto       val = aVObs[9];
               cPtxd<tUk,3>  aP_to_instr = aPoseInstr.Inverse().Value(aP_to);

               return {  aP_to_instr.z() - val };
           }
};


};


#endif  // _FORMULA_TOPO_H_
