#ifndef _FORMULA_TOPO_H_
#define _FORMULA_TOPO_H_

#include "SymbDer/SymbolicDerivatives.h"
#include "SymbDer/SymbDer_MACRO.h"
#include "ComonHeaderSymb.h"
#include "MMVII_PhgrDist.h"

/**  Class for generating topo survey equations:
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
                //  Instrument pose with 6 unknowns : 3 for center, 3 for axiator
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
               cPoseF<tUk>  aPoseInstr2RTL(aVUk,0,aVObs,0,true);
               cPtxd<tUk,3> aP_to = VtoP3(aVUk,6);
               auto       val = aVObs[9];
               cPtxd<tUk,3>  aP_to_instr = aPoseInstr2RTL.Inverse().Value(aP_to);

               auto az = ATan2( aP_to_instr.x(), aP_to_instr.y() );

               return {  DiffAngMod(az, val) };
           }
};


class cFormulaTopoZen
{
      public :

           std::string FormulaName() const { return "TopoZen";}

           std::vector<std::string>  VNamesUnknowns()  const
           {
                //  Instrument pose with 6 unknowns : 3 for center, 3 for axiator
               // target pose with 3 unknowns : 3 for center
                return  Append(NamesPose("Ci","Wi"),NamesP3("P_to"));
           }

           std::vector<std::string>    VNamesObs() const
           {
                // for the instrument pose, the 3x3 current rotation matrix as "observation/context"
                // refraction correction
                // and the measure value
                return  Append(NamesMatr("mi",cPt2di(3,3)), {"ref_cor", "val"} );
           }

           template <typename tUk>
                       std::vector<tUk> formula
                       (
                          const std::vector<tUk> & aVUk,
                          const std::vector<tUk> & aVObs
                       ) const
           {
               cPoseF<tUk>  aPoseInstr2RTL(aVUk,0,aVObs,0,true);
               cPtxd<tUk,3> aP_to = VtoP3(aVUk,6);
               auto  ref_cor = aVObs[9];
               auto      val = aVObs[10];
               cPtxd<tUk,3>  aP_to_instr = aPoseInstr2RTL.Inverse().Value(aP_to);

               auto   dist_hz =  Norm2(  cPtxd<tUk,2>(aP_to_instr.x(), aP_to_instr.y() ) );

               auto   zen = ATan2( dist_hz, aP_to_instr.z() ) - ref_cor;

               return {   DiffAngMod(zen, val) };
           }
};




class cFormulaTopoDX
{
      public :

           std::string FormulaName() const { return "TopoDX";}

           std::vector<std::string>  VNamesUnknowns()  const
           {
                //  Instrument pose with 6 unknowns : 3 for center, 3 for axiator
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
               cPoseF<tUk>  aPoseInstr2RTL(aVUk,0,aVObs,0,true);
               cPtxd<tUk,3> aP_to = VtoP3(aVUk,6);
               auto       val = aVObs[9];
               cPtxd<tUk,3>  aP_to_instr = aPoseInstr2RTL.Inverse().Value(aP_to);

               return {  aP_to_instr.x() - val };
           }
};


class cFormulaTopoDY
{
      public :

           std::string FormulaName() const { return "TopoDY";}

           std::vector<std::string>  VNamesUnknowns()  const
           {
                //  Instrument pose with 6 unknowns : 3 for center, 3 for axiator
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
               cPoseF<tUk>  aPoseInstr2RTL(aVUk,0,aVObs,0,true);
               cPtxd<tUk,3> aP_to = VtoP3(aVUk,6);
               auto       val = aVObs[9];
               cPtxd<tUk,3>  aP_to_instr = aPoseInstr2RTL.Inverse().Value(aP_to);

               return {  aP_to_instr.y() - val };
           }
};


class cFormulaTopoDZ
{
      public :

           std::string FormulaName() const { return "TopoDZ";}

           std::vector<std::string>  VNamesUnknowns()  const
           {
                //  Instrument pose with 6 unknowns : 3 for center, 3 for axiator
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
               cPoseF<tUk>  aPoseInstr2RTL(aVUk,0,aVObs,0,true);
               cPtxd<tUk,3> aP_to = VtoP3(aVUk,6);
               auto       val = aVObs[9];
               cPtxd<tUk,3>  aP_to_instr = aPoseInstr2RTL.Inverse().Value(aP_to);

               return {  aP_to_instr.z() - val };
           }
};


// -------------------------------------------

class cFormulaTopoDist
{
      public :

           std::string FormulaName() const { return "TopoDist";}

           std::vector<std::string>  VNamesUnknowns()  const
           {
                //  Instrument pose with 3 unknowns : 3 for center
               // target pose with 3 unknowns : 3 for center
                return  Append(NamesP3("P_from"),NamesP3("P_to"));
           }

           std::vector<std::string>    VNamesObs() const
           {
                // the measure value
                return  {"val"} ;
           }

           template <typename tUk>
                       std::vector<tUk> formula
                       (
                          const std::vector<tUk> & aVUk,
                          const std::vector<tUk> & aVObs
                       ) const
           {
               cPtxd<tUk,3> aP_from = VtoP3(aVUk,0);
               cPtxd<tUk,3> aP_to = VtoP3(aVUk,3);
               auto       val = aVObs[0];

               auto dist = Norm2(aP_to - aP_from);
               return {  dist - val };
           }
};


template <typename tUk>
tUk geoc2h(cPtxd<tUk,3> aP, tUk a, tUk e2) //< Compute ellips height from geocentric coords
{
    auto f = 1.-sqrt(1.-e2);
    auto R = Norm2(aP);
    // auto lambda = ATan2(aP.y(), aP.x());
    auto R2d = sqrt(aP.x()*aP.x()+aP.y()*aP.y());
    auto mu = ATan2(aP.z()*((1.-f)+(e2*a)/R), R2d);
    auto phi = ATan2(aP.z()*(1.-f)+e2*a*sin(mu)*sin(mu)*sin(mu),
                     (1.-f)*(R2d-e2*a*cos(mu)*cos(mu)*cos(mu)));
    auto h = (R2d*cos(phi)) + (aP.z()*sin(phi)) - (a*sqrt(1.-e2*sin(phi)*sin(phi)));
    return h;
}

class cFormulaTopoDH
{
      public :

           std::string FormulaName() const { return "TopoDH";}

           std::vector<std::string>  VNamesUnknowns()  const
           {
                // origin pt : 3 for center
                // target pt with 3 unknowns : 3 for center
                return  Append(NamesP3("P_from"),NamesP3("P_to"));
           }

           std::vector<std::string>    VNamesObs() const
           {
                // RTL2geoc rot + tr
                // a and e2 from ellipsoid
                // and the measure value
                return  Append(NamesMatr("RTL2Geoc_R",cPt2di(3,3)),
                               NamesP3("RTL2Geoc_T"),
                               {"a", "e2", "val"});
           }

           template <typename tUk>
                       std::vector<tUk> formula
                       (
                          const std::vector<tUk> & aVUk,
                          const std::vector<tUk> & aVObs
                       ) const
           {
               auto aP_from_RTL = VtoP3(aVUk,0);
               auto aP_to_RTL   = VtoP3(aVUk,3);
               auto           a = aVObs[12];
               auto          e2 = aVObs[13];
               auto         val = aVObs[14];
               cMatF<tUk> RTL_R = cMatF(3,3,aVObs, 0);
               auto       RTL_T = VtoP3(aVObs,9);
               cPoseF aRTL2Geoc(RTL_T, RTL_R);

               auto aP_from_geoc = aRTL2Geoc.Value(aP_from_RTL);
               auto aP_from_h = geoc2h(aP_from_geoc, a, e2);

               auto aP_to_geoc = aRTL2Geoc.Value(aP_to_RTL);
               auto aP_to_h = geoc2h(aP_to_geoc, a, e2);

               return {  aP_to_h - aP_from_h  - val };
           }
};


};


#endif  // _FORMULA_TOPO_H_
