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


class cFormulaTopoDH
{
      public :

           std::string FormulaName() const { return "TopoDH";}

           std::vector<std::string>  VNamesUnknowns()  const
           {
               //  Instrument pose with 3 unknowns : 3 for center
              // target pose with 3 unknowns : 3 for center
               return  Append(NamesP3("P_from"),NamesP3("P_to"));
           }

           std::vector<std::string>    VNamesObs() const
           {
                // RTL to GeoC transfo, as matrix + translation
                // GeoG phi (in rad) and M (=a*sqrt(1-e*e*sin(phi)*sin(phi))) for each point
                // measured value
                return  Append(NamesMatr("M_RTL",cPt2di(3,3)), NamesP3("T_RTL"),
                               {"Phi_from", "M_from", "Phi_to", "M_to", "val"});
           }

           template <typename tUk>
                       std::vector<tUk> formula
                       (
                          const std::vector<tUk> & aVUk,
                          const std::vector<tUk> & aVObs
                       ) const
           {

               cMatF<tUk> M_RTL = cMatF(3, 3, aVObs, 0);
               auto T_RTL = VtoP3(aVObs,9);
               auto Phi_from = aVObs[12];
               auto M_from = aVObs[13];
               auto Phi_to = aVObs[14];
               auto M_to = aVObs[15];
               auto val = aVObs[16];

               // convert points to GeoC
               auto aP_from_RTL = VtoP3(aVUk,0);
               auto aP_from_GeoC =  M_RTL * aP_from_RTL + T_RTL;
               auto aP_to_RTL = VtoP3(aVUk,3);
               auto aP_to_GeoC =  M_RTL * aP_to_RTL + T_RTL;

               // convert GeoC to GeoG
               /* // Iterative formula
               auto p_from = sqrt(aP_from_GeoC.x()*aP_from_GeoC.x()
                             +aP_from_GeoC.y()*aP_from_GeoC.y());
               auto h_from = p_from/cos(Phi_from) - N_from;

               auto p_to = sqrt(aP_to_GeoC.x()*aP_to_GeoC.x()
                             +aP_to_GeoC.y()*aP_to_GeoC.y());
               auto h_to = p_to/cos(Phi_to) - N_to; */

               // Bowring, 1985, The accuracy of geodetic latitude and height equations
               // https://geodesie.ign.fr/contenu/fichiers/documentation/pedagogiques/TransformationsCoordonneesGeodesiques.pdf
               auto p_from = sqrt(aP_from_GeoC.x()*aP_from_GeoC.x()
                             +aP_from_GeoC.y()*aP_from_GeoC.y());
               auto h_from = p_from*cos(Phi_from) + aP_from_GeoC.z()*sin(Phi_from) - M_from;

               auto p_to = sqrt(aP_to_GeoC.x()*aP_to_GeoC.x()
                             +aP_to_GeoC.y()*aP_to_GeoC.y());
               auto h_to = p_to*cos(Phi_to) + aP_to_GeoC.z()*sin(Phi_to) - M_to;

               auto dH = h_to - h_from;

               /*
               SymbPrint(aP_from_GeoC.x(),"aP_from_GeoC.x");
               SymbPrint(aP_from_GeoC.y(),"aP_from_GeoC.y");
               SymbPrint(aP_from_GeoC.z(),"aP_from_GeoC.z");
               SymbPrint(aP_to_GeoC.x(),"aP_to_GeoC.x");
               SymbPrint(aP_to_GeoC.y(),"aP_to_GeoC.y");
               SymbPrint(aP_to_GeoC.z(),"aP_to_GeoC.z");
               SymbPrint(h_from,"h_from");
               SymbPrint(h_to,"h_to");
               */

               return {  dH - val };
           }
};


};


#endif  // _FORMULA_TOPO_H_
