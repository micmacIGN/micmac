#ifndef _FORMULA_BLOCKRIGID_H_
#define _FORMULA_BLOCKRIGID_H_

#include "MMVII_PhgrDist.h"
#include "SymbDer/SymbolicDerivatives.h"
#include "SymbDer/SymbDer_MACRO.h"
#include "ComonHeaderSymb.h"

//  RIGIDBLOC  : all file is concerned

namespace MMVII
{
using namespace NS_SymbolicDerivative;

/**
 * @brief The cFormulaBlocRigid class, generate the equation for "rigid-pose".
 *
 * Class for generating the bloc rigid equation :
 *
 *   Let  PA and PB be two pose acquired with the rigid camera bloc,
 *   Let P1 and P2 be the, "unknown", value of rigid bloc, we note {CA;RA} 
 *   the pose of center CA & rotation RA.
 *
 *      PA = {CA ; RA} <->  PB = {CB ; RB}      
 *      P1 = {C1 ; R1} <->  P2 = {C2 ; R2}
 *
 *    Let  PA~PB  be the pose of  B relatively to A , idem P1~P2 , we have :
 *
 *           PA~PB = P1~ P2
 *
 *    Now let compute PA~PB,  let CamA,CamB be the coordinate of a point W in camera A and B 
 *     
 *            PA (CamA) =   CA +  RA * CamA  = W
 *            PA-1 PB (CamB) = PA-1(W) = CamA
 *
 *    For computation of compositio we have :
 *
 *        {CA;RA}* {CB;RB} (X) = {CA;RA} (CB+RB*X) = (CA+RA*CB + RA*RB*X)
 *        {CA;RA}* {CB;RB} = {CA+RA*CB ; RA*RB}
 *
 *    For computation of inverse  :
 *
 *         -   PA (X) =  CA +  RA *X = Y, then  X = -tRA CA + tRA Y
 *         -   PA-1 =  {-tRA CA ; tRA}
 *
 *            
 *    We have then :
 *
 *          PA~PB = PA-1 * PB = {- tRA CA; tRA} * {CB ; RB} = {- tRA CA +tRA*CB, tRA*RB}
 *
 *     And the equations :
 *       PA~PB = {tRA(CB-CA); tRA*RB}
 *
 *     Then equation is : 
 *
 *       {tRA(CB-CA); tRA*RB} = {tR1(C2-C1) ; tR1 * R2}
 *
 */


class cFormulaBlocRigid
{
      public :

           std::string FormulaName() const { return "BlocRigid";}

           std::vector<std::string>  VNamesUnknowns()  const
           {
               //  We have 4 pose  A,B,1 en 2;  each has 6 unknown : 3 for centers,3 for axiator
               //  We could write  explicitely a 24 size vector like {"CxA","CyA","CzA","WxA" .....,"Wy2","Wz2"}
               //  We prefer to  use the facility "NamesPose"

                return  Append
                        (
                            NamesPose("CA","WA"),
                            NamesPose("CB","WB"),
                            NamesPose("C1","W1"),
                            NamesPose("C2","W2")
                        );
              // Append NamesPose("CA","WA")
          }

           std::vector<std::string>    VNamesObs() const
           {
                // we have 4 pose, and for each we have the  3x3 current rotation matrix as "observation/context"
                // we coul wite explicitely the  36 size vector {"mA_00","mA_10" ... "m1_12","m_22"}
                // we prefer to use the facility ",NamesMatr"
                return  Append(NamesMatr("mA",cPt2di(3,3)),NamesMatr("mB",cPt2di(3,3)),NamesMatr("m1",cPt2di(3,3)),NamesMatr("m2",cPt2di(3,3)));
                //  Append   NamesMatr("mA",cPt2di(3,3)),
           }
          static constexpr size_t  NbUk =6;
          static constexpr size_t  NbObs=9;

          template <typename tUk>
                       std::vector<tUk> formula
                       (
                          const std::vector<tUk> & aVUk,
                          const std::vector<tUk> & aVObs
                       ) const
           {
                   // create the 4 formal poses, using unkonw (C/W) and obs
                   cPoseF<tUk>  aPoseA(aVUk,0*NbUk,aVObs,0*NbObs,true);
                   cPoseF<tUk>  aPoseB(aVUk,1*NbUk,aVObs,1*NbObs,true);
                   cPoseF<tUk>  aPose1(aVUk,2*NbUk,aVObs,2*NbObs,true);
                   cPoseF<tUk>  aPose2(aVUk,3*NbUk,aVObs,3*NbObs,true);

                   // compute relative poses B/A and 2/1
                   cPoseF<tUk>  aRelAB = aPoseA.PoseRel(aPoseB);
                   cPoseF<tUk>  aRel12 = aPose1.PoseRel(aPose2);

                   // compute difference of centers and matrices
                    cPtxd<tUk,3>  aDeltaC = aRelAB.mCenter-aRel12.mCenter;
                    cMatF<tUk>    aDeltaR = aRelAB.IJK()-aRel12.IJK();

                   //  return the differences as a size-12 vector
                   return Append
                          (
                               ToVect(aDeltaC),
                               aDeltaR.ToVect()
                          );
            }
      private :
};

/**
 * @brief The cFormulaRattBRExist class for generating the conservation of a pose to a known value
 *
 * It is used in Block Rigid because the relative pose PA being an unknown we just have to write :
 *      PA = P1
 *  where P1 is the known value
 */

class cFormulaRattBRExist
{
      public :

        std::string FormulaName() const { return "BlocRigid_RE";}

           std::vector<std::string>  VNamesUnknowns()  const
           {
                //  We have 4 pose  A,B,1 en 2;  each has 6 unknown : 3 for centers,3 for axiator
                //  We could write  explicitely a 24 size vector like {"CxA","CyA","CzA","WxA" .....,"Wy2","Wz2"}
                //  We prefer to  use the facility "NamesPose"
                return  NamesPose("CA","WA");
           }

           std::vector<std::string>    VNamesObs() const
           {
                return  Append
                        (
                               NamesMatr("mA",cPt2di(3,3)),
                               Append(  NamesP3("C1"),  NamesMatr("m1",cPt2di(3,3)))
                        );
           };
       template <typename tUk>
                       std::vector<tUk> formula
                       (
                          const std::vector<tUk> & aVUk,
                          const std::vector<tUk> & aVObs
                       ) const
           {
                   cPoseF<tUk>  aPoseA(aVUk,0,aVObs,0,true);
                   cPoseF<tUk>  aPose1(aVObs,9,aVObs,12,false);

                    cPtxd<tUk,3>  aDeltaC = aPoseA.mCenter - aPose1.mCenter;
                    cMatF<tUk>    aDeltaR = aPoseA.IJK()- aPose1.IJK();

           // ...
           // extract PoseA,PoseB,pose1, pose2

           // compute pose rel B to A,   pose rel 2 to 1
           // compute the difference


           return Append(ToVect(aDeltaC),aDeltaR.ToVect());

           //  cPoseF<tUk>  aPose1(aVUk,2*NbUk,aVObs,2*NbObs);
                   //  cPoseF<tUk>  aRelAB = aPoseA.PoseRel(aPoseB);
           // (ToVect(aDeltaC),aDeltaM.ToVect()
       }
};



class cFormulaClino
{
      public :

           std::string FormulaName() const { return "Clino_" + ToStr(mD0Corr) + "_"+ToStr(mD1Corr);}


           std::vector<std::string>  VNamesUnknowns()  const
           {
                std::vector<std::string> aVCor;
                for (int aD=mD0Corr; aD<=mD1Corr ; aD++)
                    if (aD!=1)
                        aVCor.push_back("DCorrAng_"+ToStr(aD));

                return  Append
                        (
                            NamesPose("Du","Dv") ,
                            aVCor
                        );
                ;
           }

           std::vector<std::string>    VNamesObs() const
           {
                return  Append
                        (
                            Append
                            (
                               NamesP3("PNom"),
                               NamesP3("DirU"),
                               NamesP3("DirV")
                            ),
                            NamesMatr("m1",cPt2di(3,3)),
                            {"VTeta"}
                      );
           };
           /*
       template <typename tUk>
                       std::vector<tUk> formula
                       (
                          const std::vector<tUk> & aVUk,
                          const std::vector<tUk> & aVObs
                       ) const
           {
                   cPoseF<tUk>  aPoseA(aVUk,0,aVObs,0,true);
                   cPoseF<tUk>  aPose1(aVObs,9,aVObs,12,false);

                    cPtxd<tUk,3>  aDeltaC = aPoseA.mCenter - aPose1.mCenter;
                    cMatF<tUk>    aDeltaR = aPoseA.IJK()- aPose1.IJK();

           // ...
           // extract PoseA,PoseB,pose1, pose2

           // compute pose rel B to A,   pose rel 2 to 1
           // compute the difference


           return Append(ToVect(aDeltaC),aDeltaR.ToVect());

           //  cPoseF<tUk>  aPose1(aVUk,2*NbUk,aVObs,2*NbObs);
                   //  cPoseF<tUk>  aRelAB = aPoseA.PoseRel(aPoseB);
           // (ToVect(aDeltaC),aDeltaM.ToVect()
       }
       */

        int  mD0Corr;
        int  mD1Corr;

};

};


#endif  // _FORMULA_BLOCKRIGID_H_
