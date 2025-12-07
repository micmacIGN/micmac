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
                   // create the 4 formal poses, using unkonw (C/W) and obs, Use auto incremenation by passing adresses
#if (1)
                   size_t aK0Uk=0,aK0Obs=0;
                   cPoseF<tUk>  aPoseA(aVUk,&aK0Uk,aVObs,&aK0Obs);
                   cPoseF<tUk>  aPoseB(aVUk,&aK0Uk,aVObs,&aK0Obs);
                   cPoseF<tUk>  aPose1(aVUk,&aK0Uk,aVObs,&aK0Obs);
                   cPoseF<tUk>  aPose2(aVUk,&aK0Uk,aVObs,&aK0Obs);

                   MMVII_INTERNAL_ASSERT_always(aK0Uk==aVUk.size(),"SizeUk in cFormulaBlocRigid");
                   MMVII_INTERNAL_ASSERT_always(aK0Obs==aVObs.size(),"SizeUk in cFormulaBlocRigid");
 #else
                   cPoseF<tUk>  aPoseA(aVUk,0*NbUk,aVObs,0*NbObs,true);
                  cPoseF<tUk>  aPoseB(aVUk,1*NbUk,aVObs,1*NbObs,true);
                  cPoseF<tUk>  aPose1(aVUk,2*NbUk,aVObs,2*NbObs,true);
                  cPoseF<tUk>  aPose2(aVUk,3*NbUk,aVObs,3*NbObs,true);
#endif
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
#if (1)
                   // Use new auto incr func
                   size_t aK0Uk=0,aK0Obs=0;
                   cPoseF<tUk>  aPoseA(aVUk,&aK0Uk,aVObs,&aK0Obs);
                   cPoseF<tUk>  aPose1(aVObs,&aK0Obs);

                   MMVII_INTERNAL_ASSERT_always(aK0Uk==aVUk.size(),"SizeUk in cFormulaRattBRExist");
                   MMVII_INTERNAL_ASSERT_always(aK0Obs==aVObs.size(),"SizeUk in cFormulaRattBRExist");
#else
                   cPoseF<tUk>  aPoseA(aVUk,0,aVObs,0,true);
                   cPoseF<tUk>  aPose1(aVObs,9,aVObs,12,false);
#endif

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


/**
 * @brief The cFormulaClino class
 *
 * It's not natural to add a pose when only the rotation is used, it's due to a bug in pose class
 * when we extract the rotation unknown (it does not handle correcly it indexe). Waiting for a correction
 * we have made this quick and dirty contournemnt that require a pose.
 */

class cFormulaClino
{
      public :

           std::string FormulaName() const { return "Clino_" + ToStr(mD0Corr) + "_"+ToStr(mD1Corr);}


           std::vector<std::string>  VNamesUnknowns()  const
           {
               // vector for correction of angular
                std::vector<std::string> aVCor;
                for (int aD=mD0Corr; aD<=mD1Corr ; aD++)
                    aVCor.push_back("DCorrAng_"+ToStr(aD));

                return  Append
                        (
                            NamesPose("Center","Omega"),
                            NamesP2("DuDvClino") ,
                            NamesP2("DuDvVert") ,
                            aVCor
                        );
                ;
           }

           std::vector<std::string>    VNamesObs() const
           {
               // observation is made :  3 point for coding normalized verctor +
               // matrix for rotation linearization + value of the angle
                return  Append
                        (
                            NamesMatr("m1",cPt2di(3,3)),
                            NamesObsP3Norm("Clino"),
                            NamesObsP3Norm("Vert"),
                            {"ValueTeta"}
                      );
           };

           template <typename tUk>
                       std::vector<tUk> formula
                       (
                          const std::vector<tUk> & aVUk,
                          const std::vector<tUk> & aVObs
                       ) const
           {
                   //        IndexAutoIncr(&anInd,3)
                   size_t aK0Uk=0,aK0Obs=0;
                   // rotation that is linked to clino, can be a camera rotation Cam->Word
                   cPoseF<tUk>  aRotC2M(aVUk,&aK0Uk,aVObs,&aK0Obs);

                   cP3dNorm<tUk> aClinoC(aVUk,&aK0Uk,aVObs,&aK0Obs);  //
                   cP3dNorm<tUk> aVert(aVUk,&aK0Uk,aVObs,&aK0Obs);  //


                   tUk aSinT = sin(aVObs.at(aK0Obs++));

                   MMVII_INTERNAL_ASSERT_always(aK0Obs==aVObs.size(),"SizeUk in cFormulaClino");

                   tUk aSumTeta = CreateCste(0.0,aVUk.at(0));
                   for (int aD=mD0Corr ; aD<=mD1Corr ; aD++)
                   {
                       aSumTeta = aSumTeta + aVUk.at(aK0Uk++) * powI(aSinT,aD);
                   }
                   MMVII_INTERNAL_ASSERT_always(aK0Uk==aVUk.size(),"SizeUk in cFormulaClino");

                   cPtxd<tUk,3> aClinoM =  aRotC2M.ValueVect(aClinoC.CurPt());
                   tUk   aSinus = Scal(aClinoM,aVert.CurPt());

                   return  {aSinus - aSumTeta} ;
            }

            cFormulaClino(int aD0Corr,int aD1Corr) :
                mD0Corr (aD0Corr),
                mD1Corr (aD1Corr)
            {
            }

         private :
            int  mD0Corr;
            int  mD1Corr;

};

class cFormulaVNormOrthog
{
      public :

           std::string FormulaName() const { return "VNormOrthog";}


           std::vector<std::string>  VNamesUnknowns()  const
           {
                return  Append
                        (
                            NamesP2("DuDv1") ,
                            NamesP2("DuDv2")
                        );
                ;
           }

           std::vector<std::string>    VNamesObs() const
           {

                return  Append
                        (
                            NamesObsP3Norm("_P1"),
                            NamesObsP3Norm("_P2")
                        );
           };

           template <typename tUk>
                       std::vector<tUk> formula
                       (
                          const std::vector<tUk> & aVUk,
                          const std::vector<tUk> & aVObs
                       ) const
           {
                   //        IndexAutoIncr(&anInd,3)
                   size_t aK0Uk=0,aK0Obs=0;
                   cP3dNorm<tUk> aVec1(aVUk,&aK0Uk,aVObs,&aK0Obs);  //
                   cP3dNorm<tUk> aVec2(aVUk,&aK0Uk,aVObs,&aK0Obs);  //

                   MMVII_INTERNAL_ASSERT_always(aK0Uk==aVUk.size(),"SizeUk in cFormulaVNormOrthog");
                   MMVII_INTERNAL_ASSERT_always(aK0Obs==aVObs.size(),"SizeUk in cFormulaVNormOrthog");

                   tUk aScal = Scal(aVec1.CurPt(),aVec2.CurPt());

                   return  {aScal} ;
            }


         private :
};

};


#endif  // _FORMULA_BLOCKRIGID_H_
