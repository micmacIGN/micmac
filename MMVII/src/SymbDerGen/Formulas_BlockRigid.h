#ifndef _FORMULA_BLOCKRIGID_H_
#define _FORMULA_BLOCKRIGID_H_

#include "SymbDer/SymbolicDerivatives.h"
#include "SymbDer/SymbDer_MACRO.h"
#include "ComonHeaderSymb.h"

//  RIGIDBLOC  : all file is concerned


/**  Class for generating the bloc rigid equation :
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
using namespace NS_SymbolicDerivative;

namespace MMVII
{

/** this class represent a Pose on  forumla (or real if necessary)
 *    It contains a Center and the rotation Matrix IJK
 */
template <class Type> class cPoseF
{
     public :

	cPoseF(const cPtxd<Type,3> & aCenter,const cMatF<Type> & aMat) :
             mCenter  (aCenter),
             mIJK     (aMat)
	{
	}


        cPoseF(const std::vector<Type> &  aVecUk,size_t aK0Uk,const std::vector<Type> &  aVecObs,size_t aK0Obs) :
            cPoseF<Type>
	    (
                 VtoP3(aVecUk,aK0Uk),
	     //  The matrix is Current matrix *  Axiator(-W) , the "-" in omega comming from initial convention
	     //  See  cPoseWithUK::OnUpdate()  &&  cEqColinearityCamPPC::formula
	         cMatF<Type>(3,3,aVecObs,aK0Obs)  *  cMatF<Type>::MatAxiator(-VtoP3(aVecUk,aK0Uk+3))
	    )
        {
        }

	/// A pose being considered a the, isometric, mapinc X->Tr+R*X, return pose corresponding to inverse mapping
	cPoseF<Type> Inverse() const
	{
             //  PA-1 =  {-tRA CA ; tRA}
	     cMatF<Type> aMatInv = mIJK.Transpose();
             return cPoseF<Type>(- (aMatInv* mCenter),aMatInv);
	}

	/// A pose being considered as mapping, return their composition
	cPoseF<Type> operator * (const cPoseF<Type> & aP2) const
	{
	    const cPoseF<Type> & aP1 = *this;

             //   {CA;RA}* {CB;RB} = {CA+RA*CB ; RA*RB}
            return cPoseF<Type>
                   (
		        aP1.mCenter + aP1.mIJK*mCenter,
                        aP1.mIJK * aP2.mIJK
                   );
	}

        cPoseF<Type>  PoseRel(const cPoseF<Type> & aP2) const 
        { 
            //  PA~PB = PA-1 * PB 
            return Inverse() * aP2; 
        }

        cPtxd<Type,3>  mCenter;
        cMatF<Type>    mIJK;
};

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
	   }

           std::vector<std::string>    VNamesObs() const
           {
                // we have 4 pose, and for each we have the  3x3 current rotation matrix as "observation/context"
                // we coul wite explicitely the  36 size vector {"mA_00","mA_10" ... "m1_12","m_22"}
                // we prefer to use the facility ",NamesMatr"
                return  Append
			(
			     NamesMatr("mA",cPt2di(3,3)),
			     NamesMatr("mB",cPt2di(3,3)),
			     NamesMatr("m1",cPt2di(3,3)),
			     NamesMatr("m2",cPt2di(3,3))
			);
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
                   cPoseF<tUk>  aPoseA(aVUk,0*NbUk,aVObs,0*NbObs);
                   cPoseF<tUk>  aPoseB(aVUk,1*NbUk,aVObs,1*NbObs);

                   cPoseF<tUk>  aRelAB = aPoseA.PoseRel(aPoseB);

                   cPoseF<tUk>  aPose1(aVUk,2*NbUk,aVObs,2*NbObs);
                   cPoseF<tUk>  aPose2(aVUk,3*NbUk,aVObs,3*NbObs);
                   cPoseF<tUk>  aRel12 = aPose1.PoseRel(aPose2);

		   // FakeUseIt(aRelAB);
		   // FakeUseIt(aRel12);

		   cPtxd<tUk,3>  aDeltaC =  aRelAB.mCenter - aRel12.mCenter;

		   cMatF<tUk> aDeltaM = aRelAB.mIJK - aRel12.mIJK;

		   return Append(ToVect(aDeltaC),aDeltaM.ToVect());
	   }



      private :
};


};


#endif  // _FORMULA_BLOCKRIGID_H_
