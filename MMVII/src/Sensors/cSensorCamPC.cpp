#include "include/MMVII_all.h"


/**
   \file cSensorCamPC.cpp  

   \brief file for implementing class sensor for perspective central
*/


namespace MMVII
{

/* ******************************************************* */
/*                                                         */
/*                   cSensorCamPC                          */
/*                                                         */
/* ******************************************************* */

cSensorCamPC::cSensorCamPC(const tPose & aPose,cPerspCamIntrCalib * aCalib) :
   cSensorImage  ("tttttyuiiiiiiiiiiii"),
   mPose         (aPose),
   mCalib        (aCalib),
   mOmega        (0,0,0)
{
}

void cSensorCamPC::PutUknowsInSetInterval()
{
    mSetInterv->AddOneInterv(mPose.Tr());
    mSetInterv->AddOneInterv(mOmega);
}

cPt2dr cSensorCamPC::Ground2Image(const cPt3dr & aP) const
{
        //  mPose(0,0,0) = Center, then mPose Cam->Word, then we use Inverse, BTW Inverse is as efficient as direct
     return mCalib->Value(mPose.Inverse(aP));
}

size_t  cSensorCamPC::NumXCenter() const
{
   return IndOfVal(&(mPose.Tr().x()));
}



const cPt3dr & cSensorCamPC::Center() const {return mPose.Tr();}
const cPt3dr & cSensorCamPC::Omega()  const {return mOmega;}
cPt3dr cSensorCamPC::AxeI()   const {return mPose.Rot().AxeI();}
cPt3dr cSensorCamPC::AxeJ()   const {return mPose.Rot().AxeI();}
cPt3dr cSensorCamPC::AxeK()   const {return mPose.Rot().AxeJ();}
const cIsometry3D<tREAL8> & cSensorCamPC::Pose() const {return mPose;}

/*   Let R be the rotation of pose  P=(C,P= : Cam-> Word, what is optimized in colinearity for a ground point G
 *   is Word->Cam  :
 *
 *          tR(G-C)
 *
 * So the optimal rotation R' with get satisfy the equation :
 *
 *          (1+^W) tR0 =tR'   
 *
 * The we have the formula :
 *
 *          --->   R'=R0 t(1+^W)
 *
 *  And note that for axiators :
 *
 *       t Axiator(W) = Axiator(-W)
 *
 */

void cSensorCamPC::OnUpdate()
{
	//  used above formula to modify  rotation
     mPose.SetRotation(mPose.Rot() * cRotation3D<tREAL8>::RotFromAxiator(-mOmega));
        // now this have modify rotation, the "delta" is void :
     mOmega = cPt3dr(0,0,0);
}





}; // MMVII

