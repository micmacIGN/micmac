#ifndef  _MMVII_POSE_TRIPLET_H_
#define  _MMVII_POSE_TRIPLET_H_

#include "MMVII_PCSens.h"

namespace MMVII
{

class cView
{
    public:
        typedef cIsometry3D<tREAL8>  tPose;
        cView(const tPose);

        const std::string & Name() {return mName;}
        const tPose & Pose() {return mPose;}

    private:
        const std::string           mName;
        tPose                       mPose;
        cPerspCamIntrCalib        * mCalib;

};

class cTriplet
{
    public:
        cTriplet();

    private:
};

};
#endif // _MMVII_POSE_TRIPLET_H_

