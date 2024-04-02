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
        typedef cIsometry3D<tREAL8>  tPose;
        cTriplet();

        tPose& Pose(int aK) {return mPoses[aK];}
        const tPose& Pose(int aK) const {return mPoses[aK];}

        void AddData(const cAuxAr2007&);

    private:
        std::vector<tPose> mPoses;
      /*  tPose mPose1;
        tPose mPose21;
        tPose mPose31;*/

};
void AddData(const  cAuxAr2007 &,cTripletSet &cTriplet) ;

class cTripletSet
{
    public:
        cTripletSet();

        void ToFile(const std::string&) const;

        void PushTriplet(cTriplet &);

        void AddData(const  cAuxAr2007 &);

    private:
        std::vector<cTriplet> mSet;

};

///  Global function with standard interface  required for serialization => just call member
void AddData(const  cAuxAr2007 &,cTripletSet &) ;

/*class cTripletLink
{
    public:

    private:
};*/


};
#endif // _MMVII_POSE_TRIPLET_H_

