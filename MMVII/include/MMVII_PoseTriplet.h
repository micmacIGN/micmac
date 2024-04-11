#ifndef  _MMVII_POSE_TRIPLET_H_
#define  _MMVII_POSE_TRIPLET_H_

#include "MMVII_PCSens.h"

namespace MMVII
{

class cView;
class cTriplet;
class cTripletSet;

class cView : public cMemCheck
{
    public:
        typedef cIsometry3D<tREAL8>  tPose;
        cView(const tPose,const std::string);
        cView();

        const std::string & Name() {return mName;}
        const tPose & Pose() {return mPose;}

        void AddData(const cAuxAr2007&);

    private:
        std::string           mName;
        tPose                 mPose;
        //cPerspCamIntrCalib        * mCalib;

};
void AddData(const  cAuxAr2007 &,cView &aV) ;

class cTriplet : public cMemCheck
{
    public:
        typedef cIsometry3D<tREAL8>  tPose;
        cTriplet();

        cView& Pose(int aK) {return mPoses[aK];}
        const cView& Pose(int aK) const {return mPoses[aK];}

        std::vector<cView>& PVec() {return mPoses;}
        const std::vector<cView>& PVec() const {return mPoses;}


        const double & BH() const {return mBH;}
        double &       BH() {return mBH;}
        const double & Residual() const {return mResidual;}
        double &       Residual() {return mResidual;}

        void AddData(const cAuxAr2007&);

    private:
        std::vector<cView> mPoses;

        double             mBH;
        double             mResidual;



};
void AddData(const  cAuxAr2007 &,cTriplet &) ;

class cTripletSet : public cMemCheck
{
    public:
        cTripletSet();

        /// Standard interface to write the object
        void ToFile(const std::string&) const;
        /// Standard interface to create an object
        static cTripletSet * FromFile(const std::string &) ;

        void PushTriplet(cTriplet &);

        const std::vector<cTriplet>& Set() const {return mSet;}
        std::vector<cTriplet>& Set() {return mSet;}

        void SetName(std::string& aName) {mName=aName;}
        const std::string & Name() const {return mName;}

        void AddData(const  cAuxAr2007 &);

    private:
        std::string           mName;
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

