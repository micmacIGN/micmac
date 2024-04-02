#include "MMVII_PoseTriplet.h"
#include "MMVII_Tpl_Images.h"


namespace MMVII
{

class cView; ///< class storing a view of a triplet
class cTriplet; ///< a class storing three views

typedef cIsometry3D<tREAL8>  tPose;

   /* ********************************************************** */
   /*                                                            */
   /*                        cView                               */
   /*                                                            */
   /* ********************************************************** */

cView::cView(const tPose aPose) :
    mPose(aPose)
{
};

    /* ********************************************************** */
    /*                                                            */
    /*                        cTriplet                            */
    /*                                                            */
    /* ********************************************************** */
cTriplet::cTriplet()
{
    mPoses.push_back(tPose::Identity());
    mPoses.push_back(tPose::Identity());
    mPoses.push_back(tPose::Identity());
}

void cTriplet::AddData(const cAuxAr2007 &anAuxInit)
{
    cAuxAr2007 anAux("Triplet",anAuxInit);
    //
    // Save the relative poses
    // Pose1, Pose 21, Pose31
    //
    MMVII::AddData(cAuxAr2007("Pose1",anAux),mPoses[0]);
    MMVII::AddData(cAuxAr2007("Pose21",anAux),mPoses[1]);
    MMVII::AddData(cAuxAr2007("Pose31",anAux),mPoses[2]);

}

void AddData(const cAuxAr2007& anAux,cTriplet& aTri)
{
    aTri.AddData(anAux);
}
    /* ********************************************************** */
    /*                                                            */
    /*                        cTripletSet                         */
    /*                                                            */
    /* ********************************************************** */

cTripletSet::cTripletSet()
{}

void cTripletSet::PushTriplet(cTriplet &aTri)
{
    mSet.push_back(aTri);
}

void cTripletSet::ToFile(const std::string &aName) const
{
    SaveInFile(this->mSet,aName);
}

void cTripletSet::AddData(const  cAuxAr2007 & anAuxInit)
{
     cAuxAr2007 anAux("TripletSet",anAuxInit);
     // ...
     // Put the data in  tag "cTripletSet"

     // Add data for
     //    mName
     //    ...
     //
     //MMVII::AddData(cAuxAr2007("Name",anAux),mName);
     MMVII::AddData(cAuxAr2007("Sets",anAux),mSet);


}

void AddData(const  cAuxAr2007 & anAux,cTripletSet & aSet)
{
     aSet.AddData(anAux);
}

}; // MMVII




