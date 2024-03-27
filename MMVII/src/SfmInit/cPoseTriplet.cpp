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



}; // MMVII




