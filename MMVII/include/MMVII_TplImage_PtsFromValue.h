#ifndef  _MMVII_TplImage_PtsFromValue_H_
#define  _MMVII_TplImage_PtsFromValue_H_

#include "MMVII_Image2D.h"

namespace MMVII
{


/** \file  MMVII_TplImage_PtsFromValue.h
    \brief    Get pts of image having a given value

*/

/*=======================================================*/
/*                                                       */
/*                 cGetPts_ImInterp_FromValue            */
/*                                                       */
/*=======================================================*/


/**
 *   Class for extracting a point in an image having a given value, take :
 *
 *       - the image !
 *       - the targeted value
 *       - the initial point
 *       - a direction
 *
 *    Return a point having a value close to the target one, assuming some model of interpolation
 *    (here bilinear)
*/


template <class Type> class cGetPts_ImInterp_FromValue
{
     public :
       cGetPts_ImInterp_FromValue 
       (
	    const cDataIm2D<Type> & aDIm,   ///< Image where we look
	    tREAL8 aVal,                    ///< Target value
	    tREAL8 aTol,                    ///< Tolerance to targeted value
	    cPt2dr aP0,                     ///< Initial point
	    const cPt2dr & aDir             ///< Direction of research
       ):
                mOk          (false),  // initial we have no succeded
                mDIm         (aDIm),
                mMaxIterInit (10),
                mMaxIterEnd  (20)
       {
          // [0]  first bracket the result , find a point P1 such that I(P0) > Val > I (P1) (or revers)
          tREAL8 aV0 = GetV(aP0);
          if (!CheckNoVal(aV0))  return;
          mP0IsSup = (aV0>=aVal);   // memorise if I0 > Val >I1 or I0<Val>I1

          cPt2dr aP1 = aP0 + aDir;
          double aV1 = GetV(aP1);
          if (!CheckNoVal(aV1))  return;

          int aNbIter=0;
          while ( (aV1>=aVal)==mP0IsSup )  // while bracketing if not reached
          {
               // advance one step further
                aV0 = aV1;
                aP0 = aP1;
                aP1 += aDir;
                aV1 = GetV(aP1);
                if (!CheckNoVal(aV1))  return;
                aNbIter++;
                if (aNbIter>mMaxIterInit) return;
          }

	  // now  refine interval by dicothomic cut
          tREAL8 aTol0 = std::abs(aV0-aVal);
          tREAL8 aTol1 = std::abs(aV1-aVal);
          bool  InInterv = true;
          aNbIter=0;
          while ((aTol0>aTol) && (aTol1>aTol) && InInterv  && (aNbIter<mMaxIterEnd))
          {
               aNbIter++;
               cPt2dr aNewP =  Centroid(aTol1,aP0,aTol0,aP1);  // estimat new value by interpolation
               tREAL8 aNewV = GetV(aNewP);  // value of new point
               if (!CheckNoVal(aNewV))  return;
               if (   ((aNewV<=aV0) != mP0IsSup) || ((aNewV>=aV1) != mP0IsSup) )
               {
                    InInterv = false; // then we are no longer bracketing
               }
               else
               {
                    if (   (aNewV<aVal) == mP0IsSup)   //  V1  <  NewV  < TargetVal  < V0  ( or  V0 < Val < aNewV  < V1)
                    {
                        aV1 = aNewV;
                        aTol1 = std::abs(aV1-aVal);
                        aP1 = aNewP;
                    }
                    else
                    {
                        aV0 = aNewV;
                        aTol0 = std::abs(aV0-aVal);
                        aP0 = aNewP;
                    }
               }
          }
          mPRes = (aTol0<aTol1) ? aP0 : aP1;
          mOk = true;
       }

       bool Ok() const {return mOk;}  ///< Accessor

       const cPt2dr & PRes() const  ///< Accessor to result, allowed only in case of success
       {
            MMVII_INTERNAL_ASSERT_tiny(mOk,"Try to get cGetPts_ImInterp_FromValue::PRes() w/o success");
            return mPRes;
       }

     private :

       inline tREAL8 GetV(const cPt2dr & aP) {return mDIm.DefGetVBL(aP,NoVal);}
       inline bool CheckNoVal(const tREAL8 aV)
       {
           if (aV==NoVal)
           {
               return false;
           }
           return true;
       }
       static constexpr tREAL8 NoVal= -1e10;

       bool  mP0IsSup;                ///< memorize if we have to decrease or increase
       bool   mOk;                    ///< did we get success ?
       const cDataIm2D<Type> & mDIm;  ///< memorise image
       int    mMaxIterInit;           ///< number max of iteration initial
       int    mMaxIterEnd;            ///< number max of iteration by dicothomy
       cPt2dr mPRes;                  ///< memorize the result
};

};

#endif  //  _MMVII_TplImage_PtsFromValue_H_
