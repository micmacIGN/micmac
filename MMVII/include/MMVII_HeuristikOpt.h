#ifndef  _MMVII_Heuristik_Opt_H_
#define  _MMVII_Heuristik_Opt_H_

#include "MMVII_Ptxd.h"
#include "MMVII_Mappings.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"



namespace MMVII
{

/** \file "MMVII_HeuristikOpt.h"
    \brief Declaration functionnalities for optimization that are not based on local linearization,
           typicall used on image matching 
*/

/** Class for optimizing by local descent function that are not differenciable. Method :
 *
 *  - multi-step approach
 *  - iteratively research the optimum in a neigourhood (for example can be V4 or V8 if dim=2)
 *  - take cautious not to visit twice the same point (this is the only "non trivial part")
 */

typedef  std::vector<std::list<int>> tVLInt;

template <const int Dim> class cOptimByStep
{
     public :
        typedef cPtxd<tREAL8,Dim>            tPtR;
        typedef cPtxd<int,Dim>               tPtI;
        typedef cDataMapping<tREAL8,Dim,1>   tMap;

        cOptimByStep
        (
	      const tMap & aMap,   // Function/Map to optimize
	      bool IsMin,          // do we minimize / maximize
	      tREAL8 aMaxDInfInit, // forbid point to be too far from initial position
	      int aDist1Max=Dim    // apparently no longer used
        );
        std::pair<tREAL8,tPtR>  Optim(const tPtR & ,tREAL8 aStepInit,tREAL8 aStepLim,tREAL8 aMul=0.5);

     private :
        std::pair<tREAL8,tPtR> CurValue() const;
        bool DoOneStep(tREAL8 aStep);  ///< return false if point go too far from init
        // void  OneStep(const tPtR
        tREAL8 Value(const tPtR& aPt) const;

        const tMap &                  mFunc;     ///<  contain the func to optimize
        tREAL8                        mSign;     ///< +1 for min,  -1 for max
        cWhichMin<tPtR,tREAL8>        mWMin;     ///< store best result and arg-best
        tREAL8                        mMaxDInfInit; ///< maximal dist to initial value we accept
        int                           mDist1Max; ///< max dist-1 of neighboorhooud
        std::vector<cPtxd<int,Dim>>   mINeigh;   ///< list of integer neighboorhoud
        const tVLInt &                mIndNextN; ///< given last neigh, will give fast access to next neigh
        tPtR                          mPt0;      ///< memorize init point to avoid we go too far

};

/** Class that can be used Clino+Vert or for pose estimation */

class cOptimizeRotAndVUnit  : public cDataMapping<tREAL8,5,1>
{
     public :
          typedef std::pair<tRotR,cPt3dr> tSol;

          /// Cstructor  isSignAmb -> yes for clino as it's invariant to sign chg in vertical & I &J
          cOptimizeRotAndVUnit(int aNbSamleRot,int aNbSampleSphere,bool isSignAmb);
          std::pair<tREAL8,tSol> ComputeSolInit(tREAL8 aDistNeigh,tREAL8 anEpsilon,int aNbTest,tREAL8 aDeltaSc);

     private :
          virtual tREAL8 ScoreRotAndVect (const tRotR&,const cPt3dr &) const = 0;
          std::pair<tRotR,cPt3dr>  P5toRotP3(const cPtxd<tREAL8,5>& aPt) const;
          cPt1dr  Value(const cPtxd<tREAL8,5>&) const override ;

          int  mNbSampleRot;
          int  mNbSampleSphere;
          bool mIsSignAmb;
          tRotR mCurRot0;         //< Cur value of rotation used as a starting point
          cP3dNormWithUK mCurV0;  //< Cur unit vector  used used as starting point (J,K complement)

};




}; //  namespace MMVII

#endif  //   _MMVII_Heuristik_Opt_H_
