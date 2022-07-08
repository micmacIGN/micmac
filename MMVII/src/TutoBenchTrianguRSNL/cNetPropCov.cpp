#include "TrianguRSNL.h"


namespace MMVII
{
namespace NS_Bench_RSNL
{


/**  Class for implemanting an "elementary = small" newtork,
     on which we will compute covariance that will be transfered
     in the "big" network
*/
template <class Type>  class  cElemNetwork : public cMainNetwork<Type>
{
    public :
        typedef  cMainNetwork<Type>  tMainNW;
        typedef cPtxd<Type,2>             tPt;
        typedef cPNetwork<Type>           tPNet;


        cElemNetwork(tMainNW & aMainW,const cRect2 & aRectMain);

	void CalcCov(int aNbIter);
        void PropagCov();

	 int DebugN() const {return mDebugN;}
    private :
        /// Give the homologous of point in the main network
        tPNet & MainHom(const tPNet &) const;
       
         tMainNW * mMainNW;
         cRect2    mBoxM;
         // We compute the global similitude from main to this local
         cSim2D<Type> mSimM2This;  

	 int mDebugN; 
};

/* *************************************** */
/*                                         */
/*          cElemNetwork                   */
/*                                         */
/* *************************************** */


template <class Type> cElemNetwork<Type>::cElemNetwork(tMainNW & aMainNW,const cRect2 & aBoxM) :
        // We put the local box with origin in (0,0) because frozen point are on this point
          tMainNW     (eModeSSR::eSSR_LsqDense,cRect2(cPt2di(0,0),aBoxM.Sz()),false),
          mMainNW     (&aMainNW),
          mBoxM       (aBoxM),
#if (DEBUG_RSNL)
          mSimM2This  (tPt(0,0),tPt(1,0)) 
#else 
          (cSim2D<Type>::RandomSimInv(5.0,3.0,0.3))
#endif

{
static int TheNumDebug=0;	
mDebugN = ++TheNumDebug;

     //  To have the right scale compute mSimInd2G from mSimM2This 
    this->mSimInd2G =  mSimM2This   * mMainNW->SimInd2G() ;
    // make it a copy of mMainNW with some similitude
    for (auto  & aPN :  this->mVPts)
    {
          aPN.mTheorPt = mSimM2This.Value(MainHom(aPN).mTheorPt);  // copy the geometry
          aPN.MakePosInit(AMPL_Real2Init);  // Make init position a noisy version of  real coord
   }
}

template <class Type>  cPNetwork<Type> & cElemNetwork<Type>::MainHom(const tPNet & aPN) const
{
   return mMainNW->PNetOfGrid(aPN.mInd+mBoxM.P0() );
}


template <class Type>  void cElemNetwork<Type>::CalcCov(int aNbIter)
{
     for (int aK=0 ; aK<(aNbIter-1); aK++)
         this->OneItereCompensation(false);  // Iterations with a gauge
     this->OneItereCompensation(true);       // last iteration w/o a gauge
}

template <class Type>  void cElemNetwork<Type>::PropagCov()
{
bool Bug = (mDebugN==3);

    std::vector<tPt> aVLoc;
    std::vector<tPt> aVMain;
    for (const auto & aPNet : this->mVPts)
    {
         const tPNet & aHomMain = MainHom(aPNet);
         aVLoc.push_back(aPNet.PCur());
         aVMain.push_back(aHomMain.PCur());
 if (1|| Bug)
 {
     StdOut()  << "PTS " << aPNet.PCur() << " " <<  aHomMain.PCur()  << "\n";
     StdOut()  << "  I " << aPNet.mInd << aPNet.mInd -  aHomMain.mInd  << "\n";
 }
    }
    cSim2D<Type>  aSim =  cSim2D<Type>::FromExample(aVLoc,aVMain);

    StdOut()  <<  mDebugN << " SSS " << aSim.Tr() << " " << aSim.Sc() << "\n";
    getchar();
}

/* *************************************** */
/*                                         */
/*          cMainNetwork                   */
/*                                         */
/* *************************************** */

/*
template <class Type>  void cMainNetwork<Type>::TestCov(const cRect2 &aRect)
{
     cElemNetwork<Type>  aNetElem(*this,aRect);
     aNetElem.CalcCov(10);
}
*/

template <class Type>  void cMainNetwork<Type>::TestCov()
{
     cPt2di aSz(2,2);
     cRect2  aRect(mBoxInd.P0(),mBoxInd.P1()-aSz);

     std::vector<cElemNetwork<Type> *> aVNet;

     for (const auto & aPix: aRect)
     {
         cRect2 aRect(aPix,aPix+aSz);
         auto aPtrN = new cElemNetwork<Type>(*this,aRect);
         aVNet.push_back(aPtrN);
         aPtrN->CalcCov(10);
     }

     for (int aTime=0 ; aTime<1 ; aTime++)
     {
          for (auto & aPtrNet : aVNet)
             aPtrNet->PropagCov();
     }

     DeleteAllAndClear(aVNet);
}


/* ======================================== */
/*           INSTANTIATION                  */
/* ======================================== */
#define PROP_COV_INSTANTIATE(TYPE)\
template class cElemNetwork<TYPE>;\
template class cMainNetwork<TYPE>;

PROP_COV_INSTANTIATE(tREAL4)
PROP_COV_INSTANTIATE(tREAL8)
PROP_COV_INSTANTIATE(tREAL16)

};
};
