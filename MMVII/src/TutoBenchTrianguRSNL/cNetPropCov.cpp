#include "TrianguRSNL.h"
#include "include/MMVII_Tpl_Images.h"


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
          mSimM2This  (cSim2D<Type>::RandomSimInv(5.0,3.0,0.3))
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
    std::vector<tPt> aVLoc;
    std::vector<tPt> aVMain;
 
    for (const auto & aPNet : this->mVPts)
    {
         const tPNet & aHomMain = MainHom(aPNet);
         aVLoc.push_back(aPNet.PCur());
         aVMain.push_back(aHomMain.PCur());
    }
    Type aSqResidual;
    cSim2D<Type>  aSimM2L =  cSim2D<Type>::FromExample(aVMain,aVLoc,&aSqResidual);
    tPt  aTr = aSimM2L.Tr();
    tPt  aSc = aSimM2L.Sc();
/*
    Loc =   aSimM2L * Main

    X_loc    (Trx)     (Sx   -Sy)   (X_Main)
    Y_loc =  (Try) +   (Sx   -Sy) * (Y_Main)
    
*/

     int aNbVar = this->mNum;
     std::vector<int>    aVIndTransf(this->mNum,-1);
     cDenseMatrix<Type>  aMatrixTranf(aNbVar,eModeInitImage::eMIA_Null);  ///< Square
     cDenseVect<Type>    aVecTranf(aNbVar,eModeInitImage::eMIA_Null);  ///< Square

     for (const auto & aPNet : this->mVPts)
     {
         const tPNet & aHomMain = MainHom(aPNet);
         int aKx = aPNet.mNumX;
         int aKy = aPNet.mNumY;
         aVIndTransf.at(aKx) = aHomMain.mNumX;
         aVIndTransf.at(aKy) = aHomMain.mNumY;

         aVecTranf(aKx) = aTr.x();
         aVecTranf(aKy) = aTr.y();

         aMatrixTranf.SetElem(aKx,aKx,aSc.x());
         aMatrixTranf.SetElem(aKy,aKx,-aSc.y());
         aMatrixTranf.SetElem(aKx,aKy,aSc.y());
         aMatrixTranf.SetElem(aKy,aKy,aSc.x());
     }

     if (DEBUG_RSNL)
     {
           cDenseVect<Type>    aVecLoc(aNbVar,eModeInitImage::eMIA_Null);  ///< Square
           cDenseVect<Type>    aVecGlob(aNbVar,eModeInitImage::eMIA_Null);  ///< Square
           for (const auto & aPNet : this->mVPts)
           {
               const tPNet & aHomMain = MainHom(aPNet);
               int aKx = aPNet.mNumX;
               int aKy = aPNet.mNumY;
               tPt aPLoc = aPNet.PCur();
               tPt aPGlob = aHomMain.PCur();

               aVecLoc(aKx) = aPLoc.x();
               aVecLoc(aKy) = aPLoc.y();
               aVecGlob(aKx) = aPGlob.x();
               aVecGlob(aKy) = aPGlob.y();
           }
    // aVecGlob + aVecTranf;
/*

           cDenseVect<Type>  aVLoc2 =  (aMatrixTranf * aVecGlob) + aVecTranf;
           cDenseVect<Type>  aVDif = aVLoc2 - aVecLoc;

           StdOut() << "DIF " << aVDif.L2Norm() << "\n";
*/

     }

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
