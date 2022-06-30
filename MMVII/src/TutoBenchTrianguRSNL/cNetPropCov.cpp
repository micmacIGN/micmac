#include "TrianguRSNL.h"


namespace MMVII
{
namespace NS_Bench_RSNL
{

/* ***************************************** */

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

        /// Give the homologous of point in the main network
        tPNet & MainHom(const tPNet &) const;
    private :
       
         tMainNW * mMainNW;
         cRect2    mBoxM;

         // We compute the global similitude from main to this local
         cSim2D<Type> mSimM2This;  
};


template <class Type> cElemNetwork<Type>::cElemNetwork(tMainNW & aMainNW,const cRect2 & aBoxM) :
        // We put the local box with origin in (0,0) because frozen point are on this point
          tMainNW     (eModeSSR::eSSR_LsqDense,cRect2(cPt2di(0,0),aBoxM.Sz()),false),
          mMainNW     (&aMainNW),
          mBoxM       (aBoxM),
          mSimM2This  (cSim2D<Type>::RandomSimInv(5.0,3.0,1))

{
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
   return mMainNW->PNetOfGrid(aPN.mInd+mBoxM.P1() );
}

template class cElemNetwork<tREAL8>;



};

};
