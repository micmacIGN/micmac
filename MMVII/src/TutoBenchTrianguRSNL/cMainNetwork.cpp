#include "TrianguRSNL.h"


// ==========  3 variable used for debuging  , will disappear
//

// static constexpr double SZ_RAND_TH = 0.1; // Pertubation theoriticall / regular grid
// static constexpr double THE_AMPL_P = 0.1; // Amplitude of pertubation of initial / theoreticall
// static constexpr int    THE_NB_ITER  = 10; // Sign of similitude for position init

// using namespace NS_SymbolicDerivative;
// using namespace MMVII;

namespace MMVII
{
namespace NS_Bench_RSNL
{

/* ======================================== */
/*                                          */
/*              cParamMainNW                */
/*                                          */
/* ======================================== */

cParamMainNW::cParamMainNW() :
    mAmplGrid2Real  (0.1),
    mAmplReal2Init  (0.1),
    mNoiseOnDist    (0.0),
    mFactXY         (1,1)
{
}

/* ======================================== */
/*                                          */
/*              cMainNetwork                */
/*                                          */
/* ======================================== */

template <class Type> cMainNetwork <Type>::cMainNetwork 
                      (
		            eModeSSR aMode,
                            cRect2   aRect,
			    bool WithSchurr,
                            const cParamMainNW & aParamNW,
			    cParamSparseNormalLstSq * aParamLSQ,
			    const std::vector<Type>  & aWeightSetSchur
		      ) :
    mModeSSR   (aMode),
    mParamLSQ  (aParamLSQ),
    mBoxInd    (aRect),
    mX_SzM     (mBoxInd.Sz().x()),
    mY_SzM     (mBoxInd.Sz().y()),
    mWithSchur (WithSchurr),
    mParamNW   (aParamNW),
    mNum       (0),
    mMatrP     (cMemManager::AllocMat<tPNetPtr>(mX_SzM,mY_SzM)), // alloc a grid of pointers
    mSys       (nullptr),
    mCalcD     (nullptr),
    // Amplitude of scale muste
    mSimInd2G   (cSim2D<Type>::RandomSimInv(5.0,3.0,0.1)) ,
    mBoxPts     (cBox2dr::Empty()),
    mWeightSetSchur  (aWeightSetSchur)
{
}

template <class Type> void cMainNetwork <Type>::PostInit()
{

     // generate in VPix a regular grid, put them in random order for testing more config in matrix
     std::vector<cPt2di> aVPix;
     for (const auto& aPix: mBoxInd)
         aVPix.push_back(aPix);
     aVPix = RandomOrder(aVPix);

     std::vector<Type> aVCoord0; // initial coordinates for creating unknowns
     // Initiate Pts of Networks in mVPts,
     cTplBoxOfPts<Type,2>  aBox;
     for (const auto& aPix: aVPix)
     {
         tPNet aPNet(mVPts.size(),aPix,*this);
         mVPts.push_back(aPNet);
         aBox.Add(aPNet.TheorPt());
         aBox.Add(aPNet.PosInit());

	 if (! aPNet.mSchurrPoint)
	 {
             aVCoord0.push_back(aPNet.mPosInit.x());
             aVCoord0.push_back(aPNet.mPosInit.y());
	 }
     }
     mBoxPts   =  cBox2dr(ToR(aBox.P0()),ToR(aBox.P1()));
     if (1) // Eventually put in random order to check implicit order of NumX,NumY is not used 
     {
        if (1)
	{
           mVPts = RandomOrder(mVPts);
           // StdOut() << " Dooo :  ORDER RANDOM\n";
	}
        else
           StdOut() << "NO ORDER RANDOM\n";
        // But need to reset the num of points which is used in link construction
        for (int aK=0 ; aK<int(mVPts.size()) ; aK++)
        {
            mVPts[aK].mNumPt = aK;
        }
     }
     // Put adress of points in a grid so that they are accessible by indexes
     for (auto & aPNet : mVPts)
         PNetPtrOfGrid(aPNet.mInd) = & aPNet;

         
     /*  For the schur point, the estimation of current position, is done by averaging of PCur of neighboors (left&right),
         see comment in ::PCur() 
        
         Iw we randomize the coordinate, this estimator will not converge to the theoreticall position.
         So forthe special case of schur point, we use the same formula for their theoreticall
         and the esitmator of PCur
      */

     for (auto& aPix: mBoxInd)
     {
	  tPNet &  aPSch = PNetOfGrid(aPix);
          if (aPSch.mSchurrPoint) 
          {
	      const tPNet & aPL = PNetOfGrid(aPSch.mInd+cPt2di(-1,0));  // PLeft
	      const tPNet & aPR = PNetOfGrid(aPSch.mInd+cPt2di( 1,0));  // PRight
	      aPSch.mTheorPt  =   (aPL.TheorPt()+aPR.TheorPt())/Type(2.0);

          }
     }
     
     // Initiate system "mSys" for solving
     if ((mModeSSR==eModeSSR::eSSR_LsqNormSparse)  && (mParamLSQ!=nullptr))
     {
         // LEASTSQ:CONSTRUCTOR , case Normal sparse, create first the least square
	 cLeasSq<Type>*  aLeasSQ =  cLeasSq<Type>::AllocSparseNormalLstSq(aVCoord0.size(),*mParamLSQ);
         mSys = new tSys(aLeasSQ,cDenseVect<Type>(aVCoord0));
     }
     else
     {
         // BASIC:CONSTRUCTOR other, just give the mode
         mSys = new tSys(mModeSSR,cDenseVect<Type>(aVCoord0));
     }

     // compute links between Pts of Networks,
     for (const auto& aPix: mBoxInd)
     {
	  tPNet &  aPN1 = PNetOfGrid(aPix);
          for (const auto& aNeigh: cRect2::BoxWindow(aPix,1))
          {
              if (IsInGrid(aNeigh))
              {
	          tPNet &  aPN2 = PNetOfGrid(aNeigh);
                  // Test on num to do it only one way
                  if ((aPN1.mNumPt>aPN2.mNumPt) && aPN1.AreLinked(aPN2))
	          {
                       // create the links, be careful that for pair with schur point all the links start from schur ,
		       // this will make easier the regrouping of equation concerning the same point,
		       // the logic of this lines of code take use the fact that K1 and K2 cannot be both 
                       // schur points (tested in Linked())
		
                       if (aPN1.mSchurrPoint)  // K1 is Tmp and not K2, save K1->K2
                          aPN1.mLinked.push_back(aPN2.mNumPt);
		       else if (aPN2.mSchurrPoint) // K2 is Tmp and not K1, save K2->K2
                          aPN2.mLinked.push_back(aPN1.mNumPt);
		       else // None Tmp, does not matters which way it is stored
                          aPN1.mLinked.push_back(aPN2.mNumPt);  
	          }
                  
             }
          }
     }
     // create the "functor" that will compute values and derivates
     mCalcD =  EqConsDist(true,1);
}

/// Default value return exact distance 
template <class Type>  Type cMainNetwork<Type>::ObsDist(const tPNet & aPN1,const tPNet & aPN2) const
{
     return Norm2(aPN1.TheorPt()-aPN2.TheorPt()); 
}

template <class Type>  bool cMainNetwork<Type>::OwnLinkingFiltrage(const cPt2di &,const cPt2di &) const
{
   return true;
}




template <class Type> bool  cMainNetwork <Type>::AxeXIsHoriz() const
{
     const tPt& aSc = mSimInd2G.Sc();

     return std::abs(aSc.x()) > std::abs(aSc.y());
}

template <class Type> cPtxd<Type,2>  cMainNetwork <Type>::ComputeInd2Geom(const cPt2di & anInd) const
{
    Type aIndX = anInd.x()*mParamNW.mFactXY.x();
    Type aIndY = anInd.y()*mParamNW.mFactXY.y();
// StdOut() << "SSSS " << mParamNW.mFactXY << "\n";

    return mSimInd2G.Value(tPt(aIndX,aIndY) + tPt::PRandC()*Type(mParamNW.mAmplGrid2Real)  );
}

template <class Type> cMainNetwork <Type>::~cMainNetwork ()
{
    cMemManager::FreeMat<tPNetPtr>(mMatrP,mY_SzM);
    delete mSys;
    delete mCalcD;
}

template <class Type> bool  cMainNetwork <Type>::WithSchur()  const {return mWithSchur;}
template <class Type> int&  cMainNetwork <Type>::Num() {return mNum;}

template <class Type> const cSim2D<Type>& cMainNetwork<Type>::SimInd2G() const {return mSimInd2G;}
template <class Type> const cParamMainNW& cMainNetwork<Type>::ParamNW()  const {return mParamNW;}

template <class Type> cResolSysNonLinear<Type>* cMainNetwork<Type>::Sys() {return mSys;}

template <class Type> Type cMainNetwork <Type>::CalcResidual() 
{
     Type  aSumResidual = 0;
     Type  aNbPairTested = 0;
     std::vector<tPt>  aVCur;
     std::vector<tPt>  aVTh;
     //  Compute dist to sol + add constraint for fixed var
     for (const auto & aPN : mVPts)
     {
        // Add distance between theoreticall value and curent to compute global residual
        if (! aPN.mSchurrPoint)
        {
            aNbPairTested++;
            aVCur.push_back(aPN.PCur());
            aVTh.push_back(aPN.TheorPt());
            aSumResidual += SqN2(aPN.PCur() -aPN.TheorPt());
	    // StdOut() << aPN.PCur() - aPN.TheorPt() << aPN.mInd << "\n";
            if (aPN.mFrozenX || aPN.mFrozenY)
            {
	        //  StdOut() << "CCC=> " << aPN.PCur() << aPN.TheorPt() << aPN.mInd << "\n";
            }
        }
     }
     if (0)
     {
         Type aRes;
         auto  aMap = cSim2D<Type>::StdGlobEstimate(aVCur,aVTh,&aRes);
         FakeUseIt(aMap);
         //StdOut() << "RESIDUAL By Map Fit ";
         //StdOut() << aVCur[1] - aVCur[0] / aVTh[1]-aVTh[0] <<  "\n";
         return aRes;
     }
     return sqrt(aSumResidual / aNbPairTested );
}

template <class Type> void cMainNetwork <Type>::AddGaugeConstraint(Type aWeightFix)
{
     if (aWeightFix==0) return;
     //  Compute dist to sol + add constraint for fixed var
     for (const auto & aPN : mVPts)
     {
           // EQ:FIXVAR
	   // Fix X and Y for the two given points
	   if (aPN.mFrozenY) // If Y is frozenn add equation fixing Y to its theoreticall value
	   {
              if (aWeightFix>=0)
                 mSys->AddEqFixVar(aPN.mNumY,aPN.TheorPt().y(),aWeightFix);
	      else
                 mSys->SetFrozenVar(aPN.mNumY,aPN.TheorPt().y());
	   }


	   if (aPN.mFrozenX)   // If X is frozenn add equation fixing X to its theoreticall value
	   {
              if (aWeightFix>=0)
                  mSys->AddEqFixVar(aPN.mNumX,aPN.TheorPt().x(),aWeightFix);
	      else 
                 mSys->SetFrozenVar(aPN.mNumX,aPN.TheorPt().x());
	   }
     }
}

template <class Type> Type cMainNetwork<Type>::DoOneIterationCompensation(double aWeigthGauge,bool WithCalcReset)

{
     Type   aResidual = CalcResidual() ;
     // if we are computing covariance we want it in a free network (the gauge constraint 
     // in the local network have no meaning in the coordinate of the global network)
     AddGaugeConstraint(aWeigthGauge);
     
     
     //  Add observation on distances

     for (const auto & aPN1 : mVPts)
     {
         // If PN1 is a temporary unknown we will use schurr complement
         if (aPN1.mSchurrPoint)
	 {
            // SCHURR:CALC
	    cPtxd<Type,2> aP1= aPN1.PCur(); // current value, required for linearization
            cPtxd<Type,2> aPTh1= aPN1.TheorPt(); // theoreticall value, used for test on fix var (else it's cheating to use it)
            std::vector<Type> aVTmp{aP1.x(),aP1.y()};  // vectors of temporary

	    // structure to generate "hard" constraints on temporary , cheat with theoreticall values
            std::vector<int>    aVIndFrozen;
            std::vector<Type>   aVValFrozen;
	    if (mWeightSetSchur.at(1)<0)
	    {
                aVIndFrozen.push_back(-2);
		aVValFrozen.push_back(aPTh1.y());
	    }
	    if (mWeightSetSchur.at(0)<0)
	    {
                aVIndFrozen.push_back(-1);
		aVValFrozen.push_back(aPTh1.x());
	    }

            cSetIORSNL_SameTmp<Type> aSetIO(aVTmp,aVIndFrozen,aVValFrozen); // structure to grouping all equation relative to PN1
	    // Parse all obsevation on PN1
            for (const auto & aI2 : aPN1.mLinked)
            {
                const tPNet & aPN2 = mVPts.at(aI2);
	        //std::vector<int> aVIndMixt{aPN2.mNumX,aPN2.mNumY,-1,-1};  // Compute index of unknowns for this equation
	        std::vector<int> aVIndMixt{-1,-2,aPN2.mNumX,aPN2.mNumY};  // Compute index of unknowns for this equation
                std::vector<Type> aVObs{ObsDist(aPN1,aPN2)}; // compute observations=target distance
                // Add eq in aSetIO, using CalcD intantiated with VInd,aVTmp,aVObs
		mSys->AddEq2Subst(aSetIO,mCalcD,aVIndMixt,aVObs);
	    }
	    {
                if (mWeightSetSchur.at(0)>=0) aSetIO.AddFixVarTmp(-1,aPTh1.x(), mWeightSetSchur.at(0)); // soft constraint-x  on theoreticall
                if (mWeightSetSchur.at(1)>=0) aSetIO.AddFixVarTmp(-2,aPTh1.y(), mWeightSetSchur.at(1)); // soft constraint-y  on theoreticall
                if (mWeightSetSchur.at(2)>=0) aSetIO.AddFixCurVarTmp(-1, mWeightSetSchur.at(2)); // soft constraint-x  on current
                if (mWeightSetSchur.at(3)>=0) aSetIO.AddFixCurVarTmp(-2, mWeightSetSchur.at(3)); // soft constraint-y  on current
		// StdOut() << "GGGGGgg\n";getchar();
	    }
	    mSys->AddObsWithTmpUK(aSetIO);
	 }
	 else
	 {
               // BASIC:CALC Simpler case no temporary unknown, just add equation 1 by 1
               for (const auto & aI2 : aPN1.mLinked)
               {
                    const tPNet & aPN2 = mVPts.at(aI2);
	            std::vector<int> aVInd{aPN1.mNumX,aPN1.mNumY,aPN2.mNumX,aPN2.mNumY};  // Compute index of unknowns
                    std::vector<Type> aVObs{ObsDist(aPN1,aPN2)};  // compute observations=target distance
                    // Add eq  using CalcD intantiated with VInd and aVObs
	            mSys->CalcAndAddObs(mCalcD,aVInd,aVObs);
	       }
	 }
     }

     // If we are computing for covariance : (1) the system is not inversible (no gauge constraints)
     // (2) we dont want to reset it   ;  so just skip this step
     if (WithCalcReset)
     {
        mSys->SolveUpdateReset();
     }

     return aResidual;
     // return aSumResidual / aNbPairTested ;
}
template <class Type> const Type & cMainNetwork <Type>::CurSol(int aK) const
{
    return mSys->CurSol(aK);
}


/* ======================================== */
/*                                          */
/*              cPNetwork                   */
/*                                          */
/* ======================================== */


template <class Type> cPNetwork<Type>::cPNetwork(int aNumPt,const cPt2di & anInd,cMainNetwork <Type> & aNet) :
     mNumPt    (aNumPt),
     mInd      (anInd),
     mTheorPt  (aNet.ComputeInd2Geom(mInd)),
     mNetW     (&aNet),
	//  Tricky ,for direction set cPt2di(-1,0)) to avoid interact with schurr points
	//  but is there is no schurr point, set it to cPt2di(1,0) to allow network [0,1]x[0,1]
     mFrozenX  (      ( mInd==cPt2di(0,0)) 
		  ||  (   aNet.AxeXIsHoriz() ? 
			  (mInd==cPt2di(0,1)) : 
			  (aNet.WithSchur()   ?  (mInd==cPt2di(-1,0)) : (mInd==cPt2di(1,0)))
                       )
	        ),
     mFrozenY  ( mInd==cPt2di(0,0)  ), // fix origin
     mSchurrPoint    (aNet.WithSchur() && (mInd.x()==1)),  // If test schur complement, Line x=1 will be temporary
     mNumX     (-1),
     mNumY     (-1)
{
     MakePosInit(aNet.ParamNW().mAmplReal2Init);
     
/*
     {
        double aAmplP = THE_AMPL_P; // Coefficient of amplitude of the pertubation
	Type aSysX = -mInd.x() +  mInd.y()/2.0 +std::abs(mInd.y());  // sytematism on X : Linear + non linear abs
        Type aSysY  =  mInd.y() + 4*Square(mInd.x()/aNet.NetSz()); // sytematism on U : Linear + quadratic term

        mPosInit.x() =  mTheorPt.x() + aAmplP*(aSysX + 2*RandUnif_C());;
        mPosInit.y() =  mTheorPt.y() + aAmplP*(aSysY + 2*RandUnif_C() );
     }
*/

     //  To fix globally the network (gauge) 3 coordinate are frozen, for these one the pertubation if void
     //  so that recover the good position
     //  =>>                                 NO LONGER TRUE
/*
     if (mFrozenX)
       mPosInit.x() = mTheorPt.x();
     if (mFrozenY)
       mPosInit.y() = mTheorPt.y();
*/


     if (!mSchurrPoint)
     {
        mNumX = aNet.Num()++;
        mNumY = aNet.Num()++;
     }
}

/**  To assess the correctness of our code , we must prove that we are able to recover the "real" position from
     a pertubated one;  the perturbation must be sufficiently complicated to be sure that position is not recovered 
     by "chance" , but also not to big to be sure that the gradient descent will work
     The pertubation is the a mix of sytematism and random, all is being mulitplied by some amplitude (aAmplP)
*/

template <class Type> void cPNetwork<Type>::MakePosInit(const double & aMulAmpl)
{
   double aAmplP = aMulAmpl* Norm2(mNetW->SimInd2G().Sc()); // Coefficient of amplitude of the pertubation
   Type aSysX = -mInd.x() +  mInd.y()/2.0 +std::abs(mInd.y());  // sytematism on X : Linear + non linear abs
   Type aSysY  =  mInd.y() + 4*Square(mInd.x()/mNetW->NetSz()); // sytematism on U : Linear + quadratic term

   mPosInit.x() =  mTheorPt.x() + aAmplP*(aSysX + 2*RandUnif_C());;
   mPosInit.y() =  mTheorPt.y() + aAmplP*(aSysY + 2*RandUnif_C() );
}


template <class Type> cPtxd<Type,2>  cPNetwork<Type>::PCur() const
{
	// For standard unknown, read the cur solution of the system
    if (!mSchurrPoint)
	return cPtxd<Type,2>(mNetW->CurSol(mNumX),mNetW->CurSol(mNumY));

    /*  For temporary unknown we must compute the "best guess" as we do by bundle intersection.
     
        If it was the real triangulation problem, we would compute the best circle intersection
	with all the linked points, but it's a bit complicated and we just want to check software
	not resolve the "real" problem.

	An alternative would be to use the theoreticall value, but it's too much cheating and btw
	may be a bad idea for linearization if too far from current solution.

	As an easy solution we take the midle of PCur(x-1,y) and PCur(x+1,y).
       */

    int aNbPts = 0;
    cPtxd<Type,2> aSomP(0,0);
    for (const auto & aI2 : mLinked)
    {
           const  cPNetwork<Type> & aPN2 = mNetW->PNet(aI2);
	   if (mInd.y() == aPN2.mInd.y())
	   {
               aSomP +=  aPN2.PCur();
               aNbPts++;
           }
    }
    MMVII_INTERNAL_ASSERT_bench((aNbPts==2),"Bad hypothesis for network");

    return aSomP / (Type) aNbPts;
}

template <class Type> const cPtxd<Type,2> &  cPNetwork<Type>::TheorPt() const { return mTheorPt; }
template <class Type> const cPtxd<Type,2> &  cPNetwork<Type>::PosInit() const { return mPosInit; }

template <class Type> bool cPNetwork<Type>::AreLinked(const cPNetwork<Type> & aP2) const
{
   // Precaution, a poinnt is not linked yo itself
   if (mInd== aP2.mInd) 
      return false;

   //  If two temporay point, they are not observable
   if (mSchurrPoint && aP2.mSchurrPoint)
      return false;

    //  else point are linked is they are same column, or neighbooring colums
    if (NormInf(mInd-aP2.mInd) >1) return false;

    return mNetW->OwnLinkingFiltrage(mInd,aP2.mInd);
}

/* ======================================== */
/*           INSTANTIATION                  */
/* ======================================== */
#define NETWORK_INSTANTIATE(TYPE)\
template class cMainNetwork<TYPE>;\
template class cPNetwork<TYPE>;


NETWORK_INSTANTIATE(tREAL4)
NETWORK_INSTANTIATE(tREAL8)
NETWORK_INSTANTIATE(tREAL16)
};

};
