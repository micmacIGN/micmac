#include "TrianguRSNL.h"
#include "include/MMVII_Tpl_Images.h"


namespace MMVII
{
namespace NS_Bench_RSNL
{


static constexpr int  MODE_PROPAG_COV        = 0;
static constexpr int  MODE_PROPAG_PTS_SIMUK  = 1;
static constexpr int  MODE_PROPAG_PTS_SIMFIX = 2;

//======================================

static constexpr bool  CHEATING_MAPTRANSFERT = false;
static constexpr double  CHEAT_W             =  0.8;
static constexpr int  MODE_PROPAG = MODE_PROPAG_COV;

static constexpr bool COV_WITH_GAUGE=true;

//======================================

static constexpr bool  SIMUK = ((MODE_PROPAG==MODE_PROPAG_COV) ||(MODE_PROPAG ==  MODE_PROPAG_PTS_SIMUK));
static constexpr bool  PROP_COV = (MODE_PROPAG==MODE_PROPAG_COV) ;

template <class Type>  class  cElemNetwork ;
template <class Type>  class  cCovNetwork  ;



/**  Class for implemanting an "elementary = small" newtork,
     on which we will compute covariance that will be transfered
     in the "big" network
*/



template <class Type>  class  cCovNetwork :   public cMainNetwork<Type>
{
     public :
           cCovNetwork(eModeSSR aMode,cRect2,const cParamMainNW &,cParamSparseNormalLstSq * = nullptr);
           ~cCovNetwork();

           void PostInit() override;
           void TestCov(int aNbIter);

     private :
           std::vector<cElemNetwork<Type> *> mVNetElem;
};

template <class Type>  class  cElemNetwork : public cMainNetwork<Type>
{
    public :
        typedef cMainNetwork<Type>        tMainNW;
        typedef cPtxd<Type,2>             tPt;
        typedef cPNetwork<Type>           tPNet;


        cElemNetwork(tMainNW & aMainW,const cRect2 & aRectMain);
        ~cElemNetwork();

	Type CalcCov(int aNbIter);
        void PropagCov();

        int DebugN() const {return mDebugN;}
        
        tPt  ComputeInd2Geom(const cPt2di & anInd) const override ;
        
    private :
         cPNetwork<Type> & IndMainHom(const cPt2di & anInd) const
         {
               return mMainNW->PNetOfGrid(anInd+mBoxM.P0() );
         }
         cPNetwork<Type> & MainHom(const tPNet & aPN) const
         {
               return IndMainHom(aPN.mInd);
         }
        /// Give the homologous of point in the main network
        // tPNet & MainHom(const tPNet &) const;
       

         tMainNW *                mMainNW;
         cRect2                   mBoxM;
	 int mDebugN; 
         cCalculator<double> *    mCalcCov;
         cCalculator<double> *    mCalcPtsSimFix;
         cCalculator<double> *    mCalcPtsSimVar;
         cDecSumSqLinear<Type>    mDSSL;
};

/* *************************************** */
/*                                         */
/*          cCovNetwork                    */
/*                                         */
/* *************************************** */


template <class Type>  
     cCovNetwork<Type>::cCovNetwork
     (
         eModeSSR aMode,
         cRect2 aRect,
         const cParamMainNW & aParamNW,
         cParamSparseNormalLstSq * aParamLSQ
     ) :
         cMainNetwork<Type>(aMode,aRect,false,aParamNW,aParamLSQ)
{
}

template <class Type> 
        void cCovNetwork<Type>::PostInit() 
{
     cMainNetwork<Type>::PostInit();

     cPt2di aSz(2,2);
     cRect2  aRect(this->mBoxInd.P0(),this->mBoxInd.P1()-aSz+cPt2di(1,1));

     for (const auto & aPix: aRect)
     {
         cRect2 aRect(aPix,aPix+aSz);
         auto aPtrN = new cElemNetwork<Type>(*this,aRect);
         aPtrN->PostInit();
         mVNetElem.push_back(aPtrN);
         Type aRes = aPtrN->CalcCov(10);
	 MMVII_INTERNAL_ASSERT_bench(aRes<1e-8,"No conv 4 sub net");
     }
}

template <class Type>  cCovNetwork<Type>::~cCovNetwork()
{
     DeleteAllAndClear(mVNetElem);
}




template <class Type>  void cCovNetwork<Type>::TestCov(int aNbIter)
{

     Type   aRes0 = 1.0;
     // int aNbIter = SIMUK ? 10 : 1000;
     for (int aTime=0 ; aTime<aNbIter ; aTime++)
     {
	 Type   aResidual = this->CalcResidual() ;
         if (aTime==0)
            aRes0 = aResidual;
	 StdOut()   << aTime <<  " RRR  " << aResidual << " " << aResidual/aRes0 << "\n";//  getchar();

          for (auto & aPtrNet : mVNetElem)
             aPtrNet->PropagCov();

	  this->AddGaugeConstraint(1);
	  this->mSys->SolveUpdateReset();

     }
     if (CHEATING_MAPTRANSFERT)
     {
         StdOut()  <<  "CHEATTTTTTTTTTiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiingggggggggg\n";
     }
     getchar();


}



// template class cCovNetwork<tREAL8>;


/* *************************************** */
/*                                         */
/*          cElemNetwork                   */
/*                                         */
/* *************************************** */



template <class Type> cElemNetwork<Type>::cElemNetwork(tMainNW & aMainNW,const cRect2 & aBoxM) :
        // We put the local box with origin in (0,0) because frozen point are on this point
          cMainNetwork<Type>       (eModeSSR::eSSR_LsqDense,cRect2(cPt2di(0,0),aBoxM.Sz()),false,cParamMainNW()),
          mMainNW                  (&aMainNW),
          mBoxM                    (aBoxM),
          mCalcCov                 (EqNetworkConsDistProgCov(true,1,aBoxM.Sz())),
          mCalcPtsSimFix           (EqNetworkConsDistFixPoints(true,1,aBoxM.Sz(),false)),
          mCalcPtsSimVar           (EqNetworkConsDistFixPoints(true,1,aBoxM.Sz(),true))

          // mMainNW     (&aMainNW),
          // mBoxM       (aBoxM),
{
    // to "play the game" of covariance propagation wiht unknown transformation, the elementary network
    // will have a similitude different from the main, but it must have the same scale as we define a
    // triangulation with distance conservation
  
    this->mSimInd2G  = mMainNW->SimInd2G() * cRot2D<Type>::RandomRot(4.0).Sim();
    static int TheNumDebug=0;	
    mDebugN = ++TheNumDebug;
}

template <class Type> cPtxd<Type,2>  cElemNetwork<Type>::ComputeInd2Geom(const cPt2di & anInd) const
{
     cPNetwork<Type> &  aPMain = IndMainHom(anInd);

     tPt aP = aPMain.mTheorPt;  // get the real point
     aP = mMainNW->SimInd2G().Inverse(aP) ;     // go back to index (perturbated)
     aP = this->mSimInd2G.Value(aP);  // transfom the index using the similitude of the newtork
     
     return aP;
}
/*
*/

template <class Type> cElemNetwork<Type>::~cElemNetwork()
{
    delete mCalcCov;
    delete mCalcPtsSimFix;
    delete mCalcPtsSimVar;
}
/*
template <class Type>  cPNetwork<Type> & cElemNetwork<Type>::MainHom(const tPNet & aPN) const
{
   return this->mMainNW->PNetOfGrid(aPN.mInd+this->mBoxM.P0() );
}
*/


template <class Type>  Type cElemNetwork<Type>::CalcCov(int aNbIter)
{
     for (int aK=0 ; aK<(aNbIter-1); aK++)
     {
         this->OneIterationCompensation(true,true);  // Iterations with a gauge and solve
     } 
     Type aRes = this->CalcResidual();
     this->OneIterationCompensation(COV_WITH_GAUGE,false);       // last iteration with a gauge w/o solve


     // Now get the normal matrix and vector, and decompose it in a weighted sum of square  of linear forms
     if (MODE_PROPAG == MODE_PROPAG_COV)
     {
        auto  aSL = this->mSys->SysLinear();
        auto aSol = this->mSys->CurGlobSol();
        mDSSL.Set(aSol,aSL->V_tAA(),aSL->V_tARhs());

        if (0)
        {
             StdOut() <<  "R=" << aRes<< "\n";
             for (int aK=0 ; aK<this->mSys->NbVar() ; aK++)
             {
                 StdOut()  << " " << this->mSys->CurSol(aK) ;
             }
             StdOut() <<  "\n";
             for (const auto & aPNet : this->mVPts)
             {
                 StdOut()  << " " << aPNet.TheorPt() ;
             }
             StdOut() <<  "\n";
             auto aRes = aSL->V_tAA() * this->mSys->CurGlobSol() - aSL->V_tARhs();
             StdOut() << "RES1 " << aRes.L2Norm() <<  "\n";
        }
     }


     if (0)
     {
         StdOut() << "SSSS " <<   this->mSys->SysLinear()->V_tAA ().Symetricity() << "\n" ;
         cDenseMatrix<Type> A  = this->mSys->SysLinear()->V_tAA ();
	 // A.SelfSymetrizeBottom();

         cResulSymEigenValue<Type> aRSEV = A.SymEigenValue() ;
	 const cDenseVect<Type>   &  aVP = aRSEV.EigenValues() ;
	 for (int aK=0 ; aK<int(aVP.Sz()) ; aK++)
              StdOut()  <<  aVP(aK)  << "  ";
         StdOut()  <<   "\n";
	 FakeUseIt(aVP);
     }

     return aRes;
}

template <class Type>  void cElemNetwork<Type>::PropagCov()
{
    std::vector<tPt> aVLoc;
    std::vector<tPt> aVMain;

    int aNbUkRot = SIMUK ?  3 : 0; // Number of parameters for unknown similitudes
    std::vector<int> aVIndUk(this->mVPts.size()*2+aNbUkRot,-1);  // Index of unknown, if SimUk begin with 4 Tmp-Schur for similitude
 
    for (const auto & aPNet : this->mVPts)
    {
         const tPNet & aHomMain = this->MainHom(aPNet);
         aVIndUk.at(aNbUkRot+aPNet.mNumX) = aHomMain.mNumX;
         aVIndUk.at(aNbUkRot+aPNet.mNumY) = aHomMain.mNumY;
// StdOut() << " " << aPNet.mNumX << " " << aPNet.mNumY  ;

	 if(CHEATING_MAPTRANSFERT)
	 {
             aVLoc.push_back(aPNet.mTheorPt);
             aVMain.push_back(aHomMain.mTheorPt*Type(CHEAT_W) +aHomMain.PCur()*Type(1-CHEAT_W));
	 }
	 else
	 {
             aVLoc.push_back(aPNet.PCur());
             aVMain.push_back(aHomMain.PCur());
	 }
    }

    Type aSqResidual;
    cRot2D<Type>  aRotM2L =  cRot2D<Type>::StdGlobEstimate(aVMain,aVLoc,&aSqResidual);
    {
       tPt  aSomRes(0,0);
       Type aSomDist = 0;
       int  aNb=0;
       for (const auto & aPNet : this->mVPts)
       {
            const tPNet & aHomMain = this->MainHom(aPNet);
            tPt aRes = aPNet.PCur() - aRotM2L.Value(aHomMain.PCur());
            aSomDist += Norm2(aRes);
            aSomRes = aSomRes + aRes;
//   StdOut() << "DDD " << Norm2(aPNet.PCur()-aPNet.TheorPt()) << "\n";
            aNb++;
       }
       // StdOut() << "AvgRes="  <<  aSomRes/Type(aNb)  << " AvgD=" << aSomDist/Type(aNb) <<  "\n"; getchar();
    }

    tPt  aTr   = aRotM2L.Tr();
    Type aTeta = aRotM2L.Teta();

    std::vector<Type> aVTmpRot;
    aVTmpRot.push_back(aTr.x());
    aVTmpRot.push_back(aTr.y());
    aVTmpRot.push_back(aTeta);
/*
    Loc =   aSimM2L * Main

    X_loc    (Trx)     (Sx   -Sy)   (X_Main)
    Y_loc =  (Try) +   (Sy    Sx) * (Y_Main)
    
*/


    if (PROP_COV)
    {
       cSetIORSNL_SameTmp<Type> aSetIO;
       Type aMinW(1e10);
       for (const auto anElemLin : mDSSL.VElems())
       {
           UpdateMin(aMinW,anElemLin.mW);
           cResidualWeighter<Type>  aRW(anElemLin.mW);
           std::vector<Type> aVObs = anElemLin.mCoeff.ToStdVect();
           aVObs.push_back(anElemLin.mCste);
           this->mMainNW->Sys()->AddEq2Subst(aSetIO,mCalcCov,aVIndUk,aVTmpRot,aVObs,aRW);
           //  StdOut() << "CSTE " << anElemLin.mCste << "\n";
       }
       //  StdOut() << "NBEL " << mDSSL.VElems().size() << " MinW= " << aMinW << "\n";
       this->mMainNW->Sys()->AddObsWithTmpUK(aSetIO);
    }
    else
    {
        std::vector<Type> aVObs  =  SIMUK ?  std::vector<Type>()  : aVTmpRot;
        for (const auto & aPNet : this->mVPts)
        {
             // const tPNet & aHomMain = this->MainHom(aPNet);
             tPt aPt =    aPNet.PCur();
             aVObs.push_back(aPt.x());
             aVObs.push_back(aPt.y());
        }
        if (SIMUK)
        {
            cSetIORSNL_SameTmp<Type> aSetIO;
            this->mMainNW->Sys()->AddEq2Subst(aSetIO,mCalcPtsSimVar,aVIndUk,aVTmpRot,aVObs);
            this->mMainNW->Sys()->AddObsWithTmpUK(aSetIO);
        }
        else
        {
           this->mMainNW->Sys()->CalcAndAddObs(mCalcPtsSimFix,aVIndUk,aVObs);
        }
    }


if (0)
{
     int aNbVar = this->mNum;
     std::vector<int>    aVIndTransf(this->mNum,-1);
     cDenseMatrix<Type>  aMatrixTranf(aNbVar,eModeInitImage::eMIA_Null);  ///< Square
     cDenseVect<Type>    aVecTranf(aNbVar,eModeInitImage::eMIA_Null);  ///< Square

     tPt aSc(cos(aTeta),sin(aTeta));

     for (const auto & aPNet : this->mVPts)
     {
         const tPNet & aHomMain = this->MainHom(aPNet);
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


     // Just to check that the convention regarding
     if (0 )
     {
           cDenseVect<Type>    aVecLoc(aNbVar,eModeInitImage::eMIA_Null);  ///< Square
           cDenseVect<Type>    aVecGlob(aNbVar,eModeInitImage::eMIA_Null);  ///< Square
           for (const auto & aPNet : this->mVPts)
           {
               const tPNet & aHomMain = this->MainHom(aPNet);
               int aKx = aPNet.mNumX;
               int aKy = aPNet.mNumY;
               tPt aPLoc = aPNet.PCur();
               tPt aPGlob = aHomMain.PCur();

               aVecLoc(aKx) = aPLoc.x();
               aVecLoc(aKy) = aPLoc.y();
               aVecGlob(aKx) = aPGlob.x();
               aVecGlob(aKy) = aPGlob.y();
           }

           cDenseVect<Type>  aVLoc2 =  (aMatrixTranf * aVecGlob) + aVecTranf;
           cDenseVect<Type>  aVDif = aVLoc2 - aVecLoc;

           StdOut() << "DIF " << aVDif.L2Norm() << "\n";
     }


     //   Xl  = MI * Xg + TI
     //   (Xl-Xl0)  =  MI *(Xg-Xg0)    +  TI -Xl0 + MI * Xg0
     //   E  =   tXl A Xl  -2 tXl V  = t(M Xg+ T)  A (M Xg +T) - 2 t(M Xg +T) V 
     //   E  =   tXg   (tM A M) Xg   +  2 tXg (tM A T)   - 2 tXg tM V  +Cste
     //   E   = tXg  (tM A M) Xg   - 2tXg (tM V  -tM A T)  = tXg  A' Xg  - 2tXg V'
     //    A'  =  tMAM    V' =  tMV  -tM A T

     {
         cDenseVect<Type>    aG0(aNbVar);
         for (int aK=0 ; aK<int (aVIndTransf.size()) ; aK++)
             aG0(aK) = this->mMainNW->CurSol(aVIndTransf[aK]);

         const cDenseMatrix<Type> & M =   aMatrixTranf;
         cDenseVect<Type>    T=  aVecTranf + M* aG0 - this->mSys->CurGlobSol();
         cDenseMatrix<Type> A  = this->mSys->SysLinear()->V_tAA ();
         cDenseVect<Type> V  =   this->mSys->SysLinear()->V_tARhs ();

         cDenseMatrix<Type> tM  =   M.Transpose();
         cDenseMatrix<Type> tMA  =   tM * A;

         cDenseMatrix<Type> Ap  =   tMA  * M;
         cDenseVect<Type>   Vp  =   tM * (V -A *T);

//  StdOut()  <<  "JJJJJ " <<  Ap.Symetricity() << "\n";

	 this->mMainNW->Sys()->SysLinear()->AddCov(Ap,Vp,aVIndTransf);
     }
}
}

/* ======================================== */
/*           INSTANTIATION                  */
/* ======================================== */
#define PROP_COV_INSTANTIATE(TYPE)\
template class cElemNetwork<TYPE>;\
template class cCovNetwork<TYPE>;

PROP_COV_INSTANTIATE(tREAL4)
PROP_COV_INSTANTIATE(tREAL8)
PROP_COV_INSTANTIATE(tREAL16)

};  //  namespace NS_Bench_RSNL

/* ************************************************************************ */
/*                                                                          */
/*                     cAppli_TestPropCov                                   */
/*                                                                          */
/* ************************************************************************ */
using namespace NS_Bench_RSNL;

/** A Class to make many test regarding  covariance propagation
    as things are not clear at thi step
*/

class cAppli_TestPropCov : public cMMVII_Appli
{
     public :
        cAppli_TestPropCov(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :
         int mSzMainN;
         int mSzSubN;
         int mNbItCovProp;

         cParamMainNW           mParam;
         cCovNetwork<tREAL8> * mMainNet;
};


cAppli_TestPropCov::cAppli_TestPropCov(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  ( aVArgs,aSpec),
   mSzMainN      (2),
   mSzSubN       (2),
   mNbItCovProp  (10)
{
}

cCollecSpecArg2007 & cAppli_TestPropCov::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return    anArgObl
           << Arg2007(mSzMainN,"Size of network N->[-N,N]x[NxN],i.e 2 create 25 points")
    ;
}

cCollecSpecArg2007 & cAppli_TestPropCov::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return
       anArgOpt
           << AOpt2007(mSzSubN, "SzSubN","Size of subnetwork N->[0,N[x[0,N[",{eTA2007::HDV})
           << AOpt2007(mNbItCovProp, "NbICP","Number of iteration for cov prop",{eTA2007::HDV})
           << AOpt2007(mParam.mAmplGrid2Real, "NoiseG2R","Perturbation between grid & real position",{eTA2007::HDV})
           << AOpt2007(mParam.mAmplReal2Init, "NoiseR2I","Perturbation between real & init position",{eTA2007::HDV})
   ;
}


int  cAppli_TestPropCov::Exe() 
{
   for (int aK=0 ; aK<10 ; aK++)
   {
       mMainNet = new cCovNetwork <tREAL8>(eModeSSR::eSSR_LsqDense,cRect2::BoxWindow(mSzMainN),mParam);
       mMainNet->PostInit();

       mMainNet->TestCov(mNbItCovProp);

       delete mMainNet;
   }

   return EXIT_SUCCESS;
}

tMMVII_UnikPApli Alloc_TestPropCov(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_TestPropCov(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecTestCovProp
(
     "TestCovProp",
      Alloc_TestPropCov,
      "Test on covariance propagation",
      {eApF::Test},
      {eApDT::None},
      {eApDT::Console},
      __FILE__
);


}; // namespace MMVII

