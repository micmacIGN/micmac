#include "MMVII_SysSurR.h"
#include "MMVII_Tpl_Images.h"

/* ===================================================== */
/* ===================================================== */
/* ===================================================== */

/*  To think about :

      (1)  normal matrix vs schurr
      (2)  variance of residual vs schurr
      (3)  and what about constraints ?
*/

namespace MMVII
{


class cBenchLstSqEstimUncert
{
   public :
         typedef cDenseVect<tREAL8>  tDV;

         /// Constructor, takes the dimension & a vector giving the number of sampling  for each  observation
         cBenchLstSqEstimUncert(int aDim,const std::vector<int> & aVecNbByObs);
         void DoIt(bool isDemoTest,eModeSSR aMode,const std::vector<int> & aVFrozen,const std::vector<std::vector<int>> & aVIndCstr);

          /// Destructor, free allocated object
         ~cBenchLstSqEstimUncert();

    private :


         int                          mDim;         ///< dimension of the space
         tDV                          mCommonP;     ///< comonn point to all observation
         std::vector<int>             mVecNbByObs;  ///< number of sampling of random var in each dimension
         int                          mNbObs;       ///< number of obsevation (= size of mVecNbByObs)
         tREAL8                       mStdDev;      ///< Standard deviation common to all obs
         cDecomposPAdikVar            mDecompos;    ///< help class to parse all combination of 
         cResolSysNonLinear<tREAL8> * mSys;         ///< solver
         cStrStat2<tREAL8>            mStat2;       ///<  Used for computing empirical covariance of solutions
         cDenseMatrix<tREAL8>         mMoyUnc;      ///< Used to compute average of uncertainty
         std::vector<tDV>             mVects;       ///< vector of all linear part of observation
         std::vector<std::vector<tREAL8>> mVRHS;    ///< vector of all possible RHS values of equations

};

tREAL8 CenteredNormValue(int aKV,int aNbVal)  {return  (aKV - (aNbVal-1.0) /2.0);}

/** Return of vector of NbVal regularly sampled values, having a given average and standard deviation */

std::vector<tREAL8> DiscreteRegSampledLaw(int aNbVal,tREAL8 aAvg,tREAL8 aSigma)
{
    // compute empiricall standard dev of discrete law, could use analyticall formula but
    // (1) I am lazy (2) this way can be generalized to other law (3) I am lazy ...
    cComputeStdDev<tREAL8>  aCDev0;
    for (int aKV=0 ; aKV<aNbVal ; aKV++)
        aCDev0.Add(CenteredNormValue(aKV,aNbVal));

    tREAL8 aMul = aSigma / aCDev0.StdDev(); // multiplier to have the right final std dev
    cComputeStdDev<tREAL8>  aCDev1; // Used to check final result
    std::vector<tREAL8> aResult; 
    for (int aKV=0 ; aKV<aNbVal ; aKV++)
    {
        tREAL8 aVal = aAvg + CenteredNormValue(aKV,aNbVal)  * aMul; // compute val
        aResult.push_back(aVal);  // push it in result
        aCDev1.Add(aVal);         // memorize for checking a posteriori the targeted variance
    }
    MMVII_INTERNAL_ASSERT_bench(std::abs(aCDev1.StdDev()-aSigma)<1e-5,"Var in  UC Lsq");
    return aResult;
}


cBenchLstSqEstimUncert::cBenchLstSqEstimUncert(int aDim,const std::vector<int> & aVecNbByObs) :
    mDim        (aDim),
    mCommonP    (tDV(aDim,eModeInitImage::eMIA_RandCenter)*10.0),
    mVecNbByObs (aVecNbByObs),
    mNbObs      (mVecNbByObs.size()),
    mStdDev     (RandInInterval(0.1,0.3)),
    mDecompos   (mVecNbByObs),
    // mSys        (new cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqDense,tDV(aDim,eModeInitImage::eMIA_RandCenter))),
    mSys        (nullptr), // (new cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqDense,mCommonP)),
    mStat2      (aDim),
    mMoyUnc     (aDim,eModeInitImage::eMIA_Null),
    mVects      (tDV::GenerateVectNonColin(mDim,mNbObs,0.05))
{
    // little check
    MMVII_INTERNAL_ASSERT_bench(mDim<mNbObs,"Not enough ons in cBenchLstSqEstimUncert");
}

void cBenchLstSqEstimUncert::DoIt
     (
         bool isDemoTest,
         eModeSSR aMode,
         const std::vector<int> & aVFrozen, 
         const std::vector<std::vector<int>> & aVVIndCstr
      )
{
    // ----------- [0] initialize the parameters,  re-set global var (simpler ...) -----------
    delete mSys;
    mSys = new cResolSysNonLinear<tREAL8>(aMode,mCommonP);
    mStat2 =  cStrStat2<tREAL8>(mDim);
    mMoyUnc     = cDenseMatrix<tREAL8>(mDim,eModeInitImage::eMIA_Null);


    // ----------- [1] handle the constraintes --------------------

   std::vector<bool>  aVIsFree(mDim,true);
   std::vector<int>   aVFreeVar;
         //- [1.1] fix frozen var
    for (const auto & aIndFrz : aVFrozen)
    {
          mSys->SetFrozenVarCurVal(aIndFrz);
          aVIsFree.at(aIndFrz) = false;
    }
         //- [1.2] add the linear constraints
    for (size_t aNumCstr=0 ; aNumCstr<aVVIndCstr.size() ; aNumCstr++)
    {
       auto & aVIndCstr	= aVVIndCstr.at(aNumCstr);
       tDV  aVecCstr(mDim,eModeInitImage::eMIA_Null);
       for (const auto & aIndCstr : aVIndCstr)
       {
           // some strange formula, just to be "sure" that constraint are independant
           aVecCstr(aIndCstr) = 1/(1.0+aIndCstr) +RandInInterval(0.1,0.2) + std::sin(aNumCstr+aIndCstr) ;
           aVIsFree.at(aIndCstr) = false;
       }
       // we assure that constraint is satisfied by the solution
       tREAL8 aCsteCstr = aVecCstr.DotProduct(mCommonP);
       mSys->AddConstr(aVecCstr,aCsteCstr,true);
    }

    // compute the vector of free variables
    for (size_t aKV=0 ; aKV<aVIsFree.size() ; aKV++)
       if (aVIsFree.at(aKV))
          aVFreeVar.push_back(aKV);


    int aNbCstrTot = aVFrozen.size() + aVVIndCstr.size();
    // tREAL8 aRatioUnCstr =     (mNbObs)/double(mNbObs-(mDim)) ; // gauss markov formula
    tREAL8 aRatioWithCstr =   (mNbObs+aNbCstrTot)/double(mNbObs-(mDim-aNbCstrTot)) ; // gauss markov formula

    //  -------- [2] compute the law of each variable, ----------------
    //  it's done in such way that each law has the same variance (see Homoscedasticity in wikipedia)
    for (int aKObs=0 ; aKObs< mNbObs ; aKObs++)
    {
        int aNbVal = mVecNbByObs.at(aKObs);  // number of value for this law
        tREAL8 aAvg = mCommonP.DotProduct(mVects.at(aKObs));  // assure the each equation is centerd of the common point
        mVRHS.push_back(DiscreteRegSampledLaw(aNbVal,aAvg,mStdDev));
    }

    //  -------- [3] compute the vector for testing uncertainty of linear combination

              // [3.1]  initialize 2 rand vector , with 0 on non free variables
    tDV   aComb1 (mDim,eModeInitImage::eMIA_Null);  // vector for first combination of var
    tDV   aComb2 (mDim,eModeInitImage::eMIA_Null);  // vector for second combination of var
    for (int aK=0 ; aK<mDim ; aK++)
    {
         if (aVIsFree.at(aK))  // initialse only for free var, we dont know what happens for unfree var
         {
                aComb1(aK) = RandUnif_C();
                aComb2(aK) = RandUnif_C();
         }
    }
    std::vector<cSparseVect<tREAL8>>  aVSpCombBlin
                                      {
                                          cSparseVect<tREAL8>(aComb1),
                                          cSparseVect<tREAL8>(aComb2)
                                      };
             // [3.2]  compute empiricall covariance of aComb1.X and aComb2.X
    cMatIner2Var<tREAL8>      aMI2V;  
             // [3.3]   compute the average of estimator  covariance of  aComb1.X /aComb2.X
    cWeightAv<tREAL8,tREAL8>  aWAvCov11; 
    cWeightAv<tREAL8,tREAL8>  aWAvCov12;
    cWeightAv<tREAL8,tREAL8>  aWAvCov22;

             // [3.4]   also estimator  covariance of  aComb1.X /aComb2.X , but closer to optimal way (w/o inverse)
    cDenseMatrix<tREAL8>  aLinComb12 = cDenseMatrix<tREAL8>::FromLines({aComb1,aComb2});
    cDenseMatrix<tREAL8>  aColComb12 = aLinComb12.Transpose();
    cDenseMatrix<tREAL8>  aMoyCov12(2,eModeInitImage::eMIA_Null);

            // [3.5]  Matrix for accumulating unbiased estimation Var/Cov on free var
    cDenseMatrix<tREAL8>  aMoyEstimVCV(aVFreeVar.size(),eModeInitImage::eMIA_Null);

    // --------- [4] ----------  parse all the possible combination of all the random variable --------
    for (int aK=0 ; aK<mDecompos.MulBase() ;aK++)
    {
         //  [4.1]   Add the linear observations of this configuration
         std::vector<int>  aVInd =  mDecompos.DecomposSizeBase(aK);  // decomp the index mNbObs  sub index
         for (int aKObs=0 ; aKObs< mNbObs ; aKObs++)
         {
             int aIndOfObs = aVInd.at(aKObs);
             tREAL8 aRHS =  mVRHS.at(aKObs).at(aIndOfObs);
             mSys->AddObservationLinear(1.0,mVects.at(aKObs),aRHS);
         }

         // cDenseMatrix<tREAL8>  aMUC = mSys->SysLinear()->V_tAA();   // normal matrix , do it before Reset !!
         cResult_UC_SUR<tREAL8> aRSUR(mSys,false,true,aVFreeVar,aVSpCombBlin);
         // aRSUR.SetDoComputeNormalMatrix(true);

         tDV aSol = mSys->SolveUpdateReset(0.0,{&aRSUR});  // compute the solution of this config

         cDenseMatrix<tREAL8> aMatNorm = aRSUR.NormalMatrix();

         // [4.2] Test that the variance is correctly estimated in class "cResolSysNonLinear"
         {
            cWeightAv<tREAL8,tREAL8>  aWAvResidual;      // class for averaging residual
            for (int aKObs=0 ; aKObs< mNbObs ; aKObs++)  // parse all obs
            {
                int aIndOfObs = aVInd.at(aKObs);
                tREAL8 aRHS =  mVRHS.at(aKObs).at(aIndOfObs);  // extract the righ-hand-dised
                tREAL8 aResidual = aSol.DotProduct(mVects.at(aKObs)) - aRHS;  // compute the residual of solution
                aWAvResidual.Add(1.0,Square(aResidual)); //  add the value of residual
            }
            if  (aNbCstrTot==0)
            {
               MMVII_INTERNAL_ASSERT_bench(std::abs(mSys->VarCurSol()-aWAvResidual.Average() )<1e-5,"Bench on VarInLSqa");
            }
         }
         // [4.3] compute the uncertainty as gigen in the books
         tREAL8 aFUV =  mSys->VarCurSol() *  aRatioWithCstr ; // "Facteur unitaire de variance"

         // MMVII_INTERNAL_ASSERT_bench(std::abs(mSys->VarCurSol()-aRSUR.mVarianceCur )<1e-5,"Bench FUV on VarInLSqa");
         MMVII_INTERNAL_ASSERT_bench(std::abs(aFUV-aRSUR.FUV() )<1e-5,"Bench FUV on VarInLSqa");


         cDenseMatrix<tREAL8> aMUC = aMatNorm.Inverse() * aFUV ; // gauss markov formula
         mMoyUnc = mMoyUnc + aMUC;  // average the estimator


         // Accumulate the var/covar given  by aRSUR on free var
         for (size_t aK1=0 ; aK1<aVFreeVar.size() ; aK1++)
         {
              for (size_t aK2=0 ; aK2<aVFreeVar.size() ; aK2++)
              {
                  aMoyEstimVCV.AddElem(aK1,aK2,aRSUR.UK_VarCovarEstimate(aVFreeVar[aK1],aVFreeVar[aK2]));
              }
         }

         // [4.4] in parallel accumulate for computing moments of the random variable solution
         mStat2.Add(aSol);
         
         // [4.5] compute linear combination and compute their var/covar
               // [4.5.1] Empirical computation (if we knew the law)
         tREAL8 aS1 = aSol.DotProduct(aComb1);
         tREAL8 aS2 = aSol.DotProduct(aComb2);
         aMI2V.Add(aS1,aS2);
               // [4.5.2]  estimator, using the RSUR facilities
         aWAvCov11.Add(1.0,aRSUR.CombLin_VarCovarEstimate(0,0));
         aWAvCov12.Add(1.0,aRSUR.CombLin_VarCovarEstimate(0,1));
         aWAvCov22.Add(1.0,aRSUR.CombLin_VarCovarEstimate(1,1));

               // [4.5.3] "Economical" way, dont use explicit inverse of normal matrix but solve  N * Col12 = X
               // then Lin12 * X is Lin12 N-1 Col12
         aMoyCov12 = aMoyCov12 + (aLinComb12 * aMatNorm.Solve(aColComb12)) * aFUV;
    }

    //-------------------- [5] finally compare the empiricall solution with theory

               // --- [5.1] finish computation 
    mMoyUnc = mMoyUnc * (1.0/double (mDecompos.MulBase())) ;  // average all the estimator
    aMoyEstimVCV = aMoyEstimVCV * (1.0/double (mDecompos.MulBase())) ;  // average all the estimator
    aMoyCov12 = aMoyCov12 * (1.0/double (mDecompos.MulBase())) ;  // idem 4 linear comb
    mStat2.Normalise();  // normalise the moment matrix of solutions
    aMI2V. Normalize();  // normalis moment for the 2 linear combination

               // --- [5.2] a small test, check that the average is not biased estimator
    MMVII_INTERNAL_ASSERT_bench(mCommonP.L2Dist(mStat2.Moy())<1e-5,"Avg in cBenchLstSqEstimUncert");

               // --- [5.3] Finally ...  test the validity of covariance estimatot
    cDenseMatrix<tREAL8> aMatCov = mStat2.Cov();

    for (int aK1=0 ; aK1<mDim ; aK1++)
    {
        for (int aK2=0 ; aK2<mDim ; aK2++)
        {
            if (aVIsFree.at(aK1) && aVIsFree.at(aK2)) 
            {
               tREAL8 aVUC = mMoyUnc.GetElem(aK1,aK2);
               tREAL8 aVCov = aMatCov.GetElem(aK1,aK2);

StdOut() <<  "K1K2 " << aK1 << " " << aK2 << " " << aVUC << " " << aVCov << "\n";
               MMVII_INTERNAL_ASSERT_bench(std::abs(aVUC-aVCov)<1e-5,"Variance estimator ");
            }
        }
    }

    for (size_t aK1=0 ; aK1<aVFreeVar.size() ; aK1++)
    {
        for (size_t aK2=0 ; aK2<aVFreeVar.size() ; aK2++)
        {
            tREAL8 aVUC = aMoyEstimVCV.GetElem(aK1,aK2);
            tREAL8 aVCov = aMatCov.GetElem(aVFreeVar[aK1],aVFreeVar[aK2]);
            MMVII_INTERNAL_ASSERT_bench(std::abs(aVUC-aVCov)<1e-5,"Lib-SUR Variance estimator ");
        }
    }

               // --- [5.4]   make the test with linear combination -----------
    MMVII_INTERNAL_ASSERT_bench(RelativeDifference(aMI2V.S11(),aWAvCov11.Average())<1e-5,"Variance estimator ");
    MMVII_INTERNAL_ASSERT_bench(RelativeDifference(aMI2V.S12(),aWAvCov12.Average())<1e-5,"Variance estimator ");
    MMVII_INTERNAL_ASSERT_bench(RelativeDifference(aMI2V.S22(),aWAvCov22.Average())<1e-5,"Variance estimator ");

    MMVII_INTERNAL_ASSERT_bench(RelativeDifference(aMoyCov12.GetElem(0,0),aWAvCov11.Average())<1e-5,"Variance estimator ");
    MMVII_INTERNAL_ASSERT_bench(RelativeDifference(aMoyCov12.GetElem(0,1),aWAvCov12.Average())<1e-5,"Variance estimator ");
    MMVII_INTERNAL_ASSERT_bench(RelativeDifference(aMoyCov12.GetElem(1,1),aWAvCov22.Average())<1e-5,"Variance estimator ");
/*
    StdOut() << " SSS " <<  aMI2V.S11() << " SSS " <<  aMI2V.S12() << " SSS " <<  aMI2V.S22() << "\n";
    aMoyCov12.Show();
*/
    //getchar();


    if (isDemoTest)
    {
        StdOut() << "\n******** TEST with ,"  
                 <<  " Mode=" << E2Str(aMode) 
                 << " Dim=" << mDim  
                 << " FixV=" << aVFrozen 
                 << " VIndCstr=" << aVVIndCstr 
                 << " ***************\n";

        StdOut() << " ------------------ COV ---- \n";
        aMatCov.Show() ;
        StdOut() << " ------------------ UC ---- \n";
        mMoyUnc.Show() ;
        StdOut() << " ------------------ VCV ---- \n";
        aMoyEstimVCV.Show() ;
        getchar();
    }
}

cBenchLstSqEstimUncert::~cBenchLstSqEstimUncert()
{
   delete mSys;
}

void BenchLstSqEstimUncert(cParamExeBench & aParam)
{
    if (! aParam.NewBench("LstSqUncert")) return;

    for (int aK=0 ; aK<10 ; aK++)
    {
         cBenchLstSqEstimUncert  aLstQ4B(5,{2,3,2,2,4,2,2});
         //  for (const auto aMode : {eModeSSR::eSSR_LsqDense,eModeSSR::eSSR_LsqNormSparse})
         for (const auto aMode : {eModeSSR::eSSR_LsqNormSparse,eModeSSR::eSSR_LsqDense})
         {

             aLstQ4B.DoIt(aParam.DemoTest(),aMode,{0,2},{});

             aLstQ4B.DoIt(aParam.DemoTest(),aMode,{0,2},{});
             aLstQ4B.DoIt(aParam.DemoTest(),aMode,{},{});

             aLstQ4B.DoIt(aParam.DemoTest(),aMode,{},{});
             aLstQ4B.DoIt(aParam.DemoTest(),aMode,{},{{0,1},{0,1}});
             aLstQ4B.DoIt(aParam.DemoTest(),aMode,{2},{{0,1},{0,1}});
             aLstQ4B.DoIt(aParam.DemoTest(),aMode,{},{{0,1}});
         }
    }
/*
    for (int aK=0 ; aK<1000 ; aK++)
    {
         cBenchLstSqEstimUncert  aLstQ2(2,{2,3,3,3,2,2});
         aLstQ2.DoIt(false,true,true);

         cBenchLstSqEstimUncert  aLstQ3(3,{2,3,3,3,2,2});
         aLstQ3.DoIt(false,true,true);
         cBenchLstSqEstimUncert  aLstQ4(4,{2,3,3,3,2,2});
         aLstQ4.DoIt(false,true,true);

         cBenchLstSqEstimUncert  aLstQ4B(5,{2,2,2,2,2,2,2,2});
         aLstQ4B.DoIt(false,true,true);
    }
*/
/*
*/

    aParam.EndBench();
}





};


