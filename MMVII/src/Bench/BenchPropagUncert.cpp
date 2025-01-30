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
         void DoIt(bool isDemoTest,bool WithUK,bool ShowDifRel);

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
         cStrStat2<tREAL8>            mStat2;       ///<  Used for computing covariance of solutions
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

void cBenchLstSqEstimUncert::DoIt(bool isDemoTest,bool WithFixUk,bool ShowDifRel)
{
    // ----------- [0] re-initialize the parameters --------------------
    delete mSys;
    mSys = new cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqDense,mCommonP);
    mStat2 =  cStrStat2<tREAL8>(mDim);
    mMoyUnc     = cDenseMatrix<tREAL8>(mDim,eModeInitImage::eMIA_Null);

    //  -------- [1] compute the law of each variable, ----------------
    //  it's done in such way that each law has the same variance (see Homoscedasticity in wikipedia)
    for (int aKObs=0 ; aKObs< mNbObs ; aKObs++)
    {
        int aNbVal = mVecNbByObs.at(aKObs);  // number of value for this law
        tREAL8 aAvg = mCommonP.DotProduct(mVects.at(aKObs));  // assure the each equation is centerd of the common point
        mVRHS.push_back(DiscreteRegSampledLaw(aNbVal,aAvg,mStdDev));
    }
    // bool WithFixUk = true;
    if (WithFixUk)
       mSys->SetFrozenVarCurVal(0);

    // --------- [2] ----------  parse all the possible combination of all the random variable --------
    for (int aK=0 ; aK<mDecompos.MulBase() ;aK++)
    {
         //  [2.1]   Add the linear observations of this configuration
         std::vector<int>  aVInd =  mDecompos.DecomposSizeBase(aK);  // decomp the index mNbObs  sub index
         for (int aKObs=0 ; aKObs< mNbObs ; aKObs++)
         {
             int aIndOfObs = aVInd.at(aKObs);
             tREAL8 aRHS =  mVRHS.at(aKObs).at(aIndOfObs);
             mSys->AddObservationLinear(1.0,mVects.at(aKObs),aRHS);
         }

         cDenseMatrix<tREAL8>  aMUC = mSys->SysLinear()->V_tAA();   // normal matrix , do it before Reset !!
         cResultSUR<tREAL8> aRSUR;
         tDV aSol = mSys->SolveUpdateReset(0.0,&aRSUR);  // compute the solution of this config

         aMUC = aRSUR.mtAA;

         // [2.2] Test that the variance is correctly estimated in class "cResolSysNonLinear"
         {
            cWeightAv<tREAL8,tREAL8>  aWAvResidual;      // class for averaging residual
            for (int aKObs=0 ; aKObs< mNbObs ; aKObs++)  // parse all obs
            {
                int aIndOfObs = aVInd.at(aKObs);
                tREAL8 aRHS =  mVRHS.at(aKObs).at(aIndOfObs);  // extract the righ-hand-dised
                tREAL8 aResidual = aSol.DotProduct(mVects.at(aKObs)) - aRHS;  // compute the residual of solution
                aWAvResidual.Add(1.0,Square(aResidual)); //  add the value of residual
            }
            if (!WithFixUk)
            {
               MMVII_INTERNAL_ASSERT_bench(std::abs(mSys->VarCurSol()-aWAvResidual.Average() )<1e-5,"Bench on VarInLSqa");
            }
         }
         // [2.3] compute the uncertainty as gigen in the books
         int aNbCstr = ( WithFixUk ? 1 : 0) ;
         aMUC = aMUC.Inverse() * mSys->VarCurSol() *  ((mNbObs+aNbCstr)/double(mNbObs-(mDim-aNbCstr))) ; // gauss markov formula
         mMoyUnc = mMoyUnc + aMUC;  // average the estimator

         // [2.4] in parallel accumulate for computing moments of the random variable solution
         mStat2.Add(aSol);
    }

    //-------------------- [3] finnaly compare the empiricall solution with theory

               // --- [3.1] finish computation 
    mMoyUnc = mMoyUnc * (1.0/double (mDecompos.MulBase())) ;  // average all the estimator
    mStat2.Normalise();  // normalise the moment matrix of solutions

               // --- [3.2] a small test, check that the average is not biased estimator
    MMVII_INTERNAL_ASSERT_bench(mCommonP.L2Dist(mStat2.Moy())<1e-5,"Avg in cBenchLstSqEstimUncert");

               // --- [3.3] Finally ...  test the validity of covariance estimatot
    cDenseMatrix<tREAL8> aMatCov = mStat2.Cov();
    auto aDif = aMatCov-mMoyUnc;

    if (! WithFixUk)
    {
       MMVII_INTERNAL_ASSERT_bench(aDif.DIm().L2Norm(true)<1e-5,"Covariance estimator in Bench");
    }

    if (ShowDifRel && WithFixUk)
    {
         cDenseMatrix<tREAL8>  aMCov1 = aMatCov.Crop(cPt2di(1,1),cPt2di(mDim,mDim));
         cDenseMatrix<tREAL8>  aMUC1  = mMoyUnc.Crop(cPt2di(1,1),cPt2di(mDim,mDim));

         tREAL8 aNCov = aMCov1.DIm().L2Norm() ;
         tREAL8 aNUC  =  aMUC1.DIm().L2Norm();

         tREAL8 aDifRel = (aMCov1-aMUC1).DIm().L2Norm() /(aNCov+aNUC);
         StdOut() << "DDDD " << aDifRel << "\n";
         aMCov1.Show();
         StdOut() << "------------------------------\n";
         (aMUC1 * (aNCov/aNUC)) .Show();
getchar();
    }

               // --- [3.4] In we are in demo /test, show the matrixes
    if (isDemoTest)
    {
        StdOut() <<  "\n";
    
        StdOut() << " ------------------ Dif ---- \n";
        aDif.Show() ;

        StdOut() << " ------------------ COV ---- \n";
        aMatCov.Show() ;
        StdOut() << " ------------------ UC ---- \n";
        mMoyUnc.Show() ;
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

    // cBenchLstSqEstimUncert  aLstQ5(4,{2,3,3,3,2,2},true);
    cBenchLstSqEstimUncert  aLstQ6(3,{2,3,3,3,2,2});

    aLstQ6.DoIt(aParam.DemoTest(),false,false);
    aLstQ6.DoIt(aParam.DemoTest(),true,false);
    aLstQ6.DoIt(aParam.DemoTest(),false,false);

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
/*
*/

    aParam.EndBench();
}





};


