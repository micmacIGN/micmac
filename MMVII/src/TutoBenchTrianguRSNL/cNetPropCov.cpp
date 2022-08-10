#include "TrianguRSNL.h"
#include "include/MMVII_Tpl_Images.h"

/** \file cNetPropCov.cpp
    \brief test the covariance propagation 

     Test the covariance propgation on network triangulation.

     Let note the global network (class cCovNetwork) :


      ... Pm12 --  P02  --  P12 -- P22
           |    /   |    /   |   /   |
           |   /    |   /    |  /    |
      ... Pm11 --  P01  --  P11 -- P21
           |    /   |    /   |   /   |
           |   /    |   /    |  /    |
      ... Pm10 --  P00  --  P10 -- P20
          ............................

      For each point (except when it would overflow) we create a 4 point subnetwork (class cElemNetwork),
      for example whith origin P01 we create the ABCD network and  memorizing the homologous function H:
  

            C_k  --  D_k       H_k(P01) = A_k
             |    /   |        H_k(P11) = B_k
             |   /    |        H_k(P02) = C_k
            A_k  --  B_k       H_k(P12) = D_k

                 //  kth subnetwork //


       For the simulation to be complete, the ABCD are the transformation of the Pij by a random rotation.

       For each sub-ntework we make a global optimiztion and get optimal solution, we memorize
       this solution and eventually the covariance matrix (this done arround the tag #CCM1 where
       is called the optimization).  At each iteration we estimate the rotation  R_k
       between two set of coordinates (Pij being curent value in glob, H_k(Pij) being final value in sub) such that:

                   R_k(Pij) = H_k(Pij)= hk_ij

       This estimation is done arround the tag #PC1

                       ==========================================

       In case of PtsRFix/PtsRUk  case (without covariance), we simply solve by leas-square :

              Sum(k,i,j) { || R_k(Pij) - hk_ij ||^2}  = 0  (1)

       This minimization being made on the sum all sub network and all points of the sub-network.
       The hk_ij (value at convergence) being the observation and the Pij the unknowns.  For the R_k,
       they can be considered as observation at each step (case  PtsRFix)  or temporary unknown
       that will be used with some schur complement like methods.

       The generation of the code, has be done in the class "cNetWConsDistSetPts" 
       in file "Formulas_Geom2D.h".
       

                       ==========================================

       The case covariance propagation is in fact not so different, let write :
                  
                R_k(Pij) =(Xk_ij,Yk_ij) and hk_ij =(xk_ij,yk_ij)

            And    Qk = (Xk_00 Yk_00 Xk_10 Yk_10 ....)
            And    qk = (xk_00 xk_00 xk_10 xk_10 ....)

        Contribution of network k to equation (1) can be writen as

                     ||Qk-qk||^2 t(Qk-qk) I (Qk-qk) = 0 (2)

         Where I is the identity matrix.  In covariance propgation we simply subsitituate the I
         matrix by the covariance matrix  A_k :


                 ||Qk-qk||_k ^2 =    t(Qk-qk) A_k (Qk-qk) = 0 (3)

          As we  want to use equation (3) in a least square system we use the diagonalization of A to
          write :

                  A_k = tRk D^2 Rk (4)   where Rk is orthogonal and D is diagonal
                  

         We can write  (4)  as:

                 ||Qk-qk||_k ^2 =  || DRk (Qk-qk)||^2  (5)

                
         Equation (5) correpond to a decomposition of (3) as a sum of square of linear form.  The tag #CCM2 
         in the code call the library that make the decomposition.

         The linear form are then used in global least square arround #PC2


         The generation of the code, has be done in the class "cNetworConsDistProgCov" 
         in file "Formulas_Geom2D.h". As in linear (5), the formula is pretty simple, it is :

              Sum(i,j){Lkx_ij Xk_ij  + Lky_ij Yk_ij}  - Cste_k
               
                      ==========================================
                      ===    RESULTS                         ===
                      ==========================================

With Set Pts, unknwon rotation we get :

    MMVII TestCovProp   PtsRUk  NbICP=10

    0 RResiduals :   0.722085
    1 RResiduals :   0.0623607
    2 RResiduals :   0.000504209
    3 RResiduals :   3.0851e-08
    4 RResiduals :   1.71559e-15
    .... 
    9 RResiduals :   2.38234e-15

Which "prove" that the implemantation is probably correct and convergence quite fast.

           -----------------------------------------------

With Set Pts, fix rotation we get :

    MMVII TestCovProp   PtsRFix  NbICP=100

    0 RResiduals :   0.722085
    1 RResiduals :   0.565646
    2 RResiduals :   0.500051
    3 RResiduals :   0.451668
      ..
    25 RResiduals :  0.232989
      ..
    500 RResiduals : 0.0580753
      ..
    2000 RResiduals : 0.000755532
      ..
    7000 RResiduals : 4.07673e-10
      ..
    12000 RResiduals :   3.58846e-15

We have a convergence but very slow. This experiment, if transferable to bundle adjustment,
show that we MUST consider the tranfer mapping as an unknown.

           -----------------------------------------------

With covariance propag, rotation unknown, and decomposition as sum of sqaure we get :

     MMVII TestCovProp   SomL2RUk  NbICP=10 
     0 RResiduals :   0.722085
     1 RResiduals :   0.0577052
     2 RResiduals :   0.000452253
     3 RResiduals :   2.71006e-08
       ...
     9 RResiduals :   3.8805e-15

The results are pretty identic to first case (set point, unkonwn rot), which is not a surprise,
with perfect data (no noise on observation) they both converge to the solution pretty fast.

With such data set, we would probably not get very interesting difference in adding noise, because
all the variable are homogeneous and there is no such bug correlation bewteen variable (to test ??).

The purpose of this game example is not to prove the theoreticall gain of the method, but rather to
asse the computationnel correctness of the code that implement it.  And also to use as canvas for
implementing it with more complex cases like "freenet bundle adjsustment".
 
*/




namespace MMVII
{
namespace NS_Bench_RSNL
{

//======================================


template <class Type>  class  cElemNetwork ;  // "Elementary" network
template <class Type>  class  cCovNetwork  ;  // "Big" network


/**   Class implementing a network that is "resolved" using covariance propagation.  It contains a subset
      of small network.

      The method is :
          * compute independanly the solution in small network, covariance and solution will be used
          * the iteraritvely estimate similitude between 2 network (big/small) and propagate covariance/solution

*/


template <class Type>  class  cCovNetwork :   public cMainNetwork<Type>
{
     public :
           cCovNetwork(double aWeightGCM,eModeTestPropCov,eModeSSR,cRect2,const cParamMainNW &,cParamSparseNormalLstSq * = nullptr);
           ~cCovNetwork();

           void PostInit() override;  ///< method that does the complete init

           /**  Solve the network using cov-prop on small ones ,

                Parameter CheatMT => Cheating Mappingg Transfer, if not 0, use (with weight CheatMT) the 
                real coordinate to compute the geometric mappinf (rotation) bewteen big/small. This is 
                obviously cheating as in real life we dont know the value of coordinates in big (this is
                what we want to compute).  Just for tuning and trying to understand why the fix/rotation case
                converge so slowly
           */
           void SolveByCovPropagation(double aCheatMT,int aNbIter);

     private :
	   double                            mWeightGaugeCovMat; ///< Gauge for computing cov matrices on small networks
           eModeTestPropCov                  mModeTPC;  ///<  Mode : Matric, Sum L2, Pts ...  
           std::vector<cElemNetwork<Type> *> mVNetElem; ///<  Elementary networks
};

/**  Class for implemanting an "elementary = small" newtork,
     on which we will compute covariance that will be transfered
     in the "big" network
*/

template <class Type>  class  cElemNetwork : public cMainNetwork<Type>
{
    public :
        typedef cMainNetwork<Type>        tMainNW;
        typedef cPtxd<Type,2>             tPt;
        typedef cPNetwork<Type>           tPNet;


        cElemNetwork
        (
               eModeTestPropCov,  ///< mode of propag (cov/sum l2/pts) and  (fix/uk) 
               tMainNW & aMainW,  ///< the main network it belongs to
               const cRect2 & aRectMain  ///< rectangle, typicalyy [0,2[x[0,2[
        );
        ~cElemNetwork();

        /**  "resolve" the small network, essentiall compute its solution and its covariance matrix, 
             eventually decompose in sum a square of
             linear form, will be used */
	Type ComputeCovMatrix(double aWeighGauge,int aNbIter);

        /**  Make one iteration of covariance propagation in the network*/
        void PropagCov(double aWCheatMT);

        int DebugN() const {return mDebugN;}  ///< accessor to Debugging number
        
        /**  Redefine the function Index->Geom, taking into account the network is copy (up to a rotation)
             of the subset of the big one */
        tPt  ComputeInd2Geom(const cPt2di & anInd) const override ;
        
    private :
         /// return for each node of the network, its homologous in the big one
         cPNetwork<Type> & MainHom(const tPNet & aPN) const
         {
               return IndMainHom(aPN.mInd);
         }
         /// return for each INDEX  of the network, its homologous in the big one
         cPNetwork<Type> & IndMainHom(const cPt2di & anInd) const
         {
               return mMainNW->PNetOfGrid(anInd+mBoxM.P0() );
         }

         eModeTestPropCov         mModeTPC;  ///< mode propag cov
	 bool                     mRotUk;    ///< is the rotation unknown in this mode
	 bool                     mL2Cov;    ///< is it a mode where cov is used as sum a square linear
	 bool                     mPtsAtt;   ///<  Mode attach directly topoint
         tMainNW *                mMainNW;   ///<  The main network it belongs to
         cRect2                   mBoxM;     ///<  Box of the network
	 int mDebugN;                        ///< identifier, was used in debuginng
         cCalculator<double> *    mCalcSumL2RUk;  ///< calculcator usde in mode som L2 with unknown rot
         cCalculator<double> *    mCalcPtsRFix;   ///< calculator used with known point/ Rot fix
         cCalculator<double> *    mCalcPtsSimVar;  ///< calculator used with known point/Rot unknown
         cDecSumSqLinear<Type>    mDSSL;           ///< structur for storing covariance as sum of square linear form
};

/* *************************************** */
/*                                         */
/*          cCovNetwork                    */
/*                                         */
/* *************************************** */


template <class Type>  
     cCovNetwork<Type>::cCovNetwork
     (
         double                    aWGCM,
         eModeTestPropCov          aModeTPC,
         eModeSSR                  aMode,
         cRect2                    aRect,
         const cParamMainNW &      aParamNW,
         cParamSparseNormalLstSq * aParamLSQ
     ) :
         cMainNetwork<Type>  (aMode,aRect,false,aParamNW,aParamLSQ),
	 mWeightGaugeCovMat  (aWGCM),
	 mModeTPC            (aModeTPC)
{
}


template <class Type> 
        void cCovNetwork<Type>::PostInit() 
{
     // 1-  First call the usual initialisation to create the nodes
     cMainNetwork<Type>::PostInit();

     // 2- Now create the sub network
     cPt2di aSz(2,2);
          // rectangle containings all origins of sub-networks
     cRect2  aOriginsSubN(this->mBoxInd.P0(),this->mBoxInd.P1()-aSz+cPt2di(1,1));

     for (const auto & aPix: aOriginsSubN)  // map origins
     {
         cRect2 aRect(aPix,aPix+aSz);
         auto aPtrN = new cElemNetwork<Type>(mModeTPC,*this,aRect);  // create the sub network
         aPtrN->PostInit(); // finish its initalisattion, that will use "this" (the main network)
         mVNetElem.push_back(aPtrN);
         //  compute solution and covariance in each network
         Type aRes = aPtrN->ComputeCovMatrix(mWeightGaugeCovMat,10);
         // consistancy, check that the sub-network reach convergence
	 if (aRes>=1e-8)
         {
             StdOut() << " Residual  " << aRes << "\n";
	     MMVII_INTERNAL_ASSERT_bench(false,"No conv 4 sub net");
         }
     }
}

template <class Type>  cCovNetwork<Type>::~cCovNetwork()
{
     DeleteAllAndClear(mVNetElem);
}


template <class Type>  void cCovNetwork<Type>::SolveByCovPropagation(double aCheatMT,int aNbIter)
{

     for (int aTime=0 ; aTime<aNbIter ; aTime++) // make aNbIter iteration
     {
         // compute and print the difference comuted values/ground truth
	 Type   aResidual = this->CalcResidual() ;
	 StdOut()   << aTime <<  " RResiduals :   " << aResidual <<  "\n";

          // for all subnetwork propagate the covariance
          for (auto & aPtrNet : mVNetElem)
             aPtrNet->PropagCov(aCheatMT);

          //  Add a gauge constraint for the main newtork, as all subnetnwork are computed up to a rotation
	  this->AddGaugeConstraint(10.0);
	  this->mSys->SolveUpdateReset();  // classical gauss jordan iteration

     }
     getchar();
}


/* *************************************** */
/*                                         */
/*          cElemNetwork                   */
/*                                         */
/* *************************************** */



template <class Type> 
  cElemNetwork<Type>::cElemNetwork
  (
      eModeTestPropCov aModeTPC,
      tMainNW & aMainNW,
      const cRect2 & aBoxM
  ) :
        // We put the local box with origin in (0,0) because frozen point are on this point
          cMainNetwork<Type>       (eModeSSR::eSSR_LsqDense,cRect2(cPt2di(0,0),aBoxM.Sz()),false,cParamMainNW()),
	  mModeTPC                 (aModeTPC),
	  mRotUk                   (MatchRegex(E2Str(mModeTPC),".*Uk")),
	  mL2Cov                   (MatchRegex(E2Str(mModeTPC),"SomL2.*")),
	  mPtsAtt                  (MatchRegex(E2Str(mModeTPC),"Pts.*")),
          mMainNW                  (&aMainNW),
          mBoxM                    (aBoxM),
          mCalcSumL2RUk            (EqNetworkConsDistProgCov(true,1,aBoxM.Sz())),
          mCalcPtsRFix             (EqNetworkConsDistFixPoints(true,1,aBoxM.Sz(),false)),
          mCalcPtsSimVar           (EqNetworkConsDistFixPoints(true,1,aBoxM.Sz(),true))
{
    // to "play the game" of covariance propagation wiht unknown transformation, the elementary network
    // will have a rotation different from the main, but it must have the same scale as we define a
    // triangulation with distance conservation
  
    this->mSimInd2G  = mMainNW->SimInd2G() * cRot2D<Type>::RandomRot(4.0).Sim();
    static int TheNumDebug=0;	
    mDebugN = ++TheNumDebug;
}

/*  Compute the ground truth from the index, the defaut value is randomization, this redefinition
    make the small network an exact copy, up to an arbitray rotatin, of the corresping subnetwork
    in the big one.
*/
template <class Type> cPtxd<Type,2>  cElemNetwork<Type>::ComputeInd2Geom(const cPt2di & anInd) const
{
  
     cPNetwork<Type> &  aPMain = IndMainHom(anInd); // get corresponding point
     tPt aP = aPMain.mTheorPt;  // get the ground truch point in big network
     aP = mMainNW->SimInd2G().Inverse(aP) ;  // go back to index (perturbated)
     aP = this->mSimInd2G.Value(aP);  // transfom the index using the similitude of the newtork
     
     return aP;
}

template <class Type> cElemNetwork<Type>::~cElemNetwork()
{
    delete mCalcSumL2RUk;
    delete mCalcPtsRFix;
    delete mCalcPtsSimVar;
}


template <class Type>  Type cElemNetwork<Type>::ComputeCovMatrix(double aWGaugeCovMatr,int aNbIter)
{
     // #CCM1    Iteration to compute the 
     for (int aK=0 ; aK<(aNbIter-1); aK++)
     {
         this->DoOneIterationCompensation(10.0,true);  // Iterations with a gauge and solve
     } 
     Type aRes = this->CalcResidual(); // memorization of residual

     // last iteration with a gauge w/o solve (because solving would reinit the covariance) 
     this->DoOneIterationCompensation(aWGaugeCovMatr,false);     


     // #CCM2  Now get the normal matrix and vector, and decompose it in a weighted sum of square  of linear forms
     if (mL2Cov)
     {
        auto  aSL = this->mSys->SysLinear();  // extract linear system
        auto aSol = this->mSys->CurGlobSol(); // extract solution
        mDSSL.Set(aSol,aSL->V_tAA(),aSL->V_tARhs());  // make the decomposition

     }

     return aRes;
}

template <class Type>  void cElemNetwork<Type>::PropagCov(double aWCheatMT)
{
    // ========  1- Estimate  the rotation between Big current network and final small network
    //              compute also indexes of point in big network

        // 1.0  declare vector for storing 
    std::vector<tPt> aVLoc;   // points of small network we have convergerd to
    std::vector<tPt> aVMain;  // current point of main network

    int aNbUkRot = mRotUk ?  3 : 0; // Number of parameters for unknown rotationn
    // Index of unknown, if Rotation unknown,  begin with 3 Tmp-Schur for rotation
    std::vector<int> aVIndUk(this->mVPts.size()*2+aNbUkRot,-1); 
 
        // 1.1  compute indexes and homologous points
    for (const auto & aPNet : this->mVPts)
    {
         const tPNet & aHomMain = this->MainHom(aPNet);
         // this index mapping is required because for example if first point has Num 2, and corresponding
         // global index if 36, the index 36 must be at place 2, after eventually rotations indexes
         aVIndUk.at(aNbUkRot+aPNet.mNumX) = aHomMain.mNumX;
         aVIndUk.at(aNbUkRot+aPNet.mNumY) = aHomMain.mNumY;

	 if(aWCheatMT<=0)
	 {
             // standard case
             aVLoc.push_back(aPNet.PCur());  // Cur point of local, where it has converger
             aVMain.push_back(aHomMain.PCur());  // Cur point of global, will evolve
	 }
	 else
	 {
             aVLoc.push_back(aPNet.mTheorPt);
             aVMain.push_back(aHomMain.mTheorPt*Type(aWCheatMT) +aHomMain.PCur()*Type(1-aWCheatMT));
	 }
    }

           // 1.2  estimate the rotation (done here  by ransac + several linearization+least square) #PC1
    Type aSqResidual;
    cRot2D<Type>  aRotM2L =  cRot2D<Type>::StdGlobEstimate(aVMain,aVLoc,&aSqResidual);

           // 1.3  make a vector of observtion/temp unkown of this rotation
    tPt  aTr   = aRotM2L.Tr();
    Type aTeta = aRotM2L.Teta();

    std::vector<Type> aVTmpRot;
    aVTmpRot.push_back(aTr.x());
    aVTmpRot.push_back(aTr.y());
    aVTmpRot.push_back(aTeta);

    // ========  2- Now make the process corresponding to different mode

    if (mPtsAtt)
    {
       /* ---------  2-A  case where we use   directly the points (no covariance)
              see cNetWConsDistFixPts , it return all the observation of the network
              For kieme point we have:
                Obs_k =  Rot(Plob_k) - PLoc_k
              The vector will be {Obs_0.x Obs_0.y  Obs_1.x .... }
       */

           // VectObs : (Trx Try Teta)  X1 Y1 X2 Y2 ...  
        std::vector<Type> aVObs  =  mRotUk ?  std::vector<Type>()  : aVTmpRot; // Rot is OR an observation OR an unknown
	int aNbObsRot = aVObs.size();  
	aVObs.resize(aVObs.size()+2*this->mVPts.size()); // extend to required size

        for (const auto & aPNet : this->mVPts)  // for all points of network
        {
             tPt aPt =    aPNet.PCur();
             // put points  of network as observation
             aVObs.at(aNbObsRot+aPNet.mNumX) = aPt.x(); 
             aVObs.at(aNbObsRot+aPNet.mNumY) = aPt.y();
        }
        if (mRotUk) // if rotation unknown use schurr complement or equivalent
        {
            cSetIORSNL_SameTmp<Type> aSetIO;
            // compute all the observations 
            this->mMainNW->Sys()->AddEq2Subst(aSetIO,mCalcPtsSimVar,aVIndUk,aVTmpRot,aVObs);
            // add it to system with schurr substitution
            this->mMainNW->Sys()->AddObsWithTmpUK(aSetIO);
        }
        else // just add observation if rotation is fix
        {
           this->mMainNW->Sys()->CalcAndAddObs(mCalcPtsRFix,aVIndUk,aVObs);
        }
    }
    else if (mL2Cov)
    {
       // ---------  2-B  case where we use  the decomposition covariance as sum of SqL,  #PC2
       cSetIORSNL_SameTmp<Type> aSetIO; // structure for schur subst
       for (const auto anElemLin : mDSSL.VElems()) // parse all linear system
       {
           cResidualWeighter<Type>  aRW(anElemLin.mW);  // the weigth as given by eigen values
           std::vector<Type> aVObs = anElemLin.mCoeff.ToStdVect(); // coefficient of the linear forme
           aVObs.push_back(anElemLin.mCste);  // cste  of the linear form
           // Add the equation in the structure
           this->mMainNW->Sys()->AddEq2Subst(aSetIO,mCalcSumL2RUk,aVIndUk,aVTmpRot,aVObs,aRW);
       }
       // Once all equation have been bufferd in aSetIO, add it to the system
       //  the unknown rotation will be eliminated
       this->mMainNW->Sys()->AddObsWithTmpUK(aSetIO);
    }
    else
    {
        // case where we directly add the covariance matrix, it was the way the method was initiated
        // obsoletr for now as : (1) slow if rotation is fix (2) if rotation is unknown, more complicated 
        // than sum of square  of linear forms

        // maintain it , in case we want to go back to this, but no comment in detail
/*
    Loc =   aSimM2L * Main

    X_loc    (Trx)     (Sx   -Sy)   (X_Main)
    Y_loc =  (Try) +   (Sy    Sx) * (Y_Main)
    
*/
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
         if (0)
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
	 eModeTestPropCov  mModeTPC;
         int               mSzMainN;
         int               mSzSubN;
         int               mNbItCovProp;


         cParamMainNW           mParam;
	 double                 mWeightGaugeCovMat;
	 double                 mWCheatMT;
         cCovNetwork<tREAL8> * mMainNet;
};


cAppli_TestPropCov::cAppli_TestPropCov(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli         (aVArgs,aSpec),
   mSzMainN             (2),
   mSzSubN              (2),
   mNbItCovProp         (10),
   mWeightGaugeCovMat   (1.0),
   mWCheatMT            (0.0)
{
}

cCollecSpecArg2007 & cAppli_TestPropCov::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return    anArgObl
           << Arg2007(mModeTPC,"Mode for Test Propag Covariance ",{AC_ListVal<eModeTestPropCov>()})
    ;
}

cCollecSpecArg2007 & cAppli_TestPropCov::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return
       anArgOpt
           << AOpt2007(mSzMainN, "SzMainN","Size of network N->[-N,N]x[NxN],i.e 2 create 25 points",{eTA2007::HDV})
           << AOpt2007(mSzSubN, "SzSubN","Size of subnetwork N->[0,N[x[0,N[",{eTA2007::HDV})
           << AOpt2007(mNbItCovProp, "NbICP","Number of iteration for cov prop",{eTA2007::HDV})
           << AOpt2007(mWeightGaugeCovMat, "WGCM","Weight for gauge in covariance matrix of elem networks",{eTA2007::HDV})
           << AOpt2007(mWCheatMT, "WCMT","Weight for \"cheating\" in map transfert",{eTA2007::HDV})
           << AOpt2007(mParam.mAmplGrid2Real, "NoiseG2R","Perturbation between grid & real position",{eTA2007::HDV})
           << AOpt2007(mParam.mAmplReal2Init, "NoiseR2I","Perturbation between real & init position",{eTA2007::HDV})
   ;
}


int  cAppli_TestPropCov::Exe() 
{
   for (int aK=0 ; aK<10 ; aK++)
   {
       mMainNet = new cCovNetwork <tREAL8>
	          (
		        mWeightGaugeCovMat,
		        mModeTPC,
			eModeSSR::eSSR_LsqDense,
			cRect2::BoxWindow(mSzMainN),
			mParam
		  );
       mMainNet->PostInit();

       mMainNet->SolveByCovPropagation(mWCheatMT ,mNbItCovProp);

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

