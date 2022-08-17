#include "TrianguRSNL.h"
#include "include/MMVII_Tpl_Images.h"
#include "include/MMVII_2Include_Serial_Tpl.h"

// 100 Iter
//                  Std      X*10        W/o X=1     W/o X=1 X*10
// SomL2RUk : 0.00371639 ,  0.0911945   0.00393093   0.108162
//   PtsRUk : 0.00370678 ,  0.0916735   0.0051081    0.604912






/** \file cNetPropCov.cpp
    \brief test the covariance propagation 

     Test the covariance propgation on network triangulation.

     Let note the global network (class cCovNetwork) :


      ... Pm12 --  P02  --  P12 -- P22
           |   \/   |   \/   |  \/   |
           |   /\   |   /\   |  /\   |
      ... Pm11 --  P01  --  P11 -- P21
           |   \/   |   \/   |  \/   |
           |   /\   |   /\   |  /\   |
      ... Pm10 --  P00  --  P10 -- P20
          ............................

      For each point (except when it would overflow) we create a 4 point subnetwork (class cElemNetwork),
      for example whith origin P01 we create the ABCD network and  memorizing the homologous function H:
  

            C_k -- D_k       H_k(P01) = A_k
             |  \/  |        H_k(P11) = B_k
             |  /\  |        H_k(P02) = C_k
            A_k -- B_k       H_k(P12) = D_k

                 //  kth-subnetwork //




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

MMVII TestCovProp   PtsRUk  NbICP=10 NbTest=1

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

    MMVII TestCovProp   PtsRFix  NbICP=100 NbTest=1

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
show that we MUST consider the tranfer mapping as an unknown (using, for example, schurr complement).

           -----------------------------------------------

With covariance propag, rotation unknown, and decomposition as sum of sqaure we get :

     MMVII TestCovProp   SomL2RUk  NbICP=10  NbTest=1
     0 RResiduals :   0.722085
     1 RResiduals :   0.0577052
     2 RResiduals :   0.000452253
     3 RResiduals :   2.71006e-08
       ...
     9 RResiduals :   3.8805e-15

The results are pretty identic to first case (set point, unkonwn rot), which is not a surprise,
with perfect data (no noise on observation) they both converge to the solution pretty fast.


            ======================================
            =====    ACURRACY ANALYSIS  ==========
            ======================================

Now we make the analysis of pro (and  cons ?) of covariance  analysis when the obsrvation are noisy. Basically
the conclusion of experiment is coherent with theoreticall analysis:

    * covariance is always superior or equal to basic point propagation
    * when the sub-network are well conditionned, the covariance and points propagation give the same results
    * with ill conditionned sub-newtork , the difference can be significative (up to 5 in our example)

              
                   ===============   Paramater description ==============

Here is a typical command we have run to make the test :
MMVII TestCovProp SomL2RUk  NbICP=10 NbTest=100 NoiseDist=1e-3  WConf=[0,1,1,1,1] SzMainN=5

In this command :
    NoiseDist=1e-3 -> noise we add on obsevation on distannce, it's proportionnal to the distance, we keep it low
              because are interested by relative performances, ans want to be sure of convergence

    NbICP=10 ->  mean 10 iteration to convergence (not important as long as we reach convergence)
    NbTest=100 -> number of test we do to make some average
    RXY=1.0 -> modifiy the geometry , multiply all the x-coordinate by this factor
    SzMainN=2 ->  size of the network (herer [-2,2]x[-2,2] -> give 25 points)
    WConf=[0,1,1,1,1] -> topology of subnetwork we detail bellow, it is by far, the more complex

                   ===============   WConf Paramater ==============

1) standard   [1,0,0,0,0]

If we indicate WConf=[1], it correspond to default connexion in the subnetwork as described in //  kth-subnetwork //.
Identically we could have indicate  [1,0] ... or [1,0,0,0,0], all omitted element are equivalent to 0.

2) Low connexion   [0,1,0,0,0]  or   [0,0,1,0,0]

If we indicate WConf=[0,1], it correspond to a subnetwork where we have supress the left vertical edge ,
WConf=[0,0,1]  correspond to a subnetwork where bottom horizontale edge is supressed. Purpose of this supression
is to decrease the strenght of the subtnwork (or doing it less well conditionned).

            C_k -- D_k                       C_k -- D_k  
             |  \/                            |  \/  |   
             |  /\                            |  /\  |      
            A_k -- B_k                       A_k    B_k    

         //  WConf=[0,1,0] //            //  WConf=[0,0,1] //

3) Low connexion & change mapping [0,0,0,1,0]  or   [0,0,0,0,1]

If we indicate WConf=[0,0,0,1], it correspond to  the same connexion inside the subnetwork as [0,1], but with
the differnce that D and B are not mapped to the same point in the main network, in fact they are mapped the
extrem right point on the same line. 


            C_k -- D_k       H_k(Pm10) = A_k
             |  \/           H_k(P20)  = B_k
             |  /\           H_k(Pm11) = C_k
            A_k -- B_k       H_k(P21) = D_k

 Identically with WConf=[0,0,0,0,1] , we have conexion as [0,0,1]  with a mapping of A and B to the bottom of 
the main newtork.

The purpose of this option is two have even less conditionned netork in the sub-network, without changin the main
network.

4) Conbining configuration.

It is possible to conbine these configuration, for example [0,1,1] we will compute covariance (or point prop)
with the two different set of subnetwork corresponding to [0,1,0] and [0,0,1].  Not that we dont merge the network,
so the result are different from [1,0,0@


                   ===============  Line of result   ==============
A typical line of result will be :


-> AVG=0.00746706 RefAvg=0.00744771 Med=0.00658489 RefMed=0.00655037

Where :
    AVG : is the average of square residual with point or covariance propagation
    RefAvg : residual with the full network, +- the reference
    Med and  RefMed are the median and tested method and reference


                   ===============  TEST MADE   ==============

The following experiment have been made. Each experiment has been made with PtsRUk & SomL2RUk

     --------- (1) STD :  -----------------

MMVII TestCovProp SomL2RUk  NbICP=10 NbTest=1000 NoiseDist=1e-3  WConf=[1,0,0]  RXY=1.0 SzMainN=2

We use the standard network.

     --------- (2)  X*10    -----------------
MMVII TestCovProp SomL2RUk  NbICP=10 NbTest=1000 NoiseDist=1e-3  WConf=[1,0,0]  RXY=10.0 SzMainN=2

We make a dilatation of 10 on W axes.


     --------- (3) LowCnX :  -----------------
MMVII TestCovProp SomL2RUk  NbICP=10 NbTest=1000 NoiseDist=1e-3  WConf=[0,1,0]  RXY=1.0 SzMainN=2

We suppress a conexion in subnetwork.

     --------- (4)  LowCnX X*10 :  -----------------
MMVII TestCovProp SomL2RUk  NbICP=10 NbTest=1000 NoiseDist=1e-3  WConf=[0,1,0]  RXY=10.0 SzMainN=2

We suppress a conexion in subnetwork and make a dilatation. 

     --------- (5) LowCn45  -----------------
MMVII TestCovProp SomL2RUk  NbICP=10 NbTest=100 NoiseDist=1e-3  WConf=[0,0,0,1,1] SzMainN=5

We add the two network that are badly configured (cnxion and geometry) in X and Y

     --------- (6) LowCn2345  -----------------
MMVII TestCovProp SomL2RUk  NbICP=10 NbTest=100 NoiseDist=1e-3  WConf=[0,1,1,1,1] SzMainN=5

We add the 4 network that are badly connected.

                   ==============================================================
                   ==============================================================
                   ===============  TABLE OF RESULT AND ANALYSIS   ==============
                   ==============================================================
                   ==============================================================



    ===================== TABLE for Median residual ===========================

            STD           X*10         LowCnX      LowCnX X*10    LowCn4      LowCn45    LowCn2345
SomL2RUk  0.00376762    0.0957888    0.00414344    0.137719      0.0389798   0.0127058   0.00658489
PtsRUk    0.00375196    0.0951763    0.00547995    0.3674        0.0725622   0.0435764   0.0234647
Ref       0.00369961    0.0945578    0.00376001    0.0949621     0.00678199  0.006230    0.00655037

    ===================== TABLE for average residual ===========================

    todo -> qualitatively, same tendancy than median


    ===================== Analyse of results ===========================

STD, X*10 :  for this two test, the results are pretty the same for all 3 methods of solving.  In fact the
             subnetwork are fully connected and their resolution is quite accurate.

             For X*10, the geometry is poor + there is an amplification of coordinate, the residual are mutliplied
             by 30, but it the same for the 3 methods.


LowCnX :  the subtnework is not fully connected, there is a weakness in the estimation of position of B and D
          relatively to A and C.

          The reference method is 10% better than Cov/Prop, which is significative. Most important
          Cov/Prop is 25% better than points/prop.  We can interprete it this way :

                 * the distance between BD will be poorly estimated in a network, but it will be
                   acccurately estimated in the right network (where BD be AC)

                *  the point/propagtion does not separate the two estimation of BD, while 
                   cov propagation does


LowCnX  X*10 :  is more or less the same, simply the geometry exacerbates the difference :
                  
                 * cov-prop is ~50% higher than referene
                 * point-prop is ~300% higher than reference
                 

LowCn4 :   the subnetwork are poor in geometry and in topoloy: 

             *   CovProp is  ~5 times higher than reference
             *   PtsProp is  ~11 times higher than reference

          both results are poor , even if cov-prop is ~ 2 times better than point-prop.

LowCn45 :   here we have two poor set of sub network, but that are complementary, one being poor
            vertically, the other being poor horizontally.  Not surprisignly, in the both cases,
            the results are improved compared to LowCn4 (we have more measures).
            But what is more interesing is that :

                 *  for CovProp the factor of gain between LowCn4 and LowCn45 is  3.1
                 *  for Pointrop the factor  is 1.66
                 * at the end CovProp is ~ 3.5 better than PointProp and "only" twice as bad as reference

            we can interpret this , because the two subnetwork have accuracy/unaccuracy in different direction,
            this information is vehiculated by cov-prop, while point-prop make a blind merge.

LowCn2345 :   maybe the best advocating for cov-prop ...

              The residual are almost identic between cov-prop and reference method, while point-prop
              is almost 4 times worst than the others.

              We interpret this way :

                  * in the 4 networks, there is the information for globally descring the newtok
                  * the information in  config 4 and 5 is very noisy in some direction and average blindly by point prop
                  * conversely cov-prop make a rigourous weighted average

                   ===============   ANNEX            ==============
                   ===============  LIST of Command   ==============


Case std :
MMVII TestCovProp SomL2RUk  NbICP=10 NbTest=1000 NoiseDist=1e-3  WConf=[1,0,0]  RXY=1.0 SzMainN=2
-> AVG=0.00423711 RefAvg=0.00416625 Med=0.00376762 RefMed=0.00369961

MMVII TestCovProp PtsRUk  NbICP=10 NbTest=1000 NoiseDist=1e-3  WConf=[1,0,0]  RXY=1.0 SzMainN=2 
-> AVG=0.00423332 RefAvg=0.00416625 Med=0.00375196 RefMed=0.00369961

Case X*10
MMVII TestCovProp SomL2RUk  NbICP=10 NbTest=1000 NoiseDist=1e-3  WConf=[1,0,0]  RXY=10.0 SzMainN=2
-> AVG=0.116331 RefAvg=0.114131 Med=0.0957888 RefMed=0.0945578
MMVII TestCovProp PtsRUk  NbICP=10 NbTest=1000 NoiseDist=1e-3  WConf=[1,0,0]  RXY=10.0 SzMainN=2
-> AVG=0.116915 RefAvg=0.114131 Med=0.0951763 RefMed=0.0945578

Case LowCnX :
MMVII TestCovProp SomL2RUk  NbICP=10 NbTest=1000 NoiseDist=1e-3  WConf=[0,1,0]  RXY=1.0 SzMainN=2
-> AVG=0.00463096 RefAvg=0.00429547 Med=0.00414344 RefMed=0.00376001
MMVII TestCovProp PtsRUk  NbICP=10 NbTest=1000 NoiseDist=1e-3  WConf=[0,1,0]  RXY=1.0 SzMainN=2
-> AVG=0.00605871 RefAvg=0.00429547 Med=0.00547995 RefMed=0.00376001


Case LowCnX, X*10 :
MMVII TestCovProp SomL2RUk  NbICP=10 NbTest=1000 NoiseDist=1e-3  WConf=[0,1,0]  RXY=10.0 SzMainN=2
-> AVG=0.152689 RefAvg=0.114299 Med=0.137719 RefMed=0.0949621
MMVII TestCovProp PtsRUk  NbICP=10 NbTest=1000 NoiseDist=1e-3  WConf=[0,1,0]  RXY=10.0 SzMainN=2
-> AVG=0.727023 RefAvg=0.114299 Med=0.3674 RefMed=0.0949621

Case LowCn45
MMVII TestCovProp SomL2RUk  NbICP=10 NbTest=100 NoiseDist=1e-3  WConf=[0,0,0,1,1] SzMainN=5
-> AVG=0.013597 RefAvg=0.00734164 Med=0.0127058 RefMed=0.0062305
MMVII TestCovProp PtsRUk  NbICP=10 NbTest=100 NoiseDist=1e-3  WConf=[0,0,0,1,1] SzMainN=5
-> AVG=0.0482969 RefAvg=0.00734164 Med=0.0435764 RefMed=0.0062305

Case LowCn4
MMVII TestCovProp SomL2RUk  NbICP=10 NbTest=100 NoiseDist=1e-3  WConf=[0,0,0,1,1] SzMainN=5
AVG=0.0489416 RefAvg=0.00746846 Med=0.0389798 RefMed=0.00678199
MMVII TestCovProp PtsRUk  NbICP=10 NbTest=100 NoiseDist=1e-3  WConf=[0,0,0,1,0] SzMainN=5
AVG=0.0802936 RefAvg=0.00746846 Med=0.0725622 RefMed=0.00678199

Case  LowCn2345
MMVII TestCovProp SomL2RUk  NbICP=10 NbTest=100 NoiseDist=1e-3  WConf=[0,1,1,1,1] SzMainN=5
-> AVG=0.00746706 RefAvg=0.00744771 Med=0.00658489 RefMed=0.00655037
MMVII TestCovProp PtsRUk  NbICP=10 NbTest=100 NoiseDist=1e-3  WConf=[0,1,1,1,1] SzMainN=5
-> AVG=0.024945 RefAvg=0.00744771 Med=0.0234647 RefMed=0.00655037

Case LowCnX+LowConY , X*10 :
MMVII TestCovProp SomL2RUk  NbICP=10 NbTest=1000 NoiseDist=1e-3  WConf=[0,1,1]  RXY=10.0 SzMainN=2
-> AVG=0.118326 RefAvg=0.116872 Med=0.100678 RefMed=0.0976332
MMVII TestCovProp PtsRUk  NbICP=10 NbTest=1000 NoiseDist=1e-3  WConf=[0,1,1]  RXY=10.0 SzMainN=2
-> AVG=0.378365 RefAvg=0.116872 Med=0.20398 RefMed=0.0976332



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
           typedef cPNetwork<Type>           tPNet;

           cCovNetwork
           (
                const std::vector<double> & aWeightConfigs,
                double aWeightGCM,
                eModeTestPropCov,
                eModeSSR,
                cRect2,
                const cParamMainNW &,
                cParamSparseNormalLstSq * = nullptr
           );
           ~cCovNetwork();

           void PostInit() override;  ///< method that does the complete init

           /**  Solve the network using cov-prop on small ones ,

                Parameter CheatMT => Cheating Mappingg Transfer, if not 0, use (with weight CheatMT) the 
                real coordinate to compute the geometric mappinf (rotation) bewteen big/small. This is 
                obviously cheating as in real life we dont know the value of coordinates in big (this is
                what we want to compute).  Just for tuning and trying to understand why the fix/rotation case
                converge so slowly
           */
           Type SolveByCovPropagation(double aCheatMT,int aNbIter);

            /** create some noise in the estimation of distance, to make it +or- realistic assure via a
                dictionnary that the result is always the same */
            Type FactRandomizeObsDist(const cPt2di & aP1,const cPt2di & aP2);

            ///  Redefines to add noise eventually
            Type ObsDist(const tPNet & aPN1,const tPNet & aPN2) const override;
     private :
	   double                            mWeightGaugeCovMat; ///< Gauge for computing cov matrices on small networks
           eModeTestPropCov                  mModeTPC;  ///<  Mode : Matric, Sum L2, Pts ...  
           std::vector<cElemNetwork<Type> *> mVNetElem; ///<  Elementary networks
           std::vector<Type>                 mTabErrorEstimDist; ///< Memorize the random weigth on pair of points
           std::vector<double>               mWeightConfigs;  ///< weights for different config
};

/**  Class for implemanting an "elementary = small" newtork,
     on which we will compute covariance that will be transfered
     in the "big" network
*/

template <class Type>  class  cElemNetwork : public cMainNetwork<Type>
{
    public :
        typedef cCovNetwork<Type>        tMainNW;
        typedef cPtxd<Type,2>             tPt;
        typedef cPNetwork<Type>           tPNet;


        cElemNetwork
        (
               eModeTestPropCov,  ///< mode of propag (cov/sum l2/pts) and  (fix/uk) 
               bool  LowConx,     ///< Supress cnxion between x=1
               bool  LowCony,     ///< Supress cnxion between y=1
               tMainNW & aMainW,  ///< the main network it belongs to
               const cRect2 & aRectLoc,  ///< rectangle, typicalyy [0,2[x[0,2[
               const cPt2di & aSzHomMain
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

        ///  Redefines to add noise eventually
        Type ObsDist(const tPNet & aPN1,const tPNet & aPN2) const override;

        ///   
        bool OwnLinkingFiltrage(const cPt2di & aP1,const cPt2di & aP2) const override;


        
    private :
         const cPt2di &  IndOfMainHom(const tPNet & aPN) const
         {
               return MainHomOfInd(aPN.mInd).mInd;
         }
         /// return for each node of the network, its homologous in the big one
         cPNetwork<Type> & MainHom(const tPNet & aPN) const
         {
               return MainHomOfInd(aPN.mInd);
         }
         /// return for each INDEX  of the network, its homologous in the big one
         cPNetwork<Type> & MainHomOfInd(const cPt2di & anInd) const
         {
               return mMainNW->PNetOfGrid(mBoxM.P0() + MulCByC(mSzAmpl,anInd) );
         }

         eModeTestPropCov         mModeTPC;  ///< mode propag cov
         bool                     mLowConX;  ///< supress cnx  between x=1
         bool                     mLowConY;  ///< supress cnx  between y=1
	 bool                     mRotUk;    ///< is the rotation unknown in this mode
	 bool                     mL2Cov;    ///< is it a mode where cov is used as sum a square linear
	 bool                     mPtsAtt;   ///<  Mode attach directly topoint
         tMainNW *                mMainNW;   ///<  The main network it belongs to
         cRect2                   mBoxM;     ///<  Box of the network
         cPt2di                   mSzAmpl;   ///<  Sz of correspond box in main
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
         const std::vector<double> & aWeightConfigs,
         double                    aWGCM,
         eModeTestPropCov          aModeTPC,
         eModeSSR                  aMode,
         cRect2                    aRect,
         const cParamMainNW &      aParamNW,
         cParamSparseNormalLstSq * aParamLSQ
     ) :
         cMainNetwork<Type>  (aMode,aRect,false,aParamNW,aParamLSQ),
	 mWeightGaugeCovMat  (aWGCM),
	 mModeTPC            (aModeTPC),
         mWeightConfigs      (aWeightConfigs)
{
}


template <class Type>  Type cCovNetwork<Type>::FactRandomizeObsDist(const cPt2di & aP1,const cPt2di & aP2)
{
    int aIndex = this->mBoxInd.IndexeUnorderedPair(aP1,aP2);  // make a unique indexe

    static constexpr Type aDefVal = -1e10;        // value never given by rand_C
    ResizeUp(mTabErrorEstimDist,aIndex+1,aDefVal);  // assure tab of val has right size
    if (mTabErrorEstimDist.at(aIndex) == aDefVal)
    {
        mTabErrorEstimDist.at(aIndex) = Type(RandUnif_C());
    }

    return  mTabErrorEstimDist.at(aIndex);
}

template <class Type> 
        void cCovNetwork<Type>::PostInit() 
{
     // 1-  First call the usual initialisation to create the nodes
     cMainNetwork<Type>::PostInit();

     // 2- Now create the sub network
     for (int aKConf =0 ; aKConf<int(mWeightConfigs.size()) ; aKConf++)
     {
         double aWeightConf = mWeightConfigs[aKConf];
         cPt2di aSzElem(2,2);
         cPt2di aAmpl(1,1);
         cRect2  aOriginsSubN(this->mBoxInd.P0(),this->mBoxInd.P1()-aSzElem+cPt2di(1,1));
         if (aWeightConf>0)
         {
             bool  aLowCnX = (aKConf!=0) && ((aKConf%2)==1);  // LowConx =>  1,3,5 ...
             bool  aLowCnY = (aKConf!=0) && ((aKConf%2)==0);  // LoxConY => 2,4,6,...


               // rectangle containings all origins of sub-networks
             for (const auto & aPix: aOriginsSubN)  // map origins
             {
                 if (aKConf==3)
                 {
                    aAmpl.x() =  this->mBoxInd.P1().x()- aPix.x()-1;
                 }
                 if (aKConf==4)
                 {
                    aAmpl.y() =  this->mBoxInd.P1().y()- aPix.y()-1;
                 }
                 cRect2 aRect(aPix,aPix+aSzElem);
                 auto aPtrN = new cElemNetwork<Type>(mModeTPC,aLowCnX,aLowCnY,*this,aRect,aAmpl);  // create the sub network
                 aPtrN->PostInit(); // finish its initalisattion, that will use "this" (the main network)
                 mVNetElem.push_back(aPtrN);
                 //  compute solution and covariance in each network
                 Type aRes = aPtrN->ComputeCovMatrix(mWeightGaugeCovMat,10);
                 // consistancy, check that the sub-network reach convergence, but only when observation are exact
	         if ((aRes>=1e-8) && (this->mParamNW.mNoiseOnDist==0))
                 {
                     StdOut() << " Residual  " << aRes << "\n";
	             MMVII_INTERNAL_ASSERT_bench(false,"No conv 4 sub net");
                 }
             }
         }
     }
}

template <class Type>  cCovNetwork<Type>::~cCovNetwork()
{
     DeleteAllAndClear(mVNetElem);
}


template <class Type>  Type cCovNetwork<Type>::SolveByCovPropagation(double aCheatMT,int aNbIter)
{
     Type   aResidual = 0.0;
     for (int aTime=0 ; aTime<aNbIter ; aTime++) // make aNbIter iteration
     {
         // compute and print the difference comuted values/ground truth
	 aResidual = this->CalcResidual() ;
	 StdOut()   << aTime <<  " RResiduals :   " << aResidual <<  "\n";

          // for all subnetwork propagate the covariance
          for (auto & aPtrNet : mVNetElem)
             aPtrNet->PropagCov(aCheatMT);

          //  Add a gauge constraint for the main newtork, as all subnetnwork are computed up to a rotation
	  this->AddGaugeConstraint(10.0);
	  this->mSys->SolveUpdateReset();  // classical gauss jordan iteration

     }
     StdOut()  <<  "\n";
     return aResidual;
}

template <class Type> Type  cCovNetwork<Type>::ObsDist(const tPNet & aPN1,const tPNet & aPN2) const 
{
   Type aResult =  cMainNetwork<Type>::ObsDist(aPN1,aPN2);
   Type aFactR =   const_cast<cCovNetwork<Type> *>(this)->FactRandomizeObsDist(aPN1.mInd,aPN2.mInd);
   return aResult * (1 + aFactR*this->ParamNW().mNoiseOnDist);
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
      bool             isLowConX,
      bool             isLowConY,
      tMainNW & aMainNW,
      const cRect2 & aBoxM,
      const cPt2di & aSzAmpl
  ) :
        // We put the local box with origin in (0,0) because frozen point are on this point
          cMainNetwork<Type>       (eModeSSR::eSSR_LsqDense,cRect2(cPt2di(0,0),aBoxM.Sz()),false,aMainNW.ParamNW()),
	  mModeTPC                 (aModeTPC),
          mLowConX                 (isLowConX),
          mLowConY                 (isLowConY),
	  mRotUk                   (MatchRegex(E2Str(mModeTPC),".*Uk")),
	  mL2Cov                   (MatchRegex(E2Str(mModeTPC),"SomL2.*")),
	  mPtsAtt                  (MatchRegex(E2Str(mModeTPC),"Pts.*")),
          mMainNW                  (&aMainNW),
          mBoxM                    (aBoxM),
          mSzAmpl                  (aSzAmpl),
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


///  Distance observed between 2 points, add noise
template <class Type> Type  cElemNetwork<Type>::ObsDist(const tPNet & aPN1,const tPNet & aPN2) const 
{
   Type aResult =  cMainNetwork<Type>::ObsDist(aPN1,aPN2);
   Type aFactR = mMainNW->FactRandomizeObsDist(IndOfMainHom(aPN1),IndOfMainHom(aPN2));
   return aResult * (1 + aFactR*this->ParamNW().mNoiseOnDist);
}

/*  Compute the ground truth from the index, the defaut value is randomization, this redefinition
    make the small network an exact copy, up to an arbitray rotatin, of the corresping subnetwork
    in the big one.
*/
template <class Type> cPtxd<Type,2>  cElemNetwork<Type>::ComputeInd2Geom(const cPt2di & anInd) const
{
  
     cPNetwork<Type> &  aPMain = MainHomOfInd(anInd); // get corresponding point
     tPt aP = aPMain.mTheorPt;  // get the ground truch point in big network
     aP = mMainNW->SimInd2G().Inverse(aP) ;  // go back to index (perturbated)
     aP = this->mSimInd2G.Value(aP);  // transfom the index using the similitude of the newtork
     
     return aP;
}

template <class Type> 
   bool  cElemNetwork<Type>::OwnLinkingFiltrage(const cPt2di & aP1,const cPt2di & aP2) const
{
   
   if ((aP1.x()==1) && (aP2.x()==1) && mLowConX) return false;
   if ((aP1.y()==1) && (aP2.y()==1) && mLowConY) return false;

   return true;
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
         int               mNbTest;


         cParamMainNW           mParam;
	 double                 mWeightGaugeCovMat;
	 double                 mWCheatMT;
         std::vector<double>    mWeightConfig;
         cCovNetwork<tREAL8> *  mMainNet;
         double                 mRatioXY;
};


cAppli_TestPropCov::cAppli_TestPropCov(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli         (aVArgs,aSpec),
   mSzMainN             (2),
   mSzSubN              (2),
   mNbItCovProp         (10),
   mNbTest              (20),
   mWeightGaugeCovMat   (1.0),
   mWCheatMT            (0.0),
   mWeightConfig        ({1.0}),
   mRatioXY             (1.0)
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
           << AOpt2007(mNbTest, "NbTest","Number of iteration test done",{eTA2007::HDV})
           << AOpt2007(mNbItCovProp, "NbICP","Number of iteration for cov prop",{eTA2007::HDV})
           << AOpt2007(mWeightGaugeCovMat, "WGCM","Weight for gauge in covariance matrix of elem networks",{eTA2007::HDV})
           << AOpt2007(mWCheatMT, "WCMT","Weight for \"cheating\" in map transfert",{eTA2007::HDV})
           << AOpt2007(mParam.mAmplGrid2Real, "NoiseG2R","Perturbation between grid & real position",{eTA2007::HDV})
           << AOpt2007(mParam.mAmplReal2Init, "NoiseR2I","Perturbation between real & init position",{eTA2007::HDV})
           << AOpt2007(mParam.mNoiseOnDist, "NoiseDist","Noise added on dist estim,typical 1e-3",{eTA2007::HDV})
           << AOpt2007(mWeightConfig, "WConf","Weight 4 Config [Std,LowX,LowY,FarLowX,FarLowY]",{{eTA2007::ISizeV,"[1,5]"}})
           << AOpt2007(mRatioXY, "RXY","Weight 4 Config [Std,LowX,LowY,FarLowX,FarLowY]")
   ;

}


int  cAppli_TestPropCov::Exe() 
{
   if (mRatioXY>=1)
      mParam.mFactXY = cPt2dr(mRatioXY,1);
   else
      mParam.mFactXY = cPt2dr(1,1/mRatioXY);
   std::vector<tREAL8> aVRes;
   std::vector<tREAL8> aVRefRes;
   tREAL8  aSomRes = 0.0;
   tREAL8  aSomRefRes = 0.0;
   for (int aK=0 ; aK<mNbTest ; aK++)
   {
       mMainNet = new cCovNetwork <tREAL8>
	          (
                        mWeightConfig,
		        mWeightGaugeCovMat,
		        mModeTPC,
			eModeSSR::eSSR_LsqDense,
			cRect2::BoxWindow(mSzMainN),
			mParam
		  );
       mMainNet->PostInit();

       tREAL8 aRes = mMainNet->SolveByCovPropagation(mWCheatMT ,mNbItCovProp);
       aVRes.push_back(aRes);
       aSomRes += aRes;

       double aRefRes =100;
       for (int aK=0 ; aK < 10 ; aK++)
       {
         aRefRes = mMainNet->DoOneIterationCompensation(100.0,true);
       }

       aSomRefRes += aRefRes;
       aVRefRes.push_back(aRefRes);

       delete mMainNet;
   }

   StdOut() << "========RESIDUAL AT CONVERGENCE ==== : \n";
   for (int aK=0 ; aK<mNbTest ; aK++)
       StdOut() <<  " * " << aK << " " << aVRes[aK]  << "\n";
   StdOut() << " --------------------------------- : \n";
   StdOut() << "AVG="  << aSomRes/mNbTest   
            << " RefAvg=" << aSomRefRes/mNbTest 
            << " Med=" << ConstMediane(aVRes) 
            << " RefMed=" << ConstMediane(aVRefRes) 
            << "\n";

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

