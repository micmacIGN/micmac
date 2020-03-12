#include "include/MMVII_all.h"
#include "include/MMVII_Derivatives.h"
#include "include/MMVII_FormalDerivatives.h"

#include "ceres/jet.h"

namespace  FD = NS_MMVII_FormalDerivative;
using ceres::Jet;
using MMVII::cEpsNum;


// ========== Define on Jets two optimization as we did on formal 

template <typename T, int N>
inline Jet<T, N> square(const Jet<T, N>& f) {
  return Jet<T, N>(FD::square(f.a), 2.0*f.a * f.v);
}

template <typename T, int N>
inline Jet<T, N> cube(const Jet<T, N>& f) {
  return Jet<T, N>(FD::cube(f.a), 3.0*FD::square(f.a) * f.v);
}

//=========================================

static auto BeginOfTime = std::chrono::steady_clock::now();
double TimeElapsFromT0()
{
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(now - BeginOfTime).count() / 1e6;
}


/** \file TestDynCompDerivatives.cpp
    \brief Illustration and test of formal derivative

    {I}  A detailled example of use, and test of correctness => for the final user

    {II} An example on basic function to get some insight in the way it works

    {III} A realistic example to make performance test on different methods

*/



/* The library is in the namespace NS_MMVII_FormalDerivative, we want to
use it conveniently without the "using" directive, so create an alias FD */



/* {I}   ========================  EXAMPLE OF USE :  Ratkoswky   ==========================================

   In this first basic example, we take the same model than Ceres jet, the ratkoswky
   function.

    We want to fit a curve y= F(x) with the parametric model  [Ratko] , where b1,..,b4 are the unknown and
    x,y the observations :

         y = b1 / (1+exp(b2-b3*x)) ^ 1/b4  [Ratko]
    
    we have a set of value (x1,y1), (x2,y2) ..., an initial guess of the parameter b(i) and want
    to compute optimal value. Typically we want to use non linear least-square for that, and 
    need to compute differential of  equation [Ratko] . The MMVII_FormalDerivative
    offers service for fast differentiation. The weighted least-square is another story that we dont study here.

*/


/**   RatkoswkyResidual  : residual of the Ratkoswky equation as a  function of unknowns and observation. 

        We return a vector because this is the general case to have a N-dimentional return value 
     (i.e. multiple residual like in photogrametic colinear equation).

        This template function can work on numerical type, formula, jets ....
*/

template <class Type> 
std::vector<Type> RatkoswkyResidual
                  (
                      const std::vector<Type> & aVUk,
                      const std::vector<Type> & aVObs
                  )
{
    const Type & b1 = aVUk[0];
    const Type & b2 = aVUk[1];
    const Type & b3 = aVUk[2];
    const Type & b4 = aVUk[3];

    const Type & x  = aVObs[1];  // Warn the data I got were in order y,x ..
    const Type & y  = aVObs[0];

    pow(b1,2.7);

    // Model :  y = b1 / (1+exp(b2-b3*x)) ^ 1/b4 + Error()  [Ratko]
    return { b1 / pow(1.0+exp(b2-b3*x),1.0/b4) - y } ;
}


/**  For test Declare a literal vector of pair Y,X corresponding to observations
    (not elagant but to avoid parse file in this tutorial)
*/
typedef std::vector<std::vector<double>> tVRatkoswkyData;
static tVRatkoswkyData TheVRatkoswkyData
{
     {16.08E0,1.0E0}, {33.83E0,2.0E0}, {65.80E0,3.0E0}, {97.20E0,4.0E0}, {191.55E0,5.0E0}, 
     {326.20E0,6.0E0}, {386.87E0,7.0E0}, {520.53E0,8.0E0}, {590.03E0,9.0E0}, {651.92E0,10.0E0}, 
     {724.93E0,11.0E0}, {699.56E0,12.0E0}, {689.96E0,13.0E0}, {637.56E0,14.0E0}, {717.41E0,15.0E0} 
};


/**  Use  RatkoswkyResidual on Formulas to computes its derivatives
*/

void TestRatkoswky(const tVRatkoswkyData & aVData,const std::vector<double> & aInitialGuess)
{
    size_t aNbUk = 4;
    size_t aNbObs = 2;
    assert(aInitialGuess.size()==aNbUk); // consitency test

   //-[1] ========= Create/Init the coordinator =================  
   //-    This part [1] would be executed only one time
        // Create a coordinator/context where values are stored on double and :
        //  4 unknown (b1-b4), 2 observations, a buffer of size 100
    FD::cCoordinatorF<double>  aCFD(100,aNbUk,aNbObs);

        // Create formulas of residuals, VUk and VObs are  vector of formulas for unkown and observation
    auto  aFormulaRes = RatkoswkyResidual(aCFD.VUk(),aCFD.VObs());

        // Set the formula that will be computed
    aCFD.SetCurFormulasWithDerivative(aFormulaRes);

   //-[2] ========= Now Use the coordinator to compute vals & derivatives ================= 
       // "Push" the data , this does not make the computation, data are memporized
       // In real case, you should be care to flush with EvalAndClear before you exceed buffe (100)
     for (const auto  & aVYX : aVData)
     {
          assert(aVYX.size()==aNbObs); // consitency test
          aCFD.PushNewEvals(aInitialGuess,aVYX);
     }
        // Now run the computation on "pushed" data, we have the derivative
     const std::vector<std::vector<double> *> & aVEvals = aCFD.EvalAndClear();
     assert(aVEvals.size()==aVData.size()); // Check we get the number of computation we inserted

   //-[3] ========= Now we can use the derivatives ========================== 
   //  directly on aVEvals,  or with  interface : DerComp(), ValComp()
   //  here the "use" we do is to check coherence with numerical values and derivatives

     
    for (size_t aKObs=0; aKObs<aVData.size() ; aKObs++)
    {
         // aLineRes ->  result correspond to 1 obs
         //  storing order :   V0 dV0/dX0 dV0/dX1 .... V1 dV1/dX1 .... (here only one val)
         const std::vector<double> & aLineRes = *(aVEvals.at(aKObs));

         // [3.1] Check the value
         double aValFLine =  aLineRes[0]; // we access via the vector
         double aValInterface =   aCFD.ValComp(aKObs,0); // via the interface, 0=> first value (unique here)
         double aValStd =  RatkoswkyResidual<double>(aInitialGuess,aVData[aKObs]).at(0); // Numerical value
         assert(aValFLine==aValInterface); // exactly the same
         assert(std::abs(aValFLine-aValStd)<1e-10); // may exist some numericall effect

         // [3.2]  Check the derivate
         for (size_t aKUnk=0 ; aKUnk<aNbUk ; aKUnk++)
         {
            double aDerFLine =  aLineRes[1+aKUnk]; // access via the vector
            double aDerInterface =  aCFD.DerComp(aKObs,0,aKUnk); // via the interface
            assert(aDerFLine==aDerInterface); // exactly the same
            // Compute derivate by finite difference ; 
            // RatkoswkyResidual<double> is the "standard" function operating on numbers
            // see NumericalDerivate in "MMVII_FormalDerivatives.h" 
            double aDerNum =  FD::NumericalDerivate
                              (RatkoswkyResidual<double>,aInitialGuess,aVData[aKObs],aKUnk,1e-5).at(0);

            // Check but with pessimistic majoration of error in finite difference
            assert(std::abs(aDerFLine-aDerNum)<1e-4);
         }
    }
     std::cout << "OK  TestRatkoswky \n";
}


/* {II}  ========================    ==========================================

   This second example, we take a very basic example to analyse some part of the
*/
template <class Type> 
std::vector<Type> FitCube
                  (
                      const std::vector<Type> & aVUk,
                      const std::vector<Type> & aVObs
                  )
{
    const Type & a = aVUk[0];
    const Type & b = aVUk[1];

    const Type & x  = aVObs[0];  
    const Type & y  = aVObs[1];
     
    // Naturally the user would write that
    if (false)
    {
       Type F = (a+b *x);
       return {F*F*F - y};
       return {cube(a+b *x)- y};
    }


    // but here we want to test the reduction process
    return {(a+b *x)*(x*b+a)*(a+b *x) - y};
}

void InspectCube()
{
    std::cout <<  "===================== TestRatkoswky  ===================\n";

    // Create a context where values are stored on double and :
    //    2 unknown, 2 observations, a buffer of size 100
    //    aCFD(100,2,2) would have the same effect for the computation
    //    The variant with vector of string, will fix the name of variables, it
    //    will be usefull when will generate code and will want  to analyse it
    FD::cCoordinatorF<double>  aCFD(100,{"a","b"},{"x","y"});

    // Inspect vector of unknown and vector of observations
    {  
        for (const auto & aF : aCFD.VUk())
           std::cout << "Unknowns : "<< aF->Name() << "\n";
        for (const auto & aF : aCFD.VObs())
           std::cout << "Observation : "<< aF->Name() << "\n";
    }

    // Create the formula corresponding to residual
    std::vector<FD::cFormula<double>>  aVResidu = FitCube(aCFD.VUk(),aCFD.VObs());
    FD::cFormula<double>  aResidu = aVResidu[0];
 
    // Inspect the formula 
    std::cout  << "RESIDU FORMULA, Num=" << aResidu->NumGlob() << " Name=" <<  aResidu->Name() <<"\n";
    std::cout  << " PP=[" << aResidu->InfixPPrint() <<"]\n";

    // Inspect the derivative  relatively to b
    auto aDerb = aResidu->Derivate(1);  
    std::cout  << "DERIVATE FORMULA , Num=" << aDerb->NumGlob() << " Name=" <<  aDerb->Name() <<"\n";
    std::cout  << " PP=[" << aDerb->InfixPPrint() <<"]\n";

    // Set the formula that will be computed
    aCFD.SetCurFormulasWithDerivative(aVResidu);
    
    // Print stack of formula
    std::cout << "====== Stack === \n";
    aCFD.ShowStackFunc();

    getchar();
}

/* {III}  ========================  Test perf on colinearit equation =================================

       On this example 

*/



#define NB_UK  19
#define NB_OBS 11

/*  Capital letter for 3D variable/formulas and small for 2D */
template <class TypeUk,class TypeObs> std::vector<TypeUk> FraserCamColinearEq
                  (
                      const std::vector<TypeUk> & aVUk,
                      const std::vector<TypeObs> & aVObs
                  )
{
    assert (aVUk.size() ==NB_UK) ;// FD::UserSError("Bad size for unknown");
    assert (aVObs.size()==NB_OBS) ;// FD::UserSError("Bad size for observations");

    // 0 - Ground Coordinates of projected point
    const auto & XGround = aVUk[0];
    const auto & YGround = aVUk[1];
    const auto & ZGround = aVUk[2];
// std::cout << "LLL " << __LINE__ << " " << XGround.mNum << " " << YGround.mNum << " " << ZGround.mNum << "\n";

    // 1 - Pose / External parameter 
        // 1.1  Coordinate of camera center
    const auto & C_XCam = aVUk[3];
    const auto & C_YCam = aVUk[4];
    const auto & C_ZCam = aVUk[5];

        // 1.2  Coordinate of Omega vector coding the unknown "tiny" rotation
    const auto & Wx = aVUk[6];
    const auto & Wy = aVUk[7];
    const auto & Wz = aVUk[8];

    // 2 - Intrinsic parameters
         // 2.1 Principal point  and Focal
    const auto & xPP = aVUk[ 9];
    const auto & yPP = aVUk[10];
    const auto & zPP = aVUk[11]; // also named as focal

         // Also in this model we confond Principal point and distorsion center, name 
         // explicitely the dist center case we change our mind
    const auto & xCD = xPP;
    const auto & yCD = yPP;

         // 2.2  Radial  distortions coefficients
    const auto & k2D = aVUk[12];
    const auto & k4D = aVUk[13];
    const auto & k6D = aVUk[14];

         // 2.3  Decentric distorstion
    const auto & p1 = aVUk[15];
    const auto & p2 = aVUk[16];

         // 2.3  Affine distorsion
    const auto & b1 = aVUk[17];
    const auto & b2 = aVUk[18];

   // Vector P->Cam
    auto  XPC = XGround-C_XCam;
    auto  YPC = YGround-C_YCam;
    auto  ZPC = ZGround-C_ZCam;


    // Coordinate of points in  camera coordinate system, do not integrate "tiny" rotation

    auto  XCam0 = aVObs[0] * XPC +  aVObs[1]* YPC +  aVObs[2]*ZPC;
    auto  YCam0 = aVObs[3] * XPC +  aVObs[4]* YPC +  aVObs[5]*ZPC;
    auto  ZCam0 = aVObs[6] * XPC +  aVObs[7]* YPC +  aVObs[8]*ZPC;

// std::cout << "LLL " << __LINE__ << " " << aVObs[0] << " " << aVObs[1] << " " << aVObs[2] << "\n";

    // Now "tiny" rotation
    //  Wx      X      Wy * Z - Wz * Y
    //  Wy  ^   Y  =   Wz * X - Wx * Z
    //  Wz      Z      Wx * Y - Wy * X

     //  P =  P0 + W ^ P0 

    auto  XCam = XCam0 + Wy * ZCam0 - Wz * YCam0;
    auto  YCam = YCam0 + Wz * XCam0 - Wx * ZCam0;
    auto  ZCam = ZCam0 + Wx * YCam0 - Wy * XCam0;

    // Projection :  (xPi,yPi,1) is the bundle direction in camera coordinates

    auto xPi =  XCam/ZCam;
    auto yPi =  YCam/ZCam;
// std::cout << "LLL " << __LINE__ << " " << XCam.mNum << " " << YCam.mNum << " " << ZCam.mNum << "\n";
// std::cout << "LLL " << __LINE__ << " " << xPi.mNum << " " << yPi.mNum << "\n";

    // Coordinate relative to distorsion center
    auto xC =  xPi-xCD;
    auto yC =  yPi-yCD;
    auto x2C = square(xC);  // Use the indermediar value to (probably) optimize Jet
    auto y2C = square(yC);
    auto xyC = xC * yC;
    auto Rho2C = x2C + y2C;

   // Compute the distorsion
    auto rDist = k2D*Rho2C + k4D * square(Rho2C) + k6D*cube(Rho2C);
    auto affDist = b1 * xC + b2 * yC;
    auto decX = p1*(3.0*x2C + y2C) +  p2*(2.0*xyC);
    auto decY = p2*(3.0*y2C + x2C) +  p1*(2.0*xyC);
    

    auto xDist =  xPi + xC * rDist + decX + affDist;
    auto yDist =  yPi + yC * rDist + decY ;

   // Use principal point and focal
    auto xIm =  xPP  + zPP  * xDist;
    auto yIm =  yPP  + zPP  * yDist;

// std::cout << "LLL " << __LINE__ << " " << xIm.mNum << " " << yIm.mNum << "\n";

    auto x_Residual = xIm -  aVObs[ 9];
    auto y_Residual = yIm -  aVObs[10];

// getchar();

    return {x_Residual,y_Residual};
}


class cTestFraserCamColinearEq
{
    public :
       cTestFraserCamColinearEq(int aSzBuf,bool Show);

    private :
       
       static const int  TheNbUk  = 19;
       static const int  TheNbObs = 11;
       typedef Jet<double,NB_UK>  tJets;
       typedef cEpsNum<NB_UK>     tEps;

       static const  std::vector<std::string> TheVNamesUnknowns;
       static const  std::vector<std::string> TheVNamesObs;

       /// Return unknowns vect after fixing XYZ (ground point)
       const std::vector<double> & VUk(double X,double Y,double Z);
       /// Return observation vect t after fixing I,J (pixel projection)
       const std::vector<double> & VObs(double I,double J);

 
       FD::cCoordinatorF<double>  mCFD;  /// Coordinator for formal derivative
       std::vector<double>        mVUk;  /// Buffer for computing the unknown
       std::vector<double>        mVObs; /// Buffer for computing the unknown
};


const std::vector<double> & cTestFraserCamColinearEq::VUk(double X,double Y,double Z)
{
   mVUk[0] = X;
   mVUk[1] = Y;
   mVUk[2] = Z;

   return mVUk;
}

const std::vector<double> & cTestFraserCamColinearEq::VObs(double I,double J)
{
     mVObs[ 9] = I;
     mVObs[10] = J;
    
    return mVObs;
}



const std::vector<std::string> 
  cTestFraserCamColinearEq::TheVNamesUnknowns
  {
      "XGround","YGround","ZGround",            // Unknown 3D Point
      "XCam","YCam","ZCam", "Wx","Wy","Wz",     // External Parameters
      "ppX","ppY","ppZ",                        // Internal : principal point + focal
      "k2","k4","k6", "p1","p2","b1","b2"       // Distorsion (radiale/ Decentric/Affine)
  };

const std::vector<std::string> 
  cTestFraserCamColinearEq::TheVNamesObs
  {
        "oR00","oR01","oR02","oR10","oR11","oR12","oR20","oR21","oR22",
        "oXIm","oYIm"
  };




cTestFraserCamColinearEq::cTestFraserCamColinearEq(int aSzBuf,bool Show) :
    // mCFD (aSzBuf,19,11) would have the same effect, but future generated code will be less readable
     mCFD  (aSzBuf,TheVNamesUnknowns,TheVNamesObs),
     mVUk  (TheNbUk,0.0),
     mVObs (TheNbObs,0.0)
{
   // In unknown, we set everything to zero exepct focal to 1
   mVUk[11] = 1.0;
   // In obs, we set the current matrix to Id
   mVObs[0] = mVObs[4] = mVObs[8] = 1;

   double aT0 = TimeElapsFromT0();

   auto aVFormula = FraserCamColinearEq(mCFD.VUk(),mCFD.VObs());
   if (Show)
   {
       mCFD.SetCurFormulas({aVFormula[0]});
       int aNbRx = mCFD.VReached().size() ;
       mCFD.SetCurFormulas(aVFormula);
       int aNbRxy = mCFD.VReached().size() ;

       std::cout << "NbReached x:" << aNbRx << "  xy:" << aNbRxy << "\n";
        
       mCFD.SetCurFormulas({aVFormula[0]});
       mCFD.ShowStackFunc();
   }
   mCFD.SetCurFormulasWithDerivative(aVFormula);

   double aT1 = TimeElapsFromT0();
    
   std::cout << "TestFraser NbEq=" << mCFD.VReached().size() << " TimeInit=" << (aT1-aT0) << "\n";

   
   // mCFD.ShowStackFunc();

   int aNbTestTotal =  1e5; ///< Approximative number of Test
   int aNbTestWithBuf = aNbTestTotal/aSzBuf;  ///< Number of time we will fill the buffer
   aNbTestTotal = aNbTestWithBuf * aSzBuf; ///< Number of test with one equation

   // Here we initiate with "peferct" projection, to check something
   const std::vector<double> & aVUk  =  VUk(1.0,2.0,10.0);
   const std::vector<double> & aVObs =  VObs(0.101,0.2); 
   
   // Make the computation with jets
   double TimeJets = TimeElapsFromT0();
   std::vector<tJets> aJetRes;
   {
        std::vector<tJets>  aVJetUk;
        for (int aK=0 ; aK<NB_UK ; aK++)
            aVJetUk.push_back(tJets(aVUk[aK],aK));

        for (int aK=0 ; aK<aNbTestTotal ; aK++)
        {
            aJetRes = FraserCamColinearEq(aVJetUk,aVObs);
        }
        TimeJets = TimeElapsFromT0() - TimeJets;
   }

   double TimeEps = TimeElapsFromT0();
   std::vector<tEps> aEpsRes;
   {
        std::vector<tEps >  aVEpsUk;
        for (int aK=0 ; aK<NB_UK ; aK++)
            aVEpsUk.push_back(tEps(aVUk[aK],aK));
        for (int aK=0 ; aK<aNbTestTotal ; aK++)
        {
            aEpsRes = FraserCamColinearEq(aVEpsUk,aVObs);
        }
        TimeEps = TimeElapsFromT0() - TimeEps;
   }

   // Make the computation with formal deriv buffered
   double TimeBuf = TimeElapsFromT0();
   {
       for (int aK=0 ; aK<aNbTestWithBuf ; aK++)
       {
           // Fill the buffers with data
           for (int aKInBuf=0 ; aKInBuf<aSzBuf ; aKInBuf++)
               mCFD.PushNewEvals(aVUk,aVObs);
           // Evaluate the derivate once buffer is full
           mCFD.EvalAndClear();
       }
       TimeBuf = TimeElapsFromT0() - TimeBuf;
   }

   for (int aKVal=0 ; aKVal<int(aEpsRes.size()) ; aKVal++)
   {
      std::cout << "VALssss " << aKVal << "\n";
      std::cout << "  J:" <<  aJetRes[aKVal].a 
                <<  " E:" << aEpsRes[aKVal].mNum 
                <<  " F:" << mCFD.ValComp(0,aKVal) << "\n";
      for (int aKVar=0;  aKVar< NB_UK ; aKVar++)
      {
           std::cout << "  dJ:" << aJetRes[aKVal].v[aKVar]
                     <<  " dE:" << aEpsRes[aKVal].mEps[aKVar]
                     <<  " dF:" << mCFD.DerComp(0,aKVal,aKVar) << "\n";
      }
   }

   std::cout 
         << " TimeJets= " << TimeJets 
         << " TimeEps= " << TimeEps 
         << " TimeBuf= " << TimeBuf
         << "\n";
}

void TestFraserCamColinearEq()
{
   {
       cTestFraserCamColinearEq (1000,false);
       //  cTestFraserCamColinearEq (100);
       //  cTestFraserCamColinearEq (10);
       cTestFraserCamColinearEq (1,false);
   }
/*
   {
        using MMVII::cEpsNum;
        std::vector<cEpsNum<NB_UK> >  aVUk;
        for (int aK=0 ; aK<NB_UK ; aK++)
            aVUk.push_back(cEpsNum<NB_UK>(1.0,aK));

        std::vector<double>  aVObs;
        for (int aK=0 ; aK<NB_OBS ; aK++)
            aVObs.push_back(1/(1.0+aK));

       auto aVRes = FraserCamColinearEq(aVUk,aVObs);
   }
   {
        std::vector<Jet<double,NB_UK> >  aVUk;
        for (int aK=0 ; aK<NB_UK ; aK++)
            aVUk.push_back(Jet<double,NB_UK>(1.0,aK));

        std::vector<double>  aVObs;
        for (int aK=0 ; aK<NB_OBS ; aK++)
            aVObs.push_back(1/(1.0+aK));

       auto aVRes = FraserCamColinearEq(aVUk,aVObs);
   }
*/

   getchar();
}


/* -------------------------------------------------- */

namespace  MMVII
{
    void BenchCmpOpVect();
};

void   BenchFormalDer()
{
    // MMVII::BenchCmpOpVect();
    TestFraserCamColinearEq();
   // Run TestRatkoswky with static obsevation an inital guess 
    TestRatkoswky(TheVRatkoswkyData,{100,10,1,1});
    InspectCube() ;
    // cTestOperationVector<float,90>::DoIt();
    // cTestOperationVector<float,128>::DoIt();
    // cTestOperationVector<double,128>::DoIt();
    getchar();
}


/*
--- Form[0] => C0 ; Val=0
--- Form[1] => C1 ; Val=1
--- Form[2] => C2 ; Val=2
-0- Form[3] => a
-0- Form[4] => b
-0- Form[5] => x
-0- Form[6] => y
-1- Form[7] => F4*F5      // bx
-2- Form[8] => F7+F3      // a+bx
-3- Form[9] => F8*F8      // (a+bx) ^2
-4- Form[10] => F8*F9     // (a+bx) ^ 3
-5- Form[11] => F10-F6    // (a+bx)^3-y
-3- Form[12] => F8*F5     // x(a+bx)
-4- Form[13] => F12+F12   // 2x(a+bx)
-5- Form[14] => F13*F8    // 2x(a+bx)^2
-4- Form[15] => F9*F5     // x (a+bx)^2
-6- Form[16] => F14+F15   // 3 x(a+bx)^2
-3- Form[17] => F8+F8     // 2 (a+bx) 
-4- Form[18] => F8*F17    // 2 (a+bx) ^2
-5- Form[19] => F18+F9    // 3 (a+bx) ^2
REACHED 5 3 6 4 7 8 9 17 12 10 18 15 13 11 14 19 16   // Reached formula in their computation order
CUR 11 19 16   // Computed formula
*/



