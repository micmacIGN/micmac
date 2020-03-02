#include "include/MMVII_FormalDerivatives.h"
/** \file TestDynCompDerivatives.cpp
    \brief Illustration and test of formal derivative

    {I}  A detailled example of use, and test of correctness => for the final user
*/



/* The library is in the namespace NS_MMVII_FormalDerivative, we want to
use it conveniently without the "using" directive, so create an alias FD */

namespace  FD = NS_MMVII_FormalDerivative;


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
     assert(aVEvals.size()==aVData.size());

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

   This second example, we take a very basic example to analyse
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
 
    Type F = (a + b *x);

    return {F*F*F - y};
}

void InspectCube()
{
    std::cout <<  "===================== TestRatkoswky  ===================\n";

    // Create a context where values are stored on double and :
    //  4 unknown (b1-b4), 2 observations, a buffer of size 100
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
    std::cout  << "RESIDU FORMULA=[" << aResidu->InfixPPrint() <<"]\n";

    // Inspect the derivative  relatively to b2
    std::cout  << "DERIVEATE FORMULA=[" << aResidu->Derivate(1)->InfixPPrint() <<"]\n";

    // Set the formula that will be computed
    aCFD.SetCurFormulasWithDerivative(aVResidu);
    
    // Print stack of formula
    aCFD.ShowStackFunc();

    getchar();
}

#include "ExternalInclude/Eigen/Dense"  // TODO => replace with standard eigen file

/* -------------------------------------------------- */


/*
template <class Type> 
std::vector<Type> ResiduRat43
                  (
                     const std::vector<Type> & aVUk,
                     const std::vector<Type> & aVObs
                  )
*/


/* *************************************************** */
/* *************************************************** */
/* *                                                 * */
/* *        TEST                                     * */
/* *                                                 * */
/* *************************************************** */
/* *************************************************** */

using FD::square;

std::vector<double> VRand(unsigned aSeed,int aNb)
{
    std::srand(aSeed);
    std::vector<double> aVRes;
    for (int aK=0 ; aK<aNb ; aK++)
    {
       double aV =  std::rand()/((double) RAND_MAX );
       aVRes.push_back(aV);
    }

    return aVRes;
}

template <class Type> 
std::vector<Type> Residu
                  (
                     const std::vector<Type> & aVUk,
                     const std::vector<Type> & aVObs
                  )
{
    const Type & X0 = aVUk.at(0);
    const Type & X1 = aVUk.at(1);
    const Type & X2 = aVUk.at(2);

    const Type & V0 = aVObs.at(0);
    const Type & V1 = aVObs.at(1);

    Type aF0 =  2.0 *X0 + X0*X1*X2 + pow(square(V0)+square(X0-X2),(X1*V1)/X2);
    Type aF1 =  log(square(X0+X1+X2+V0+V1));
    Type aF2 =  -aF0 + aF1;
             
    return {aF0,aF1,aF2};
}


void TestDyn()
{
    int aNbUk  = 3;
    int aNbObs = 2;

    FD::cCoordinatorF<double>  aCFD(100,aNbUk,aNbObs);
    aCFD.SetCurFormulasWithDerivative(Residu(aCFD.VUk(),aCFD.VObs()));

    int aNbT = 2;
    unsigned aSeedUk=333, aSeedObs=222;
    for (int aKTest=0 ; aKTest<aNbT ; aKTest++)
    {
       std::vector<double> aVUk  = VRand(aKTest+aSeedUk ,aNbUk);
       std::vector<double> aVObs = VRand(aKTest+aSeedObs,aNbObs);
       aCFD.PushNewEvals(aVUk,aVObs);
    }
    aCFD.EvalAndClear();

    for (int aKTest=0 ; aKTest<aNbT ; aKTest++)
    {
       // const std::vector<double> & aLineDyn =  *(aVDyn[aKTest]);
       std::vector<double> aVUk  = VRand(aKTest+aSeedUk ,aNbUk);
       std::vector<double> aVObs = VRand(aKTest+aSeedObs,aNbObs);
       std::vector<double> aVRes = Residu(aVUk,aVObs);
       int aNbRes = aVRes.size();

       for (int aKx=0 ; aKx<aNbUk ; aKx++)
       {
           double aEps = 1e-5;
           std::vector<double> aVUkP  = aVUk;
           std::vector<double> aVUkM  = aVUk;
           aVUkP[aKx] += aEps;
           aVUkM[aKx] -= aEps;

           std::vector<double> aVResP = Residu(aVUkP,aVObs);
           std::vector<double> aVResM = Residu(aVUkM,aVObs);
           for (int aKRes=0 ; aKRes<aNbRes ; aKRes++)
           {
               double aDerNum  = (aVResP[aKRes]-aVResM[aKRes]) / (2*aEps);
               double aDerForm = aCFD.DerComp(aKTest,aKRes,aKx);
               double aDif = std::abs(aDerNum-aDerForm);
               assert(aDif<1e-4);
           }
       }
       for (int aKRes=0 ; aKRes<aNbRes ; aKRes++)
       {
           double aDif = std::abs(aCFD.ValComp(aKTest,aKRes) - aVRes[aKRes] );
           assert(aDif<1e-7);
       }

    }
    aCFD.ShowStackFunc();
    getchar();
}



typedef  double TypeTest;
typedef  FD::cFormula <TypeTest>  tFormulaTest;

// #include "include/MMVII_all.h"
// #include "include/MMVII_Derivatives.h"






#define SzTEigen 90
typedef float tTEigen;
typedef  Eigen::Array<tTEigen,1,Eigen::Dynamic>  tEigenSubArray;
typedef  Eigen::Map<tEigenSubArray > tEigenWrap;
void   BenchFormalDer()
{
   // Run TestRatkoswky with static obsevation an inital guess 
    TestRatkoswky(TheVRatkoswkyData,{100,10,1,1});
    InspectCube() ;

    // TestDyn();
    if (1)
    {
        Eigen::Array<tTEigen, 1, SzTEigen>  aAFix = Eigen::Array<tTEigen, 1, SzTEigen>::Random();
        // Eigen::Array<tTEigen,Eigen::Dynamic,Eigen::Dynamic>   aADyn(1,SzTEigen);
        // Eigen::Array<tTEigen,Eigen::Dynamic,1>   aADyn(SzTEigen);
        Eigen::Array<tTEigen,1,Eigen::Dynamic>   aADyn(SzTEigen);
        Eigen::Array<tTEigen,Eigen::Dynamic,Eigen::Dynamic>   aADyn1(1,1);
        Eigen::Array<tTEigen,Eigen::Dynamic,Eigen::Dynamic>   aADyn2(1,SzTEigen);


        for (int aX=0 ; aX<SzTEigen ; aX++)
        {
            aAFix(0,aX)  = 10 + 2.0*aX;
            aAFix(0,aX)  = 1;
            aAFix(0,aX)  = 10 + 2.0*aX;
        }
        aAFix = 1;
        aADyn = aAFix;
       
        aADyn1(0,0) = 1.0;
         
#if (WITH_MMVII)
        int aNb=1e7;
        double aT0 = cMMVII_Appli::CurrentAppli().SecFromT0(); 

        for (int aK=0 ; aK<aNb ; aK++)
        {
             aAFix = aAFix + aAFix -10;
             aAFix = (aAFix + 10)/2;
        }
        double aT1 = cMMVII_Appli::CurrentAppli().SecFromT0(); 

        for (int aK=0 ; aK<aNb ; aK++)
        {
             aADyn = aADyn + aADyn -10;
             aADyn = (aADyn + 10)/2;
        }
        double aT2 = cMMVII_Appli::CurrentAppli().SecFromT0(); 

        if (0)
        {
           for (int aK=0 ; aK<aNb*SzTEigen ; aK++)
           {
               aADyn1 = aADyn1 + aADyn1 -10;
               aADyn1 = (aADyn1 + 10)/2;
           }
        }
        double aT3 = cMMVII_Appli::CurrentAppli().SecFromT0(); 

        for (int aK=0 ; aK<aNb ; aK++)
        {
             Eigen::Array<tTEigen,1,Eigen::Dynamic>   aBloc 
                // = aADyn.topLeftCorner(1,SzTEigen);
                // = aADyn.block(0,0,1,SzTEigen);
                = aADyn.head(SzTEigen-1);
             aBloc = aBloc + aBloc -10;
             aBloc = (aBloc + 10)/2;
             if (aK==0)
             {
                  std::cout << "AAAAADr  " << &(aBloc(0,0)) - &(aADyn(0,0)) << "\n";
                  std::cout << "AAAAADr  " << aBloc(0,0)   << " " << aADyn(0,0) << "\n";
             }
        }
        double aT4 = cMMVII_Appli::CurrentAppli().SecFromT0(); 

        for (int aK=0 ; aK<aNb ; aK++)
        {
            for (int aX=0 ; aX<SzTEigen ; aX++)
            {
                aADyn2(aX) = aADyn2(aX) + aADyn2(aX) -10;
                aADyn2(aX) = (aADyn2(aX) + 10)/2;
            }
        }
        double aT5 = cMMVII_Appli::CurrentAppli().SecFromT0(); 

        for (int aK=0 ; aK<aNb ; aK++)
        {
            tTEigen * aData = &  aADyn(0) ;
            for (int aX=0 ; aX<SzTEigen ; aX++)
            {
                aData[aX] =  aData[aX] + aData[aX] -10;
                aData[aX] = (aData[aX] + 10)/2;
            }
        }
        double aT6 = cMMVII_Appli::CurrentAppli().SecFromT0(); 

        for (int aK=0 ; aK<aNb ; aK++)
        {
             tEigenWrap aWrap(&aADyn(0),1,SzTEigen-1);
             // aWrap += aWrap ;
             // aWrap += 10;
             aWrap = aWrap + aWrap -10;
             aWrap = (aWrap + 10)/2;
        }
        double aT7 = cMMVII_Appli::CurrentAppli().SecFromT0(); 

        std::cout << " T01-EigenFix " << aT1-aT0 << " T12-EigenDyn " << aT2-aT1 
                  << " T23 " << aT3-aT2 << " T34-EigenBloc " << aT4-aT3  << "\n"
                  << " T45-EigenElem " << aT5-aT4 << " T56_RawData " << aT6-aT5 
                  << " T67-EigenWrap " << aT7-aT6 
                  << "\n";
        std::cout << "FIXSZ " << aAFix.rows() << " C:" <<  aAFix.cols() << "\n";
        std::cout << "DYNSZ " << aADyn.rows() << " C:" <<  aADyn.cols() << "\n";
#endif
    }


    {
       int aNbUk  = 3;
       int aNbObs = 5;
       FD::cCoordinatorF<TypeTest>  aCFD(100,aNbUk,aNbObs);

       std::vector<TypeTest> aVUk(aNbUk,0.0);
       std::vector<TypeTest> aVObs(aNbObs,0.0);
       aCFD.PushNewEvals(aVUk,aVObs);
       aCFD.EvalAndClear();

       tFormulaTest  X0 = aCFD.VUk().at(0);
       if (0)
       {
          FD::cCoordinatorF<TypeTest>  aCFD2(100,3,5);
          tFormulaTest  B0 = aCFD2.VUk().at(0);
          X0 + B0;
       }
       tFormulaTest  X1 = aCFD.VUk().at(1);
       tFormulaTest  X2 = aCFD.VUk().at(2);

       tFormulaTest  aF0 =  X0 ;
       for (int aK=0 ; aK<5 ; aK++)
       {
           std::cout << "K= " << aK << " R=" << aF0->RecursiveRec() << "\n";
           aF0 = aF0 + aF0;
       }
/*
       tFormulaTest  aF1 = aF0 + aF0;
       tFormulaTest  aF2 = aF1 + aF1;
       tFormulaTest  aF3 = aF2 + aF2;
       tFormulaTest  aF4 = aF3 + aF3;
       std::cout << "Re=" << aF->InfixPPrint() << "\n";
*/
   

       tFormulaTest  aF = (X0+X1) * (X0 +square(X2)) - exp(-square(X0))/X0;
       // tFormulaTest  aF = X0 * X0;
       tFormulaTest  aFd0 = aF->Derivate(0);

       std::cout << "F=" << aF->InfixPPrint() << "\n";
       std::cout << "Fd=" << aFd0->InfixPPrint() << "\n";

       // aF->ComputeBuf();
       std::vector<tFormulaTest> aVF{aF0,aF0};
       aCFD.SetCurFormulas(aVF);
       aCFD.SetCurFormulasWithDerivative(aVF);

       aCFD.ShowStackFunc();
/*
       aCFD.CsteOfVal(3.14);
       aCFD.CsteOfVal(3.14);
       tFormulaTest  aU0 = aCFD.VUK()[0];
       tFormulaTest  aU1 = aCFD.VUK()[1];
       tFormulaTest  aO0 = aCFD.VObs()[0];
       tFormulaTest  aO1 = aCFD.VObs()[1];
       tFormulaTest  aO2 = aCFD.VObs()[2];

       tFormulaTest  aSom00 = aU0 + aO0;
       tFormulaTest  aSomInv00 = aO0 + aU0;
       tFormulaTest  aSom11 = aO1 + aU1;

       tFormulaTest  aSom0 = aCFD.VUK()[0] + aCFD.Cste0();
       tFormulaTest  aSom1 = aCFD.VUK()[0] + aCFD.Cste1();

       tFormulaTest  aSom3 = aCFD.VUK()[0] + 3.14;
       tFormulaTest  aSom4 = 3.14 + aCFD.VUK()[0] ;
       std::cout << "TEST ADD CST " << aSom0->Name() << " " << aSom1->Name() << "\n";
       std::cout << "TEST ADD CST " << aSom3->Name() << " " << aSom4->Name() << "\n";

       aO0+aO1;
       aO1+aO2;
       aO0+(aO1+aO2);
       {
          tFormulaTest aS=(aO0+aO1)*(aO2+2.1);
          std::cout << "PP=" << aS->InfixPPrint() << "\n";
       }
*/

       // aPtr->IsCste0();
       

       // std::shared_ptr<cFuncFormalDer <8,double> > aF1  =
       
    }
    // new cCoordinatorF<double,100> (3,5);

    

    int i=10;
    std::string aStr = "i="+ std::to_string(i);
    std::cout  << "BenchFormalDerBenchFormalDerBenchFormalDer " << aStr << "\n";

    Eigen::MatrixXf m(10,20);
    Eigen::MatrixXf aM2 = m.topLeftCorner(8,15);


    Eigen::Array<double, 2, 25>  a;

    std::cout << "MMMM R:" << m.rows() << " C:" <<  m.cols() << "\n";
    std::cout << "MMMM R:" << aM2.rows() << " C:" <<  aM2.cols() << "\n";
    std::cout << "MMMM A:" << a.rows() << " C:" <<  a.cols() << "\n";

    getchar();
}


