#include "include/SymbDer/SymbolicDerivatives.h"

double TimeElapsFromT0()
{
    static auto BeginOfTime = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(now - BeginOfTime).count() / 1e6;
}

namespace  SD = NS_SymbolicDerivative;


#if (MMVII_WITH_CERES)
#include "ceres/jet.h"
using ceres::Jet;

// ========= Define on Jets some function that make them work like formula and nums
// and also may optimize the computation so that comparison is fair
    

template <typename T, int N> inline Jet<T, N> square(const Jet<T, N>& f) 
{
  return Jet<T, N>(SD::square(f.a), (2.0*f.a) * f.v);
}

template <typename T, int N> inline Jet<T, N> cube(const Jet<T, N>& f) 
{
  T a2 = SD::square(f.a);
  return Jet<T, N>(f.a*a2, (3.0*a2) * f.v);
}


template <typename T, int N> inline Jet<T, N> powI(const Jet<T, N>& aJ,const int & aExp) 
{
   // In this case avoid compute 1/x and multiply by x
   if (aExp==0) return Jet<T,N>(1.0);

   // make a single computation of pow
   T aPm1 = SD::powI(aJ.a,aExp-1);
   return Jet<T,N>(aJ.a*aPm1,(aExp*aPm1)*aJ.v);
}

template <class T,const int N> Jet<T, N>  CreateCste(const T & aV,const Jet<T, N>&) 
{
    return Jet<T, N>(aV);
}

#else
#endif

//=========================================

//=========================================


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

void TestRatkoswky(bool Show,const tVRatkoswkyData & aVData,const std::vector<double> & aInitialGuess)
{
    size_t aNbUk = 4;
    size_t aNbObs = 2;
    assert(aInitialGuess.size()==aNbUk); // consitency test

   //-[1] ========= Create/Init the coordinator =================  
   //-    This part [1] would be executed only one time
        // Create a coordinator/context where values are stored on double and :
        //  4 unknown (b1-b4), 2 observations, a buffer of size 100
    SD::cCoordinatorF<double>  aCFD("RatKowsky",100,aNbUk,aNbObs);

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
            double aDerNum =  SD::NumericalDerivate
                              (RatkoswkyResidual<double>,aInitialGuess,aVData[aKObs],aKUnk,1e-5).at(0);

            // Check but with pessimistic majoration of error in finite difference
            assert(std::abs(aDerFLine-aDerNum)<1e-4);
         }
    }
    if (Show)
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
       return {cube(a+b *x)- y};
       Type F = (a+b *x);
       return {F*F*F - y};
    }


    // but here we want to test the reduction process
    return {(a+b *x)*(x*b+a)*(a+b *x) - y};
}

void LocInspectCube()
{
    std::cout <<  "===================== TestFoncCube  ===================\n";

    // Create a context where values are stored on double and :
    //    2 unknown, 2 observations, a buffer of size 100
    //    aCFD(100,2,2) would have the same effect for the computation
    //    The variant with vector of string, will fix the name of variables, it
    //    will be usefull when will generate code and will want  to analyse it
    SD::cCoordinatorF<double>  aCFD("FitCube",100,{"a","b"},{"x","y"});

    // Inspect vector of unknown and vector of observations
    {  
        for (const auto & aF : aCFD.VUk())
           std::cout << "Unknowns : "<< aF->Name() << "\n";
        for (const auto & aF : aCFD.VObs())
           std::cout << "Observation : "<< aF->Name() << "\n";
    }

    // Create the formula corresponding to residual
    std::vector<SD::cFormula<double>>  aVResidu = FitCube(aCFD.VUk(),aCFD.VObs());
    SD::cFormula<double>  aResidu = aVResidu[0];
 
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

}

/* {III}  ========================  Test perf on colinearit equation ==========================

    On this example we use the colinearity equation which is central in bundle adjustment. We
    use it with different type of camera :

       * a Fraser camera, which a very current as it modelize physically the
         main distorsion with a relatively low number of parameters

       * a polynomial model that is more mathemitcall that can be used to approximate any
         function;  the degree is parametrizable inside a template

Now we make a macro description of the main classes :

   class cEqCoLinearity<class TypeDist> :  implement the colinarity equation;

        It is parametrized by the type of distortion TypeDist; the TypeDist
        will define the mathematicall function of distorsion
        the TypeDist will also define the types of "scalars" for unknowns 
        and observation. Scalar can be jet,cFormula, num.
        See class cTplFraserDist for a detailled example of requirement to
        be a valid dist.

        Fundamental static method : Residual , for a given value of unknown and obs,
        return the vector of residual (x,y).

    cCountDist<const int> a Class for centralizing the counting of unknown and obs
        
    class cTplFraserDist<TypeUk,TypeObs> : public cCountDist<7>
         a class implemanting fraser distorsion

    class cTplPolDist<TypeUk,TypeObs,Deg>
         a class implemanting polynomial distorsion of degre "Deg"
*/

/*
    ================ Mathematical formalization ===============

    The unknown are :

       * Pg = PGround
       * C  = Center of camera
       * R  = Orientation of camera Rotation Matrix
       * F,PP = Principal point and focal
       * Distortion

    Then, without distorsion, we have alreay 12 unkwnown

    We write the rotation as  R = A * R0 , were R0 is the current value of the
    rotation. Because :
       * it avoid the the classical guimbal-lock problem
       * as A is a rotation close to identity, we ca, write A = I + ^W were W 
         is a small vector , and ^W is the vector product matrix
   
   Then we have 11 obsevation : 

        * 9 for matrix R0
        * 2 for image point PIm

   The equation is

        * Pc  =  (I +^W) R0 (Pg-C)  for Pc= coordinate in camera repair
        * PPi = Pp + F*(Pc.x,Pc.y) / Pc.z   for projection without distorsion
        
    =>    PIm = Dist(Proj) 
*/

/**
    A class for sharing the number of unknown & observations. One template parameter
    the number of unknown of the distortion
*/

template <const int NbParamD> class cCountDist
{
    public :
        static const int TheNbDist        =  NbParamD;
        static const int TheNbCommonUk    = 12 ;
        static const int TheNbUk          = TheNbCommonUk + TheNbDist;
        static const int TheNbObs         = 11;

        static const std::vector<std::string> & VNamesObs()
        {
            static std::vector<std::string> TheVObs
            {
                "oR00","oR01","oR02","oR10","oR11","oR12","oR20","oR21","oR22", // Rotation
                "oXIm","oYIm"  // Image point
            };
            return TheVObs;
        };
        static const std::vector<std::string> & VUnkGlob ()
        {
            static std::vector<std::string> TheVUk
            {
              "XGround","YGround","ZGround",        // Unknown 3D Point
              "XCam","YCam","ZCam", "Wx","Wy","Wz", // External Parameters
              "ppX","ppY","ppZ"                     // Internal : principal point + focal
            };
            return TheVUk;
         }

};

/**  Class implementing the fraser model distortion.

     It contains the 4 prerequirement to be a Valid distorsion class usable in
     cEqCoLinearity<TypeDist>  :

        *  definition of type of unknown tUk
        *  definition of type of obs     tObs
        *  definition of the vector of names of unknown VNamesUnknowns(),
           this vector contain the 12 global unknown + those specific to dist
        *  method Dist for computing the distortion

     The method Dist take as parameters :

        * xPi, yPi
        * the vector of unknown , the 12 first value are those  describes above
        * the vector of observation, it is not used for now, but maybe 
          will be later for some dist ?

*/

class cTplFraserDist : public cCountDist<7>
{
  public :
    /// Usable for message, also for name generation in formulas
    static std::string  NameModel() {return "Fraser";}

    static const std::vector<std::string>&  VNamesUnknowns()
    {
      static std::vector<std::string>  TheV;
      // Add name of distorsion to others unkonw
      if (TheV.empty())
      {
        TheV = VUnkGlob();
        // k2,k4,k6  Distorsion radiale; p1 p2 Decentric ; b1 b2 Affine
        for (auto aS :{"k2","k4","k6", "p1","p2","b1","b2"}) 
           TheV.push_back(aS);
      }
 
      return  TheV;
    }

    template<typename tUk, typename tObs>
    static std::vector<tUk> Dist (
                                 const tUk & xPi,const tUk & yPi, 
                                 const std::vector<tUk> & aVUk, const std::vector<tObs> & 
                             )
    {
         // In this model we confond Principal point and distorsion center, 
         const auto & xCD = aVUk[ 9];
         const auto & yCD = aVUk[10];

         //  Radial  distortions coefficients
         const auto & k2D = aVUk[12];
         const auto & k4D = aVUk[13];
         const auto & k6D = aVUk[14];

         //   Decentric distorstion
         const auto & p1 = aVUk[15];
         const auto & p2 = aVUk[16];

         //   Affine distorsion
         const auto & b1 = aVUk[17];
         const auto & b2 = aVUk[18];

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

         return {xDist,yDist};
    }
  private :
};




/*
    Class for implementing a polynomial distorsion. The maximal degree
  is the last parameter of this template class.

    In this model, we want to be abble to approximat any smooth function,
  so a priori we will use all the monome under a given degree.
    Let D be the degree the  distortion will be :

      Distx = Som (dx_ij   X^i Y^j)   i+j<=D
      Disty = Som (dy_ij   X^i Y^j)   i+j<=D


     But it we want to avoid sur parametrization, we have to be cautious and avoid
   certain monoms because they redundant, or highly correlated, with some external
   parameter (rotation) or other internal parameter ( PP,focal). So we avoid :

     - Degre 0 =>  all because principal point

     - Degre 1 => 
           * we note  (dx_10 dx_01 dy_10 dy_01) the degree 1 parameter
           * (1 0 0 1) is focal, so avoid it 
           * (0 -1 1 0) is a pure rotation and redundant with rotation around axe, avoid it
           * so we have to select a complementary base,
           * (1 0 0 0)  (0 1 0 0)  is a complementary base 

        So we avoid  degree 1 in Y

     - Degre 2 :

          * Rotation arround X  + linear are highly correlated to X2 + affine, so we muste
            so avoid X2 in X
          * Idem avoid Y2 in Y


    Finnaly we have :

        *  (D+1) (D+2) /2  monome in X, same for Y
        *  6 monome to avoid

     ----------------------------------------------------------------

   In this class we have the 4 requirement as in Fraser. We have also two
   facility function :

   * bool OkMonome(bool isX,int aDegX,int aDegY) indicate if a monome of
     degree Dx,Dy is not to avoid (the bool isX means if it for Dx or Dy as
     the rule is not the same)
          
   * void InitDegreeMonomes(xDx,xDy,yDx,yDy);
        The 4 parameters are & of vector of int, as a result they contain the
      degrees of the monomes :
          DistX =  Som ( X ^ xDx[K]  Y ^ yDy[k])
        
*/


/**  Class implementing a polynomial distorsion of degree  Deg
*/


template <const int Deg> class cTplPolDist :
       public cCountDist<(Deg+1)*(Deg+2) -6> 
{
    public :
       typedef cCountDist<(Deg+1)*(Deg+2) -6>  tCountDist;

       /// Usable for message, also for name generation in formulas
       static std::string  NameModel() {return "XYPol_Deg"+std::to_string(Deg);}
   
       // Vectors of names of unknowns
       static const std::vector<std::string>&  VNamesUnknowns()
       {
         static std::vector<std::string>  TheV;
         if (TheV.empty()) // First call
         {
            // Get the common unknowns
            TheV = tCountDist::VUnkGlob();

            // Get the degrees of monomes
            std::vector<int>  aXDx,aXDy,aYDx,aYDy;
            InitDegreeMonomes(aXDx,aXDy,aYDx,aYDy);
 
           // Add the name of monomes for X Dist
            for (size_t aK=0 ; aK<aXDx.size() ; aK++)
            {
                TheV.push_back
                (
                     "xDistPol_" 
                    + std::to_string(aXDx.at(aK))  + "_"
                    + std::to_string(aXDy.at(aK))
                );
            }

           // Add the name of monomes for Y Dist
            for (size_t aK=0 ; aK<aYDx.size() ; aK++)
            {
                TheV.push_back
                (
                     "yDistPol_" 
                    + std::to_string(aYDx.at(aK))  + "_"
                    + std::to_string(aYDy.at(aK))
                );
            }
         }
         return  TheV;
       }

       // Vectors of names of unknowns

       template<typename tUk, typename tObs>
       static std::vector<tUk> Dist (
                                 const tUk & xPi,const tUk & yPi, 
                                 const std::vector<tUk> & aVUk, const std::vector<tObs> & 
                             )
       {
           static std::vector<int>  aXDx,aXDy,aYDx,aYDy;
            if (aXDx.empty()) // first call compute degree of monomes
               InitDegreeMonomes(aXDx,aXDy,aYDx,aYDy);
             
            //  We compute here the  Value of monomes : X^i and Y^j , 
            // this is an optimisation for jets, probably not usefull for formula, but does not hurt either
            std::vector<tUk> aVMonX;  
            std::vector<tUk> aVMonY;  

            // We can compute it using powI optimized functionc, or using a recurence formula
            // According to type, the optimal computation may not be the same
            // On tests it seems more or less equivalent ....
            if (0)
            {  
               // Case using powI
               for (int aD=0 ;aD<=Deg ; aD++)
               {
                  aVMonX.push_back(powI(xPi,aD));
                  aVMonY.push_back(powI(yPi,aD));
               }
            }
            else
            {
               // Case using recurence   X^(k+1) = X^k *X
               aVMonX.push_back(CreateCste(1.0,xPi));
               aVMonY.push_back(CreateCste(1.0,xPi));
               for (int aD=1 ;aD<=Deg ; aD++)
               {
                  aVMonX.push_back(aVMonX.back()*xPi);
                  aVMonY.push_back(aVMonY.back()*yPi);
               }
            }
            // Initialisze  with identity
            auto xDist =  xPi;
            auto yDist =  yPi;

            int anInd =  tCountDist::TheNbCommonUk;  // Unkown on dist are stored after common 
            // Be carefull to be coherent with VNamesUnknowns
            for (size_t aK=0; aK<aXDx.size() ; aK++)
                xDist = xDist+aVMonX.at(aXDx.at(aK))*aVMonY.at(aXDy.at(aK))*aVUk.at(anInd++);

            for (size_t aK=0; aK<aYDx.size() ; aK++)
                yDist = yDist+aVMonX.at(aYDx.at(aK))*aVMonY.at(aYDy.at(aK))*aVUk.at(anInd++);

            return {xDist,yDist};
       }

    private :
       // indicate if X^DegX  Y ^DegY is to avoid for xDist/yDist 
       static bool OkMonome(bool isX,int aDegX,int aDegY)
       {
            if ((aDegX ==0) && (aDegY==0)) return false;  // degre 0 : avoid
            if ((!isX) && ((aDegX + aDegY) ==1))       return false;  //  degre 1 in dY : avoid
            if (isX &&    (aDegX==2) &&  (aDegY ==0))  return false;  //  X2 in dX avoid
            if ((!isX) && (aDegX==0) &&  (aDegY ==2))  return false;  //  Y2 in dY avoid
   
            return true;  // then ok
       }

       static inline void InitDegreeMonomes
            (
                std::vector<int>  & aXDx,  // Degre in x of X component
                std::vector<int>  & aXDy,  // Degre in y of X component
                std::vector<int>  & aYDx,  // Degre in x of Y component
                std::vector<int>  & aYDy   // Degre in y of Y component
            )
        {
            for (int aDx=0 ; aDx<=Deg ; aDx++)
            {
                for (int aDy=0 ; (aDx+aDy)<=Deg ; aDy++)
                {
                    if (OkMonome(true,aDx,aDy))
                    {
                        aXDx.push_back(aDx);
                        aXDy.push_back(aDy);
                    }

                    if (OkMonome(false,aDx,aDy))
                    {
                        aYDx.push_back(aDx);
                        aYDy.push_back(aDy);
                    }
                }
            }
        }
};




template <class TypeDist>  class cEqCoLinearity
{
  public :
    static const int  TheNbUk  = TypeDist::TheNbUk;
    static const int  TheNbObs = TypeDist::TheNbObs;
    static const std::vector<std::string>& VNamesUnknowns() { return TypeDist::VNamesUnknowns();}
    static const std::vector<std::string>& VNamesObs() { return TypeDist::VNamesObs();}
    static std::string NameModel() { return TypeDist::NameModel();}

       /*  Capital letter for 3D variable/formulas and small for 2D */
    template <typename tUk, typename tObs>
    static     std::vector<tUk> Residual
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  )
    {
        assert (aVUk.size() ==TheNbUk) ;  // SD::UserSError("Bad size for unknown");
        assert (aVObs.size()==TheNbObs) ;// SD::UserSError("Bad size for observations");

        // 0 - Ground Coordinates of projected point
        const auto & XGround = aVUk[0];
        const auto & YGround = aVUk[1];
        const auto & ZGround = aVUk[2];

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

       // Vector P->Cam
        auto  XPC = XGround-C_XCam;
        auto  YPC = YGround-C_YCam;
        auto  ZPC = ZGround-C_ZCam;


        // Coordinate of points in  camera coordinate system, do not integrate "tiny" rotation

        auto  XCam0 = aVObs[0] * XPC +  aVObs[1]* YPC +  aVObs[2]*ZPC;
        auto  YCam0 = aVObs[3] * XPC +  aVObs[4]* YPC +  aVObs[5]*ZPC;
        auto  ZCam0 = aVObs[6] * XPC +  aVObs[7]* YPC +  aVObs[8]*ZPC;

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

        // Now compute the distorsion
        auto   aVDist = TypeDist::Dist(xPi,yPi,aVUk,aVObs);
        const auto & xDist =  aVDist[0];
        const auto & yDist =  aVDist[1];

       // Use principal point and focal to compute image projection
        auto xIm =  xPP  + zPP  * xDist;
        auto yIm =  yPP  + zPP  * yDist;


       // substract image observations to have a residual
        auto x_Residual = xIm -  aVObs[ 9];
        auto y_Residual = yIm -  aVObs[10];


        return {x_Residual,y_Residual};
    }
};



template <class FORMULA>  class cTestEqCoL
{
    public :
       cTestEqCoL(int aSzBuf,int aLevel,bool ShowGlob,bool ShowDetail);

    private :
       static const int  TheNbUk = FORMULA::TheNbUk;
       static const int  TheNbObs = FORMULA::TheNbObs;

       typedef SD::cCoordinatorF<double>    tCoord;
       typedef typename tCoord::tFormula    tFormula;

       /// Return unknowns vect after fixing XYZ (ground point)
       const std::vector<double> & VUk(double X,double Y,double Z)
       {
          mVUk[0] = X;
          mVUk[1] = Y;
          mVUk[2] = Z;

          return mVUk;
       }
       /// Return observation vect  after fixing I,J (pixel projection)
       const std::vector<double> & VObs(double I,double J)
       {
          mVObs[ 9] = I;
          mVObs[10] = J;
    
         return mVObs;
       }

 
       tCoord                     mCFD;  ///< Coordinator for formal derivative
       std::vector<double>        mVUk;  ///< Buffer for computing the unknown
       std::vector<double>        mVObs; ///< Buffer for computing the unknown
};




template <class FORMULA>
cTestEqCoL<FORMULA>::cTestEqCoL(int aSzBuf,int aLevel,bool ShowGlob,bool ShowAllDetail) :
     // mCFD (aSzBuf,TheNbUk,TheNbObs), //  would have the same effect, but future generated code will be less readable
     mCFD  (FORMULA::NameModel(),aSzBuf,FORMULA::VNamesUnknowns(),FORMULA::VNamesObs()),
     mVUk  (TheNbUk,0.0),
     mVObs (TheNbObs,0.0)
{
   // In unknown, we set everything to zero exepct focal to 1
   mVUk[11] = 1.0;
   // In obs, we set the current matrix to Id
   mVObs[0] = mVObs[4] = mVObs[8] = 1;

   double aT0 = TimeElapsFromT0();

//   auto aVFormula = FraserFuncCamColinearEq(mCFD.VUk(),mCFD.VObs());
   auto aVFormula = FORMULA::Residual (mCFD.VUk(),mCFD.VObs());
   
   if (ShowGlob)
   {
       mCFD.SetCurFormulas({aVFormula[0]});
       int aNbRx = mCFD.VReached().size() ;
       mCFD.SetCurFormulas(aVFormula);
       int aNbRxy = mCFD.VReached().size() ;

       std::cout << "NbReached x:" << aNbRx << "  xy:" << aNbRxy << "\n";
        
       mCFD.SetCurFormulas({aVFormula[0]});
       if (ShowAllDetail)
           mCFD.ShowStackFunc();
   }
   mCFD.SetCurFormulasWithDerivative(aVFormula);
   //  In tentative to reduce the size, print the statistiq on all operators
   if (ShowGlob)
   {
      const std::vector<tFormula>& aVR =mCFD.VReached();
      int aNbTot=0;
      int aNbPl=0;
      for (const auto  & aF : aVR)
      {
          aNbTot++;
          std::string anOp =  aF->NameOperator() ;
          // std::cout << "Opp= " << anOp << "\n";
          if ((anOp=="+") || (anOp=="*"))
          {
             aNbPl++;
          }
      }
      std::cout 
                 << " NbTotOper=" << aNbTot
                 << " +or*=" << aNbPl
                 << "\n";
   }

   double aT1 = TimeElapsFromT0();
    
   if (ShowGlob)
       std::cout << "Test "  +  FORMULA::NameModel()
                 << ", SzBuf=" << aSzBuf 
                 << ", NbEq=" << mCFD.VReached().size() 
                 << ", TimeInit=" << (aT1-aT0) << "\n";

   
   // mCFD.ShowStackFunc();

   int aNbTestTotal =  std::min(1e5,1e3*pow(aLevel,1.5)) ; ///< Approximative number of Test
  

   int aNbTestWithBuf = std::ceil(aNbTestTotal/double(aSzBuf));  ///< Number of time we will fill the buffer
   aNbTestTotal = aNbTestWithBuf * aSzBuf; ///< Number of test with one equation

   // Here we initiate with "peferct" projection, to check something
   const std::vector<double> & aVUk  =  VUk(1.0,2.0,10.0);
   const std::vector<double> & aVObs =  VObs(0.101,0.2); 
   
   // Make the computation with jets
   double TimeJets = TimeElapsFromT0();
#if (MMVII_WITH_CERES)
   typedef Jet<double,TheNbUk> tJets;
   std::vector<tJets> aJetRes;
   {
        std::vector<tJets>  aVJetUk;
        for (int aK=0 ; aK<TheNbUk ; aK++)
            aVJetUk.push_back(tJets(aVUk[aK],aK));

        for (int aK=0 ; aK<aNbTestTotal ; aK++)
        {
            aJetRes = FORMULA::Residual (aVJetUk,aVObs);
        }
        TimeJets = TimeElapsFromT0() - TimeJets;
   }


   for (int aKVal=0 ; aKVal<int(aJetRes.size()) ; aKVal++)
   {
      double aVJ = aJetRes[aKVal].a;
      // double aVE = aEpsRes[aKVal].mNum ;
      double aVF = mCFD.ValComp(0,aKVal); 
      // std::cout << "VALssss " << aKVal << "\n";
      // std::cout << "  J:" <<  aVJ <<  " E:" << aVE <<  " F:" << aVF << "\n";
      //  assert(std::abs(aVJ-aVE)<1e-5);
      // assert(std::abs(aVJ-aVF)<1e-5);

      SD::AssertAlmostEqual(aVJ,aVF,1e-5);
      // FD::AssertAlmostEqual(aVJ,aVF,1e-5);
      for (int aKVar=0;  aKVar< TheNbUk ; aKVar++)
      {
           double aDVJ = aJetRes[aKVal].v[aKVar];
           //  double aDVE = aEpsRes[aKVal].mEps[aKVar] ;
           double aDVF = mCFD.DerComp(0,aKVal,aKVar); 
      //  FD::AssertAlmostEqual(aDVJ,aDVE,1e-5);
           SD::AssertAlmostEqual(aDVJ,aDVF,1e-5);
      //  std::cout << "  dJ:" << aDVJ <<  " dE:" << aDVE <<  " dF:" << aDVF << "\n";
      }
   }
#endif
   
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

   if (ShowGlob)
       std::cout 
             << " TimeJets= " << TimeJets 
             // << " TimeEps= " << TimeEps 
             << " TimeBuf= " << TimeBuf
             << "\n\n";
}


void TestFraserCamColinearEq(bool Show,int aLevel)
{
   {
      SD::cCoordinatorF<double>  aCoord("test",100,4,2);
      SD::cFormula<double>     aFPi = aCoord.CsteOfVal(3.14);
      SD::cFormula<double>     aFE = aCoord.CsteOfVal(exp(1));
      SD::cFormula<double>     aFCste = aFE+aFPi;
      if (Show)
         std::cout  << " CSTE=[" << aFCste->InfixPPrint() <<"]\n";

      SD::cFormula<double>     aX =  aCoord.VUk()[0];
      SD::cFormula<double>     aY =  aCoord.VUk()[1];
      SD::cFormula<double>     aZ =  aCoord.VUk()[2];
      SD::cFormula<double>     aT =  aCoord.VUk()[3];
      SD::cFormula<double>     aMX = - aX;
      SD::cFormula<double>     aMMX = - aMX;
      if (Show) 
          std::cout  << " -X, --X=[" << aMX->InfixPPrint() << " " << aMMX->InfixPPrint() <<"]\n";

      SD::cFormula<double>     aPipX =  aFPi - aMX;
      if (Show) 
         std::cout  << " piPx= [" << aPipX->InfixPPrint() << "," << aPipX->Name()  <<"]\n";


      SD::cFormula<double>     XpX =  aX + aX;
      if (Show) 
         std::cout  << " XpX= [" << XpX->InfixPPrint()  <<"]\n";


      SD::cFormula<double>     XpPiX =  aZ+ aX + aY + aX * aFPi + aT;

      if (Show) 
      {
         std::cout  << " XpPiX= [" << XpPiX->InfixPPrint()  <<"]\n";
         aCoord.ShowStackFunc();
         std::cout  << " -------------------------\n";
      }
   }


   for (auto SzBuf : {1000,1})
   {
       cTestEqCoL<cEqCoLinearity<cTplFraserDist>> (SzBuf,aLevel,Show,false);
       cTestEqCoL<cEqCoLinearity<cTplPolDist<7>>> (SzBuf,aLevel,Show,false);
       cTestEqCoL<cEqCoLinearity<cTplPolDist<2>>> (SzBuf,aLevel,Show,false);

       if (Show)
          std::cout << "======================\n";
   }


}


/* -------------------------------------------------- */

void   Bench_powI(bool Show,int aLevel)
{
    // Test that function powI gives the same results than pow
    // Test alsp for jets, the value and the derivatives
    for (int aK=-4 ; aK<44 ; aK++)
    {
        double aV= 1.35;
        double aP1= pow(aV,double(aK));
        double aP2= SD::powI(aV,aK);
        SD::AssertAlmostEqual(aP1,aP2,1e-8);

        //TestDerNumJetBasic();
#if (MMVII_WITH_CERES)
        Jet<double,1> aJ0= powI(Jet<double,1> (aV,0),aK);
        SD::AssertAlmostEqual(aP1,aJ0.a,1e-8);

        double aEps = 1e-7;
        double aP1Minus = pow(aV-aEps,double(aK));
        double aP1Plus  = pow(aV+aEps,double(aK));
        double aNumDer = (aP1Plus-aP1Minus) / (2.0*aEps);
        SD::AssertAlmostEqual(aNumDer,aJ0.v[0],1e-8);
#endif

    } 

    // Bench on time performance
    int aNb= std::min(1e9,1e4*pow(1+aLevel,3));

         // Using std::pow
    double aT0 = TimeElapsFromT0();
    double aS=0;
    for (int aK=0 ; aK<aNb ; aK++)
        aS+=std::pow(1.3,7);

         // Using powI
    double aT1 = TimeElapsFromT0();
    for (int aK=0 ; aK<aNb ; aK++)
        aS-=  SD::powI(1.3,7);

         // Using pow7 => supress the switch
    double aT2 = TimeElapsFromT0();
    for (int aK=0 ; aK<aNb ; aK++)
        aS-=SD::pow7(1.3);

    double aT3 = TimeElapsFromT0();

    if (Show)
    {
        std::cout << "PowR " << aT1-aT0 
              << " PowI " << aT2-aT1 
              << " P7 " << aT3-aT2  
              << " SOM=" << aS << "\n\n";
    }
}

namespace MMVII
{
void InspectCube()
{
     LocInspectCube();
}

void   BenchFormalDer(int aLevel, bool Show)
{
   {
      // Run TestRatkoswky with static obsevation an inital guess 
       // Make a basic test with Raykoswky function, check if value
       // and derivative are the same with numerics and symbolic
       TestRatkoswky(Show,TheVRatkoswkyData,{100,10,1,1});

       // Check correctnes of PowI, wich is a bit trick
       Bench_powI(Show,aLevel);

       // Check correctness and efficiency on a  "real" example with a colinearity equation
       // check is made between Jet and formal derive
       TestFraserCamColinearEq(Show,aLevel);
   }
}
};


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

(a+bx)^3
3 (a+bx)^2 
3 x (a+bx)^2

REACHED 5 3 6 4 7 8 9 17 12 10 18 15 13 11 14 19 16   // Reached formula in their computation order
CUR 11 19 16   // Computed formula
*/



