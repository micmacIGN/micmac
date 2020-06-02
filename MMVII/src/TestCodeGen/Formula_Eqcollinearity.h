#ifndef FORMULA_EQCOLLINEARITY_H
#define FORMULA_EQCOLLINEARITY_H

#include <assert.h>
#include <vector>
#include <string>

#include "ceres/ceres.h"

namespace FD = NS_MMVII_FormalDerivative;


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
    static constexpr int NbUk()  { return TheNbUk;}
    static constexpr int NbObs() { return TheNbObs;}
    static const std::vector<std::string>& VNamesUnknowns() { return TypeDist::VNamesUnknowns();}
    static const std::vector<std::string>& VNamesObs() { return TypeDist::VNamesObs();}
    static std::string FormulaName() { return "EqColLinearity" + TypeDist::NameModel();}

       /*  Capital letter for 3D variable/formulas and small for 2D */
    template <typename tUk, typename tObs>
    static     std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  )
    {
        assert (aVUk.size() ==TheNbUk) ;  // FD::UserSError("Bad size for unknown");
        assert (aVObs.size()==TheNbObs) ;// FD::UserSError("Bad size for observations");

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
    private:
    static const int  TheNbUk  = TypeDist::TheNbUk;
    static const int  TheNbObs = TypeDist::TheNbObs;

};



#endif // FORMULA_EQCOLLINEARITY_H
