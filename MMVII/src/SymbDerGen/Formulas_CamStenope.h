#ifndef FORMULA_CAMSTENOPE_H
#define FORMULA_CAMSTENOPE_H



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


/* For decentrik distorstion we have

      XN = Xc(Xc2+Yc2)^N    and  YN = Yc(Xc2+Yc2)^N

  (dXN,dYN)
   -------  =[R2^N+2NX^2R2^(N-1),2NXYR2^(N-1)] :N=1=>[R2+2X^2,2XY]=[3X2+Y2,2XY]
     dCx

  (dXN,dYN)
   -------  =[2XYNR2^(N-1), R2^N+2NY^2R^(N-1)  ] :N=1=>[2XY,R2+2Y^2]=[2XY,X2+3Y2]
     dCy
 
*/

template <const int DegRad,const int DegDec,const int DegreUniv> class cMMVIIUnivDist
{
    public :

       /// Usable for message, also for name generation in formulas
       static std::string  NameModel() 
       {
           return    std::string("Dist")
                   + std::string("_Rad") + std::to_string(DegRad)
                   + std::string("_Dec") + std::to_string(DegDec)
                   + std::string("_XY") + std::to_string(DegreUniv);

       }

       // Vectors of names of unknowns
/*
       static const std::vector<std::string>  NamesParamsOfDist(bool Colinear)
       {
         std::vector<std::string>  TheV;
         if (TheV.empty()) // First call
         {
            // Get the common unknowns
            TheV = tCountDist::VUnknown(Colinear);

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
*/

       // Vectors of names of unknowns

/*
       template<typename tUk> static std::vector<tUk> 
                PProjToImNorm 
                (
                     const tUk & xPi,const tUk & yPi,
                     const std::vector<tUk> & aVParam, 
                     unsigned int   NumFirstDistParam
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

            unsigned int anInd =  NumFirstDistParam;
            // Be carefull to be coherent with VNamesUnknowns
            for (size_t aK=0; aK<aXDx.size() ; aK++)
                xDist = xDist+aVMonX.at(aXDx.at(aK))*aVMonY.at(aXDy.at(aK))*aVParam.at(anInd++);

            for (size_t aK=0; aK<aYDx.size() ; aK++)
                yDist = yDist+aVMonX.at(aYDx.at(aK))*aVMonY.at(aYDy.at(aK))*aVParam.at(anInd++);

            assert(anInd==aVParam.size()); // be sure we have consumed exacly the parameters
            return {xDist,yDist};
       }
*/

       // indicate if X^DegX  Y ^DegY is to avoid for xDist/yDist
       // We must create a free familly of polynome taking into account :
       //    * the poolynoms existing in radial distorsion
       //    * the polymoms existing in decentrik distorsion
       //    * the polymoms modelized by focal an principal points
       //    * the polynom that can be approximed by rotation arround projection

       static bool OkMonome
                   (
                       bool isX,  // Is it x component of distorsion
                       int aDegX, // Degree in x
                       int aDegY  // Degree in y
                   )
       {
            // degre 0 : avoid, it's PP
            if ((aDegX ==0) && (aDegY==0)) 
               return false;  

            //  because of redundancy with focal & rotation arround Z axe, we must avoid
            // 2 out of 4 degree 1 monoms, we suppress arbitrarily the Y function 
            // (because its coherent with most current  convention on "fraser" model)

            if ((!isX) && ((aDegX + aDegY) ==1))    
               return false; 

            // because of rednuncy with non plane rotation, we supress 2 degree 2 function
            // there many choice possible, for the first one, considering a rotation arround
            // x axes + other monomes can create a taylor devlopement of Dx= X2

               //  X2 in dX avoid
            if (isX &&    (aDegX==2) &&  (aDegY ==0))  
                return false;  

               //  Y2 in dY avoid
            if ((!isX) && (aDegX==0) &&  (aDegY ==2))  
               return false;  

            // Now we must avoid redundancy with radial distortion, 
            //    x(x2+y2)^n  , y (x2+y2)^n, arbirarily, we supress x3,x5,.. in dx
            //  (aDegX>2)=> for deg 1, already many rules and no radial param                                       
            if (isX && (aDegY ==0) && ((aDegX%2)==1) && (aDegX>2) && (aDegX<=(1+2*DegRad)))
               return false;  

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
            for (int aDx=0 ; aDx<=DegreUniv ; aDx++)
            {
                for (int aDy=0 ; (aDx+aDy)<=DegreUniv ; aDy++)
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
    private :
};

/*
template <class TypeDist>  class cEqDist
{
  public :
    typedef typename TypeDist::tCountDist  tCountDist;

    static constexpr int NbUk()  { return TheNbUk;}
    static constexpr int NbObs() { return TheNbObs;}
    static const std::vector<std::string>& VNamesUnknowns() 
    {
         static std::vector<std::string>  TheV {"xPi","yPi"};
         return TheV;
    }
    static const std::vector<std::string> VNamesObs() 
    { 
       return TypeDist::NamesParamsOfDist(false);
    }
      static std::vector<std::string>  TheV;

    static const int TheNbUk = 2;
    static const int TheNbObs = TypeDist::TheNbDist + tCountDist::TheNbObsDist;
     
    static std::string FormulaName() { return "EqDist" + TypeDist::NameModel();}

    //  tUk and tObs differs, because with Jets  obs are double

       / *  Capital letter for 3D variable/formulas and small for 2D * /
    template <typename tUk, typename tObs>
    static     std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  )
    {
        assert (aVUk.size() ==TheNbUk) ;  // SD::UserSError("Bad size for unknown");
        assert (aVObs.size()==TheNbObs) ;// SD::UserSError("Bad size for observations");


        int aNumInd = 0;
        // 0 - Ground Coordinates of projected point
        const auto & xPi = aVUk.at(aNumInd++);
        const auto & yPi = aVUk.at(aNumInd++);
        assert(aNumInd==TheNbUk); // be sure we have consumed exacly the parameters

        // Now compute the distorsion
        auto   aVDist = TypeDist::PProjToImNorm(xPi,yPi,aVObs,tCountDist::TheNbObsDist);
        const auto & xDist =  aVDist.at(0);
        const auto & yDist =  aVDist.at(1);

        const auto & xPP = aVObs.at(0);
        const auto & yPP = aVObs.at(1);
        const auto & zPP = aVObs.at(2);

        auto xIm =  xPP  + zPP  * xDist;
        auto yIm =  yPP  + zPP  * yDist;

        return {xIm,yIm};
     }
};
*/



/*
template <class TypeDist>  class cEqCoLinearity
{
  public :
    typedef typename TypeDist::tCountDist  tCountDist;

    static constexpr int NbUk()  { return TheNbUk;}
    static constexpr int NbObs() { return TheNbObs;}
    static const std::vector<std::string> VNamesUnknowns() { return TypeDist::NamesParamsOfDist(true);}
    static const std::vector<std::string>& VNamesObs() { return tCountDist::VNamesObsCoLinear();}
    static std::string FormulaName() { return "EqColLinearity" + TypeDist::NameModel();}

    //  tUk and tObs differs, because with Jets  obs are double

       / *  Capital letter for 3D variable/formulas and small for 2D * /
    template <typename tUk, typename tObs>
    static     std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  )
    {
        assert (aVUk.size() ==TheNbUk) ;  // SD::UserSError("Bad size for unknown");
        assert (aVObs.size()==TheNbObs) ;// SD::UserSError("Bad size for observations");

        int aNumInd = 0;
        // 0 - Ground Coordinates of projected point
        const auto & XGround = aVUk.at(aNumInd++);
        const auto & YGround = aVUk.at(aNumInd++);
        const auto & ZGround = aVUk.at(aNumInd++);

        // 1 - Pose / External parameter
            // 1.1  Coordinate of camera center
        const auto & C_XCam = aVUk.at(aNumInd++);
        const auto & C_YCam = aVUk.at(aNumInd++);
        const auto & C_ZCam = aVUk.at(aNumInd++);

            // 1.2  Coordinate of Omega vector coding the unknown "tiny" rotation
        const auto & Wx = aVUk.at(aNumInd++);  // 6
        const auto & Wy = aVUk.at(aNumInd++);  // 7
        const auto & Wz = aVUk.at(aNumInd++);  // 8

        // 2 - Intrinsic parameters
             // 2.1 Principal point  and Focal
        const auto & xPP = aVUk.at(aNumInd++); // 9
        const auto & yPP = aVUk.at(aNumInd++); // 10
        const auto & zPP = aVUk.at(aNumInd++); // 11 also named focal
         assert(aNumInd==tCountDist::TheNbUkCoLinear); // be sure we have consumed exacly the parameters

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
        auto   aVDist = TypeDist::PProjToImNorm(xPi,yPi,aVUk,tCountDist::TheNbUkCoLinear);
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
    static const int  TheNbUk  = TypeDist::TheNbDist + tCountDist::TheNbUkCoLinear;
    static const int  TheNbObs = tCountDist::TheNbObsColinear;
};
*/



#endif // FORMULA_CAMSTENOPE_H
