#ifndef _FORMULA_CENTRAL_PROJ_H_
#define _FORMULA_CENTRAL_PROJ_H_

#include "MMVII_Ptxd.h"
#include "MMVII_PCSens.h"

#include "SymbDer/SymbolicDerivatives.h"
#include "SymbDer/SymbDer_MACRO.h"
#include "ComonHeaderSymb.h"

using namespace NS_SymbolicDerivative;



namespace MMVII
{
#if (0)

#endif

/* ********************************************************* */
/* ********************************************************* */
/* **                                                     ** */
/* **                PROJECTION                           ** */
/* **                                                     ** */
/* ********************************************************* */
/* ********************************************************* */

/*  For being a projection, class must define two functions :

     * Proj  3D->2D , the projection itself, it must work on formula as
       it will be used in code generation

     * ToDirBundle  2D->3D , more a less the invert, retun for a given 2 point p
       a 3D point P (the "direction") such that Proj(P) = p
*/

std::string FormulaName_ProjDir(eProjPC aProj);
std::string FormulaName_ProjInv(eProjPC aProj);

   // ==========================================================

///< Define definition domain for projection ok with z>0
class cDefinedZPos
{
   public :
        template<typename tScal> static double P3DIsDef(const  cPtxd<tScal,3>  & aP)
        {
           return VUnit(aP).z() ;
        }
};
///< Define definition domain for projection ok except at pole
class cUnDefinedAtPole
{
   public :
        template<typename tScal> static double P3DIsDef(const  cPtxd<tScal,3>  & aP)
        {
           return Norm2(VUnit(aP)-cPtxd<tScal,3>(0,0,-1)) ;
        }
};

       /**  Basic projection */

class cProjStenope : public cDefProjPerspC
{
   public :

        //  static const std::string & NameProj() {static std::string aName("Stenope"); return aName;}
        static eProjPC  TypeProj() {return eProjPC::eStenope;}


	///< no continuous extension for Z<=0
        tREAL8  P3DIsDef(const cPt3dr & aP) const override  {return cDefinedZPos::P3DIsDef(aP);}

        ///  questionable, as always defined could return 1.0, but it make sense to compute a distance to infinite point
        tREAL8  P2DIsDef(const cPt2dr & aP) const override
        {
            return 1.0/(1.0+Norm2(aP));
        }


        template<typename tScal> static std::vector<tScal> Proj(const  std::vector<tScal> & aXYZ)
        {
           MMVII_INTERNAL_ASSERT_tiny(aXYZ.size()==3,"Inconsistent param number");

           const auto & aX = aXYZ.at(0);
           const auto & aY = aXYZ.at(1);
           const auto & aZ = aXYZ.at(2);

           return {aX/aZ,aY/aZ};
        }

        template <typename tScal> static  std::vector<tScal>  ToDirBundle(const std::vector<tScal>  & aXY)
        {
           MMVII_INTERNAL_ASSERT_tiny(aXY.size()==2,"Inconsistent param number");

           const auto & aX = aXY.at(0);
           const auto & aY = aXY.at(1);
           tScal aC1 = CreateCste(1.0,aX);

           return {aX,aY,aC1};
        }
};

     /** class that allow to generate code from formulas */
/*
template <class  T ypeProj>  class cFormulaProj
{
    public :
};
*/

   // ==========================================================

       /**  fisheye equidistant  */

class cProjFE_EquiDist : public cDefProjPerspC
{
   public :

        //static const std::string & NameProj() {static std::string aName("FE_EquiDist"); return aName;}
        static eProjPC  TypeProj() {return eProjPC::eFE_EquiDist;}
	///< fish eye can have FOV over 180, and mathematicall formula is defined execte at pole (axe 0 0 -1 )
        tREAL8  P3DIsDef(const cPt3dr & aP) const override  {return cUnDefinedAtPole::P3DIsDef(aP);}

        /// image of pole converge to the circle of ray PI, 
        tREAL8  P2DIsDef(const cPt2dr & aP) const override
        {
            return M_PI - Norm2(aP) ;
	}
/*
  theoretically :
         R = sqrt(X^2+Y2)  tan(teta)=R/Z   teta = atan2(R,Z)
         X,Y,Z =>  teta * (X/R,Y/R)

  all fine   BUT in (0,0)  (X/R,Y/R) is not defined and cannot be prolonged continuously,
  maybe its not a problem because it is multiplied by teta that tend to 0. However I prefer
  to have only explicitely C-infinite function.

  So we define AtanXY_sX  by :
      *  AtanXY_sX(A,B) = atan2(A,B)/A  if A!=0
      *  AtanXY_sX(A,B) = 1             if A==0  
  This prolongation makes the function continuous, and in fact  C-infinite

  For the computing implementation, we must avoid 0/0, and in fact division by tiny
  numbers.  So there is a test and if X is sufficiently big regarding to R, we use
  the atan/R , else we use an ad-hoc taylor expansion (1-X^2/3 ....)
   
  AtanXY_sX  using atan/X for non small value and taylor expansion for too small values

  Finnaly we can write
         X,Y,Z =>  AtanXY_sX(R,X) * (X,Y)
*/

        template<typename tScal> static std::vector<tScal> Proj(const  std::vector<tScal> & aXYZ)
        {
           MMVII_INTERNAL_ASSERT_tiny(aXYZ.size()==3,"Inconsistent param number");

           const auto & aX = aXYZ.at(0);
           const auto & aY = aXYZ.at(1);
           const auto & aZ = aXYZ.at(2);
           const auto  r = NormL2V2(aX,aY);
           const auto aTeta_sR  = AtanXsY_sX(r,aZ);

           return {aX*aTeta_sR,aY*aTeta_sR};
        }
/*  For "inverse" mapping, we have :
     U = teta X/R   V = teta Y/R  with X2+Y2=R2

      teta = sqrt(U2+V2)

     the direction is   :
            (X/R sin(teta), Y/R sin(teta), cos(teta) )
         = (U/teta sin(teta), V/teta sin(teta), cos(teta))
         = (U sinc(teta), V sinc(teta), cos(teta))
    As AtanXsY_sX,  the sinC implementation is a mix of sin/x and taylor expension close to 0
*/
        template <typename tScal> static std::vector<tScal>  ToDirBundle(const std::vector<tScal> & aXY)
        {
           MMVII_INTERNAL_ASSERT_tiny(aXY.size()==2,"Inconsistent param number");

           auto aTeta = NormL2Vec2(aXY);
           auto aSinC = sinC(aTeta);

           return {aXY[0]*aSinC,aXY[1]*aSinC,cos(aTeta)};
        }
};

   // ==========================================================


       /**  fisheye orthographic  */


class cProjStereroGraphik : public cDefProjPerspC
{
   public :
        // static const std::string & NameProj() {static std::string aName("FE_StereoGr"); return aName;}
        static eProjPC  TypeProj() {return eProjPC::eStereroGraphik;}
	//
	///< fish eye can have FOV over 180, and mathematicall formula is defined execte at pole (axe 0 0 -1 )
        tREAL8  P3DIsDef(const cPt3dr & aP) const override  {return cUnDefinedAtPole::P3DIsDef(aP);}

        /// more or less as stenope, defined everywhere but be not to close to infinite point ...
        tREAL8  P2DIsDef(const cPt2dr & aP) const override
        {
            return 1 / (1+Norm2(aP));
	}
/*  Theory :
        r=sqrt(X2+Y2)  R = sqrt(X2+Y2+Z2)   r' = r/R  z'=Z/R

        In the vertical plane containing the bundle, the sphere of direction intersect in a
        circle; the coordinate of the bundle on a circle  is P=[r',1-z'], let Q=[0,2] 
        be the top of the circle, we must compute the intersection of line PQ with line y=0,
        they intersect on a point (x',0) and we have 

           [x'  -r' ]
           [-2  1+z']  =0  => x' = 2r'/(1+z')

        x' is the distance, to have the coordinat we multiply by vector X/r  Y/r

      (X/r)                 (X/r)               (X)
      (Y/r) * 2r'/(1+z') =  (Y/r) * 2r /(R+Z) = (Y) * 2 /(R+Z)

Also :
       t2 = 2T/(1-T^2)   T^2 +2T/t2 -1 =0
*/
        template<typename tScal> static std::vector<tScal> Proj(const  std::vector<tScal> & aXYZ)
        {
           MMVII_INTERNAL_ASSERT_tiny(aXYZ.size()==3,"Inconsistent param number");

           const auto & aX = aXYZ.at(0);
           const auto & aY = aXYZ.at(1);
           const auto & aZ = aXYZ.at(2);
           tScal aC2 = CreateCste(2.0,aX);

           const auto  aR = NormL2V3(aX,aY,aZ);
           const auto  aMul = aC2 /(aR+aZ);
           return {aMul*aX,aMul*aY};
        }
/* Inverse we have:

       U = 2X/(R+Z)   V=2Y/(R+Z)  or U=2x'/(1+z')  V=2y'/ (1+z')  (with x'=X/R ..z'=Z/R)
    
   and x'^2 + y'^2 + z'^2 = 1  substituing with x'=U(1+z')/2 and noting r2=U^2+V^2

   we have :  z'^2 + r2/4 (1+z')^2 = 1

    This give two solution, one z'=-1 has no interest (intersection with top of the circle) and the 
    other :
         z' = 2/(1+r2/4) -1, and then x'=U/(1+r2/4) ...
   

*/
        template <typename tScal> static std::vector<tScal>  ToDirBundle(const  std::vector<tScal> & aXY)
        {
           // auto aDiv = (1+SqN2(aP)/4.0);
           MMVII_INTERNAL_ASSERT_tiny(aXY.size()==2,"Inconsistent param number");

           const auto & aX = aXY.at(0);
           const auto & aY = aXY.at(1);
           tScal aC1 = CreateCste(1.0,aX);
           tScal aC2 = CreateCste(2.0,aX);
           tScal aC4 = CreateCste(4.0,aX);

           auto aDiv = (aC1+SqNormL2V2(aX,aY)/aC4);
           return {aX/aDiv,aY/aDiv,aC2/aDiv-aC1};
        }
};



   // ==========================================================

       /**  Orthographic Projection  */

class cProjOrthoGraphic : public cDefProjPerspC
{
   public :
/* Quite basic :
    Direct N = vunit(P)  => N.x,Ny
    Invese Norm(N) = sin(alpha) , cos = sqrt(1-sin^2) 
*/
        // static const std::string & NameProj() {static std::string aName("FE_OrthoGr"); return aName;}
        static eProjPC  TypeProj() {return eProjPC::eOrthoGraphik;}
	///< no continuous extension for Z<=0
        tREAL8  P3DIsDef(const cPt3dr & aP) const override  {return cDefinedZPos::P3DIsDef(aP);}

        ///  The equator project on circle of ray 1,  over its is degenerate and the formula cannot be inverted
        tREAL8  P2DIsDef(const cPt2dr & aP) const override
        {
            return 1.0 - Norm2(aP);
        }



        template<typename tScal> static std::vector<tScal> Proj(const  std::vector<tScal> & aXYZ)
        {
           MMVII_INTERNAL_ASSERT_tiny(aXYZ.size()==3,"Inconsistent param number");

           const auto & aX = aXYZ.at(0);
           const auto & aY = aXYZ.at(1);
           const auto & aZ = aXYZ.at(2);

           auto aR = sqrt(Square(aX)+Square(aY)+Square(aZ));
           return {aX/aR,aY/aR};
        }

        template <typename tScal> static std::vector<tScal>  ToDirBundle(const  std::vector<tScal> & aXY)
        {
           MMVII_INTERNAL_ASSERT_tiny(aXY.size()==2,"Inconsistent param number");

           const auto & aX = aXY.at(0);
           const auto & aY = aXY.at(1);
           tScal aC1 = CreateCste(1.0,aX);

           // return {aX,aY,sqrt(std::max(0.0,1.0-SqN2(aP)))};
           //  !!  Warning  -> have supress the max for sort term derivation
           // StdOut() << "Warning cProjOrthoGraphic::ToDirBundle \n";
           return {aX,aY,sqrt(aC1-SqNormL2V2(aX,aY))};

        }
};

   // ==========================================================
   
/*
     Theory  :

         Bundle  :
	    U,V  ->  
	           X = cos V sin U
		   Y = sin V
		   Z = cos V cos U

           X,Y,Z : 
              R = sqrt(X^2+Y^2+Z^2) 
	      x = X/R  y= Y/R  z=Z/R

          V = ASin(y)
	  U = atan2(x,z) = atan2(X,Z)
 */
class cProj_EquiRect  : public cDefProjPerspC
{
   public :
        bool  HasRadialSym() const override {return false;}
        static eProjPC  TypeProj() {return eProjPC::eEquiRect;}
	///< always defines, but singular when x,z=0
        tREAL8  P3DIsDef(const cPt3dr & aP) const override  
	{
             return 1.0 -std::abs(aP.y()/Norm2(aP));
	}

        ///  can be prolongated at  equtor not at the pole
        tREAL8  P2DIsDef(const cPt2dr & aP) const override
        {
            tREAL8 aU = aP.x();
            tREAL8 aV = aP.y();
            return std::min
		    (
		       std::min(M_PI+aU,M_PI-aU)     ,
		       std::min(M_PI_2+aV,M_PI_2-aV)
		    );
        }
        template<typename tScal> static std::vector<tScal> Proj(const  std::vector<tScal> & aXYZ)
	{
           const auto & aX = aXYZ.at(0);
           const auto & aY = aXYZ.at(1);
           const auto & aZ = aXYZ.at(2);

	   auto aR = sqrt(Square(aX)+Square(aY)+Square(aZ));

	   return  {ATan2(aX,aZ),ASin(aY/aR)} ;
	}

        template <typename tScal> static std::vector<tScal> ToDirBundle(const  std::vector<tScal> & aXY)
        {
           MMVII_INTERNAL_ASSERT_tiny(aXY.size()==2,"Inconsistent param number");

           const auto & aU = aXY.at(0);
           const auto & aV = aXY.at(1);
           auto CosV  = cos(aV);

           return {CosV * sin(aU) ,  sin(aV) , CosV *cos(aU)};
        }
};



       /**  EquiSolid (Equal-Area,equivalente) Projection  */

class cProjFE_EquiSolid  : public cDefProjPerspC
{
   public :
        // static const std::string & NameProj() {static std::string aName("FE_EquiSolid"); return aName;}
        static eProjPC  TypeProj() {return eProjPC::eFE_EquiSolid;}

	///< like equidisant 
        tREAL8  P3DIsDef(const cPt3dr & aP) const override  {return cUnDefinedAtPole::P3DIsDef(aP);}

        ///  The equator project on circle of ray 1,  over its is degenerate and the formula cannot be inverted
        tREAL8  P2DIsDef(const cPt2dr & aP) const override
        {
            return 2-Norm2(aP);
        }
/* Theory
    R2 = X2 +Y2+Z2   , r2 = X2+Y2
    (X,Y,Z)  = R (sin A,0,cos A) ->  2 sin(A/2) = L  
    sin(A/2) =  sqrt((1-cosA)/2)  cos(A) = Z/R
    L = 2 sqrt((1-Z/R)/2)
    (X,Y,Z) => L (X/r,Y/r)
    LX/r = 2 X/r sqrt((1-Z/R)/2)  = 2X/R sqrt((R2-ZR)/2r2) =2X/R sqrt(R(Z-R)(Z+R)/2r2(Z+R))
    = 2X/R sqrt(R/2(R+Z))
    
    Finally (X,Y,Z) => 2(X/R,Y/R) sqrt(R/2(R+Z))
*/

        template<typename tScal> static std::vector<tScal> Proj(const  std::vector<tScal> & aXYZ)
        {
           MMVII_INTERNAL_ASSERT_tiny(aXYZ.size()==3,"Inconsistent param number");

           const auto & aX = aXYZ.at(0);
           const auto & aY = aXYZ.at(1);
           const auto & aZ = aXYZ.at(2);
           tScal aC2 = CreateCste(2.0,aX);

           auto aR = sqrt(Square(aX)+Square(aY)+Square(aZ));
           auto aDiv = sqrt(aC2*aR*(aR+aZ));
           return {(aC2*aX)/aDiv,(aC2*aY)/aDiv};
        }
/*
   Theory
     r2=U2+V2     r=2sin(a/2)  a = 2 asin(r)

    U/r sin(A) = U/(2sin(a/2)) (2sin(a/2)cos(a/2))  = U cos(a/2)
*/

        template <typename tScal> static std::vector<tScal> ToDirBundle(const  std::vector<tScal> & aXY)
        {
           MMVII_INTERNAL_ASSERT_tiny(aXY.size()==2,"Inconsistent param number");

           const auto & aX = aXY.at(0);
           const auto & aY = aXY.at(1);
           tScal aC2 = CreateCste(2.0,aX);

           auto r = NormL2Vec2(aXY);
           auto A = aC2 * ASin(r/aC2);
           auto cAs2 = cos(A/aC2);

           return {aX*cAs2,aY*cAs2,cos(A)};
        }
};


template <typename tProj> class   cGenCode_ProjDir
{
	public :
            static const std::vector<std::string> VNamesUnknowns() { return {"PtX","PtY","PtZ"}; }
            static const std::vector<std::string> VNamesObs()      { return {}; }

            std::string FormulaName() const { return  FormulaName_ProjDir(tProj::TypeProj());}

	    template <typename tUk>
                     std::vector<tUk> formula
                     (
                           const std::vector<tUk> & aVUk,
                           const std::vector<tUk> & aVObs
                      ) const
             {
		     return tProj::Proj(aVUk);
             }


};

template <typename tProj> class   cGenCode_ProjInv
{
	public :
            static const std::vector<std::string> VNamesUnknowns() { return {"PixI","PixJ"}; }
            static const std::vector<std::string> VNamesObs()      { return {}; }

            std::string FormulaName() const { return FormulaName_ProjInv(tProj::TypeProj());}

	    template <typename tUk>
                     std::vector<tUk> formula
                     (
                           const std::vector<tUk> & aVUk,
                           const std::vector<tUk> & aVObs
                      ) const
             {
		     return tProj::ToDirBundle(aVUk);
             }


};

/**  Proj manipulate vector for code generation, but for testing it is easier to manipulate points */
template <typename tProj> class   cHelperProj
{
	public :

           template<typename tScal> static cPtxd<tScal,2>  Proj(const  cPtxd<tScal,3> & aXYZ)
	   {
		   //return  cPtxd<tScal,2>::FromStdVector(tProj::Proj(aXYZ.ToStdVector()));
		   return  VtoP2(tProj::Proj(ToVect(aXYZ)));
	   }
           template<typename tScal> static cPtxd<tScal,3>  ToDirBundle(const  cPtxd<tScal,2> & aXYZ)
	   {
		   //return  cPtxd<tScal,3>::FromStdVector(tProj::ToDirBundle(aXYZ.ToStdVector()));
		   return  VtoP3(tProj::ToDirBundle(ToVect(aXYZ)));
	   }

};






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



};//  namespace MMVII
#endif // _FORMULA_CENTRAL_PROJ_H_
