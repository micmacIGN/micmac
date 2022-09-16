#ifndef FORMULA_CAMSTENOPE_H
#define FORMULA_CAMSTENOPE_H

#include "include/MMVII_all.h"
#include "include/SymbDer/SymbolicDerivatives.h"
#include "include/SymbDer/SymbDer_MACRO.h"
#include "ComonHeaderSymb.h"

namespace NS_SymbolicDerivative
{
//  add sinus cardinal and arc-sinus as operator on formulas
MACRO_SD_DEFINE_STD_UNARY_FUNC_OP_DERIVABLE(MMVII,ASin,DerASin)
MACRO_SD_DEFINE_STD_UNARY_FUNC_OP_DERIVABLE(MMVII,sinC,DerSinC)

/// See bellow the needs of a differentiable operator AtanXsY_sX=atan(X/Y)/X C-infinite
MACRO_SD_DEFINE_STD_BINARY_FUNC_OP_DERIVABLE(MMVII,AtanXsY_sX,DerXAtanXsY_sX,DerYAtanXsY_sX)

}
using namespace NS_SymbolicDerivative;

namespace MMVII
{


/*  
    ================ Physicall modelisation ===============

 Ground          Camera            P Proj        Pdistorded            Pixel
 coordinate     Coordinate                                              
     Pg  ------>    Pc   -------->    Pp   ----->  Pd   -------------->    Pix
           |                |                |                |
    Camera Center     Stenope          Fraser, radial,     Focal, 
   & Orientation     or  Fisheye      Polynomial...       principal point

   6 param           0 param          2 or 7 or ...        3 param
                                      or many  param       
   
    Extrinseq                           Intrinseq           Intrinseq

*/

class cMMVIIUnivDist;  // can generate distorsion as fun R2->R2, or basis of these function 
class cDefinedZPos;    //  define definition validity for some projection (Z>0)
class cUnDefinedAtPole;  // define definition validity for some projection , define every where but at pole
      // projection
class cProjStenope ;      //  classical  stenope
class cProjFE_EquiDist;    // fish-eye equidisant
class cProjStereroGraphik ; // stereographic
class cProjOrthoGraphic ;   // orthographic
class cProjFE_EquiSolid ;   // fish-eye equi solid
 // to do xyz <--> teta phi for "360" degree images
 
template <typename tProj> class   cGenCode_ProjDir;  // class to generate code for projection  x,y,z -> i,j
template <typename tProj> class   cGenCode_ProjInv;  // class to generate code for projection   i,j -> x,y,z

template <typename TypeDist>  class cEqDist;  // class to generate code for direct distorsion  Pp->Pd
template <typename TypeDist>  class cEqIntr;  // class for generating    Pp-> Pix



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



/* ********************************************************* */
/* ********************************************************* */
/* **                                                     ** */
/* **               DISTORSION                            ** */
/* **                                                     ** */
/* ********************************************************* */
/* ********************************************************* */

/*  In MMVII , was made the choice of having a unique template class of distorsion,
  knowing that it is the sum of three term, each term is paremitrisable, it include almost
  all the model existing in MMV1 and is much easier to maintain.


   # [3SUM] The distorsion is the sum of :

      *  a radial model assuming perfect cylindric symetric of the sensor
      *  a decentric model,  taking into account small misalingment in lenses
      *  a polynomial model taking into account non planarity but, with sufficient degree,
         possibly any deformation

   # [PPF]  In our model, the distorsion operates before PP and focal, so they operates on
   undimentionnal numbers, and all the coefficient are undimentionned.

   # [PP=CD] The model assumme PP and distorsion center are identic (because having them
     different would creat  high correlation if we add also decentric). 

   # [CD00]  As a consequence  , the center of radial distorsion is (0,0)
     (=> (0,0)  is principal point, because [PPF], and then distorsion center because [PP=CD]).

   # [RADEVEN] Classicaly, assuming distorsion is C-infinite, leads to have only even coeffs

   # [DEC] The decentric distorsion modelize the difference optical axes, that can also be
     modelize as combination of several radial functions, the parametric model can be reduced
     to differenciation of the radial distorsion as a function of the center.

      We have  the Nth function of radial distorsion as :
         XN = Xc(Xc2+Yc2)^N    and  YN = Yc(Xc2+Yc2)^N
      And then :
       (dXN,dYN)
        -------  =[R2^N+2NX^2R2^(N-1),2NXYR2^(N-1)] :N=1=>[R2+2X^2,2XY]=[3X2+Y2,2XY]
        dCx

       (dXN,dYN)
        -------  =[2XYNR2^(N-1), R2^N+2NY^2R^(N-1)  ] :N=1=>[2XY,R2+2Y^2]=[2XY,X2+3Y2]
        dCy
 
      If we limit to the first term, we recognize current term in the literrature.
 
   # [POL] The polynomial distorsion is meant  to approximate any smooth function.
    Let D be the degree the  polynomial distortion we will have :
      Distx = Som (dx_ij   X^i Y^j)   i+j<=D
      Disty = Som (dy_ij   X^i Y^j)   i+j<=D
    However, to avoid a redundant coding of the distorsion there
    is several dx_ij and dy_ij that we will not include because they are already
    modelized.  The decision for each coefficient depends of the degree and the fact that 
    it is x or y. Precisely the rules are :

     - Degre 0 =>  avoid  all because principal point already modelize translation
     - Degre 1, noting  [dx_10 dx_01 dy_10 dy_01] the degree 1 parameter
           * [1 0 0 1] is a pure homothety, modelize by  focal, so avoid it
           * [0 -1 1 0] is a pure rotation and redundant with rotation around Z-axe, avoid it
           * so we have to select a complementary base, many choice are possible, we select
            [1 0 0 0]  [0 1 0 0] because it is coherent with literature (Fraser's)
     - Degre 2 :
          * Rotation arround X  + linear are highly correlated to X2 + affine, so we muste
            so avoid X2 in X
          * Idem avoid Y2 in Y

      - Degre 2,4 ... EVEN :
         If the decentrik distorsion is of degree  Dec, avoid X^k Y^k for k<=Dec

      - Degre 3,5, ... ODD
         If the radial  distorsion is of degree  Rad, avoid dx= X^(2k+1) for k <= Rad

    Finnaly we have :

*/


/**  This class aims to be the universol model of distorsion in MicMac, it is templ
*/

// template <const int TheDegRad,const int TheDegDec,const int TheDegUniv> class cMMVIIUnivDist
class cMMVIIUnivDist
{
    public :
       
       const int & DegRad()  const  {return mTheDegRad;}
       const int & DegDec()  const {return mTheDegDec;}
       const int & DegUniv() const {return mTheDegUniv;}

       /// Usable for message, also for name generation in formulas
       std::string  NameModel()  const
       {
           return    std::string("Dist")
                   + std::string(mForBase?"_Base" : "")
                   + std::string("_Rad") + std::to_string(DegRad())
                   + std::string("_Dec") + std::to_string(DegDec())
                   + std::string("_XY") + std::to_string(DegUniv())
           ;

       }
       /// Used to indicate reason why monom sould be removes. To maintain for debug  ?
       void ShowElim(const std::string aMesWhy, bool isX,  int aDegX, int aDegY ) const
       {
          // StdOut() << aMesWhy << " " << (isX?"x":"y")    << " " << aDegX  << " " << aDegY << "\n";
       }


       /** 
        Indicate if X^DegX  Y ^DegY is to maintain in  Dist X or Dist Y (isX means DistX)
        We must create a free familly of polynome taking into account :
           * the poolynoms existing in radial distorsion
           * the polymoms existing in decentrik distorsion
           * the polymoms modelized by focal an principal points
          * the polynom that can be approximed by rotation arround projection
       */


       bool OkMonome
                   (
                       bool isX,  // Is it x component of distorsion
                       int aDegX, // Degree in x
                       int aDegY  // Degree in y
                   ) const
       {
            // degre 0 : avoid, it's already modelized by PP
            if ((aDegX ==0) && (aDegY==0)) 
               return false;  

            //  because of redundancy with focal & rotation arround Z axe, we must avoid
            // 2 out of 4 degree 1 monoms, we suppress arbitrarily the Y function 
            // (because its coherent with most current  convention on "fraser" model :
            //  dx = b1 x + b2 y ...

            if ((!isX) && ((aDegX + aDegY) ==1))    
            {
               ShowElim("Aff",isX,aDegX,aDegY);
               return false; 
            }

            // because of redundaucy with non plane rotation, we supress 2 degree 2 function
            // there many choice possible, for the first one, considering a rotation arround
            // x axes + other monomes can create a taylor devlopement of Dx= X2

               //  X2 in dX avoid
            if (isX &&    (aDegX==2) &&  (aDegY ==0))  
            {
                ShowElim("ROT",isX,aDegX,aDegY);
                return false;  
            }

               //  Y2 in dY avoid
            if ((!isX) && (aDegX==0) &&  (aDegY ==2))  
            {
               ShowElim("ROT",isX,aDegX,aDegY);
               return false;  
            }

            // Now we must avoid redundancy with radial distortion, 
            //    x(x2+y2)^n  , y (x2+y2)^n, arbirarily, we supress x3,x5,.. in dx
            //  (aDegX>2)=> for deg 1, already many rules and no radial param                                       
            if (isX && (aDegY ==0) && ((aDegX%2)==1) && (aDegX>2) && (aDegX<=(1+2*DegRad())))
            {
               ShowElim("RAD",isX,aDegX,aDegY);
               return false;  
            }

            // Now we must avoid redundancy with decentrik distorsion
            //  [R2^N+2NX^2R2^(N-1),2NXYR2^(N-1)] :N=1=>[R2+2X^2,2XY]=[3X2+Y2,2XY]
            // [2XYNR2^(N-1), R2^N+2NY^2R^(N-1)  ] :N=1=>[2XY,R2+2Y^2]=[2XY,X2+3Y2]
            // The terme to supress is a bit arbitrary, BUT we cannot supress x2 in x, because
            // it is already supress by rotations, select X2 in Y and Y2 in X

            //   supress X^n Y^n, not X^0Y^0 (already PP) and 
            if ( (aDegX==aDegY) && (aDegX>=1) && (aDegX<=DegDec()))
            {
               // StdOut() << "DEC " << (isX?"x":"y")    << " " << aDegX  << " " << aDegY << "\n";
               ShowElim("DEC",isX,aDegX,aDegY);
               return false;
            }

            return true;  // then ok
       }

       /**  Vector describing the attribute of each parameters, 
            This explicit description will be used for :

                *  formula generation . 
                *  name generation
                *  generating description or accessing parameters by names
       */

       std::vector<cDescOneFuncDist>  VDescParams() const
       {
           // static_assert(DegRad>=DegDec(),"Too much decentrik");
              std::vector<cDescOneFuncDist>  VDesc;
              // Generate description of radial parameters, x used for num, y not used => -1
              for (int aDR=1 ; aDR<=DegRad() ; aDR++)
              {
                  VDesc.push_back(cDescOneFuncDist(eTypeFuncDist::eRad,cPt2di(aDR,-1)));
              }

              // Generate description of decentrik parameter, x used for num, y not used => -1
              for (int aDC=1 ; aDC<=DegDec() ; aDC++)
              {
                  VDesc.push_back(cDescOneFuncDist(eTypeFuncDist::eDecX,cPt2di(aDC,-1)));
                  VDesc.push_back(cDescOneFuncDist(eTypeFuncDist::eDecY,cPt2di(aDC,-1)));
              }

              // Generate description of monomes in X and Y that are to maintain
              for (int aDx=0 ; aDx<=DegUniv() ; aDx++)
              {
                  for (int aDy=0 ; (aDx+aDy)<=DegUniv() ; aDy++)
                  {
                      cPt2di aDXY(aDx,aDy);
                      if (OkMonome(true,aDx,aDy))
                      {
                         VDesc.push_back(cDescOneFuncDist(eTypeFuncDist::eMonX,aDXY));
                      }

                      if (OkMonome(false,aDx,aDy))
                      {
                         VDesc.push_back(cDescOneFuncDist(eTypeFuncDist::eMonY,aDXY));
                      }
                  }
              }
              return VDesc;
        }

       /**  Names of all param, used to automatize symbolic derivation */
        const std::vector<std::string>  VNamesParams() const
        {
           std::vector<std::string>  VNames;
           if (VNames.empty() && (!mForBase))
           {
               for (const auto & aDesc : VDescParams())
               {
                  VNames.push_back(aDesc.mName);
               }
           }
           return VNames;
        }

       /**  Main method  Generate the formula of distorsion, tScal can be a formula */
        template<typename tScal> std::vector<tScal> 
                PProjToImNorm 
                (
                     const tScal & xIn,const tScal & yIn,
                     const std::vector<tScal> &  aVParam, 
                     unsigned int              aK0P
                ) const
       {
           tScal aC0 = CreateCste(0.0,xIn);
           tScal aC1 = CreateCste(1.0,xIn);
           std::vector<tScal>  aVBaseX;
           std::vector<tScal>  aVBaseY;

           int aPowR2 = std::max(DegRad(),DegDec());
           int aPowMon = std::max(2,DegUniv());

           //  Generate the momom X^k and Y^k in vectors
           std::vector<tScal> aVMonX;  ///< aVMonX[k] contains X^k
           std::vector<tScal> aVMonY;  ///< aVMonY[k] contains Y^k
           aVMonX.push_back(aC1);
           aVMonY.push_back(aC1);
           for (int aD=1 ;aD<=aPowMon ; aD++)
           {
              aVMonX.push_back(aVMonX.back()*xIn);
              aVMonY.push_back(aVMonY.back()*yIn);
           }

           //  Generate power of R^2 in a vector
           tScal aX2 = aVMonX.at(2) ; // Not sur its better than xIn * xIn
           tScal aY2 = aVMonY.at(2);
           tScal aXY = xIn * yIn;
           tScal aRho2 = aX2 + aY2;
           std::vector<tScal> aVRho2; ///< aVRho2[k] contains (X^2+Y^2) ^ k
           aVRho2.push_back(aC1);
           for (int aD=1 ;aD<=aPowR2 ; aD++)
           {
              aVRho2.push_back(aVRho2.back()*aRho2);
           }
           // auto 

           tScal aDR2   =  aC0; ///< will contain 1 + K1 R^2 + K2 R ^4 + ...
           tScal aDecX  =  aC0; ///< will contain X component of decentric distorsion
           tScal aDecY  =  aC0; ///< will contain Y component of decentric distorsion
           tScal aPolX  =  aC0; ///< will contain X component of polynomial distorsion
           tScal aPolY  =  aC0; ///< will contain Y component of polynomial distorsion
           for (const auto & aDesc : VDescParams())
           {
              tScal aParam = mForBase ? aC0 : aVParam.at(aK0P++); // If for base, Param empty
              tScal aBaseX = aC0;  //  because must init, init ok for monoms
              tScal aBaseY = aC0;  //  because must init, init ok for monoms
              if (aDesc.mType <= eTypeFuncDist::eDecY)
              {
                 int aNum = aDesc.mNum;
                 tScal aR2N   = aVRho2.at(aNum);
                 tScal aR2Nm1 = aVRho2.at(aNum-1);
                 if (aDesc.mType==eTypeFuncDist::eRad)
                 {
                    aDR2 = aDR2 + aParam * aR2N  ;
                    aBaseX = xIn*aR2N;
                    aBaseY = yIn*aR2N;
                 }
                 //  [R2^N+2NX^2R2^(N-1),2NXYR2^(N-1)] :N=1=>[R2+2X^2,2XY]=[3X2+Y2,2XY]
                 else if (aDesc.mType==eTypeFuncDist::eDecX)
                 {
                    aBaseX = aR2N + aX2*(2.0*aNum*aR2Nm1);
                    aBaseY = aXY*(2.0*aNum*aR2Nm1);
                    aDecX = aDecX + aParam* aBaseX;
                    aDecY = aDecY + aParam* aBaseY;
                 }
                 // [2XYNR2^(N-1), R2^N+2NY^2R^(N-1)  ] :N=1=>[2XY,R2+2Y^2]=[2XY,X2+3Y2]
                 else if (aDesc.mType==eTypeFuncDist::eDecY)
                 {
                    aBaseX = aXY*(2.0*aNum*aR2Nm1);
                    aBaseY = aR2N + aY2*(2.0*aNum*aR2Nm1);
                    aDecX = aDecX + aParam* aBaseX;
                    aDecY = aDecY + aParam* aBaseY;
                 }
              }
              else
              {
                 cPt2di aD = aDesc.mDegMon;
                 tScal aMon   =  aVMonX.at(aD.x()) * aVMonY.at(aD.y());
                 if (aDesc.mType==eTypeFuncDist::eMonX)
                 {
                    aPolX = aPolX + aParam* aMon;
                    aBaseX = aMon;
                 }
                 else
                 {
                    aPolY = aPolY + aParam* aMon;
                    aBaseY = aMon;
                 }
              }
              if (mForBase)
              {
                 aVBaseX.push_back(aBaseX);
                 aVBaseY.push_back(aBaseY);
              }
           }
           tScal xDist =  xIn + xIn*aDR2 + aDecX + aPolX;
           tScal yDist =  yIn + yIn*aDR2 + aDecY + aPolY;
           MMVII_INTERNAL_ASSERT_always(aK0P==aVParam.size(),"Inconsistent param number");

           if (mForBase) 
              return Append(aVBaseX,aVBaseY) ;

           return        {xDist,yDist}  ;
       }

       cMMVIIUnivDist(const int & aDegRad,const int & aDegDec,const int & aDegUniv,bool ForBase) :
          mTheDegRad  (aDegRad),
          mTheDegDec  (aDegDec),
          mTheDegUniv (aDegUniv),
          mForBase    (ForBase)
       {
       }



    private :
       int    mTheDegRad;
       int    mTheDegDec;
       int    mTheDegUniv;
       bool   mForBase;   // If true, generate the base of function and not the sum
};


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


   // ==========================================================

///< Define definition domain for projection ok with z>0
class cDefinedZPos
{
   public :
        template<typename tScal> static double DegreeDef(const  cPtxd<tScal,3>  & aP)
        {
           return VUnit(aP).z() ;
        }
};
///< Define definition domain for projection ok except at pole
class cUnDefinedAtPole
{
   public :
        template<typename tScal> static double DegreeDef(const  cPtxd<tScal,3>  & aP)
        {
           return Norm2(VUnit(aP)-cPtxd<tScal,3>(0,0,-1)) ;
        }
};

       /**  Basic projection */

class cProjStenope : public cDefinedZPos
{
   public :
        //  static const std::string & NameProj() {static std::string aName("Stenope"); return aName;}
        static eProjPC  TypeProj() {return eProjPC::eStenope;}

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

class cProjFE_EquiDist : public cUnDefinedAtPole
{
   public :

        //static const std::string & NameProj() {static std::string aName("FE_EquiDist"); return aName;}
        static eProjPC  TypeProj() {return eProjPC::eFE_EquiDist;}
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


class cProjStereroGraphik : public cUnDefinedAtPole
{
   public :
        // static const std::string & NameProj() {static std::string aName("FE_StereoGr"); return aName;}
        static eProjPC  TypeProj() {return eProjPC::eStereroGraphik;}
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

class cProjOrthoGraphic : public cDefinedZPos
{
   public :
/* Quite basic :
    Direct N = vunit(P)  => N.x,Ny
    Invese Norm(N) = sin(alpha) , cos = sqrt(1-sin^2) 
*/
        // static const std::string & NameProj() {static std::string aName("FE_OrthoGr"); return aName;}
        static eProjPC  TypeProj() {return eProjPC::eOrthoGraphik;}

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

           // return {aX,aY,sqrt(std::max(0.0,1.0-SqN2(aP)));
           //  !!  Warning  -> have supress the max for sort term derivation
           // StdOut() << "Warning cProjOrthoGraphic::ToDirBundle \n";
           MMVII_WARGING("Warning cProjOrthoGraphic::ToDirBundle");
           return {aX,aY,sqrt(aC1-SqNormL2V2(aX,aY))};

        }
};

   // ==========================================================
   

       /**  EquiSolid (Equal-Area,equivalente) Projection  */

class cProjFE_EquiSolid  : public cUnDefinedAtPole
{
   public :
        // static const std::string & NameProj() {static std::string aName("FE_EquiSolid"); return aName;}
        static eProjPC  TypeProj() {return eProjPC::eFE_EquiSolid;}
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
           // auto A = 2 * std::asin(std::min(1.0,r/2.0));
           MMVII_WARGING("Warning cProjFE_EquiSolid::ToDirBundle");
           auto A = aC2 * ASin(r/aC2);
           auto cAs2 = cos(A/aC2);

           return {aX*cAs2,aY*cAs2,cos(A)};
        }
};

std::string FormulaName_ProjDir(eProjPC aProj);
std::string FormulaName_ProjInv(eProjPC aProj);

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

/**  Proj manipulate vector for code generation, but for testing its easier to manipulate points */
template <typename tProj> class   cHelperProj
{
	public :

           template<typename tScal> static cPtxd<tScal,2>  Proj(const  cPtxd<tScal,3> & aXYZ)
	   {
		   return  cPtxd<tScal,2>::FromStdVector(tProj::Proj(aXYZ.ToStdVector()));
	   }
           template<typename tScal> static cPtxd<tScal,3>  ToDirBundle(const  cPtxd<tScal,2> & aXYZ)
	   {
		   return  cPtxd<tScal,3>::FromStdVector(tProj::ToDirBundle(aXYZ.ToStdVector()));
	   }

};





/*
 * Dont know if its usefull for now
template <typename tScal>  
  inline std::vector<tScal> 
    TransformPPxyz(const std::vector<tScal> & aPts,const std::vector<tScal> & aParam,int aInd)
{
        const auto & xIn =  aPts.at(0);
        const auto & yIn =  aPts.at(1);

        const auto & xPP = aParam.at(aInd++);
        const auto & yPP = aParam.at(aInd++);
        const auto & zPP = aParam.at(aInd++);

        auto xOut =  xPP  + zPP  * xIn;
        auto yOut =  yPP  + zPP  * yIn;

        return {xOut,yOut};
}
*/

template <typename TypeDist>  class cEqDist
{
  public :
    cEqDist(const TypeDist & aDist) : mDist      (aDist) { }
    static std::vector<std::string>  VNamesUnknowns() {return {"xPi","yPi"}; }
    const std::vector<std::string>    VNamesObs() const {return mDist.VNamesParams();}
     
    std::string FormulaName() const { return "EqDist_" + mDist.NameModel();}

    template <typename tUk> 
             std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tUk> & aVObs
                  ) const
    {
        return  mDist.PProjToImNorm(aVUk.at(0),aVUk.at(1),aVObs,0);
    }
  private :
    TypeDist                 mDist;
};



template <typename TypeDist>  class cEqIntr
{
  public :
    cEqIntr(const TypeDist & aDist) :
       mDist      (aDist),
       mVNamesObs (Append({"PPx","PPy","PPz"},mDist.VNamesParams()))
    {
    }

    static const std::vector<std::string> VNamesUnknowns() { return {"xPi","yPi"}; }
    const std::vector<std::string> & VNamesObs()  const
    { 
       return mVNamesObs;
    }
     
    std::string FormulaName() const { return "EqIntr_" + mDist.NameModel();}

       /*  Capital letter for 3D variable/formulas and small for 2D */
    template <typename tUk> 
             std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tUk> & aVObs
                  ) const
    {

        // Now compute the distorsion
        auto XY = mDist.PProjToImNorm(aVUk.at(0),aVUk.at(1),aVObs,3);

        return {  aVObs[0] + aVObs[2] * XY[0],
                  aVObs[1] + aVObs[2] * XY[1]    };
     }
  private :
    TypeDist                 mDist;
    std::vector<std::string> mVNamesObs; 
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
#endif // FORMULA_CAMSTENOPE_H
