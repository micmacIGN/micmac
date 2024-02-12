#ifndef _FORMULA_SENSOR_GEN_H_
#define _FORMULA_SENSOR_GEN_H_

// PUSHB


#include "SymbDer/SymbDer_Common.h"
#include "MMVII_Ptxd.h"
#include "MMVII_Stringifier.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_PhgrDist.h"

#include "ComonHeaderSymb.h"

using namespace NS_SymbolicDerivative;


namespace MMVII
{
/* 
GLOBAL SCOPE and HYPOTHESIS, LIMITATION :

 Classes used for defining bundle adjusment of  "generic" model of sensor, i.e a model for wich we dont
have access to the physicall model that would allow a rigourous implementation. We suppose we only have access to :

    - the projection function  Pi(x,y,z) =>  i,j
    - the derivative of Pi  :   d Pi/dx ... , that can be computed by anyway (handcraft, formal, finit diff) does matter

This function are regarded has black box, and at the level of this classes, what will be communicated is only values of these
functions in some points.  Also in this file we make the hypthesis that the correction between initial model and adjusted model
is just a  2D deformations "D2"  so that :

         Pi' =   D2 o Pi

More precisely we assume that D2 is a polynomial a total degree N

    D2(x,y) = Sum_{i+j<=N}  a_ij x^i y^j + b_ij x^i y^j

The limitation 

*/

/** "Helper" class for generating the distorsion, it can be used in 3 contexts :

      - generate a calculator for computing the distorsion itself
      - generate a caculator for computing the colinearity equation
      - generate the base of functions  used to compute by least square such function from a set of samples,
        i.e given  pair of points {p_k,q_k}, compute the  D2  such that D2(p_k) = q_k, it is used typicall for
        computing an invers mapping

     The fact of using this helper class, assure the coherence in convention in the 3 context.  

     For example with degree "1" :

      in mode standard(no base) "Func"  we will return :

           { a_00+a_10*x+a_01*y , b_00+b_10*x+b_01*y} 

      while in mode Base "Funcs" we will return :

           {  a_00 , 0    ,  a_10*x ,  0      ,   a_01*y ,  0
              0    , b_00 ,  0      ,  b_10*x ,   0      ,  b_01*y}

     The class construct an explict representation of base of polynom "mVDesc", which is more descriptive.
*/

class cDistPolyn2D
{
   public :

      /// constructor : memorise parameter and build the explicit representation of base functions (nomoms)
      cDistPolyn2D(int aDegree,bool ForBase,bool InitDesc=true) :
         mDegree      (aDegree),
         mForBase     (ForBase)
      {
          if (InitDesc)
          {
             for (int aDegX=0 ; aDegX<= mDegree; aDegX++)
	     {
                 for (int aDegY=0 ; (aDegX+aDegY) <= mDegree; aDegY++)
	         {
                     //  Add    Dx = X^i Y^j
		     mVDesc.push_back(cDescOneFuncDist(eTypeFuncDist::eMonX,cPt2di(aDegX,aDegY),eModeDistMonom::eModeStd));
                     //  Add    Dy = X^i Y^j
		     mVDesc.push_back(cDescOneFuncDist(eTypeFuncDist::eMonY,cPt2di(aDegX,aDegY),eModeDistMonom::eModeStd));
	         }
	     }
	  }
      }
      // create an identifier that will be used for naming code generation
      std::string  NamesDist() const
      {
            return std::string(mForBase?"Dist":"Base") +  "_Polyn2D_Degree_" +ToStr(mDegree) ;
      }
      //  std::string  NamesDist() const {return NamesDist(mDegree,mForBase);}


      // vector of name that can be used as unknown or as observation
      std::vector<std::string> VNamesParam()  const
      { 
           std::vector<std::string> aRes ;
           for (const auto & aDesc : mVDesc) 
	   {
               aRes.push_back(aDesc.mName);
	   }
           return aRes;
      }

      // compute the set of functions
      template<typename tScal> std::vector<tScal>
                Funcs
                (
                     const tScal & xIn,const tScal & yIn,
                     const std::vector<tScal> &  aVParam,
                     unsigned int              aK0P
                ) const
       {
          tScal  aFunc0 = CreateCste(0.0,xIn);
          tScal  aSumX =  aFunc0;
          tScal  aSumY =  aFunc0;
          std::vector<tScal>  aVBaseX;
          std::vector<tScal>  aVBaseY;

          for (size_t aKDesc=0 ; aKDesc<mVDesc.size() ; aKDesc++)
          {
              const cDescOneFuncDist & aDesc = mVDesc[aKDesc];
              tScal aMon = aVParam[aK0P+aKDesc] * powI(xIn,aDesc.mDegMon.x()) * powI(yIn,aDesc.mDegMon.y()) ;

              if (aDesc.mType == eTypeFuncDist::eMonX)
              {
                   aSumX = aSumX + aMon;
                   aVBaseX.push_back(aMon);
                   aVBaseY.push_back(aFunc0);
              }
              else if (aDesc.mType == eTypeFuncDist::eMonY)
              {
                   aSumY = aSumY + aMon;
                   aVBaseX.push_back(aFunc0);
                   aVBaseY.push_back(aMon);
              }
              else
              {
                   MMVII_INTERNAL_ERROR("cEqColinSensGenPolyn2D  : should not be here");
              }
          }

          if (mForBase)
          {
              return Append(aVBaseX,aVBaseY);
          }
          return  {aSumX,aSumY};

       }


      int                             mDegree;
      std::vector<cDescOneFuncDist>   mVDesc;
      bool                            mForBase;
};


/**  Class to generate the colinearity equation of a generic sensor with 2D polynomial conversion
*/

class cEqColinSensGenPolyn2D
{
  public :
    cEqColinSensGenPolyn2D(int aDegree,bool InitDesc=true) :
       mDistPol2D (aDegree,false,InitDesc)
    {
    }

    //  The unknown are constitued of coordinate of ground -point (which is the base of bundle adjustment theory) and
    //  the parameter of sensor which, here, is made of 2D corrective polynomial
    std::vector<std::string> VNamesUnknowns()  const
    { 
         std::vector<std::string> aRes {"XGround","YGround","ZGround"};
         return Append(aRes,mDistPol2D.VNamesParam());
    }

    //    As the class know nothing on the sensor , all the information must be passed for a given point
    //  as data (observation/context) describing the "tangent" application,  more precicsely we must give :
    // 
    //      * the value of grounf point "P0x, P0y, P0z"
    //      * the gradient of projection for I    "dIdX,dIDY,dIdZ"  and J "dJdX,dJDY,dJdZ"
    //      * the projection of ground point "IP0,JP0"
    // 
    //    Also the vector of observation contains as usual the measure of the point in image "IObs","JObs"

    std::vector<std::string> VNamesObs() const      
    { 
                // 0               2              4                    7                       10
         return {"IObs","JObs",   "IP0","JP0",    "P0x","P0y","P0z",   "dIdX","dIDY","dIdZ",   "dJdX","dJDY","dJdZ"};
    }

    std::string FormulaName() const { return  mDistPol2D.NamesDist() + "_EqColin";}

    template <typename tUk,typename tObs> 
             std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  )  const
    {
          //  The 3D point is an unknown
          cPtxd<tUk,3>  aPUk = VtoP3(aVUk,0);

          // we compute the formal expression that gives the projection taking into account the differential of
          // projection that is given as obsevations 

          cPtxd<tObs,2>  aPixObs = VtoP2(aVObs,0);  // extract the measurement (like tie point)

          cPtxd<tObs,2>   aPix0   = VtoP2(aVObs,2);  // extract projection of estimation
          cPtxd<tObs,3>   aP0  = VtoP3(aVObs,4);     //  Estimation of point, where the linearisation was made
          cPtxd<tUk,3>  aDP = aPUk-aP0;              //  differnce between unknonw and estimatio,
          cPtxd<tObs,3>   aGradI = VtoP3(aVObs,7);   //  extract gradient of "I"
          cPtxd<tObs,3>   aGradJ = VtoP3(aVObs,10);  //  extract gradient of "J"
          tUk  aDI =  PScal(aDP,aGradI);             //   formula giving I difference between unkonwn & estimatation
          tUk  aDJ =  PScal(aDP,aGradJ);             //   formula giving J difference between unkonwn & estimatation

          cPtxd<tUk,2> aPix = aPix0 + cPtxd<tUk,2>(aDI,aDJ);  // formula giving  the  exact projection of unknown point

          cPtxd<tUk,2>   aDist =   VtoP2(mDistPol2D.Funcs(aPix.x(),aPix.y(),aVUk,3)) ;


          cPtxd<tUk,2>  aResidual = aPix +  aDist -  aPixObs;

          return ToVect(aResidual);
     }

  private :
     cDistPolyn2D                    mDistPol2D;
};

class cEqDistPolyn2D
{
  public :
    cEqDistPolyn2D(int aDegree,bool InitDesc=true) :
       mDistPol2D (aDegree,false,InitDesc)
    {
    }
    std::vector<std::string>  VNamesUnknowns()  const
    {
            return {"xPix","yPix"};
    }
    const std::vector<std::string>    VNamesObs() const {return mDistPol2D.VNamesParam();}
     
    std::string FormulaName() const { return  mDistPol2D.NamesDist() + "_EqDist";}

    template <typename tUk> 
             std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tUk> & aVObs
                  ) const
    {
         auto  aDist =   mDistPol2D.Funcs(aVUk.at(0),aVUk.at(1),aVObs,0);

         return  {  aVUk.at(0)+aDist.at(0)  ,  aVUk.at(1)+aDist.at(1) } ;
    }
  private :
     cDistPolyn2D                    mDistPol2D;
};



};//  namespace MMVII

#endif // _FORMULA_SENSOR_GEN_H_
