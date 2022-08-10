#ifndef _FORMULA_GEOMED_H_
#define _FORMULA_GEOMED_H_

#include "ComonHeaderSymb.h"

using namespace NS_SymbolicDerivative;


namespace MMVII
{

/**  Class for generating code relative to 2D-distance conservation for triangalution simulation */

class cDist2DConservation
{
  public :
    cDist2DConservation() 
    {
    }

    static const std::vector<std::string> VNamesUnknowns() { return {"x1","y1","x2","y2"}; }
    static const std::vector<std::string> VNamesObs()      { return {"D"}; }

    std::string FormulaName() const { return "Dist2DCons";}

    template <typename tUk> 
             static std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tUk> & aVObs
                  ) // const
    {
          const auto & x1 = aVUk[0];
          const auto & y1 = aVUk[1];
          const auto & x2 = aVUk[2];
          const auto & y2 = aVUk[3];

          const auto & ObsDist  = aVObs[0];  
	  const auto aCst1 = CreateCste(1.0,x1);  // create a symbolic formula for constant 1


          return { sqrt(square(x1-x2) + square(y1-y2))/ObsDist - aCst1 } ;
     }
};

/**  Class for generating code relative to 2D-"RATIO of distance"  */

class cRatioDist2DConservation
{
  public :
    cRatioDist2DConservation() 
    {
    }

    static const std::vector<std::string> VNamesUnknowns() { return {"x1","y1","x2","y2","x3","y3"}; }
    static const std::vector<std::string> VNamesObs()      { return {"D12","D13","D23"}; }

    std::string FormulaName() const { return "RatioDist2DCons";}

    template <typename tUk> 
             std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tUk> & aVObs
                  ) const
    {
          const auto & x1 = aVUk[0];
          const auto & y1 = aVUk[1];
          const auto & x2 = aVUk[2];
          const auto & y2 = aVUk[3];
          const auto & x3 = aVUk[4];
          const auto & y3 = aVUk[5];

          const auto & Obs_d12  = aVObs[0];  
          const auto & Obs_d13  = aVObs[1];  
          const auto & Obs_d23  = aVObs[2];  

          const auto r12 =  sqrt(square(x1-x2) + square(y1-y2)) / Obs_d12 ;
          const auto r13 =  sqrt(square(x1-x3) + square(y1-y3)) / Obs_d13 ;
          const auto r23 =  sqrt(square(x2-x3) + square(y2-y3)) / Obs_d23 ;
          return { r12-r13,r12-r23,r13-r23};
     }
};

/** class for covariance propag   :
     Uknonw  :  Similitude + 4 points of a small network    Tr(x,y)  Sc(x,y)   P0(x,y) ... P3(x,y)

     Observation  :   Linera coeff of 4 pt + cste    :     L0(x,y)  .. L3(x,y)   Cst
 */

class cBaseNetCDPC
{
   public :
       cBaseNetCDPC(const cPt2di  & aSzN) :
           mSzN       (aSzN),
           mNbPts     (mSzN.x() * mSzN.y()),
           mNbCoord   (2*mNbPts)
       {
       }
       static std::vector<std::string>  VectSim (bool WithSim)  
       {
            return WithSim ? std::vector<std::string> {"x_tr","y_tr","x_sc","y_sc"} :  EMPTY_VSTR;
       } 
       static std::vector<std::string>  VectRot (bool WithRot)  
       {
            return WithRot ? std::vector<std::string> {"x_tr","y_tr","teta"} :  EMPTY_VSTR;
       } 
       cPt2di mSzN;
       int    mNbPts;
       int    mNbCoord;
};

class cNetworConsDistProgCov : public  cBaseNetCDPC
{
      public :
          cNetworConsDistProgCov(const cPt2di  & aSzN) :
                cBaseNetCDPC(aSzN)
          {
          }
          std::string FormulaName() const { return "PropCovNwCD_" + ToStr(mNbPts) ;}

          const std::vector<std::string> VNamesUnknowns()  const
          {
               std::vector<std::string>  aRes = VectRot(true);
               for  (int aK=0 ; aK<mNbPts ; aK++)
               {
                   aRes.push_back("x_P" + ToStr(aK));
                   aRes.push_back("y_P" + ToStr(aK));
               }
               return aRes;
          }
          const std::vector<std::string> VNamesObs()       const
          { 
               std::vector<std::string>  aRes;
               for  (int aKVar=0 ; aKVar<mNbPts ; aKVar++)
               {
                    aRes.push_back("xLEq_" +  ToStr(aKVar));
                    aRes.push_back("yLEq_" +  ToStr(aKVar));
               }
               aRes.push_back("Cste");
               return aRes;
          }

          template <typename tUk> 
                     std::vector<tUk> formula
                     (
                          const std::vector<tUk> & aVUk,
                          const std::vector<tUk> & aVObs
                     ) const
           {

                 cPtxd<tUk,2>  aTr(aVUk[0],aVUk[1]);
                 tUk aTeta = aVUk[2];
                 cPtxd<tUk,2>  aSc(cos(aTeta),sin(aTeta));

                 int aIndUk   = 3;

                 int aIndObs  = 0;

                 tUk  aResidual =  CreateCste(0.0,aTeta);  // create a symbolic formula for constant 0

                 for  (int aKVar=0 ; aKVar<mNbPts ; aKVar++)
                 {
                      tUk  aX = aVUk[aIndUk++];
                      tUk  aY = aVUk[aIndUk++];
                      tUk  aLX = aVObs[aIndObs++];
                      tUk  aLY = aVObs[aIndObs++];
                      cPtxd<tUk,2> aPLoc = aTr + aSc*cPtxd<tUk,2>(aX,aY);
                      aResidual = aResidual + aLX * aPLoc.x() + aLY * aPLoc.y();
                 }
                 aResidual = aResidual -  aVObs[aIndObs++];  // substract constant
                 return {aResidual};
           }
          
      public :
};

/** XXXXXX ag*/

class cNetWConsDistSetPts : public  cBaseNetCDPC
{
      public :
          cNetWConsDistSetPts(const cPt2di  & aSzN,bool RotIsUk) :
                cBaseNetCDPC(aSzN),
                mRotIsUk    (RotIsUk)
          {
          }
          std::string FormulaName() const 
          { 
               return "SetPointNwCD_" + std::string(mRotIsUk ? "SimUK" : "SimFix") + ToStr(mNbPts) ;
          }

          const std::vector<std::string> VNamesUnknowns()  const
          {
               // The vector of unknown containt the similitude iff mRotIsUk
               std::vector<std::string> aRes =   VectRot(mRotIsUk) ;
               for  (int aK=0 ; aK<mNbPts ; aK++)
               {
                   aRes.push_back("x_P" + ToStr(aK));
                   aRes.push_back("y_P" + ToStr(aK));
               }
               return aRes;
          }
          const std::vector<std::string> VNamesObs()       const
          { 
               // If similitude is not unknown then it's an observation
               std::vector<std::string> aRes =   VectRot(!mRotIsUk) ;
               for  (int aKVar=0 ; aKVar<mNbPts ; aKVar++)
               {
                    aRes.push_back("xRef_" +  ToStr(aKVar));
                    aRes.push_back("yRef_" +  ToStr(aKVar));
               }
               return aRes;
          }

          template <typename tUk> 
                     std::vector<tUk> formula
                     (
                          const std::vector<tUk> & aVUk,
                          const std::vector<tUk> & aVObs
                     ) const
           {
                 // Rot is OR unkown OR observation
                 const std::vector<tUk> & aVRot =  mRotIsUk ? aVUk : aVObs;
                 cPtxd<tUk,2>  aTr(aVRot[0],aVRot[1]);
                 tUk aTeta = aVRot[2];
                 //  vector of rotation
                 cPtxd<tUk,2>  aSc(cos(aTeta),sin(aTeta));

                 int aIndUk   =  mRotIsUk ? 3 : 0;
                 int aIndObs  =  3 - aIndUk;

                 std::vector<tUk> aVecResidual;

                 for  (int aKVar=0 ; aKVar<mNbPts ; aKVar++)
                 {
                      tUk  aX = aVUk[aIndUk++];
                      tUk  aY = aVUk[aIndUk++];
                      tUk  aRefX = aVObs[aIndObs++];
                      tUk  aRefY = aVObs[aIndObs++];
                      cPtxd<tUk,2> aPLoc = aTr + aSc*cPtxd<tUk,2>(aX,aY);
                      aVecResidual.push_back(aPLoc.x()-aRefX);
                      aVecResidual.push_back(aPLoc.y()-aRefY);
                 }
                 return aVecResidual;
           }
          
      public :
           bool     mRotIsUk; // is the similitude Glob->Loc an unknown or an observation
};


};//  namespace MMVII

#endif // _FORMULA_GEOMED_H_
