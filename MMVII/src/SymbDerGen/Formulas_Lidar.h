#ifndef _FORMULA_LIDAR_H_
#define _FORMULA_LIDAR_H_


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
      Let I be an image , Intr the intrinsic paramater of the camera, P=(R,C) the pose ,
      Q=(x,y,z) a point , q=(i,j) the projection of q in I.  We write :

          -  q  = Intr (tR (Q-C)) 

      As  in this first approach we impose that intrinsiq parameter are fixed, we set  :

                            d Intr
           Intr(Q) = q0 + x ------  + ...   =  q0 + Jac(Intr) * Q = q0 + JI * Q
                             dx

      The value of JI will have been  computed from method  DiffGround2Im

      Identically for the  the radiometry of I, to have the image derivable we will write  :

          I(q) =  I0 +  x (dI/dx) ....

      The value here will have been computed using the method GetValueAndGradInterpol

*/

class cRadiomLidarIma
{
   protected :
     template <typename tUk,typename tObs> 
          tUk Radiom_PerpCentrIntrFix
          (
               const std::vector<tUk> &  aVUk,
	       size_t                 &  aIndUk,
               const std::vector<tObs>&  aVObs,
	       size_t                 &  aIndObs
          ) const 
     {
        // read the unknowns
        cPtxd<tUk,3>  aCCcam   = VtoP3AuoIncr(aVUk,&aIndUk);
        cPtxd<tUk,3>  aW       = VtoP3AuoIncr(aVUk,&aIndUk);
     
        // read the observation
        cPtxd<tUk,3>  aPGround  = VtoP3AuoIncr(aVObs,&aIndObs);  //  PGround
        cMatF<tUk>    aRotInit (3,3,&aIndObs,aVObs);             // Curent value of rotatuin
								
        cPtxd<tUk,3>  aPCamInit = VtoP3AuoIncr(aVObs,&aIndObs);  // Current value of PGround in camera coordinate
        cPtxd<tUk,3>  aGradI  = VtoP3AuoIncr(aVObs,&aIndObs);    // gradient / PCam  of abscisse of projection in image
        cPtxd<tUk,3>  aGradJ  = VtoP3AuoIncr(aVObs,&aIndObs);    // gradient / PCam  of ordonate of projection in image
								
        tUk  aRadiom0 = aVObs.at(aIndObs++);                     // radiometry in image of current proj
        cPtxd<tUk,2>  aGradR = VtoP2AuoIncr(aVObs,&aIndObs);     // gradient of radiometry in image

        // compute the position of the point in camera coordinates
        cPtxd<tUk,3>  aVCP = aPGround - aCCcam;             // vector  CamCenter -> PGround
        cMatF<tUk> aDeltaRot =  cMatF<tUk>::MatAxiator(aW); // unknown "small rotation"
        cPtxd<tUk,3> aPCam =  aDeltaRot * (aRotInit * aVCP); // point in camera coordinate, taking into account the unknowns

        // compute the position of projected point
        cPtxd<tUk,3>  aDPCam = aPCam-aPCamInit;                //  difference Unknown/Curent of point in camera coordinate
        tUk  aDI = PScal(aGradI,aDPCam);                       // variation of abscissa of projection
        tUk  aDJ = PScal(aGradJ,aDPCam);                       // variation of ordinate of projection
                    
        // compute the radiometry
        return aRadiom0 + PScal(aGradR,cPtxd<tUk,2>(aDI,aDJ));
     }

     std::vector<std::string>  NamesPoseUK() const {return Append(NamesP3("mCCam"),NamesP3("mOmega"));}

     std::vector<std::string>  VectObsPPose() const 
     {
          return Append(NamesP3("mPGround"),NamesMatr("mRot0",cPt2di(3,3)));
     }

     std::vector<std::string>  VectObsPCam() const 
     {
          return Append(NamesP3("mPCam0"),NamesP3("mGradPCam_i"),NamesP3("mGradPCam_j")) ;
     }

     std::vector<std::string>  VectObsRadiom() const 
     {
	     return Append({"Rad0"},NamesP2("GradRad"));
     }
};


class cEqLidarImPonct : public cRadiomLidarIma
{
     public :
            template <typename tUk,typename tObs> 
                  std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  )  const
            {
                 // read the unknowns
		 size_t aIndUk = 0;
		 size_t aIndObs = 0;

		 tUk aRadiom = Radiom_PerpCentrIntrFix(aVUk,aIndUk,aVObs,aIndObs);
		 tUk  aRadiomTarget = aVUk.at(aIndUk++);

		 return {aRadiom- aRadiomTarget};
             }

            std::vector<std::string> VNamesUnknowns()  const {return Append(NamesPoseUK(),{"TargetRad"});}
            std::vector<std::string> VNamesObs() const      
            {
		    return Append(VectObsPPose() , VectObsPCam() , VectObsRadiom());
            }
            std::string FormulaName() const { return  "EqLidarImPonct";}

     private :
};


};//  namespace MMVII

#endif // _FORMULA_LIDAR_H_
