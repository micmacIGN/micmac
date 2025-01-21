#ifndef _FORMULA_LIDAR_H_
#define _FORMULA_LIDAR_H_


#include "SymbDer/SymbDer_Common.h"
#include "SymbDer/SymbDer_MACRO.h"
#include "MMVII_Ptxd.h"
#include "MMVII_Stringifier.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_PhgrDist.h"

#include "ComonHeaderSymb.h"



using namespace NS_SymbolicDerivative;
namespace NS_SymbolicDerivative
{
MACRO_SD_DEFINE_STD_BINARY_FUNC_OP_DERIVABLE(MMVII,NormalisedRatioPos,Der_NormalisedRatio_I1Pos,Der_NormalisedRatio_I2Pos)
};


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
                 // to complete

        // read the unknowns
        cPtxd<tUk,3>  aCCam   = VtoP3AuoIncr(aVUk,&aIndUk);  // camera center
        cPtxd<tUk,3>  aW      = VtoP3AuoIncr(aVUk,&aIndUk);  // camera infinitesimal rotation

        // read the observation
        cMatF<tObs>    aRotInit (3,3,&aIndObs,aVObs);       // Curent value of rotation
        cPtxd<tObs,3>  aPGround   = VtoP3AuoIncr(aVObs,&aIndObs);  // Value of 3D ground point 
        cPtxd<tObs,3>  aPCamInit   = VtoP3AuoIncr(aVObs,&aIndObs);  // Current value of 3D point in camera system
        cPtxd<tObs,3>  aGradProjI  = VtoP3AuoIncr(aVObs,&aIndObs);  // I(abscissa) of gradient / PCamera of projection 
        cPtxd<tObs,3>  aGradProjJ  = VtoP3AuoIncr(aVObs,&aIndObs);  // J(ordinate) of gradient / PCamera of projection

        tUk aRadiomInit  = aVObs.at(aIndObs++); // extract the radiometry of image
        cPtxd<tObs,2>  aGradIm  = VtoP2AuoIncr(aVObs,&aIndObs);  // extract the gradient of image

        // compute the position of the point in camera coordinates
        cPtxd<tUk,3> aVCP = aPGround - aCCam;  // "vector"  Center -> PGround
        cMatF<tUk>   aDeltaRot =  cMatF<tUk>::MatAxiator(aW); // small rotation associated to W
        cPtxd<tUk,3> aPCoordCam = aDeltaRot * aRotInit * aVCP;

        //                                       d Intr 
        // Intr(Pose(Pground)) =  Intr(PCam0)  + ------- * (Pose(Pground) - PCam0)
        //                                       dcam
        //                                          
        cPtxd<tUk,3> aDeltaPCam = aPCoordCam-aPCamInit;  // difference Unknown point in cam coord, vs its current value
        tUk aDelta_I = PScal(aDeltaPCam,aGradProjI); // scalar product gradient with diff
        tUk aDelta_J = PScal(aDeltaPCam,aGradProjJ); // scalar product gradient with diff
                    
        // compute the radiometry
        return  aRadiomInit + PScal(aGradIm,cPtxd<tObs,2>(aDelta_I,aDelta_J));
     }

     static std::vector<std::string>  NamesPoseUK()  {return Append(NamesP3("mCCam"),NamesP3("mOmega"));}

     static std::vector<std::string>  VectObsPPose()  
     {
          return NamesMatr("mRot0",cPt2di(3,3));
     }

     static std::vector<std::string>  VectObsPCam() 
     {
          return Append
                 (
                      NamesP3("mPGround"),  // Ground point 
                      NamesP3("mPCam0"),    // initial current value of PGround in camera system
                      NamesP3("mGradPCam_i"),   //  (d PIm / d PCam ).i
                      NamesP3("mGradPCam_j")    //  (d PIm / d PCam) .j
                 ) ;
     }

     static std::vector<std::string>  VectObsRadiom()  
     {
             // Radiom + grad /i,j
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

		tUk aRadiomTarget = aVUk.at(aIndUk++);
		tUk aRadiom = Radiom_PerpCentrIntrFix(aVUk,aIndUk,aVObs,aIndObs);

		 return {aRadiom - aRadiomTarget};
             }

            std::vector<std::string> VNamesUnknowns()  const {return Append({"TargetRad"},NamesPoseUK());}
            static std::vector<std::string> VNamesObs() 
            {
		    return Append(VectObsPPose() , VectObsPCam() , VectObsRadiom());
            }
            std::string FormulaName() const { return  "EqLidarImPonct";}

     private :
};

/*
     ak ri  +bk =     Rj,k
     Som(ri^2) = 1
     Som(ri) = 0

      C(r0,rj)  = C(R0,Rj)
*/
    
class cEqLidarImCensus : public cRadiomLidarIma
{
     public :
            template <typename tUk,typename tObs> 
                  std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  )  const
            {
                 // to complete

		 return {aVUk.at(0)};
            }

            std::vector<std::string> VNamesUnknowns()  const {return Append({"TargetRatio"},NamesPoseUK());}
            std::vector<std::string> VNamesObs() const      
            {
		    std::vector<std::string>  aV0 = cEqLidarImPonct::VNamesObs();
		    return Append(AddPostFix(aV0,"_0"),AddPostFix(aV0,"_1"));
            }
            std::string FormulaName() const { return  "EqLidarImCensus";}

     private :
};

class cEqLidarImCorrel : public cRadiomLidarIma
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

		tUk aRadiomTarget = aVUk.at(aIndUk++);
		tUk aCoefMul      = aVUk.at(aIndUk++);
		tUk aCoefAdd      = aVUk.at(aIndUk++);
		tUk aRadiom = Radiom_PerpCentrIntrFix(aVUk,aIndUk,aVObs,aIndObs);

		 return {aCoefAdd + aCoefMul * aRadiom - aRadiomTarget};
             }

            std::vector<std::string> VNamesUnknowns()  const {return Append({"TargetRad","CoefMul","CoefAdd"},NamesPoseUK());}
            static std::vector<std::string> VNamesObs() 
            {
		    return Append(VectObsPPose() , VectObsPCam() , VectObsRadiom());
            }
            std::string FormulaName() const { return  "EqLidarImCorrel";}

     private :
};



};//  namespace MMVII

#endif // _FORMULA_LIDAR_H_
