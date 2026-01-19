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
      Q=(x,y,z) a point , q=(i,j) the projection of q in I.  We write:

          -  q  = Intr (tR (Q-C)) 

      As  in this first approach we impose that intrinsiq parameter are fixed, we set:

                            d Intr
           Intr(Q) = q0 + x ------  + ...   =  q0 + Jac(Intr) * Q = q0 + JI * Q
                             dx

      The value of JI will have been  computed from method  DiffGround2Im

      Identically for the  the radiometry of I, to have the image derivable we will write:

          I(q) =  I0 +  x (dI/dx) ....

      The value here will have been computed using the method GetValueAndGradInterpol

*/

class cRadiomLidarIma
{
   public:
    cRadiomLidarIma(bool aIsPoseScanUk) : mIsPoseScanUk(aIsPoseScanUk) {}
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
        // uk only if aUkScanPose
        cPtxd<tUk,3>  aCScan;  // scan center
        cPtxd<tUk,3>  aWScan;  // scan infinitesimal rotation
        if (mIsPoseScanUk)
        {
            aCScan   = VtoP3AutoIncr(aVUk,&aIndUk);
            aWScan   = VtoP3AutoIncr(aVUk,&aIndUk);
        }

        cPtxd<tUk,3>  aCCam   = VtoP3AutoIncr(aVUk,&aIndUk);  // camera center
        cPtxd<tUk,3>  aWCam      = VtoP3AutoIncr(aVUk,&aIndUk);  // camera infinitesimal rotation

        // read the observation
        cMatF<tObs>    aRotScanInit; // dummy value, matrix not used if no scan pose
        if (mIsPoseScanUk)
        {
            aRotScanInit = cMatF<tObs>(3,3,&aIndObs,aVObs);         // Curent value of scan rotation
        }

        cMatF<tObs>    aRotCamInitTr (3,3,&aIndObs,aVObs);           // Curent value of camera rotation, transposed
        cPtxd<tObs,3>  aPScan      = VtoP3AutoIncr(aVObs,&aIndObs);  // Value of 3D point
        cPtxd<tObs,3>  aPCamInit   = VtoP3AutoIncr(aVObs,&aIndObs);  // Current value of 3D point in camera system
        cPtxd<tObs,3>  aGradProjI  = VtoP3AutoIncr(aVObs,&aIndObs);  // I(abscissa) of gradient / PCamera of projection
        cPtxd<tObs,3>  aGradProjJ  = VtoP3AutoIncr(aVObs,&aIndObs);  // J(ordinate) of gradient / PCamera of projection

        tUk aRadiomInit  = aVObs.at(aIndObs++); // extract the radiometry of image
        cPtxd<tObs,2>  aGradIm  = VtoP2AutoIncr(aVObs,&aIndObs);  // extract the gradient of image

        // compute the position of the point in camera coordinates
        cPtxd<tUk,3> aPGround = aPScan; // when no scan pose uk
        if (mIsPoseScanUk)
        {
            cMatF<tUk>   aDeltaScanRot =  cMatF<tUk>::MatAxiator(-aWScan); // transpose small rotation associated to W
            aPGround = aRotScanInit * aDeltaScanRot * aPScan + aCScan;
        }
        cPtxd<tUk,3> aVCP = aPGround - aCCam;  // "vector"  Center -> PGround
        cMatF<tUk>   aDeltaCamRot =  cMatF<tUk>::MatAxiator(aWCam); // small rotation associated to W
        cPtxd<tUk,3> aPCoordCam = aDeltaCamRot * aRotCamInitTr * aVCP;

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

     std::vector<std::string>  NamesPoseUK() const {
        if (mIsPoseScanUk)
            return Append(NamesP3("mCScan"),NamesP3("mOmegaScan"), NamesP3("mCCam"),NamesP3("mOmegaCam"));
        else
            return Append(NamesP3("mCCam"),NamesP3("mOmegaCam"));
     }

     std::vector<std::string>  VectObsPPose() const
     {
        if (mIsPoseScanUk)
            return Append(NamesMatr("mRot0Scan",cPt2di(3,3)), NamesMatr("mRot0CamTr",cPt2di(3,3)));
        else
            return NamesMatr("mRot0CamTr",cPt2di(3,3));
     }

     static std::vector<std::string>  VectObsPCam() 
     {
          return Append
                 (
                      NamesP3("mPScan"),  // scan 3D point
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

     bool mIsPoseScanUk;
};

class cEqLidarImPonct : public cRadiomLidarIma
{
     public :
        cEqLidarImPonct(bool aIsPoseScanUk) : cRadiomLidarIma(aIsPoseScanUk) {}
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
            std::vector<std::string> VNamesObs() const
            {
                return Append(VectObsPPose() , VectObsPCam() , VectObsRadiom());
            }
            std::string FormulaName() const {
                return  mIsPoseScanUk?"EqLidarImPonctPose":"EqLidarImPonct";
            }

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
        cEqLidarImCensus(bool aIsPoseScanUk) : cRadiomLidarIma(aIsPoseScanUk) {}
            template <typename tUk,typename tObs> 
                  std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  )  const
            {
                size_t aIndUk = 0;
                size_t aIndObs = 0;

                // target unknown ratio between central pixel and neighbouring
                tUk aTargetRatio = aVUk.at(aIndUk++);
                // radiometry of central pixel
                tUk aRadiom0 = Radiom_PerpCentrIntrFix(aVUk,aIndUk,aVObs,aIndObs);
                // !! Carefull !! : aIndUk has been incremented, reset it 
                aIndUk=1;
                // radiometry of neighbour pixel
                tUk aRadiom1 = Radiom_PerpCentrIntrFix(aVUk,aIndUk,aVObs,aIndObs);

                return {NormalisedRatioPos(aRadiom0,aRadiom1) - aTargetRatio};
            }

            std::vector<std::string> VNamesUnknowns()  const {return Append({"TargetRatio"},NamesPoseUK());}
            std::vector<std::string> VNamesObs() const
            {
                std::vector<std::string>  aV0 = Append(VectObsPPose() , VectObsPCam() , VectObsRadiom());
                // we duplicate the observation for 2 pixels of the pair (central / periph)
                return Append(AddPostFix(aV0,"_0"),AddPostFix(aV0,"_1"));
            }
            std::string FormulaName() const {
                return  mIsPoseScanUk?"EqLidarImCensusPose":"EqLidarImCensus";
            }

     private :
};

class cEqLidarImCorrel : public cRadiomLidarIma
{
     public :
        cEqLidarImCorrel(bool aIsPoseScanUk) : cRadiomLidarIma(aIsPoseScanUk) {}
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
            std::vector<std::string> VNamesObs() const
            {
                return Append(VectObsPPose() , VectObsPCam() , VectObsRadiom());
            }
            std::string FormulaName() const {
                return  mIsPoseScanUk?"EqLidarImCorrelPose":"EqLidarImCorrel";
            }

     private :
};


/*
 * scan to scan adjustment:
 * frome scan A patch, in Camera3D coords => PoseA to get world coords
 * => InvPoseB to get scan B Camera3D coords => Proj to get scan B raster pixel => get distanceB
 * error = distanceB - distance scanB to ground point
 *
 * */
class cEqLidarLidar
{
public :
    template <typename tUk,typename tObs>
    std::vector<tUk> formula
        (
            const std::vector<tUk> & aVUk,
            const std::vector<tObs> & aVObs
            )  const
    {
        size_t aIndUk = 0;
        size_t aIndObs = 0;
        // read the unknowns
        cPtxd<tUk,3>  aCScanA   = VtoP3AutoIncr(aVUk,&aIndUk);
        cPtxd<tUk,3>  aWScanA   = VtoP3AutoIncr(aVUk,&aIndUk);
        cPtxd<tUk,3>  aCScanB   = VtoP3AutoIncr(aVUk,&aIndUk);
        cPtxd<tUk,3>  aWScanB   = VtoP3AutoIncr(aVUk,&aIndUk);

        // read the observations
        cMatF<tObs> aRotScanAInit   = cMatF<tObs>(3,3,&aIndObs,aVObs);         // Curent value of scan A rotation
        cMatF<tObs> aRotScanBTrInit = cMatF<tObs>(3,3,&aIndObs,aVObs);         // Curent value of scan B rotation, transposed

        cPtxd<tObs,3>  aPScanA     = VtoP3AutoIncr(aVObs,&aIndObs);  // Value of point in scan A frame
        cPtxd<tObs,3>  aPScanB0    = VtoP3AutoIncr(aVObs,&aIndObs);  // Initial value of point in scan B frame
        tObs           aDistB0     = aVObs.at(aIndObs++);            // Current distance read in raster B
        cPtxd<tObs,3>  aGradProjBI = VtoP3AutoIncr(aVObs,&aIndObs);  // I(abscissa) of gradient / PCamera of projection
        cPtxd<tObs,3>  aGradProjBJ = VtoP3AutoIncr(aVObs,&aIndObs);  // J(ordinate) of gradient / PCamera of projection
        cPtxd<tObs,2>  aGradDistB  = VtoP2AutoIncr(aVObs,&aIndObs);  // extract the gradient of distance raster

        cMatF<tUk>   aDeltaScanARot =  cMatF<tUk>::MatAxiator(-aWScanA); // transpose small rotation associated to W
        cPtxd<tUk,3> aPGround = aRotScanAInit * aDeltaScanARot * aPScanA + aCScanA;

        tUk aDistGround = Norm2(aPGround - aCScanB);

        cMatF<tUk>   aDeltaScanBRot =  cMatF<tUk>::MatAxiator(aWScanB); // small rotation associated to W
        cPtxd<tUk,3> aPCoordB = aDeltaScanBRot * aRotScanBTrInit * (aPGround - aCScanB);

        //                                       d Intr
        // Intr(Pose(Pground)) =  Intr(PCam0)  + ------- * (Pose(Pground) - PCam0)
        //                                       dcam
        //
        cPtxd<tUk,3> aDeltaPScanB = aPCoordB-aPScanB0;  // difference Unknown point in scan B coord, vs its current value
        tUk aDelta_I = PScal(aDeltaPScanB,aGradProjBI); // scalar product gradient with diff
        tUk aDelta_J = PScal(aDeltaPScanB,aGradProjBJ); // scalar product gradient with diff

        // compute the distance
        tUk aDistB = aDistB0 + PScal(aGradDistB,cPtxd<tObs,2>(aDelta_I,aDelta_J));
        return  { aDistGround - aDistB };
    }

    std::vector<std::string> VNamesUnknowns()  const
    {
        return Append(NamesP3("OmegaScanA"), NamesP3("OmegaScanB"));
    }
    std::vector<std::string> VNamesObs() const
    {
        return Append(Append(NamesMatr("RotScanA",cPt2di(3,3)),
                             NamesMatr("RotScanBTr",cPt2di(3,3)),
                             NamesP3("PScanA"),
                             NamesP3("PScanB0")),
                      Append({"DistB0"},
                             NamesP3("GradProjBI"),   //  (d PIm / d PCam ).i
                             NamesP3("GradProjBJ"),   //  (d PIm / d PCam) .j
                             NamesP2("GradDistB")));
    }
    std::string FormulaName() const {
        return  "EqLidarLidar";
    }

private :
};





};//  namespace MMVII

#endif // _FORMULA_LIDAR_H_
