#include "V1VII.h"
#include "MMVII_util.h"

namespace MMVII
{

/*   ************************************************* */
/*                                                     */
/*         cExportV1StenopeCalInterne                  */
/*                                                     */
/*   ************************************************* */

/**   generate correspodance  3d/2d for a given camerastenope.

*/
void DoCorresp
     (
        cSet2D3D & aSet,      ///< result where put the correspondance
	CamStenope * aCamV1,  ///< camera-stenope-V1  generating the correspondance
	int aNbPointPerDim,   ///< number of point in each dimension
	int aNbLayer          ///< number of layer
     )
{
   size_t aNbPtsMin = Square(aNbPointPerDim);
   cPt2di aSzCam = ToI(ToMMVII(aCamV1->SzPixel()));

   // generate 2-point on camera,  take into account the fact that some points are invalid (as with fish-eye)
   std::vector<Pt2dr> aVP2;
   while (aVP2.size() < aNbPtsMin)
   {
       aVP2.clear();
       for (int aKX=0 ; aKX<aNbPointPerDim ; aKX++)
       {
           double aX = ((aKX+0.5) / aNbPointPerDim) * aSzCam.x();
           for (int aKY=0 ; aKY<aNbPointPerDim ; aKY++)
           {
               double aY = ((aKY+0.5) / aNbPointPerDim) * aSzCam.y();
               Pt2dr aPImV1(aX,aY);

               if (aCamV1->CaptHasData(aPImV1))
               {
                  aVP2.push_back(aPImV1);
               }
           }
       }
       // if not enough point increase
        double aMul = sqrt(   (aNbPtsMin*1.1)  /   std::max(double(aVP2.size()),1.0) );
       aNbPointPerDim = std::max(aNbPointPerDim+2,int(aNbPointPerDim*aMul));
   }

   // now generate the points 
   aSet.Clear();
   for (int aLayer=0 ; aLayer<aNbLayer ; aLayer++)
   {
       for (const auto & aPImV1 : aVP2)
       {
           Pt3dr aPGroundV1 = aCamV1->ImEtProf2Terrain(aPImV1,aLayer+1);
           aSet.AddPair(cPair2D3D(ToMMVII(aPImV1),ToMMVII(aPGroundV1)));
       }
   }
}

cExportV1StenopeCalInterne::cExportV1StenopeCalInterne(bool isForCalib,const std::string& aFile,int aNbPointPerDim,int aNbLayer) :
   mPose (cIsometry3D<tREAL8>::Identity())
{
	// do we read only a calibration or calib+pose
   CamStenope * aCamV1 =  isForCalib                            ?
	                     Std_Cal_From_File (aFile)          :
	                     BasicCamOrientGenFromFile (aFile)  ;

   //  Do the easy stuff for parameters having obvious  equivalence  V1/V2
   mSzCam = ToI(ToMMVII(aCamV1->SzPixel()));
   mFoc   =  aCamV1->Focale();
   mPP    =  ToMMVII(aCamV1->PP());


   // determine equivalent model of projection, if any
   if (isForCalib)
   {
       cCalibrationInternConique  aCIC =  StdGetFromPCP(aFile,CalibrationInternConique);
       MMVII_INTERNAL_ASSERT_strong(aCIC.CalibDistortion().size()==1,"Dont handle multiple dist in V1->V2 conv");
       auto aCal =  aCIC.CalibDistortion().at(0);

       //  standard model => stenope
       if (aCal.ModRad().IsInit() || aCal.ModPhgrStd().IsInit() || aCal.ModNoDist().IsInit())
       {
           eProj = eProjPC::eStenope;
       }
       else if (aCal.ModUnif().IsInit())
       {
           eModelesCalibUnif  aTypeV1 =  aCal.ModUnif().Val().TypeModele();

           int iV1 = int(aTypeV1);

	   // three fish eye
           if (aTypeV1==eModele_FishEye_10_5_5)   
              eProj = eProjPC::eFE_EquiDist;
           else if (aTypeV1==eModele_EquiSolid_FishEye_10_5_5)
              eProj = eProjPC::eFE_EquiSolid;
           else if (aTypeV1==eModele_Stereographik_FishEye_10_5_5)
              eProj = eProjPC::eStereroGraphik;
           else if // maybe no exact corresp for distorsion, but stenope
            (
                         ( (iV1>=(int) eModeleEbner) && (iV1<=(int) eModelePolyDeg7))
		    ||   ((aTypeV1>= (int) eModele_DRad_PPaEqPPs) && (iV1<=int(eModelePolyDeg1)))
	    )
            {
                 eProj = eProjPC::eStenope;
	    }
            else 
            {
		MMVII_INTERNAL_ERROR("Impossible internal calibration conversion from MMV1");
            }
       }
       else
       {
             MMVII_INTERNAL_ERROR("Impossible internal calibration conversion from MMV1");
       }
   }
   else
   {
	   // read the name of calibration
       cOrientationConique aOriConique=StdGetFromPCP(aFile,OrientationConique);
       MMVII_INTERNAL_ASSERT_strong(aOriConique.FileInterne().IsInit(),"File interne absent");
       mNameCalib = aOriConique.FileInterne().Val();

       // In MMV1  Orient  M->C,  in MMV2 Pose C->M , so reciproc image are used
       ElRotation3D  aOriC2M = aCamV1->Orient();
       cPt3dr aC = ToMMVII(aOriC2M.ImRecAff(Pt3dr(0,0,0)));
       cPt3dr aI = ToMMVII(aOriC2M.IRecVect(Pt3dr(1,0,0)));
       cPt3dr aJ = ToMMVII(aOriC2M.IRecVect(Pt3dr(0,1,0)));
       cPt3dr aK = ToMMVII(aOriC2M.IRecVect(Pt3dr(0,0,1)));

       mPose = cIsometry3D<tREAL8>(aC,cRotation3D<tREAL8>(MatFromCols(aI,aJ,aK),false));
   }


   if ((aNbLayer>0) && (aNbPointPerDim>0))
      DoCorresp(mCorresp,aCamV1,aNbPointPerDim,aNbLayer);
}

};
