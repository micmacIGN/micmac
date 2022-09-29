#include "include/V1VII.h"

namespace MMVII
{

/*   ************************************************* */
/*                                                     */
/*         cExportV1StenopeCalInterne                  */
/*                                                     */
/*   ************************************************* */

/*
 CamStenope * Std_Cal_From_File
             (
                 const std::string & aNameFile,
                 const std::string &  aNameTag = "CalibrationInternConique"
             );

*/

cExportV1StenopeCalInterne::cExportV1StenopeCalInterne(const std::string& aFile,int aNbPointPerDim,int aNbLayer)
{
   CamStenope * aCamV1 =  Std_Cal_From_File (aFile);

   //  Do the easy stuff for parameters having obvious  equivalence  V1/V2
   mSzCam = ToI(ToMMVII(aCamV1->SzPixel()));
   mFoc   =  aCamV1->Focale();
   mPP    =  ToMMVII(aCamV1->PP());

   cCalibrationInternConique  aCIC =  StdGetFromPCP(aFile,CalibrationInternConique);

   MMVII_INTERNAL_ASSERT_strong(aCIC.CalibDistortion().size()==1,"Dont handle multiple dist in V1->V2 conv");

   // determine equivalent model of projection, if any
   {
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
   bool  GotEnoughPoint=false;
   while (! GotEnoughPoint)
   {

         mCorresp.Clear();
	 int aNb = 0;
         for (int aKX=0 ; aKX<aNbPointPerDim ; aKX++)
	 {
             double aX = ((aKX+0.5) / aNbPointPerDim) * mSzCam.x();
             for (int aKY=0 ; aKY<aNbPointPerDim ; aKY++)
	     {
                  double aY = ((aKY+0.5) / aNbPointPerDim) * mSzCam.y();
                  Pt2dr aPImV1(aX,aY);

		  if (aCamV1->CaptHasData(aPImV1))
		  {
                      aNb++;
		      for (int aLayer=0 ; aLayer<aNbLayer ; aLayer++)
		      {
                           Pt3dr aPGroundV1 = aCamV1->ImEtProf2Terrain(aPImV1,aLayer+1);
                           mCorresp.AddPair(cPair2D3D(ToMMVII(aPImV1),ToMMVII(aPGroundV1)));
		      }
		  }
	     }
	 }
	 GotEnoughPoint = (aNb >= 500);
         aNbPointPerDim = std::max(aNbPointPerDim+2,int(aNbPointPerDim*1.5));
   }
}

};
