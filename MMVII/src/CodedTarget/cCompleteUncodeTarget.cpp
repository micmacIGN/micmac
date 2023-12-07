#include "MMVII_Geom2D.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Sensor.h"
#include "MMVII_PCSens.h"
#include "CodedTarget.h"
#include "CodedTarget_Tpl.h"
#include "MMVII_2Include_Serial_Tpl.h"

/*   Modularistion
 *   Code extern tel que ellipse
 *   Ellipse => avec centre
 *   Pas de continue
 */


namespace MMVII
{

	/*
struct A 
{
};
struct B
{
	A mA;
};

void f ()
{
	B aB;
	A aA = aB;
}
*/


/*  *********************************************************** */
/*                                                              */
/*           cAppliCompletUncodedTarget                         */
/*                                                              */
/*  *********************************************************** */

class cAppliCompletUncodedTarget : public cMMVII_Appli
{
     public :

        cAppliCompletUncodedTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

        void ShouldNotCompile()
        {
           ///  WHY DOES THIS COMPILE ???
           if (0)
           {
              // cPhotogrammetricProject &  aPPPP = *this;
              // cPhotogrammetricProject * aPPPP = this;
              cPhotogrammetricProject  aPPPP = * this;
              FakeUseIt(aPPPP);
           }
        }

     private :
        int Exe() override;


	void CompleteAll();
	void CompleteOneGCP(const cMes1GCP & aGCP);

        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;



	cPhotogrammetricProject        mPhProj;
        std::string                    mSpecImIn;

	tREAL8                         mThresholdDist;

	std::string                    mPatternNormal;
        cPt3dr                         mNormal;
	std::vector<tREAL8>            mThreshRay;  // Ratio Max Min

        std::string                    mNameIm;
        cSensorImage *                 mSensor;
        cSensorCamPC *                 mCamPC;
        cSetMesImGCP                   mMesImGCP;
        cSetMesPtOf1Im                 mImageM;
        std::vector<cSaveExtrEllipe>   mVSEE;
	std::string                    mNameReportEllipse;
};

cAppliCompletUncodedTarget::cAppliCompletUncodedTarget
(
    const std::vector<std::string> & aVArgs,
    const cSpecMMVII_Appli & aSpec
) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this),
   mNormal       (0,0,1)
   // mThreshRay    {1.03,4.85,5.05}
{
}

        // cExtract_BW_Target * 
cCollecSpecArg2007 & cAppliCompletUncodedTarget::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return
            anArgObl
         << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
	 << mPhProj.DPOrient().ArgDirInMand()
         << Arg2007(mThresholdDist,"Threshold on distance for in pixel")

   ;
}

cCollecSpecArg2007 & cAppliCompletUncodedTarget::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return 
                  anArgOpt
	     <<   mPhProj.DPPointsMeasures().ArgDirInputOptWithDef("Std")
	     <<   mPhProj.DPPointsMeasures().ArgDirOutOptWithDef("Completed")
             <<   AOpt2007(mThreshRay,"ThRay","Threshold for ray [RatioMax,RMin,RMax]",{{eTA2007::ISizeV,"[3,3]"}})
             <<   AOpt2007(mPatternNormal,"PatNorm","If estimate normal, pattern for point involved")
          ;
}

void cAppliCompletUncodedTarget::CompleteOneGCP(const cMes1GCP & aGCP)
{
    // if has already been selected, nothing to do
    if (mImageM.NameHasMeasure(aGCP.mNamePt))
    {
       return;
    }

    // if 3D point not visible , reject
    if (! mSensor->IsVisible(aGCP.mPt))
       return;

    cPt2dr aProjIm = mSensor->Ground2Image(aGCP.mPt);
    cMesIm1Pt * aMes = mImageM.NearestMeasure(aProjIm);

    // if projection too far reject
    if (Norm2(aProjIm-aMes->mPt)>mThresholdDist)
       return;

    // get the ellipse that has the same (temporary) code than the point
    auto  anIt_SEE = find_if(mVSEE.begin(),mVSEE.end(),[aMes](const auto& aM){return aM.mNameCode==aMes->mNamePt;});
    if (anIt_SEE==mVSEE.end())   // should not happen
    {
       MMVII_INTERNAL_ERROR("Could not find ellipse");
       return;
    }

    // Now test shape of ellispe compared to theoreticall ground pose
 
    cPlane3D aPlaneT  = cPlane3D::FromPtAndNormal(aGCP.mPt,mNormal);     // 3D plane of the ellispe
    cEllipse aEl = mSensor->EllipseIm2Plane(aPlaneT,anIt_SEE->mEllipse,50);  // ellipse in ground coordinate

    tREAL8 aL1 = aEl.LSa();   // gread axe
    tREAL8 aL2 = aEl.LGa();   // small axe
			      
    tREAL8 aRatio = aL2/aL1;  // ratio (should be  equal to 1)
    tREAL8 aRMoy = std::sqrt(aL1*aL2);  // ray, to compare to theoretical (for ex 5 mm for3D AICON)

   AddOneReportCSV(mNameReportEllipse,{mNameIm,aGCP.mNamePt,ToStr(aRMoy),ToStr(aRatio)});


    if (  IsInit(&mThreshRay) &&
	  (
                 (aRatio > mThreshRay[0])
             ||  (aRMoy  < mThreshRay[1])
             ||  (aRMoy  > mThreshRay[2])
	  )
       )
    {
       return;
    }

    if (false && (LevelCall()==0))  // print info if was done whith only one image
    {
        StdOut() << "NNN=" << aGCP.mNamePt  << " DistReproj: " << Norm2(aProjIm-aMes->mPt) 
	         <<  " Excentricity*1000=" << (1-aL2/aL1) *1000 << " Ray=" << std::sqrt(aL1*aL2) << "\n";
    }
    aMes->mNamePt = aGCP.mNamePt; // match suceed, give the right name
    anIt_SEE->mNameCode = aGCP.mNamePt;
}

void cAppliCompletUncodedTarget::CompleteAll()
{
     for (const auto & aGCP : mMesImGCP.MesGCP())
     {
         CompleteOneGCP(aGCP);
     }
}



int  cAppliCompletUncodedTarget::Exe()
{
   mPhProj.FinishInit();

   mNameReportEllipse = "EllipsesDim";
   InitReport(mNameReportEllipse,"csv",true);


   if (RunMultiSet(0,0))  // If a pattern was used, run in // by a recall to itself  0->Param 0->Set
   {
       AddOneReportCSV(mNameReportEllipse,{"Image","Pt","Ray","Ratio"});
      int aRes =  ResultMultiSet();
      if (aRes!=EXIT_SUCCESS) return aRes;

      // DPPointsMeasures()

      return EXIT_SUCCESS;
   }

   mNameIm = FileOfPath(mSpecImIn);
   mPhProj.LoadSensor(mNameIm,mSensor,mCamPC,false);

   //   load CGP
   mPhProj.LoadGCP(mMesImGCP);
   mPhProj.LoadIm(mMesImGCP,*mSensor);
   mImageM = mPhProj.LoadMeasureIm(mNameIm);


   // evenntually estimate normal from a subset of point
   if (IsInit(&mPatternNormal))
   {
       std::vector<cPt3dr> aVPt;
       for (const auto & aGCP :  mMesImGCP.MesGCP())
       {
           if (MatchRegex(aGCP.mNamePt,mPatternNormal))
	   {
                aVPt.push_back(aGCP.mPt);
	   }
       }
       auto [aPlane,aCost] = cPlane3D::RansacEstimate(aVPt,true);
       mNormal = aPlane.AxeK();
       if (LevelCall() ==0)
           StdOut() <<  "Normal,  Dist=" << aCost << " Axe=" <<  mNormal << std::endl;
   }



   std::string  aNameE = cSaveExtrEllipe::NameFile(mPhProj,mMesImGCP.MesImInitOfName(mNameIm),true);
   ReadFromFile(mVSEE,aNameE);

   CompleteAll();
   // mCamPC = mPhProj.AllocCamPC(FileOfPath(mSpecImIn),true);

   // StdOut()  << mNameIm << " Fff=" << mCamPC->InternalCalib()->F()  << " "<<  mCamPC->NameImage() << std::endl;


   mPhProj.SaveMeasureIm(mImageM);
   // Save GCP because they will probaly be re-used, but do it only once at first call, else risk of simultaneaous writting
   if (KthCall()==0)
   {
       //mPhProj.SaveGCP(mMesImGCP,"");
       mPhProj.CpGCP();
   }

   aNameE = cSaveExtrEllipe::NameFile(mPhProj,mMesImGCP.MesImInitOfName(mNameIm),false);
   SaveInFile(mVSEE,aNameE);

   return EXIT_SUCCESS;
}

/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

tMMVII_UnikPApli Alloc_CompletUncodedTarget(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliCompletUncodedTarget(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecCompletUncodedTarget
(
     "CodedTargetCompleteUncoded",
      Alloc_CompletUncodedTarget,
      "Complete detection, with uncoded target",
      {eApF::ImProc,eApF::CodedTarget},
      {eApDT::Image,eApDT::Xml},
      {eApDT::Xml},
      __FILE__
);


};

