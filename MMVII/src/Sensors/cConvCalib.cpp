#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"

/**
   \file cConvCalib.cpp  testgit

   \brief file for conversion between calibration (change format, change model) and tests
*/


namespace MMVII
{


static const std::string ThePatOriV1 = "Orientation-(.*)\\.xml";
static std::string V1NameOri2NameImage(const std::string & aNameOri) {return ReplacePattern(ThePatOriV1,"$1",aNameOri);}


/* ************************************************************* */
/*                                                               */
/*                       cV1PCConverter                          */
/*                                                               */
/* ************************************************************* */

class cV1PCConverter : public cCorresp32_BA
{
    public :
         typedef cIsometry3D<tREAL8>   tPose;
	 /** Take as input the name xml-v1 internal calib and create Calib+Pos+Set , return then 
	    the cV1PCConverter  used in bench & command 4 convert */
         static cV1PCConverter *  AllocV1Converter
		 (
                      const std::string & aFullName,
                      bool HCG,
                      bool  CenterFix,
                      const std::string & aNameCalOut="",
                      tREAL8 aDS=1.0,
                      cPt3di aDeg = cPt3di(3,1,1),
                      const std::string & aNameCalIn ="" //  in case file interne doesnt not exist
                 );

	 /** Alloc a calib from name : create  the converter ,do the iteration, if already created return same object */
         static cPerspCamIntrCalib *       AllocCalibV1
		 (
                       const std::string & aFullName,
                       const std::string & aNameCalOut="",
                       tREAL8 aDS=1.0,
                       cPt3di aDeg=cPt3di(3,1,1),
                       const std::string & aNameCalIn=""
                 );

	 /** */
         static cSensorCamPC * AllocSensorPCV1
                (
                     const std::string& aNameIm,
                     const std::string & aFullName,
                     const std::string& aNameCalOut="",
                     tREAL8 aDS=1.0,
                     cPt3di aDegree=cPt3di(3,1,1),
                     const std::string& aNameCalIn=""
                );

         const cSensorCamPC  &       CamPC() const {return *mCamPC;}
         cPerspCamIntrCalib *    Calib() {return mCamPC->InternalCalib();}

	 ~cV1PCConverter();
    protected :
         cV1PCConverter
         (
	      cSensorCamPC *,
              const cSet2D3D &,
              bool    HardConstrOnGCP=true , // do we fix GCP,  false make sense in test mode
              bool    CenterFix=true        // do we fix centre of projection,  false make sense in test mode
         );
	 cSensorCamPC       *               mCamPC;

};


cV1PCConverter::~cV1PCConverter()
{
      mSetInterv.SIUK_Reset();
      delete mCamPC;
}


     // ==============  constructor & destructor ================

cV1PCConverter::cV1PCConverter
(
     cSensorCamPC *          aCamPC,
     const cSet2D3D &        aSetCorresp,
     bool                    HardConstrOnGCP,
     bool                    CenterFix
) :
     cCorresp32_BA   (aCamPC,aSetCorresp),
     mCamPC          (aCamPC)
{
	mFGC = HardConstrOnGCP;
	mCFix = CenterFix;
}


     // ==============  conversion for V1 ================

cV1PCConverter *  cV1PCConverter::AllocV1Converter
(
      const std::string & aFullName,
      bool HCG,
      bool  CenterFix,
      const std::string & aNameCalOut,
      tREAL8 aDS,
      cPt3di aDeg,
      const std::string & aNameCalIn
)
{
     //  [1]   ==========   Raw-read the parameters from V1 ================
     bool  isForTest = (!HCG) ||  (! CenterFix);
     cExportV1StenopeCalInterne  aExp(true,aFullName,30,2,aDS,aNameCalIn);
     cIsometry3D<tREAL8>   aPose0 = cIsometry3D<tREAL8>::Identity();

     // [2] ============  in mode test perturbate internal et external parameters =================
     if (isForTest)
     {
         aExp.mFoc *=  (1.0 + 0.05*RandUnif_C());
         aExp.mPP = MulCByC(aExp.mPP,  cPt2dr(1,1)+cPt2dr::PRandC()*0.05);
         aPose0 = cIsometry3D<tREAL8>
                         (
                              cPt3dr::PRandC() * (CenterFix ? 0.0 : 0.1),
                              cRotation3D<tREAL8>::RandomRot(0.05)
                         );
     }

     // [3]   ============= 
        // aFullName = ".../Ori-MMV1/AutoCal_Foc-60000_Cam-NIKON_D810.xml"   =>  "Foc-60000_Cam-NIKON_D810" = aNameCam
     std::string aNameCam = LastPrefix(FileOfPath(aFullName,false));
     aNameCam =  cPerspCamIntrCalib::PrefixName() +  ReplacePattern("AutoCal_(.*)","$1",aNameCam);
     
     if (aNameCalOut!="")
	     aNameCam  = aNameCalOut;
        //  Data part for  internal calibration w/o distorsion
     cDataPerspCamIntrCalib aDataCalib(aNameCam,aExp.eProj,aDeg,aExp.mFoc,aExp.mSzCam);
     aDataCalib.PushInformation("Converted from MMV1");  // just for info
     cPerspCamIntrCalib * aCalib = cPerspCamIntrCalib::Alloc(aDataCalib); // the calib itself

     cSensorCamPC * aCamPC = new cSensorCamPC("NONE",aPose0,aCalib);
     return new cV1PCConverter(aCamPC,aExp.mCorresp,HCG,CenterFix); // We have Calib+Pose+corresp : go
}

cPerspCamIntrCalib * cV1PCConverter::AllocCalibV1
(
      const std::string & aFullName,
      const std::string & aNameCalOut,
      tREAL8 aDS,
      cPt3di aDeg,
      const std::string & aNameCalIn
)
{ 
     // If object already created
     static std::map<std::string,cPerspCamIntrCalib *> TheMap;
     cPerspCamIntrCalib * & aPersp = TheMap[aFullName];

     if (aPersp==0)
     {
         // Create the converter
         cV1PCConverter * aConvertor = cV1PCConverter::AllocV1Converter(aFullName,true,true,aNameCalOut,aDS,aDeg,aNameCalIn);
	 // make the bundle adjustment
         for (int aK=0 ; aK<10 ; aK++)
         {
            aConvertor->OneIteration();
         }

         aPersp = aConvertor->Calib();
	 cMMVII_Appli::AddObj2DelAtEnd(aPersp); // deletion will be done at end
	 delete aConvertor;  // delete the convertor that was created for this task
     }

     return aPersp;
}


cSensorCamPC * cV1PCConverter::AllocSensorPCV1
(
      const std::string & aNameIm,
      const std::string & aFullName,
      const std::string& aNameCalOut,
      tREAL8 aDS,
      cPt3di aDegree,
      const std::string& aNameCalIn
)
{
     cExportV1StenopeCalInterne  aExp(false,aFullName,0,0,aDS,aNameCalIn); // Alloc w/o  3d-2d correspondance

     std::string aNameCal = DirOfPath(aFullName,false) + FileOfPath(aExp.mNameCalib,false);
     cPerspCamIntrCalib * aCalib =  cV1PCConverter::AllocCalibV1(aNameCal,aNameCalOut,aDS,aDegree,aNameCalIn);

     return new cSensorCamPC(aNameIm,aExp.mPose,aCalib);
}



   /* ********************************************************** */
   /*                                                            */
   /*               BENCH PART                                   */
   /*                                                            */
   /* ********************************************************** */



void BenchCentralePerspective_ImportCalibV1(cParamExeBench & aParam,const std::string & aName,bool HCG,bool  CenterFix,double aAccuracy)
{
static int aCpt=0; aCpt++;

     std::string aFullName = cMMVII_Appli::CurrentAppli().InputDirTestMMVII() + "Ori-MMV1" +  StringDirSeparator() + aName;


     cV1PCConverter * aConv =   cV1PCConverter::AllocV1Converter(aFullName,HCG,CenterFix);

     const cSensorCamPC  &       aCamPC =  aConv->CamPC() ;
     cPerspCamIntrCalib *        aCalib =  aConv->Calib() ;

     double aResidual  = 10;
     for (int aK=0 ; aK<20 ; aK++)
     {
        aConv->OneIteration();
        aResidual  = aCamPC.AvgSqResidual(aConv->SetCorresp());

        if (aResidual<aAccuracy)
        {
            // create a new file , to avoid reading in map in "FromFile"
	    std::string aNameTmp = cMMVII_Appli::CurrentAppli().TmpDirTestMMVII() + "TestCalib_" + ToStr(aCpt) + "." + GlobTaggedNameDefSerial();
	    // std::string aNameTmp = cMMVII_Appli::CurrentAppli().TmpDirTestMMVII() + aCalib->Name() + ".xml";
	    aCalib->SetName(FileOfPath(Prefix(aNameTmp),false));
	    aCalib->ToFile(aNameTmp);

	    cPerspCamIntrCalib *  aCam2 = cPerspCamIntrCalib::FromFile(aNameTmp);
            cSensorCamPC          aSensor2 ("NONE",aConv->CamPC().Pose(),aCam2) ;
	    double aR2 = aSensor2.AvgSqResidual(aConv->SetCorresp());
           
	    //  Accuracy must be as good as initial camera, but small diff possible due to string conv
            MMVII_INTERNAL_ASSERT_bench(aR2< aAccuracy+1e-5  ,"Reload camera  xml");
	    // diff of accuracy should be tiny
            MMVII_INTERNAL_ASSERT_bench(std::abs(aR2-aResidual)< 1e-10  ,"Reload camera  xml");

	    delete aConv;
	    //delete aCam2; dont delete -> create by FromFile, autom delete at end
	    delete aCalib; // create at hand  -> delete
            return;
        }
     }
     StdOut() << "---------------RR=" <<  aResidual  << std::endl;
     MMVII_INTERNAL_ASSERT_bench(false ,"No convergence in BenchCentralePerspective_ImportV1");
}



void BenchPoseImportV1(const std::string & aNameOriV1,double anAccuracy)
{
     std::string aFullName = cMMVII_Appli::CurrentAppli().InputDirTestMMVII() + "Ori-MMV1" +  StringDirSeparator() + aNameOriV1;

     // AllocSensorPCV1

     cExportV1StenopeCalInterne  aExp(false,aFullName,10); 
     cSensorCamPC  *aPC  =  cV1PCConverter::AllocSensorPCV1(V1NameOri2NameImage(aNameOriV1),aFullName);
     double aResidual  =  aPC->AvgSqResidual(aExp.mCorresp);


     for (const auto & aCorresp : aExp.mCorresp.Pairs())
     {
         cPt3dr aPGround = aCorresp.mP3;
	 cPt3dr aPImAndD = aPC->Ground2ImageAndDepth(aPGround);
	 cPt3dr aPG2 = aPC->ImageAndDepth2Ground(aPImAndD);

	 double aDist = Norm2(aPGround - aPG2);
          MMVII_INTERNAL_ASSERT_bench(aDist<1e-5 ,"I&Depth inversion");
     }

     MMVII_INTERNAL_ASSERT_bench(aResidual<anAccuracy ,"No convergence in BenchCentralePerspective_ImportV1");

     std::string aNameTmp = cMMVII_Appli::CurrentAppli().TmpDirTestMMVII() +  aPC->NameOriStd();  // "ccTestOri.xml";
     aPC->ToFile(aNameTmp);


     cSensorCamPC  *aPC2  =  cSensorCamPC::FromFile(aNameTmp,false);
     double aR2 = aPC2->AvgSqResidual(aExp.mCorresp) ;

     // StdOut() << "BenchPoseImportV1 " << aR2 <<  " " <<  aNameTmp << std::endl; getchar();

     MMVII_INTERNAL_ASSERT_bench(aR2<anAccuracy ,"No Conv in Reimport cam");


     delete aPC;
     delete aPC2;

}

void BenchCentralePerspective_ImportV1(cParamExeBench & aParam)
{
    BenchPoseImportV1("Orientation-_DSC8385.tif.xml" ,1e-5);
    BenchPoseImportV1("Orientation-Img0937.tif.xml"  ,1e-5);

    for (int aK=0 ; aK<3 ; aK++)
    {
        BenchCentralePerspective_ImportCalibV1(aParam,"AutoCal_Foc-60000_Cam-NIKON_D810.xml",false,false,1e-5);
        BenchCentralePerspective_ImportCalibV1(aParam,"AutoCal_Foc-60000_Cam-NIKON_D810.xml",false,true ,1e-5);
        BenchCentralePerspective_ImportCalibV1(aParam,"AutoCal_Foc-60000_Cam-NIKON_D810.xml",true ,false,1e-5);
        BenchCentralePerspective_ImportCalibV1(aParam,"AutoCal_Foc-60000_Cam-NIKON_D810.xml",true ,true ,1e-5);

        BenchCentralePerspective_ImportCalibV1(aParam,"AutoCal_Foc-11500_Cam-imx477imx477-1.xml",false,false,1e-3);
    }
}

    // ===============================================================================================
    // ===============================================================================================
    // ===============================================================================================


   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_OriConvV1V2                         */
   /*                                                            */
   /* ********************************************************** */

class cAppli_OriConvV1V2 : public cMMVII_Appli
{
     public :
        cAppli_OriConvV1V2(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :
	std::string              mDirMMV1;
	cPhotogrammetricProject  mPhProj;
	bool                     mUseStdNameCalib;
	tREAL8                   mDownScale;
	cPt3di                   mDegree;
	std::string              mFileInterne;
};

cAppli_OriConvV1V2::cAppli_OriConvV1V2(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mPhProj          (*this),
   mUseStdNameCalib (false),
   mDownScale       (1.0),
   mDegree          (3,1,1),
   mFileInterne     ("")
{
}

cCollecSpecArg2007 & cAppli_OriConvV1V2::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mDirMMV1 ,"Input Orientation for MMV1 Files")
	      <<  mPhProj.DPOrient().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_OriConvV1V2::ArgOpt(cCollecSpecArg2007 & anArgOpt) 
{
   return    anArgOpt
          << AOpt2007(mUseStdNameCalib,"USNC","Use Standard name for calib (when metadata can be  computed) ",{eTA2007::HDV})
          << AOpt2007(mDownScale,"DS","Down scale, if want to generate smaller ",{eTA2007::HDV})
          << AOpt2007(mDegree,"Degree","Degree of distorsion Rad/Dec/Gen",{eTA2007::HDV})
          << AOpt2007(mFileInterne,"FileInterne","Indicate tag <FileInterne> to MMV1, is absent")
   ;
}


int cAppli_OriConvV1V2::Exe()
{
    mPhProj.FinishInit();

    //std::string aPatV1 = "Orientation-(.*)\\.xml";
    std::vector<std::string> aListOriV1 = GetFilesFromDir(mDirMMV1,AllocRegex(ThePatOriV1),true);

    for (const auto & aNameOri : aListOriV1)
    {
        std::string aNameIm = V1NameOri2NameImage(aNameOri); // ReplacePattern(ThePatOriV1,"$1",aName);
	std::string aNameCalib ="";
	if (mUseStdNameCalib)
           aNameCalib = mPhProj.StdNameCalibOfImage(aNameIm);

        cSensorCamPC * aPC =  cV1PCConverter::AllocSensorPCV1(aNameIm,mDirMMV1+StringDirSeparator()+aNameOri,aNameCalib,mDownScale,mDegree,mFileInterne);

	mPhProj.SaveCamPC(*aPC);

	/*
        StdOut() << "N="  << aNameOri << " " << aPC->Center() 
		<< " CalN=" << aPC->InternalCalib()->Name() 
		<< " ### [" <<   aNameIm << "]\n";
		*/


	delete aPC;
    }


    return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_OriConvV1V2(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_OriConvV1V2(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_OriConvV1V2
(
     "V1OriConv",
      Alloc_OriConvV1V2,
      "Convert orientation of MMV1  to MMVII",
      {eApF::Ori},
      {eApDT::Orient},
      {eApDT::Orient},
      __FILE__
);


}; // MMVII

