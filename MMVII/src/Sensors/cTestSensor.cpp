#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Sensor.h"


namespace MMVII
{
/* =============================================== */
/*                                                 */
/*                 cAppliTestSensor                */
/*                                                 */
/* =============================================== */

/**  An application for  testing the accuracy of a sensor : 
        - consistency of direct/inverse model
        - (optionnaly) comparison with a ground truth
 */

class cAppliTestSensor : public cMMVII_Appli
{
     public :
        cAppliTestSensor(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
	std::vector<std::string>  Samples() const override;


        void  DoOneImage(const std::string & aNameIm);

	///  Test that the accuracy of ground truth, i.e Proj(P3) = P2
        void TestGroundTruth(const  cSensorImage & aSI) const;
	///  Test coherence of Direct/Inverse model, i.e Id = Dir o Inv = Inv o Dir
        void TestCoherenceDirInv(const  cSensorImage & aSI) const;
	///  Export the synthetic 3d-2d measure
        void ExportMeasures(const  cSensorImage & aSI) const;
	///  Test coherence of Graddient  Grad/Dif finite ~ Grad analytic
        void TestCoherenceGrad(const  cSensorImage & aSI) const;
	///  Test coherence of Graddient  Grad/Dif finite ~ Grad analytic
        void TestPoseLineSensor(const  cSensorImage & aSI) const;


        cPhotogrammetricProject  mPhProj;
        std::string              mPatImage;
        bool                     mShowDetail;

	std::vector<int>         mSzGenerate;
	bool                     mTestCorDirInv;
	tREAL8                   mMaxDistImCorDirInv;  // if used in bench will make assertiob on max val
        cPt2dr                   mDefIntDepth;  // Defautlt interval of depth if sensor has depth functs
        bool                     mDoTestGrad;
	std::vector<tREAL8>      mTestPLS;   // Test Pose Line Sensor

        cPt2dr                   mCurIntZD ;   // curent interval of Z or Depth
	bool                     mCurWDepth;   // does current sensor use depth or Z
        cSet2D3D                 mCurS23;      // current set of "3d-2d" correspondance
};

std::vector<std::string>  cAppliTestSensor::Samples() const
{
   return {
              "MMVII TestSensor SPOT_1B.tif SPOT_Init InPointsMeasure=XingB"
	};
}

cAppliTestSensor::cAppliTestSensor(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli    (aVArgs,aSpec),
    mPhProj         (*this),
    mShowDetail     (false),
    mSzGenerate     {15,3},
    mTestCorDirInv  (true),
    mDefIntDepth    (1.0,2.0),
    mDoTestGrad     (false)
{
}

cCollecSpecArg2007 & cAppliTestSensor::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
      return    anArgObl
             << Arg2007(mPatImage,"Name of input Image", {eTA2007::FileDirProj,{eTA2007::MPatFile,"0"}})
             << mPhProj.DPOrient().ArgDirInMand()
      ;
}

cCollecSpecArg2007 & cAppliTestSensor::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return anArgOpt
               << mPhProj.DPGndPt3D().ArgDirInOpt()
               << mPhProj.DPGndPt3D().ArgDirOutOpt()
               << mPhProj.DPGndPt2D().ArgDirInOpt()
               << mPhProj.DPGndPt2D().ArgDirOutOpt()
               << AOpt2007(mShowDetail,"ShowD","Show detail",{eTA2007::HDV})
               << AOpt2007(mTestCorDirInv,"TestCDI","Test coherence of direct/invers model",{eTA2007::HDV})
               << AOpt2007(mSzGenerate,"SzGen","Sz gen",{eTA2007::HDV,{eTA2007::ISizeV,"[2,2]"}})
               << AOpt2007(mDoTestGrad,"TestGrad","Test coherence anlytic grad/finit diff (if different)",{eTA2007::HDV})
               << AOpt2007(mTestPLS,"TestPLS","Test Pose Line Sensor",{{eTA2007::ISizeV,"[2,2]"}})
               << AOpt2007(mMaxDistImCorDirInv,"CCImDI","Check coherence image dist inverse")
            ;
}

void cAppliTestSensor::TestGroundTruth(const  cSensorImage & aSI) const
{
    // Load mesure from standard MMVII project
    cSetMesGndPt aSetMes;
    mPhProj.LoadGCP3D(aSetMes);
    mPhProj.LoadIm(aSetMes,aSI.NameImage());
    cSet2D3D aSetM23;
    aSetMes.ExtractMes1Im(aSetM23,aSI.NameImage());

    cStdStatRes  aStCheckIm;  //  Statistic of reproj errorr

    for (const auto & aPair : aSetM23.Pairs()) // parse all pair to accumulate stat of errors
    {
         cPt3dr  aPGr = aPair.mP3;
         cPt2dr  aPIm = aSI.Ground2Image(aPGr);
	 tREAL8 aDifIm = Norm2(aPIm-aPair.mP2);
	 aStCheckIm.Add(aDifIm);

         if (mShowDetail) 
         {
             StdOut()  << "ImGT=" <<  aDifIm << std::endl;

         }
    }
    StdOut() << "     ==============  Accuracy / Ground trurh =============== " << std::endl;
    StdOut()  << "    Avg=" <<  aStCheckIm.Avg() << ",  Worst=" << aStCheckIm.Max()  << " Med=" << aStCheckIm.ErrAtProp(0.5) << "\n";
}

void cAppliTestSensor::ExportMeasures(const  cSensorImage & aSI) const
{
    cSetMesGnd3D      aSet3D(aSI.NameImage());
    cSetMesPtOf1Im  aSet2D(aSI.NameImage());
    int aNb=0;
    for (const auto & aPair : mCurS23.Pairs())
    {
        std::string aName = "Pt_"+ ToStr(aNb) + "-" + aSI.NameImage();
        aNb++;
        aSet2D.AddMeasure(cMesIm1Pt(aSI.Ground2Image(aPair.mP3),aName,1.0));
        aSet3D.AddMeasure3D(cMes1Gnd3D(aPair.mP3,aName,1.0));

    }
    mPhProj.SaveGCP3D(aSet3D);
    mPhProj.SaveMeasureIm(aSet2D);
}

void cAppliTestSensor::TestCoherenceGrad(const  cSensorImage & aSI) const
{
    cStdStatRes aStatProj;
    cStdStatRes aStatGradI;
    cStdStatRes aStatGradJ;
    for (const auto & aPair : mCurS23.Pairs())
    {
	tProjImAndGrad  aP1 = aSI.DiffGround2Im (aPair.mP3);
	tProjImAndGrad  aP2 = aSI.DiffG2IByFiniteDiff (aPair.mP3);

	aStatProj.Add(Norm2(aP1.mPIJ - aP2.mPIJ));
	aStatGradI.Add(Norm2(aP1.mGradI - aP2.mGradI));
	aStatGradJ.Add(Norm2(aP1.mGradJ - aP2.mGradJ));


        // DiffGround2Im DiffG2IByFiniteDiff
    }
    StdOut() << "  ==============  Derivate Analytic/finite differnce =============== " << std::endl;
    StdOut() << "  Proj=" << aStatProj.Avg() 
	     << " GradI=" << aStatGradI.Avg() 
	     << " GradJ=" << aStatGradJ.Avg() << "\n";
}

void cAppliTestSensor::TestPoseLineSensor(const  cSensorImage & aSI) const
{
    bool RowIsCol = false;
    int  aNbRow =  mTestPLS.at(0);
    int  aNbInRow =  mTestPLS.at(1);
    int aNbTotRow =  RowIsCol ? aSI.Sz().x() : aSI.Sz().y() ;

    for (int aKRow=0 ; aKRow< aNbRow ; aKRow++)
    {
        tREAL8 aCoord = ((aKRow +0.5) / double(aNbRow)) * aNbTotRow;
	bool Ok;
	std::vector<tREAL8> aResidual;

	StdOut()  <<  "CCC="  << aCoord << aSI.Sz() << " " << aNbTotRow << std::endl;

	tPoseR aPose = aSI.GetPoseLineSensor(aCoord,RowIsCol,aNbInRow,&Ok,&aResidual);

	StdOut()  << " C="  << aPose.Tr() << " R=" << aResidual << " "   << aKRow << " " << aNbRow << "\n";
    }
}

void cAppliTestSensor::TestCoherenceDirInv(const  cSensorImage & aSI) const
{
     
     cStdStatRes  aStConsistIm;  // stat for image consit  Proj( Proj-1(PIm)) ?= PIm
     cStdStatRes  aStConsistGr;  // stat for ground consist  Proj-1 (Proj(Ground)) ?= Ground
     // cStdStatRes  aStConsistGr;  // stat for ground consist  Proj-1 (Proj(Ground)) ?= Ground

     for (const auto & aPair : mCurS23.Pairs())
     {
         cPt3dr  aPGr = aPair.mP3;
         cPt3dr  aPIm (aPair.mP2.x(),aPair.mP2.y(),aPGr.z());


         cPt3dr  aPIm2 ;
         cPt3dr  aPGr2 ;
	
	 if (mCurWDepth)
	 {
	    aPIm2 = aSI.Ground2ImageAndDepth(aSI.ImageAndDepth2Ground(aPIm));
	    aPGr2 = aSI.ImageAndDepth2Ground(aSI.Ground2ImageAndDepth(aPGr));
	 }
	 else
	 {
	    aPIm2 = aSI.Ground2ImageAndZ(aSI.ImageAndZ2Ground(aPIm));
	    aPGr2 = aSI.ImageAndZ2Ground(aSI.Ground2ImageAndZ(aPGr));
	 }
	 tREAL8 aDifIm = Norm2(aPIm-aPIm2);
	 aStConsistIm.Add(aDifIm);

	 tREAL8 aDifGr = Norm2(aPGr-aPGr2);
	 aStConsistGr.Add(aDifGr);

	 // StdOut() <<  "DiiFimm " << aDifIm << "\n"; getchar();
     }

     StdOut() << "     ==============  Consistencies Direct/Inverse =============== " << std::endl;

     StdOut() << "     * Image :  Avg=" <<   aStConsistIm.Avg() 
	      <<  ", Worst=" << aStConsistIm.Max()  
	      <<  ", Med=" << aStConsistIm.ErrAtProp(0.5)  
              << std::endl;

     if (IsInit(& mMaxDistImCorDirInv))
     {
         MMVII_INTERNAL_ASSERT_tiny(aStConsistIm.Max()<mMaxDistImCorDirInv,"Consist Dir/Inv for tested camera");
     }

     StdOut() << "     * Ground:  Avg=" <<   aStConsistGr.Avg() 
	      <<  ", Worst=" << aStConsistGr.Max()  
	      <<  ", Med=" << aStConsistGr.ErrAtProp(0.5)  
              << std::endl;
}

void  cAppliTestSensor::DoOneImage(const std::string & aNameIm)
{
    cSensorImage *  aSI =  mPhProj.ReadSensor(FileOfPath(aNameIm,false /* Ok Not Exist*/),true/*DelAuto*/,false /* Not SVP*/);

    
    StdOut() << std::endl;
    StdOut() << "******************************************************************" << std::endl;
    StdOut() <<  "  Image=" <<  aNameIm  << "  NAMEORI=[" << aSI->NameOriStd()  << "]" << std::endl;


     //  Compute a set of synthetic  correspondance 3d-2d
     mCurWDepth = ! aSI->HasIntervalZ();  // do we use Im&Depth or Image&Z
     mCurIntZD = mDefIntDepth;
     if (mCurWDepth)
     {  // if depth probably doent matter which one is used
     }
     else
     {
        mCurIntZD = aSI->GetIntervalZ(); // at least with RPC, need to get validity interval
     }
     int mNbByDim = mSzGenerate.at(0);
     int mNbDepth = mSzGenerate.at(1);
     mCurS23 = aSI->SyntheticsCorresp3D2D(mNbByDim,mNbDepth,mCurIntZD.x(),mCurIntZD.y(),mCurWDepth);

    // cSensorImage *  aSI =  AllocAutoSensorFromFile(mNameRPC,mNameImage);

    if (mPhProj.DPGndPt3D().DirInIsInit() && mPhProj.DPGndPt2D().DirInIsInit())
       TestGroundTruth(*aSI);

    if (mTestCorDirInv)
       TestCoherenceDirInv(*aSI);

    if (mPhProj.DPGndPt3D().DirOutIsInit() && mPhProj.DPGndPt2D().DirOutIsInit())
       ExportMeasures(*aSI);

    if (mDoTestGrad)
        TestCoherenceGrad(*aSI);

    if (IsInit(&mTestPLS))
       TestPoseLineSensor(*aSI);

    // delete aSI;
}



int cAppliTestSensor::Exe()
{
    mPhProj.FinishInit();
    /*  Version "à la main" on parcourt les explicitement les images
    for (const auto & aNameIm :  VectMainSet(0))
    {
         DoOneImage(aNameIm);
    }
    */

    if (RunMultiSet(0,0))
    {
       return ResultMultiSet();
    }
    DoOneImage(UniqueStr(0));
    return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_TestImportSensors(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
      return tMMVII_UnikPApli(new cAppliTestSensor(aVArgs,aSpec));
}


cSpecMMVII_Appli  TheSpecTestSensor
(
     "TestSensor",
      Alloc_TestImportSensors,
      "Test orientation functions of a sensor : coherence Direct/Inverse, ground truth 2D/3D correspondance, generate 3d-2d corresp",
      {eApF::Ori},
      {eApDT::Ori,eApDT::GndPt2D,eApDT::GndPt3D},
      {eApDT::Console},
      __FILE__
);

};
