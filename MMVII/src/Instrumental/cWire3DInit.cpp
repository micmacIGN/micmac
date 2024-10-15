#include "MMVII_BlocRig.h"

#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_2Include_Serial_Tpl.h"

//  RIGIDBLOC


/**
 
   \file cWire3DInit.cpp

   This file contains a command for computing the 3D position of a wire from multiple
   images. 

 */

namespace MMVII
{

/* ==================================================== */
/*                                                      */
/*          cAppli_CalibratedSpaceResection             */
/*                                                      */
/* ==================================================== */

class cAppli_Wire3DInit : public cMMVII_Appli
{
     public :

        cAppli_Wire3DInit(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

	//  std::vector<std::string>  Samples() const override;

     private :
        void MakeOneIm(const std::string & aNameIm);
        void TestWire3D(const std::vector<cSensorCamPC *> & aVCam);

        cPhotogrammetricProject  mPhProj;

	std::string                 mSpecImIn;
        std::list<cBlocOfCamera *>  mListBloc;
        cBlocOfCamera *             mTheBloc;

};

/*
std::vector<std::string>  cAppli_Wire3DInit::Samples() const
{
    return {
	     "MMVII BlockCamInit SetFiltered_GCP_OK_Resec.xml   BA_311_B   '(.*)_(.*).JPG' [1,2]  Rig_311_B"
    };
}
*/


cAppli_Wire3DInit::cAppli_Wire3DInit
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this)
{
}


cCollecSpecArg2007 & cAppli_Wire3DInit::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
             <<  Arg2007(mSpecImIn,"Pattern/file for images", {{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}}  )
             <<  mPhProj.DPOrient().ArgDirInMand()
             <<  mPhProj.DPPointsMeasures().ArgDirInMand()
             <<  mPhProj.DPRigBloc().ArgDirInMand()
           ;
}


cCollecSpecArg2007 & cAppli_Wire3DInit::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return    anArgOpt
	    /*
             << AOpt2007(mNameBloc,"NameBloc","Set the name of the bloc ",{{eTA2007::HDV}})
             << AOpt2007(mMaster,"Master","Set the name of the master bloc, is user wants to enforce it ")
             << AOpt2007(mShowByBloc,"ShowByBloc","Show matricial organization by bloc ",{{eTA2007::HDV}})
             << AOpt2007(mShowBySync,"ShowBySync","Show matricial organization by sync ",{{eTA2007::HDV}})
             << AOpt2007(mTestRW,"TestRW","Call test en Read-Write ",{{eTA2007::HDV}})
             << AOpt2007(mTestNoDel,"TestNoDel","Force a memory leak error ",{{eTA2007::HDV}})
	     */
    ;
}

void cAppli_Wire3DInit::TestWire3D(const std::vector<cSensorCamPC *> & aVCam)
{

     std::vector<cPlane3D>  aVPlane;
     std::vector<cSensorCamPC *>  aVCamOk;
     std::vector<tSeg2dr>         aVSegOk;
     for (const auto & aCam : aVCam)
     {
          cLinesAntiParal1Im   aSetL  = mPhProj.ReadLines(aCam->NameImage());
	  const std::vector<cOneLineAntiParal> & aVL  = 	aSetL.mLines;

	  StdOut() << " NBL=" << aCam->NameImage()  << " NBL=" << aVL.size() << "\n";


	  // At this step we dont handle multiple lines
	  if (aVL.size()==1)
	  {
             tSeg2dr aSeg = aVL.at(0).mSeg;

	     cPt2dr aPIm1 = aSeg.P1();
	     cPt2dr aPIm2 = aSeg.P2();
	     cPt3dr aPG1 = aCam->ImageAndDepth2Ground(cPt3dr(aPIm1.x(),aPIm1.y(),1.0));
	     cPt3dr aPG2 = aCam->ImageAndDepth2Ground(cPt3dr(aPIm2.x(),aPIm2.y(),1.0));
	     cPt3dr aPG0 = aCam->Center();
	     aVPlane.push_back(cPlane3D::From3Point(aPG0,aPG1,aPG2));
	     aVCamOk.push_back(aCam);
	     aVSegOk.push_back(aSeg);
	     // StdOut() <<  "     " <<  aSeg.P1() << " " << aSeg.P2() << "\n";
	  }
     }

     if (aVPlane.size() >= 3)
     {
	 cSegmentCompiled<tREAL8,3> aWire = cPlane3D::InterPlane(aVPlane);
	 for (size_t aKC=0 ; aKC<aVCamOk.size() ; aKC++)
	 {
             cSensorCamPC * aCam = aVCamOk[aKC];
             cPerspCamIntrCalib * aCalib = aCam->InternalCalib();
             cPt2dr aPIm1 =  aVSegOk[aKC].P1();
             cPt2dr aPIm2 =  aVSegOk[aKC].P2();
	     int aNbSeg = 5;

	     cPt2dr  aPU1 = aCalib->Undist(aPIm1);
	     cPt2dr  aPU2 = aCalib->Undist(aPIm2);
	     cSegment2DCompiled aSegU(aPU1,aPU2);

	     for (int aKS=0 ; aKS<=aNbSeg ; aKS++)
	     {
		  tREAL8 aW1 = 1.0 - aKS/tREAL8(aNbSeg); 

		  cPt2dr aPU = aPU1 * aW1 + aPU2 * (1-aW1);

		  cPt2dr aPIm = aCalib->Redist(aPU);
                  cSegmentCompiled<tREAL8,3> aBundIm = aCam->Image2Bundle(aPIm);
                  cPt3dr  aPWire =  BundleInters(aWire,aBundIm,1.0);

		  cPt2dr aProjW = aCam->Ground2Image(aPWire);
		  cPt2dr aPUW = aCalib->Undist(aProjW);

	          StdOut() << "DIST3 "  << aBundIm.Dist(aPWire) 
			  << " DistPix=" << Norm2(aProjW-aPIm)  
			  << " Prof "  << Norm2(aPWire-aCam->Center())
			  << " DistUPix=" << aSegU.DistLine(aPUW)
			  << "\n";
	     }
	     /*
		
			     cSegmentCompiled<tREAL8,3> aB1 = aVCamOk[aKC]->Image2Bundle( aVSegOk[aKC].P1());
             cPt3dr  aPW1 =  BundleInters(aWire,aB1,1.0);

             cSegmentCompiled<tREAL8,3> aB2 = aVCamOk[aKC]->Image2Bundle( aVSegOk[aKC].P2());
             cPt3dr  aPW2 =  BundleInters(aWire,aB2,1.0);

	     StdOut() << "DIST "  << aB1.Dist(aPW1) << " " << aB2.Dist(aPW2) << "\n";

	     cPt2dr aProjW1 = 
	     */

	 }
     }
      

     StdOut() << "NBPL " << aVPlane.size() << "\n";
}


void cAppli_Wire3DInit::MakeOneIm(const std::string & aNameIm)
{
     std::string anIdBloc = mTheBloc->IdBloc(aNameIm);
     std::string anIdSync = mTheBloc->IdSync(aNameIm);
     cBlocOfCamera::tMapStrPoseUK& aMap = mTheBloc-> MapStrPoseUK();

     StdOut() << "NBCAM="  << aNameIm 
	     << "  IdBloc=" << anIdBloc
	     << "  IdSync=" << anIdSync
	     << "  Size=" << aMap.size()
	     << "\n";


     std::vector<cSensorCamPC *>  aVCam;
     for (const auto & [aNameBl,aPoseUK] : aMap)
     {
         std::string aNameIm = mTheBloc->Ids2Image(aNameBl,anIdSync);
         cPerspCamIntrCalib *  aIntr = mPhProj.InternalCalibFromImage(aNameIm);
	 aVCam.push_back(new cSensorCamPC (aNameIm,aPoseUK.Pose(),aIntr));
     }

     TestWire3D(aVCam);

     DeleteAllAndClear(aVCam);

     StdOut() << "\n";
}

int cAppli_Wire3DInit::Exe()
{
    mPhProj.FinishInit();  // the final construction of  photogrammetric project manager can only be done now

    mListBloc = mPhProj.ReadBlocCams();
    MMVII_INTERNAL_ASSERT_tiny(mListBloc.size()==1,"Number of bloc ="+ ToStr(mListBloc.size()));

    mTheBloc = *(mListBloc.begin());
    for (const auto & aNameIm : VectMainSet(0))
    {
        MakeOneIm(aNameIm);
        // cSensorCamPC * aCam = 
        // mTheBloc->
    }

    DeleteAllAndClear(mListBloc);
    return EXIT_SUCCESS;
}                                       

/* ==================================================== */
/*                                                      */
/*               MMVII                                  */
/*                                                      */
/* ==================================================== */

tMMVII_UnikPApli Alloc_Wire3DInit(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_Wire3DInit(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_Wire3DInit
(
      "Wire3DInit",
      Alloc_Wire3DInit,
      "Compute 3D position of wire",
      {eApF::Ori},
      {eApDT::Orient}, 
      {eApDT::Xml}, 
      __FILE__
);


}; // MMVII

