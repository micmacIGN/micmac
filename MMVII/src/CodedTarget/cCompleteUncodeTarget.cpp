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
using namespace  cNS_CodedTarget;

/*  *********************************************************** */
/*                                                              */
/*           cAppliCompletUncodedTarget                         */
/*                                                              */
/*  *********************************************************** */

class cAppliCompletUncodedTarget : public cMMVII_Appli
{
     public :

        cAppliCompletUncodedTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;


	void CompleteAll();
	void CompleteOneGCP(const cMes1GCP & aGCP);

        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

	cPhotogrammetricProject     mPhProj;
        cPt3dr                      mNormal;
	tREAL8                      mRayTarget;

        std::string                 mSpecImIn;
        std::string                 mNameIm;
        cSensorImage *              mSensor;
        cSensorCamPC *              mCamPC;
        cSetMesImGCP                mMesImGCP;
};

cAppliCompletUncodedTarget::cAppliCompletUncodedTarget
(
    const std::vector<std::string> & aVArgs,
    const cSpecMMVII_Appli & aSpec
) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this),
   mNormal       (0,0,1)
{
}

        // cExtract_BW_Target * 
cCollecSpecArg2007 & cAppliCompletUncodedTarget::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return
            anArgObl
         << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
	 << mPhProj.DPOrient().ArgDirInMand()

   ;
}

cCollecSpecArg2007 & cAppliCompletUncodedTarget::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return 
                  anArgOpt
	     <<   mPhProj.DPPointsMeasures().ArgDirInputOptWithDef("Std")
             << AOpt2007(mRayTarget,"RayTarget","Ray for target (else estimate automatically)")
		/*
             << AOpt2007(mB,"VisuEllipse","Make a visualisation extracted ellispe & target",{eTA2007::HDV})
             << mPhProj.DPMask().ArgDirInOpt("TestMask","Mask for selecting point used in detailed mesg/output")
             << AOpt2007(mPBWT.mMinDiam,"DiamMin","Minimum diameters for ellipse",{eTA2007::HDV})
             << AOpt2007(mPBWT.mMaxDiam,"DiamMax","Maximum diameters for ellipse",{eTA2007::HDV})
             << AOpt2007(mRatioDMML,"RDMML","Ratio Distance minimal bewteen local max /Diam min ",{eTA2007::HDV})
             << AOpt2007(mVisuLabel,"VisuLabel","Make a visualisation of labeled image",{eTA2007::HDV})
             << AOpt2007(mVisuElFinal,"VisuEllipse","Make a visualisation extracted ellispe & target",{eTA2007::HDV})
             << AOpt2007(mPatHihlight,"PatHL","Pattern for highliting targets in visu",{eTA2007::HDV})
	     */
          ;
}


void cAppliCompletUncodedTarget::CompleteOneGCP(const cMes1GCP & aGCP)
{
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

   if (RunMultiSet(0,0))  // If a pattern was used, run in // by a recall to itself  0->Param 0->Set
   {
      return ResultMultiSet();
   }

   mNameIm = FileOfPath(mSpecImIn);
   mPhProj.LoadSensor(mNameIm,mSensor,mCamPC,true);

   mPhProj.LoadGCP(mMesImGCP);
   mPhProj.LoadIm(mMesImGCP,*mSensor);

   // mCamPC = mPhProj.AllocCamPC(FileOfPath(mSpecImIn),true);

   StdOut()  << mNameIm << " Fff=" << mCamPC->InternalCalib()->F()  << " "<<  mCamPC->NameImage() << "\n";


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

