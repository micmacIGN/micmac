#include "MMVII_PCSens.h"
#include "MMVII_ImageInfoExtract.h"
#include "MMVII_ExtractLines.h"


namespace MMVII
{

/* =============================================== */
/*                                                 */
/*                 cAppliLine2DTo3D               */
/*                                                 */
/* =============================================== */


/** 
 * 
 */

#if (0)
class cAppliLine2DTo3D : public cMMVII_Appli
{
     public :
        cAppliLine2DTo3D(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
     private :

	cPt2dr Redist(const cPt2dr &) const;
	cPt2dr Undist(const cPt2dr &) const;

        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
	std::vector<std::string>  Samples() const override;

        cPerspCamIntrCalib *     mCalib;

        virtual ~cAppliLine2DTo3D();
};


cAppliLine2DTo3D::cAppliLine2DTo3D(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli      (aVArgs,aSpec),
    mPhProj           (*this),
    mCalib            (nullptr)
{
}

cAppliExtractLine::~cAppliExtractLine()
{
     delete mExtrL;
     DeleteAllAndClear(mVPS);
}

cCollecSpecArg2007 & cAppliExtractLine::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
      return    anArgObl
             <<  Arg2007(mPatImage,"Name of input Image", {eTA2007::FileDirProj,{eTA2007::MPatFile,"0"}})
	     <<  Arg2007(mLineIsWhite," True : its a light line , false dark ")
      ;
}

cCollecSpecArg2007 & cAppliExtractLine::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return anArgOpt
               << mPhProj.DPOrient().ArgDirInOpt("","Folder for calibration to integrate distorsion")
	       << AOpt2007(mAffineMax,"AffineMax","Affinate the local maxima",{eTA2007::HDV})
	       << AOpt2007(mShowSteps,"ShowSteps","Show detail of computation steps by steps",{eTA2007::HDV})
	       << AOpt2007(mZoomImL,"ZoomImL","Zoom for images of line",{eTA2007::HDV})
	       << AOpt2007(mRelThrsCumulLow,"ThrCumLow","Low Thresold relative for cumul in histo",{eTA2007::HDV})
	       << AOpt2007(mRelThrsCumulHigh,"ThrCumHigh","Low Thresold relative for cumul in histo",{eTA2007::HDV})
               << mPhProj.DPPointsMeasures().ArgDirInOpt("","Folder for ground truth measure")
            ;
}

std::vector<std::string>  cAppliExtractLine::Samples() const
{
   return {
              "MMVII ExtractLine 'DSC_.*.JPG' ShowSteps=1 InOri=FB"
	};
}

int cAppliExtractLine::Exe()
{
    return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_AppliExtractLine(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
      return tMMVII_UnikPApli(new cAppliExtractLine(aVArgs,aSpec));
}


cSpecMMVII_Appli  TheSpecAppliExtractLine
(
     "ExtractLine",
      Alloc_AppliExtractLine,
      "Extraction of lines",
      {eApF::Ori},
      {eApDT::Ori,eApDT::GCP},
      {eApDT::Console},
      __FILE__
);
#endif

};
