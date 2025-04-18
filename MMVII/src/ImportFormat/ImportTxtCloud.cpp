#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_ReadFileStruct.h"
#include "MMVII_util_tpl.h"
#include "MMVII_PointCloud.h"


/**
   \file cConvCalib.cpp  testgit

   \brief file for conversion between calibration (change format, change model) and tests
*/


namespace MMVII
{
   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_ImportLines                         */
   /*                                                            */
   /* ********************************************************** */

class cAppli_ImportTxtCloud : public cMMVII_Appli
{
     public :
        cAppli_ImportTxtCloud(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        std::vector<std::string>  Samples() const override;
     private :


	// Mandatory Arg
	std::string              mNameFile;
	std::string              mFormat;

	// Optionall Arg
	cNRFS_ParamRead            mParamNSF;

	//   Format specif
	std::string              mNameX;
	std::string              mNameY;
	std::string              mNameZ;
	std::string              mNameR;
	std::string              mSpecFormatMand;
	std::string              mSpecFormatTot;

        bool                     mMode8B;   ///< do we use 8-bytes
        cPt3dr                   mOffset;  ///< offset to subsract
        bool                     mOffsetIsP0;   ///< do we use First point as offset
	tREAL8                   mRoundingOffset;
        std::string              mNameOut;

};

cAppli_ImportTxtCloud::cAppli_ImportTxtCloud(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli    (aVArgs,aSpec),
   mNameX          ("X"),
   mNameY          ("Y"),
   mNameZ          ("Z"),
   mNameR          ("R"),
   mSpecFormatMand (mNameX+mNameY+mNameZ),
   mSpecFormatTot  (mSpecFormatMand + "/"+ mNameR),
   mMode8B         (true),
   mOffset         (0,0,0),
   mOffsetIsP0     (false),
   mRoundingOffset (1e4)
{
}

cCollecSpecArg2007 & cAppli_ImportTxtCloud::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mNameFile ,"Name of Input File",{eTA2007::FileAny})
	      // <<  Arg2007(mFormat   ,"Format of file as for in spec :  \"" + mSpecFormatTot + "\"")
	      <<  Arg2007(mFormat   ,cNewReadFilesStruct::MsgFormat(mSpecFormatTot))
           ;
}

cCollecSpecArg2007 & cAppli_ImportTxtCloud::ArgOpt(cCollecSpecArg2007 & anArgOpt) 
{
    mParamNSF.AddArgOpt(anArgOpt);

    return    anArgOpt
           << AOpt2007(mOffset,"OffsetValue","Offset to add to pixels",{eTA2007::HDV})
           << AOpt2007(mOffsetIsP0,"OffsetIsP0","Use first points as offset (with some rounding)",{eTA2007::HDV})
           << AOpt2007(mRoundingOffset,"OffsetRounding","Value use for rounding when P0 is offset",{eTA2007::HDV})
           << AOpt2007(mMode8B,"Bytes8","Do we use 8/4 bytes for storing points",{eTA2007::HDV})
           << AOpt2007(mNameOut,"Out","Name of output, def=In+\".dmp\"")

    ;
}


int cAppli_ImportTxtCloud::Exe()
{

    //cTriangulation3D<tREAL8> aTT(mNameFile);
    // StdOut() << "NB " << aTT.NbPts() << "\n";


    cNewReadFilesStruct aNRFS(mFormat,mSpecFormatMand,mSpecFormatTot);
    aNRFS.ReadFile(mNameFile,mParamNSF);
    if (!IsInit(&mNameOut))
        mNameOut = LastPrefix(mNameFile)+".dmp";

    cPointCloud aPC(mMode8B);
    aPC.SetOffset(mOffset);

    for (size_t aK=0 ; aK<aNRFS.NbLineRead() ; aK++)
    {
        cPt3dr aPt = aNRFS.GetPt3dr_XYZ (aK) ;
	if (mOffsetIsP0 && (aK==0))
	{
           cPt3di aOffsDiv = Pt_round_down(aPt/mRoundingOffset);
           mOffset = ToR(aOffsDiv) * mRoundingOffset;
           aPC.SetOffset(mOffset);
	}

        aPC.AddPt(aPt);
    }
    SaveInFile(aPC,mNameOut);

    return EXIT_SUCCESS;
}

std::vector<std::string>  cAppli_ImportTxtCloud::Samples() const
{
   return 
   {
          // "MMVII ImportLine Data-Input/BlaAllLine.txt \"Im?X1Y1X2Y2SigmaBla\" CERNFils \"Comment=#\""
   };
}


tMMVII_UnikPApli Alloc_ImportTxtCloud(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ImportTxtCloud(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ImportTxtCloud
(
     "ImportTxtCloud",
      Alloc_ImportTxtCloud,
      "Import/Convert cloud point in txt format (ply ...)",
      {eApF::Lines},
      {eApDT::Lines},
      {eApDT::Lines},
      __FILE__
);

}; // MMVII

