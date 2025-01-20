#include "MMVII_PCSens.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"
#include "MMVII_2Include_Serial_Tpl.h"

#include "MMVII_ReadFileStruct.h"

#include "MMVII_util_tpl.h"


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

class cAppli_ImportLines : public cMMVII_Appli
{
     public :
        cAppli_ImportLines(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        std::vector<std::string>  Samples() const override;
     private :

	cPhotogrammetricProject  mPhProj;

	// Mandatory Arg
	std::string              mNameFile;
	std::string              mFormat;

	// Optionall Arg
	cNRFS_ParamRead            mParamNSF;

	//   Format specif
	std::string              mNameIm;
	std::string              mNameX1;
	std::string              mNameY1;
	std::string              mNameX2;
	std::string              mNameY2;
	std::string              mNameSigma;
	std::string              mNameWidth;
	std::string              mSpecFormatMand;
	std::string              mSpecFormatTot;
        cPt2dr                   mOffset;

};

cAppli_ImportLines::cAppli_ImportLines(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli    (aVArgs,aSpec),
   mPhProj         (*this),
   mNameIm         ("Im"),
   mNameX1         ("X1"),
   mNameY1         ("Y1"),
   mNameX2         ("X2"),
   mNameY2         ("Y2"),
   mNameSigma      ("Sigma"),
   mNameWidth      ("Width"),
   mSpecFormatMand (mNameIm+mNameX1+mNameY1+mNameX2+mNameY2),
   mSpecFormatTot  (mSpecFormatMand+"/"+mNameSigma +mNameWidth),
   mOffset         (0,0)
{
	// std::map<std::string,int>  aMap{{"2",2}};
}

cCollecSpecArg2007 & cAppli_ImportLines::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mNameFile ,"Name of Input File",{eTA2007::FileAny})
	      // <<  Arg2007(mFormat   ,"Format of file as for in spec :  \"" + mSpecFormatTot + "\"")
	      <<  Arg2007(mFormat   ,cNewReadFilesStruct::MsgFormat(mSpecFormatTot))
              <<  mPhProj.DPGndPt2D().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_ImportLines::ArgOpt(cCollecSpecArg2007 & anArgOpt) 
{
    mParamNSF.AddArgOpt(anArgOpt);

    return    anArgOpt
           << mPhProj.ArgSysCo()
           << AOpt2007(mOffset,"Offset","Offset to add to pixels",{eTA2007::HDV})

    ;
}


int cAppli_ImportLines::Exe()
{
    mPhProj.FinishInit();
    // BenchcNewReadFilesStruct();

    cNewReadFilesStruct aNRFS(mFormat,mSpecFormatMand,mSpecFormatTot);
    aNRFS.ReadFile(mNameFile,mParamNSF);


   // Create a structure of map, because there can exist multiple line/image
   std::map<std::string,cLinesAntiParal1Im> aMap;

    bool WithSigma = aNRFS.FieldIsKnown(mNameSigma);
    bool WithWidth = aNRFS.FieldIsKnown(mNameWidth);
    for (size_t aK=0 ; aK<aNRFS.NbLineRead() ; aK++)
    {
         // Create potentially a new set of line for image
         std::string aNameIm =  aNRFS.GetValue<std::string>(mNameIm,aK);
         cLinesAntiParal1Im  & aLAP = aMap[aNameIm];
         aLAP.mNameIm  = aNameIm;

	 //  Add a new line
         cOneLineAntiParal aLine;
         cPt2dr aP1=aNRFS.GetPt2dr(aK,mNameX1,mNameY1) + mOffset;
         cPt2dr aP2=aNRFS.GetPt2dr(aK,mNameX2,mNameY2) + mOffset;

	 if (aP1.x() > -100)  // CERN CONVENTION FOR FALSE SEG
	 {
	     aLine.mSeg = tSeg2dr(aP1,aP2);

	     if (WithSigma)
                aLine.mSigmaLine = aNRFS.GetFloat(mNameSigma,aK);
	     if (WithWidth)
                aLine.mWidth = aNRFS.GetFloat(mNameWidth,aK);

	     aLAP.mLines.push_back(aLine);
	 }
    }

    for (const auto & [aStr,aL]  : aMap)
        mPhProj.SaveLines(aL);


    return EXIT_SUCCESS;
}

std::vector<std::string>  cAppli_ImportLines::Samples() const
{
   return 
   {
          "MMVII ImportLine Data-Input/BlaAllLine.txt \"Im?X1Y1X2Y2SigmaBla\" CERNFils \"Comment=#\""
   };
}


tMMVII_UnikPApli Alloc_ImportLines(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ImportLines(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ImportLines
(
     "ImportLine",
      Alloc_ImportLines,
      "Import/Convert Set of lines extracted",
      {eApF::Lines},
      {eApDT::Lines},
      {eApDT::Lines},
      __FILE__
);

}; // MMVII

