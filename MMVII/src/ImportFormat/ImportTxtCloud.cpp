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

class cAppli_ImportTxtCloud : public cMMVII_Appli
{
     public :
        cAppli_ImportTxtCloud(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
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
	std::string              mNameX;
	std::string              mNameY;
	std::string              mNameZ;
	std::string              mNameR;
	std::string              mSpecFormatMand;
	std::string              mSpecFormatTot;

        cPt2dr                   mOffset;

};

cAppli_ImportTxtCloud::cAppli_ImportTxtCloud(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli    (aVArgs,aSpec),
   mPhProj         (*this),
   mNameX          ("X"),
   mNameY          ("Y"),
   mNameZ          ("Z"),
   mNameR          ("R"),
   mSpecFormatMand (mNameX+mNameY+mNameZ),
   mSpecFormatTot  (mSpecFormatMand + "/"+ mNameR),
   mOffset         (0,0)
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
           << AOpt2007(mOffset,"Offset","Offset to add to pixels",{eTA2007::HDV})

    ;
}


int cAppli_ImportTxtCloud::Exe()
{
    mPhProj.FinishInit();

    //cTriangulation3D<tREAL8> aTT(mNameFile);
    // StdOut() << "NB " << aTT.NbPts() << "\n";


    cNewReadFilesStruct aNRFS(mFormat,mSpecFormatMand,mSpecFormatTot);
StdOut() << "JJJJJJJJJJJJJJJ\n";
    aNRFS.ReadFile(mNameFile,mParamNSF);
StdOut() << "HHHHHHHH\n";

/*


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

*/

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

