#include "MMVII_PCSens.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"
#include "MMVII_2Include_Serial_Tpl.h"


/**
   \file cConvCalib.cpp  testgit

   \brief file for conversion between calibration (change format, change model) and tests
*/


namespace MMVII
{

   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_MergeMesImGCP                       */
   /*                                                            */
   /* ********************************************************** */

class cAppli_MergeMesImGCP : public cMMVII_Appli
{
     public :
        cAppli_MergeMesImGCP(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        // std::vector<std::string>  Samples() const override;
     private :

	cPhotogrammetricProject  mPhProj;
        std::string              mSpecImIn;
        std::string              mFolder2;

	// Mandatory Arg

};

cAppli_MergeMesImGCP::cAppli_MergeMesImGCP(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this)
{
}

cCollecSpecArg2007 & cAppli_MergeMesImGCP::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
   return      anArgObl
            << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
            << mPhProj.DPPointsMeasures().ArgDirInMand()
            << mPhProj.DPPointsMeasures().ArgDirInMand("Folder second set",&mFolder2)
            << mPhProj.DPPointsMeasures().ArgDirOutMand()
   ;
 
}

cCollecSpecArg2007 & cAppli_MergeMesImGCP::ArgOpt(cCollecSpecArg2007 & anArgOpt) 
{
      return  anArgOpt
/*
            << AOpt2007(mL0,"NumL0","Num of first line to read",{eTA2007::HDV})
            << AOpt2007(mLLast,"NumLast","Num of last line to read (-1 if at end of file)",{eTA2007::HDV})
            << AOpt2007(mPatIm,"PatIm","Pattern for transforming name [Pat,Replace]",{{eTA2007::ISizeV,"[2,2]"}})
            << AOpt2007(mPatPt,"PatPt","Pattern for transforming/select pt [Pat,Replace] ",{{eTA2007::ISizeV,"[2,2]"}})
            << AOpt2007(mImFilter,"ImFilter","File/Pattern for selecting images")
            << AOpt2007(mOffset,"Offset","Offset to add to image measures",{eTA2007::HDV})
*/
   ;
}

int cAppli_MergeMesImGCP::Exe()
{
   mPhProj.FinishInit();
   std::vector<std::string> aSetNames = VectMainSet(0);

   for (const auto & aNameIma : aSetNames)
   {
        cSetMesPtOf1Im aSet(aNameIma);
        if (mPhProj.HasMeasureIm(aNameIma))
           aSet.AddSetMeasure(mPhProj.LoadMeasureIm(aNameIma),true,true);

        if (mPhProj.HasMeasureImFolder(mFolder2,aNameIma))
           aSet.AddSetMeasure(mPhProj.LoadMeasureImFromFolder(mFolder2,aNameIma),true,true);

        mPhProj.SaveMeasureIm(aSet);
   }

    // cSetMesPtOf1Im LoadMeasureIm(const std::string &,bool InDir=true) const;
    // cSetMesPtOf1Im LoadMeasureImFromFolder(const std::string & aFolder,const std::string &) const;


    return EXIT_SUCCESS;
}


// std::vector<std::string>  cAppli_MergeMesImGCP::Samples() const { return {}; }

/*********************************************************************/
/*                                                                   */
/*                       ImportTiePMul                               */
/*                                                                   */
/*********************************************************************/

tMMVII_UnikPApli Alloc_MergeMesImGCP(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_MergeMesImGCP(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_MergeMesImGCP
(
     "MergeMesImGCP",
      Alloc_MergeMesImGCP,
      "Merge different files of image measur of GCP",
      {eApF::GCP},
      {eApDT::GCP},
      {eApDT::GCP},
      __FILE__
);



}; // MMVII

