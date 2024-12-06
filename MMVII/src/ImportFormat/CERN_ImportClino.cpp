#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"
#include "MMVII_Stringifier.h"

/**
   \file cConvCalib.cpp  testgit

   \brief file for conversion between calibration (change format, change model) and tests
*/


namespace MMVII
{

   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_CERN_ImportClino                    */
   /*                                                            */
   /* ********************************************************** */

class cAppli_CERN_ImportClino : public cMMVII_Appli
{
     public :
        cAppli_CERN_ImportClino(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :

        void MakeOneDir(const std::string & aDir,cMMVII_Ofs &) const;
	cPhotogrammetricProject  mPhProj;

	// Mandatory Arg
	std::string               mDirData;
	std::string               mNameRes;
	std::vector<std::string>  Samples() const override;

	std::vector<std::string>  mNamesClino;
        std::string               mNameFile;
        std::string               mPatIm;
};

cAppli_CERN_ImportClino::cAppli_CERN_ImportClino(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this),
   mNamesClino   {"A1","B1","B2","A2"},
   mNameFile     ("ClinoValue.json"),
   mPatIm        ("043.*")
{
}

cCollecSpecArg2007 & cAppli_CERN_ImportClino::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mDirData ,"Folder where data are to be researched",{{eTA2007::FolderAny}})
	      <<  Arg2007(mNameRes ,"Name of result file")
           ;
}

cCollecSpecArg2007 & cAppli_CERN_ImportClino::ArgOpt(cCollecSpecArg2007 & anArgFac) 
{
    
    return anArgFac
           << AOpt2007(mNamesClino,"NameClino","Name of Clino",{eTA2007::HDV})
           << AOpt2007(mNameFile,"NameFile","Name of file in each folder",{eTA2007::HDV})
           << AOpt2007(mPatIm,"PatIm","Pattern for extracting image from folder",{eTA2007::HDV})
       //  << AOpt2007(mNbDigName,"NbDigName","Number of digit for name, if fixed size required (only if int)")
       //  << AOpt2007(mL0,"NumL0","Num of first line to read",{eTA2007::HDV})
       //  << AOpt2007(mLLast,"NumLast","Num of last line to read (-1 if at end of file)",{eTA2007::HDV})
       //  << AOpt2007(mPatternTransfo,"PatName","Pattern for transforming name (first sub-expr)")
    ;
}

//template<class Type> inline Type GetV(std::istringstream & iss);


void cAppli_CERN_ImportClino::MakeOneDir(const std::string & aDir,cMMVII_Ofs & anOFS) const
{
    std::string aNameF = aDir + StringDirSeparator() +  mNameFile;
    std::ifstream infile(aNameF);

    std::vector<std::string>  aVFileIm =  GetFilesFromDir(aDir+StringDirSeparator(),AllocRegex(mPatIm));

    MMVII_INTERNAL_ASSERT_tiny(aVFileIm.size()==1,"cAppli_CERN_ImportClino : bad size for image pattern match");
    anOFS.Ofs() << aVFileIm.at(0) ;
    StdOut() << "DDD " << aDir << " " << aVFileIm << "\n";

    std::string line;
    int aNumL = 0;
    cCarLookUpTable aLUT;
    aLUT.InitIdGlob();
    aLUT.Init("[],",' ');
    while (std::getline(infile, line))
    {
        line = aLUT.Translate(line);
	std::istringstream iss(line);

	for (const auto & aNameClino : mNamesClino)
	{
		tREAL8 aAvg = GetV<tREAL8>(iss,aNameF,aNumL);
		tREAL8 aStdDev = GetV<tREAL8>(iss,aNameF,aNumL);
		StdOut() << "   * " << aNameClino  << " " << aAvg << " " << aStdDev << "\n";
		anOFS.Ofs() << " " << aNameClino  << " " << aAvg << " " << aStdDev ;
	}
        aNumL++;
    }
    anOFS.Ofs() << std::endl;
}


int cAppli_CERN_ImportClino::Exe()
{

/*
 *    To redo completely, CERN having adpoted a more standard format, this command is a priori useless
 *
 *    Now it may be usefull, rewritten in much more general way, in collecting file from different folders
 *    from different files.
*/

    mPhProj.FinishInit();

    tNameSelector   aSelec = AllocRegex("Calibration_Clino_.*");
    // std::vector<std::string>   aTEST = RecGetFilesFromDir(mDirData,aSelec,0,100);
    // StdOut() << "TTT=" << aTEST << "\n";


    std::vector<std::string>   aLD = GetSubDirFromDir(mDirData,aSelec);
    std::sort(aLD.begin(),aLD.end());

    cMMVII_Ofs anOFS(mNameRes,eFileModeOut::CreateText);
    for (const auto & aDir : aLD)
        MakeOneDir(aDir,anOFS);


    return EXIT_SUCCESS;
}


std::vector<std::string>  cAppli_CERN_ImportClino::Samples() const
{
    return {"MMVII CERN_ImportClino  ./ MMC.txt PatIm=\"043.*\" NameClino=[A,B,C,D]  NameFile=ClinoValue.json"};
}


tMMVII_UnikPApli Alloc_CERN_ImportClino(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_CERN_ImportClino(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_CERN_ImportClino
(
     "CERN_ImportClino",
      Alloc_CERN_ImportClino,
      "A temporary command to arrange clino format from file splited in different folders",
      {eApF::Ori},
      {eApDT::Ori},
      {eApDT::Ori},
      __FILE__
);
/*
*/


}; // MMVII

