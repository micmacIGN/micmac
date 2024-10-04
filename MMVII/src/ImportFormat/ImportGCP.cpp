#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"
#include <regex>

/**
   \file cConvCalib.cpp  testgit

   \brief file for conversion between calibration (change format, change model) and tests
*/


namespace MMVII
{

   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_ImportGCP                           */
   /*                                                            */
   /* ********************************************************** */

class cAppli_ImportGCP : public cMMVII_Appli
{
     public :
        cAppli_ImportGCP(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
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
	std::string                mNameGCP;
	int                        mL0;
	int                        mLLast;
	char                       mComment;
	int                        mNbDigName;
	std::vector<std::string>   mPatternTransfo;        
	double                     mMulCoord;  ///< Coordinates multiplicator (to change units)
	double                     mSigma;
	std::string                mPatternAddInfoFree;
};

cAppli_ImportGCP::cAppli_ImportGCP(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this),
   mL0           (0),
   mLLast        (-1),
   mMulCoord     (1.0),
   mSigma        (1.0),
   mPatternAddInfoFree("")
{
}

cCollecSpecArg2007 & cAppli_ImportGCP::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mNameFile ,"Name of Input File",{eTA2007::FileAny})
              <<  Arg2007(mFormat,"Format of file as for ex \"SNASXYZSS\" ")
              << mPhProj.DPPointsMeasures().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_ImportGCP::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
       << AOpt2007(mNameGCP,"NameGCP","Name of GCP set")
       << AOpt2007(mNbDigName,"NbDigName","Number of digit for name, if fixed size required (only if int)")
       << AOpt2007(mL0,"NumL0","Num of first line to read",{eTA2007::HDV})
       << AOpt2007(mLLast,"NumLast","Num of last line to read (-1 if at end of file)",{eTA2007::HDV})
       << AOpt2007(mPatternTransfo,"PatName","Pattern for transforming name (first sub-expr)",{{eTA2007::ISizeV,"[2,2]"}})
       << mPhProj.ArgChSys(true)  // true =>  default init with None
       << AOpt2007(mMulCoord,"MulCoord","Coordinate multiplier, used to change unity as meter to mm")
       << AOpt2007(mSigma,"Sigma","Sigma for all coords (covar is 0). -1 to make all points free",{eTA2007::HDV})
       << AOpt2007(mComment,"Comment","Character for commented line")
       << AOpt2007(mPatternAddInfoFree,"AddInfoFree","All points whose Additional Info matches this pattern are Free")
    ;
}


int cAppli_ImportGCP::Exe()
{
    int aComment = -1;
    if (IsInit(&mComment))
        aComment = mComment;

    mPhProj.FinishInit();
    std::vector<std::vector<std::string>> aVNames;
    std::vector<std::vector<double>> aVNums;
    std::vector<cPt3dr> aVXYZ,aVWKP;


    MMVII_INTERNAL_ASSERT_tiny(CptSameOccur(mFormat,"XYZN")==1,"Bad format vs NXYZ");

    ReadFilesStruct
    (
        mNameFile, mFormat,
        mL0, mLLast, aComment,
        aVNames,aVXYZ,aVWKP,aVNums
    );


    if (! IsInit(&mNameGCP))
    {
       mNameGCP = FileOfPath(mNameFile,false);
       if (IsPrefixed(mNameGCP))
         mNameGCP = LastPrefix(mNameGCP);
    }

    cChangeSysCo & aChSys = mPhProj.ChSysCo();

    cSetMesGCP aSetM(mNameGCP);

    size_t  aRankP_InF = mFormat.find('N');
    size_t  aRankA_InF = mFormat.find('A');
    size_t  aRankP = (aRankP_InF<aRankA_InF) ? 0 : 1;
    size_t  aRankA = 1 - aRankP;
    bool aHasAdditionalInfo = aRankA_InF != mFormat.npos;
    bool aUseAddInfoFree = IsInit(&mPatternAddInfoFree);
    std::regex aRegexAddInfoFree;
    try {
        aRegexAddInfoFree = std::regex(mPatternAddInfoFree);
    } catch (std::regex_error&) {
        MMVII_UserError(eTyUEr::eBadPattern,"Invalid regular expression for AddInfoFree : '" + mPatternAddInfoFree + "'");
    } catch (...) {
        throw;
    }
    if (aUseAddInfoFree && !aHasAdditionalInfo)
            MMVII_UserError(eTyUEr::eBadOptParam,"AddInfoFree specified but no 'A' in format string");

    // compute output RTL if necessary
    if (mPhProj.ChSysCo().SysTarget()->getType()==eSysCo::eRTL && !mPhProj.ChSysCo().SysTarget()->isReady())
    {
        cWeightAv<tREAL8,cPt3dr> aAvgPt;

        for (size_t aK=0 ; aK<aVXYZ.size() ; aK++)
        {
            aAvgPt.Add(1.0,aVXYZ[aK]);
        }
        std::string aRTLName = mPhProj.ChSysCo().SysTarget()->Def();
        mPhProj.ChSysCo().setTargetsysCo(mPhProj.CreateSysCoRTL(
                                             aAvgPt.Average(),
                                             mPhProj.ChSysCo().SysOrigin()->Def()));
        SaveInFile(mPhProj.ChSysCo().SysTarget()->toSysCoData(),
                   mPhProj.getDirSysCo() + aRTLName + "." + GlobTaggedNameDefSerial());

    }


    for (size_t aK=0 ; aK<aVXYZ.size() ; aK++)
    {
        auto aSigma = mSigma;
        std::string aName = aVNames.at(aK).at(aRankP);

        if (IsInit(&mPatternTransfo))
        {
	//	 aName = PatternKthSubExpr(mPatternTransfo,1,aName);
            aName = ReplacePattern(mPatternTransfo.at(0),mPatternTransfo.at(1),aName);
        }

        if (IsInit(&mNbDigName))
            aName =   ToStr(cStrIO<int>::FromStr(aName),mNbDigName);

        std::string aAdditionalInfo = "";
        if (aHasAdditionalInfo)
        {
            aAdditionalInfo = aVNames.at(aK).at(aRankA);
            if (aUseAddInfoFree && std::regex_match(aAdditionalInfo, aRegexAddInfoFree))
                aSigma = -1;
        }
        aSetM.AddMeasure(cMes1GCP(aChSys.Value(aVXYZ[aK]*mMulCoord),aName,aSigma,aAdditionalInfo));
    }

    mPhProj.SaveGCP(aSetM);
    mPhProj.SaveCurSysCoGCP(aChSys.SysTarget());
   

    return EXIT_SUCCESS;
}


std::vector<std::string>  cAppli_ImportGCP::Samples() const
{
   return 
   {
       "MMVII ImportGCP  2023-10-06_15h31PolarModule.coo  NXYZ Std  NumL0=14 NumLast=34  PatName=\"P\\.(.*)\" NbDigName=4",
       "MMVII ImportGCP  Pannel5mm.obc  NXYZ Std NbDigName=4 ChSys=[LocalPannel]"
   };
}



tMMVII_UnikPApli Alloc_ImportGCP(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ImportGCP(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ImportGCP
(
     "ImportGCP",
      Alloc_ImportGCP,
      "Import/Convert basic GCP file in MMVII format",
      {eApF::GCP},
      {eApDT::GCP},
      {eApDT::GCP},
      __FILE__
);


}; // MMVII

