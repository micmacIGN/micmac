#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"
#include <regex>

#include "MMVII_ReadFileStruct.h"

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
        cNRFS_ParamRead            mParamNSF;
	int                        mNbDigName;
	std::vector<std::string>   mPatternTransfo;        
	double                     mMulCoord;  ///< Coordinates multiplicator (to change units)
	double                     mDefSigma;
	std::string                mPatternAddInfoFree;

	std::string                mFieldGCP;  ///  Field for name of GCP
	std::string                mFieldX;    ///  Field for GCP.x
	std::string                mFieldY;    ///  Field for GCP.y
	std::string                mFieldZ;    ///  Field for GCP.z
					       //
	std::string                mFieldAI;   ///  Field for Additional Info
	std::string                mFieldSx;   ///  Field for Sigma.x
	std::string                mFieldSy;   ///  Field for Sigma.y
	std::string                mFieldSz;   ///  Field for Sigma.z
	std::string                mFieldSxyz; ///  Field for Sigma common to x,y & z

	std::string                mSpecFormatMand;  /// Specification for mandatory field
	std::string                mSpecFormatTot;   /// Specification for optionnal field
};

cAppli_ImportGCP::cAppli_ImportGCP(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this),
   mMulCoord     (1.0),
   mDefSigma     (1.0),
   mPatternAddInfoFree(""),   // Pattern on Additional Info, for specifying free points
   mFieldGCP             ("N"),
   mFieldX               ("X"),
   mFieldY               ("Y"),
   mFieldZ               ("Z"),
   mFieldAI              ("A"),
   mFieldSx              ("Sx"),
   mFieldSy              ("Sy"),
   mFieldSz              ("Sz"),
   mFieldSxyz            ("Sxyz"),
   mSpecFormatMand       (mFieldGCP+mFieldX+mFieldY+mFieldZ),
   mSpecFormatTot        (mSpecFormatMand + " / " + mFieldSx + mFieldSy + mFieldSz + mFieldAI + mFieldSxyz)
   
{
}

cCollecSpecArg2007 & cAppli_ImportGCP::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mNameFile ,"Name of Input File",{eTA2007::FileAny})
              <<  Arg2007(mFormat,cNewReadFilesStruct::MsgFormat(mSpecFormatTot))
              << mPhProj.DPPointsMeasures().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_ImportGCP::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    mParamNSF.AddArgOpt(anArgOpt);
    return anArgOpt
       << AOpt2007(mNameGCP,"NameGCP","Name of GCP set")
       << AOpt2007(mNbDigName,"NbDigName","Number of digit for name, if fixed size required (only if int)")
       << AOpt2007(mPatternTransfo,"PatName","Pattern for transforming name (first sub-expr)",{{eTA2007::ISizeV,"[2,2]"}})
       << mPhProj.ArgChSys(true)  // true =>  default init with None
       << AOpt2007(mMulCoord,"MulCoord","Coordinate multiplier, used to change unity as meter to mm")
       << AOpt2007(mDefSigma,"Sigma","Sigma for all coords (covar is 0). -1 to make all points free",{eTA2007::HDV})
       << AOpt2007(mPatternAddInfoFree,"AddInfoFree","All points whose Additional Info matches this pattern are Free")
    ;
}

int cAppli_ImportGCP::Exe()
{
    mPhProj.FinishInit();

    //  If name of GCP set is not init, fix it with name of file
    if (! IsInit(&mNameGCP))
    {
       mNameGCP = FileOfPath(mNameFile,false);
       if (IsPrefixed(mNameGCP))
         mNameGCP = LastPrefix(mNameGCP);
    }
    cSetMesGCP aSetM(mNameGCP);

    //  Extract chang of coordinate, if not set  handled by default
    cChangeSysCo & aChSys = mPhProj.ChSysCo();

    cNewReadFilesStruct aNRFS(mFormat,mSpecFormatMand,mSpecFormatTot);
    aNRFS.ReadFile(mNameFile,mParamNSF);


    bool withAddInfo  = aNRFS.FieldIsKnown(mFieldAI);
    bool withPatternAddInfoFree = IsInit(&mPatternAddInfoFree);

    bool wSigmaX       = aNRFS.FieldIsKnown(mFieldSx);
    bool wSigmaY       = aNRFS.FieldIsKnown(mFieldSy);
    bool wSigmaZ       = aNRFS.FieldIsKnown(mFieldSz);
    bool wSigmaXYZ      = aNRFS.FieldIsKnown(mFieldSxyz);

    // too complicate to handle partiall case of fixing sigma, and btw, not pertinent ?
    MMVII_INTERNAL_ASSERT_tiny((wSigmaX==wSigmaY)&&(wSigmaY==wSigmaZ),"Sigma xyz, must have all or none");
    // no sens to have both individual and global sigma
    MMVII_INTERNAL_ASSERT_tiny((wSigmaX+wSigmaXYZ+IsInit(&mDefSigma) <=1 ),"Must choose between :  individual sigma,global sigma, default sigma");


/*  => JMM, JO : would be better to make a centralized check in "MatchRegex" ?
    std::regex aRegexAddInfoFree;
    try {
        aRegexAddInfoFree = std::regex(mPatternAddInfoFree);
    } catch (std::regex_error&) {
        MMVII_UserError(eTyUEr::eBadPattern,"Invalid regular expression for AddInfoFree : '" + mPatternAddInfoFree + "'");
    } catch (...) {
        throw;
    }
*/

    // coherence check ;  JMM => I changed the test, becaus it seems more logical ???  
    if (withPatternAddInfoFree  && (!withAddInfo))
       MMVII_UserError(eTyUEr::eBadOptParam,"AddInfoFree specified but no 'A' in format string");

    // Extract now as we will used it twice
    std::vector<cPt3dr> aVPts;
    for (size_t aKL=0 ; aKL<aNRFS.NbLineRead() ; aKL++)
        aVPts.push_back(aNRFS.GetPt3dr_XYZ(aKL));

    // JMM, JO : make a function, of SysCo creation, as it will be re used with conversion of orientation
    // compute output RTL if necessary
    mPhProj.InitSysCoRTLIfNotReady(cWeightAv<tREAL8,cPt3dr>::AvgCst(aVPts));

    for (size_t aKL=0 ; aKL<aNRFS.NbLineRead() ; aKL++)
    {
        std::string aNamePoint  =  aNRFS.GetStr(mFieldGCP,aKL);

	//  eventually transformate the name using specified pattern
        if (IsInit(&mPatternTransfo))
        {
            aNamePoint = ReplacePattern(mPatternTransfo.at(0),mPatternTransfo.at(1),aNamePoint);
        }

	//  Eventually  fix the number of digit (error if it's not an int)
        if (IsInit(&mNbDigName))
            aNamePoint =   ToStr(cStrIO<int>::FromStr(aNamePoint),mNbDigName);

        std::string aAdditionalInfo =   withAddInfo  ? aNRFS.GetStr(mFieldAI,aKL)  : "";

        tREAL8 aSigma = mDefSigma;
        if (wSigmaXYZ)
           aSigma = aNRFS.GetFloat(mFieldSxyz,aKL);
	//  check AddInfoFree at end, because we can have sigma in format but point is free
        if (withPatternAddInfoFree && MatchRegex(aNamePoint,mPatternAddInfoFree))
           aSigma = -1;

        cMes1GCP aMesGCP(aChSys.Value(aVPts[aKL]*mMulCoord),aNamePoint,aSigma,aAdditionalInfo);
        if (wSigmaX && (aSigma>0)) // dont use sigma if point is free
            aMesGCP.SetSigma2(aNRFS.GetPt3dr(aKL,mFieldSx,mFieldSy,mFieldSz));
        aSetM.AddMeasure(aMesGCP);
    }
    mPhProj.SaveGCP(aSetM);
    mPhProj.SaveCurSysCoGCP(aChSys.SysTarget());

    return EXIT_SUCCESS;
}

#if (0)
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
#endif


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

