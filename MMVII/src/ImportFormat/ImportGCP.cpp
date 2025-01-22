#include "MMVII_PCSens.h"
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
              << mPhProj.DPGndPt3D().ArgDirOutMand()
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
    cSetMesGnd3D aSetM(mNameGCP);

    //  Extract chang of coordinate, if not set  handled by default
    cChangeSysCo & aChSys = mPhProj.ChSysCo();

    cNewReadFilesStruct aNRFS(mFormat,mSpecFormatMand,mSpecFormatTot);
    aNRFS.ReadFile(mNameFile,mParamNSF);


    bool withAddInfo  = aNRFS.FieldIsKnown(mFieldAI);
    bool withPatternAddInfoFree = IsInit(&mPatternAddInfoFree);

    bool wSigmaX       = aNRFS.FieldIsKnown(mFieldSx);
    [[maybe_unused]] bool wSigmaY       = aNRFS.FieldIsKnown(mFieldSy);     // May be unused depending of 'The_MMVII_DebugLevel'
    [[maybe_unused]] bool wSigmaZ       = aNRFS.FieldIsKnown(mFieldSz);
    bool wSigmaXYZ      = aNRFS.FieldIsKnown(mFieldSxyz);

    // too complicate to handle partiall case of fixing sigma, and btw, not pertinent ?
    MMVII_INTERNAL_ASSERT_User((wSigmaX==wSigmaY)&&(wSigmaY==wSigmaZ),eTyUEr::eUnClassedError,"Sigma xyz, must have all or none");
    // no sens to have both individual and global sigma
    MMVII_INTERNAL_ASSERT_User((wSigmaX+wSigmaXYZ+IsInit(&mDefSigma) <=1 ),eTyUEr::eUnClassedError,"Must choose between :  individual sigma,global sigma, default sigma");


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

    // coherence check
    if (withPatternAddInfoFree  && (!withAddInfo))
       MMVII_UserError(eTyUEr::eBadOptParam,"AddInfoFree specified but no 'A' in format string");

    // Extract now as we will used it twice
    std::vector<cPt3dr> aVPts;
    for (size_t aKL=0 ; aKL<aNRFS.NbLineRead() ; aKL++)
        aVPts.push_back(aNRFS.GetPt3dr_XYZ(aKL));

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
        if (withPatternAddInfoFree && MatchRegex(aAdditionalInfo,mPatternAddInfoFree))
           aSigma = -1;

        cMes1Gnd3D aMesGCP(aChSys.Value(aVPts[aKL]*mMulCoord),aNamePoint,aSigma,aAdditionalInfo);
        if (wSigmaX && (aSigma>0)) // dont use sigma if point is free
            aMesGCP.SetSigma2(aNRFS.GetPt3dr(aKL,mFieldSx,mFieldSy,mFieldSz));
        aSetM.AddMeasure3D(aMesGCP);
    }
    mPhProj.SaveGCP3D(aSetM, mPhProj.DPGndPt3D().DirOut());
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
      {eApDT::ObjCoordWorld},
      {eApDT::ObjCoordWorld},
      __FILE__
);


}; // MMVII

