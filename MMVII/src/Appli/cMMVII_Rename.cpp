#include "cMMVII_Appli.h"
#include "MMVII_Sys.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Sensor.h"

#include <regex>


namespace MMVII
{

/* ==================================================== */
/*                                                      */
/*          cAppli_DicoRename                           */
/*                                                      */
/* ==================================================== */


class cAppli_DicoRename   : public cMMVII_Appli
{
     public :
        cAppli_DicoRename  (const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli &);  ///< constructor
        int Exe() override;                                             ///< execute action
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override; ///< return spec of  mandatory args
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override; ///< return spec of optional args
     protected :
     private :
        cPhotogrammetricProject  mPhProj;
	std::string              mNameFileTxtIn;
        std::string              mFormat;
	std::vector<std::string> mPatIm;
	std::string              mNameDico;
        int                      mL0;
        int                      mLLast;
        char                     mComment;
	std::string              mSeparator;
        int                      mNbMinTieP;
	std::vector<std::string> mNameFilesListIm;

};

cAppli_DicoRename::cAppli_DicoRename(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli      (aVArgs,aSpec),
    mPhProj           (*this),
    mL0               (0),
    mLLast            (-1),
    mComment          ('#'),
    mSeparator        ("@"),
    mNbMinTieP        (0),
    mNameFilesListIm  {"AllImDicoIn.xml","AllImDicoOut.xml"}
{
}

cCollecSpecArg2007 & cAppli_DicoRename::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
              <<  Arg2007(mNameFileTxtIn ,"Name of Input File")
              <<  Arg2007(mFormat   ,"Format of file as for ex \"SNSXYZSS\" ")
              <<  Arg2007(mPatIm ,"Substitution pattern [Pattern,SubstIn,SubstOut]",{{eTA2007::ISizeV,"[3,3]"}})
              <<  Arg2007(mNameDico ,"Name for output dictionnary")
           ;
}

cCollecSpecArg2007 & cAppli_DicoRename::ArgOpt(cCollecSpecArg2007 & anArgObl)
{

    return anArgObl
       << AOpt2007(mL0,"NumL0","Num of first line to read",{eTA2007::HDV})
       << AOpt2007(mLLast,"NumLast","Num of last line to read (-1 if at end of file)",{eTA2007::HDV})
       << AOpt2007(mComment,"Com","Carac for commentary",{eTA2007::HDV})
       << AOpt2007(mNameFilesListIm,"Files","Name file to transform [Input,Output]",{{eTA2007::ISizeV,"[2,2]"},eTA2007::HDV})
       << AOpt2007(mNbMinTieP,"NbMinTiep","Number minimal of tie point for save, set -1 if save w/o tiep",{eTA2007::HDV})

       <<   mPhProj.DPMulTieP().ArgDirInOpt()
       <<   mPhProj.DPMulTieP().ArgDirOutOpt()

       <<   mPhProj.DPPointsMeasures().ArgDirInOpt()
       <<   mPhProj.DPPointsMeasures().ArgDirOutOpt()
     ;
}


int cAppli_DicoRename::Exe()
{
    mPhProj.FinishInit();

    std::vector<std::vector<std::string>> aVVNames;
    std::vector<std::vector<double>> aVNums;
    std::vector<cPt3dr> aVXYZ,aVWKP;

    ReadFilesStruct
    (
        mNameFileTxtIn, mFormat,
        mL0, mLLast, mComment,
        aVVNames,aVXYZ,aVWKP,aVNums,
        false
    );

    std::map<std::string,std::string>  aDico;
    for (auto & aVNames  : aVVNames)
    {
         std::string aCatName = aVNames.at(0);
         for (size_t aKName=1 ; aKName<aVNames.size() ; aKName++)
             aCatName = aCatName + mSeparator + aVNames.at(aKName);

	 std::string  aNameIn  = ReplacePattern(mPatIm.at(0),mPatIm.at(1),aCatName);
	 std::string  aNameOut = ReplacePattern(mPatIm.at(0),mPatIm.at(2),aCatName);

	 aDico[aNameIn] = aNameOut;

	 // StdOut()  << "DddDgkI :: " << aNameIn  << " => " << aNameOut << "\n";
    }

    SaveInFile(aDico,mNameDico);

/*
    if (IsInit(&mNameFiles))
    {
        auto aSetIn = ToVect(SetNameFromString(mNameFiles.at(0),true));
        tNameSet aSetIn;
        tNameSet aSetOut;

        for (const auto & aNameIn : aSetIn)
        {
            const auto & anIter = aDico.find(aNameIn);
            if (anIter != aDico.end())
            {
               aSetOut.Add(anIter->second);
            }
        }
        SaveInFile(aSetIn,mNameFiles.at(0));
        SaveInFile(aSetOut,mNameFiles.at(1));
    }
*/

    bool  isInitMTP = mPhProj.DPMulTieP().DirInIsInit();
    if (isInitMTP)
    {
       MMVII_INTERNAL_ASSERT_User(mPhProj.DPMulTieP().DirOutIsInit(),eTyUEr::eUnClassedError,"MulTieP In w/o Out");
    }
    tNameSet aSetIn;
    tNameSet aSetOut;
    for (const auto & [aNameIn,aNameOut] :  aDico)
    {
       bool  hasTieP =    isInitMTP
                       && ExistFile(mPhProj.DPMulTieP().FullDirIn()+ mPhProj.NameMultipleTieP(aNameIn));
       int aNbTieP = isInitMTP ? -1 : 0;
       if (hasTieP)
       {
           cVecTiePMul  aVTPM("toto");
           mPhProj.ReadMultipleTieP(aVTPM,aNameIn);
           aVTPM.mNameIm = aNameOut;
           aNbTieP = aVTPM.mVecTPM.size();
           mPhProj.SaveMultipleTieP(aVTPM,aNameOut);
       }
       else
       {
       }

       if (aNbTieP >= mNbMinTieP)
       {
           aSetIn.Add(aNameIn);
           aSetOut.Add(aNameOut);
       }
    }
    SaveInFile(aSetIn ,mNameFilesListIm.at(0));
    SaveInFile(aSetOut,mNameFilesListIm.at(1));

    if (mPhProj.DPPointsMeasures().DirInIsInit())
    {
       MMVII_INTERNAL_ASSERT_User(mPhProj.DPPointsMeasures().DirOutIsInit(),eTyUEr::eUnClassedError,"Measure In w/o Out");
       for (const auto & aPair :  aDico)
       {
           if (ExistFile(mPhProj.NameMeasureGCPIm(aPair.first,true)))
           {
              cSetMesPtOf1Im  aSMes = mPhProj.LoadMeasureIm(aPair.first);
              aSMes.SetNameIm(aPair.second);
              mPhProj.SaveMeasureIm(aSMes);
           }
       }
       mPhProj.CpGCP();
    }

    return EXIT_SUCCESS;
}

tMMVII_UnikPApli Alloc_DicoRename(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_DicoRename(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecDicoRename
(
    "UtiDicoRename",
    Alloc_DicoRename,
    "This command create a dictionnary after parsing a file, can be used for renaming",
    {eApF::Project},
    {eApDT::FileSys},
    {eApDT::FileSys},
    __FILE__
);



/* ==================================================== */
/*                                                      */
/*          cAppli_Rename                               */
/*                                                      */
/* ==================================================== */


class cAppli_Rename : public cMMVII_Appli
{
     public :
        cAppli_Rename(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli &);  ///< constructor
        int Exe() override;                                             ///< execute action
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override; ///< return spec of  mandatory args
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override; ///< return spec of optional args
     protected :
     private :
        std::vector<std::string>  Samples() const override;

        void TestSet(const std::string & aName);

        std::string               mPattern;
        std::string               mSubst;
        std::vector<std::string>  mArithmReplace;
        bool                      mDoReplace;

        std::set<std::string>     mSetOut;
};



cCollecSpecArg2007 & cAppli_Rename::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return anArgObl
            << Arg2007(mPattern,"Pattern of file to replace",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
            << Arg2007(mSubst,"Pattern of substituion")
;
}

cCollecSpecArg2007 & cAppli_Rename::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
            << AOpt2007(mDoReplace,"DoReplace","do the replacement ",{{eTA2007::HDV}})
            << AOpt2007(mArithmReplace,"AR","arthim repacement like [+,33,2,4] to add 33 to second expr and put on 4 digt ",{{eTA2007::ISizeV,"[3,4]"}})
            ;
}


cAppli_Rename::cAppli_Rename(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli (aVArgs,aSpec),
  mDoReplace   (false)
{
}

void cAppli_Rename::TestSet(const std::string & aNameOut)
{
    if (BoolFind(mSetOut,aNameOut))
    {
        MMVII_UnclasseUsEr("Proposed replacement would create a conflict");
    }
    mSetOut.insert(aNameOut);
}

std::vector<std::string>  cAppli_Rename::Samples() const
{
  return {"MMVII UtiRename \"948_(.*).JPG\" \"\\$&\" AR=[-,1,1] DoReplace=true"};
}

int cAppli_Rename::Exe()
{
    std::set<std::string> aSetStr;
    StdOut() <<  "============= Proposed replacement  ====== " << std::endl;

    std::vector<std::pair<std::string,std::string>  > aVInOut;


     std::regex aPat(mPattern);

    for (const auto & aStrIn0 : VectMainSet(0))
    {
        std::string aStrIn = aStrIn0;
        if (IsInit(&mArithmReplace))
	{
             std::string anOp = mArithmReplace[0];
	     int aOffset = cStrIO<int>::FromStr(mArithmReplace[1]);
	     int aKExpr = cStrIO<int>::FromStr(mArithmReplace[2]);

             std::smatch aBoundMatch;
             bool aGotMatch = std::regex_search(aStrIn0, aBoundMatch, aPat);
	     Fake4ReleaseUseIt(aGotMatch);
             MMVII_INTERNAL_ASSERT_tiny(aGotMatch,"cCRegex::BoundsMatch no match");

	     if ((aKExpr<0)||(aKExpr >= (int) aBoundMatch.size()))
	     {
                  MMVII_UnclasseUsEr("Num of expr incompatible with pattern : " + mArithmReplace[2]);
	     }

	     // auto aMatch  = aBoundMatch[aKExpr];

	     std::string aStrNumIn = aBoundMatch[aKExpr];
	     int aNum    = cStrIO<int>::FromStr(aStrNumIn);

	     if (anOp=="+")
                aNum += aOffset;
	     else if (anOp=="-")
                aNum -= aOffset;
	     else if (anOp=="%")
                aNum %= aOffset;
	     else
	     {
                MMVII_UnclasseUsEr("Bad operand in arithmetic : " + mArithmReplace[0]);
	     }

	     int aNbDig = (mArithmReplace.size()> 3) ? cStrIO<int>::FromStr(mArithmReplace[3]) : aStrNumIn.size();
	     std::string aStrNumOut = ToStr(aNum,aNbDig);

	     aStrIn.replace(aBoundMatch.position(aKExpr),aBoundMatch.length(aKExpr),aStrNumOut);
	}
        std::string aStrOut =  ReplacePattern(mPattern,mSubst,aStrIn);
        StdOut() << "[" << aStrIn0  << "] ";
        if (IsInit(&mArithmReplace))
           StdOut() << " ==> [" << aStrIn  << "] ";

        StdOut() << " ==> [" << aStrOut  << "]  " << std::endl;

        // TestSet(aStrIn0);
        TestSet(aStrOut);
        aVInOut.push_back(std::pair<std::string,std::string>(aStrIn0,aStrOut));
    }

    for (const auto & aPair : aVInOut)
    {
       // auto [aStrIn0,aStrOut] = aPair;
       auto aStrOut = aPair.second;
       if (ExistFile(aStrOut) && (! BoolFind(mSetOut,aStrOut)))
       {
           MMVII_UnclasseUsEr("File already exist");
       }
    }

    std::string aPrefTmp = "MMVII_Tmp_Replace_"+ PrefixGMA() + "_";

    if (mDoReplace)
    {
        // In case "input" intersect "outout", put first "input" in "tmp" file,
        for (const auto & aPair : aVInOut)
        {
            auto [aStrIn0,aStrOut] = aPair;
            StdOut() << "mv " << aStrIn0  << " " << aPrefTmp+aStrIn0  << std::endl;
	    RenameFiles(aStrIn0,aPrefTmp+aStrIn0);
        }
	// the put, safely, "tmp" in "output"
        for (const auto & aPair : aVInOut)
        {
            auto [aStrIn0,aStrOut] = aPair;
            StdOut() << "mv " << aPrefTmp+ aStrIn0  << " " << aStrOut  << std::endl;
	    RenameFiles(aPrefTmp+aStrIn0,aStrOut);
        }
    }
    return EXIT_SUCCESS;
}



tMMVII_UnikPApli Alloc_Rename(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_Rename(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecRename
(
    "UtiRename",
    Alloc_Rename,
    "This command is rename files using expr and eventually arithmetic",
    {eApF::Project},
    //  {eApF::ManMMVII, eApF::Project},  JOE ?  j'ai enleve eApF::ManMMVI, je sais plus qui l'a mis
    {eApDT::FileSys},
    {eApDT::FileSys},
    __FILE__
);

}
