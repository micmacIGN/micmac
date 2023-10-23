#include "cMMVII_Appli.h"
#include "MMVII_Sys.h"
#include "MMVII_DeclareCste.h"


#include <regex>


namespace MMVII
{

/* ==================================================== */
/*                                                      */
/*          cAppli_Rename                          */
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

int cAppli_Rename::Exe()
{
    std::set<std::string> aSetStr;
    StdOut() <<  "============= Proposed replacement  ====== \n";

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

        StdOut() << " ==> [" << aStrOut  << "]  \n";

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
            StdOut() << "mv " << aStrIn0  << " " << aPrefTmp+aStrIn0  << "\n";
	    RenameFiles(aStrIn0,aPrefTmp+aStrIn0);
        }
	// the put, safely, "tmp" in "output"
        for (const auto & aPair : aVInOut)
        {
            auto [aStrIn0,aStrOut] = aPair;
            StdOut() << "mv " << aPrefTmp+ aStrIn0  << " " << aStrOut  << "\n";
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
    {eApF::ManMMVII, eApF::Project},
    {eApDT::FileSys},
    {eApDT::FileSys},
    __FILE__
);

}
