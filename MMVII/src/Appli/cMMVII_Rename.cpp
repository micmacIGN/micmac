#include "cMMVII_Appli.h"
#include "MMVII_Sys.h"
#include "MMVII_DeclareCste.h"


#include <regex.h>


namespace MMVII
{

// int regexec(const regex_t *preg, const char *string, size_t nmatch, regmatch_t *pmatch, int eflags);
void FFF(const std::string & aPattern)
{
   size_t aNbPar = std::count(aPattern.begin(), aPattern.end(), '('); //)
FakeUseIt(aNbPar);
   regex_t    preg;
   regcomp(&preg, nullptr, 0);
   regexec(&preg,nullptr,1,nullptr,33);
}

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
        std::set<std::string>     mSet;
        bool                      mDoReplace;
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
            << AOpt2007(mArithmReplace,"AR","arthim repacement like [+,2,33,4] to add 33 to second expr and put on 4 digt ",{{eTA2007::ISizeV,"[3,4]"}})
            ;
}


cAppli_Rename::cAppli_Rename(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli (aVArgs,aSpec),
  mDoReplace   (false)
{
}

void cAppli_Rename::TestSet(const std::string & aName)
{
    if (BoolFind(mSet,aName))
    {
        MMVII_UnclasseUsEr("Proposed replacement would create a conflict");
    }
    mSet.insert(aName);
}

int cAppli_Rename::Exe()
{
    std::set<std::string> aSetStr;
    StdOut() <<  "============= Proposed replacement  ====== \n";

    std::vector<std::pair<std::string,std::string>  > aVInOut;


    for (const auto & aStrIn : VectMainSet(0))
    {
        std::string aStrOut =  ReplacePattern(mPattern,mSubst,aStrIn);
        StdOut() << "[" << aStrIn  << "] ==> [" << aStrOut  << "]\n";

        TestSet(aStrIn);
        TestSet(aStrOut);
        aVInOut.push_back(std::pair<std::string,std::string>(aStrIn,aStrOut));
    }

    if (mDoReplace)
    {
        for (const auto & aPair : aVInOut)
        {
            auto [aStrIn,aStrOut] = aPair;
            StdOut() << "[" << aStrIn  << "] MMMMMVVVVVV [" << aStrOut  << "]\n";
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
