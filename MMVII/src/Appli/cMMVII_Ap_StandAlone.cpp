#include "include/MMVII_all.h"

namespace MMVII {

class cDummyAppli : public cMMVII_Appli
{
public :
    cDummyAppli(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &aSpec)
        : cMMVII_Appli(aVArgs,aSpec)
    {
        SetNot4Exe();
    }
    int Exe() override
    {
        return EXIT_SUCCESS;
    }
    cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override
    {
        return anArgObl;
    }
    cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override
    {
        return anArgOpt;
    }
};

static tMMVII_UnikPApli Alloc_DummyAppli(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cDummyAppli(aVArgs,aSpec));
}

void InitStandAloneAppli(const char* aAppName, const char *aComment)
{
    static cSpecMMVII_Appli TheSpecDummyAppli(
                aAppName,
                Alloc_DummyAppli,
                aComment,
                {eApF::Test},
                {eApDT::None},
                {eApDT::None},
                __FILE__
            );
    static cDummyAppli cDummyAppli({},TheSpecDummyAppli);
}

int InitStandAloneAppli(const cSpecMMVII_Appli & aSpec, int argc, char*argv[])
{
    std::vector<std::string> aVArgs;
    for (int aK=0 ; aK<argc; aK++)
        aVArgs.push_back(argv[aK]);
    return aSpec.AllocExecuteDestruct(aVArgs);
}

} // namespace MMVII
