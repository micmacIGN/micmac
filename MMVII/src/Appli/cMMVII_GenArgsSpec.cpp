#include "cMMVII_Appli.h"
#include "MMVII_DeclareCste.h"


namespace MMVII
{


/* ==================================================== */
/*                                                      */
/*          cAppli_GenArgsSpec                          */
/*                                                      */
/* ==================================================== */


class cAppli_GenArgsSpec : public cMMVII_Appli
{
     public :
        cAppli_GenArgsSpec(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli &);  ///< constructor
        int Exe() override;                                             ///< execute action
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override; ///< return spec of  mandatory args
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override; ///< return spec of optional args
    private:
        std::string mSpecFileName;
};



cCollecSpecArg2007 & cAppli_GenArgsSpec::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return anArgObl;
}

cCollecSpecArg2007 & cAppli_GenArgsSpec::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return anArgOpt
          << AOpt2007(mSpecFileName,"Out","Destination file",{eTA2007::Output,eTA2007::HDV});
}


cAppli_GenArgsSpec::cAppli_GenArgsSpec(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli (aVArgs,aSpec)
{
}

int cAppli_GenArgsSpec::Exe()
{
    std::vector<std::string> appletsWithGUI;
    std::vector<std::string> appletsWithoutGUI;

    std::string aDesc;
    std::string aErrors;

    cMultipleOfs aEfs(std::cerr);
    cMultipleOfs aOfs(std::cout);
    std::unique_ptr<cMMVII_Ofs> aOFile;

    if (mSpecFileName.size() != 0) {
        aOfs.Clear();
        aOFile.reset(new cMMVII_Ofs(mSpecFileName,eFileModeOut::CreateText));
        aOfs.Add(aOFile->Ofs());
    }

    std::vector<std::string> aVArgs;
    aVArgs.push_back(mArgv[0]);             // MMV2
    aVArgs.push_back(mArgv[1]);             // will be replaced by anAppli name


    aEfs << "Generating command line specifications ...\n\n";

    aDesc  = "{\n";
    aDesc += "  \"config\": {\n";
    aDesc += "    \"DirBin2007\":\"" + DirBinMMVII() + "\",\n";
    aDesc += "    \"Bin2007\":\"" + FullBin() + "\",\n";

    aDesc += "    \"MMVIIDirPhp\":\"" + MMVII_DirPhp + "\",\n";
    aDesc += "    \"MMVIITestDir\":\"" + MMVIITestDir + "\",\n";


    aDesc += "    \"extensions\": {" ;

    bool firstEta = true;
    for (const auto& [anETA2077,anExtList] : MMVIISupportedFilesExt) {
        if (!firstEta)
            aDesc += ",";
        aDesc += "\n";
        firstEta = false;
        aDesc += "      \"" + E2Str(anETA2077) + "\": [" ;
        bool firstExt = true;
        for (const auto& anExt : anExtList) {
            if (!firstExt)
                aDesc += ",";
            aDesc += "\n";
            firstExt = false;
            aDesc += "        \"" + anExt + "\"" ;
        }
        aDesc += "\n      ]" ;
    }
    aDesc += "\n    }\n" ;  // Extensions
    aDesc += "  },\n" ;     // Config

    aDesc += "  \"applets\": [\n";

    bool first = true;
    for (const auto & aSpec : cSpecMMVII_Appli::VecAll())
    {
        bool gui = true;
        for (auto& feature: aSpec->Features()) {
            if (feature == eApF::NoGui) {
                gui = false;
                break;
            }
        }
        if (gui)
            appletsWithGUI.push_back(aSpec->Name());
        else
            appletsWithoutGUI.push_back(aSpec->Name());

        if (!first)
            aDesc += ",\n";
        first = false;
        aVArgs[1] = aSpec->Name();
        tMMVII_UnikPApli anAppli = aSpec->Alloc()(aVArgs,*aSpec);
        anAppli->SetNot4Exe();
        anAppli->InitParam(&aDesc, &aErrors);
    }
    aDesc += "\n  ]\n";
    aDesc += "}\n";

    aOfs << aDesc;

    aEfs << aErrors;
    aEfs << "Specifications with GUI generated for:\n";
    aEfs << "  " << appletsWithGUI << "\n\n";
    aEfs << "Specifications with NO GUI generated for:\n";
    aEfs << "  " << appletsWithoutGUI << "\n";
    return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_GenArgsSpec(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_GenArgsSpec(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecGenArgsSpec
(
    "GenArgsSpec",
    Alloc_GenArgsSpec,
    "This command is used to generate arguments specifications",
    {eApF::ManMMVII, eApF::NoGui},
    {eApDT::ToDef},
    {eApDT::ToDef},
    __FILE__
);

}
