#include "cMMVII_Appli.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_GenArgsSpec.h"


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
        cGenArgsSpecContext mArgsSpecs;
        std::string mSpecFileName;
        bool mQuiet;
        bool mNoInfo;
};


// !!! Modify mArgsSpecs if eTA2007 changes !!
cAppli_GenArgsSpec::cAppli_GenArgsSpec(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli (aVArgs,aSpec),
  mArgsSpecs(eTA2007::FileImage,eTA2007::File3DRegion,eTA2007::Orient,eTA2007::MulTieP),
    mQuiet(false),mNoInfo(false)
{
}


cCollecSpecArg2007 & cAppli_GenArgsSpec::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return anArgObl;
}

cCollecSpecArg2007 & cAppli_GenArgsSpec::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return anArgOpt
          << AOpt2007(mSpecFileName,"Out","Destination file",{eTA2007::Output,eTA2007::HDV})
          << AOpt2007(mNoInfo,"NoInfo","if True, don't ouput informations",{eTA2007::HDV})
          << AOpt2007(mQuiet,"Quiet","if True, don't ouput errors/informations",{eTA2007::HDV});
}



int cAppli_GenArgsSpec::Exe()
{
    cGenArgsSpecContext aArgsSpecs(eTA2007::FileImage,eTA2007::File3DRegion,eTA2007::Orient,eTA2007::MulTieP);

    std::vector<std::string> appletsWithGUI;
    std::vector<std::string> appletsWithoutGUI;

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

    if (! (mQuiet || mNoInfo))
        aEfs << "Generating command line specifications ...\n\n";

    aArgsSpecs.jsonSpec = "{\n";
    aArgsSpecs.jsonSpec += "  \"config\": {\n";
    aArgsSpecs.jsonSpec += "    \"DirBin2007\":\"" + DirBinMMVII() + "\",\n";
    aArgsSpecs.jsonSpec += "    \"Bin2007\":\"" + FullBin() + "\",\n";

    aArgsSpecs.jsonSpec += "    \"MMVIIDirPhp\":\"" + MMVII_DirPhp + "\",\n";
    aArgsSpecs.jsonSpec += "    \"MMVIITestDir\":\"" + MMVIITestDir + "\",\n";

    aArgsSpecs.jsonSpec += "    \"eTa2007FileTypes\": [";
    for (eTA2007 aTA2007 = aArgsSpecs.firstFileType; aTA2007 <= aArgsSpecs.lastFileType;) {
        if (aTA2007 != aArgsSpecs.firstFileType)
            aArgsSpecs.jsonSpec += ",";
        aArgsSpecs.jsonSpec += "\"" + E2Str(aTA2007) + "\"";
        aTA2007 = static_cast<eTA2007>(static_cast<int>(aTA2007) + 1);
    }
    aArgsSpecs.jsonSpec += "],\n";

    aArgsSpecs.jsonSpec += "    \"eTa2007DirTypes\": [";
    for (eTA2007 aTA2007 = aArgsSpecs.firstDirType; aTA2007 <= aArgsSpecs.lastDirType;) {
        if (aTA2007 != aArgsSpecs.firstDirType)
            aArgsSpecs.jsonSpec += ",";
        aArgsSpecs.jsonSpec += "\"" + E2Str(aTA2007) + "\"";
        aTA2007 = static_cast<eTA2007>(static_cast<int>(aTA2007) + 1);
    }
    aArgsSpecs.jsonSpec += "],\n";

    aArgsSpecs.jsonSpec += "    \"extensions\": {" ;
    bool firstEta = true;
    for (const auto& [aTA2007,anExtList] : MMVIISupportedFilesExt) {
        if (!firstEta)
            aArgsSpecs.jsonSpec += ",";
        aArgsSpecs.jsonSpec += "\n";
        firstEta = false;
        aArgsSpecs.jsonSpec += "      \"" + E2Str(aTA2007) + "\": [" ;
        bool firstExt = true;
        for (const auto& anExt : anExtList) {
            if (!firstExt)
                aArgsSpecs.jsonSpec += ",";
            aArgsSpecs.jsonSpec += "\n";
            firstExt = false;
            aArgsSpecs.jsonSpec += "        \"" + anExt + "\"" ;
        }
        aArgsSpecs.jsonSpec += "\n      ]" ;
    }
    aArgsSpecs.jsonSpec += "\n    }\n" ;  // Extensions
    aArgsSpecs.jsonSpec += "  },\n" ;     // Config

    aArgsSpecs.jsonSpec += "  \"applets\": [\n";

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
            aArgsSpecs.jsonSpec += ",\n";
        first = false;
        aVArgs[1] = aSpec->Name();
        tMMVII_UnikPApli anAppli = aSpec->Alloc()(aVArgs,*aSpec);
        anAppli->SetNot4Exe();
        anAppli->InitParam(&aArgsSpecs);
    }
    aArgsSpecs.jsonSpec += "\n  ]\n";
    aArgsSpecs.jsonSpec += "}\n";

    aOfs << aArgsSpecs.jsonSpec;

    if (!mQuiet)
        aEfs << aArgsSpecs.errors;

    if (! (mQuiet || mNoInfo)) {
        aEfs << "\nSpecifications with GUI generated for:\n";
        aEfs << "  " << appletsWithGUI << "\n\n";
        aEfs << "Specifications with NO GUI generated for:\n";
        aEfs << "  " << appletsWithoutGUI << "\n";
    }
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

} // namespace MMVII
