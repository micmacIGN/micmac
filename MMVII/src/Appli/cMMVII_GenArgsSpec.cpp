#include "cMMVII_Appli.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_GenArgsSpec.h"


namespace MMVII
{

namespace GenArgsInternal
{
static const std::vector<eTA2007> prjSubDirList =                           // clazy:exclude=non-pod-global-static
{
    eTA2007::Orient,
    eTA2007::RadiomData,
    eTA2007::RadiomModel,
    eTA2007::MeshDev,
    eTA2007::Mask,
    eTA2007::MetaData,
    eTA2007::PointsMeasure,
    eTA2007::TieP,
    eTA2007::MulTieP,
    eTA2007::RigBlock,
    eTA2007::SysCo,
};

static const std::map<eTA2007,std::vector<std::string>> fileList =          // clazy:exclude=non-pod-global-static
{
    {eTA2007::FileImage,{".tif",".tiff",".jpg",".jpeg",".png",".cr2",".crw",".nef"}},
    {eTA2007::FileCloud,{".ply"}},
    {eTA2007::File3DRegion,{".*"}},
};

} // namespace GenArgsInternal

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
    mArgsSpecs(GenArgsInternal::prjSubDirList, GenArgsInternal::fileList),
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

    mArgsSpecs.jsonSpec = "{\n";
    mArgsSpecs.jsonSpec += "  \"config\": {\n";
    mArgsSpecs.jsonSpec += "    \"DirBin2007\":\"" + DirBinMMVII() + "\",\n";
    mArgsSpecs.jsonSpec += "    \"Bin2007\":\"" + FullBin() + "\",\n";

    mArgsSpecs.jsonSpec += "    \"MMVIIDirPhp\":\"" + MMVII_DirPhp + "\",\n";
    mArgsSpecs.jsonSpec += "    \"MMVIITestDir\":\"" + MMVIITestDir + "\",\n";

    mArgsSpecs.jsonSpec += "    \"eTa2007FileTypes\": [";
    bool first = true;
    for (const auto& [aTA2007,anExtList]  : mArgsSpecs.fileTypes) {
        if (! first)
            mArgsSpecs.jsonSpec += ",";
        first = false;
        mArgsSpecs.jsonSpec += "\"" + E2Str(aTA2007) + "\"";
    }
    mArgsSpecs.jsonSpec += "],\n";

    mArgsSpecs.jsonSpec += "    \"eTa2007DirTypes\": [";
    first = true;
    for (const auto &aTA2007 : mArgsSpecs.prjSubDirList) {
        if (! first)
            mArgsSpecs.jsonSpec += ",";
        first = false;
        mArgsSpecs.jsonSpec += "\"" + E2Str(aTA2007) + "\"";
    }
    mArgsSpecs.jsonSpec += "],\n";

    mArgsSpecs.jsonSpec += "    \"extensions\": {" ;
    first = true;
    for (const auto& [aTA2007,anExtList]  : mArgsSpecs.fileTypes) {
        if (! first)
            mArgsSpecs.jsonSpec += ",";
        first = false;
        mArgsSpecs.jsonSpec += "\n      \"" + E2Str(aTA2007) + "\": [" ;
        bool firstExt = true;
        for (const auto& anExt : anExtList) {
            if (!firstExt)
                mArgsSpecs.jsonSpec += ",";
            firstExt = false;
            mArgsSpecs.jsonSpec += "\n        \"" + anExt + "\"" ;
        }
        mArgsSpecs.jsonSpec += "\n      ]" ;
    }
    mArgsSpecs.jsonSpec += "\n    }\n" ;  // Extensions
    mArgsSpecs.jsonSpec += "  },\n" ;     // Config

    mArgsSpecs.jsonSpec += "  \"applets\": [\n";

    first = true;
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
            mArgsSpecs.jsonSpec += ",\n";
        first = false;
        aVArgs[1] = aSpec->Name();
        tMMVII_UnikPApli anAppli = aSpec->Alloc()(aVArgs,*aSpec);
        anAppli->SetNot4Exe();
        anAppli->InitParam(&mArgsSpecs);
    }
    mArgsSpecs.jsonSpec += "\n  ]\n";
    mArgsSpecs.jsonSpec += "}\n";

    aOfs << mArgsSpecs.jsonSpec;

    if (!mQuiet)
        aEfs << mArgsSpecs.errors;

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
