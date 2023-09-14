#include "cMMVII_Appli.h"
#include "MMVII_Sys.h"
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
     protected :
     private :
        std::string mSpecFileName;
};



cCollecSpecArg2007 & cAppli_GenArgsSpec::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return anArgObl;
}

cCollecSpecArg2007 & cAppli_GenArgsSpec::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
            << AOpt2007(mSpecFileName,"Out","Destination file",{eTA2007::Output,eTA2007::HDV})
            ;
}


cAppli_GenArgsSpec::cAppli_GenArgsSpec(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli (aVArgs,aSpec)
{
    mSpecFileName = DirBinMMVII() + "MMVII_argsspec.json";
}

int cAppli_GenArgsSpec::Exe()
{
    std::vector<std::string> appletsWithGUI;
    std::vector<std::string> appletsWithoutGUI;

    std::vector<std::string> aVArgs;
    aVArgs.push_back(mArgv[0]);             // MMV2
    aVArgs.push_back(mArgv[1]);             // will be replaced by anAppli name


    cMMVII_Ofs  aOfs(mSpecFileName, eFileModeOut::CreateText);

    StdOut() << "Generating specifications in :\n";
    StdOut() << "  " << mSpecFileName << "\n\n";


    aOfs.Ofs() << "{\n";
    aOfs.Ofs() << "  \"config\": {\n";
    aOfs.Ofs() << "    \"DirBin2007\":\"" + DirBinMMVII() + "\",\n";
    aOfs.Ofs() << "    \"Bin2007\":\"" + FullBin() + "\",\n";

    aOfs.Ofs() << "    \"MMVIIDirOrient\":\"" + MMVIIDirOrient + "\",\n";
    aOfs.Ofs() << "    \"MMVIIDirHomol\":\"" + MMVIIDirHomol + "\",\n";
    aOfs.Ofs() << "    \"MMVIIDirMeshDev\":\"" + MMVIIDirMeshDev + "\",\n";
    aOfs.Ofs() << "    \"MMVIIDirRadiom\":\"" + MMVIIDirRadiom + "\",\n";
    aOfs.Ofs() << "    \"MMVIITestDir\":\"" + MMVIITestDir + "\",\n";


    aOfs.Ofs() << "    \"extensions\": {" ;

    bool firstEta = true;
    for (const auto& [anETA2077,anExtList] : MMVIISupportedFilesExt) {
        if (!firstEta)
            aOfs.Ofs() << ",";
        aOfs.Ofs() << "\n";
        firstEta = false;
        aOfs.Ofs() << "      \"" << E2Str(anETA2077) << "\": [" ;
        bool firstExt = true;
        for (const auto& anExt : anExtList) {
            if (!firstExt)
                aOfs.Ofs() << ",";
            aOfs.Ofs() << "\n";
            firstExt = false;
            aOfs.Ofs() << "        \"" << anExt << "\"" ;
        }
        aOfs.Ofs() << "\n      ]" ;
    }
    aOfs.Ofs() << "\n    }\n" ;  // Extensions
    aOfs.Ofs() << "  },\n" ;     // Config

    aOfs.Ofs() << "  \"applets\": [\n";

    bool first = true;
    for (const auto & aSpec : cSpecMMVII_Appli::VecAll())
    {
        bool gui = true;
        for (auto& feature: aSpec->Features()) {
            if (feature == eApF::NoGui) {
                gui= false;
                break;
            }
        }
        if (gui)
            appletsWithGUI.push_back(aSpec->Name());
        else
            appletsWithoutGUI.push_back(aSpec->Name());

        std::string argDesc;
        if (!first)
            aOfs.Ofs()<<",\n";
        first = false;
        aVArgs[1] = aSpec->Name();
        tMMVII_UnikPApli anAppli = aSpec->Alloc()(aVArgs,*aSpec);
        anAppli->SetNot4Exe();
        anAppli->InitParam(&argDesc);
        aOfs.Ofs() <<  argDesc;
    }
    aOfs.Ofs() << "\n  ]\n";
    aOfs.Ofs() << "}\n";
    aOfs.Ofs().close();

    StdOut() << "Specifications with GUI generated for:\n";
    StdOut() << "  " << appletsWithGUI << "\n\n";
    StdOut() << "Specifications with NO GUI generated for:\n";
    StdOut() << "  " << appletsWithoutGUI << "\n";
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
