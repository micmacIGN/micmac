#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Topo.h"

#include "../src/Topo/ctopodata.h"

namespace MMVII
{

/* ********************************************************** */
/*                                                            */
/*                 cAppli_ImportOBS                           */
/*                                                            */
/* ********************************************************** */

class cAppli_ImportOBS : public cMMVII_Appli
{
public :
    cAppli_ImportOBS(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
    int Exe() override;
    cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
    cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

    std::vector<std::string>  Samples() const override;
private :

    cPhotogrammetricProject  mPhProj;

    // Mandatory Arg
    std::string              mNameObsFile;
    std::string              mNameObsDir;
};

cAppli_ImportOBS::cAppli_ImportOBS(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli  (aVArgs,aSpec),
    mPhProj       (*this)
{
}

cCollecSpecArg2007 & cAppli_ImportOBS::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
             << Arg2007(mNameObsFile ,"Name of Obs Input File",{eTA2007::FileAny})
             << mPhProj.DPTopoMes().ArgDirOutMand()
                 ;
}

cCollecSpecArg2007 & cAppli_ImportOBS::ArgOpt(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
               ;
}


int cAppli_ImportOBS::Exe()
{
    mPhProj.FinishInit();

    //cBA_Topo aBA_Topo(nullptr, nullptr);
    cTopoData aAllTopoDataIn;
    std::string aPost = Postfix(mNameObsFile,'.',true);
    if (UCaseEqual(aPost,"obs"))
    {
        aAllTopoDataIn.InsertCompObsFile( mNameObsFile );
    } else {
        MMVII_INTERNAL_ASSERT_User(false, eTyUEr::eUnClassedError,
                                   "Error: obs file has not the correct \".obs\" extension")
    }
    //mPhProj.SaveTopoMes(*mTopo);
    std::string aInputFileDir;
    std::string aInputFileFile;
    SplitDirAndFile(aInputFileDir, aInputFileFile, mNameObsFile, true);
    CopyFile(mNameObsFile,  mPhProj.DPTopoMes().FullDirOut() + aInputFileFile);
    StdOut() << mNameObsFile << " copied into " << mPhProj.DPTopoMes().FullDirOut() << "\n";

    return EXIT_SUCCESS;
}


std::vector<std::string>  cAppli_ImportOBS::Samples() const
{
    return
    {
        "MMVII ImportOBS  toto.obs  Toto1"
    };
}



tMMVII_UnikPApli Alloc_ImportOBS(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
    return tMMVII_UnikPApli(new cAppli_ImportOBS(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ImportOBS
(
        "ImportOBS",
        Alloc_ImportOBS,
        "Import Obs file in MMVII project",
        {eApF::Topo},
        {eApDT::Topo},
        {eApDT::Topo},
        __FILE__
        );


}; // MMVII

