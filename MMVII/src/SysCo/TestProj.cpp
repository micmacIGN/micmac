#include "MMVII_PCSens.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"
#include "MMVII_Matrix.h"

/**
   \file TestProj.cpp

   \brief appli to show proj log to check for missing grids
*/


namespace MMVII
{

   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_TestProj                            */
   /*                                                            */
   /* ********************************************************** */

class cAppli_TestProj : public cMMVII_Appli
{
public :
    cAppli_TestProj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
    int Exe() override;
    cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
    cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

    std::vector<std::string>  Samples() const override;
private :
    cPhotogrammetricProject  mPhProj;

    // Mandatory Arg
    std::string              mNameSysIn;
	std::string              mNameSysOut;

    // Optional Arg
    cPt3dr           mTestPoint;
};

cAppli_TestProj::cAppli_TestProj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli  (aVArgs,aSpec),
    mPhProj       (*this)
{
}

cCollecSpecArg2007 & cAppli_TestProj::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
            <<  Arg2007(mNameSysIn, "Input SysCo definition")
            <<  Arg2007(mNameSysOut,"Output SysCo definition")
                 ;
}

cCollecSpecArg2007 & cAppli_TestProj::ArgOpt(cCollecSpecArg2007 & anArgObl)
{
    
    return    anArgObl
              << AOpt2007(mTestPoint,"TestPoint","Point in input SysCo to check transformation",{{}})
                 ;
}


int cAppli_TestProj::Exe()
{
    mPhProj.FinishInit();
    tPtrSysCo aSysIn = mPhProj.ReadSysCo(mNameSysIn, true);
    tPtrSysCo aSysOut = mPhProj.ReadSysCo(mNameSysOut, true);
    cChangeSysCo aChSys(aSysIn,aSysOut);

    if (IsInit(&mTestPoint))
    {
        auto aTestPointOut = aChSys.Value(mTestPoint);
        auto aTestPointOutIn = aChSys.Inverse(aTestPointOut);
        StdOut() << "                SysIn                 =>                 SysOut                  =>             SysIn\n";
        StdOut() << std::fixed << std::showpoint << std::setprecision(5) << mTestPoint << "  =>  " << aTestPointOut << "  =>  " << aTestPointOutIn << "\n";
    }
    return EXIT_SUCCESS;
}


std::vector<std::string>  cAppli_TestProj::Samples() const
{
    return {"MMVII TestProj \"L93\" \"EPSG:5698\" TestPoint=[657723,6860710,0] # check if able to convert L93+height into L93+altitude"};
}



tMMVII_UnikPApli Alloc_TestProj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
    return tMMVII_UnikPApli(new cAppli_TestProj(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_TestProj
(
        "TestProj",
        Alloc_TestProj,
        "Test Proj",
        {eApF::SysCo},
        {eApDT::GCP,eApDT::SysCo},
        {eApDT::GCP,eApDT::SysCo},
        __FILE__
        );


}; // MMVII

