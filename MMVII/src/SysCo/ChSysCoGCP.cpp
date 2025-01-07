#include "MMVII_PCSens.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"
#include "MMVII_Matrix.h"

/**
   \file cConvCalib.cpp  testgit

   \brief file for conversion between calibration (change format, change model) and tests
*/


namespace MMVII
{

   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_ImportGCP                           */
   /*                                                            */
   /* ********************************************************** */

class cAppli_ChSysCoGCP : public cMMVII_Appli
{
     public :
        cAppli_ChSysCoGCP(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        std::vector<std::string>  Samples() const override;
     private :

	cPhotogrammetricProject  mPhProj;

	// Mandatory Arg
	std::string              mSpecIm;
	std::string              mNameSysIn;
	std::string              mNameSysOut;

};

cAppli_ChSysCoGCP::cAppli_ChSysCoGCP(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this)
{
}

cCollecSpecArg2007 & cAppli_ChSysCoGCP::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mNameSysOut ,"Output SysCo definition")
              <<  mPhProj.DPGndPt3D().ArgDirInMand()
              <<  mPhProj.DPGndPt3D().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_ChSysCoGCP::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    
    return      anArgObl
            << AOpt2007(mNameSysIn,"SysIn","Input SysCo definition, default=found in CurSysCo.xml (if exists)")
    ;
}


int cAppli_ChSysCoGCP::Exe()
{
    mPhProj.FinishInit();

    std::vector<std::string>  aListNameGCPIn = mPhProj.ListFileGCP("");

    tPtrSysCo aSysOut = mPhProj.ReadSysCo(mNameSysOut);

    tPtrSysCo aSysIn (nullptr);
    if (IsInit(&mNameSysIn))
       aSysIn = mPhProj.ReadSysCo(mNameSysIn);
    else
       aSysIn = mPhProj.CurSysCoGCP();


    // StdOut() << "HHHHHH " << mNameSysOut << " " << mNameSysIn << "\n";

    cChangeSysCo aChSys(aSysIn,aSysOut);


    for (const auto & aNameGPCIN : aListNameGCPIn)
    {
    // StdOut() << "KKKKKKK " <<   aChSys.Value(cPt3dr(0,0,0)) << aChSys.Inverse(cPt3dr(0,0,0))  << "\n";
        cSetMesGnd3D aMesGCP = cSetMesGnd3D::FromFile(aNameGPCIN);
        aMesGCP.ChangeCoord(aChSys);
        mPhProj.SaveGCP3D(aMesGCP, mPhProj.DPGndPt3D().DirOut());
    }

    // copy the System of coordinate in the GCP-folder
    mPhProj.SaveCurSysCoGCP(aSysOut);

    return EXIT_SUCCESS;
}


std::vector<std::string>  cAppli_ChSysCoGCP::Samples() const
{
   return {"MMVII SysCoCreateRTL "};
}



tMMVII_UnikPApli Alloc_ChSysCoGCP(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ChSysCoGCP(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ChSysCoGCP
(
     "GCPChSysCo",
      Alloc_ChSysCoGCP,
      "Change SysCo of GCP",
      {eApF::SysCo,eApF::Ori},
      {eApDT::Ori,eApDT::SysCo},
      {eApDT::Ori,eApDT::SysCo},
      __FILE__
);


}; // MMVII

