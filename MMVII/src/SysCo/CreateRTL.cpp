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

/** A "Tiny" application that create a coodinate sys  local to the "Site" ,

   Create an application for that, may seem "a hammer for a fly", but this is the "price to pay"
 to warantee the respect of MMVII's naming/formating convention.
*/

class cAppli_CreateSysCoLoc : public cMMVII_Appli
{

  public :
    cAppli_CreateSysCoLoc(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
        cMMVII_Appli  (aVArgs,aSpec),
        mPhProj       (*this)
    {
    }

    cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override 
    {
        return      anArgObl
                <<  Arg2007(mStringSys ,"String for the coordinate system")
                <<  Arg2007(mOutFile ,"Name of file to save")
        ;
    }

    cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override {return anArgOpt;}

    std::vector<std::string>  Samples() const override
    {
        return {
             "MMVII SysCoCreateLocal \'RTL*6.1*46.24*0*EPSG:4326\'  CERN"
        };
    }
    int Exe() override;

private :

    cPhotogrammetricProject  mPhProj;

    // Mandatory Arg
    std::string              mStringSys;
    std::string              mOutFile;

};


int cAppli_CreateSysCoLoc::Exe()
{
    mPhProj.FinishInit();

    tPtrSysCo aSysCo = mPhProj.ReadSysCo(mStringSys);
    mPhProj.SaveSysCo(aSysCo,mOutFile);

    return EXIT_SUCCESS;
}

tMMVII_UnikPApli Alloc_CreateSysCoLoc(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
    return tMMVII_UnikPApli(new cAppli_CreateSysCoLoc(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_CreateSysCoLoc
(
        "SysCoCreateLocal",
        Alloc_CreateSysCoLoc,
        "Create a coordinate system in the standard folder",
        {eApF::SysCo},
        {eApDT::ObjCoordWorld,eApDT::Ori,eApDT::SysCo},
        {eApDT::ObjCoordWorld,eApDT::Ori,eApDT::SysCo},
        __FILE__
);

   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_CreateRTL                           */
   /*                                                            */
   /* ********************************************************** */

class cAppli_CreateRTL : public cMMVII_Appli
{
public :
    cAppli_CreateRTL(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
    int Exe() override;
    cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
    cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

    std::vector<std::string>  Samples() const override;
private :

    cPhotogrammetricProject  mPhProj;

    // Mandatory Arg
    std::string              mSpecIm;
    std::string              mDefSysCoIn;
    std::string              mDefSysCoOut;

    // Optional Arg
    cPt3dr           mOrigin;
    tREAL8           mZ0;
    tREAL8           mEpsDer ;
};

cAppli_CreateRTL::cAppli_CreateRTL(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli  (aVArgs,aSpec),
    mPhProj       (*this),
    mEpsDer       (200.0)
{
}

cCollecSpecArg2007 & cAppli_CreateRTL::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
            <<  Arg2007(mSpecIm ,"Name of Input File",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
             <<  Arg2007(mDefSysCoOut ,"Output coordinate system name")
                 ;
}

cCollecSpecArg2007 & cAppli_CreateRTL::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    
    return    anArgObl
              << mPhProj.DPOrient().ArgDirInOpt()
            //  <<  mPhProj.DPOrient().ArgDirOutOpt()
              << mPhProj.DPGndPt3D().ArgDirInOpt()
              << AOpt2007(mOrigin,"Origin","Force origin of RTL Measures",{{}})
              << AOpt2007(mZ0,"Z0","Force altitute of RTL Measures",{{}})
              << AOpt2007(mEpsDer,"EpsDer","Epislon 4 computing derivative",{{eTA2007::HDV}})
              << AOpt2007(mDefSysCoIn ,"SysIn" ,"Input coordinate system (default from GCP or Orient)")
                 ;
}


int cAppli_CreateRTL::Exe()
{
    mPhProj.FinishInit();

    cWeightAv<tREAL8,cPt3dr> aAvgSens;
    bool isInitSens  =false;
    bool isInitGCP  =false;

    std::string aDefSysCoIn;
    if (mPhProj.DPOrient().DirInIsInit())
    {
        mPhProj.DPOrient().CheckDirExists(true, true);
        auto aSysIn = mPhProj.CurSysCoOri(true);
        if (aSysIn.get())
            aDefSysCoIn = aSysIn->Def();

        for (const auto & aNameIm : VectMainSet(0))
        {
            cSensorImage* aSI = mPhProj. ReadSensor(aNameIm,true,false);
            cPt3dr aCenter;
            const cPt3dr * aCOfFP = aSI->CenterOfFootPrint();
            if (aCOfFP)
                aCenter = *aCOfFP;
            else
                aCenter = aSI->PseudoCenterOfProj();  // should always exist, but bot 100% implemanted 4 now

            aAvgSens.Add(1.0,aCenter);
            isInitSens = true;
        }
    }

    cSetMesGndPt aMesIm;
    cWeightAv<tREAL8,cPt3dr> aAvgGCP;
    if (mPhProj.DPGndPt3D().DirInIsInit())
    {
        auto aSysIn = mPhProj.CurSysCoGCP(true);
        if (aSysIn.get())
            aDefSysCoIn = aSysIn->Def();

        mPhProj.LoadGCP3D(aMesIm);
        for (const auto & aGCP : aMesIm.MesGCP())
        {
            aAvgGCP.Add(1,aGCP.mPt);
            isInitGCP = true;
        }
    }
    SetIfNotInit(mDefSysCoIn,aDefSysCoIn);

    if (! IsInit(&mOrigin))
    {
        MMVII_INTERNAL_ASSERT_User(isInitSens||isInitGCP,eTyUEr::eUnClassedError,"No data for init center");

        const cWeightAv<tREAL8,cPt3dr>	 &  aAvgXY  = isInitSens ? aAvgSens : aAvgGCP ;
        const cWeightAv<tREAL8,cPt3dr>	 &  aAvgZ   = isInitGCP ?  aAvgGCP  : aAvgSens;

        mOrigin.x() = aAvgXY.Average().x();
        mOrigin.y() = aAvgXY.Average().y();
        mOrigin.z() =  aAvgZ.Average().z();
    }

    if (IsInit(&mZ0))
        mOrigin.z() = mZ0;


    tPtrSysCo aSysRTL = mPhProj.CreateSysCoRTL(mOrigin,mDefSysCoIn);

    mPhProj.SaveSysCo(aSysRTL,mDefSysCoOut);


    return EXIT_SUCCESS;
}


std::vector<std::string>  cAppli_CreateRTL::Samples() const
{
    return {"MMVII SysCoCreateRTL AllIM.xml RTL InOri=SatWGS84 z0=0"};
}



tMMVII_UnikPApli Alloc_CreateRTL(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
    return tMMVII_UnikPApli(new cAppli_CreateRTL(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_CreateRTL
(
        "SysCoCreateRTL",
        Alloc_CreateRTL,
        "Create RTL (local tangent repair)",
        {eApF::SysCo},
        {eApDT::ObjCoordWorld,eApDT::Ori,eApDT::SysCo},
        {eApDT::ObjCoordWorld,eApDT::Ori,eApDT::SysCo},
        __FILE__
);

}; // MMVII

