#include "cMMVII_Appli.h"
#include "MMVII_PCSens.h"
#include "MMVII_Geom2D.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Triangles.h"
#include "MMVII_Image2D.h"
#include "MMVII_ZBuffer.h"
#include "MeshDev.h"
#include "MMVII_Sys.h"
#include "MMVII_Radiom.h"
#include <fstream>
#include <iostream>
#include <StdAfx.h>



namespace MMVII
{
 class cAppliNuageBascule : public cMMVII_Appli,
                            public cAppliParseBoxIm<tREAL4>
{
    private:
        int Exe() override;
        int ExeOnParsedBox() override;

        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        std::string mNameCloud2D_DepthIn;
        std::string mNameIm;
        std::string mNameMasq1;
        std::string mNameCorrelIm;

        cSensorCamPC * mCamPC;
        cSensorCamPC * mSecCamPC;
        cIm2D<tREAL4> mImPx1;
        cIm2D<tU_INT1> mImMasq1;
        cIm2D<tREAL4> mImRed;

        // photogrammetric project
        cPhotogrammetricProject mPhProj;
        cTriangulation3D<tREAL8> * mTri3D;
        std::string mNameResult;
        eModeGeom mModeGeom;
    public:
        cAppliNuageBascule(const std::vector<std::string> & aVArgs, const cSpecMMVII_Appli & aSpec );
        typedef cAppliParseBoxIm<tREAL4> tAPBI;
        typedef tAPBI::tIm               tImAPBI;
        typedef tAPBI::tDataIm           tDImAPBI;
        std::string Name_Epip(std::string & aName);
        void MakeDepthImage();
        void MakeBascule();

};


cAppliNuageBascule::cAppliNuageBascule(const std::vector<std::string> & aVArgs, const cSpecMMVII_Appli & aSpec ):
    cMMVII_Appli(aVArgs,aSpec),
    cAppliParseBoxIm<tREAL4>(*this,eForceGray::No,cPt2di(2000,2000),cPt2di(150,50),true),
    mCamPC(nullptr),
    mSecCamPC(nullptr),
    mImPx1(cPt2di(1,1)),
    mImMasq1(cPt2di(1,1)),
    mImRed(cPt2di(1,1)),
    mPhProj(*this),
    mTri3D(nullptr),
    mNameResult("Bascule_"),
    mModeGeom(eModeGeom::eGEOM_EPIP)
{
}


cCollecSpecArg2007 & cAppliNuageBascule::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
           <<   Arg2007(mNameIm, "Name of master image to get camera", {{eTA2007::MPatFile,"0"},eTA2007::FileDirProj} )
           <<   Arg2007(mNameCloud2D_DepthIn,"Name of input depth map", {{eTA2007::MPatFile,"0"},eTA2007::FileDirProj} )
           <<   mPhProj.DPOrient().ArgDirInMand()
           <<   mPhProj.DPMeshDev().ArgDirOutMand()
        ;
}


cCollecSpecArg2007 & cAppliNuageBascule::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return anArgOpt
           << AOpt2007(mModeGeom,"ModeGeom","Either epipolar geometry, def=geometry of input depth",{AC_ListVal<eModeGeom>()})
           << AOpt2007(mNameResult,"Out"," prefix of output files, default=Bascule.tif",{eTA2007::HDV})
           << AOpt2007(mNameMasq1, "Masq1","Masq of first image if any")
           /*<< AOpt2007(mResolZBuf,"ResZBuf","Resolution of ZBuffer", {eTA2007::HDV})
           << AOpt2007(mDoImages,"DoIm","Do images", {eTA2007::HDV})
           << AOpt2007(mNbPixImRedr,"NbPixIR","Resolution of ZBuffer", {eTA2007::HDV})
           << AOpt2007(mMII,"MII","Margin Inside Image (for triangle validation)", {eTA2007::HDV})
           << AOpt2007(mSKE,CurOP_SkipWhenExist,"Skip command when result exist")*/
           << AOptBench()
           //<< mPhProj.DPRadiomData().ArgDirOutOpt()
        ;

}


void cAppliNuageBascule::MakeDepthImage()
{
    mImRed.DIm().Resize(CurSzIn(),eModeInitImage::eMIA_Null);

    cPt2di aP0 = CurBoxIn().P0();
    cPt2di aPix1;
    for(const auto & aPix: mImRed.DIm())
    {
        aPix1 = aPix+aP0;
        double aX2= aPix1.x() + mImPx1.DIm().GetV(aPix1);
        cPt2dr aPix2(aX2, aPix1.y());
        tSeg3dr aP2TerCam1 = mCamPC->Image2Bundle(ToR(aPix1));
        tSeg3dr aP2TerCam2= mSecCamPC->Image2Bundle(aPix2);
        cPt3dr aPZ=RobustBundleInters({aP2TerCam1,aP2TerCam2});
        mImRed.DIm().SetV(aPix,aPZ.z());
    }
    APBI_WriteIm(mNameResult,mImRed);
}

int cAppliNuageBascule::ExeOnParsedBox()
{
    mImPx1= APBI_ReadIm<tREAL4>(mNameCloud2D_DepthIn);
    mImMasq1 = ReadMasqWithDef(CurBoxIn(), mNameMasq1);

    if (mSecCamPC)
        MakeDepthImage();

    // Make Bascule and Save
    //MakeBascule();

    return EXIT_SUCCESS;
}

std::string cAppliNuageBascule::Name_Epip(std::string & aName)
{
    std::string aTab[4]={"Epi_Im1_Right","Epi_Im2_Right","Epi_Im1_Left","Epi_Im2_Left"};
    std::string aTabMatch[4]={"Epi_Im2_Left","Epi_Im1_Left","Epi_Im2_Right","Epi_Im1_Right"};

    std::string aNameSec="";

    for(int i=0; i<4;i++)
    {
        if (MatchPattern(aName,aTab[i]))
        {
            aNameSec = aTabMatch[i];
            break;
        }
    }
    return aNameSec;
}

int cAppliNuageBascule::Exe()
{
    mPhProj.FinishInit();
    // read camera
    mCamPC =mPhProj.ReadCamPC(mNameIm,true);

    if (mModeGeom==eModeGeom::eGEOM_EPIP) // compute secondary epipolar image
        {
            std::string aSecNameIm = Name_Epip(mNameIm);
            mSecCamPC = mPhProj.ReadCamPC(aSecNameIm,true);
        }

    if (RunMultiSet(0,0))
        return ResultMultiSet();

    APBI_ExecAll();

    return EXIT_SUCCESS;
}



/*  ============================================= */
/*       ALLOCATION                               */
/*  ============================================= */

tMMVII_UnikPApli Alloc_NuageBascule(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
    return tMMVII_UnikPApli(new cAppliNuageBascule(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecNuageBascule
    (
        "NuageBascule",
        Alloc_NuageBascule,
        "Bascule of Pax/Depth maps to ground",
        {eApF::Cloud},
        {eApDT::Image},
        {eApDT::Image},
        __FILE__
        );


};
