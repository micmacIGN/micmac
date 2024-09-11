#include "BundleAdjustment.h"

/**
   \file cAppliTopoAdj.cpp

*/
#include "MMVII_Topo.h"



namespace MMVII
{

   /* ********************************************************** */
   /*                                                            */
   /*                 cAppliTopoAdj                              */
   /*                                                            */
   /* ********************************************************** */

class cAppliTopoAdj : public cMMVII_Appli
{
public :
    cAppliTopoAdj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
    int Exe() override;
    cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
    cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
private :

    std::vector<tREAL8>  ConvParamStandard(const std::vector<std::string> &,size_t aSzMin,size_t aSzMax) ;
    /// New Method for multiple GCP : each
    void  AddOneSetGCP(const std::vector<std::string> & aParam);
    void  AddOneSetTieP(const std::vector<std::string> & aParam);

    std::string               mSpecImIn;

    std::string               mDataDir;  /// Default Data dir for all

    cPhotogrammetricProject   mPhProj;
    cMMVII_BundleAdj          mBA;

    double  mGCPW;
    std::vector<std::vector<std::string>>  mAddGCPW; // In case there is multiple GCP Set
    std::string               mGCPFilter;  // pattern to filter names of GCP
    std::string               mGCPFilterAdd;  // pattern to filter GCP by additional info
    std::vector<std::string>  mTiePWeight;
    std::vector<std::vector<std::string>>  mAddTieP; // In case there is multiple GCP Set
    std::vector<double>       mBRSigma; // RIGIDBLOC
    std::vector<double>       mBRSigma_Rat; // RIGIDBLOC

    int                       mNbIter;

    std::string               mPatParamFrozCalib;
    std::string               mPatFrosenCenters;
    std::string               mPatFrosenOrient;
    std::vector<tREAL8>       mViscPose;
    tREAL8                    mLVM;  ///< Levenberk Markard
    std::vector<std::string>  mVSharedIP;  ///< Vector for shared intrinsic param
};

cAppliTopoAdj::cAppliTopoAdj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli    (aVArgs,aSpec),
   mDataDir        ("Std"),
   mPhProj         (*this),
   mBA             (&mPhProj),
   mGCPW           (1.),
   mGCPFilter      (""),
   mGCPFilterAdd   (""),
   mNbIter         (10),
   mLVM            (0.0)
{
}

cCollecSpecArg2007 & cAppliTopoAdj::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
            << mPhProj.DPTopoMes().ArgDirInMand("Dir for Topo measures")
            << mPhProj.DPTopoMes().ArgDirOutMand("Dir for Topo measures output")
            << mPhProj.DPPointsMeasures().ArgDirInMand("Dir for points initial coordinates")
            << mPhProj.DPPointsMeasures().ArgDirOutMand("Dir for points final coordinates")
           ;
}

cCollecSpecArg2007 & cAppliTopoAdj::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    
    return 
          anArgOpt

      << AOpt2007(mGCPW,"GCPW", "Constrained GCP weight factor (default: 1)")
      << AOpt2007(mDataDir,"DataDir","Default data directories ",{eTA2007::HDV})
      << AOpt2007(mNbIter,"NbIter","Number of iterations",{eTA2007::HDV})
      << AOpt2007(mGCPFilter,"GCPFilter","Pattern to filter GCP by name")
      << AOpt2007(mGCPFilterAdd,"GCPFilterAdd","Pattern to filter GCP by additional info")
      << mPhProj.DPPointsMeasures().ArgDirOutOpt("GCPDirOut","Dir for output GCP")
      << AOpt2007(mLVM,"LVM","Levenberg–Marquardt parameter (to have better conditioning of least squares)",{eTA2007::HDV})
    ;
}



std::vector<tREAL8>  cAppliTopoAdj::ConvParamStandard(const std::vector<std::string> & aVParStd,size_t aSzMin,size_t aSzMax)
{
    if ((aVParStd.size() <aSzMin) || (aVParStd.size() >aSzMax))
    {
        MMVII_UnclasseUsEr("Bad size of AddOneSetGCP, exp in [3,6] got : " + ToStr(aVParStd.size()));
    }

    std::vector<tREAL8>  aRes;  // then weight must be converted from string to double
    for (size_t aK=1 ; aK<aVParStd.size() ; aK++)
        aRes.push_back(cStrIO<double>::FromStr(aVParStd.at(aK)));

    return aRes;
}

// VParam standar is done from  Folder +  weight of size [1]
void  cAppliTopoAdj::AddOneSetGCP(const std::vector<std::string> & aVParStd)
{
    std::string aFolder = aVParStd.at(0);  // folder
    std::vector<tREAL8>  aGCPW = {cStrIO<double>::FromStr(aVParStd.at(1)), 1.}; // with fake im weight

    MMVII_INTERNAL_ASSERT_User(aGCPW[0]>0., eTyUEr::eUnClassedError, "Error: GCPW must be > 0")

    //  load the GCP
    cSetMesImGCP  aFullMesGCP; 
    mPhProj.LoadGCPFromFolder(aFolder,aFullMesGCP,
                              {mBA.getTopo(),&mBA.getVGCP()},
                              "",mGCPFilter,mGCPFilterAdd);

    //here no 2d mes, fake it
    cSetMesPtOf1Im aSetMesIm("none");
    aFullMesGCP.AddMes2D(aSetMesIm);
    cSetMesImGCP * aMesGCP = aFullMesGCP.FilterNonEmptyMeasure(0);

    cStdWeighterResidual aWeighter(aGCPW,1);
    mBA.AddGCP(aFolder,aGCPW.at(0),aWeighter,aMesGCP);
}


void  cAppliTopoAdj::AddOneSetTieP(const std::vector<std::string> & aVParStd)
{
    std::string aFolder = aVParStd.at(0);  // folder
    std::vector<tREAL8>  aTiePW = ConvParamStandard(aVParStd,3,6);
    cStdWeighterResidual aWeighter(aTiePW,0);
    mBA.AddMTieP(aFolder,AllocStdFromMTPFromFolder(aFolder,VectMainSet(0),mPhProj,false,true,false),aWeighter);
}


int cAppliTopoAdj::Exe()
{
    mPhProj.DPPointsMeasures().SetDirInIfNoInit(mDataDir);
    mPhProj.FinishInit();


    mBA.AddTopo();

    std::vector<std::string> aGCPW;
    aGCPW.push_back(cStrIO<double>::ToStr(mGCPW));
    std::vector<std::string>  aVParamStdGCP{mPhProj.DPPointsMeasures().DirIn()};
    AppendIn(aVParamStdGCP,aGCPW);
    AddOneSetGCP(aVParamStdGCP);

    // Add  the potential suplementary GCP
    for (const auto& aGCP : mAddGCPW)
        AddOneSetGCP(aGCP);


    for (int aKIter=0 ; aKIter<mNbIter ; aKIter++)
    {
        mBA.OneIterationTopoOnly(mLVM, true);
    }

    mBA.Save_newGCP();
    mBA.SaveTopo(); // just for debug for now

    return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_TopoAdj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliTopoAdj(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_TopoAdj
(
     "TopoAdj",
      Alloc_TopoAdj,
      "Topo adjustment",
      {eApF::Topo},
      {eApDT::GCP},
      {eApDT::GCP},
      __FILE__
);

/*
*/

}; // MMVII

