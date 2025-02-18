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

    void  AddOneSetGCP3D(const std::string & aFolderIn, const std::string &aFolderOut, tREAL8 aWFactor); // aFolderOut="" if no out

    std::string               mSpecImIn;

    std::string               mDataDir;  /// Default Data dir for all

    cPhotogrammetricProject   mPhProj;
    cMMVII_BundleAdj          mBA;

    std::vector<std::vector<std::string>>  mGCP3D; // gcp ground coords with sigma factor and optional output dir
    std::string               mGCPFilter;  // pattern to filter names of GCP
    std::string               mGCPFilterAdd;  // pattern to filter GCP by additional info
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
            << Arg2007( mGCP3D, "GCP ground coords and sigma factor, SG=0 fix, SG<0 schurr elim, SG>0 and optional output dir [[Folder,SigG,FOut?],...]]")
           ;
}

cCollecSpecArg2007 & cAppliTopoAdj::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    
    return 
          anArgOpt
      << AOpt2007(mDataDir,"DataDir","Default data directories ",{eTA2007::HDV})
      << AOpt2007(mNbIter,"NbIter","Number of iterations",{eTA2007::HDV})
      << AOpt2007(mGCPFilter,"GCPFilter","Pattern to filter GCP by name")
      << AOpt2007(mGCPFilterAdd,"GCPFilterAdd","Pattern to filter GCP by additional info")
      << AOpt2007(mLVM,"LVM","Levenberg–Marquardt parameter (to have better conditioning of least squares)",{eTA2007::HDV})
    ;
}

void  cAppliTopoAdj::AddOneSetGCP3D(const std::string & aFolderIn, const std::string & aFolderOut, tREAL8 aWFactor)
{
    cSetMesGndPt  aFullMesGCP;
    cMes3DDirInfo * aMesDirInfo = cMes3DDirInfo::addMes3DDirInfo(mBA.getGCP(), aFolderIn, aFolderOut, aWFactor);
    mPhProj.LoadGCP3DFromFolder(aFolderIn, aFullMesGCP, aMesDirInfo, "", mGCPFilter, mGCPFilterAdd);
    auto aFullMes3D = aFullMesGCP.ExtractSetGCP("???");
    mBA.AddGCP3D(aMesDirInfo,aFullMes3D);
}


int cAppliTopoAdj::Exe()
{
    mPhProj.FinishInit();

    for (const auto& aVStrGCP : mGCP3D)
    {
        // expected: [Folder,SigG,FOut?]
        if ((aVStrGCP.size() <2) || (aVStrGCP.size() >3))
        {
            MMVII_UnclasseUsEr("Bad size of GCP3D, exp in [2,3] got : " + ToStr(aVStrGCP.size()));
        }
        AddOneSetGCP3D(aVStrGCP[0],aVStrGCP.size()>2?aVStrGCP[2]:"",cStrIO<double>::FromStr(aVStrGCP[1]));
    }

    mBA.AddTopo();

    //here no 2d mes, fake it
    cMes2DDirInfo * aMes2DDirInfo = cMes2DDirInfo::addMes2DDirInfo(mBA.getGCP(), "in",cStdWeighterResidual());
    cSetMesPtOf1Im aSetMesPtOf1Im;
    mBA.AddGCP2D(aMes2DDirInfo, aSetMesPtOf1Im, nullptr, eLevelCheck::NoCheck);


    for (int aKIter=0 ; aKIter<mNbIter ; aKIter++)
    {
        bool isLastIter =  (aKIter==(mNbIter-1)) ;
        mBA.OneIteration(mLVM,isLastIter);
    }

    mBA.Save_newGCP3D();
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
      {eApDT::ObjCoordWorld, eApDT::Topo},
      {eApDT::ObjCoordWorld},
      __FILE__
);

/*
*/

}; // MMVII

