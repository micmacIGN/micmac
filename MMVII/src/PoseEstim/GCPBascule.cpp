#include "cMMVII_Appli.h"
#include "MMVII_PCSens.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Matrix.h"


namespace MMVII
{
class cAppli_GCPBascule : public cMMVII_Appli
{
    public:
        cAppli_GCPBascule(const std::vector<std::string> & aVArgs, const cSpecMMVII_Appli & aSpec);
        void AddData(const cAuxAr2007 & anAuxInit);

    private:
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
        cPhotogrammetricProject mPhProj;
        std::string             mSpecImIn;
        bool                    mShow;
        cSimilitud3D<tREAL8>    mSim;
        double                  mScale;
        cPt3dr                  mTr;
        cRotation3D<tREAL8>     mRot;
        double                  mRes2;
        bool                    mCSVReport;
        bool                    mWriteSim;

};

cCollecSpecArg2007 & cAppli_GCPBascule::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
        << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}}) //input pattern of images
        << mPhProj.DPOrient().ArgDirInMand()                                                             //input orientation
        << mPhProj.DPGndPt2D().ArgDirInMand()                                                            //input (2d) image measurements of GCPs
        << mPhProj.DPGndPt3D().ArgDirInMand()                                                            //input (3d) ground measurements of GCPs
        << mPhProj.DPOrient().ArgDirOutMand()                                                            //output orientation expressed in GCPs frame
      ;
}

cCollecSpecArg2007 & cAppli_GCPBascule::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return anArgOpt
        << AOpt2007(mShow,"Show","Show some useful details",{eTA2007::HDV})
        << AOpt2007(mCSVReport,"CSVReport","Save residuals to a csv file",{eTA2007::HDV})
        << AOpt2007(mWriteSim,"WriteSim","Save Similarity parameters to a file",{eTA2007::HDV})
    ;
}

cAppli_GCPBascule::cAppli_GCPBascule(const std::vector<std::string> & aVArgs, const cSpecMMVII_Appli & aSpec):
    cMMVII_Appli(aVArgs,aSpec),
    mPhProj     (*this),
    mShow       (false),
    mCSVReport  (true),
    mWriteSim   (false)
{
}

void cAppli_GCPBascule::AddData(const cAuxAr2007 & anAuxInit)
{
    cAuxAr2007 anAux("Simlitude3D",anAuxInit);
    MMVII::AddData(cAuxAr2007("Scale",anAux),mScale);
}

void AddData(const cAuxAr2007 & anAuxInit, cAppli_GCPBascule & aAppliBascule)
{
    aAppliBascule.AddData(anAuxInit);
}


int cAppli_GCPBascule::Exe()
{

    mPhProj.FinishInit();

    std::vector<std::string> aVImg = VectMainSet(0);

    //to store GCPs
    cSetMesGndPt aSet;

    //load input 3D measures
    mPhProj.LoadGCP3D(aSet);

    //load input 2D measures
    for(const auto & aNameIm : aVImg)
    {
        //read sensor
        cSensorCamPC* aCam = mPhProj.ReadCamPC(aNameIm,true);

        //load 2D measure
        mPhProj.LoadIm(aSet,nullptr,*aCam);
    }

    cSetMesGnd3D aInSet, aOutSet;

    //pseudo-intersection
    for(const auto & aMesIm : aSet.MesImOfPt())
    {
        int aNumPt = aMesIm.NumPt();
        const cMes1Gnd3D & aMes3D = aSet.MesGCPOfNum(aNumPt);

        if(aMesIm.VMeasures().size() >= 2)
        {
            cMes1Gnd3D aInMes;
            aInMes.mPt = aSet.BundleInter(aMesIm);
            aInMes.mNamePt = aMes3D.mNamePt;
            aInSet.AddMeasure3D(aInMes);

            cMes1Gnd3D aOutMes;
            aOutMes.mPt = aMes3D.mPt;
            aOutMes.mNamePt = aMes3D.mNamePt;
            aOutSet.AddMeasure3D(aOutMes);
            
            if(mShow)
            {
                StdOut() << "Name = " << aInMes.mNamePt << " Orig_Frame = " << aInMes.mPt << " Target_Frame = " << aOutMes.mPt << "\n";
            }

        }
    }

    //we need at least 3 correspondences
    if(aInSet.Measures().size() < 3)
    {
        StdOut() << "Not enough points ! " << "\n";
        return EXIT_FAILURE;
    }

    //vectors to store points for StdGlobEstimate()
    std::vector<cPt3dr> aInPts, aOutPts;
    for(const auto & aMes : aInSet.Measures())
        aInPts.push_back(aMes.mPt);
    for(const auto & aMes : aOutSet.Measures())
        aOutPts.push_back(aMes.mPt);

    //estimate 3d similarity
    mSim = mSim.StdGlobEstimate(aInPts,aOutPts,&mRes2,nullptr,cParamCtrlOpt::Default());

    if(mShow)
    {
        StdOut() << "Sim residual = " << mRes2 << "\n";
        StdOut() << "Scale = "        << mSim.Scale() << "\n";
        StdOut() << "Translation = "  << mSim.Tr() << "\n";
        StdOut() << "Rotation = "     << mSim.Rot().Mat() << "\n";
    }
    
    //apply similarity to input ori
    for(const auto & aIm : aVImg)
    {
        //read sensor
        cSensorCamPC* aCam = mPhProj.ReadCamPC(aIm,true);

        //get image pose
        const tPoseR & aPose = aCam->Pose();

        //apply sim to pose
        tPoseR aTransformedPose = TransfoPose(mSim,aPose);
        aCam->SetPose(aTransformedPose);

        //save
        mPhProj.SaveSensor(*aCam);
    }

    //generate report
    if(mCSVReport)
    {
        //csv file name
        std::string aReportFileName = "3D-Sim-Residuals";

        //csv header
        InitReportCSV(aReportFileName,"csv","false",{"Pt_name","delta_x","delta_y","delta_z"});

        //vector to store transformed points
        std::vector<cPt3dr> aVTransformedPts;

        for(size_t i=0; i<aInSet.Measures().size(); i++)
        {
            cPt3dr aInPt = aInSet.Measures().at(i).mPt;
            cPt3dr aTransformedPt = mSim.Value(aInPt);
            cPt3dr aOutPt = aOutSet.Measures().at(i).mPt;

            //residuals
            double dx =  aTransformedPt.x() - aOutPt.x();
            double dy =  aTransformedPt.y() - aOutPt.y();
            double dz =  aTransformedPt.z() - aOutPt.z();

            //add to csv file
            AddOneReportCSV(aReportFileName,{aInSet.Measures().at(i).mNamePt,ToStr(dx),ToStr(dy),ToStr(dz)});
        }
    }

    //write similarity parameters to .xml file
    if(mWriteSim)
    {
        //file name
        std::string aNameFile = mPhProj.DPOrient().FullDirOut()
                                + "3D-Similarity"
                                + "_"
                                + mPhProj.DPOrient().DirIn()
                                + "_"
                                + mPhProj.DPOrient().DirOut()
                                + std::string(".xml");
        //assign
        mScale = mSim.Scale();
        mTr = mSim.Tr();
        mRot = mSim.Rot();

        //save file
        SaveInFile(*this,aNameFile);

        StdOut() << "Similarity parameters wrote to file : " << aNameFile << "\n";
    }

    
    return EXIT_SUCCESS;
}

tMMVII_UnikPApli Alloc_GCPBascule(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
      return tMMVII_UnikPApli(new cAppli_GCPBascule(aVArgs,aSpec));
}

cSpecMMVII_Appli TheSpec_GCPBascule
(
     "GCPBascule",
      Alloc_GCPBascule,
      "Perform a bascule based on GCPs",
      {eApF::Ori,eApF::GCP},                      //features
      {eApDT::ObjCoordWorld, eApDT::ObjMesInstr}, //inputs
      {eApDT::Console},                           //output
      __FILE__
);

} //MMVII
