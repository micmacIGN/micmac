#include "cMMVII_Appli.h"
#include "MMVII_PCSens.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Matrix.h"


namespace MMVII
{
class cAppli_GCPAbsOri : public cMMVII_Appli
{
    public:
        cAppli_GCPAbsOri(const std::vector<std::string> & aVArgs, const cSpecMMVII_Appli & aSpec);
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

cCollecSpecArg2007 & cAppli_GCPAbsOri::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
        << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}}) //input pattern of images
        << mPhProj.DPOrient().ArgDirInMand()                                                             //input orientation
        << mPhProj.DPGndPt2D().ArgDirInMand()                                                            //input (2d) image measurements of GCPs
        << mPhProj.DPGndPt3D().ArgDirInMand()                                                            //input (3d) ground measurements of GCPs
        << mPhProj.DPOrient().ArgDirOutMand()                                                            //output orientation expressed in GCPs frame
      ;
}

cCollecSpecArg2007 & cAppli_GCPAbsOri::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return anArgOpt
        << AOpt2007(mShow,"Show","Show some useful details",{eTA2007::HDV})
        << AOpt2007(mCSVReport,"CSVReport","Save residuals to a csv file",{eTA2007::HDV})
        << AOpt2007(mWriteSim,"WriteSim","Save Similarity parameters to a file",{eTA2007::HDV})
    ;
}

cAppli_GCPAbsOri::cAppli_GCPAbsOri(const std::vector<std::string> & aVArgs, const cSpecMMVII_Appli & aSpec):
    cMMVII_Appli(aVArgs,aSpec),
    mPhProj     (*this),
    mShow       (false),
    mCSVReport  (true),
    mWriteSim   (true)
{
}

void cAppli_GCPAbsOri::AddData(const cAuxAr2007 & anAuxInit)
{
    cAuxAr2007 anAux("3D-Similarity",anAuxInit);
    MMVII::AddData(cAuxAr2007("Scale",anAux),mScale);
    MMVII::AddData(cAuxAr2007("Translation",anAux),mTr);
    MMVII::AddData(cAuxAr2007("Rotation",anAux),mRot);
}

void AddData(const cAuxAr2007 & anAuxInit, cAppli_GCPAbsOri & aAppliAbsOri)
{
    aAppliAbsOri.AddData(anAuxInit);
}

int cAppli_GCPAbsOri::Exe()
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

    //vectors to store points for StdGlobEstimate()
    std::vector<cPt3dr> aInPts, aOutPts;
    //vector to store GCP names
    std::vector<std::string> aPtsNames;

    //pseudo-intersection - fill point and name vectors
    for(const auto & aMesIm : aSet.MesImOfPt())
    {
        int aNumPt = aMesIm.NumPt();
        const cMes1Gnd3D & aMes3D = aSet.MesGCPOfNum(aNumPt);

        if(aMesIm.VMeasures().size() >= 2)
        {
            //compute bundle intersection - origine frame coordinates
            cPt3dr aInPt = aSet.BundleInter(aMesIm);
            //retreive target frame coordinates
            cPt3dr aOutPt = aMes3D.mPt;
            //pt name
            std::string aNamePt = aMes3D.mNamePt;
            
            //store coordinates and names
            aInPts.push_back(aInPt);
            aOutPts.push_back(aOutPt);
            aPtsNames.push_back(aNamePt);
            
            if(mShow)
            {
                StdOut() << "Name = " << aNamePt << " Origin_Frame = " << aInPt 
                         << "  &&  " 
                         << "Target_Frame = " << aOutPt 
                         << "\n";
            }
        }
    }

    //we need at least 3 correspondences
    if(aInPts.size() < 3)
    {
        StdOut() << "Not enough points (the minimum is 3) ! " << "\n";
        return EXIT_FAILURE;
    }

    //estimate 3d similarity
    mSim = mSim.StdGlobEstimate(aInPts,aOutPts,&mRes2,nullptr,cParamCtrlOpt::Default());

    if(mShow)
    {
        StdOut() << "Similarity residual = " << mRes2 << "\n";
        StdOut() << "Similarity scale = "        << mSim.Scale() << "\n";
        StdOut() << "Similarity translation = "  << mSim.Tr() << "\n";
        StdOut() << "Similarity rotation = "     << mSim.Rot().Mat() << "\n";
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
        std::string aReportFileName = std::string("3D-Similarity-Residuals") 
                                    + "_"
                                    + mPhProj.DPOrient().DirIn()
                                    + "_"
                                    + mPhProj.DPOrient().DirOut();

        //csv header
        InitReportCSV(aReportFileName,"csv","false",{"GCP_name","delta_x","delta_y","delta_z"});

        for(size_t i=0; i<aInPts.size(); i++)
        {
            cPt3dr aTransformedPt = mSim.Value(aInPts[i]);

            //residuals
            double dx =  aTransformedPt.x() - aOutPts[i].x();
            double dy =  aTransformedPt.y() - aOutPts[i].y();
            double dz =  aTransformedPt.z() - aOutPts[i].z();

            //add to csv file
            AddOneReportCSV(aReportFileName,{aPtsNames[i],ToStr(dx),ToStr(dy),ToStr(dz)});
        }
    }

    //write similarity parameters to .xml file
    if(mWriteSim)
    {
        //file name
        std::string aNameFile = mPhProj.DPOrient().FullDirOut()
                                + "3D-Similarity-Parameters"
                                + "_"
                                + mPhProj.DPOrient().DirIn()
                                + "_"
                                + mPhProj.DPOrient().DirOut()
                                + ".xml";
        //assign
        mScale = mSim.Scale();
        mTr = mSim.Tr();
        mRot = mSim.Rot();

        //save file
        SaveInFile(*this,aNameFile);
        
        if(mShow)
        {
            StdOut() << "3D similarity parameters wrote to file: " << aNameFile << "\n";
        }
        
    }

    return EXIT_SUCCESS;
}

tMMVII_UnikPApli Alloc_GCPAbsOri(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
      return tMMVII_UnikPApli(new cAppli_GCPAbsOri(aVArgs,aSpec));
}

cSpecMMVII_Appli TheSpec_GCPAbsOri
(
     "GCPAbsOri",
      Alloc_GCPAbsOri,
      "Generating an Absolute Orientation using a 3D similarity transformation based on GCPs",
      {eApF::Ori,eApF::GCP},                      //features
      {eApDT::ObjCoordWorld, eApDT::ObjMesInstr}, //inputs
      {eApDT::Console},                           //output
      __FILE__
);

} //MMVII
