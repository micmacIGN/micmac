#include "cMMVII_Appli.h"
#include "MMVII_PCSens.h"

namespace MMVII
{

class cAppli_ExportUndistMesIm : public cMMVII_Appli
{
    public:
        cAppli_ExportUndistMesIm(const std::vector<std::string> & aVArgs, const cSpecMMVII_Appli & aSpec);

    private:
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

        cPhotogrammetricProject  mPhProj;
        std::string              mSpecImIn;
        bool                     mShow;

};

cAppli_ExportUndistMesIm::cAppli_ExportUndistMesIm(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli    (aVArgs,aSpec),
    mPhProj         (*this),
    mShow     (false)

{
}

cCollecSpecArg2007 & cAppli_ExportUndistMesIm::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
             << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
             << mPhProj.DPOrient().ArgDirInMand()
             << mPhProj.DPGndPt3D().ArgDirInMand()
             << mPhProj.DPGndPt2D().ArgDirInMand()
      ;
}


cCollecSpecArg2007 & cAppli_ExportUndistMesIm::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return anArgOpt
               <<  mPhProj.DPGndPt2D().ArgDirOutOptWithDef("Undist")
               << AOpt2007(mShow,"ShowD","Show details",{eTA2007::HDV})
            ;
}

int cAppli_ExportUndistMesIm::Exe()
{
    mPhProj.FinishInit();

    //read the image pattern
    std::vector<std::string> aVecIm = VectMainSet(0);//interface to MainSet
    
    for (const std::string& aCImage : aVecIm)
    {

		cSetMesGndPt aSetMes;

		//load calibration
		cPerspCamIntrCalib * aCal = mPhProj.InternalCalibFromStdName(aCImage);

		//load GCPs
		mPhProj.LoadGCP3D(aSetMes);

		//load image measurements
		mPhProj.LoadIm(aSetMes,aCImage);

		//image measurements to export
		cSetMesPtOf1Im  aSetMesOut(FileOfPath(aCImage));

		for(const auto & aVMes : aSetMes.MesImInit())
		{
			std::string aNameImage = aVMes.NameIm();

			std::vector<cMesIm1Pt> aVMesIm =  aVMes.Measures();
			
			for(const auto & aMes : aVMes.Measures())
			{
				std::string aGcpName = aMes.mNamePt;

				cPt2dr aPtIm = aMes.mPt;

				cPt2dr aPtImUndist = aCal->Undist(aPtIm);

				if(mShow)
				{
					std::cout << aNameImage << "," << aGcpName << "," << aPtImUndist.x() << "," << aPtImUndist.y() << std::endl;
				}
				
				//fill aSetMesOut
				cMesIm1Pt aMesIm1Pt(aPtImUndist,aGcpName,1.0);
				aSetMesOut.AddMeasure(aMesIm1Pt);

			}
		}
		
		//write in a file
		mPhProj.SaveMeasureIm(aSetMesOut);
	}

    return EXIT_SUCCESS;
}

tMMVII_UnikPApli Alloc_Test_ExportUndistMesIm(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
      return tMMVII_UnikPApli(new cAppli_ExportUndistMesIm(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ExportUndistMesIm
(
     "ExportUndistMesIm",
      Alloc_Test_ExportUndistMesIm,
      "Export image points measurements corrected from distorsion",
      {eApF::Ori,eApF::GCP},
      {eApDT::Ori,eApDT::ObjMesInstr},
      {eApDT::Console},
      __FILE__
);

}
