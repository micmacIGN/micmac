#include "cMMVII_Appli.h"
#include "MMVII_PCSens.h"

#include "V1VII.h"


namespace MMVII
{

class cAppli_MMV2_MesIm_2_MMV1 : public cMMVII_Appli
{
    public:
        cAppli_MMV2_MesIm_2_MMV1(const std::vector<std::string> & aVArgs, const cSpecMMVII_Appli & aSpec);

    private:
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

        cPhotogrammetricProject  mPhProj;
        std::string              mSpecImIn;
        std::string               mNameFile;
        bool                     mShow;

};

cAppli_MMV2_MesIm_2_MMV1::cAppli_MMV2_MesIm_2_MMV1(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli    (aVArgs,aSpec),
    mPhProj         (*this),
    mShow     (false)

{
}

cCollecSpecArg2007 & cAppli_MMV2_MesIm_2_MMV1::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
             << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
             << mPhProj.DPGndPt2D().ArgDirInMand()
             << Arg2007(mNameFile  ,"Name of V1-image-measure file (\""+MMVII_NONE +"\" if none !)",{eTA2007::FileTagged})
      ;
}


cCollecSpecArg2007 & cAppli_MMV2_MesIm_2_MMV1::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return anArgOpt
               << AOpt2007(mShow,"ShowD","Show details",{eTA2007::HDV})
            ;
}

int cAppli_MMV2_MesIm_2_MMV1::Exe()
{
    mPhProj.FinishInit();

    //read the image pattern
    std::vector<std::string> aVecIm = VectMainSet(0);//interface to MainSet
    
#if (MMVII_KEEP_LIBRARY_MMV1)
    //MicMac v1
    cSetOfMesureAppuisFlottants aDico;
    
    for (const std::string& aCImage : aVecIm)
    {

		cSetMesGndPt aSetMes;

		//load GCPs
		// mPhProj.LoadGCP3D(aSetMes);

		//load image measurements
		mPhProj.LoadIm(aSetMes,aCImage);

		//image measurements to export
		cSetMesPtOf1Im  aSetMesOut(FileOfPath(aCImage));
		
		//MicMac v1
		cMesureAppuiFlottant1Im aMAF;

		for(const auto & aVMes : aSetMes.MesImInit())
		{
			std::string aNameImage = aVMes.NameIm();
			std::vector<cMesIm1Pt> aVMesIm =  aVMes.Measures();
			
			//MicMac v1
			aMAF.NameIm() = aNameImage;
			
			for(const auto & aMes : aVMes.Measures())
			{
				std::string aGcpName = aMes.mNamePt;
				cPt2dr aPtIm = aMes.mPt;
				
				//MicMac v1
				cOneMesureAF1I aOAF1I;
				Pt2dr aPt;
				aPt.x = aPtIm.x();
				aPt.y = aPtIm.y();
				aOAF1I.NamePt() = aGcpName;
				aOAF1I.PtIm() = aPt;

				if(mShow)
				{
					std::cout << aNameImage << "," << aGcpName << "," << aPtIm.x() << "," << aPtIm.y() << std::endl;
				}
				
				//add to the dico
				aMAF.OneMesureAF1I().push_back(aOAF1I);

			}
			
			
		}
		
		//append to the dico
		aDico.MesureAppuiFlottant1Im().push_back(aMAF);
	}
	
	//write image measure in MicMac v1 .xml format
	MakeFileXML(aDico,mNameFile);
#else //  (MMVII_KEEP_LIBRARY_MMV1)
     StdOut()  << " \n\n";
     StdOut()  << " ********************************************************************************************************\n";
     StdOut()  << " * Use of MMV1 Library is deprecated in this distrib, see with MicMac's administrator how to install it *\n";
     StdOut()  << " ********************************************************************************************************\n";
     MMVII_INTERNAL_ERROR("Deprecated use of MMV1's library");
#endif //  (MMVII_KEEP_LIBRARY_MMV1)

    return EXIT_SUCCESS;
}

tMMVII_UnikPApli Alloc_Test_MMV2_MesIm_2_MMV1(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
      return tMMVII_UnikPApli(new cAppli_MMV2_MesIm_2_MMV1(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_MMV2_MesIm_2_MMV1
(
     "MMV2_MesIm_2_MMV1",
      Alloc_Test_MMV2_MesIm_2_MMV1,
      "Export image measurements format from MicMac v2 to MicMac v1",
      {eApF::GCP},
      {eApDT::ObjMesInstr},
      {eApDT::ObjMesInstr},
      __FILE__
);

}
