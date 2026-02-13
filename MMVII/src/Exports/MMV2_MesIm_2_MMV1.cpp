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

        cPhotogrammetricProject mPhProj;
        std::string             mSpecImIn;
        std::string             mNameFile;
        bool                    mShow;
        bool                    mFiltNONE; 

};

cAppli_MMV2_MesIm_2_MMV1::cAppli_MMV2_MesIm_2_MMV1(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mShow        (false),
    mFiltNONE    (true)

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
               << AOpt2007(mFiltNONE,"FiltNONE","Do not export points with name starting with NONE",{eTA2007::HDV})
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
        //std::string aNameImage = aCImage;

        if(mPhProj.HasMeasureIm(aCImage))
        {
            //MicMac v1
            cMesureAppuiFlottant1Im aMAF;
            aMAF.NameIm() = aCImage;

            //retreive set of measure in an image
            cSetMesPtOf1Im  aSet = mPhProj.LoadMeasureIm(aCImage);

            if(mShow)
            {
                std::cout << "Image: " << aCImage
                          << "\t#Nb Img Measure: " << aSet.Measures().size()
                          << std::endl;
            }
            
            //retreive vector of measure of a point in an image
            for(const auto & aMes : aSet.Measures())
            {
                std::string aGcpName = aMes.mNamePt;
                if (!(mFiltNONE && (aGcpName.substr(0, 4) == "NONE")))
                {
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
                        std::cout << aMAF.NameIm() << "," << aGcpName << "," << aPtIm.x() << "," << aPtIm.y() << std::endl;
                    }
                
                    //add to the dico
                    aMAF.OneMesureAF1I().push_back(aOAF1I);
                }
                    }

            //append to the dico
            aDico.MesureAppuiFlottant1Im().push_back(aMAF);
        }
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
