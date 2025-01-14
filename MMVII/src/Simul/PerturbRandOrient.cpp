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
   /*                 cAppli_PerturbRandomOri                    */
   /*                                                            */
   /* ********************************************************** */

class cAppli_PerturbRandomOri : public cMMVII_Appli
{
     public :
        cAppli_PerturbRandomOri(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        std::vector<std::string>  Samples() const override;
     private :

	cPhotogrammetricProject  mPhProj;

	std::string              mSpecIm;

        tREAL8                   mRandOri;
        tREAL8                   mRandC;
};

cAppli_PerturbRandomOri::cAppli_PerturbRandomOri(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this)
{
}

cCollecSpecArg2007 & cAppli_PerturbRandomOri::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mSpecIm ,"Name of Input File",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
              <<  mPhProj.DPOrient().ArgDirInMand()
              <<  mPhProj.DPOrient().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_PerturbRandomOri::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    
    return      anArgObl
            << AOpt2007(mRandOri,"RandOri","Random perturbation on orientations")
            << AOpt2007(mRandC  ,"RandC"  ,"Random perturbation on center")
    ;
}


int cAppli_PerturbRandomOri::Exe()
{
    mPhProj.FinishInit();

    for (const auto & aNameIm : VectMainSet(0))
    {
        cSensorImage* aSI = mPhProj.ReadSensor(aNameIm,true,false);

        
        mPhProj.SaveSensor(*aSI);
    }
    return EXIT_SUCCESS;
}


std::vector<std::string>  cAppli_PerturbRandomOri::Samples() const
{
   return {"NO SAMPLES FOR NOW"};
}



tMMVII_UnikPApli Alloc_PerturbRandomOri(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_PerturbRandomOri(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ChSysCo
(
     "OriPerturbRandom",
      Alloc_PerturbRandomOri,
      "Perturbate random de orientation (for simulations)",
      {eApF::Ori,eApF::Simul},
      {eApDT::Ori},
      {eApDT::Ori},
      __FILE__
);

}; // MMVII

