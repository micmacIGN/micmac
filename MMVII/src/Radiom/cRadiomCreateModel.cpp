#include "MMVII_Sensor.h"
#include "MMVII_PCSens.h"
#include "MMVII_Radiom.h"
#include "MMVII_Stringifier.h"
#include "MMVII_2Include_Serial_Tpl.h"


namespace MMVII
{
     /* =============================================== */
     /*                       ::                        */
     /* =============================================== */

class cAppliCreateModelRadiom : public cMMVII_Appli
{
     public :

        cAppliCreateModelRadiom(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

     // --- constructed ---
        cPhotogrammetricProject            mPhProj;     ///< The Project, as usual
	std::string                        mNamePatternIm;
	size_t                             mDegreeRadSens;
	size_t                             mDegreeIma;
};

cCollecSpecArg2007 & cAppliCreateModelRadiom::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return anArgObl
          <<   Arg2007(mNamePatternIm,"Name of image", {{eTA2007::MPatFile,"0"},eTA2007::FileDirProj} )
          <<   mPhProj.DPRadiomModel().ArgDirOutMand()
          <<   mPhProj.DPOrient().ArgDirInMand("InputCalibration")

   ;
}

cCollecSpecArg2007 & cAppliCreateModelRadiom::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
          <<   mPhProj.DPRadiomModel().ArgDirInOpt("XXX","YYY")
	  <<   AOpt2007(mDegreeIma ,"DegIma" ,"Degree for per image polynomial",{eTA2007::HDV})
	  <<   AOpt2007(mDegreeRadSens,"DegRadSens","Degree for per sens radial model",{eTA2007::HDV})

   ;
}

int cAppliCreateModelRadiom::Exe()
{
    mPhProj.FinishInit();

    for (const auto & aNameIm : VectMainSet(0))
    {

	  cRadialCRS * aRCRS = mPhProj.CreateNewRadialCRS(mDegreeRadSens,aNameIm);

	  {
	       cCalRadIm_Pol* aCRIP = new cCalRadIm_Pol(aRCRS,mDegreeIma,aNameIm);
	       mPhProj.SaveCalibRad(*aCRIP);
	       delete aCRIP;
          }

	  if (mPhProj.DPRadiomModel().DirInIsInit())
	  {
               cCalibRadiomIma *  aCRI = mPhProj.ReadCalibRadiomIma(aNameIm);
	       StdOut() << "ZZZZ= " << aCRI->NameIm() << "\n";
	       delete aCRI;
	  }

    }

    return EXIT_SUCCESS;
}

cAppliCreateModelRadiom::cAppliCreateModelRadiom(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli      (aVArgs,aSpec),
    mPhProj           (*this),
    mDegreeRadSens    (5),
    mDegreeIma        (0)
{
}


tMMVII_UnikPApli Alloc_RadiomCreateModel(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliCreateModelRadiom(aVArgs,aSpec));
}


cSpecMMVII_Appli  TheSpecRadiomCreateModel
(
     "RadiomCreateModel",
      Alloc_RadiomCreateModel,
      "Create an initial neutral radiometric model",
      {eApF::Radiometry},
      {eApDT::Radiom},
      {eApDT::Radiom},
      __FILE__
);



};
