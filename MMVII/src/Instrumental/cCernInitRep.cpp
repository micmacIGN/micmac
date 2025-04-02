#include "MMVII_BlocRig.h"

#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_2Include_Serial_Tpl.h"

#include "MMVII_Clino.h"


/**
 

 */

namespace MMVII
{

/** In case we are working in non verticalized system (for ex no target, no gps ...) , it may
    be necessary to extract vertical in the camera coordinate  system, using the boresight calibration
    and the clino measures
*/


class cGetVerticalFromClino
{
    public :
       cGetVerticalFromClino(const cCalibSetClino &,const std::vector<tREAL8> &);
    private :
       tREAL8 Score(const cPt3dr & aDir);
       const cCalibSetClino & mCalibs; 
       const std::vector<tREAL8> & mAngles;
};

/* ==================================================== */
/*                                                      */
/*                  cAppli_CernInitRep                  */
/*                                                      */
/* ==================================================== */

class cAppli_CernInitRep : public cMMVII_Appli
{
     public :

        cAppli_CernInitRep(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

        void ProcessOneBloc(const std::vector<cSensorCamPC *> &);
	//std::vector<std::string>  Samples() const override;

     private :
        cPhotogrammetricProject  mPhProj;
        std::string              mSpecIm;
        cBlocOfCamera *          mTheBloc;
        cSetMeasureClino         mMesClino;
// ReadMeasureClino(const std::string * aPatSel=nullptr) const;


};

cCollecSpecArg2007 & cAppli_CernInitRep::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
             <<  Arg2007(mSpecIm,"Pattern/file for images", {{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}}  )
             <<  mPhProj.DPOrient().ArgDirInMand()
             <<  mPhProj.DPRigBloc().ArgDirInMand()
             <<  mPhProj.DPMeasuresClino().ArgDirInMand()
             <<  mPhProj.DPClinoMeters().ArgDirInMand()
           ;
}

cCollecSpecArg2007 & cAppli_CernInitRep::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return    anArgOpt
    ;
}

cAppli_CernInitRep::cAppli_CernInitRep
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this),
     mTheBloc      (nullptr)
{
}

void cAppli_CernInitRep::ProcessOneBloc(const std::vector<cSensorCamPC *> & aVPC)
{
   MMVII_INTERNAL_ASSERT_tiny(aVPC.size()>=2,"Not enough cam in cAppli_CernInitRep::ProcessOneBloc");

   std::string anId = mTheBloc->IdSync(aVPC.at(0)->NameImage());
   const  cOneMesureClino & aMes = *  mMesClino.MeasureOfId(anId);
   std::string aNameIm = mMesClino.NameOfIm(aMes);
   cPerspCamIntrCalib *  aCalib = mPhProj.InternalCalibFromImage(aNameIm);


   StdOut() << " ID=" << anId << " Ang=" << aMes.Angles()  << " CAM=" << aNameIm << " F=" << aCalib->F() << "\n";
   StdOut() << " NAMES=" <<  mMesClino.NamesClino() << "\n";

   cCalibSetClino aSetC = mPhProj.ReadSetClino(*aCalib,mMesClino.NamesClino());
   for (const auto & aCalC : aSetC.ClinosCal())
   {
       aCalC.Rot().Mat().Show();
       StdOut() << " ==================================================\n";
   }
}


int cAppli_CernInitRep::Exe()
{
    mPhProj.FinishInit();  // the final construction of  photogrammetric project manager can only be done now

    mTheBloc = mPhProj.ReadUnikBlocCam();

    mMesClino = mPhProj.ReadMeasureClino();


    std::vector<std::vector<cSensorCamPC *>>  aVVC = mTheBloc->GenerateOrientLoc(mPhProj,VectMainSet(0));
    for (auto & aVC : aVVC)
    {
        ProcessOneBloc(aVC);
        DeleteAllAndClear(aVC);
    }

    delete mTheBloc;

    return EXIT_SUCCESS;
}                                       

/* ==================================================== */
/*                                                      */
/*               MMVII                                  */
/*                                                      */
/* ==================================================== */

tMMVII_UnikPApli Alloc_CernInitRep(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_CernInitRep(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_CernInitRep
(
      "CERN_InitRep",
      Alloc_CernInitRep,
      "Initialize the repere local to wire/sphere/clino",
      {eApF::Ori},
      {eApDT::Orient}, 
      {eApDT::Xml}, 
      __FILE__
);

}; // MMVII

