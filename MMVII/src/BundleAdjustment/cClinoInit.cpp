#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"


/**
   \file GCPQuality.cpp


 */

namespace MMVII
{

class cClinoCal1Mes
{
    public :
        cClinoCal1Mes(cSensorCamPC * aCam,const std::vector<tREAL8> & aVAngles);

        cSensorCamPC *         mCam;
	std::vector<cPt2dr>     mVDir;
	cPt3dr                 mVertInLoc;  ///  Vertical in camera coordinates

        tREAL8  ScoreWPK(int aKClino,const  cRotation3D<tREAL8> &aR) const;
};


cClinoCal1Mes::cClinoCal1Mes(cSensorCamPC * aCam,const std::vector<tREAL8> & aVAngles) :
    mCam       (aCam),
    mVertInLoc (mCam->V_W2L(cPt3dr(0,0,1)))
{
    for (auto & aTeta : aVAngles)
    {
        mVDir.push_back(FromPolar(1.0,aTeta));
    }
}

tREAL8  cClinoCal1Mes::ScoreWPK(int aKClino,const  cRotation3D<tREAL8>& aCam2Clino) const
{
     cPt3dr  aVClin =  aCam2Clino.Value(mVertInLoc);

     cPt2dr aNeedleDir =  VUnit(Proj(aVClin));

     return Norm2(aNeedleDir- mVDir[aKClino]);
}




/* ==================================================== */
/*                                                      */
/*          cAppli_CalibratedSpaceResection             */
/*                                                      */
/* ==================================================== */

class cAppli_ClinoInit : public cMMVII_Appli
{
     public :
        cAppli_ClinoInit(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

     private :
        cPhotogrammetricProject  mPhProj;
        std::string              mNameClino;   ///  Pattern of xml file
	std::vector<std::string>  mPrePost;   ///  Pattern of xml file
	
};

cAppli_ClinoInit::cAppli_ClinoInit
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this)
{
}


cCollecSpecArg2007 & cAppli_ClinoInit::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
              <<  Arg2007(mNameClino,"Name of inclination file",{eTA2007::FileDirProj})
              <<  Arg2007(mPrePost,"[Prefix,PostFix] to compute image name",{{eTA2007::ISizeV,"[2,2]"}})
              <<  mPhProj.DPOrient().ArgDirInMand()
           ;
}

cCollecSpecArg2007 & cAppli_ClinoInit::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return    anArgOpt
	    /*
	     << AOpt2007(mGeomFiedlVec,"GFV","Geom Fiel Vect for visu [Mul,Witdh,Ray,Zoom?=2]",{{eTA2007::ISizeV,"[3,4]"}})
	     */
    ;
}

int cAppli_ClinoInit::Exe()
{
    mPhProj.FinishInit();


    std::vector<std::vector<std::string>> aVNames;
    std::vector<std::vector<double>>      aVAngles;
    std::vector<cPt3dr>                   aVFakePts;



    ReadFilesStruct
    (
         mNameClino,
	 "NNFNF",
	 0,-1,
	 -1,
	 aVNames,
	 aVFakePts,aVFakePts,
	 aVAngles,
	 false
    );

    for (size_t aKLine=0 ; aKLine<aVNames.size() ; aKLine++)
    {
        std::string aNameIm = mPrePost[0] +  aVNames[aKLine][0] + mPrePost[1];
	cSensorCamPC * aCam = mPhProj.ReadCamPC(aNameIm,true);

	StdOut() << aNameIm  << " " << aVAngles[aKLine] << " FF=" << aCam->Pose().Tr() << "\n";

    }
    return EXIT_SUCCESS;
}                                       

/* ==================================================== */
/*                                                      */
/*               MMVII                                  */
/*                                                      */
/* ==================================================== */


tMMVII_UnikPApli Alloc_ClinoInit(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ClinoInit(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ClinoInit
(
     "ClinoInit",
      Alloc_ClinoInit,
      "Initialisation of inclinometer",
      {eApF::Ori},
      {eApDT::Orient},
      {eApDT::Xml},
      __FILE__
);

/*
*/




}; // MMVII

