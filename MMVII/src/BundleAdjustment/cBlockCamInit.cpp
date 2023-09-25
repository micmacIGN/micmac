#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"


/**
   \file GCPQuality.cpp


 */

namespace MMVII
{

	/*

class cBlocCam
{
      public :
         void Add(const std::string & aNameIm,);

         cBlocCam(const std::string aPattern&,cPt2di aNum);
	 // cSensorCamPC
      private :
	  cBijectiveMapI2O  

	  std::string  mPattern;
	  cPt2di       mNum;
          // t2MapStrInt  mMapIntCam;
          // t2MapStrInt  mMapIntBlock;
};
*/

/**  Structure of block data
 
         - there can exist several block, each block has its name
	 - the result are store in a folder, one file by block
	 - 

     This command will create a block init  and save it in a given folder


     For now, just check the rigidity => will implemant in detail with Celestin ...

*/


/* ==================================================== */
/*                                                      */
/*          cAppli_CalibratedSpaceResection             */
/*                                                      */
/* ==================================================== */

class cAppli_BlockCamInit : public cMMVII_Appli
{
     public :

        cAppli_BlockCamInit(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

     private :
        void  MakeOneIm(const std::string & aNameIm);
        void  MakeGlobReports();
        void  BeginReport();
        void  ReportsByGCP();
        void  ReportsByCam();

        std::string              mSpecImIn;   ///  Pattern of xml file
        cPhotogrammetricProject  mPhProj;
	std::string              mPattern;
	cPt2di                   mNumSub;
};

cAppli_BlockCamInit::cAppli_BlockCamInit
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this)
{
}



cCollecSpecArg2007 & cAppli_BlockCamInit::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
              <<  Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
              <<  mPhProj.DPOrient().ArgDirInMand()
              <<  Arg2007(mPattern,"Pattern for images specifing sup expr")
              <<  Arg2007(mNumSub,"Num of sub expr for x:block and  y:image")
           ;
}

cCollecSpecArg2007 & cAppli_BlockCamInit::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return    anArgOpt
	    /*
	     << AOpt2007(mGeomFiedlVec,"GFV","Geom Fiel Vect for visu [Mul,Witdh,Ray,Zoom?=2]",{{eTA2007::ISizeV,"[3,4]"}})
	     */
    ;
}


int cAppli_BlockCamInit::Exe()
{
    mPhProj.FinishInit();

    for (const auto & anIm : VectMainSet(0))
    {
        std::string aStrBlock = PatternKthSubExpr(mPattern,mNumSub.x(),FileOfPath(anIm));
        std::string aStrIma   = PatternKthSubExpr(mPattern,mNumSub.y(),FileOfPath(anIm));


        StdOut() << " Bl=" << aStrBlock  << " ; Ima=" << aStrIma  << "   <<== " << anIm << "\n";
    }

    return EXIT_SUCCESS;
}                                       

/* ==================================================== */
/*                                                      */
/*               MMVII                                  */
/*                                                      */
/* ==================================================== */


tMMVII_UnikPApli Alloc_BlockCamInit(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_BlockCamInit(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_BlockCamInit
(
     "BlockCamInit",
      Alloc_BlockCamInit,
      "Reports on GCP projection",
      {eApF::GCP,eApF::Ori},
      {eApDT::Orient},
      {eApDT::Xml},
      __FILE__
);

/*
*/




}; // MMVII

