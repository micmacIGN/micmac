#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"


/**
   \file GCPQuality.cpp


 */

namespace MMVII
{

class cBlocCam
{
      public :
          void Add(cSensorCamPC *);
          cBlocCam(const std::string & aPattern,cPt2di aNum);
          void Finish();

      private :
	  // typedef cBijectiveMapI2O<cSensorCamPC *>  tMap;
	  void AssertFinish();
	  void AssertNotFinish();

	  std::pair<std::string,std::string>   BlocTime(cSensorCamPC*);
	  std::pair<int,int>                   IndBlocTime(cSensorCamPC*);

	  bool         mClosed;
	  std::string  mPattern;
	  cPt2di       mNumSub;

          t2MapStrInt  mMapIntTime;
          t2MapStrInt  mMapIntBlock;

	  std::vector<cSensorCamPC*>  mVAllCam;
	  std::vector<std::vector<cSensorCamPC*> >  mV_TB;  // mV_TB[KTime][KBlock]
};

cBlocCam::cBlocCam(const std::string & aPattern,cPt2di aNum):
    mClosed  (false),
    mPattern (aPattern),
    mNumSub  (aNum)
{
}

void cBlocCam::Finish()
{
    AssertNotFinish();
    mClosed = true;

    for (size_t aKTime=0 ; aKTime<mMapIntTime.size() ; aKTime++)
    {
        mV_TB.push_back(std::vector<cSensorCamPC*>(mMapIntBlock.size(),nullptr));
    }

    for (const auto & aCam : mVAllCam)
    {
	auto [aIndBloc,aIndTime] = IndBlocTime(aCam);
	mV_TB.at(aIndTime).at(aIndBloc) = aCam;
    }

    for (const auto & aVBl : mV_TB)
    {
        for (const auto  &aCam : aVBl)
	{
            if (aCam==nullptr)
               StdOut() <<  "    00000000000000000\n";
	    else
               StdOut() <<  "    " << aCam->NameImage() << "\n";
	}
        StdOut() << "============================================================\n";
    }
}

void cBlocCam::AssertNotFinish() {MMVII_INTERNAL_ASSERT_tiny(!mClosed,"cBlocCam::AssertNotFinish");}
void cBlocCam::AssertFinish() {MMVII_INTERNAL_ASSERT_tiny(mClosed,"cBlocCam::AssertFinish");}

void cBlocCam::Add(cSensorCamPC * aCam)
{
    AssertNotFinish();

    if (MatchRegex(aCam->NameImage(),mPattern))
    {
	auto [aStrBlock,aStrTime] = BlocTime(aCam);

        mMapIntBlock.Add(aStrBlock,true);
        mMapIntTime.Add(aStrTime,true);

        mVAllCam.push_back(aCam);
    }
}

std::pair<std::string,std::string>   cBlocCam::BlocTime(cSensorCamPC* aCam)
{
    std::string aNameCam = aCam->NameImage();

    std::string aStrBlock = PatternKthSubExpr(mPattern,mNumSub.x(),aNameCam);
    std::string aStrTime  = PatternKthSubExpr(mPattern,mNumSub.y(),aNameCam);

    return std::pair<std::string,std::string>(aStrBlock,aStrTime);
}

std::pair<int,int>  cBlocCam::IndBlocTime(cSensorCamPC* aCam)
{
    auto [aStrBlock,aStrTime] = BlocTime(aCam);

    return std::pair<int,int>(mMapIntBlock.Obj2I(aStrBlock),mMapIntTime.Obj2I(aStrTime));
}


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

    cBlocCam aBloc(mPattern,mNumSub);
    for (const auto & anIm : VectMainSet(0))
    {
	cSensorCamPC * aCam = mPhProj.ReadCamPC(anIm,true);
	aBloc.Add(aCam);
    }
    aBloc.Finish();

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

