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

          void Show() const;

          void EstimateInit();
          void EstimateInit(size_t aKB1,size_t aKB2,cMMVII_Appli &);
      private :


	  typedef std::pair<std::string,std::string> tPairStr;
	  typedef std::pair<int,int>                 tPairInd;

	  const std::string & NamePoseId(size_t) const;
	  const std::string & NameSyncId(size_t) const;


	  // typedef cBijectiveMapI2O<cSensorCamPC *>  tMap;
	  void AssertFinish();
	  void AssertNotFinish();

	  tPairStr   ComputeStrPoseIdSyncId(cSensorCamPC*);
	  tPairInd   ComputeIndPoseIdSyncId(cSensorCamPC*);

	  cSensorCamPC *& CamOfIndexes(const tPairInd&);

	  bool         mClosed;
	  std::string  mPattern;
	  cPt2di       mNumSub;

          mutable t2MapStrInt  mMapIntSyncId;
          mutable t2MapStrInt  mMapIntPoseId;
	  size_t       mNbSyncId;
	  size_t       mNbPoseId;

	  std::vector<cSensorCamPC*>  mVAllCam;
	  std::vector<std::vector<cSensorCamPC*> >  mV_TB;  // mV_TB[KSyncId][KBlock]
};

cBlocCam::cBlocCam(const std::string & aPattern,cPt2di aNum):
    mClosed  (false),
    mPattern (aPattern),
    mNumSub  (aNum)
{
}

const std::string & cBlocCam::NamePoseId(size_t anInd) const { return *mMapIntPoseId.I2Obj(anInd,false); }

const std::string & cBlocCam::NameSyncId(size_t anInd) const { return *mMapIntSyncId.I2Obj(anInd,false); }


void cBlocCam::Finish()
{
    mNbSyncId = mMapIntSyncId.size();
    mNbPoseId = mMapIntPoseId.size();
    AssertNotFinish();
    mClosed = true;

    for (size_t aKSyncId=0 ; aKSyncId<mNbSyncId ; aKSyncId++)
    {
        mV_TB.push_back(std::vector<cSensorCamPC*>(mMapIntPoseId.size(),nullptr));
    }

    for (const auto & aCam : mVAllCam)
    {
        CamOfIndexes(ComputeIndPoseIdSyncId(aCam)) = aCam;
    }
}

cSensorCamPC *& cBlocCam::CamOfIndexes(const tPairInd & aPair)
{
	return  mV_TB.at(aPair.second).at(aPair.first);
}
void cBlocCam::Show() const
{
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
	auto [aStrBlock,aStrSyncId] = ComputeStrPoseIdSyncId(aCam);

        mMapIntPoseId.Add(aStrBlock,true);
        mMapIntSyncId.Add(aStrSyncId,true);

        mVAllCam.push_back(aCam);
    }
}

cBlocCam::tPairStr  cBlocCam::ComputeStrPoseIdSyncId(cSensorCamPC* aCam)
{
    std::string aNameCam = aCam->NameImage();

    std::string aStrBlock = PatternKthSubExpr(mPattern,mNumSub.x(),aNameCam);
    std::string aStrSyncId  = PatternKthSubExpr(mPattern,mNumSub.y(),aNameCam);

    return std::pair<std::string,std::string>(aStrBlock,aStrSyncId);
}

cBlocCam::tPairInd cBlocCam::ComputeIndPoseIdSyncId(cSensorCamPC* aCam)
{
    auto [aStrBlock,aStrSyncId] = ComputeStrPoseIdSyncId(aCam);

    return std::pair<int,int>(mMapIntPoseId.Obj2I(aStrBlock),mMapIntSyncId.Obj2I(aStrSyncId));
}

void cBlocCam::EstimateInit(size_t aKB1,size_t aKB2,cMMVII_Appli & anAppli)
{
    std::string  anIdReport =  "Detail_" +  NamePoseId(aKB1) + "_" +   NamePoseId(aKB2) ;

    anAppli.InitReport(anIdReport,"csv",false);
    anAppli.AddOneReportCSV(anIdReport,{"SyncId","x","y","z","w","p","k"});

    cPt3dr aSomTr;
    cPt4dr a4;
    size_t aNb=0;
    for (size_t aKT=0 ; aKT<mNbSyncId ; aKT++)
    {
        cSensorCamPC * aCam1 = CamOfIndexes(tPairInd(aKB1,aKT));
        cSensorCamPC * aCam2 = CamOfIndexes(tPairInd(aKB2,aKT));

	// P1(C1)=W   P2(C2) =  W   P1-1 P2 (C2)  = C1
        if ((aCam1!=nullptr) && (aCam2!=nullptr))
        {
           const cIsometry3D<tREAL8> & aP1 = aCam1->Pose();
           const cIsometry3D<tREAL8> & aP2 = aCam2->Pose();

	   // cIsometry3D<tREAL8>  aP2To1 = aP1 * aP2.MapInverse();
	   cIsometry3D<tREAL8>  aP2To1 =  aP1.MapInverse() * aP2;
	   const auto & aRot = aP2To1.Rot();
	   const auto & aMat = aRot.Mat();
	   cPt3dr aTr = aP2To1.Tr();
	   aSomTr += aTr;
	   cPt3dr aWPK = aRot.ToWPK();

	   a4  +=  MatrRot2Quat(aMat);
	   aNb++;

	   std::vector<std::string>  aVReport{
		                              NameSyncId(aKT),
                                              ToStr(aTr.x()),ToStr(aTr.y()),ToStr(aTr.z()),
		                              ToStr(aWPK.x()),ToStr(aWPK.y()),ToStr(aWPK.z())
	                                   };
           anAppli.AddOneReportCSV(anIdReport,aVReport);
	   StdOut()  << "TR =" << aP2To1.Tr() << " Q=" << MatrRot2Quat(aMat) << " "<< aRot.ToWPK()  << "\n";
        }
    }

    aSomTr = aSomTr / tREAL8(aNb);
    a4  = a4  / tREAL8(aNb);
    a4 = VUnit(a4);

    tREAL8 aSomEcTr = 0;
    tREAL8 aSomEc4 = 0;

    for (size_t aKT=0 ; aKT<mNbSyncId ; aKT++)
    {
        cSensorCamPC * aCam1 = CamOfIndexes(tPairInd(aKB1,aKT));
        cSensorCamPC * aCam2 = CamOfIndexes(tPairInd(aKB2,aKT));

        if ((aCam1!=nullptr) && (aCam2!=nullptr))
        {
           const cIsometry3D<tREAL8> & aP1 = aCam1->Pose();
           const cIsometry3D<tREAL8> & aP2 = aCam2->Pose();
	   cIsometry3D<tREAL8>  aP2To1 =  aP1.MapInverse() * aP2;

	   aSomEcTr += SqN2(aSomTr-aP2To1.Tr());
	   aSomEc4  += SqN2(a4- MatrRot2Quat(aP2To1.Rot().Mat()));
        }
    }

    StdOut() << " KKKKK " << aSomTr << " " << a4 << "\n";
    StdOut()  << "DISPTR " << std::sqrt(aSomEcTr/(aNb-1))  << " DISPROT=" << std::sqrt(aSomEc4/(aNb-1)) << "\n";
    StdOut() << " BBB " << NamePoseId(aKB1) <<  " " << NamePoseId(aKB2) << "\n";
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
    aBloc.Show();
    aBloc.EstimateInit(0,1,*this);

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
      "Initialisation of bloc camera",
      {eApF::GCP,eApF::Ori},
      {eApDT::Orient},
      {eApDT::Xml},
      __FILE__
);

/*
*/




}; // MMVII

