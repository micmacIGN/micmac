#include "MMVII_InstrumentalBlock.h"

#include "cMMVII_Appli.h"
#include "MMVII_2Include_Serial_Tpl.h"




/**
  \file cInstrumentalBloc.cpp


  \brief This file contains the core implemantation of Block of rigid instrument
 
*/

namespace MMVII
{


/* *************************************************************** */
/*                                                                 */
/*                        cIrbComp_TimeS                             */
/*                                                                 */
/* *************************************************************** */

cIrbComp_TimeS::cIrbComp_TimeS (const cIrbComp_Block & aCompBlock) :
    mCompBlock (aCompBlock),
    mSetCams   (aCompBlock)
{
}

const cIrbComp_CamSet & cIrbComp_TimeS::SetCams() const {return mSetCams;}
const cIrbComp_Block & cIrbComp_TimeS::CompBlock() const {return mCompBlock;}

/* *************************************************************** */
/*                                                                 */
/*                        cIrbComp_Block                            */
/*                                                                 */
/* *************************************************************** */

    //  -------------------------- "Constructors"  --------------------------------------------------------

cIrbComp_Block::cIrbComp_Block(const cIrbCal_Block & aCalBlock) :
   mCalBlock   (aCalBlock),
   mPhProj (nullptr)
{
}

cIrbComp_Block::cIrbComp_Block(const std::string & aNameFile) :
    cIrbComp_Block(SimpleCopyObjectFromFile<cIrbCal_Block>(aNameFile))
{
}

cIrbComp_Block::cIrbComp_Block(const cPhotogrammetricProject& aPhProj,const std::string & aNameBloc) :
    cIrbComp_Block  (aPhProj.NameRigBoI(aNameBloc,true))
{
    mPhProj   = &aPhProj;
}

    //  -------------------------- "Modificators=progressive construction"  --------------------------------------------

cIrbComp_TimeS &  cIrbComp_Block::DataOfTimeS(const std::string & aTS)
{
    // possibly add an empty cIrbComp_TimeS if noting at aTS
    mDataTS.emplace(aTS,*this);

    // extract result mDataTS[aTS]  that should exist now
    auto  anIter = mDataTS.find(aTS);
    MMVII_INTERNAL_ASSERT_tiny(anIter!=mDataTS.end(),"cIrbComp_Block::DataOfTimeS");
    return anIter->second;
}

void cIrbComp_Block::AddImagePose(const tPoseR& aPose,const std::string & aNameIm,bool okImNotInBloc)
{
    // extract the name of the calibration 
    std::string aNameCal = PhProj().StdNameCalibOfImage(aNameIm);

    // extract the specification of the camera in the block
    cIrbCal_Cam1 *  aCInRBoI = mCalBlock.mSetCams.CamFromNameCalib(aNameCal,okImNotInBloc);
    if (aCInRBoI==nullptr)
       return;

    // if the image does not belong to this block
    if (!aCInRBoI->ImageIsInBlock(aNameIm))
    {
        return;
    }

    // extract time stamp
    std::string aTimeS = aCInRBoI->TimeStamp(aNameIm);
    // cIrbComp_TimeS &  cIrbComp_Block::DataOfTimeS(const std::string & aTS)
    cIrbComp_TimeS &  aDataTS =  DataOfTimeS(aTimeS);

    // StdOut() << " III=" << aNameIm << " CCC=" << aNameCal << " Ptr=" << aTimeS << "\n";
    aDataTS.mSetCams.AddImagePose(aCInRBoI->Num(),aPose,aNameIm);
}

void cIrbComp_Block::AddImagePose(const std::string & aNameIm,bool  okImNotInBloc)
{
    bool hasPose;
    tPoseR aPose = PhProj().ReadPoseCamPC(aNameIm,&hasPose);
    if (hasPose)
    {
         AddImagePose(aPose,aNameIm,okImNotInBloc);
    }
}

    //  -------------------------- "computation"  --------------------------------------------

std::pair<tPoseR,cIrb_SigmaPoseRel> cIrbComp_Block::ComputeCalibCamsInit(int aKC1,int aKC2) const
{
   // [0]  Compute relative poses for each time stamps where it exist
   std::vector<tPoseR> aVPoseRel;  // vector of relative pose
   std::vector<std::string> aVTS;
   for (const auto & [aName,aDataTS] :  mDataTS)
   {
       const cIrbComp_CamSet & aSetC = aDataTS.SetCams();
        if (aSetC.HasPoseRel(aKC1,aKC2))
        {
            tPoseR aPose = aSetC.PoseRel(aKC1,aKC2);
            aVPoseRel.push_back(aPose);
            aVTS.push_back(aName);
        }
   }

   // [1]  Compute medians, used to have an order of magnitude
   tREAL8 aMedTr=0,aMedRot=0;
   {
       std::vector<tREAL8>  aVDistTr;
       std::vector<tREAL8>  aVDistRot;
       for (size_t aKP1 =0 ; aKP1<aVPoseRel.size() ; aKP1++)
       {
           for (size_t aKP2 =aKP1+1 ; aKP2<aVPoseRel.size() ; aKP2++)
           {
                aVDistTr.push_back(Norm2(aVPoseRel.at(aKP1).Tr()-aVPoseRel.at(aKP2).Tr()));
                aVDistRot.push_back(aVPoseRel.at(aKP1).Rot().Dist(aVPoseRel.at(aKP2).Rot()));
           }
       }
       aMedTr  =  NonConstMediane(aVDistTr);
       aMedRot =  NonConstMediane(aVDistRot);
   }
   

   // [2]  Exract the robust center
   int    aK1Min    = -1;
   tREAL8 aMinDGlob = 1e10;
   tREAL8 aMinDTr   = 1e10;
   tREAL8 aMinDRot  = 1e10;

   for (size_t aKP1 =0 ; aKP1<aVPoseRel.size() ; aKP1++)
   {
       tREAL8 aSumDGlob = 0.0;
       tREAL8 aSumDTr   = 0.0;
       tREAL8 aSumDRot  = 0.0;
       cStdStatRes      aStat;
       for (size_t aKP2 =0 ; aKP2<aVPoseRel.size() ; aKP2++)
       {
            tREAL8 aDTr = Norm2(aVPoseRel.at(aKP1).Tr()-aVPoseRel.at(aKP2).Tr());
            tREAL8 aDRot = aVPoseRel.at(aKP1).Rot().Dist(aVPoseRel.at(aKP2).Rot());
            aSumDTr   += aDTr;
            aSumDRot  += aDRot;
            aSumDGlob += aDTr + aDRot * (aMedTr/aMedRot);
       }
       // StdOut() << "SOM=" << aSumDGlob/aVPoseRel.size() << "\n";
       if (aSumDGlob<aMinDGlob )
       {
           aK1Min    = aKP1;
           aMinDGlob = aSumDGlob ;
           aMinDTr   = aSumDTr   ;
           aMinDRot  = aSumDRot  ;
       }
   }
   aMinDGlob /= aVPoseRel.size() ;
   aMinDTr   /= aVPoseRel.size() ;
   aMinDRot  /= aVPoseRel.size() ;

   /*
   StdOut() << "K1/K2=" << aKC1<< aKC2 
            << " Med , Tr=" <<  aMedTr << " Rot=" <<  aMedRot
	    << " TS "  << aVTS.at(aK1Min) << " DTr=" <<  aMinDTr << " DRot=" << aMinDRot 
	    << "\n";
	    */

   return std::pair<tPoseR,cIrb_SigmaPoseRel>
	  (
              aVPoseRel.at(aK1Min),
              cIrb_SigmaPoseRel(aKC1,aKC2,aMinDTr,aMinDRot)
	  );
}

    //  -------------------------- "Accessors"  --------------------------------------------------------
   
const cIrbCal_CamSet &  cIrbComp_Block::SetOfCalibCams() const { return mCalBlock.SetCams(); }
const cPhotogrammetricProject & cIrbComp_Block::PhProj()
{
    MMVII_INTERNAL_ASSERT_strong(mPhProj,"No PhProj for cIrbComp_Block");
    return *mPhProj;
}
const cIrbCal_Block & cIrbComp_Block::CalBlock() const {return mCalBlock;}
cIrbCal_Block & cIrbComp_Block::CalBlock() {return mCalBlock;}
size_t  cIrbComp_Block::NbCams() const  {return SetOfCalibCams().NbCams();}


/* *************************************************************** */
/*                                                                 */
/*                        cIrbCal_Clino1                          */
/*                                                                 */
/* *************************************************************** */

cIrbCal_Clino1::cIrbCal_Clino1(const std::string & aName) :
   mName         (aName),
   mIsInit       (false),
   mOrientInBloc (tRotR::Identity()),
   mSigmaR       (-1)
{
}

cIrbCal_Clino1::cIrbCal_Clino1() :
   cIrbCal_Clino1 (MMVII_NONE)
{
}

void cIrbCal_Clino1::AddData(const  cAuxAr2007 & anAux)
{
      MMVII::AddData(cAuxAr2007("Name",anAux),mName);
      MMVII::AddData(cAuxAr2007("IsInit",anAux),mIsInit);
      MMVII::AddData(cAuxAr2007("OrientInBloc",anAux),mOrientInBloc);
      MMVII::AddData(cAuxAr2007("SigmaR",anAux),mSigmaR);
}
void AddData(const  cAuxAr2007 & anAux,cIrbCal_Clino1 & aClino)
{
    aClino.AddData(anAux);
}

const std::string & cIrbCal_Clino1::Name() const {return mName;}


/* *************************************************************** */
/*                                                                 */
/*                        cIrbCal_ClinoSet                            */
/*                                                                 */
/* *************************************************************** */

cIrbCal_ClinoSet::cIrbCal_ClinoSet()
{
}

void cIrbCal_ClinoSet::AddData(const  cAuxAr2007 & anAux)
{
     MMVII::StdContAddData(cAuxAr2007("Set_Clinos",anAux),mVClinos);
}

void AddData(const  cAuxAr2007 & anAux,cIrbCal_ClinoSet & aSetClino)
{
    aSetClino.AddData(anAux);
}


cIrbCal_Clino1 * cIrbCal_ClinoSet::ClinoFromName(const std::string& aName)
{
    for (auto&  aClino : mVClinos)
        if (aClino.Name() == aName)
           return & aClino;
    return nullptr;
}

void cIrbCal_ClinoSet::AddClino(const std::string & aName,bool SVP)
{
   cIrbCal_Clino1 * aClino = ClinoFromName(aName);
   cIrbCal_Clino1 aNewClino (aName);
   if (aClino)
   {
       MMVII_INTERNAL_ASSERT_strong(SVP,"cIrbCal_Block::AddClino, cal already exist for " + aName);
       *aClino = aNewClino;
   }
   else
   {
      mVClinos.push_back(aNewClino);
   }
}

/* *************************************************************** */
/*                                                                 */
/*                        cIrbCal_Block                            */
/*                                                                 */
/* *************************************************************** */

const std::string  cIrbCal_Block::theDefaultName = "TheBlock";  /// in most application there is only one block

cIrbCal_Block::cIrbCal_Block(const std::string& aName) :
     mNameBloc (aName)
{
}


void  cIrbCal_Block::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("Cams",anAux),mSetCams);	
    MMVII::AddData(cAuxAr2007("Clinos",anAux),mSetClinos);	
}
void AddData(const  cAuxAr2007 & anAux,cIrbCal_Block & aRBoI)
{
    aRBoI.AddData(anAux);
}


const std::string &      cIrbCal_Block::NameBloc() const {return mNameBloc;}
cIrbCal_CamSet &        cIrbCal_Block::SetCams() {return mSetCams;}
const cIrbCal_CamSet &  cIrbCal_Block::SetCams() const {return mSetCams;}
cIrbCal_ClinoSet &      cIrbCal_Block::SetClinos() {return mSetClinos;}

/* *************************************************************** */
/*                                                                 */
/*                        cAppli_EditBlockInstr                    */
/*                                                                 */
/* *************************************************************** */

class cAppli_EditBlockInstr : public cMMVII_Appli
{
     public :

        cAppli_EditBlockInstr(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

        std::vector<std::string>  Samples() const override;

     private :
        cPhotogrammetricProject   mPhProj;
        std::string               mNameBloc;
	std::vector<std::string>  mVPatsIm4Cam;
	bool                      mFromScratch;
};

cAppli_EditBlockInstr::cAppli_EditBlockInstr(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli  (aVArgs,aSpec),
    mPhProj       (*this),
    mNameBloc     (cIrbCal_Block::theDefaultName),
    mFromScratch  (false)
{
}


std::vector<std::string>  cAppli_EditBlockInstr::Samples() const 
{
   return 
   {
       "MMVII EditBlockInstr Bl0  PatsIm4Cam='[.*_(.*).tif]' InMeasureClino=MesClin_043",
       "MMVII EditBlockInstr Bl0  PatsIm4Cam='[.*tif,.*_(.*).tif,Fils-100.xml]' InMeasureClino=MesClin_043"
   };
}


cCollecSpecArg2007 & cAppli_EditBlockInstr::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
     return anArgObl
             <<  mPhProj.DPBlockInstr().ArgDirInMand()
     ;

}

cCollecSpecArg2007 & cAppli_EditBlockInstr::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
	return anArgOpt
            << AOpt2007(mNameBloc,"NameBloc","Set the name of the bloc ",{{eTA2007::HDV}})
            << AOpt2007(mVPatsIm4Cam,"PatsIm4Cam","Pattern images [PatSelOnDisk,PatTimeStamp?,PatSelInBlock?]",{{eTA2007::ISizeV,"[1,3]"}})
            << AOpt2007(mFromScratch,"FromScratch","Do we start from a new file, even if already exist",{{eTA2007::HDV}})
            << mPhProj.DPBlockInstr().ArgDirOutOpt()
            << mPhProj.DPMeasuresClino().ArgDirInOpt()
        ;
}


int cAppli_EditBlockInstr::Exe() 
{
    mPhProj.DPBlockInstr().SetDirOutInIfNotInit();
    mPhProj.FinishInit();

    cIrbCal_Block *  aBlock =      mFromScratch                       ?
	                       new cIrbCal_Block                      :
	                       mPhProj.ReadRigBoI(mNameBloc,SVP::Yes) ;

    if (IsInit(&mVPatsIm4Cam))
    {
        std::string aPatSelOnDisk = mVPatsIm4Cam.at(0);
        std::string aPatTimeStamp = GetDef(mVPatsIm4Cam,1,aPatSelOnDisk);
        std::string aPatSelIm = GetDef(mVPatsIm4Cam,2,aPatTimeStamp);

        auto aVNameIm = ToVect(SetNameFromString(aPatSelOnDisk,true));
        for (const auto & aNameIm : aVNameIm)
        {
            std::string aNameCal = mPhProj.StdNameCalibOfImage(aNameIm);
	    aBlock->SetCams().AddCam(aNameCal,aPatTimeStamp,aPatSelIm,SVP::Yes);
        }
    }
    if (mPhProj.DPMeasuresClino().DirInIsInit())
    {
         cSetMeasureClino aMesClin =  mPhProj.ReadMeasureClino();
         for (const auto & aName : aMesClin.NamesClino())
         {
             aBlock->SetClinos().AddClino(aName,SVP::Yes);
         }
    }


    mPhProj.SaveRigBoI(*aBlock);

    delete aBlock;
    return EXIT_SUCCESS;
}


    /* ==================================================== */
    /*                                                      */
    /*               MMVII                                  */
    /*                                                      */
    /* ==================================================== */


tMMVII_UnikPApli Alloc_EditBlockInstr(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_EditBlockInstr(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_EditBlockInstr
(
     "BlockInstrEdit",
      Alloc_EditBlockInstr,
      "Create/Edit a block of instruments",
      {eApF::BlockInstr},
      {eApDT::BlockInstr},
      {eApDT::BlockInstr},
      __FILE__
);

/* *************************************************************** */
/*                                                                 */
/*               cAppli_BlockInstrInitCam                          */
/*                                                                 */
/* *************************************************************** */

class cAppli_BlockInstrInitCam : public cMMVII_Appli
{
     public :

        cAppli_BlockInstrInitCam(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
        int Exe() override;
        // std::vector<std::string>  Samples() const ;

     private :
        cPhotogrammetricProject   mPhProj;
        std::string               mSpecImIn;
        cIrbComp_Block *           mBlock;
        std::string               mNameBloc;
};


cAppli_BlockInstrInitCam::cAppli_BlockInstrInitCam(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mBlock       (nullptr),
    mNameBloc    (cIrbCal_Block::theDefaultName)
{
}

cCollecSpecArg2007 & cAppli_BlockInstrInitCam::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
     return anArgObl
             <<  Arg2007(mSpecImIn,"Pattern/file for images", {{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}}  )
             <<  mPhProj.DPBlockInstr().ArgDirInMand()
             <<  mPhProj.DPOrient().ArgDirInMand()
             <<  mPhProj.DPBlockInstr().ArgDirOutMand()
     ;
}

cCollecSpecArg2007 & cAppli_BlockInstrInitCam::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
	return anArgOpt
            << AOpt2007(mNameBloc,"NameBloc","Name of bloc to calib ",{{eTA2007::HDV}})
        ;
}

int cAppli_BlockInstrInitCam::Exe()
{
    mPhProj.FinishInit();

    mBlock = new cIrbComp_Block(mPhProj,mNameBloc);

    for (const auto & aNameIm :  VectMainSet(0))
    {
       mBlock->AddImagePose(aNameIm);
       // mBlock->AddImagePose(aNameIm);
    }

    for (size_t aKC1=0 ; aKC1<mBlock->NbCams(); aKC1++)
    {
        for (size_t aKC2=0 ; aKC2<mBlock->NbCams(); aKC2++)
        {
            const auto & [aPose,aSigma] = mBlock->ComputeCalibCamsInit(aKC1,aKC2);
            if (aKC1 < aKC2)
            {
                mBlock->CalBlock().SetCams().SetSigma(aSigma);
            }
            if ((int)aKC1==mBlock->CalBlock().SetCams().NumMaster())
            {
               StdOut() << "aKC2aKC2aKC2aKC2 " << aKC2 << "\n";
               mBlock->CalBlock().SetCams().KthCam(aKC2).SetPose(aPose);
            }
        }
    }

    mPhProj.SaveRigBoI(mBlock->CalBlock());

    delete mBlock;
    return EXIT_SUCCESS;
}

    /* ==================================================== */
    /*                                                      */
    /*               MMVII                                  */
    /*                                                      */
    /* ==================================================== */


tMMVII_UnikPApli Alloc_BlockInstrInitCam(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_BlockInstrInitCam(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_BlockInstrInitCam
(
     "BlockInstrInitCam",
      Alloc_BlockInstrInitCam,
      "Init  camera poses inside a block of instrument",
      {eApF::BlockInstr,eApF::Ori},
      {eApDT::BlockInstr,eApDT::Ori},
      {eApDT::BlockInstr},
      __FILE__
);

/* *************************************************************** */
/*                                                                 */
/*               cPhotogrammetricProject                           */
/*                                                                 */
/* *************************************************************** */

std::string   cPhotogrammetricProject::NameRigBoI(const std::string & aName,bool isIn) const
{
    return DPBlockInstr().FullDirInOut(isIn) + aName + "." + GlobTaggedNameDefSerial();
}

cIrbCal_Block *  cPhotogrammetricProject::ReadRigBoI(const std::string & aName,bool SVP) const
{
    std::string aFullName  = NameRigBoI(aName,IO::In);
    cIrbCal_Block * aRes = new cIrbCal_Block(aName);

    if (! ExistFile(aFullName))  // if it doesnt exist and we are OK, it return a new empty bloc
    {
        MMVII_INTERNAL_ASSERT_User_UndefE(SVP,"cIrbCal_Block file dont exist");
    }
    else
    {
        ReadFromFile(*aRes,aFullName);
    }

    return aRes;
}

void   cPhotogrammetricProject::SaveRigBoI(const cIrbCal_Block & aBloc) const
{
      SaveInFile(aBloc,NameRigBoI(aBloc.NameBloc(),IO::Out));
}


};

