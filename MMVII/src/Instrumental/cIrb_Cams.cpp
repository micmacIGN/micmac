#include "MMVII_InstrumentalBlock.h"

/**
  \file cIrb_Cams.cpp

  \brief This file contains the class relative to cameras in rigid blocks
*/

namespace MMVII
{


/* *************************************************************** */
/*                                                                 */
/*                        cIrbComp_Cam1                           */
/*                                                                 */
/* *************************************************************** */

cIrbComp_Cam1::cIrbComp_Cam1() :
  mCamPC (nullptr)
{
}

void cIrbComp_Cam1::Init(cSensorCamPC * aCamPC)
{
    if (mCamPC)
    {
        MMVII_INTERNAL_ERROR("Multiple init in cIrbComp_Cam1 for : " + mCamPC->NameImage());
    }
    mCamPC = aCamPC;
}

bool cIrbComp_Cam1::IsInit() const {return mCamPC!=nullptr;}
const  cSensorCamPC * cIrbComp_Cam1::CamPC() const {return mCamPC;}
cSensorCamPC * cIrbComp_Cam1::CamPC() {return mCamPC;}
tPoseR  cIrbComp_Cam1::Pose() const {return mCamPC->Pose();}
std::string cIrbComp_Cam1::NameIm() const{return mCamPC->NameImage();}

tPoseR cIrbComp_Cam1::PosBInSysA(const cIrbComp_Cam1 & aCamB) const
{
    MMVII_INTERNAL_ASSERT_tiny(mCamPC&&(aCamB.mCamPC),"cIrbComp_Cam1::PosBInSysA no init");
    //      (A->W) -1  * (B->W) 
    return mCamPC->Pose().MapInverse() * aCamB.mCamPC->Pose();
}

/* *************************************************************** */
/*                                                                 */
/*                        cIrbComp_CamSet                          */
/*                                                                 */
/* *************************************************************** */

cIrbComp_CamSet::cIrbComp_CamSet(const cIrbComp_Block & aCompBlock) :
    mBlock          (aCompBlock),
    mVCompPoses     (aCompBlock.SetOfCalibCams().NbCams())
{
}
void cIrbComp_CamSet::AddImagePose(int anIndex,cSensorCamPC * aCamPC)
{
   mVCompPoses.at(anIndex).Init(aCamPC);
}

bool   cIrbComp_CamSet::HasPoseRel(size_t aK1,size_t aK2) const
{
    return mVCompPoses.at(aK1).IsInit() && mVCompPoses.at(aK2).IsInit() ;
}

tPoseR cIrbComp_CamSet::PoseRel(size_t aK1,size_t aK2) const
{
   return mVCompPoses.at(aK1).PosBInSysA(mVCompPoses.at(aK2));
}

const std::vector<cIrbComp_Cam1> &  cIrbComp_CamSet::VCompPoses() const {return mVCompPoses;}

cIrbComp_Cam1 & cIrbComp_CamSet::KthCam(int aK)
{
    return  mVCompPoses.at(aK);
}


/* *************************************************************** */
/*                                                                 */
/*                        cIrbCal_Cam1                            */
/*                                                                 */
/* *************************************************************** */

cIrbCal_Cam1::cIrbCal_Cam1(int aNum,const std::string & aNameCal,const std::string & aTimeStamp,const std::string & aPatImSel) :
     mNum           (aNum),
     mNameCal       (aNameCal),
     mPatTimeStamp  (aTimeStamp),
     mSelIsPat      (true),
     mImSelect      (aPatImSel),
     mIsInit        (false),
     mPoseInBlock   (nullptr)
{
}

cIrbCal_Cam1::~cIrbCal_Cam1()
{/*
    if (mPoseInBlock)
        mPoseInBlock->OUK_Reset();
        */
    delete mPoseInBlock;
}

cIrbCal_Cam1::cIrbCal_Cam1()  :
    cIrbCal_Cam1(-1,MMVII_NONE,MMVII_NONE,MMVII_NONE)
{
}

void cIrbCal_Cam1::UnInit()
{
   delete mPoseInBlock;
    mPoseInBlock = nullptr;
    mIsInit = false;
}


void cIrbCal_Cam1::SetPose(const tPoseR & aPose)
{

    if (mPoseInBlock==nullptr)
        mPoseInBlock = new cPoseWithUK;

   mPoseInBlock->SetPose(aPose);

   mIsInit      = true;

}

bool          cIrbCal_Cam1::IsInit() const{  return mIsInit;}
const tPoseR & cIrbCal_Cam1::PoseInBlock() const
{
    MMVII_INTERNAL_ASSERT_tiny(mIsInit,"IrbCal_Cam1::PoseInBlock");
    return mPoseInBlock->Pose();
}

tPoseR cIrbCal_Cam1::PosBInSysA(const cIrbCal_Cam1 & aCamB) const
{
    MMVII_INTERNAL_ASSERT_tiny(mIsInit&&(aCamB.mIsInit),"cIrbCal_Cam1::PosBInSysA no init");
    //      (A->W) -1  * (B->W)
    return PoseInBlock().MapInverse() * aCamB.PoseInBlock();
}


std::string  cIrbCal_Cam1::TimeStamp(const std::string & aNameImage) const
{
  return  ReplacePattern(mPatTimeStamp,"$1",aNameImage);
}

bool  cIrbCal_Cam1::ImageIsInBlock(const std::string & aNameImage) const
{
  return  MatchRegex(aNameImage,mImSelect);
}


const std::string & cIrbCal_Cam1::NameCal() const { return mNameCal; }
int cIrbCal_Cam1::Num() const {return mNum;}


cPoseWithUK&  cIrbCal_Cam1::PoseUKInBlock()
{
    MMVII_INTERNAL_ASSERT_tiny(mIsInit,"IrbCal_Cam1::PoseUKInBlock");
    return *mPoseInBlock;
}


void cIrbCal_Cam1::AddData(const  cAuxAr2007 & anAux)
{
      MMVII::AddData(cAuxAr2007("Num",anAux),mNum);
      MMVII::AddData(cAuxAr2007("NameCalib",anAux),mNameCal);
      MMVII::AddData(cAuxAr2007("PatTimeStamp",anAux),mPatTimeStamp);
      MMVII::AddData(cAuxAr2007("SelIsPat",anAux),mSelIsPat);
      MMVII::AddData(cAuxAr2007("ImSelect",anAux),mImSelect);
      MMVII::AddData(cAuxAr2007("IsInit",anAux),mIsInit);

      tPoseR aPose = mPoseInBlock ? mPoseInBlock->Pose() : tPoseR::Identity();

      MMVII::AddData(cAuxAr2007("Pose",anAux),aPose);
      if (anAux.Input())
      {
          SetPose(aPose);
      }

}

void AddData(const  cAuxAr2007 & anAux,cIrbCal_Cam1 & aCam)
{
    aCam.AddData(anAux);
}




/* *************************************************************** */
/*                                                                 */
/*                        cIrbCal_CamSet                           */
/*                                                                 */
/* *************************************************************** */

cIrbCal_CamSet::cIrbCal_CamSet()  :
    mNumMaster (-1)
{
}


void  cIrbCal_CamSet::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("NumMaster",anAux),mNumMaster);
    MMVII::StdContAddData(cAuxAr2007("Set_Cams",anAux),mVCams);
    
    // check the coherence of num master
    MMVII_INTERNAL_ASSERT_strong
    (
         (mNumMaster>=-1) && (mNumMaster<(int)mVCams.size()),
         "Bad num master for cIrbCal_CamSet"
    );
}
void AddData(const  cAuxAr2007 & anAux,cIrbCal_CamSet & aCams)
{
    aCams.AddData(anAux);
}


void cIrbCal_CamSet::AddCam
     (
         const std::string & aNameCalib,
	 const std::string & aTimeStamp,
	 const std::string & aPatImSel,
	 bool OkAlreadyExist 
     )
{
   cIrbCal_Cam1 * aCam = CamFromNameCalib(aNameCalib,SVP::Yes);
   int aNum = aCam ?  aCam->Num() : int(mVCams.size()) ;
   cIrbCal_Cam1 aNewCam (aNum,aNameCalib,aTimeStamp,aPatImSel);
   // in case already exist, we may ovewrite (multiple edit)
   if (aCam)
   {
       MMVII_INTERNAL_ASSERT_strong(OkAlreadyExist,"cIrbCal_Block::AddCam, cal already exist for " + aNameCalib);
       aCam->UnInit();
       *aCam = aNewCam;
   }
   else
   {
      mVCams.push_back(aNewCam);
   }
}


size_t  cIrbCal_CamSet::NbCams() const { return  mVCams.size();}
int     cIrbCal_CamSet::NumMaster() const{    return mNumMaster;}
cIrbCal_Cam1 &  cIrbCal_CamSet::MasterCam() {return mVCams.at(mNumMaster);}


void  cIrbCal_CamSet::SetNumMaster(int aNum)
{
    mNumMaster = aNum;
}


cIrbCal_Cam1 &       cIrbCal_CamSet::KthCam(size_t aK)       {return  mVCams.at(aK);}
const cIrbCal_Cam1 & cIrbCal_CamSet::KthCam(size_t aK) const {return  mVCams.at(aK);}

std::vector<cIrbCal_Cam1> &      cIrbCal_CamSet::VCams()
{
    return mVCams;
}
cIrbCal_Cam1 * cIrbCal_CamSet::CamFromNameCalib(const std::string& aNameCalib,bool SVP)
{
    int aK = IndexCamFromNameCalib(aNameCalib,SVP);
    return (aK>=0) ? & mVCams.at(aK)  : nullptr;
}
int cIrbCal_CamSet::IndexCamFromNameCalib(const std::string& aNameCalib,bool SVP)
{
    for (size_t aK=0 ; aK< mVCams.size() ; aK++)
        if ( mVCams.at(aK).NameCal() == aNameCalib)
           return aK;
    MMVII_INTERNAL_ASSERT_strong(SVP,"Cannot get calib for camera " + aNameCalib);
    return -1;
}

tPoseR cIrbCal_CamSet::PoseRel(size_t aK1,size_t aK2) const
{
   return mVCams.at(aK1).PosBInSysA(mVCams.at(aK2));
}


/*
void cIrbCal_CamSet::SetSigma(const cIrb_SigmaPoseRel& aNewS)
{
   bool isFound=false;
   for (auto & aSigm : mVSigmas)
   {
       if ((aSigm.mK1==aNewS.mK1) && (aSigm.mK2==aNewS.mK2))
       {
           isFound=true;
           aSigm = aNewS;
       }
   }

   if (!isFound)
      mVSigmas.push_back(aNewS);
}
*/

};

