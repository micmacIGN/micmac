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
  mIsInit  (false),
  mPoseInW (tPoseR::RandomIsom3D(10)),
  mNameIm  ()
{
}

void cIrbComp_Cam1::Init(const tPoseR& aPoseInW,const std::string & aNameIm)
{
    if (mIsInit)
    {
        MMVII_INTERNAL_ERROR("Multiple init in cIrbComp_Cam1 for : " + aNameIm);
    }
    mIsInit    = true;
    mPoseInW   = aPoseInW;
    mNameIm    = aNameIm;
}

bool cIrbComp_Cam1::IsInit() const {return mIsInit;}

tPoseR cIrbComp_Cam1::PosBInSysA(const cIrbComp_Cam1 & aCamB) const
{
    MMVII_INTERNAL_ASSERT_tiny(mIsInit&&(aCamB.mIsInit),"cIrbComp_Cam1::PosBInSysA no init");
    //      (A->W) -1  * (B->W) 
    return mPoseInW.MapInverse() * aCamB.mPoseInW;
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
void cIrbComp_CamSet::AddImagePose(int anIndex,const tPoseR& aPose,const std::string & aNameIm)
{
   mVCompPoses.at(anIndex).Init(aPose,aNameIm);
}

bool   cIrbComp_CamSet::HasPoseRel(size_t aK1,size_t aK2) const
{
    return mVCompPoses.at(aK1).IsInit() && mVCompPoses.at(aK2).IsInit() ;
}

tPoseR cIrbComp_CamSet::PoseRel(size_t aK1,size_t aK2) const
{
   return mVCompPoses.at(aK1).PosBInSysA(mVCompPoses.at(aK2));
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
     mPoseInBlock   (tPoseR::Identity())
{
}

cIrbCal_Cam1::cIrbCal_Cam1()  :
    cIrbCal_Cam1(-1,MMVII_NONE,MMVII_NONE,MMVII_NONE)
{
}

void cIrbCal_Cam1::SetPose(const tPoseR & aPose)
{
   mPoseInBlock = aPose;
   mIsInit      = true;
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

void cIrbCal_Cam1::AddData(const  cAuxAr2007 & anAux)
{
      MMVII::AddData(cAuxAr2007("Num",anAux),mNum);
      MMVII::AddData(cAuxAr2007("NameCalib",anAux),mNameCal);
      MMVII::AddData(cAuxAr2007("PatTimeStamp",anAux),mPatTimeStamp);
      MMVII::AddData(cAuxAr2007("SelIsPat",anAux),mSelIsPat);
      MMVII::AddData(cAuxAr2007("ImSelect",anAux),mImSelect);
      MMVII::AddData(cAuxAr2007("IsInit",anAux),mIsInit);
      MMVII::AddData(cAuxAr2007("Pose",anAux),mPoseInBlock);
}

void AddData(const  cAuxAr2007 & anAux,cIrbCal_Cam1 & aCam)
{
    aCam.AddData(anAux);
}

/* *************************************************************** */
/*                                                                 */
/*                        cIrb_SigmaPoseRel                        */
/*                                                                 */
/* *************************************************************** */

cIrb_SigmaInstr::cIrb_SigmaInstr() :
    cIrb_SigmaInstr(0.0,0.0,0.0,0.0)
{
}

cIrb_SigmaInstr::cIrb_SigmaInstr(tREAL8 aW,tREAL8 aSigTr,tREAL8 aSigRot,tREAL8 aSigGlob) :
    mSumW     (aW),
    mSumWTr   (aSigTr),
    mSumWRot  (aSigRot),
    mSumWGlob (aSigGlob)
{
}

void cIrb_SigmaInstr::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("SumW",anAux)     ,mSumW);
    MMVII::AddData(cAuxAr2007("SumWTr",anAux)   ,mSumWTr);     anAux.Ar().AddComment("SigTr="+ToStr(SigmaTr()));
    MMVII::AddData(cAuxAr2007("SumWRot",anAux)  ,mSumWRot);    anAux.Ar().AddComment("SigRot="+ToStr(SigmaRot()));
    MMVII::AddData(cAuxAr2007("SumWGlob",anAux) ,mSumWGlob);
}


void AddData(const  cAuxAr2007 & anAux,cIrb_SigmaInstr & aSig)
{
    aSig.AddData(anAux);
}


void  cIrb_SigmaInstr::AddNewSigma(const cIrb_SigmaInstr & aS2, const tREAL8 &aW)
{
  mSumW     += aW * aS2.mSumW;
  mSumWTr   += aW * aS2.mSumWTr;
  mSumWRot  += aW * aS2.mSumWRot;
  mSumWGlob += aW * aS2.mSumWGlob;
}

tREAL8 cIrb_SigmaInstr::SigmaTr() const
{
   MMVII_INTERNAL_ASSERT_tiny(mSumWTr>0,"cIrb_SigmaInstr::SigmaGlob");
   return mSumWTr / mSumW;
}

tREAL8 cIrb_SigmaInstr::SigmaRot() const
{
   MMVII_INTERNAL_ASSERT_tiny(mSumWRot>0,"cIrb_SigmaInstr::SigmaGlob");
   return mSumWRot / mSumW;
}

tREAL8 cIrb_SigmaInstr::SigmaGlob() const
{
   MMVII_INTERNAL_ASSERT_tiny(mSumWGlob>0,"cIrb_SigmaInstr::SigmaGlob");
   return mSumWGlob / mSumW;
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
       *aCam = aNewCam;
   }
   else
   {
      mVCams.push_back(aNewCam);
   }
}

size_t  cIrbCal_CamSet::NbCams() const { return  mVCams.size();}
int     cIrbCal_CamSet::NumMaster() const
{
    return mNumMaster;
}

void  cIrbCal_CamSet::SetNumMaster(int aNum)
{
    mNumMaster = aNum;
}

cIrbCal_Cam1 &       cIrbCal_CamSet::KthCam(size_t aK)       {return  mVCams.at(aK);}
const cIrbCal_Cam1 & cIrbCal_CamSet::KthCam(size_t aK) const {return  mVCams.at(aK);}

cIrbCal_Cam1 * cIrbCal_CamSet::CamFromNameCalib(const std::string& aNameCalib,bool SVP)
{
    for (auto&  aCam : mVCams)
        if (aCam.NameCal() == aNameCalib)
           return & aCam;
    MMVII_INTERNAL_ASSERT_strong(SVP,"Cannot get calib for camera " + aNameCalib);
    return nullptr;
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

