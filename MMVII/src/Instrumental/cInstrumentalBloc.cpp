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
/*                        cIrb_SigmaInstr                          */
/*                                                                 */
/* *************************************************************** */


cIrb_SigmaInstr::cIrb_SigmaInstr() :
    cIrb_SigmaInstr(0.0,0.0,0.0,0.0)
{
}


cIrb_SigmaInstr::cIrb_SigmaInstr(tREAL8 aWTr,tREAL8 aWRot,tREAL8 aSigTr,tREAL8 aSigRot) :
    mAvgSigTr   (),
    mAvgSigRot  ()
{
    mAvgSigTr.Add(aWTr,aSigTr);
    mAvgSigRot.Add(aWRot,aSigRot);
}

void cIrb_SigmaInstr::AddData(const  cAuxAr2007 & anAux)
{
 //   StdOut() << "cIrb_SigmaInstr::AddData " << mAvgSigTr.Nb() << " " << mAvgSigRot.Nb() << "\n";

    MMVII::AddData(cAuxAr2007("Tr",anAux) ,mAvgSigTr);
    if (mAvgSigTr.SW() > 0)
       anAux.Ar().AddComment("SigTr="+ToStr(mAvgSigTr.Average()));

    MMVII::AddData(cAuxAr2007("Rot",anAux) ,mAvgSigRot);
    if (mAvgSigRot.SW() >0 )
    {
       anAux.Ar().AddComment
       (
             "SigRot="
           + ToStr(mAvgSigRot.Average()) + " rad, "
           + ToStr(Rad2DMgon(mAvgSigRot.Average())) + " dmgon "
       );
    }
}


void AddData(const  cAuxAr2007 & anAux,cIrb_SigmaInstr & aSig)
{
    aSig.AddData(anAux);
}



void  cIrb_SigmaInstr::AddNewSigma(const cIrb_SigmaInstr & aS2)
{
  mAvgSigTr.Add(aS2.mAvgSigTr);
  mAvgSigRot.Add(aS2.mAvgSigRot);
}

tREAL8 cIrb_SigmaInstr::SigmaTr() const
{
   return mAvgSigTr.Average();
}

tREAL8 cIrb_SigmaInstr::SigmaRot() const
{
    return mAvgSigRot.Average();

}

/* *************************************************************** */
/*                                                                 */
/*                        cIrb_Desc1Intsr                          */
/*                                                                 */
/* *************************************************************** */


cIrb_Desc1Intsr::cIrb_Desc1Intsr (eTyInstr aType,const std::string & aNameInstr ) :
    mType      (aType),
    mNameInstr (aNameInstr)
{
}

cIrb_Desc1Intsr::cIrb_Desc1Intsr():
    cIrb_Desc1Intsr(eTyInstr::eNbVals,MMVII_NONE)
{
}

void cIrb_Desc1Intsr::AddData(const  cAuxAr2007 & anAux)
{
    EnumAddData(anAux,mType,"Type");
    MMVII::AddData(cAuxAr2007("NameInstr",anAux) ,mNameInstr);
    MMVII::AddData(cAuxAr2007("Sigma",anAux) ,mSigma);
}

void AddData(const  cAuxAr2007 & anAux,cIrb_Desc1Intsr & aDesc)
{
   aDesc.AddData(anAux);
}

const cIrb_SigmaInstr & cIrb_Desc1Intsr::Sigma() const {return mSigma;}

void cIrb_Desc1Intsr::SetSigma(const cIrb_SigmaInstr& aSigma)
{
   mSigma = aSigma;
}

void cIrb_Desc1Intsr::ResetSigma()
{
    SetSigma(cIrb_SigmaInstr());
}

void  cIrb_Desc1Intsr::AddNewSigma(const cIrb_SigmaInstr& aSigAdd)
{
    mSigma.AddNewSigma(aSigAdd);
}

eTyInstr             cIrb_Desc1Intsr::Type() const {return mType;}
const std::string &  cIrb_Desc1Intsr::NameInstr() const {return mNameInstr;}


/* *************************************************************** */
/*                                                                 */
/*                        cIrbComp_TimeS                             */
/*                                                                 */
/* *************************************************************** */

cIrbComp_TimeS::cIrbComp_TimeS (const cIrbComp_Block & aCompBlock) :
    mCompBlock       (aCompBlock),
    mSetCams         (aCompBlock),
    mPoseInstrIsInit (false),
    mPoseInstr       (tPoseR::Identity())
{
}

const cIrbComp_CamSet & cIrbComp_TimeS::SetCams() const {return mSetCams;}
cIrbComp_CamSet & cIrbComp_TimeS::SetCams() {return mSetCams;}
const cIrbComp_ClinoSet & cIrbComp_TimeS::SetClino() const {return mSetClino;}

const cIrbComp_Block & cIrbComp_TimeS::CompBlock() const {return mCompBlock;}

const cIrbCal_Block & cIrbComp_TimeS::CalBlock() const{return  mCompBlock.CalBlock();}

void cIrbComp_TimeS::SetClinoValues(const cOneMesureClino& aMeasure)
{
    mSetClino.SetClinoValues(aMeasure);
}

// cIrbComp_Block & cIrbComp_TimeS::CompBlock()  {return mCompBlock;}

void cIrbComp_TimeS::ComputePoseInstrument(const std::vector<int>& aSetNumCam,bool SVP)
{
    mPoseInstrIsInit = false;
   // static tTypeMap  Centroid(const std::vector<tTypeMap> & aV,const std::vector<Type> &);
    std::vector<tPoseR> aVPose;
    std::vector<tREAL8> aVWeight;

    // tREAL8 aSumW = 0;
    const cIrbCal_CamSet & aSetCalCams = mCompBlock.SetOfCalibCams() ;


   // for (size_t aKP=0 ; aKP< aSetCalCams.NbCams() ; aKP++)
    for (auto aKP : aSetNumCam)
    {
         const cIrbCal_Cam1 & aCalCams = aSetCalCams.KthCam(aKP);
         const cIrbComp_Cam1 & aCompCam = mSetCams.KthCam(aKP) ;
         if (aCalCams.IsInit() && aCompCam.IsInit())
         {
             const  cIrb_SigmaInstr & aSigma  = CalBlock().DescrIndiv(aCalCams.NameCal()).Sigma();
             tREAL8 aSig2 = Square(aSigma.SigmaRot()) +  Square(aSigma.SigmaTr()) ;
             //  PoseCal  :  Cam->CalCoord     PoseIm  Cam->Word
             //  PoseCal * PoseIm-1  :  Word -> CalCoord
             tPoseR aPoseCam2Word = aCompCam.Pose();
             tPoseR aPoseCam2Cal = aCalCams.PoseInBlock();
             tPoseR aPosWord2Cal = aPoseCam2Cal*aPoseCam2Word.MapInverse();

           //  StdOut()  << "    * PPPppPPp " << aPosWord2Cal.Tr() << " " << aPosWord2Cal.Rot().ToWPK() << "\n";

             aVWeight.push_back(1/aSig2);
             aVPose.push_back(aPosWord2Cal);
         }
    }
//    StdOut() << " ============================================================\n";
    if (! aVWeight.empty())
    {
        mPoseInstrIsInit = true;
        mPoseInstr = tPoseR::Centroid(aVPose,aVWeight);
    }
    else
    {
        MMVII_INTERNAL_ASSERT_tiny(SVP,"Cannot do ComputePoseInstrument");
    }
}

tREAL8 cIrbComp_TimeS::ScoreDirClino(const cPt3dr& aDirClino,size_t aKClino) const
{
    cPt3dr aDirLoc = mPoseInstr.Rot().Inverse(aDirClino);
    // cPt3dr aDirLoc = mPoseInstr.Rot().Value(aDirClino);
/*
    StdOut () << " aKClino " << aKClino << " " << mSetClino.NbMeasure() << "\n";
    std::abs(aDirLoc.z() - std::sin(mSetClino.KthMeasure(aKClino).Angle()) );
    StdOut() << " ScoreDirClino " << __LINE__ << "\n";
*/

    return std::abs(aDirLoc.z() - std::sin(mSetClino.KthMeasure(aKClino).Angle()) );

    // return std::abs(aDirLoc.z() - mSetClino.KthMeasure(aKClino).Angle() );

}



/* *************************************************************** */
/*                                                                 */
/*                        cIrbComp_Block                            */
/*                                                                 */
/* *************************************************************** */

    //  -------------------------- "Constructors"  --------------------------------------------------------


cIrbComp_Block::cIrbComp_Block( cIrbCal_Block * aCalBlock,bool IsAdopted) :
   mCalBlock   (aCalBlock),
   mCalIsAdopted (IsAdopted),
   mPhProj (nullptr)
{
}


cIrbComp_Block::~cIrbComp_Block()
{
   if (mCalIsAdopted)
      delete mCalBlock;
}


cIrbComp_Block::cIrbComp_Block(const std::string & aNameFile) :
    cIrbComp_Block (new  cIrbCal_Block,true)
{
    ReadFromFile(*mCalBlock,aNameFile);
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

void cIrbComp_Block::AddImagePose(cSensorCamPC * aCamPC,bool okImNotInBloc)
{
    // extract the name of the calibration 
    std::string aNameIm = aCamPC->NameImage();
    tPoseR aPose = aCamPC->Pose();
    std::string aNameCal = PhProj().StdNameCalibOfImage(aNameIm);

    // extract the specification of the camera in the block
    cIrbCal_Cam1 *  aCInRBoI = mCalBlock->mSetCams.CamFromNameCalib(aNameCal,okImNotInBloc);
    if (aCInRBoI==nullptr)
       return;

    // if the image does not belong to this block
    if (!aCInRBoI->ImageIsInBlock(aNameIm))
    {
        MMVII_INTERNAL_ASSERT_tiny(okImNotInBloc,"Image is not in bloc : "+aNameIm);
        return;
    }

    // extract time stamp
    std::string aTimeS = aCInRBoI->TimeStamp(aNameIm);
    // cIrbComp_TimeS &  cIrbComp_Block::DataOfTimeS(const std::string & aTS)
    cIrbComp_TimeS &  aDataTS =  DataOfTimeS(aTimeS);

    // StdOut() << " III=" << aNameIm << " CCC=" << aNameCal << " Ptr=" << aTimeS << "\n";
    aDataTS.mSetCams.AddImagePose(aCInRBoI->Num(),aCamPC);
}

void cIrbComp_Block::AddImagePose(const std::string & aNameIm,bool  okImNotInBloc)
{
 //   bool hasPose;
    cSensorCamPC * aCamPC = PhProj().ReadCamPC(aNameIm,DelAuto::Yes,SVP::Yes);

  //  tPoseR aPose = PhProj().ReadPoseCamPC(aNameIm,&hasPose);
    if (aCamPC!=0)
    {
         AddImagePose(aCamPC,okImNotInBloc);
    }
}


void cIrbComp_Block::SetClinoValues(const cSetMeasureClino& aSetM,bool OkNewTimeS)
{
    MMVII_INTERNAL_ASSERT_tiny(aSetM.NamesClino() == mCalBlock->SetClinos().VNames(),"Names differs in SetClinoValues");
   for (const auto & aMeasure : aSetM.SetMeasures())
   {
       // we test before, because in case does not exist, it will
       if (!OkNewTimeS)
       {
           MMVII_INTERNAL_ASSERT_tiny(MapBoolFind(mDataTS,aMeasure.Ident()),"SetClinoValues new clino ident refuted for "+aMeasure.Ident());
       }
       cIrbComp_TimeS &     aTS = DataOfTimeS(aMeasure.Ident());
       aTS.SetClinoValues(aMeasure);
   }
}

void cIrbComp_Block::SetClinoValues(bool OkNewTimeS)
{
    cSetMeasureClino aSetMeasures = mPhProj->ReadMeasureClino();
    SetClinoValues(aSetMeasures,OkNewTimeS);
}


    //  -------------------------- "computation"  --------------------------------------------
void cIrbComp_Block::ComputePoseInstrument(bool SVP)
{
    std::vector<int> aSetNumCam = SetOfCalibCams().NumPoseInstr();

StdOut() << "ComputePoseInstrumentComputePoseInstrument= " << aSetNumCam << "\n";

    for (auto & [aTimes,aDataTS] : mDataTS)
        aDataTS.ComputePoseInstrument(aSetNumCam,SVP);
}

typename cIrbComp_Block::tResCompCal cIrbComp_Block::ComputeCalibCamsInit(int aKC1,int aKC2) const
{
   // [0]  Compute relative poses for each time stamps where it exist
   std::vector<tPoseR> aVPoseRel;  // vector of existing relative pose
   std::vector<std::string> aVTS;  // vector of existing time_stamp
   for (const auto & [aName,aDataTS] :  mDataTS) // parse Time-Stamps and data associated
   {
       const cIrbComp_CamSet & aSetC = aDataTS.SetCams(); // extract data for cameras
        if (aSetC.HasPoseRel(aKC1,aKC2))
        {
            tPoseR aPose = aSetC.PoseRel(aKC1,aKC2);
            aVPoseRel.push_back(aPose);
            aVTS.push_back(aName);
        }
   }
   int aNbP = aVPoseRel.size();
   if (aNbP<2)
       return  tResCompCal(-1.0,tPoseR::Identity(),cIrb_SigmaInstr());

   // [1]  Compute medians of residuals, used to have an order of magnitude of the sigmas
   // required for mixing sigma on tr with sigma on rot
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
   

   // [2]  Exract the robust center, ie the one minimizing the sum of distance
   // to the other
   int    aK1Min    = -1;
   tREAL8 aMinDGlob = 1e10;
   tREAL8 aMinDTr   = 1e10;
   tREAL8 aMinDRot  = 1e10;
   tREAL8 aRatioTrRot = (aMedTr/aMedRot);
   tREAL8 aRatioNb = 1.0 / tREAL8(aNbP-1);

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
            aSumDGlob += ( aDTr + aDRot * aRatioTrRot) / (1+aRatioTrRot);
       }
       // StdOut() << "SOM=" << aSumDGlob/aVPoseRel.size() << "\n";
       if (aSumDGlob<aMinDGlob )
       {
           aK1Min    = aKP1;
           aMinDGlob = aSumDGlob * aRatioNb;
           aMinDTr   = aSumDTr   * aRatioNb;
           aMinDRot  = aSumDRot  * aRatioNb;
       }
   }


   /*
   StdOut() << "K1/K2=" << aKC1<< aKC2 
            << " Med , Tr=" <<  aMedTr << " Rot=" <<  aMedRot
	    << " TS "  << aVTS.at(aK1Min) << " DTr=" <<  aMinDTr << " DRot=" << aMinDRot 
	    << "\n";
	    */

   return tResCompCal(aMinDGlob,aVPoseRel.at(aK1Min), cIrb_SigmaInstr (aNbP,aNbP,aMinDTr,aMinDRot));
}


    //  -------------------------- "Accessors"  --------------------------------------------------------
   
const cIrbCal_CamSet &  cIrbComp_Block::SetOfCalibCams() const { return mCalBlock->SetCams(); }
const cPhotogrammetricProject & cIrbComp_Block::PhProj()
{
    MMVII_INTERNAL_ASSERT_strong(mPhProj,"No PhProj for cIrbComp_Block");
    return *mPhProj;
}
const cIrbCal_Block & cIrbComp_Block::CalBlock() const {return *mCalBlock;}
cIrbCal_Block & cIrbComp_Block::CalBlock() {return *mCalBlock;}
size_t  cIrbComp_Block::NbCams() const  {return SetOfCalibCams().NbCams();}

const typename cIrbComp_Block::tContTimeS & cIrbComp_Block::DataTS() const {return mDataTS;}
 typename cIrbComp_Block::tContTimeS & cIrbComp_Block::DataTS()  {return mDataTS;}


tREAL8 cIrbComp_Block::ScoreDirClino(const cPt3dr& aDir,size_t aKClino) const
{
    cWeightAv<tREAL8,tREAL8>  aWAvg;

     for (const auto & [aTimeS,aDataTS]: mDataTS)
     {
         // StdOut() << " DDDTssSS " << aTimeS << "\n";
         if (aKClino < aDataTS.SetClino().NbMeasure())
            aWAvg.Add(1.0,aDataTS.ScoreDirClino(aDir,aKClino));
     }

     return aWAvg.Average();
}


/* *************************************************************** */
/*                                                                 */
/*                        cIrb_CstrRelRot                          */
/*                                                                 */
/* *************************************************************** */

cIrb_CstrRelRot::cIrb_CstrRelRot(const tRotR & anOri,const tREAL8 & aSigma) :
    mOri    (anOri),
    mSigma (aSigma)
{
}

cIrb_CstrRelRot::cIrb_CstrRelRot() :
    cIrb_CstrRelRot(tRotR::Identity(),-1.0)
{

}

void cIrb_CstrRelRot::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("Ori",anAux),mOri);
    MMVII::AddData(cAuxAr2007("Sigma",anAux),mSigma);

}

void AddData(const  cAuxAr2007 & anAux,cIrb_CstrRelRot & aICRR)
{
    aICRR.AddData(anAux);
}

/* *************************************************************** */
/*                                                                 */
/*                        cIrb_CstrOrthog                          */
/*                                                                 */
/* *************************************************************** */


cIrb_CstrOrthog::cIrb_CstrOrthog(const tREAL8 & aSigma) :
    mSigma (aSigma)
{
}

cIrb_CstrOrthog::cIrb_CstrOrthog() :
    cIrb_CstrOrthog(-1.0)
{

}


tREAL8 cIrb_CstrOrthog::Sigma() const {return mSigma;}


void cIrb_CstrOrthog::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("Sigma",anAux),mSigma);

}

void AddData(const  cAuxAr2007 & anAux,cIrb_CstrOrthog & aICO)
{
    aICO.AddData(anAux);
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
    mSetCams.mCalBlock = this;
    mSetClinos.mCalBlock = this;

}


void  cIrbCal_Block::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("Cams",anAux),mSetCams);	
    MMVII::AddData(cAuxAr2007("Clinos",anAux),mSetClinos);	

    MMVII::StdMapAddData(cAuxAr2007("SigmasPairs",anAux),mSigmaPair);
    MMVII::StdMapAddData(cAuxAr2007("DescrIndiv",anAux),mDescrIndiv);
    MMVII::StdMapAddData(cAuxAr2007("CstrRelRot",anAux),mCstrRelRot);
    MMVII::StdMapAddData(cAuxAr2007("CstrOrthog",anAux),mCstrOrthog);
}

void AddData(const  cAuxAr2007 & anAux,cIrbCal_Block & aRBoI)
{
    aRBoI.AddData(anAux);
}

cIrb_Desc1Intsr &  cIrbCal_Block::AddSigma_Indiv(std::string aNameInstr,eTyInstr aTypeInstr)
{
    auto  anIter = mDescrIndiv.find(aNameInstr);
    if (anIter== mDescrIndiv.end())
    {
        mDescrIndiv[aNameInstr] = cIrb_Desc1Intsr(aTypeInstr,aNameInstr);
        anIter = mDescrIndiv.find(aNameInstr);
    }
    cIrb_Desc1Intsr &  anInstr = anIter->second;

    MMVII_INTERNAL_ASSERT_tiny(anInstr.Type()== aTypeInstr,"Chang type in cIrbCal_Block::AddSigma_Indiv");
    MMVII_INTERNAL_ASSERT_tiny(anInstr.NameInstr()== aNameInstr,"Chang type in cIrbCal_Block::AddSigma_Indiv");

    return anInstr;
}

void  cIrbCal_Block::AddSigma_Indiv(std::string aNameInstr,eTyInstr aTypeInstr, const cIrb_SigmaInstr & aSigma)
{
    AddSigma_Indiv(aNameInstr,aTypeInstr).AddNewSigma(aSigma);
}

void  cIrbCal_Block::AddCstrRelRot(std::string aN1,std::string aN2,tREAL8 aSigma,tRotR anOri)
{
   if (aN1>aN2)
   {
       std::swap(aN1,aN2);
       anOri = anOri.MapInverse();
   }

   DescrIndiv(aN1);
   DescrIndiv(aN2);

   mCstrRelRot[tNamePair(aN1,aN2)] = cIrb_CstrRelRot(anOri,aSigma);
}

void  cIrbCal_Block::AddCstrRelOrthog(std::string aN1,std::string aN2,tREAL8 aSigma)
{
   if (aN1>aN2)
   {
       std::swap(aN1,aN2);
   }

   DescrIndiv(aN1);
   DescrIndiv(aN2);

   mCstrOrthog[tNamePair(aN1,aN2)] = cIrb_CstrOrthog(aSigma);
}

void cIrbCal_Block::AddSigma(std::string aN1,eTyInstr aType1,std::string aN2,eTyInstr aType2,const cIrb_SigmaInstr & aSig)
{
    AddSigma_Indiv(aN1,aType1,aSig);
    AddSigma_Indiv(aN2,aType2,aSig);
    // mSigmaInd[aN1].AddNewSigma(aSig);
    // mSigmaInd[aN2].AddNewSigma(aSig);
    mSigmaPair[tNamePair(aN1,aN2)].AddNewSigma(aSig);
}


const std::string &      cIrbCal_Block::NameBloc() const {return mNameBloc;}
cIrbCal_CamSet &        cIrbCal_Block::SetCams() {return mSetCams;}
const cIrbCal_CamSet &  cIrbCal_Block::SetCams() const {return mSetCams;}
cIrbCal_ClinoSet &      cIrbCal_Block::SetClinos() {return mSetClinos;}


const  std::map<tNamePair,cIrb_SigmaInstr> &  cIrbCal_Block::SigmaPair() const {return mSigmaPair; }
const std::map<std::string,cIrb_Desc1Intsr> & cIrbCal_Block::DescrIndiv() const {return mDescrIndiv;}
const std::map<tNamePair,cIrb_CstrOrthog> &  cIrbCal_Block::CstrOrthog() const {return mCstrOrthog;}

void cIrbCal_Block::SetSigmaPair(const  std::map<tNamePair,cIrb_SigmaInstr> & aSigmaPair)
{

    StdOut () << "    ============  SIGMA by PAIR, for cams in bloc ===========\n";
    for (const auto&  [aCple,aSig] : aSigmaPair )
    {
        StdOut()
                << " Tr=" << aSig.SigmaTr() << " Rot=" << aSig.SigmaRot()
                << " : " << aCple.V1() << " " << aCple.V2()
                << "\n";
    }
    mSigmaPair = aSigmaPair;
}

void cIrbCal_Block::SetSigmaIndiv(const  std::map<tNamePair,cIrb_SigmaInstr> & aSigmaPair)
{

    StdOut() << "AddNewSigma SetSigmaIndiv \n";
    std::set<std::string> aSetInstr;
    for (const auto&  [aCple,aSig] : aSigmaPair )
    {
        aSetInstr.insert(aCple.V1());
        aSetInstr.insert(aCple.V2());
    }

    for (const auto&   aNameInstr : aSetInstr )
    {
        NC_DescrIndiv(aNameInstr).ResetSigma();
    }

    for (const auto&  [aCple,aSig] : aSigmaPair )
    {
        NC_DescrIndiv(aCple.V1()).AddNewSigma(aSig);
        NC_DescrIndiv(aCple.V2()).AddNewSigma(aSig);
        //StdOut() << " TTRRR=" << NC_DescrIndiv(aCple.V1()).Sigma().SigmaTr() << "\n";
    }
}


const cIrb_Desc1Intsr &  cIrbCal_Block::DescrIndiv(const std::string & aNameInstr) const
{
   return *(MapGet(mDescrIndiv,aNameInstr));
}

cIrb_Desc1Intsr &  cIrbCal_Block::NC_DescrIndiv(const std::string & aNameInstr)
{
   return const_cast<cIrb_Desc1Intsr &> (DescrIndiv(aNameInstr));
}

void cIrbCal_Block::AvgPairSigma(eTyInstr aTyTarg1,eTyInstr aTyTarg2)
{
    tIntPair aPairTarg((int)aTyTarg1,(int)aTyTarg2); // Index use for T1,T2 <=> T2,T1
    std::vector<cIrb_SigmaInstr*> aVSig; // memorize
    cIrb_SigmaInstr  aSigGlob;

    for ( auto & [aPair,aSigma] : mSigmaPair)
    {
        eTyInstr aTy1 = DescrIndiv(aPair.V1()).Type();
        eTyInstr aTy2 = DescrIndiv(aPair.V2()).Type();

        if (aPairTarg== tIntPair(int(aTy1),int(aTy2)))
        {
            aSigGlob.AddNewSigma(aSigma);
            aVSig.push_back(&aSigma);
        }
    }

    for (auto aPtrS : aVSig)
        *aPtrS = aSigGlob;
}

void cIrbCal_Block::AvgIndivSigma(eTyInstr aTyTarg)
{
    cIrb_SigmaInstr  aSigGlob;
    std::vector<cIrb_Desc1Intsr*> aVInstr;

    for ( auto & [aPair,aInstr] : mDescrIndiv)
    {
        if (aInstr.Type() == aTyTarg)
        {
            aSigGlob.AddNewSigma(aInstr.Sigma());
            aVInstr.push_back(&aInstr);
        }
    }

    for (auto aPtrI : aVInstr)
        aPtrI->SetSigma(aSigGlob);

}

void cIrbCal_Block::AvgIndivSigma()
{
    AvgIndivSigma(eTyInstr::eCamera);
}

void cIrbCal_Block::AvgPairSigma()
{
   AvgPairSigma(eTyInstr::eCamera,eTyInstr::eCamera);
}

void cIrbCal_Block::AvgSigma()
{
    AvgPairSigma();
    AvgIndivSigma();
}





};

