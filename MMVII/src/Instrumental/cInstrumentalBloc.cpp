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
       anAux.Ar().AddComment("SigRot="+ToStr(mAvgSigRot.Average()));

  //  StdOut() << "cIrb_SigmaInstr::AddData::Done \n";
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
    mCompBlock (aCompBlock),
    mSetCams   (aCompBlock)
{
}

const cIrbComp_CamSet & cIrbComp_TimeS::SetCams() const {return mSetCams;}
cIrbComp_CamSet & cIrbComp_TimeS::SetCams() {return mSetCams;}

const cIrbComp_Block & cIrbComp_TimeS::CompBlock() const {return mCompBlock;}
// cIrbComp_Block & cIrbComp_TimeS::CompBlock()  {return mCompBlock;}


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

    //  -------------------------- "computation"  --------------------------------------------

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
    mSetCams.mCalBlock = this;
}


void  cIrbCal_Block::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("Cams",anAux),mSetCams);	
    MMVII::AddData(cAuxAr2007("Clinos",anAux),mSetClinos);	

    MMVII::StdMapAddData(cAuxAr2007("SigmasPairs",anAux),mSigmaPair);
    MMVII::StdMapAddData(cAuxAr2007("SigmasIndiv",anAux),mSigmaInd);

}
void AddData(const  cAuxAr2007 & anAux,cIrbCal_Block & aRBoI)
{
    aRBoI.AddData(anAux);
}

cIrb_Desc1Intsr &  cIrbCal_Block::AddSigma_Indiv(std::string aNameInstr,eTyInstr aTypeInstr)
{
    auto  anIter = mSigmaInd.find(aNameInstr);
    if (anIter== mSigmaInd.end())
    {
        mSigmaInd[aNameInstr] = cIrb_Desc1Intsr(aTypeInstr,aNameInstr);
        anIter = mSigmaInd.find(aNameInstr);
    }
    cIrb_Desc1Intsr &  anInstr = anIter->second;

    MMVII_INTERNAL_ASSERT_tiny(anInstr.Type()== aTypeInstr,"Chang type in cIrbCal_Block::AddSigma_Indiv");
    MMVII_INTERNAL_ASSERT_tiny(anInstr.NameInstr()== aNameInstr,"Chang type in cIrbCal_Block::AddSigma_Indiv");

    return anInstr;
}

void  cIrbCal_Block::AddSigma_Indiv(std::string aNameInstr,eTyInstr aTypeInstr, const cIrb_SigmaInstr & aSigma)
{
    /*
    auto  anIter = mSigmaInd.find(aNameInstr);
    if (anIter== mSigmaInd.end())
    {
        mSigmaInd[aNameInstr] = cIrb_Desc1Intsr(aTypeInstr,aNameInstr);
        anIter = mSigmaInd.find(aNameInstr);
    }
    MMVII_INTERNAL_ASSERT_tiny(anIter->second.Type()== aTypeInstr,"Chang type in cIrbCal_Block::AddSigma_Indiv");
    MMVII_INTERNAL_ASSERT_tiny(anIter->second.NameInstr()== aNameInstr,"Chang type in cIrbCal_Block::AddSigma_Indiv");
*/

    AddSigma_Indiv(aNameInstr,aTypeInstr).AddNewSigma(aSigma);
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
void cIrbCal_Block::SetSigmaPair(const  std::map<tNamePair,cIrb_SigmaInstr> & aSigmaPair)
{

    for (const auto&  [aCple,aSig] : aSigmaPair )
    {
        StdOut() << " SP=" << aCple.V1() << " " << aCple.V2() << " " << aSig.SigmaTr() << " " << aSig.SigmaRot() << "\n";
    }
    mSigmaPair = aSigmaPair;
}

const cIrb_Desc1Intsr &  cIrbCal_Block::SigmaInd(const std::string & aNameInstr) const
{
   return *(MapGet(mSigmaInd,aNameInstr));
}


void cIrbCal_Block::AvgPairSigma(eTyInstr aTyTarg1,eTyInstr aTyTarg2)
{
    tIntPair aPairTarg((int)aTyTarg1,(int)aTyTarg2); // Index use for T1,T2 <=> T2,T1
    std::vector<cIrb_SigmaInstr*> aVSig; // memorize
    cIrb_SigmaInstr  aSigGlob;

    for ( auto & [aPair,aSigma] : mSigmaPair)
    {
        eTyInstr aTy1 = SigmaInd(aPair.V1()).Type();
        eTyInstr aTy2 = SigmaInd(aPair.V2()).Type();

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

    for ( auto & [aPair,aInstr] : mSigmaInd)
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


/* *************************************************************** */
/*                                                                 */
/*                        cAppli_EditBlockInstr                    */
/*                                                                 */
/* *************************************************************** */

/*
class cAppli_EditBlockInstr : public cMMVII_Appli
{
     public :

        cAppli_EditBlockInstr(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

        std::vector<std::string>  Samples() const override;

     private :
        cPhotogrammetricProject   mPhProj;       //< As usual ....
        std::string               mNameBloc;     //< Name of the block edited (generally default MMVII)
        std::vector<std::string>  mVPatsIm4Cam;  //< Patterns for cam structure : [PatSelOnDisk,PatTimeStamp?,PatSelInBlock?]
        bool                      mFromScratch;  //< If exist file : Reset of Modify ?
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

    cIrbCal_Block *  aBlock =    mFromScratch                           ?
                                 new cIrbCal_Block                      :
                                 mPhProj.ReadRigBoI(mNameBloc,SVP::Yes) ;

    // if we add structure for camera
    if (IsInit(&mVPatsIm4Cam))
    {
        std::string aPatSelOnDisk = mVPatsIm4Cam.at(0);
        std::string aPatTimeStamp = GetDef(mVPatsIm4Cam,1,aPatSelOnDisk);
        std::string aPatSelIm = GetDef(mVPatsIm4Cam,2,aPatTimeStamp);

        auto aVNameIm = ToVect(SetNameFromString(aPatSelOnDisk,true));
        std::set<std::string> aSetNameCal;
        for (const auto & aNameIm : aVNameIm)
        {
            std::string aNameCal = mPhProj.StdNameCalibOfImage(aNameIm);
            if (! BoolFind(aSetNameCal,aNameCal))
            {
                aSetNameCal.insert(aNameCal);
                aBlock->SetCams().AddCam(aNameCal,aPatTimeStamp,aPatSelIm,SVP::Yes);
            }
        }
    }

    // if we add the structure for clinometers
    if (mPhProj.DPMeasuresClino().DirInIsInit())
    {
         cSetMeasureClino aMesClin =  mPhProj.ReadMeasureClino();
         for (const auto & aName : aMesClin.NamesClino())
         {
             aBlock->SetClinos().AddClino(aName,SVP::Yes);
         }
    }

    // save the result on disk
    mPhProj.SaveRigBoI(*aBlock);

    delete aBlock;
    return EXIT_SUCCESS;
}
*/

    /* ==================================================== */
    /*                                                      */
    /*               MMVII                                  */
    /*                                                      */
    /* ==================================================== */

/*

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
*/


};

