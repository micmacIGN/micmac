#include "MMVII_InstrumentalBlock.h"

/**
  \file cIrb_Cams.cpp

  \brief This file contains the class relative to cameras in rigid blocks
*/

namespace MMVII
{

/* *************************************************************** */
/*                                                                 */
/*                        cIrbCal_Clino1                          */
/*                                                                 */
/* *************************************************************** */

cIrbCal_Clino1::cIrbCal_Clino1(const std::string & aName) :
   mName         (aName),
   mIsInit       (false),
   mTrInBlock    (nullptr),
   mPolCorr      (new cVectorUK({0.0,1.0,0.0,0.0},mName))
{
}

cIrbCal_Clino1::cIrbCal_Clino1() :
   cIrbCal_Clino1 (MMVII_NONE)
{
}


cIrbCal_Clino1::~cIrbCal_Clino1()
{
    //for (int aK=0 ; aK<10 ; aK++)
    //   StdOut() <<   "cIrbCal_Clino1::~cIrbCal_Clino1cIrbCal_Clino1::~cIrbCal_Clino1()\n ";
    // delete mTrInBlock;
}

const std::string & cIrbCal_Clino1::Name() const {return mName;}
cVectorUK &      cIrbCal_Clino1::PolCorr() {return *mPolCorr;}
cP3dNormWithUK&  cIrbCal_Clino1::CurPNorm()
{
    MMVII_INTERNAL_ASSERT_tiny(mTrInBlock!=0,"cIrbCal_Clino1::CurPNorn");
    return *mTrInBlock;
}


void cIrbCal_Clino1::AddData(const  cAuxAr2007 & anAux)
{
      MMVII::AddData(cAuxAr2007("Name",anAux),mName);
      MMVII::AddData(cAuxAr2007("IsInit",anAux),mIsInit);
      MMVII::StdContAddData(cAuxAr2007("PolCorr",anAux),mPolCorr->Vect());

      cPt3dr aPN  = mTrInBlock ? mTrInBlock->GetPNorm() : cPt3dr(0,0,0);

      MMVII::AddData(cAuxAr2007("PtNorm",anAux),aPN);
      // In case input, we dont want to create a point if was not saved
      if (anAux.Input())
      {
           if (!IsNull(aPN))
               SetPNorm(aPN);
      }
      else
      {
          if (mTrInBlock)
          {
              MMVII_INTERNAL_ASSERT_tiny(IsNull(mTrInBlock->DuDv()),"DuDv not null");
          }
      }

}

void AddData(const  cAuxAr2007 & anAux,cIrbCal_Clino1 & aClino)
{
    aClino.AddData(anAux);
}

void cIrbCal_Clino1::SetPNorm(const cPt3dr &aPNorm)
{
  if (mTrInBlock==nullptr)
  {
      mTrInBlock.reset(new cP3dNormWithUK(aPNorm,"BlockClino",mName));
  }
  else
      mTrInBlock->SetPNorm(aPNorm);
   mIsInit      = true;
}






/* *************************************************************** */
/*                                                                 */
/*                        cIrbCal_ClinoSet                            */
/*                                                                 */
/* *************************************************************** */

cIrbCal_ClinoSet::cIrbCal_ClinoSet() :
    mCalBlock (nullptr)
{
}

std::vector<std::string> cIrbCal_ClinoSet::VNames() const
{
    std::vector<std::string> aRes;

    for (auto&  aClino : mVClinos)
        aRes.push_back(aClino.Name());

    return aRes;
}

void cIrbCal_ClinoSet::AddData(const  cAuxAr2007 & anAux)
{
     MMVII::StdContAddData(cAuxAr2007("Set_Clinos",anAux),mVClinos);
}

void AddData(const  cAuxAr2007 & anAux,cIrbCal_ClinoSet & aSetClino)
{
    aSetClino.AddData(anAux);
}

int  cIrbCal_ClinoSet::IndexClinoFromName(const std::string& aName,bool OkNone) const
{
    for (size_t aK=0 ; aK<mVClinos.size() ; aK++)
        if (mVClinos.at(aK).Name() == aName)
           return aK;
    MMVII_INTERNAL_ASSERT_strong(OkNone,"cIrbCal_ClinoSet::IndexClinoFromName Not found");
    return -1;
}

//cIrbCal_Clino1 &  ClinoOfName(const std::string& aName);


cIrbCal_Clino1 * cIrbCal_ClinoSet::ClinoFromName(const std::string& aName,bool OkNone)
{
    int aK = IndexClinoFromName(aName,OkNone);

    return (aK>=0) ? &mVClinos.at(aK) : nullptr;
}

void cIrbCal_ClinoSet::AddClino(const std::string & aName,tREAL8 aSigma,bool SVP)
{
   StdOut() << "  AddClino, Sigma=" << Rad2DMgon(aSigma) << "DMgon\n";
   cIrbCal_Clino1 * aClino = ClinoFromName(aName,true);
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
   auto & aDescInstr = mCalBlock->AddSigma_Indiv(aName,eTyInstr::eClino);
   if (aSigma>0)
       aDescInstr.SetSigma(cIrb_SigmaInstr(0.0,1.0,0.0,aSigma));
}

size_t cIrbCal_ClinoSet::NbClino() const {return mVClinos.size();}

cIrbCal_Clino1 &  cIrbCal_ClinoSet::KthClino(int aK)
{
    return mVClinos.at(aK);
}



/* *************************************************************** */
/*                                                                 */
/*                        cIrbComp_Clino1                          */
/*                                                                 */
/* *************************************************************** */

cIrbComp_Clino1::cIrbComp_Clino1(tREAL8 anAngle) :
    mAngle (anAngle)
{

}

tREAL8 cIrbComp_Clino1::Angle() const {return mAngle;}


/* *************************************************************** */
/*                                                                 */
/*                        cIrbComp_ClinoSet                        */
/*                                                                 */
/* *************************************************************** */

cIrbComp_ClinoSet::cIrbComp_ClinoSet()
{

}

void cIrbComp_ClinoSet::SetClinoValues(const cOneMesureClino& aMeas)
{
    for (const auto & anA : aMeas.Angles())
        mVCompClinos.push_back(cIrbComp_Clino1(anA));
}

size_t cIrbComp_ClinoSet::NbMeasure() const
{
    return mVCompClinos.size();
}


const cIrbComp_Clino1 & cIrbComp_ClinoSet::KthMeasure(int aK) const
{
    return mVCompClinos.at(aK);
}

};

