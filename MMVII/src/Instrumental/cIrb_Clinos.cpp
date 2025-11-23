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
   mSigmaR       (-1)
{
}

cIrbCal_Clino1::cIrbCal_Clino1() :
   cIrbCal_Clino1 (MMVII_NONE)
{
}

const std::string & cIrbCal_Clino1::Name() const {return mName;}


void cIrbCal_Clino1::AddData(const  cAuxAr2007 & anAux)
{
      MMVII::AddData(cAuxAr2007("Name",anAux),mName);
      MMVII::AddData(cAuxAr2007("IsInit",anAux),mIsInit);
      MMVII::AddData(cAuxAr2007("SigmaR",anAux),mSigmaR);

      cPt3dr aPN  = mTrInBlock ? mTrInBlock->GetPNorm() : cPt3dr(0,0,0);

      MMVII::AddData(cAuxAr2007("PtNorm",anAux),aPN);
      if (anAux.Input() && mIsInit)
      {
          mTrInBlock->SetPNorm(aPN);
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
      mTrInBlock = new cP3dNormWithUK(aPNorm,"BlockClino","Name");
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

int  cIrbCal_ClinoSet::IndexClinoFromName(const std::string& aName) const
{
    for (size_t aK=0 ; aK<mVClinos.size() ; aK++)
        if (mVClinos.at(aK).Name() == aName)
           return aK;
    return -1;
}

cIrbCal_Clino1 * cIrbCal_ClinoSet::ClinoFromName(const std::string& aName)
{
    int aK = IndexClinoFromName(aName);

    return (aK>=0) ? &mVClinos.at(aK) : nullptr;
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
      mCalBlock->AddSigma_Indiv(aName,eTyInstr::eClino);
   }
}
size_t cIrbCal_ClinoSet::NbClino() const {return mVClinos.size();}




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

