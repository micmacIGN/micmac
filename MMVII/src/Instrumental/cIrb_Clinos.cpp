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
      mCalBlock->AddSigma_Indiv(aName,eTyInstr::eClino);
   }
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


};

