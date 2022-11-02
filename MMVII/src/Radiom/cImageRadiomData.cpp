#include "MMVII_Ptxd.h"
#include "MMVII_Radiom.h"
#include "MMVII_Stringifier.h"
#include "MMVII_2Include_Serial_Tpl.h"


namespace MMVII
{

   /* =============================================== */
   /*                                                 */
   /*              cImageRadiomData                   */
   /*                                                 */
   /* =============================================== */

cImageRadiomData::cImageRadiomData(const std::string & aNameIm,int aNbChanel,bool withPoint) :
    mIndexWellOrdered  (true),
    mNameIm            (aNameIm),
    mNbChanel          (aNbChanel),
    mWithPoints        (withPoint),
    mVRadiom           (aNbChanel)
{
}

void cImageRadiomData::AddIndex(tIndex anIndex)
{
    if (mIndexWellOrdered && (!mVIndex.empty()) && (anIndex<=mVIndex.back()))
       mIndexWellOrdered = false;
     mVIndex.push_back(anIndex);
}


void cImageRadiomData::AddObsGray(tIndex anIndex,tRadiom aRadiom)
{
    MMVII_INTERNAL_ASSERT_tiny(mNbChanel==1,"Bad Nb Channel for cImageRadiomData (expected 1)");

    AddIndex(anIndex);
    mVRadiom.at(0).push_back(aRadiom);
}

void cImageRadiomData::MakeOrdered(cSetIntDyn & aSID)
{
   if (mIndexWellOrdered) return;

   // Make index & invert
   aSID.Clear();
   for (const auto & anInd : mVIndex)
       aSID.AddInd(anInd);
   aSID.MakeInvertIndex();


   mIndexWellOrdered = true;
}

void cImageRadiomData::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("WellOrdered",anAux),mIndexWellOrdered);
    MMVII::AddData(cAuxAr2007("NameIm",anAux),mNameIm);
    MMVII::AddData(cAuxAr2007("NbChanels",anAux),mNbChanel);
    MMVII::AddData(cAuxAr2007("WithPts",anAux),mWithPoints);
    MMVII::AddData(cAuxAr2007("Index",anAux),mVIndex);
    MMVII::AddData(cAuxAr2007("Pts",anAux),mVPts);
    MMVII::AddData(cAuxAr2007("Radioms",anAux),mVRadiom);
}

/*
*/


};
