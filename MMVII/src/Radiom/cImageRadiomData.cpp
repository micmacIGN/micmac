#include "MMVII_Ptxd.h"
#include "MMVII_Radiom.h"
#include "MMVII_Stringifier.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Bench.h"


namespace MMVII
{

/*
     std::vector<bool>    mOccupied;     ///< direct acces to the belonging  [0 1 0 0 1 0 1 0]
     std::vector<size_t>  mVIndOcc;      ///< list  of element               [1 4 6]
     std::vector<int   >  mVInvertInd;   ///< if created, give for an index its rank [ _ 0 _ _  1 _ 2 _]
*/


template <class Type> void Order(std::vector<Type> & aVec,const std::vector<size_t> aVInd0,const cSetIntDyn & aSet,std::vector<Type> & aBuf)
{
    aBuf.resize(aVec.size());

    for (size_t aK=0 ; aK<aVec.size() ; aK++)
        aBuf[ aSet.mVInvertInd[aVInd0[aK]] ] =  aVec[aK];

    aVec = aBuf;
}


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
    mVVRadiom           (aNbChanel)
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
    mVVRadiom.at(0).push_back(aRadiom);
}

void cImageRadiomData::MakeOrdered()
{
   if (mIndexWellOrdered) 
      return;

   static cSetIntDyn aSID(1);

   // Make index & invert
   aSID.Clear();
   for (const auto & anInd : mVIndex)
       aSID.AddInd(anInd);
   aSID.MakeInvertIndex();
  

   if (mWithPoints)
   {
       static std::vector<tPtMem>  aBufPts;
       Order(mVPts,mVIndex,aSID,aBufPts);
   }
   for (auto & aVecRadiom : mVVRadiom)
   {
        static tVRadiom  aBufRadiom;
	Order(aVecRadiom,mVIndex,aSID,aBufRadiom);
   }
   mVIndex = aSID.mVIndOcc;

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
    MMVII::AddData(cAuxAr2007("Radioms",anAux),mVVRadiom);
}
/*
*/

void cImageRadiomData::Bench(cParamExeBench & aParam)
{
     if (! aParam.NewBench("Radiom")) return;

      for (int aTime=0 ; aTime<3 ; aTime++)
      {
          for (int aK=0 ;aK<10 ; aK++)
          {
              int aNb =  (1+aK) * 3;
              std::vector<int> aPermut = RandPerm(aNb);

	      cImageRadiomData aIRD("TestIRD.tif",1,false);

	      for (size_t aK=0 ; aK<aPermut.size() ; aK++)
                  aIRD.AddObsGray(Square(aPermut[aK]),aPermut[aK]*10);

              aIRD.MakeOrdered();

	      for (int aK=0; aK<aNb ; aK++)
	      {
                  MMVII_INTERNAL_ASSERT_bench(int(aIRD.mVIndex[aK])==Square(aK)," cImageRadiomData -> Index");
                  MMVII_INTERNAL_ASSERT_bench(aIRD.mVVRadiom[0][aK]==(aK*10)," cImageRadiomData -> Index");
	      }
          }
      }

      aParam.EndBench();
}



};
