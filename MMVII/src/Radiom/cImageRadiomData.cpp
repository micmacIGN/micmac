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

cImageRadiomData * cImageRadiomData::FromFile(const std::string & aNameFile)
{
    cImageRadiomData * aRes = new cImageRadiomData("yy",1,false);
    ReadFromFile(*aRes,aNameFile);

    return aRes;
}


void cImageRadiomData::AddIndex(tIndex anIndex)
{
    if (mIndexWellOrdered && (!mVIndex.empty()) && (anIndex<=mVIndex.back()))
       mIndexWellOrdered = false;
     mVIndex.push_back(anIndex);
}

void cImageRadiomData::CheckAndAdd(tIndex anIndex,tRadiom aRadiom,int aNbCh,bool WithPoint)
{
    MMVII_INTERNAL_ASSERT_tiny(mNbChanel==aNbCh,"Bad Nb Channel for cImageRadiomData (expected 1)");
    MMVII_INTERNAL_ASSERT_tiny(mWithPoints==WithPoint,"Expcted poinst in cImageRadiomData::AddObsGray");

    AddIndex(anIndex);
    mVVRadiom.at(0).push_back(aRadiom);
}


void cImageRadiomData::AddObsGray(tIndex anIndex,tRadiom aRadiom)
{
    CheckAndAdd(anIndex,aRadiom,1,false);
}

void cImageRadiomData::AddObsGray(tIndex anIndex,tRadiom aRadiom,const tPtMem &  aPt)
{
    CheckAndAdd(anIndex,aRadiom,1,true);
    mVPts.push_back(aPt);
}

void cImageRadiomData::AddObsRGB(tIndex anIndex,tRadiom aR0,tRadiom aR1,tRadiom aR2)
{
    CheckAndAdd(anIndex,aR0,3,false);
    mVVRadiom.at(1).push_back(aR1);
    mVVRadiom.at(2).push_back(aR2);
}

void cImageRadiomData::AddObsRGB(tIndex anIndex,tRadiom aR0,tRadiom aR1,tRadiom aR2,const tPtMem &  aPt)
{
    CheckAndAdd(anIndex,aR0,3,true);
    mVVRadiom.at(1).push_back(aR1);
    mVVRadiom.at(2).push_back(aR2);
    mVPts.push_back(aPt);
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
void AddData(const  cAuxAr2007 & anAux,cImageRadiomData & aIRD)
{
    aIRD.AddData(anAux);
}

void cImageRadiomData::Bench(cParamExeBench & aParam)
{
     if (! aParam.NewBench("Radiom")) return;

      for (int aTime=0 ; aTime<3 ; aTime++)
      {
          for (int aKTest=0 ;aKTest<10 ; aKTest++)
          {
              int aNb =  (1+aKTest) * 3;
              std::vector<int> aPermut = RandPerm(aNb);
	      bool RandOrder = (aKTest%2);

	      cImageRadiomData aIRD("TestIRD.tif",1,false);

              for (size_t aKPt=0 ; aKPt<aPermut.size() ; aKPt++)
              {
                  int aInd = aPermut[aKPt];
		  if (! RandOrder) aInd = aKPt;
                  aIRD.AddObsGray(Square(aInd),aInd*10);
              }
	      if (! RandOrder)
                  MMVII_INTERNAL_ASSERT_bench(aIRD.mIndexWellOrdered," cImageRadiomData ordering");

              aIRD.MakeOrdered();

	      for (int aKPt=0; aKPt<aNb ; aKPt++)
	      {
                  MMVII_INTERNAL_ASSERT_bench(int(aIRD.mVIndex[aKPt])==Square(aKPt)," cImageRadiomData -> Index");
                  MMVII_INTERNAL_ASSERT_bench(aIRD.mVVRadiom[0][aKPt]==(aKPt*10)," cImageRadiomData -> Index");
	      }
          }
      }


      for (int aTime=0 ; aTime<3 ; aTime++)
      {
          for (const auto & aNBC : {1,3})
	  {
              for (const auto & WithPt : {true,false})
	      {
                  cImageRadiomData aIRD("TestIRD.tif",aNBC,WithPt);

		  int aNbPts = 5 + RandUnif_N(10);

		  for (int aKPt=0 ; aKPt < aNbPts ; aKPt++)
		  {
                      tRadiom aR1 = aKPt;
                      tRadiom aR2 = aKPt+1;
                      tRadiom aR3 = aKPt+1;

		      size_t  anIndex = 3 * aNbPts  + aKPt * ((aKPt%2) ? 1 : -1) ; // formula to make index not monoton
		      tPtMem aPt(anIndex+3,aKPt*5);
										   //
                      if (aNBC==3)
                      {
                         if (WithPt)
                            aIRD.AddObsRGB(anIndex,aR1,aR2,aR3,aPt);
			 else
                            aIRD.AddObsRGB(anIndex,aR1,aR2,aR3);
                      }
		      else
                      {
                         if (WithPt)
                            aIRD.AddObsGray(anIndex,aR1,aPt);
			 else
                            aIRD.AddObsGray(anIndex,aR1);
                      }
		  }
		  std::string aNameDmp = cMMVII_Appli::CurrentAppli().TmpDirTestMMVII() +"ImageRadiomData.dmp";

		  SaveInFile(aIRD,aNameDmp);

                  cImageRadiomData* aIRD2 = cImageRadiomData::FromFile(aNameDmp);

		  for (int aKOrder=0 ; aKOrder<2 ; aKOrder++)
                  {
                      bool WellO = aIRD.mIndexWellOrdered;
                      MMVII_INTERNAL_ASSERT_bench(aIRD2->mNbChanel==aNBC,"cImageRadiomData Serial : NBC");
                      MMVII_INTERNAL_ASSERT_bench(aIRD2->mWithPoints==WithPt,"cImageRadiomData Serial : NBC");
                      MMVII_INTERNAL_ASSERT_bench(aIRD2->mNameIm==aIRD.mNameIm,"cImageRadiomData Serial : NBC");
                      MMVII_INTERNAL_ASSERT_bench(aIRD2->mIndexWellOrdered==WellO,"cImageRadiomData Serial : NBC");

                      MMVII_INTERNAL_ASSERT_bench(WellO==(aKOrder!=0),"cImageRadiomData Serial : NBC");

                      for (int aKPt=0 ; aKPt < aNbPts ; aKPt++)
                      {
                          size_t aIndex = aIRD.mVIndex.at(aKPt);
                          MMVII_INTERNAL_ASSERT_bench(aIRD2->mVIndex.at(aKPt)==aIndex,"cImageRadiomData Serial : INDEX");
		          if (WithPt)
		          {
                              const tPtMem  & aPt = aIRD.mVPts.at(aKPt);
                              MMVII_INTERNAL_ASSERT_bench(aIRD2->mVPts.at(aKPt)==aPt,"cImageRadiomData Serial : Pt");
                              MMVII_INTERNAL_ASSERT_bench(aPt.x()== int(aIndex+3) ,"cImageRadiomData Serial : Pt");
		          }
			  for (int aKC=0 ; aKC<aNBC ; aKC++)
			  {
                              MMVII_INTERNAL_ASSERT_bench
                              (
                                  aIRD2->mVVRadiom.at(aKC).at(aKPt)==aIRD.mVVRadiom.at(aKC).at(aKPt),
                                  "cImageRadiomData Serial : radiom"
                              );
			  }
                      }
		      aIRD2->MakeOrdered();
		      aIRD.MakeOrdered();
                  }

		  delete aIRD2;
	      }
	  }
      }

      aParam.EndBench();
}



};
