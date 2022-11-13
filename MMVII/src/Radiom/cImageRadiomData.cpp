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
    mLimitIndex        (0),
    mVVRadiom          (aNbChanel)
{
}

cImageRadiomData * cImageRadiomData::FromFile(const std::string & aNameFile)
{
    cImageRadiomData * aRes = new cImageRadiomData("yy",1,false);
    ReadFromFile(*aRes,aNameFile);

    return aRes;
}
void cImageRadiomData::ToFile(const std::string & aNameFile) const
{
     SaveInFile(*this,aNameFile);
}

std::string cImageRadiomData::NameFileOfImage(const std::string& aNameIm)
{
      return "RadiomData-" + aNameIm + ".dmp";
}
std::string cImageRadiomData::NameFile() const
{
      return NameFileOfImage(mNameIm);
}

const std::vector<size_t> &    cImageRadiomData::VIndex() const {return mVIndex;}
const cImageRadiomData::tVPt & cImageRadiomData::VPts() const {return mVPts;}


const cImageRadiomData::tPtMem & cImageRadiomData::Pt(size_t aIndex) const {return mVPts.at(aIndex);}

cImageRadiomData::tRadiom cImageRadiomData::Gray(size_t aIndex) const
{
   if (mNbChanel==1) return mVVRadiom[0].at(aIndex);
   int aSom=0;
   for (int aKC=0 ; aKC<mNbChanel ; aKC++)
       aSom += mVVRadiom[aKC].at(aIndex);

   return aSom/mNbChanel;
}


const cImageRadiomData::tVRadiom & cImageRadiomData::VRadiom(size_t aKCh) const {return mVVRadiom.at(aKCh);}
size_t cImageRadiomData::LimitIndex() const{return mLimitIndex;}



void cImageRadiomData::AddIndex(tIndex anIndex)
{
    if (mIndexWellOrdered && (!mVIndex.empty()) && (anIndex<=mVIndex.back()))
       mIndexWellOrdered = false;
     mVIndex.push_back(anIndex);
     mLimitIndex = std::max(mLimitIndex,anIndex+1);
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

void cImageRadiomData::AddObs_Adapt(tIndex anIndex,tRadiom aR0,tRadiom aR1,tRadiom aR2,const tPtMem &  aPt)
{
    if (mNbChanel==1)
    {
         tRadiom aGray = (aR0+aR1+aR2)/3;
	 if (mWithPoints)
            AddObsGray(anIndex,aGray,aPt);
	 else
            AddObsGray(anIndex,aGray);
    }
    else if (mNbChanel==3)
    {
	 if (mWithPoints)
            AddObsRGB(anIndex,aR0,aR1,aR2,aPt);
	 else
            AddObsRGB(anIndex,aR0,aR1,aR2);
    }

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

void cImageRadiomData::GetIndexCommon(std::vector<tPt2di> & aRes,cImageRadiomData & aIRD2)
{
    aRes.clear();
    MakeOrdered();
    aIRD2.MakeOrdered();

     // inspired from https://en.cppreference.com/w/cpp/algorithm/set_intersection
     // BTW cannot used set_intersection as we dont want the index but the "index of the index" ...

    size_t aI1=0;
    size_t aI2=0;
    size_t aSz1=mVIndex.size();
    size_t aSz2=aIRD2.mVIndex.size();

    while ( (aI1 != aSz1) && (aI2 != aSz2)) 
    {
        if (mVIndex[aI1] < aIRD2.mVIndex[aI2]) 
        {
            ++aI1;
        } 
	else  
        {
            if (mVIndex[aI1] == aIRD2.mVIndex[aI2])
	    {
                aRes.push_back(cPt2di(aI1++,aI2));
            }
            ++aI2;
        }
    }
}


void cImageRadiomData::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("WellOrdered",anAux),mIndexWellOrdered);
    MMVII::AddData(cAuxAr2007("NameIm",anAux),mNameIm);
    MMVII::AddData(cAuxAr2007("NbChanels",anAux),mNbChanel);
    MMVII::AddData(cAuxAr2007("WithPts",anAux),mWithPoints);
    MMVII::AddData(cAuxAr2007("Index",anAux),mVIndex);
    MMVII::AddData(cAuxAr2007("LimitIndex",anAux),mLimitIndex);
    MMVII::AddData(cAuxAr2007("Pts",anAux),mVPts);
    MMVII::AddData(cAuxAr2007("Radioms",anAux),mVVRadiom);
}
void AddData(const  cAuxAr2007 & anAux,cImageRadiomData & aIRD)
{
    aIRD.AddData(anAux);
}

static void  InitRandIRD (cSetIntDyn & aSet,cImageRadiomData & aIRD,int aNb,double aProba)
{
     for (int aK=0 ; aK<aNb ; aK++)
     {
          bool inSide = (RandUnif_0_1() > aProba);
          if (inSide)
          {
             aIRD.AddObsGray(aK,aK*3);
	     aSet.AddInd(aK);
          }
     }
}

void cImageRadiomData::Bench(cParamExeBench & aParam)
{
     if (! aParam.NewBench("Radiom")) return;

      //   Check "MakeOrdered"  and "mIndexWellOrdered"
      //  in this test we generate a random (or not) permutation , the index are N^2, the value N*10
      //  we check that  after ordering
      //       *  Ind[K] = K^2
      //       *  Val[K] = K * 10
      //  if permutaion was not randomizedwe check also that mIndexWellOrdered is true

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

      //  Check  read/write  SaveInFile  and  "ToFile" and "FromFile"
      //  we generate cImageRadiomData, with and w/o Pt, with and w/o RGB,
      //  and check that what we write is what we read

      for (int aTime=0 ; aTime<3 ; aTime++)  // do several time the test
      {
          for (const auto & aNBC : {1,3})  // check Gray and RGB
	  {
              for (const auto & WithPt : {true,false})  // check with and w/o points
	      {
                  // generate a IRD with predictible value
                  cImageRadiomData aIRD("TestIRD.tif",aNBC,WithPt);
		  int aNbPts = 5 + RandUnif_N(10);
		  for (int aKPt=0 ; aKPt < aNbPts ; aKPt++)
		  {
                      tRadiom aR1 = aKPt;
                      tRadiom aR2 = aKPt+1;
                      tRadiom aR3 = aKPt+1;

		      size_t  anIndex = 3 * aNbPts  + aKPt * ((aKPt%2) ? 1 : -1) ; // formula to make index not monoton
		      tPtMem aPt(anIndex+3,aKPt*5);
                      aIRD.AddObs_Adapt(anIndex,aR1,aR2,aR3,aPt);
		  }
		  std::string aNameDmp = cMMVII_Appli::CurrentAppli().TmpDirTestMMVII() +"ImageRadiomData.dmp";
		  aIRD.ToFile(aNameDmp);

		  // read it in a new IRD
                  cImageRadiomData* aIRD2 = cImageRadiomData::FromFile(aNameDmp);

		  for (int aKOrder=0 ; aKOrder<2 ; aKOrder++)
                  {
                      // check global values are the same
                      bool WellO = aIRD.mIndexWellOrdered;
                      MMVII_INTERNAL_ASSERT_bench(aIRD2->mNbChanel==aNBC,"cImageRadiomData Serial : NBC");
                      MMVII_INTERNAL_ASSERT_bench(aIRD2->mWithPoints==WithPt,"cImageRadiomData Serial : NBC");
                      MMVII_INTERNAL_ASSERT_bench(aIRD2->mNameIm==aIRD.mNameIm,"cImageRadiomData Serial : NBC");
                      MMVII_INTERNAL_ASSERT_bench(aIRD2->mIndexWellOrdered==WellO,"cImageRadiomData Serial : NBC");

                      MMVII_INTERNAL_ASSERT_bench(WellO==(aKOrder!=0),"cImageRadiomData Serial : NBC");

		      // Check Radiom and Pts are the same
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

      for (int aTime=0 ; aTime<10 ; aTime++)  // do several time the test
      {
           int aNb1 = 1 + RandUnif_N(20);
           int aNb2 = 1 + RandUnif_N(20);
	   if ((aTime==0) || (aTime==1))  aNb1=0;
	   if ((aTime==1) || (aTime==2))  aNb2=0;

           for (int aState=0 ; aState<6 ; aState++)  // do several time the test
           {
                double aProba1 = 0.5;
                double aProba2 = 0.5;

	        if ((aState==0) || (aState==2) ) aProba1 =0;
	        if ((aState==1) || (aState==2) ) aProba2 =0;
	        if (aState==3)                   aProba1 =1.0;

                cSetIntDyn aSet1(aNb1);
                cSetIntDyn aSet2(aNb2);
                cImageRadiomData aIRD1("TestIRD.tif",1,false);
                cImageRadiomData aIRD2("TestIRD.tif",1,false);

		InitRandIRD(aSet1,aIRD1,aNb1,aProba1);
		InitRandIRD(aSet2,aIRD2,aNb2,aProba2);

		std::vector<cPt2di> aVPairI;
		aIRD1.GetIndexCommon(aVPairI,aIRD2);

                // check all ind x,y in aVPairI belong simultaneousy to IRD1 & IRD2, then  erase them
                for (const auto & aPt : aVPairI)
                {
                     size_t aI1 = aIRD1.VIndex()[aPt.x()];
                     size_t aI2 = aIRD2.VIndex()[aPt.y()];
                     MMVII_INTERNAL_ASSERT_bench(aSet1.mOccupied.at(aI1),"GetIndexCommon set1");
                     MMVII_INTERNAL_ASSERT_bench(aSet2.mOccupied.at(aI2),"GetIndexCommon set2");
		     aSet1.mOccupied.at(aI1) = false;
		     aSet2.mOccupied.at(aI2) = false;
                }

		// check that after erase there is no more common indexe
		for (int aK=0 ; aK< std::min(aNb1,aNb2) ; aK++)
                     MMVII_INTERNAL_ASSERT_bench(!(aSet1.mOccupied.at(aK) && aSet2.mOccupied.at(aK)),"GetIndexCommon inter");

           }
      }

      aParam.EndBench();
}

   /* =============================================== */
   /*                                                 */
   /*              cFusionIRDSEt                      */
   /*                                                 */
   /* =============================================== */

cFusionIRDSEt::cFusionIRDSEt(size_t aNbMax) :
        mVVIndexes (aNbMax)
{
}
void cFusionIRDSEt::Resize(size_t aNbMax)
{
    mVVIndexes.resize(aNbMax);
}

void cFusionIRDSEt::AddIndex(int aNumIm, const std::vector<tIndex> & aVInd)
{
     for (size_t aK=0 ; aK<aVInd.size() ; aK++)
     {
         mVVIndexes.at(aVInd[aK]).push_back(tImInd(aNumIm,aK));
     }
}

void cFusionIRDSEt::FilterSzMin(size_t aSzMin)
{
     std::vector<tV1Index > aDup;
     for (const auto & aVInd : mVVIndexes)
     {
         if (aVInd.size() >= aSzMin)
            aDup.push_back(aVInd);
     }
     mVVIndexes = aDup;
}

const std::vector<cFusionIRDSEt::tV1Index > & cFusionIRDSEt::VVIndexes() const {return mVVIndexes;}





};
