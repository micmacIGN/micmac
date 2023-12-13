#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_TplHeap.h"

#include "TieP.h"


/**
   \file cConvCalib.cpp  testgit

   \brief file for conversion between calibration (change format, change model) and tests
*/


namespace MMVII
{


/* **************************************** */
/*                                          */
/*             c1ConfigLogMTP               */
/*                                          */
/* **************************************** */

c1ConfigLogMTP::c1ConfigLogMTP() :
    mNbPts (0)
{
}

void c1ConfigLogMTP::SetIndIm(const  std::vector<int> & aIndIm) { mIndIm = aIndIm; }
void c1ConfigLogMTP::SetNbPts(size_t aNbPts) { mNbPts = aNbPts; }
void c1ConfigLogMTP::SetIdP0(size_t anIdP0)  { mIdP0  = anIdP0; }

void  c1ConfigLogMTP::AddData(const cAuxAr2007 & anAux)
{
     size_t aNbIm = mIndIm.size();
     MMVII::AddData(cAuxAr2007("NbIm", anAux),aNbIm);
     if (anAux.Ar().Input())
     {
        mIndIm.resize(aNbIm);
     }
     AddTabData(cAuxAr2007("IndIm", anAux),mIndIm.data(),mIndIm.size());
     MMVII::AddData(cAuxAr2007("IdP0", anAux),mIdP0);
     MMVII::AddData(cAuxAr2007("NbPts", anAux),mNbPts);
}
void AddData(const cAuxAr2007 & anAux,c1ConfigLogMTP & aConf){aConf.AddData(anAux);}

void c1ConfigLogMTP::NewIndIm(std::vector<int> & aVecNewInd,const std::vector<int> & aLut) const
{
    aVecNewInd.clear();

    for (const auto & mOldIndIm : mIndIm)
    {
        int aNewIndIm = aLut.at(mOldIndIm);
	if (aNewIndIm >=0)
           aVecNewInd.push_back(aNewIndIm);
    }
}

size_t c1ConfigLogMTP::NbPts() const {return mNbPts;}

/* **************************************** */
/*                                          */
/*            cGlobConfLogMTP               */
/*                                          */
/* **************************************** */

cGlobConfLogMTP::cGlobConfLogMTP() {}

cGlobConfLogMTP::cGlobConfLogMTP(std::vector<std::string> & aVNamesIm,size_t aNbConfig) :
	mVNamesIm (aVNamesIm),
	mConfigs  (aNbConfig)
{
}

const std::vector<std::string> &  cGlobConfLogMTP::VNamesIm() const {return mVNamesIm;}

c1ConfigLogMTP & cGlobConfLogMTP::KthConf(size_t aK) { return mConfigs.at(aK);}
const std::vector<c1ConfigLogMTP> & cGlobConfLogMTP::Configs() const {return mConfigs;}

void  cGlobConfLogMTP::AddData(const cAuxAr2007 & anAux)
{
     MMVII::AddData(cAuxAr2007("Images", anAux),mVNamesIm);
     MMVII::AddData(cAuxAr2007("Configs", anAux),mConfigs);
}

void AddData(const cAuxAr2007 & anAux,cGlobConfLogMTP & aGlobConf){aGlobConf.AddData(anAux);}

/* **************************************** */
/*                                          */
/*                 cReadMTP_Large           */
/*                                          */
/* **************************************** */

// Not finish, theoretically adpated to  "big" data with optimization by pre-compiled index

cReadMTP_Large::cReadMTP_Large(const std::vector<std::string> aVNewNames,const std::string aNameConfig)
{
    cGlobConfLogMTP aFullConfig;
    ReadFromFile(aFullConfig,aNameConfig);
    const std::vector<std::string> &  aVOldNames =  aFullConfig.VNamesIm();

    //  [1]  Compute the mapping  Old -> Num
    std::vector<int> aVOld2New(aVOldNames.size(),-1);
    size_t aOldNum =0;
    size_t aNewNum =0;

    for (const auto & aNameOri : aVOldNames)
    {
        bool Got = std::binary_search(aVNewNames.begin(),aVNewNames.end(),aNameOri);
	if (Got)
	{
           aVOld2New[aOldNum] = aNewNum;

	   aNewNum++;
	}
	aOldNum++;
    }

    // [2] compute the new config and their size 
    // 
    //    Map   Config  ->  IdInt
    //    Tab[IdConf]   ->  Pair*
    //    Tab[IdPt]  ->  IdConf
    //
    //     Tab [IdConfig]  ->  Config, Pts
   
    std::map<std::vector<int>,size_t>  aMapCptInd;
    for (const auto & a1Conf : aFullConfig.Configs())
    {
        std::vector<int> aVNewInd;
        a1Conf.NewIndIm(aVNewInd,aVOld2New);
	if (aVNewInd.size() >= 2)
	{
           aMapCptInd[aVNewInd] += a1Conf.NbPts();
	}
    }

    for (const auto & aPair : aMapCptInd)
        StdOut() << "PPPP : " << aPair.first << " = " << aPair.second << std::endl;
}

/* **************************************** */
/*                                          */
/*                 cReadMTP_Std             */
/*                                          */
/* **************************************** */

class cObjHeapMTP
{
     public :
        int mIdPt;
        int mNumCurInIm;
        int mIdIm;
	int mHeapIndex;
};

class  cParamHeap_MTP
{
     public :
        typedef cObjHeapMTP tObj;

        bool operator() (const cObjHeapMTP & aP1,const cObjHeapMTP & aP2) const  
	{
            if (aP1.mIdPt < aP2.mIdPt) return true;
            if (aP1.mIdPt > aP2.mIdPt) return false;
            return (aP1.mIdIm < aP2.mIdIm);
	}

        static void SetIndex(tObj & aObj,tINT4 i) { aObj.mHeapIndex = i;} 
        static int  GetIndex(const tObj & aObj) {return aObj.mHeapIndex;}

};
static cParamHeap_MTP The_cParamHeap_MTP;



class cReadMTP_Std 
{
      public :
              typedef cIndexedHeap<cObjHeapMTP,cParamHeap_MTP,cParamHeap_MTP> tHeap;
	      cReadMTP_Std
              (
                     const std::vector<std::string> &aVNames,
		     cPhotogrammetricProject & aPhProj,
		     bool WithInd,
		     bool WithSensor
              );

	      cComputeMergeMulTieP *      CompMerge();
	      bool WithSensor() const { return mWithSensor;}

      private :
              bool GetNextConfig();
	      bool                        mWithIndex;
	      bool                        mWithSensor;
	      std::vector<cVecTiePMul>    mVTpm;
	      tHeap                       mHeap;
	      cComputeMergeMulTieP      * mCompMerge;
};


bool cReadMTP_Std::GetNextConfig()
{
      std::vector<cObjHeapMTP>       aNewConfig;
      std::vector<int>  aIdConf;

      cObjHeapMTP *  aPt0 = mHeap.Lowest();
      if (aPt0==nullptr) return false;

      int aIdPt0 = aPt0->mIdPt;

      cObjHeapMTP * aPt = nullptr;
      while ( (aPt=mHeap.Lowest())  && (aPt->mIdPt==aIdPt0))
      {
	  aNewConfig.push_back(*aPt);
	  aIdConf.push_back(aPt->mIdIm);
	  aPt->mNumCurInIm++;  // we must progress in parsing point of image
          const std::vector<cTiePMul> & aVec = mVTpm.at(aPt->mIdIm).mVecTPM;
	  // all the point of this image have been reached 
	  if (aPt->mNumCurInIm== (int) aVec.size())
	  {
              mHeap.Pop();
	  }
	  else
	  {
              aPt->mIdPt = aVec.at(aPt->mNumCurInIm).mId; // else we update the new IdGlob
	      mHeap.UpDate(*aPt);  // Heap must be reorganized after this modificatio;
	  }
      }


      if (aNewConfig.size() >= 2)  // no use to have singleton tie point ...
      {
          cVal1ConfTPM & aVal =  mCompMerge->Pts()[aIdConf];
          std::vector<cPt2dr> &  aVPtsOut =  aVal.mVPIm;
	  for (const auto  & aPt :  aNewConfig)
	  {
              const std::vector<cTiePMul> & aVecIn = mVTpm.at(aPt.mIdIm).mVecTPM;
	      aVPtsOut.push_back(aVecIn.at(aPt.mNumCurInIm).mPt);
	  }
          if (mWithIndex)
          {
             aVal.mVIdPts.push_back(aIdPt0);
          }
      }

      return true;
}


cReadMTP_Std::cReadMTP_Std
(
    const std::vector<std::string> &aVNames,
    cPhotogrammetricProject & aPhProj,
    bool  WithIndex,
    bool  WithSensor
) :
     mWithIndex   (WithIndex),
     mWithSensor  (WithSensor),
     mVTpm        (aVNames.size(),cVecTiePMul("")),
     mHeap        (The_cParamHeap_MTP),
     mCompMerge   (new cComputeMergeMulTieP(aVNames,nullptr,(WithSensor ? &aPhProj : nullptr)))
{
     for (size_t aKIm=0 ; aKIm<aVNames.size() ; aKIm++)
     {
	  aPhProj.ReadMultipleTieP(mVTpm[aKIm],aVNames[aKIm],true);  // true=SVP , ok if no file
	  const auto & aVec = mVTpm[aKIm].mVecTPM;
	  if (! aVec.empty())
	  {
             cObjHeapMTP  aPt;
	     aPt.mIdPt = aVec[0].mId;
	     aPt.mNumCurInIm = 0;
	     aPt.mIdIm = aKIm;
	     aPt.mHeapIndex = HEAP_NO_INDEX;   // probably not necessary, but no harm
	     mHeap.Push(aPt);
	  }
     }

     while ( GetNextConfig()) ;
}

cComputeMergeMulTieP * cReadMTP_Std::CompMerge() {return mCompMerge;}

cComputeMergeMulTieP * AllocStdFromMTP
                      (
                            const std::vector<std::string> &aVNames,
                            cPhotogrammetricProject & aPhProj,
                            bool  WithIndexPt,
                            bool  WithSensor,
                            bool  WithIndexImages
                      )
{
    cReadMTP_Std aRStd(aVNames,aPhProj,WithIndexPt,WithSensor);

    if (WithIndexImages)
        aRStd.CompMerge()->SetImageIndexe();

    return aRStd.CompMerge();
}


}; // MMVII

