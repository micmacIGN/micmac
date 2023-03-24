#include "cMMVII_Appli.h"
#include "MMVII_Tpl_Images.h"
#include "SymbDer/SymbDer_Common.h"
#include "MMVII_SysSurR.h"

using namespace NS_SymbolicDerivative;
using namespace MMVII;

namespace MMVII
{

/* ******************************** */
/*       cGetAdrInfoParam           */
/* ******************************** */


	/*
template <class Type> cGetAdrInfoParam<Type>::cGetAdrInfoParam(const std::string & aPattern) :
	mPattern  (AllocRegex(aPattern))
{
}
*/

template <class Type> cGetAdrInfoParam<Type>::cGetAdrInfoParam(const std::string & aPattern,cObjWithUnkowns<Type> & aObj) :
      // cGetAdrInfoParam<Type>(aPattern)
	mPattern  (AllocRegex(aPattern))
{
     std::vector<cObjWithUnkowns<Type> *>  aVObj = aObj.RecursGetAllUK() ;

     for (auto  aPtr : aVObj)
     {
          aPtr->GetAdrInfoParam(*this);
     }
}

template <class Type> void cGetAdrInfoParam<Type>::TestParam(tObjWUK * anObj,Type * anAdr,const std::string & aName)
{
    if (mPattern.Match(aName))
    {
       mVObjs.push_back(anObj);
       mVNames.push_back(aName);
       mVAdrs.push_back(anAdr);
    }
}

template <class Type> const std::vector<std::string>  &   cGetAdrInfoParam<Type>::VNames() const { return mVNames; }
template <class Type> const std::vector<Type*> &        cGetAdrInfoParam<Type>::VAdrs()  const {return mVAdrs;}
template <class Type> const std::vector<cObjWithUnkowns<Type>*>& cGetAdrInfoParam<Type>::VObjs()  const {return mVObjs;}

template <class Type> void cGetAdrInfoParam<Type>::ShowAllParam(cObjWithUnkowns<Type> & anObj)
{
    cGetAdrInfoParam aGAIP(".*",anObj);

    StdOut() << "===============  Avalaible names =================\n";
    for (const auto & aName  : aGAIP.VNames())
        StdOut()  << "  -[ " << aName << "]\n";
}

template <class Type> void cGetAdrInfoParam<Type>::PatternSetToVal(const std::string & aPattern,tObjWUK & aObj,const Type & aVal)
{
    cGetAdrInfoParam<Type> aGAIP(aPattern,aObj);
    for (auto & anAdr : aGAIP.mVAdrs)
        *anAdr = aVal;
}

/* ******************************** */
/*       cSetInterUK_MultipeObj     */
/* ******************************** */

//  put all value to "bull shit"
template <class Type> cObjWithUnkowns<Type>::cObjWithUnkowns() 
{
   Reset();
}


template <class Type> 
     std::vector<cObjWithUnkowns<Type> *> cObjWithUnkowns<Type>::GetAllUK()
{
  return std::vector<tPtrOUK> {this};
}

template <class Type>
     std::vector<cObjWithUnkowns<Type> *> cObjWithUnkowns<Type>::RecursGetAllUK()
{
   std::vector<cObjWithUnkowns<Type> *>  aRes = {this};

   for (size_t aK0 = 0;  aK0<aRes.size() ; aK0++)
   {
       cObjWithUnkowns<Type> * aCur = aRes[aK0];
       std::vector<cObjWithUnkowns<Type> *> aVecNew = aCur->GetAllUK();
       for (const auto & aPtr : aVecNew)
       {
           if (aPtr != aCur)
              aRes.push_back(aPtr);
       }
   }

   return aRes;
}

template <class Type> cObjWithUnkowns<Type>::~cObjWithUnkowns() 
{
    // MMVII_WARGNING("cObjWithUnkowns mSetInterv==nullptr to reactivate");	
     MMVII_INTERNAL_ASSERT_tiny(mSetInterv==nullptr,"Unfreed object in cObjWithUnkowns");
}

template <class Type> void cObjWithUnkowns<Type>::Reset()
{
   mSetInterv = nullptr;
   mNumObj    = -1;
   mIndUk0    = -1;
   mIndUk1    = -1;
}

// default, dont need update
template <class Type> void cObjWithUnkowns<Type>::OnUpdate()
{
}

template <class Type> bool cObjWithUnkowns<Type>::UkIsInit()  const
{
   return mSetInterv != nullptr;
}

// add indexes  of unknown in a vect, note that indexes are consecutives even if unknown are no in object
template <class Type> void cObjWithUnkowns<Type>::PushIndexes(std::vector<int> & aVect)
{
     for (int aInd=mIndUk0; aInd<mIndUk1 ; aInd++)
         aVect.push_back(aInd);
}


template <class Type> size_t cObjWithUnkowns<Type>::IndOfVal(const Type * aVal) const
{
    return mSetInterv->IndOfVal(*this,aVal);
}

template <class Type> 
    void  cObjWithUnkowns<Type>::GetAdrInfoParam(cGetAdrInfoParam<Type> &) 
{
    MMVII_INTERNAL_ERROR("No default AdrParamFromPattern");
}


template <class Type> int cObjWithUnkowns<Type>::IndUk0() const {return mIndUk0;}
template <class Type> int cObjWithUnkowns<Type>::IndUk1() const {return mIndUk1;}

/* ******************************** */
/*       cSetInterUK_MultipeObj     */
/* ******************************** */

template <class Type> cSetInterUK_MultipeObj<Type>::cSetInterUK_MultipeObj() :
    mNbUk (0)
{
}

template <class Type> void  cSetInterUK_MultipeObj<Type>::Reset()
{
    for (auto &   aVinterv : mVVInterv) // parse object to reset them
    {
        aVinterv.mObj->Reset();
    }
    mVVInterv.clear();
    mNbUk = 0;
}

template <class Type> cSetInterUK_MultipeObj<Type>::~cSetInterUK_MultipeObj()
{
   Reset();
}

template <class Type> void  cSetInterUK_MultipeObj<Type>::AddOneObj(cObjWithUnkowns<Type> * anObj)
{
     // 4 now adopt a rather restrictive/cautious policy,  dont allow simultaneous use of same object
     // in different optimizer.  BTW, if it was happening to be needed, a probably easy solution would
     // be to have for each object a stack of SetInterv.  Woul work as long as the intrication of
     // simultaneaous optimization work in a FILO (first-in/last-out) mode. Which is probably not so
     // restrictive.

     MMVII_INTERNAL_ASSERT_tiny(anObj->mSetInterv==nullptr,"Multiple set interv simultaneously for same object");

     // initialise members of anObj (execpt mIndUk1)
     anObj->mIndUk0 = mNbUk;
     anObj->mNumObj = mVVInterv.size();
     anObj->mSetInterv = this;

     mVVInterv.push_back(cSetIntervUK_OneObj<Type>(anObj)); // add the object in it stack
     anObj->PutUknowsInSetInterval(); // call the object for it to communicate its intervall 

     anObj->mIndUk1 = mNbUk ; // initialise  en of interval
}

template <class Type> void  cSetInterUK_MultipeObj<Type>::AddOneObjIfRequired(cObjWithUnkowns<Type> * anObj)
{
      if (anObj->mSetInterv!=nullptr) 
         return;
      AddOneObj(anObj);
}

        //  ================= method for adding interval of unknowns ======================

template <class Type> void cSetInterUK_MultipeObj<Type>::AddOneInterv(Type * anAdr,size_t aSz)
{
    mNbUk += aSz;
    mVVInterv.back().mVInterv.push_back(cOneInteralvUnkown<Type>(anAdr,aSz));
}

template <class Type> void cSetInterUK_MultipeObj<Type>::AddOneInterv(Type & aVal)
{
    AddOneInterv(&aVal,1);
}

template <class Type> void cSetInterUK_MultipeObj<Type>::AddOneInterv(std::vector<Type> & aV)
{
    AddOneInterv(aV.data(),aV.size());
}

template <class Type> void cSetInterUK_MultipeObj<Type>::AddOneInterv(cPtxd<Type,3> & aPt)
{
    AddOneInterv(aPt.PtRawData(),3);
}

template <class Type> void cSetInterUK_MultipeObj<Type>::AddOneInterv(cPtxd<Type,2> & aPt)
{
    AddOneInterv(aPt.PtRawData(),2);
}

        //  ================= method for transforming unknown of each object in a global one ======================
	
   /*  internal method, used by  GetVUnKnowns and SetVUnKnowns */

template <class Type> void cSetInterUK_MultipeObj<Type>::IO_UnKnowns(cDenseVect<Type> & aVect,bool isSetUK)
{
    size_t anIndex=0; // index that will parse all unknowns

    for (auto &   aVinterv : mVVInterv) // parse object
    {
        for (auto & anInterv : aVinterv.mVInterv) // parse interv of 1 object
        {
            for (size_t aK=0 ; aK<anInterv.mNb ; aK++)  // parse element of the interv
            {
                Type & aVal = aVect(anIndex++); // memorize ref to element of vector and increase
                if (isSetUK)
                    anInterv.mVUk[aK] = aVal;
                else
                    aVal =  anInterv.mVUk[aK];
            }
        }
        if (isSetUK)
            aVinterv.mObj->OnUpdate();
    }
}

template <class Type> cDenseVect<Type>  cSetInterUK_MultipeObj<Type>::GetVUnKnowns() const
{
    cDenseVect<Type> aRes(mNbUk);
    const_cast<cSetInterUK_MultipeObj<Type>*>(this)->IO_UnKnowns(aRes,false);

    return aRes;
}

template <class Type> void  cSetInterUK_MultipeObj<Type>::SetVUnKnowns(const cDenseVect<Type> & aVect)
{
     IO_UnKnowns(const_cast<cDenseVect<Type> &>(aVect),true);
}

template <class Type> size_t  cSetInterUK_MultipeObj<Type>::IndOfVal(const cObjWithUnkowns<Type>& anObj,const Type * aVal) const
{
    const cSetIntervUK_OneObj<Type> & aSI = mVVInterv.at(anObj.mNumObj);
    MMVII_INTERNAL_ASSERT_tiny(aSI.mObj==&anObj,"Incoherence in cSetInterUK_MultipeObj::InOfVal");

    size_t aRes = anObj.mIndUk0;
    for (const auto & anInterv : aSI.mVInterv)
    {
        if ((aVal>=anInterv.mVUk) && (aVal <(anInterv.mVUk+anInterv.mNb)) )
		return  aRes + (aVal-anInterv.mVUk);
	aRes += anInterv.mNb;
    }

    MMVII_INTERNAL_ERROR("IndOfVal cannot be found");

    return 0;
}


template class cGetAdrInfoParam<tREAL4>;
template class cGetAdrInfoParam<tREAL8>;
template class cGetAdrInfoParam<tREAL16>;

template class cObjWithUnkowns<tREAL4>;
template class cSetInterUK_MultipeObj<tREAL4>;
template class cObjWithUnkowns<tREAL8>;
template class cSetInterUK_MultipeObj<tREAL8>;
template class cObjWithUnkowns<tREAL16>;
template class cSetInterUK_MultipeObj<tREAL16>;

};
