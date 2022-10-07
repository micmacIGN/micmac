#include "include/MMVII_all.h"
#include "include/MMVII_Tpl_Images.h"


using namespace NS_SymbolicDerivative;
using namespace MMVII;

namespace MMVII
{

/* ******************************** */
/*       cSetInterUK_MultipeObj     */
/* ******************************** */

template <class Type> cObjWithUnkowns<Type>::cObjWithUnkowns() :
   mSetInterv (nullptr),
   mNumObj    (-1),
   mIndUk0    (-1),
   mIndUk1    (-1)
{
}

template <class Type> void cObjWithUnkowns<Type>::OnUpdate()
{
}

template <class Type> bool cObjWithUnkowns<Type>::UkIsInit()  const
{
   return mSetInterv != nullptr;
}

template <class Type> void cObjWithUnkowns<Type>::FillIndexes(std::vector<int> & aVect)
{
     for (int aInd=mIndUk0; aInd<mIndUk1 ; aInd++)
         aVect.push_back(aInd);
}


/* ******************************** */
/*       cSetInterUK_MultipeObj     */
/* ******************************** */

template <class Type> cSetInterUK_MultipeObj<Type>::cSetInterUK_MultipeObj() :
    mNbUk (0)
{
}

template <class Type> cSetInterUK_MultipeObj<Type>::~cSetInterUK_MultipeObj()
{
    for (auto &   aVinterv : mVVInterv) // parse object
    {
        aVinterv.mObj->mSetInterv = nullptr;
    }
}

template <class Type> void  cSetInterUK_MultipeObj<Type>::AddOneObj(cObjWithUnkowns<Type> * anObj)
{
     MMVII_INTERNAL_ASSERT_tiny(anObj->mSetInterv==nullptr,"Multiple set interv simultaneously for same object");

     anObj->mIndUk0 = mNbUk;
     anObj->mNumObj = mVVInterv.size();
     anObj->mSetInterv = this;
     mVVInterv.push_back(cSetIntervUK_OneObj<Type>(anObj));
     anObj->PutUknowsInSetInterval();

     anObj->mIndUk1 = mNbUk ;
}

template <class Type> void cSetInterUK_MultipeObj<Type>::AddOneInterv(Type * anAdr,size_t aSz)
{
    mNbUk += aSz;
    mVVInterv.back().mVInterv.push_back(cOneInteralvUnkown<Type>(anAdr,aSz));
}
template <class Type> void cSetInterUK_MultipeObj<Type>::AddOneInterv(std::vector<Type> & aV)
{
    AddOneInterv(aV.data(),aV.size());
}

template <class Type> void cSetInterUK_MultipeObj<Type>::AddOneInterv(cPtxd<Type,3> & aPt)
{
    AddOneInterv(aPt.PtRawData(),3);
}

template <class Type> void cSetInterUK_MultipeObj<Type>::IO_UnKnowns(cDenseVect<Type> & aVect,bool forExport)
{
    size_t anIndex=0;

    for (auto &   aVinterv : mVVInterv) // parse object
    {
        for (auto & anInterv : aVinterv.mVInterv) // parse interv of 1 object
        {
            for (size_t aK=0 ; aK<anInterv.mNb ; aK++)  // parse element of the interv
            {
                Type & aVal = aVect(anIndex++);
                if (forExport)
                    aVal =  anInterv.mVUk[aK];
                else
                    anInterv.mVUk[aK] = aVal;
            }
        }
        if (!forExport)
            aVinterv.mObj->OnUpdate();
    }
}

template <class Type> cDenseVect<Type>  cSetInterUK_MultipeObj<Type>::GetVUnKnowns() const
{
    cDenseVect<Type> aRes(mNbUk);
    const_cast<cSetInterUK_MultipeObj<Type>*>(this)->IO_UnKnowns(aRes,true);

    return aRes;
}

template <class Type> void  cSetInterUK_MultipeObj<Type>::SetVUnKnowns(const cDenseVect<Type> & aVect)
{
     IO_UnKnowns(const_cast<cDenseVect<Type> &>(aVect),false);
}


template class cObjWithUnkowns<tREAL8>;
template class cSetInterUK_MultipeObj<tREAL8>;

};
