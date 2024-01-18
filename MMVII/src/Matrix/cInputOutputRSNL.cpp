
#include "MMVII_Tpl_Images.h"

#include "MMVII_SysSurR.h"

using namespace NS_SymbolicDerivative;
using namespace MMVII;

namespace MMVII
{



template <class T1,class T2> void ConvertVWD(cInputOutputRSNL<T1> & aIO1 , const cInputOutputRSNL<T2> & aIO2)
{
     Convert(aIO1.mWeights,aIO2.mWeights);
     Convert(aIO1.mVals,aIO2.mVals);

     aIO1.mDers.resize(aIO2.mDers.size());
     for (size_t aKDer=0 ; aKDer<aIO1.mDers.size() ; aKDer++)
         Convert(aIO1.mDers.at(aKDer),aIO2.mDers.at(aKDer));
}

/* ************************************************************ */
/*                                                              */
/*                cInputOutputRSNL                              */
/*                                                              */
/* ************************************************************ */


template <class Type>  cInputOutputRSNL<Type>::cInputOutputRSNL(const tVectInd& aVInd,const tStdVect & aVObs):
     mGlobVInd  (aVInd),
     mVObs      (aVObs),
     mNbTmpUk   (0)
{

    //  Check consistency on temporary indexes
    for (const auto & anInd : aVInd)
    {
        if (cSetIORSNL_SameTmp<Type>::IsIndTmp(anInd))
        {
	    mNbTmpUk++;
        }
    }
    // MMVII_INTERNAL_ASSERT_tiny(mNbTmpUk==mVTmpUk.size(),"Size Tmp/subst in  cInputOutputRSNL");
}


template <class Type>  cInputOutputRSNL<Type>::cInputOutputRSNL(bool Fake,const cInputOutputRSNL<tREAL8> & aR_IO) :
	cInputOutputRSNL<Type>
	(
	     aR_IO.mGlobVInd,
	     VecConvert<Type,tREAL8>(aR_IO.mVObs)
	)
{
	ConvertVWD(*this,aR_IO);
}


template <class Type> Type cInputOutputRSNL<Type>::WeightOfKthResisual(int aK) const
{
   switch (mWeights.size())
   {
	   case 0 :  return 1.0;
	   case 1 :  return mWeights[0];
	   default  : return mWeights.at(aK);
   }
}
template <class Type> size_t cInputOutputRSNL<Type>::NbUkTot() const
{
	return mGlobVInd.size() ;
}

template <class Type> bool cInputOutputRSNL<Type>::IsOk() const
{
     if (mVals.size() !=mDers.size()) 
        return false;

     if (mVals.empty())
        return false;

     {
         size_t aNbUk = NbUkTot();
         for (const auto & aDer : mDers)
             if (aDer.size() != aNbUk)
                return false;
     }

     {
         size_t aSzW =  mWeights.size();
         if ((aSzW>1) && (aSzW!= mVals.size()))
            return false;
     }
     return true;
}





/* ************************************************************ */
/*                                                              */
/*                cSetIORSNL_SameTmp                            */
/*                                                              */
/* ************************************************************ */

template <class Type> cSetIORSNL_SameTmp<Type>::cSetIORSNL_SameTmp
                      (
		           const tStdVect & aValTmpUk,
                           const tVectInd & aVFix,
		           const tStdVect & aValFix
                      ) :
        mVFix              (aVFix),
	mValFix            (aValFix),
	mOk                (false),
	mNbTmpUk           (aValTmpUk.size()),
	mValTmpUk          (aValTmpUk),
	mVarTmpIsFrozen    (mNbTmpUk,false),
	mValueFrozenVarTmp (mNbTmpUk,-283971), // random val
	mNbEq              (0),
	mSetIndTmpUk       (mNbTmpUk)
{
    MMVII_INTERNAL_ASSERT_tiny((aVFix.size()==aValFix.size()) || aValFix.empty(),"Bad size for fix var tmp");

    for (size_t aKInd=0 ; aKInd<aVFix.size() ; aKInd++)
    {
        int anIndFix = aVFix[aKInd];
	Type aVal = aValFix.empty() ? Val1TmpUk(anIndFix)  : aValFix.at(aKInd);

	// Need to fix the var that will be elimined, need to do it now because line after will change
        AddFixVarTmp(anIndFix,aVal,1.0); 
        mVarTmpIsFrozen.at(ToIndTmp(anIndFix)) = true;
        mValueFrozenVarTmp.at(ToIndTmp(anIndFix)) = aVal;
    }
}

template <class Type> int cSetIORSNL_SameTmp<Type>::NbRedundacy() const
{
   return mNbEq -mNbTmpUk;
}

template <class Type> cSetIORSNL_SameTmp<Type>::cSetIORSNL_SameTmp(bool Fake,const cSetIORSNL_SameTmp<tREAL8> & aR_Set)  :
	cSetIORSNL_SameTmp<Type> 
	(
	     VecConvert<Type,tREAL8>(aR_Set.mValTmpUk),
	     aR_Set.mVFix,
	     VecConvert<Type,tREAL8>(aR_Set.mValFix)
	)
{
    for (const auto & anIO : aR_Set.mVEq)
         AddOneEq(cInputOutputRSNL<Type>(false,anIO));
}




template <class Type> size_t cSetIORSNL_SameTmp<Type>::ToIndTmp(int anInd) { return -(anInd+1); }
template <class Type> bool   cSetIORSNL_SameTmp<Type>::IsIndTmp(int anInd) 
{ 
    return anInd<0; 
}
template <class Type> size_t cSetIORSNL_SameTmp<Type>::NbTmpUk() const { return mNbTmpUk; }
template <class Type> const std::vector<Type> & cSetIORSNL_SameTmp<Type>::ValTmpUk() const { return mValTmpUk; }
template <class Type> Type  cSetIORSNL_SameTmp<Type>::Val1TmpUk(int aInd) const { return mValTmpUk.at(ToIndTmp(aInd));}



template <class Type> void cSetIORSNL_SameTmp<Type>::AddOneEq(const tIO_OneEq & anIO_In)
{
    mVEq.push_back(anIO_In);
    tIO_OneEq & anIO = mVEq.back();

    MMVII_INTERNAL_ASSERT_tiny(anIO.IsOk(),"Bad size for cInputOutputRSNL");

    // for (const auto & anInd : anIO.mGlobVInd)
    for (size_t aKInd=0 ; aKInd<anIO.mGlobVInd.size() ;aKInd++)
    {
        int anIndSigned = anIO.mGlobVInd[aKInd];
        if (IsIndTmp(anIndSigned))
	{
           size_t aIndPos = ToIndTmp(anIndSigned);
           mSetIndTmpUk.AddInd(aIndPos); // add it to the computed list of indexes
           if (mVarTmpIsFrozen.at(aIndPos))
           {
              Type aDeltaVar = mValueFrozenVarTmp.at(aIndPos) - mValTmpUk.at(aIndPos);
              for (size_t aKEq=0 ; aKEq<anIO.mVals.size() ; aKEq++)
              {
                   Type & aVDer = anIO.mDers.at(aKEq).at(aKInd);
		   anIO.mVals[aKEq]  +=  aVDer * aDeltaVar;
		   aVDer = 0;
              }
           }
	}
    }

    mNbEq += anIO.mVals.size();
    if 
    (
            (mNbEq >= mNbTmpUk)  // A priori there is no use to less or equal equation, this doesnt give any constraint
                                 // but useful for GCP with 3d constraints and no 2d obs
            && ( mSetIndTmpUk.NbElem()== mNbTmpUk)  // we are sure to have good index, because we cannot add oustide
    )
    {
        mOk = true; 
    }
}

template <class Type> void   cSetIORSNL_SameTmp<Type>::AddFixVarTmp (int aInd,const Type& aVal,const Type& aWeight)
{
     MMVII_INTERNAL_ASSERT_tiny
     (
	 cSetIORSNL_SameTmp<Type>::IsIndTmp(aInd),
	 "Non tempo index in AddFixVarTmp"
     );

     // tVectInd aVInd{anInd};

     cInputOutputRSNL<Type> aIO({aInd},{});
     aIO.mWeights.push_back(aWeight);
     aIO.mDers.push_back({1.0});
     Type aDVal = Val1TmpUk(aInd)-aVal;
     aIO.mVals.push_back({aDVal});

     AddOneEq(aIO);
}

template <class Type> void   cSetIORSNL_SameTmp<Type>::AddFixCurVarTmp (int aInd,const Type& aWeight)
{
     AddFixVarTmp(aInd,Val1TmpUk(aInd),aWeight); 
}

template <class Type> 
    const std::vector<cInputOutputRSNL<Type> >& 
          cSetIORSNL_SameTmp<Type>::AllEq() const
{
     return mVEq;
}

template <class Type> void cSetIORSNL_SameTmp<Type>::AssertOk() const
{
      MMVII_INTERNAL_ASSERT_tiny(mOk,"Not enough eq to use tmp unknowns");
}



/* ************************************************************ */
/*                                                              */
/*                  INSTANTIATION                               */
/*                                                              */
/* ************************************************************ */

#define INSTANTIATE_RESOLSYSNL(TYPE)\
template class  cInputOutputRSNL<TYPE>;\
template class  cSetIORSNL_SameTmp<TYPE>;\
template void ConvertVWD(cInputOutputRSNL<tREAL8> & aIO1 , const cInputOutputRSNL<TYPE> & aIO2);
         

INSTANTIATE_RESOLSYSNL(tREAL4)
INSTANTIATE_RESOLSYSNL(tREAL8)
INSTANTIATE_RESOLSYSNL(tREAL16)


};
