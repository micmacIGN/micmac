
#include "MMVII_Tpl_Images.h"

#include "MMVII_SysSurR.h"

using namespace NS_SymbolicDerivative;
using namespace MMVII;

namespace MMVII
{

/* ************************************************************ */
/*                                                              */
/*                cOneLinearConstraint                          */
/*                                                              */
/* ************************************************************ */

/**  Class for a "sparse" dense vector,  i.e a vector that is represented by a dense vector
 */

template <class Type> class cDSVec
{
    public :
       cDSVec(size_t aNbVar);

       void Add(const Type &,int anInd);


       cDenseVect<Type>  mVec;
       cSetIntDyn        mSet;
};

template <class Type> cDSVec<Type>::cDSVec(size_t aNbVar) :
     mVec (aNbVar,eModeInitImage::eMIA_Null),
     mSet (aNbVar)
{
}

template <class Type> void cDSVec<Type>::Add(const Type & aVal,int anInd)
{
     mVec(anInd) += aVal;
     mSet.AddIndFixe(anInd);
}

/*    Class for handling linear constraint in non linear optimization system.
 *    Note the constraint on a vector X as :
 *
 *         mL . X = mC      where mL is a non null vector
 *
 *    The way it is done :
 *
 *       (1) We select an arbitray non null coord of L  Li!=0; (something like the biggest one)
 *       (2) We suppose Li=1.0  (in fact we have it by  mL = mL/Li  , mC = mC/Li)
 *       (3) Let not X' the vector X without Xi
 *       (4) we have Xi =  mC- mL X'
 *       (5) Each time we add a new obs in sytem :
 *            A.X = B
 *            A.x-B =  A' X' -B + Ai Xi =  A' X' -B + Ai (mC-mL X')  
 *
 *            (A'-Ai mL) = B
 *
 */


template <class Type>  class cOneLinearConstraint
{
     public :
       typedef cSparseVect<Type> tSV;
       /**  In Cstr we can fix the index of subst, if it value -1 let the system select the best , fixing can be usefull in case
	* of equivalence
	*/
        cOneLinearConstraint(const tSV&aLP,const Type& aCste);


	void  SubstractIn(cOneLinearConstraint<Type> & aToSub,cDSVec<Type>  & aBuf);

	void ModifDenseEqLinear(cDenseVect<Type> & aCoeff,Type &  aRHS, const cDenseVect<Type> &  aCurSol);
     private :

	void  AddBuf(cDSVec<Type>  & aBuf,const Type & aMul,int aI2Avoid);

        tSV       mLP;     /// Linear part
        int       mISubst; /// Indexe which is substituted
	Type      mCste;   /// Constant of the constrainte   
};

template <class Type> cOneLinearConstraint<Type>::cOneLinearConstraint(const tSV&aLP,const Type& aCste) :
	mLP     (aLP),
	mISubst (-1),
	mCste   (aCste)
{
}

template <class Type> void  cOneLinearConstraint<Type>::AddBuf(cDSVec<Type>  & aBuf,const Type & aMul,int aI2Avoid)
{
    for (const auto & aPair : mLP.IV())
    {
         if (aPair.mInd != aI2Avoid)
	 {
              aBuf.Add(aPair.mInd,aPair.mVal * aMul);
	 }
    }
}

template <class Type> void cOneLinearConstraint<Type>::SubstractIn(cOneLinearConstraint<Type> & aToSub,cDSVec<Type>  & aBuf)
{
    // substract constant
    aToSub.mCste -=  mCste;

    aToSub.AddBuf(aBuf, 1.0,mISubst);
    this ->AddBuf(aBuf,-1.0,mISubst);

    aToSub.mLP.Reset();
    for (const auto &  aInd : aBuf.mSet.mVIndOcc)
    {
         aToSub.mLP.AddIV(aInd,aBuf.mVec(aInd));
    }
}



/*
	mLP  {}
{
    const typename tSV::tCont & aVPair = aLP.IV();
    // if indexe was not forced or is already, get the "best" one
    if ((aKPair<0) || aSetSubst.mOccupied.at(aVPair.at(aKPair).mInd))
    {
       cWhichMax<int,Type> aMaxInd(-1,0);
       // extract the index, not occupied
       for (size_t aKP=0 ; aKP<aVPair.size() ; aKP++)
       {
           if (!aSetSubst.mOccupied.at(aVPair.at(aKP).mInd))
	   {
               aMaxInd.Add(aKP,std::abs(aVPair.at(aKP).mVal));
	   }
       }
       MMVII_INTERNAL_ASSERT_tiny(aMaxInd.IsInit(),"Cannot get index in cOneLinearConstraint");
       aKPair = aMaxInd.IndexExtre();
    }

    // now store the result
    mISubst = aVPair.at(aKPair).mInd;  // The indexe that will be susbstitued
    Type aV0 = aVPair.at(aKPair).mVal; // Value we divide all
    mCste = aCste / aV0;  // normalized constant

    for (const auto & aPair : aVPair)
    {
        if (aPair.mInd != mISubst)
        {
             mLP.AddIV(aPair.mInd,aPair.mVal/aV0);
        }
    }
}


template <class Type> void cOneLinearConstraint<Type>::ModifDenseEqLinear(cDenseVect<Type> & aCoeff,Type &  aRHS, const cDenseVect<Type> & )
{
}
*/

#define INSTANTIATE_LINEAER_CONSTR(TYPE)\
template class  cOneLinearConstraint<TYPE>;\
template class  cDSVec<TYPE>;

INSTANTIATE_LINEAER_CONSTR(tREAL16)
INSTANTIATE_LINEAER_CONSTR(tREAL8)
INSTANTIATE_LINEAER_CONSTR(tREAL4)

// template class  cOneLinearConstraint<tREAL16>;
// template class  cOneLinearConstraint<tREAL8>;
// template class  cOneLinearConstraint<tREAL4>;

};
