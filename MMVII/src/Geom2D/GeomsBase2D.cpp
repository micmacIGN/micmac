#include "include/MMVII_all.h"

namespace MMVII
{

/* ========================== */
/*         cBox2di            */
/* ========================== */

cBox2di DilateFromIntervPx(const cBox2di & aBox,int aDPx0,int aDPx1)
{
   cPt2di aP0 = aBox.P0();
   cPt2di aP1 = aBox.P1();
   return  cBox2di
           (
                cPt2di(aP0.x()+aDPx0,aP0.y()),
                cPt2di(aP1.x()+aDPx1,aP1.y())
           );
}


/* ========================== */
/*    cSegment2DCompiled      */
/* ========================== */

template <class Type> cSegment2DCompiled<Type>::cSegment2DCompiled(const tPt& aP1,const tPt& aP2) :
    cSegmentCompiled<Type,2> (aP1,aP2),
    mNorm            (Rot90(this->mTgt))
{
}

template <class Type> cPtxd<Type,2> cSegment2DCompiled<Type>::ToCoordLoc(const tPt& aPt) const
{
    tPt   aV1P = aPt - this->mP1;
    return tPt(Scal(this->mTgt,aV1P),Scal(mNorm,aV1P));
}

template <class Type> cPtxd<Type,2> cSegment2DCompiled<Type>::FromCoordLoc(const tPt& aPt) const
{
    return  this->mP1 + this->mTgt*aPt.x()  + mNorm*aPt.y();
}


/* ========================== */
/*          cSim2D            */
/* ========================== */

template <class Type>  cSim2D<Type> cSim2D<Type>::FromExample(const tPt & aP0In,const tPt & aP1In,const tPt & aP0Out,const tPt & aP1Out )  
{
    tPt aScale = (aP1Out-aP0Out)  /  (aP1In-aP0In);

    return cSim2D<Type>(aP0Out-aScale*aP0In,aScale);
}

template <class Type>  cSim2D<Type> cSim2D<Type>::RandomSimInv(const Type & AmplTr,const Type & AmplSc,const Type & AmplMinSc)
{
    return cSim2D<Type>
	   (
	       tPt::PRandC() * AmplTr,
	       tPt::PRandUnitDiff(tPt(0,0),AmplMinSc/AmplSc)*AmplSc
	   );
}



template <class Type>  cSimilitud3D<Type> cSim2D<Type>::Ext3D() const
{
     Type aNSc = Norm2(mSc);
     cDenseMatrix<Type> aMRot2 = MatOfMul (mSc/aNSc);
     cDenseMatrix<Type> aMRot3 = aMRot2.ExtendSquareMatId(3);

     return cSimilitud3D<Type>
	    (
	         aNSc,
		 TP3z0(mTr),
                 cRotation3D<Type>(aMRot3,false)
	    );

}

/* ========================== */
/*          cAffin2D          */
/* ========================== */

template <class Type>  cAffin2D<Type>::cAffin2D(const tPt & aTr,const tPt & aImX,const tPt aImY) :
    mTr     (aTr),
    mVX     (aImX),
    mVY     (aImY),
    mDelta  (mVX ^ mVY),
    mVInvX  (mDelta  ? tPt(mVY.y(),-mVX.y()) /mDelta : tPt(0,0)),
    mVInvY  (mDelta  ? tPt(-mVY.x(),mVX.x()) /mDelta : tPt(0,0))
{
}
template <class Type>  const int cAffin2D<Type>::NbDOF() {return 6;}
template <class Type> cPtxd<Type,2>  cAffin2D<Type>::Value(const tPt & aP) const 
{
    return  mTr + mVX * aP.x() + mVY *aP.y();
}
template <class Type> cPtxd<Type,2>  cAffin2D<Type>::Inverse(const tPt & aP) const 
{
    return   mVInvX * (aP.x()-mTr.x()) + mVInvY * (aP.y()-mTr.y());
}
template <class Type>  cAffin2D<Type> cAffin2D<Type>::MapInverse() const 
{
	return tTypeMapInv ( VecInverse(-mTr), mVInvX, mVInvY);
}




template <class Type> cPtxd<Type,2>  cAffin2D<Type>::VecValue(const tPt & aP) const 
{
    return   mVX * aP.x() + mVY *aP.y();
}
template <class Type> cPtxd<Type,2>  cAffin2D<Type>::VecInverse(const tPt & aP) const 
{
    return   mVInvX * aP.x() + mVInvY *aP.y();
}


template <class Type>  cAffin2D<Type> cAffin2D<Type>::AllocRandom(const Type & aDeltaMin)
{
   tPt aP0(0,0);
   tTypeMap aRes(aP0,aP0,aP0);
   while (std::abs(aRes.mDelta)<aDeltaMin)
	   aRes =tTypeMap(tPt::PRandC()*Type(10.0),tPt::PRandC()*Type(2.0),tPt::PRandC()*Type(2.0));
   return aRes;
}

template <class Type>  cAffin2D<Type> cAffin2D<Type>::operator *(const tTypeMap& aMap2) const
{
	return tTypeMap 
		( 
		    mTr + VecValue(aMap2.mTr), 
		    VecValue(aMap2.mVX), 
		    VecValue(aMap2.mVY)
		);
}

template <class Type>  cAffin2D<Type> cAffin2D<Type>::Translation(const tPt  & aTr)
{
	return tTypeMap ( aTr, tPt(1,0),tPt(0,1));
}

template <class Type>  cAffin2D<Type> cAffin2D<Type>::Rotation(const Type  & aTeta)
{
	tPt aImX =FromPolar<Type>(Type(1.0),aTeta);
	return tTypeMap (tPt(0,0),aImX,Rot90(aImX));
}

template <class Type>  cAffin2D<Type> cAffin2D<Type>::Homot(const Type & aScale)
{
	return tTypeMap (tPt(0,0), tPt(aScale,0),tPt(0,aScale));
}

template <class Type>  cAffin2D<Type> cAffin2D<Type>::HomotXY(const Type & aScaleX,const Type & aScaleY)
{
	return tTypeMap (tPt(0,0), tPt(aScaleX,0),tPt(0,aScaleY));
}


template <class Type>  const Type& cAffin2D<Type>::Delta() const {return mDelta;}
template <class Type>  const cPtxd<Type,2> & cAffin2D<Type>::Tr() const {return mTr;}
template <class Type>  const cPtxd<Type,2> & cAffin2D<Type>::VX() const {return mVX;}
template <class Type>  const cPtxd<Type,2> & cAffin2D<Type>::VY() const {return mVY;}
template <class Type>  const cPtxd<Type,2> & cAffin2D<Type>::VInvX() const {return mVInvX;}
template <class Type>  const cPtxd<Type,2> & cAffin2D<Type>::VInvY() const {return mVInvY;}




/* ========================== */
/*             ::             */
/* ========================== */

template <class Type> cDenseMatrix<Type> MatOfMul (const cPtxd<Type,2> & aP)
{
    cDenseMatrix<Type> aRes(2);

    SetCol(aRes,0,aP);         // P * (1,0)
    SetCol(aRes,1,Rot90(aP));  // P * (0,1)

    return aRes;
}

/* ========================== */
/*       INSTANTIATION        */
/* ========================== */

#define INSTANTIATE_GEOM_REAL(TYPE)\
template class cSegment2DCompiled<TYPE>;\
template class  cAffin2D<TYPE>;

INSTANTIATE_GEOM_REAL(tREAL4)
INSTANTIATE_GEOM_REAL(tREAL8)
INSTANTIATE_GEOM_REAL(tREAL16)



#define MACRO_INSTATIATE_GEOM2D(TYPE)\
template  cSim2D<TYPE> cSim2D<TYPE>::RandomSimInv(const TYPE & AmplTr,const TYPE & AmplSc,const TYPE & AmplMinSc);\
template  cSim2D<TYPE> cSim2D<TYPE>::FromExample(const tPt & aP0In,const tPt & aP1In,const tPt & aP0Out,const tPt & aP1Out )  ;\
template  cSimilitud3D<TYPE> cSim2D<TYPE>::Ext3D() const;\
template  cDenseMatrix<TYPE> MatOfMul (const cPtxd<TYPE,2> & aP);


MACRO_INSTATIATE_GEOM2D(tREAL4)
MACRO_INSTATIATE_GEOM2D(tREAL8)
MACRO_INSTATIATE_GEOM2D(tREAL16)



};
