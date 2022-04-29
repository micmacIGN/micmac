#include "include/MMVII_all.h"

namespace MMVII
{

/* ========================== */
/*    cSegment2DCompiled      */
/* ========================== */

template <class Type> cSegment2DCompiled<Type>::cSegment2DCompiled(const tPt& aP1,const tPt& aP2) :
    cSegment<Type,2> (aP1,aP2),
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
class cSegment2DCompiled<TYPE>;

INSTANTIATE_GEOM_REAL(tREAL4)
INSTANTIATE_GEOM_REAL(tREAL8)
INSTANTIATE_GEOM_REAL(tREAL16)



#define MACRO_INSTATIATE_GEOM2D(TYPE)\
template  cSim2D<TYPE> cSim2D<TYPE>::FromExample(const tPt & aP0In,const tPt & aP1In,const tPt & aP0Out,const tPt & aP1Out )  ;\
template  cSimilitud3D<TYPE> cSim2D<TYPE>::Ext3D() const;\
template  cDenseMatrix<TYPE> MatOfMul (const cPtxd<TYPE,2> & aP);


MACRO_INSTATIATE_GEOM2D(tREAL4)
MACRO_INSTATIATE_GEOM2D(tREAL8)
MACRO_INSTATIATE_GEOM2D(tREAL16)



};
