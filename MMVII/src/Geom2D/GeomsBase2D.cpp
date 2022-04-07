#include "include/MMVII_all.h"

namespace MMVII
{


template <class Type>  cSim2D<Type> cSim2D<Type>::FromExample(const tPt & aP0In,const tPt & aP1In,const tPt & aP0Out,const tPt & aP1Out )  
{
    tPt aScale = (aP1Out-aP0Out)  /  (aP1In-aP0In);

    return cSim2D<Type>(aP0Out-aScale*aP1In,aScale);
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


template <class Type> cDenseMatrix<Type> MatOfMul (const cPtxd<Type,2> & aP)
{
    cDenseMatrix<Type> aRes(2);

    SetCol(aRes,0,aP);         // P * (1,0)
    SetCol(aRes,1,Rot90(aP));  // P * (0,1)

    return aRes;
}



/* ========================== */
/*          ::                */
/* ========================== */

#define MACRO_INSTATIATE_GEOM2D(TYPE)\
template  cSim2D<TYPE> cSim2D<TYPE>::FromExample(const tPt & aP0In,const tPt & aP1In,const tPt & aP0Out,const tPt & aP1Out )  ;\
template  cSimilitud3D<TYPE> cSim2D<TYPE>::Ext3D() const;\
template  cDenseMatrix<TYPE> MatOfMul (const cPtxd<TYPE,2> & aP);

/*

template  cSim2D<tREAL8> cSim2D<tREAL8>::FromExample(const tPt & aP0In,const tPt & aP1In,const tPt & aP0Out,const tPt & aP1Out )  ;
#define MACRO_INSTATIATE_GEOM2D(TYPE)\
template cPtxd<TYPE,3>  operator ^ (const cPtxd<TYPE,3> & aP1,const cPtxd<TYPE,3> & aP2);\
template cDenseMatrix<TYPE> MatFromCols(const cPtxd<TYPE,3>&,const cPtxd<TYPE,3>&,const cPtxd<TYPE,3>&);\
template cDenseMatrix<TYPE> MatFromLines(const cPtxd<TYPE,3>&,const cPtxd<TYPE,3>&,const cPtxd<TYPE,3>&);\
template cPtxd<TYPE,3>  PFromNumAxe(int aNum);\
template cPtxd<TYPE,3>  VOrthog(const cPtxd<TYPE,3> & aP);

MACRO_INSTATIATE_GEOM2D(tREAL4)
*/


MACRO_INSTATIATE_GEOM2D(tREAL4)
MACRO_INSTATIATE_GEOM2D(tREAL8)
MACRO_INSTATIATE_GEOM2D(tREAL16)



};
