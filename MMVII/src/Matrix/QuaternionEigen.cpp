#include "include/MMVII_all.h"

#include "MMVII_EigenWrap.h"
#include "ExternalInclude/Eigen/Geometry" 

using namespace Eigen;

namespace MMVII
{

// Eigen::Quaterniond q;
// Eigen::Matrix3d R = q.normalized().toRotationMatrix();


template <class Type> Quaternion<Type> ToEQ(const cPtxd<Type,4> & aP)
{
   return  Quaternion<Type>(aP.x(),aP.y(),aP.z(),aP.t());
}

template <class Type> cPtxd<Type,4>  FromEQ(const Quaternion<Type> & aQ)
{
   return  cPtxd<Type,4>(aQ.w(),aQ.x(),aQ.y(),aQ.z());
}

template<class T> cPtxd<T,4>  operator * (const cPtxd<T,4> & aP1,const cPtxd<T,4> & aP2)
{
  return FromEQ(ToEQ(aP1)*ToEQ(aP2));
}

template<class T> cDenseMatrix<T>  Quat2MatrRot  (const cPtxd<T,4> & aPt)
{
    Quaternion<T> aQ = ToEQ(aPt);
    cDenseMatrix<T> aRes(3,3);

    cNC_EigenMatWrap<T> aWrap(aRes);
    aWrap.EW()  = aQ.normalized().toRotationMatrix();

    return aRes;
}


template<class T> cPtxd<T,4>  MatrRot2Quat  (const cDenseMatrix<T> & aMat)
{
   MMVII_INTERNAL_ASSERT_medium(aMat.Sz()==cPt2di(3,3),"Bad sz for MatrRot2Quat");
   cConst_EigenMatWrap<T> aWrap(aMat);
   // Quaternion<T> aQ(aWrap.EW());  => DONT COMPILE ?!
   Matrix3<T> mat(aWrap.EW());
   Quaternion<T> aQ(mat);
   return FromEQ(aQ);
}


/*
template <class Type>  void Toto()
{
   cNC_EigenMatWrap<Type> aWrap_U(aRes.mMatU);
   aWrap_U.EW() = aJacSVD.matrixU();

   cNC_EigenMatWrap<Type> aWrap_V(aRes.mMatV);
   aWrap_V.EW() = aJacSVD.matrixV();

   cNC_EigenColVectWrap<Type>  aWrapSVal(aRes.mSingularValues);
   aWrapSVal.EW() =  aJacSVD.singularValues();

   return aRes;
}
*/

/* ===================================================== */
/* =====              INSTANTIATION                ===== */
/* ===================================================== */


#define INSTANTIATE_QUATERNION(TYPE)\
template cPtxd<TYPE,4>  MatrRot2Quat  (const cDenseMatrix<TYPE> & aMat);\
template  cDenseMatrix<TYPE>  Quat2MatrRot  (const cPtxd<TYPE,4> & aPt);\
template  cPtxd<TYPE,4>  operator * (const cPtxd<TYPE,4> & aP1,const cPtxd<TYPE,4> & aP2);\
template  Quaternion<TYPE> ToEQ(const cPtxd<TYPE,4> & aP);\
template  cPtxd<TYPE,4>  FromEQ(const Quaternion<TYPE> & aP);

INSTANTIATE_QUATERNION(tREAL4)
INSTANTIATE_QUATERNION(tREAL8)
INSTANTIATE_QUATERNION(tREAL16)


};
