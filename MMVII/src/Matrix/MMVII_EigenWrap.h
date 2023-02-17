#ifndef  _MMVII_EigenWrap_H_
#define  _MMVII_EigenWrap_H_

#include "Eigen/Dense"
#include "MMVII_Matrix.h"


using namespace Eigen;

namespace MMVII
{

/* ============================================= */
/*      Eigen Wrappers class                     */
/* ============================================= */

     //   ---  Matrix Wrapper --- 

template <class Type> class cNC_EigenMatWrap
{
   public :
       typedef  Matrix<Type,Dynamic,Dynamic,RowMajor>  tEigenMat;
       typedef  Map<tEigenMat > tEigenWrap;
       // typedef  Map<Matrix<Type,Dynamic,Dynamic,RowMajor> > tEigenWrap;
       cNC_EigenMatWrap (cDenseMatrix<Type> & aDM) :
          mMat (aDM.DIm().RawDataLin(),aDM.Sz().y(),aDM.Sz().x())
       {
       }
       tEigenWrap & EW() {return mMat;}
   private :
       tEigenWrap mMat;
};

template <class Type> class cConst_EigenMatWrap
{
   public :
       typedef  const Matrix<Type,Dynamic,Dynamic,RowMajor>  tEigenMat;
       typedef  Map<tEigenMat >                              tEigenWrap;
       // typedef  Map<const Matrix<Type,Dynamic,Dynamic,RowMajor> > tEigenWrap;


       cConst_EigenMatWrap (const cDenseMatrix<Type> & aDM) :
          mMat (aDM.DIm().RawDataLin(),aDM.Sz().y(),aDM.Sz().x())
       {
       }
       const tEigenWrap & EW() {return mMat;}
   private :
       tEigenWrap mMat;
};

template <class Type> class cConst_EigenTransposeMatWrap
{
   public :
       typedef  const Matrix<Type,Dynamic,Dynamic,ColMajor>  tEigenMat;
       typedef  Map<tEigenMat >                              tEigenWrap;
       // typedef  Map<const Matrix<Type,Dynamic,Dynamic,RowMajor> > tEigenWrap;


       cConst_EigenTransposeMatWrap (const cDenseMatrix<Type> & aDM) :
          mMat (aDM.DIm().RawDataLin(),aDM.Sz().x(),aDM.Sz().y())
       {
       }
       const tEigenWrap & EW() {return mMat;}
   private :
       tEigenWrap mMat;
};





     //   ---  Line Vector Wrapper --- 

template <class Type> class cNC_EigenLineVectWrap
{
   public :
       typedef  RowVector<Type,Dynamic>   tEigenVect;
       typedef  Map<tEigenVect >          tEigenWrap;
       // typedef  Map<RowVector<Type,Dynamic> >  tEigenWrap;
       cNC_EigenLineVectWrap (cDenseVect<Type> & aVecL) :
          mVecL (aVecL.RawData(),aVecL.Sz())
       {
       }
       tEigenWrap & EW() {return mVecL;}
   private :
       tEigenWrap mVecL;
};

template <class Type> class cConst_EigenLineVectWrap
{
   public :
       typedef  const RowVector<Type,Dynamic>   tEigenVect;
       typedef  Map<tEigenVect >                tEigenWrap;
       // typedef  Map<const RowVector<Type,Dynamic> >  tEigenWrap;
       cConst_EigenLineVectWrap (const cDenseVect<Type> & aVecL) :
          mVecL (aVecL.RawData(),aVecL.Sz())
       {
       }
       tEigenWrap & EW() {return mVecL;}
   private :
       tEigenWrap mVecL;
};

     //   ---  Column Vector Wrapper --- 

template <class Type> class cNC_EigenColVectWrap
{
   public :
       typedef  Map<Vector<Type,Dynamic> >  tEigenWrap;
       cNC_EigenColVectWrap (cDenseVect<Type> & aVecC) :
          mVecC (aVecC.RawData(),aVecC.Sz())
       {
       }
       tEigenWrap & EW() {return mVecC;}
   private :
       tEigenWrap mVecC;
};

template <class Type> class cConst_EigenColVectWrap
{
   public :
       typedef  Map<const Vector<Type,Dynamic> >  tEigenWrap;
       cConst_EigenColVectWrap (const cDenseVect<Type> & aVecC) :
          mVecC (aVecC.RawData(),aVecC.Sz())
       {
       }
       cConst_EigenColVectWrap (const std::vector<Type> & aVecC) :
          mVecC (aVecC.data(),aVecC.size())
       {
       }

       tEigenWrap & EW() {return mVecC;}
   private :
       tEigenWrap mVecC;
};
};

#endif //   _MMVII_EigenWrap_H_

