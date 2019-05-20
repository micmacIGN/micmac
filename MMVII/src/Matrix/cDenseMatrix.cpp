#include "include/MMVII_all.h"

namespace MMVII
{

/** Unopt dense Matrix are not very Usefull in "final" version
of  MMVII, but it's the opportunity to test the default method
implemanted in cDataMatrix */


template <class Type> class cUnOptDenseMatrix : public cDataMatrix
{
    public :
        typedef cIm2D<Type> tIm;
        typedef cDataIm2D<Type> tDIm;

        cUnOptDenseMatrix(int aX,int aY);

        tMatrElem GetElem(int aX,int  aY) const override ;
        void  SetElem(int  aX,int  aY,const tMatrElem &) override;
   protected :
       tDIm & DIm() {return mIm.DIm();}
       const tDIm & DIm() const {return mIm.DIm();}
       
       tIm  mIm;
};

template <class Type> tMatrElem cUnOptDenseMatrix<Type>::GetElem(int aX,int  aY) const 
{
    return DIm().GetV(cPt2di(aX,aY));
}

template <class Type> void cUnOptDenseMatrix<Type>::SetElem(int aX,int  aY,const tMatrElem & aV) 
{
    DIm().SetV(cPt2di(aX,aY),aV);
}

template <class Type> cUnOptDenseMatrix<Type>::cUnOptDenseMatrix(int aX,int aY) :
                           cDataMatrix(aX,aY),
                           mIm(cPt2di(aX,aY))
{
}
     

/* ===================================================== */
/* ===================================================== */
/* ===================================================== */

static double FTestMatr(const cPt2di & aP)
{
    return 1 + 1/(aP.x()+2.45) + 1/(aP.y()*aP.y() + 3.14);
}

static double FTestVect(const int & aK)
{
    return  (aK+3.0) / (aK+17.899 + 1e-2 * aK * aK);
}


template <class TypeMatr,class TypeVect>  void TplBenchMatrix(int aSzX,int aSzY)
{
    TypeMatr aM(aSzX,aSzY);
    TypeMatr aMt(aSzY,aSzX);
    for (const auto & aP : aM)
    {
        aM.SetElem(aP.x(),aP.y(),FTestMatr(aP));
        aMt.SetElem(aP.y(),aP.x(),FTestMatr(aP));
    }
    // Bench Col mult
    {
        cDenseVect<TypeVect> aVIn(aSzX),aVOut(aSzY);
        for (int aX=0 ; aX<aSzX ; aX++)
        {
            aVIn(aX) = FTestVect(aX);
        }
        aM.MulCol(aVOut,aVIn);
        for (int aY=0 ; aY<aSzY ; aY++)
        {
             double aV1 =  aVOut(aY);
             double aV2 =  aM.MulCol(aY,aVIn);
             double aV3 = 0 ;
             for (int aX=0 ; aX<aSzX ; aX++)
                 aV3 +=  FTestVect(aX) * FTestMatr(cPt2di(aX,aY));
             // std::cout << "VVV "<< aV1 -  aV2 << " " << aV3-aV2 << "\n";;
             MMVII_INTERNAL_ASSERT_bench(std::abs(aV1-aV2)<1e-5,"Bench Matrixes");
             MMVII_INTERNAL_ASSERT_bench(std::abs(aV2-aV3)<1e-5,"Bench Matrixes");
        }
    }
    // Bench Line mult
    {
        cDenseVect<TypeVect> aVIn(aSzY),aVOut(aSzX);
        for (int aY=0 ; aY<aSzY ; aY++)
        {
            aVIn(aY) = FTestVect(aY);
        }
        aM.MulLine(aVOut,aVIn);
        for (int aX=0 ; aX<aSzX ; aX++)
        {
             double aV1 =  aVOut(aX);
             double aV2 =  aM.MulLine(aX,aVIn);
             double aV3 = 0 ;
             for (int aY=0 ; aY<aSzY ; aY++)
                 aV3 +=  FTestVect(aY) * FTestMatr(cPt2di(aX,aY));
             MMVII_INTERNAL_ASSERT_bench(std::abs(aV1-aV2)<1e-5,"Bench Matrixes");
             MMVII_INTERNAL_ASSERT_bench(std::abs(aV2-aV3)<1e-5,"Bench Matrixes");
        }
    }
}


void BenchDenseMatrix0()
{
    TplBenchMatrix<cUnOptDenseMatrix<tREAL8>,tREAL4 > (3,2);
    TplBenchMatrix<cUnOptDenseMatrix<tREAL8>,tREAL16 > (3,2);
    TplBenchMatrix<cUnOptDenseMatrix<tREAL8>,tREAL8 > (2,3);
    TplBenchMatrix<cUnOptDenseMatrix<tREAL4>,tREAL8 > (2,3);
}


#define INSTANTIATE_DENSE_MATRICES(Type)\
template  class  cUnOptDenseMatrix<Type>;\


INSTANTIATE_DENSE_MATRICES(tREAL4)
INSTANTIATE_DENSE_MATRICES(tREAL8)
INSTANTIATE_DENSE_MATRICES(tREAL16)
/*
*/



};
