#include "include/MMVII_all.h"

namespace MMVII
{

    //  To test Error_Handler mecanism

static std::string MesNegSz="Negative size in rect object";
static std::string  TestErHandler;
void TestBenchRectObjError(const std::string & aType,const std::string &  aMes,const char * aFile,int aLine)
{
   TestErHandler = aMes;
}

/* ========================== */
/*          cRectObj          */
/* ========================== */

/// Computation to get the point we have at end of iterating a rectangle
template <const int Dim> cPtxd<int,Dim> CalPEnd(const cPtxd<int,Dim> & aP0,const cPtxd<int,Dim> & aP1)
{
    cPtxd<int,Dim> aRes = aP0;
    aRes[Dim-1] = aP1[Dim-1] ;
    return aRes;
}


template <const int Dim>   cRectObj<Dim>::cRectObj(const cPtxd<int,Dim> & aP0,const cPtxd<int,Dim> & aP1) :
     mP0  (aP0),
     mP1  (aP1),
     mSz  (aP0),
     mBegin  (*this,mP0),
     mEnd    (*this,CalPEnd(aP0,aP1)),
     mNbElem (1)
{
    for (int aK=Dim-1 ; aK>=0 ; aK--)
    {
       mSz[aK] = mP1[aK] - mP0[aK];
       MMVII_INTERNAL_ASSERT_strong(mSz[aK]>0,MesNegSz);
       mNbElem *= mSz[aK];
       mSzCum[aK] = mNbElem;
    }
    // std::cout << mNbElem << "\n";
}


template class cRectObj<1>;
template class cRectObj<2>;
template class cRectObj<3>;


/* ========================== */
/*          cBaseImage        */
/* ========================== */


template <class Type,const int Dim> 
    cBaseImage<Type,Dim>::cBaseImage(const cPtxd<int,Dim> & aP0,const cPtxd<int,Dim> & aP1,Type *aDataLin) :
        cRectObj<Dim>(aP0,aP1),
        mDoAlloc (aDataLin==0),
        mDataLin (mDoAlloc ? cMemManager::Alloc<Type>(NbElem())  : aDataLin)
{
}



template <class Type,const int Dim> 
    cBaseImage<Type,Dim>::~cBaseImage()
{
   if (mDoAlloc)
      cMemManager::Free(mDataLin);
}

/*
template class cBaseImage<tREAL4,1>;
template class cBaseImage<tREAL4,2>;
template class cBaseImage<tREAL4,3>;
*/

#define MACRO_INSTANTIATE_cBaseImage(aType)\
template class cBaseImage<aType,1>;\
template class cBaseImage<aType,2>;\
template class cBaseImage<aType,3>;


MACRO_INSTANTIATE_cBaseImage(tINT1)
MACRO_INSTANTIATE_cBaseImage(tINT2)
MACRO_INSTANTIATE_cBaseImage(tINT4)

MACRO_INSTANTIATE_cBaseImage(tU_INT1)
MACRO_INSTANTIATE_cBaseImage(tU_INT2)
MACRO_INSTANTIATE_cBaseImage(tU_INT4)

MACRO_INSTANTIATE_cBaseImage(tREAL4)
MACRO_INSTANTIATE_cBaseImage(tREAL8)
/*
*/


/* ========================== */
/*          cBenchBaseImage   */
/* ========================== */


    ///  cBenchBaseImage
class     cBenchBaseImage
{
    public :
    static void DoBenchBI();
    static void DoBenchRO();
};

void cBenchBaseImage::DoBenchBI()
{
    cMemState  aState = cMemManager::CurState() ;
    {
        
        // cBaseImage<tREAL4,2> aBI(cPt2di(2,3),cPt2di(10,9));
    }
    cMemManager::CheckRestoration(aState);
}


void cBenchBaseImage::DoBenchRO()
{
   {
      cRectObj<1>  aR(cPt1di(2),cPt1di(10));
      MMVII_INTERNAL_ASSERT_bench(cPt1di(aR.Sz()) ==cPt1di(8),"Bench sz RectObj");
      MMVII_INTERNAL_ASSERT_bench(aR.NbElem()==8,"Bench sz RectObj");
   }

   // Test the SetErrorHandler mecanism, abality to recover on error
   {
       MMVII_SetErrorHandler(TestBenchRectObjError);
       cRectObj<1>  aR(cPt1di(10),cPt1di(0));
       MMVII_RestoreDefaultHandle();
       MMVII_INTERNAL_ASSERT_bench(MesNegSz==TestErHandler,"Handler mechanism");
   }
   {
      cRectObj<2>  aR(cPt2di(2,3),cPt2di(10,9));
      MMVII_INTERNAL_ASSERT_bench(cPt2di(aR.Sz()) ==cPt2di(8,6),"Bench sz RectObj");
      MMVII_INTERNAL_ASSERT_bench(aR.NbElem()==48,"Bench sz RectObj");

      MMVII_INTERNAL_ASSERT_bench(aR.Inside(cPt2di(2,8)),"Bench inside rect");
      MMVII_INTERNAL_ASSERT_bench(aR.Inside(cPt2di(9,3)),"Bench inside rect");
      MMVII_INTERNAL_ASSERT_bench(!aR.Inside(cPt2di(2,9)),"Bench inside rect");
      MMVII_INTERNAL_ASSERT_bench(!aR.Inside(cPt2di(1,8)),"Bench inside rect");
      MMVII_INTERNAL_ASSERT_bench(!aR.Inside(cPt2di(10,3)),"Bench inside rect");
      MMVII_INTERNAL_ASSERT_bench(!aR.Inside(cPt2di(9,2)),"Bench inside rect");
      MMVII_INTERNAL_ASSERT_bench(!aR.Inside(cPt2di(10,2)),"Bench inside rect");
   }
}

    //----------  ::  external call ---------

void BenchBaseImage()
{
    cBenchBaseImage::DoBenchBI();
}

void BenchRectObj()
{
    cBenchBaseImage::DoBenchRO();
}




};
