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
       MMVII_INTERNAL_ASSERT_strong(mSz[aK]>=0,MesNegSz);
       mNbElem *= mSz[aK];
       mSzCum[aK] = mNbElem;
    }
    // std::cout << mNbElem << "\n";
}

template <const int Dim> bool cRectObj<Dim>::operator == (const cRectObj<Dim> aR2) const 
{
    return (mP0==aR2.mP0) && (mP1==aR2.mP1);
}

template <const int Dim> bool cRectObj<Dim>::IncludedIn(const cRectObj<Dim> & aR2) const
{
    return SupEq(mP0,aR2.mP0) && InfEq(mP1,aR2.mP1) ;
}

template <const int Dim> cRectObj<Dim> cRectObj<Dim>::Translate(const cPtxd<int,Dim> & aTr) const
{
   return cRectObj<Dim>(mP0+aTr,mP1+aTr);
}


template <const int Dim> cPtxd<int,Dim>  cRectObj<Dim>::FromNormaliseCoord(const cPtxd<double,Dim> & aPN) const 
{
    cPtxd<int,Dim> aRes;
    for (int aK=0 ; aK<Dim ; aK++)
    {
        aRes[aK] = mP0[aK] + round_down(mSz[aK]*aPN[aK]);
    }
    return Proj(aRes);
}

template <const int Dim>  cPtxd<double,Dim>  cRectObj<Dim>::RandomNormalised() 
{
   cPtxd<double,Dim>  aRes;
   for (int aK=0 ; aK<Dim ; aK++)
   {
        aRes[aK] = RandUnif_0_1();
   }
   return aRes;
}

template <const int Dim> cPtxd<int,Dim>   cRectObj<Dim>::GeneratePointInside() const
{
   return FromNormaliseCoord(RandomNormalised());
}

template <const int Dim> cRectObj<Dim>  cRectObj<Dim>::GenerateRectInside(double aPowSize) const
{
    cPtxd<int,Dim> aP0;
    cPtxd<int,Dim> aP1;
    for (int aK=0 ; aK<Dim ; aK++)
    {
        double aSzRed = pow(RandUnif_0_1(),aPowSize);
        double aX0 = (1-aSzRed) * RandUnif_0_1();
        double aX1 = aX0 + aSzRed;
        int aI0 = round_down(aX0*mSz[aK]);
        int aI1 = round_down(aX1*mSz[aK]);
        aI1 = std::min(mP1[aK]-1,std::max(aI1,aI0+1));
        aI0  = std::max(mP0[aK],std::min(aI0,aI1-1));
        aP0[aK] = aI0;
        aP1[aK] = aI1;

    }
    return cRectObj<Dim>(aP0,aP1);
}




// (const cPtxd<int,Dim> & aP) const 


template class cRectObj<1>;
template class cRectObj<2>;
template class cRectObj<3>;

template <> const cRectObj<1> cRectObj<1>::Empty00(cPt1di(0),cPt1di(0));
template <> const cRectObj<2> cRectObj<2>::Empty00(cPt2di(0,0),cPt2di(0,0));
template <> const cRectObj<3> cRectObj<3>::Empty00(cPt3di(0,0,0),cPt3di(0,0,0));


/* ========================== */
/*          cDataImGen        */
/* ========================== */


template <class Type,const int Dim> 
    cDataImGen<Type,Dim>::cDataImGen(const cPtxd<int,Dim> & aP0,const cPtxd<int,Dim> & aP1,Type *aDataLin) :
        cRectObj<Dim>(aP0,aP1),
        mDoAlloc (aDataLin==0),
        mDataLin (mDoAlloc ? cMemManager::Alloc<Type>(NbElem())  : aDataLin)
{
}


template <class Type,const int Dim> 
    cDataImGen<Type,Dim>::~cDataImGen()
{
   if (mDoAlloc)
      cMemManager::Free(mDataLin);
}

template <class Type,const int Dim> void  cDataImGen<Type,Dim>::InitRandom()
{
   for (tINT8 aK=0 ; aK< NbElem() ; aK++)
       mDataLin[aK] = tTraits::RandomValue();
}

/*
template class cDataImGen<tREAL4,1>;
template class cDataImGen<tREAL4,2>;
template class cDataImGen<tREAL4,3>;
*/

#define MACRO_INSTANTIATE_cDataImGen(aType)\
template class cDataImGen<aType,1>;\
template class cDataImGen<aType,2>;\
template class cDataImGen<aType,3>;


MACRO_INSTANTIATE_cDataImGen(tINT1)
MACRO_INSTANTIATE_cDataImGen(tINT2)
MACRO_INSTANTIATE_cDataImGen(tINT4)

MACRO_INSTANTIATE_cDataImGen(tU_INT1)
MACRO_INSTANTIATE_cDataImGen(tU_INT2)
MACRO_INSTANTIATE_cDataImGen(tU_INT4)

MACRO_INSTANTIATE_cDataImGen(tREAL4)
MACRO_INSTANTIATE_cDataImGen(tREAL8)
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
        
        // cDataImGen<tREAL4,2> aBI(cPt2di(2,3),cPt2di(10,9));
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
