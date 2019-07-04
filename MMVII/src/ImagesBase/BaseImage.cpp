#include "include/MMVII_all.h"

// #include <Eigen/Dense>

namespace MMVII
{

/* ========================== */
/*     cDataGenUnTypedIm      */
/* ========================== */


template <const int Dim> cDataGenUnTypedIm<Dim>::cDataGenUnTypedIm
                         (
                             const cPtxd<int,Dim> & aP0,
                             const cPtxd<int,Dim> & aP1
                         )  :
                            cPixBox<Dim>(aP0,aP1)
{
}


/* ========================== */
/*          cPtxd             */
/* ========================== */


/*
template <class Type,const int Dim> cPtxd<Type,Dim>  cPtxd<Type,Dim>::PCste(const Type & aVal)
{
    cPtxd<Type,Dim> aRes;
    for (int aK=0 ; aK<Dim; aK++)
        aRes.mCoords[aK]= aVal;
    return aRes;
}
*/


/* ========================== */
/*          ::                */
/* ========================== */
    //  To test Error_Handler mecanism

static std::string MesNegSz="Negative size in rect object";
static std::string  TestErHandler;
void TestBenchRectObjError(const std::string & aType,const std::string &  aMes,const char * aFile,int aLine)
{
   TestErHandler = aMes;
}

/* ========================== */
/*          cPixBox           */
/* ========================== */

/// Computation to get the point we have at end of iterating a rectangle
template <const int Dim> cPtxd<int,Dim> CalPEnd(const cPtxd<int,Dim> & aP0,const cPtxd<int,Dim> & aP1)
{
    cPtxd<int,Dim> aRes = aP0;
    aRes[Dim-1] = aP1[Dim-1] ;
    return aRes;
}

template <const int Dim>   cPixBox<Dim>::cPixBox(const cPtxd<int,Dim> & aP0,const cPtxd<int,Dim> & aP1,bool AllowEmpty) :
     cTplBox<int,Dim>(aP0,aP1,AllowEmpty),
     mBegin  (*this,aP0),
     mEnd    (*this,CalPEnd(aP0,aP1))
{
}

template <const int Dim>   cPixBox<Dim>::cPixBox(const cPixBox<Dim> & aR) :
   cPixBox<Dim>(aR.mP0,aR.mP1,true)
{
}

template <const int Dim>   cPixBox<Dim>::cPixBox(const cTplBox<int,Dim> & aR) :
   cPixBox<Dim>(aR.P0(),aR.P1(),true)
{
}


template <const int Dim> tINT8  cPixBox<Dim>::IndexeLinear(const tPt & aP) const
{
   tINT8 aRes = 0;
   for (int aK=0 ; aK<Dim ; aK++)
      aRes += tINT8(aP[aK]-tBox::mP0[aK]) * tINT8(tBox::mSzCum[aK]);
   return aRes;
}

/* ========================== */
/*          cTplBox           */
/* ========================== */

template <class Type,const int Dim>   
   cTplBox<Type,Dim>::cTplBox
   (
       const cPtxd<Type,Dim> & aP0,
       const cPtxd<Type,Dim> & aP1,
       bool AllowEmpty
   ) :
       mP0  (aP0),
       mP1  (aP1),
       mSz  (aP0),
       mNbElem (1)
{
    //for (int aK=Dim-1 ; aK>=0 ; aK--)
    for (int aK=0 ; aK<Dim ; aK++)
    {
       mSz[aK] = mP1[aK] - mP0[aK];
       if (AllowEmpty)
       {
          MMVII_INTERNAL_ASSERT_strong(mSz[aK]>=0,MesNegSz);
       }
       else
       {
          MMVII_INTERNAL_ASSERT_strong(mSz[aK]>0,MesNegSz);
       }
       mSzCum[aK] = mNbElem;
       mNbElem *= mSz[aK];
    }
}



template <class Type,const int Dim> bool  cTplBox<Type,Dim>::IsEmpty() const
{
   return mNbElem == 0;
}




template <class Type,const int Dim> void cTplBox<Type,Dim>::AssertSameArea(const cTplBox<Type,Dim> & aR2) const
{
    MMVII_INTERNAL_ASSERT_strong((*this)==aR2,"Rect obj were expected to have identic area");
}
template <class Type,const int Dim> void cTplBox<Type,Dim>::AssertSameSz(const cTplBox<Type,Dim> & aR2) const
{
    MMVII_INTERNAL_ASSERT_strong(Sz()==aR2.Sz(),"Rect obj were expected to have identic size");
}


template <class Type,const int Dim> bool cTplBox<Type,Dim>::operator == (const tBox & aR2) const 
{
    return (mP0==aR2.mP0) && (mP1==aR2.mP1);
}

template <class Type,const int Dim> bool cTplBox<Type,Dim>::IncludedIn(const tBox & aR2) const
{
    return SupEq(mP0,aR2.mP0) && InfEq(mP1,aR2.mP1) ;
}

template <class Type,const int Dim> cTplBox<Type,Dim> cTplBox<Type,Dim>::Translate(const cPtxd<Type,Dim> & aTr) const
{
   return cTplBox<Type,Dim>(mP0+aTr,mP1+aTr);
}


template <class Type,const int Dim> cPtxd<Type,Dim>  cTplBox<Type,Dim>::FromNormaliseCoord(const cPtxd<double,Dim> & aPN) const 
{
    // MMVII_INTERNAL_ASSERT_strong(false,"To Change 
    cPtxd<Type,Dim> aRes;
    for (int aK=0 ; aK<Dim ; aK++)
    {
        // aRes[aK] = mP0[aK] + round_down(mSz[aK]*aPN[aK]);
        aRes[aK] = mP0[aK] + tBaseNumTrait<Type>::RoundDownToType(mSz[aK]*aPN[aK]);
    }
    return Proj(aRes);
}

template <class Type,const int Dim>  cPtxd<double,Dim>  cTplBox<Type,Dim>::RandomNormalised() 
{
   cPtxd<double,Dim>  aRes;
   for (int aK=0 ; aK<Dim ; aK++)
   {
        aRes[aK] = RandUnif_0_1();
   }
   return aRes;
}

template <class Type,const int Dim> cPtxd<Type,Dim>   cTplBox<Type,Dim>::GeneratePointInside() const
{
   return FromNormaliseCoord(RandomNormalised());
}

template <class Type,const int Dim> cTplBox<Type,Dim>  cTplBox<Type,Dim>::GenerateRectInside(double aPowSize) const
{
    cPtxd<Type,Dim> aP0;
    cPtxd<Type,Dim> aP1;
    for (int aK=0 ; aK<Dim ; aK++)
    {
        double aSzRed = pow(RandUnif_0_1(),aPowSize);
        double aX0 = (1-aSzRed) * RandUnif_0_1();
        double aX1 = aX0 + aSzRed;
        int aI0 = round_down(aX0*mSz[aK]);
        int aI1 = round_down(aX1*mSz[aK]);
        aI1 = std::min(int(mP1[aK]-1),std::max(aI1,aI0+1));
        aI0  = std::max(int(mP0[aK]),std::min(aI0,aI1-1));
        aP0[aK] = aI0;
        aP1[aK] = aI1;

    }
    return cTplBox<Type,Dim>(aP0,aP1);
}
#if (0)
#endif



#define MACRO_INSTATIATE_PRECT_DIM(DIM)\
template class cTplBox<tINT4,DIM>;\
template class cTplBox<tREAL8,DIM>;\
template class cPixBox<DIM>;\
template class cDataGenUnTypedIm<DIM>;\
template <> const cPixBox<DIM> cPixBox<DIM>::TheEmptyBox(cPtxd<int,DIM>::PCste(0),cPtxd<int,DIM>::PCste(0),true);



MACRO_INSTATIATE_PRECT_DIM(1)
MACRO_INSTATIATE_PRECT_DIM(2)
MACRO_INSTATIATE_PRECT_DIM(3)
/*

*/


/* ========================== */
/*          cDataTypedIm      */
/* ========================== */


template <class Type,const int Dim> 
    cDataTypedIm<Type,Dim>::cDataTypedIm(const cPtxd<int,Dim> & aP0,const cPtxd<int,Dim> & aP1,Type *aRawDataLin,eModeInitImage aModeInit) :
        cDataGenUnTypedIm<Dim>(aP0,aP1),
        mDoAlloc (aRawDataLin==0),
        mRawDataLin (mDoAlloc ? cMemManager::Alloc<Type>(NbElem())  : aRawDataLin),
        mNbElemMax  (NbElem())
{
   Init(aModeInit);
}

template <class Type,const int Dim>
    void cDataTypedIm<Type,Dim>::Resize(const cPtxd<int,Dim> & aP0,const cPtxd<int,Dim> & aP1,eModeInitImage aModeInit) 
{
    //  WARNING : this work because cDataGenUnTypedIm only calls cRectObj
    //     DO NOT WORK all stuff like :  this->cDataGenUnTypedIm<Dim>::cDataGenUnTypedIm(aP0,aP1);
    // static_cast<cRectObj<Dim>&>(*this) = cRectObj<Dim>(aP0,aP1);

    // this-> cPixBox<Dim>::cRectObj(aP0,aP1);

    // To call the copy constructor of cPixBox, we use a placemennt new
    // Not the best C++, but I don't success to do it other way as constructor cannot  be called explicitely
    new (static_cast<cPixBox<Dim>*>(this)) cPixBox<Dim>(aP0,aP1);

    if (cMemManager::Resize(mRawDataLin,0,mNbElemMax,0,NbElem()))
    {
        mDoAlloc = true;
    }
    Init(aModeInit);
}



template <class Type,const int Dim> 
    cDataTypedIm<Type,Dim>::~cDataTypedIm()
{
   if (mDoAlloc)
      cMemManager::Free(mRawDataLin);
}


template <class Type,const int Dim>  
        double cDataTypedIm<Type,Dim>::L1Dist(const cDataTypedIm<Type,Dim> & aI2) const
{
    tPB::AssertSameArea(aI2);
    double aRes = 0.0;
    for (int aK=0 ; aK<NbElem() ; aK++)
       aRes += std::fabs(mRawDataLin[aK]-aI2.mRawDataLin[aK]);

   return aRes/NbElem();
}
template <class Type,const int Dim>  
        double cDataTypedIm<Type,Dim>::L2Dist(const cDataTypedIm<Type,Dim> & aI2) const
{
    tPB::AssertSameArea(aI2);
    double aRes = 0.0;
    for (int aK=0 ; aK<NbElem() ; aK++)
       aRes += R8Square(mRawDataLin[aK]-aI2.mRawDataLin[aK]);

   return sqrt(aRes/NbElem());
}
template <class Type,const int Dim>  
        double cDataTypedIm<Type,Dim>::LInfDist(const cDataTypedIm<Type,Dim> & aI2) const
{
    tPB::AssertSameArea(aI2);
    double aRes = 0.0;
    for (int aK=0 ; aK<NbElem() ; aK++)
       aRes = std::max(aRes,(double)std::fabs(mRawDataLin[aK]-aI2.mRawDataLin[aK]));

   return aRes;
}



template <class Type,const int Dim>  
        double cDataTypedIm<Type,Dim>::L1Norm() const
{
    double aRes = 0.0;
    for (int aK=0 ; aK<NbElem() ; aK++)
       aRes += std::fabs(mRawDataLin[aK]);

   return aRes/NbElem();
}
template <class Type,const int Dim>  
        double cDataTypedIm<Type,Dim>::L2Norm() const
{
    double aRes = 0.0;
    for (int aK=0 ; aK<NbElem() ; aK++)
       aRes += R8Square(mRawDataLin[aK]);

   return sqrt(aRes/NbElem());
}
template <class Type,const int Dim>  
        double cDataTypedIm<Type,Dim>::LInfNorm() const
{
    double aRes = 0.0;
    for (int aK=0 ; aK<NbElem() ; aK++)
       aRes = std::max(aRes,(double) std::fabs(mRawDataLin[aK]));

   return aRes;
}


template <class Type,const int Dim> void  cDataTypedIm<Type,Dim>::DupIn(cDataTypedIm<Type,Dim> & aIm) const
{
    tPB::AssertSameSz(aIm);
    MemCopy(aIm.RawDataLin(),RawDataLin(),NbElem());
    // MMVII_INTERNAL_ASSERT_strong(mSz[aK]>=0,"");
}




template <class Type,const int Dim> void  cDataTypedIm<Type,Dim>::InitCste(const Type & aVal)
{
   if (aVal==0)
   {
      InitNull();
   }
   else
   {
      for (tINT8 aK=0 ; aK< NbElem() ; aK++)
           mRawDataLin[aK] = aVal;
   }
}

template <class Type,const int Dim> void  cDataTypedIm<Type,Dim>::InitRandom()
{
   for (tINT8 aK=0 ; aK< NbElem() ; aK++)
       mRawDataLin[aK] = tTraits::RandomValue();
}

template <class Type,const int Dim> void  cDataTypedIm<Type,Dim>::InitRandom(const Type & aV0,const Type &aV1)
{
   for (tINT8 aK=0 ; aK< NbElem() ; aK++)
   {
       mRawDataLin[aK] = Type(aV0 + (aV1-aV0) *RandUnif_0_1());
       if (mRawDataLin[aK]==aV1) 
           mRawDataLin[aK]--;
   }
}




template <class Type,const int Dim> void  cDataTypedIm<Type,Dim>::InitRandomCenter()
{
   for (tINT8 aK=0 ; aK< NbElem() ; aK++)
       mRawDataLin[aK] = tTraits::RandomValueCenter();
}

template <class Type,const int Dim> void  cDataTypedIm<Type,Dim>::InitDirac(const cPtxd<int,Dim> & aP,const Type &  aVal)
{
    InitNull();
    mRawDataLin[tPB::IndexeLinear(aP)] = aVal;
}

template <class Type,const int Dim> void  cDataTypedIm<Type,Dim>::InitDirac(const Type &  aVal)
{
    InitDirac((tPB::mP0+tPB::mP1)/2,aVal);
}



template <class Type,const int Dim> void  cDataTypedIm<Type,Dim>::InitNull()
{
    MEM_RAZ(mRawDataLin,NbElem());
}

template <class Type,const int Dim> void  cDataTypedIm<Type,Dim>::InitId()
{
   // Check it is a square matrix
   MMVII_INTERNAL_ASSERT_bench((Dim==2)  ,"Init Id : dim !=2");
   for (int aK=0 ; aK<Dim ; aK++)
   {
       MMVII_INTERNAL_ASSERT_bench((P0()[aK]==0)  ,"Init Id P0!= (0,0)");
   }
   MMVII_INTERNAL_ASSERT_bench((P1()[0]==P1()[1])  ,"Init Id, non square image");
   
   InitNull();
   for (int aK=0 ;  aK<NbElem()  ; aK += P1()[0]+1)
   {
       mRawDataLin[aK] = 1;
   }
}

template <class Type,const int Dim> void  cDataTypedIm<Type,Dim>::Init(eModeInitImage aMode)
{
    switch(aMode)
    {
       case eModeInitImage::eMIA_Rand        : InitRandom(); return;
       case eModeInitImage::eMIA_RandCenter  : InitRandomCenter(); return;
       case eModeInitImage::eMIA_Null        : InitNull(); return;
       case eModeInitImage::eMIA_MatrixId    : InitId(); return;
       case eModeInitImage::eMIA_NoInit      : ;
    }
}


/*
template class cDataTypedIm<tREAL4,1>;
template class cDataTypedIm<tREAL4,2>;
template class cDataTypedIm<tREAL4,3>;
*/

#define MACRO_INSTANTIATE_cDataTypedIm(aType)\
template class cDataTypedIm<aType,1>;\
template class cDataTypedIm<aType,2>;\
template class cDataTypedIm<aType,3>;


MACRO_INSTANTIATE_cDataTypedIm(tINT1)
MACRO_INSTANTIATE_cDataTypedIm(tINT2)
MACRO_INSTANTIATE_cDataTypedIm(tINT4)

MACRO_INSTANTIATE_cDataTypedIm(tU_INT1)
MACRO_INSTANTIATE_cDataTypedIm(tU_INT2)
MACRO_INSTANTIATE_cDataTypedIm(tU_INT4)

MACRO_INSTANTIATE_cDataTypedIm(tREAL4)
MACRO_INSTANTIATE_cDataTypedIm(tREAL8)
MACRO_INSTANTIATE_cDataTypedIm(tREAL16)
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
        
        // cDataTypedIm<tREAL4,2> aBI(cPt2di(2,3),cPt2di(10,9));
    }
    cMemManager::CheckRestoration(aState);
}


void cBenchBaseImage::DoBenchRO()
{
   {
      cPixBox<1>  aR(cPt1di(2),cPt1di(10));
      MMVII_INTERNAL_ASSERT_bench(cPt1di(aR.Sz()) ==cPt1di(8),"Bench sz RectObj");
      MMVII_INTERNAL_ASSERT_bench(aR.NbElem()==8,"Bench sz RectObj");
   }

   // Test the SetErrorHandler mecanism, abality to recover on error
   {
       MMVII_SetErrorHandler(TestBenchRectObjError);
       cPixBox<1>  aR(cPt1di(10),cPt1di(0));
       MMVII_RestoreDefaultHandle();
       MMVII_INTERNAL_ASSERT_bench(MesNegSz==TestErHandler,"Handler mechanism");
   }
   {
      cPixBox<2>  aR(cPt2di(2,3),cPt2di(10,9));
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
