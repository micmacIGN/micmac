#include "MMVII_Images.h"
#include "MMVII_Image2D.h"
#include <algorithm>
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


template <class Type> cDataGenUnTypedIm<2> * Tpl_ReadIm2DGen(const cDataFileIm2D &aDFI,const cBox2di & aBox)
{
   cDataIm2D<Type> * aDIm  = new  cDataIm2D<Type>(cPt2di(0,0),aBox.Sz());
   aDIm->Read(aDFI,aBox.P0());
   return aDIm;
}

cDataGenUnTypedIm<2> * ReadIm2DGen(const std::string &aName,cBox2di  aBox)
{
    cDataFileIm2D  aDFI = cDataFileIm2D::Create(aName,eForceGray::Yes);

    if (aBox.IsEmpty())
        aBox = cBox2di(cPt2di(0,0),aDFI.Sz());

    switch (aDFI.Type()) {
        case eTyNums::eTN_U_INT1 :   return Tpl_ReadIm2DGen<tU_INT1>(aDFI,aBox);
        case eTyNums::eTN_U_INT2 :   return Tpl_ReadIm2DGen<tU_INT2>(aDFI,aBox);
        case eTyNums::eTN_INT1 :     return Tpl_ReadIm2DGen<tINT1>(aDFI,aBox);
        case eTyNums::eTN_INT2 :     return Tpl_ReadIm2DGen<tINT2>(aDFI,aBox);
        case eTyNums::eTN_INT4 :     return Tpl_ReadIm2DGen<tINT4>(aDFI,aBox);
        case eTyNums::eTN_REAL4 :    return Tpl_ReadIm2DGen<tREAL4>(aDFI,aBox);
        default :
            MMVII_INTERNAL_ERROR("Unhandled type in ReadIm2DGen");
            return nullptr;
    }

    MMVII_INTERNAL_ERROR("Unhandled type in ReadIm2DGen");
    return nullptr;
}

cDataGenUnTypedIm<2> * ReadIm2DGen(const std::string &aName)
{
    return ReadIm2DGen(aName,cBox2di::Empty());
}

template <const int Dim>
std::pair<tREAL8,cPt2dr> cDataGenUnTypedIm<Dim>::GetValueAndGradInterpol(const cDiffInterpolator1D &,const cPt2dr & aP) const
{
    MMVII_INTERNAL_ERROR("Unhandled type in cDataGenUnTypedIm::GetValueAndGradInterpol");
    return {0.,{0.,0.}};
}


/* ========================== */
/*          ::                */
/* ========================== */
    //  To test Error_Handler mecanism

static std::string MesNegSz="Negative size in rect object";
static std::string  TestErHandler;
static void TestBenchRectObjError(const std::string & aType,const std::string &  aMes,const char * aFile,int aLine)
{
   TestErHandler = aMes;
}



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

template <>   cDataTypedIm<tREAL8,1> * cDataTypedIm<tREAL8,1>::AllocIm(const tPix& aPix)
{
   return new cDataIm1D<tREAL8>(cPt1di(0),aPix);
}
template <>   cDataTypedIm<tREAL8,2> * cDataTypedIm<tREAL8,2>::AllocIm(const tPix& aPix)
{
   return new cDataIm2D<tREAL8>(cPt2di(0,0),aPix);
}
template <>   cDataTypedIm<tREAL8,3> * cDataTypedIm<tREAL8,3>::AllocIm(const tPix& aPix)
{
   return new cDataIm3D<tREAL8>(aPix);
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
        double cDataTypedIm<Type,Dim>::L1Dist(const cDataTypedIm<Type,Dim> & aI2,bool isAvg) const
{
    tPB::AssertSameArea(aI2);
    double aRes = 0.0;
    for (int aK=0 ; aK<NbElem() ; aK++)
       aRes += std::fabs(mRawDataLin[aK]-aI2.mRawDataLin[aK]);

   return isAvg ? aRes/NbElem() : aRes;
}

template <class Type,const int Dim>  
        double cDataTypedIm<Type,Dim>::SqL2Dist(const cDataTypedIm<Type,Dim> & aI2,bool isAvg) const
{
    tPB::AssertSameArea(aI2);
    double aRes = 0.0;
    for (int aK=0 ; aK<NbElem() ; aK++)
       aRes += R8Square(mRawDataLin[aK]-aI2.mRawDataLin[aK]);

   return isAvg ? aRes/NbElem() : aRes;
}

template <class Type,const int Dim>  
        double cDataTypedIm<Type,Dim>::L2Dist(const cDataTypedIm<Type,Dim> & aI2,bool isAvg) const
{
   return sqrt(SqL2Dist(aI2,isAvg));
}


template <class Type,const int Dim>  
        double cDataTypedIm<Type,Dim>::SafeMaxRelDif(const cDataTypedIm<Type,Dim> & aI2,tREAL8 aEps) const
{
    tPB::AssertSameArea(aI2);
    double aRes = 0;
    for (int aK=0 ; aK<NbElem() ; aK++)
    {
       tREAL8 aV1 = mRawDataLin[aK];
       tREAL8 aV2 = aI2.mRawDataLin[aK];
       UpdateMax(aRes,fabs(aV1-aV2)/std::max(aEps,std::max(std::fabs(aV1),std::fabs(aV2))));
    }

   return aRes;
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
        double cDataTypedIm<Type,Dim>::L1Norm(bool isAvg) const
{
    double aRes = 0.0;
    for (int aK=0 ; aK<NbElem() ; aK++)
       aRes += std::fabs(mRawDataLin[aK]);

   return isAvg ? aRes/NbElem() : aRes;
}
template <class Type,const int Dim>  
        double cDataTypedIm<Type,Dim>::SqL2Norm(bool isAvg) const
{
    double aRes = 0.0;
    for (int aK=0 ; aK<NbElem() ; aK++)
       aRes += R8Square(mRawDataLin[aK]);

   return isAvg ? aRes/NbElem() : aRes;
}

template <class Type,const int Dim>  
        double cDataTypedIm<Type,Dim>::L2Norm(bool isAvg) const
{
   return sqrt(SqL2Norm(isAvg));
}

template <class Type,const int Dim>  
        double cDataTypedIm<Type,Dim>::LInfNorm() const
{
    double aRes = 0.0;
    for (int aK=0 ; aK<NbElem() ; aK++)
       aRes = std::max(aRes,(double) std::fabs(mRawDataLin[aK]));

   return aRes;
}

template <class Type,const int Dim>  Type     cDataTypedIm<Type,Dim>::MinVal() const
{
    Type aRes = mRawDataLin[0];
    for (int aK=1 ; aK<NbElem() ; aK++)
        UpdateMin(aRes,mRawDataLin[aK]);
   return aRes;
}
template <class Type,const int Dim>  Type     cDataTypedIm<Type,Dim>::MaxVal() const
{
    Type aRes = mRawDataLin[0];
    for (int aK=1 ; aK<NbElem() ; aK++)
        UpdateMax(aRes,mRawDataLin[aK]);
   return aRes;
}
template <class Type,const int Dim>  tREAL16     cDataTypedIm<Type,Dim>::SomVal() const
{
    tREAL16 aRes = mRawDataLin[0];
    for (int aK=1 ; aK<NbElem() ; aK++)
        aRes += mRawDataLin[aK];
   return aRes;
}
template <class Type,const int Dim>  tREAL16     cDataTypedIm<Type,Dim>::MoyVal() const
{
   return SomVal() / NbElem();
}



template <class Type,const int Dim> void  cDataTypedIm<Type,Dim>::DupIn(cDataTypedIm<Type,Dim> & aIm) const
{
    tPB::AssertSameSz(aIm);
    MemCopy(aIm.RawDataLin(),RawDataLin(),NbElem());
    // MMVII_INTERNAL_ASSERT_strong(mSz[aK]>=0,"");
}

template <class Type,const int Dim> void  cDataTypedIm<Type,Dim>::DupInVect(std::vector<Type> & aVec) const
{
    aVec.resize(NbElem());
    MemCopy(aVec.data(),RawDataLin(),NbElem());
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

template <class Type,const int Dim> void  cDataTypedIm<Type,Dim>::InitBorder(const Type & aVal)
{
   int aLarg = 1;
   if (MinAbsCoord(tPB::Sz()) > (2*aLarg))
   {
      cBorderPixBox<Dim> aBorder(this->RO(),aLarg);

      for (const auto & aP : aBorder)
      {
          mRawDataLin[tPB::IndexeLinear(aP)] = aVal;
      }
   }
   else
   {
      InitCste(aVal);
   }
}

template <class Type,const int Dim> void  cDataTypedIm<Type,Dim>::InitInteriorAndBorder(const Type & aVInt,const Type & aVB) 
{
   InitCste(aVInt);
   InitBorder(aVB);
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
       case eModeInitImage::eMIA_V1          : InitCste(1); return;
       case eModeInitImage::eMIA_MatrixId    : InitId(); return;
       case eModeInitImage::eMIA_NoInit      : ;
    }
}


template <class Type,const int Dim> int  cDataTypedIm<Type,Dim>::VI_GetV(const cPtxd<int,Dim> & aP)  const 
{
    tPB::AssertInside(aP);
    return round_ni(mRawDataLin[tPB::IndexeLinear(aP)]);
}
template <class Type,const int Dim> double  cDataTypedIm<Type,Dim>::VD_GetV(const cPtxd<int,Dim> & aP)  const 
{
    tPB::AssertInside(aP);
    return mRawDataLin[tPB::IndexeLinear(aP)];
}
template <class Type,const int Dim> void  cDataTypedIm<Type,Dim>::VI_SetV(const cPtxd<int,Dim> & aP,const int & aV)  
{
    tPB::AssertInside(aP);
    mRawDataLin[tPB::IndexeLinear(aP)] = tNumTrait<Type>::Trunc(aV);
}

template <class Type,const int Dim> void  cDataTypedIm<Type,Dim>::VD_SetV(const cPtxd<int,Dim> & aP,const double & aV)  
{
    tPB::AssertInside(aP);
    MMVII_INTERNAL_ASSERT_tiny(tNumTrait<Type>::ValueOk(aV),"Bad Value in VD_SetV");
    mRawDataLin[tPB::IndexeLinear(aP)] = tNumTrait<Type>::RoundNearestToType(aV);
}

template<class Type,const int Dim> void  cDataTypedIm<Type,Dim>::ChSignIn(cDataTypedIm<Type,Dim> & aRes) const
{
   this->AssertSameArea(aRes);
   auto  aOut =  aRes.mRawDataLin;
   auto  aIn =   mRawDataLin;
   auto aNbElem = NbElem();

// msvc++ :  disable: warning C4146: unary minus operator applied to unsigned type, result still unsigned
#ifdef _WIN32
# pragma warning( push )
# pragma warning( disable : 4146 )
#endif

   for (int aX=0 ; aX<aNbElem ; aX++)
       aOut[aX] = -aIn[aX];

#ifdef _WIN32
# pragma warning( pop )
#endif
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
MACRO_INSTANTIATE_cDataTypedIm(tINT8)

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
   //     => do it only if verification is high;  else the error will not be detected and check cannot work 
   if (The_MMVII_DebugLevel >= The_MMVII_DebugLevel_InternalError_tiny)
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
