
#include "include/MMVII_2Include_Serial_Tpl.h"
// #include "include/MMVII_Tpl_Images.h"

namespace MMVII
{

/* ========================== */
/*     cDataGenDimTypedIm     */
/* ========================== */

template <class Type> cDataGenDimTypedIm<Type>::cDataGenDimTypedIm() :
    mDim        (1),
    mNbElem     (1),
    mSz         (mDim,eModeInitImage::eMIA_V1),
    mMulSz      (mDim,eModeInitImage::eMIA_V1),
    mRawDataLin (nullptr)
{
}

template <class Type> void cDataGenDimTypedIm<Type>::Resize(const tIndex& aSz) 
{
   delete mRawDataLin;
   mDim    =  aSz.Sz();
   mSz     =  tIndex(mDim);
   mMulSz  =  tIndex(mDim);

   mNbElem = 1;
   for (int aDim=0 ; aDim<mDim ; aDim++)
   {
       int aVal = aSz(aDim);
       MMVII_INTERNAL_ASSERT_tiny(aVal>0,"Size not positive in cDataGenDimTypedIm");
       mSz(aDim) = aVal;
       mMulSz(aDim) = mNbElem;
       mNbElem *= aVal;
   }
   mRawDataLin = new Type [mNbElem];
   for (int aK=0 ; aK<mNbElem ; aK++)
       mRawDataLin[aK] = 0;
}

template <class Type> cDataGenDimTypedIm<Type>::cDataGenDimTypedIm(const tIndex& aSz) :
   cDataGenDimTypedIm<Type>()
{
   Resize(aSz);
}

template <class Type> cDataGenDimTypedIm<Type>::~cDataGenDimTypedIm()
{
   delete[] mRawDataLin;
}

template <class Type> void cDataGenDimTypedIm<Type>::AddData(const cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("Sz",anAux),mSz);
    if (anAux.Input())
       Resize(mSz.Dup()); // Dup : else size effet in re-init

   TplAddRawData(anAux,mRawDataLin,mNbElem);
}

/*
template <class Type> cDataGenDimTypedIm<Type>::cDataGenDimTypedIm(const tIndex& aSz) :
   mDim    (aSz.Sz()),
   mSz     (mDim),
   mMulSz  (mDim)
{
    mNbElem = 1;
    for (int aDim=0 ; aDim<mDim ; aDim++)
    {
        int aVal = aSz(aDim);
        MMVII_INTERNAL_ASSERT_tiny(aVal>0,"Size not positive in cDataGenDimTypedIm");
        mSz(aDim) = aVal;
        mMulSz(aDim) = mNbElem;
        mNbElem *= aVal;
    }
    mRawDataLin = new Type [mNbElem];
}
*/

template <class Type> int cDataGenDimTypedIm<Type>::NbElem() const {return mNbElem;}
template <class Type> Type * cDataGenDimTypedIm<Type>::RawDataLin() const {return mRawDataLin;}


template <class Type> const Type & cDataGenDimTypedIm<Type>::GetV(const tIndex& anIndex) const
{
   return mRawDataLin[Adress(anIndex)];
}

template <class Type> void cDataGenDimTypedIm<Type>::SetV(const tIndex& anIndex,const tBase &aVal) 
{
   MMVII_INTERNAL_ASSERT_tiny(tNumTrait<Type>::ValueOk(aVal),"Bad val in cDataGenDimTypedIm");
   mRawDataLin[Adress(anIndex)] = aVal;
}

template <class Type> void cDataGenDimTypedIm<Type>::AddV(const tIndex& anIndex,const tBase &anIncr) 
{
   Type & aDest = mRawDataLin[Adress(anIndex)];
   tBase  aNewV = aDest + anIncr;
   MMVII_INTERNAL_ASSERT_tiny(tNumTrait<Type>::ValueOk(aNewV),"Bad val in cDataGenDimTypedIm");
   mRawDataLin[Adress(anIndex)] = aNewV;
}


template <class Type> const cDenseVect<int> & cDataGenDimTypedIm<Type>::Sz() const
{
    return mSz;
}


template <class Type> int cDataGenDimTypedIm<Type>::Adress(const tIndex& anIndex) const
{
   AssertOk(anIndex);
   int aAdr = 0;
   for (int aDim=0 ; aDim<mDim ; aDim++)
   {
       aAdr += mMulSz(aDim) * anIndex(aDim);
   }

   return aAdr;
}


template <class Type> void cDataGenDimTypedIm<Type>::RecAddNLinearVal
    (const tRIndex& aRIndex,const double & aVal,tIndex& aIIndex,int aDim) 
{
   tREAL4 aCoord = aRIndex(aDim);
   int    aICoord = round_down(aCoord);
   tREAL4 aP1 =  (aCoord-aICoord);
   tREAL4 aP0 =  1-aP1;

   tREAL4 aV0 =  aP0 * aVal;
   tREAL4 aV1 =  aP1 * aVal;
   bool isEnd = (aDim==mDim-1);

   aIIndex(aDim) = aICoord;
   if (isEnd)
      AddV(aIIndex,aV0);
   else
      RecAddNLinearVal(aRIndex,aV0,aIIndex,aDim+1);


   if (aP1>0) 
   {
      aIIndex(aDim) = aICoord+1;
      if (isEnd)
         AddV(aIIndex,aV1);
      else
         RecAddNLinearVal(aRIndex,aV1,aIIndex,aDim+1);
   }

}


template <class Type>  tREAL4   cDataGenDimTypedIm<Type>::RecGetNLinearVal(const tRIndex& aRIndex,tIndex& aIIndex,int aDim) const
{
   tREAL4 aCoord = aRIndex(aDim);
   int    aICoord = round_down(aCoord);
   tREAL4 aP1 =  (aCoord-aICoord);
   tREAL4 aP0 =  1-aP1;

   aIIndex(aDim) = aICoord;
   bool isEnd = (aDim==mDim-1);

   tREAL4 aRes =  aP0 * (isEnd ? GetV(aIIndex)  : RecGetNLinearVal(aRIndex,aIIndex,aDim+1));

   if (aP1>0) 
   {
      aIIndex(aDim) = aICoord+1;
      aRes +=  aP1 * (isEnd ? GetV(aIIndex)  : RecGetNLinearVal(aRIndex,aIIndex,aDim+1));
   }

   return aRes;
}

template <class Type>  tREAL4   cDataGenDimTypedIm<Type>::GetNLinearVal(const tRIndex& aRI) const
{
    tIndex aII(mDim);
    return RecGetNLinearVal(aRI,aII,0);
}

template <class Type>  void   cDataGenDimTypedIm<Type>::AddNLinearVal(const tRIndex& aRI,const double & aVal) 
{
    tIndex aII(mDim);
    RecAddNLinearVal(aRI,aVal,aII,0);
}




template <class Type> void cDataGenDimTypedIm<Type>::PrivateAssertOk(const tIndex& anIndex) const
{
    MMVII_INTERNAL_ASSERT_always(anIndex.Sz()==mDim,"Bad dim for cDataGenDimTypedIm");
    for (int aDim=0 ; aDim<mDim ; aDim++)
    {
        int aVal = anIndex(aDim);
        MMVII_INTERNAL_ASSERT_always((aVal>=0) && (aVal<mSz(aDim)),"Out pts in cDataGenDimTypedIm");
    }
}

template <class Type> cIm2D<Type>  cDataGenDimTypedIm<Type>::ToIm2D() const
{
    MMVII_INTERNAL_ASSERT_always(mDim==2,"Bad dim for cDataGenDimTypedIm");
    return cIm2D<Type>(cPt2di::FromVect(mSz),mRawDataLin);
}

/* ========================== */
/*     cBenchImNDim<Type>     */
/* ========================== */

template <class Type>  class cBenchImNDim
{
   public :
      cBenchImNDim();
      ~cBenchImNDim();

      typedef cDenseVect<int> tIndex;
      typedef cDataGenDimTypedIm<Type> tIm;

      tIndex mSz;
      tIm    mIm;
      int    mDim;
      Type   Func(const tIndex & anIndex) const;

      void ExploreRec(tIndex &,int aDim,int aMode);
      void FinaleRec(tIndex &,int aMode);
      void AssertAllValEq(const Type &);
};

template <class Type>  Type cBenchImNDim<Type>::Func(const tIndex & anIndex) const
{
    Type aRes = 0.0;
    for (int aD=0 ; aD<mDim ; aD++)
    {
       aRes += anIndex(aD) /(1.0 + atan(aD *aD * anIndex(aD)));

    }
    return aRes;
}

template <class Type>  void cBenchImNDim<Type>::AssertAllValEq(const Type & aVal)
{
   for (int aK=0 ; aK<mIm.NbElem(); aK++)
   {
       MMVII_INTERNAL_ASSERT_bench(mIm.RawDataLin()[aK]==aVal,"BENCH AssertAllValEq");
   }
}


template <class Type>  void cBenchImNDim<Type>::FinaleRec(tIndex & anIndex,int aMode)
{
   if (aMode==0)
   {
       mIm.SetV(anIndex,0);
   }
   else if (aMode==1)
   {
       Type aV = mIm.GetV(anIndex);
       MMVII_INTERNAL_ASSERT_bench(aV==0,"BENCH cBenchImNDim,V!=0");
       mIm.SetV(anIndex,1);
   }
   else if (aMode==2)
   {
       Type aV = Func(anIndex);
       mIm.SetV(anIndex,aV);
   }
   else if (aMode==3)
   {
       Type aV1 = Func(anIndex);
       Type aV2 = mIm.GetV(anIndex);
       // StdOut() << " vvvvvVVv=" << aV1-aV2 << " " << aV1 << " " << aV2 << "\n";
       MMVII_INTERNAL_ASSERT_bench(std::abs(aV1-aV2)<1e-5,"Final Rec");
   }
}

template <class Type>  void cBenchImNDim<Type>::ExploreRec(tIndex & anIndex,int aDim,int aMode)
{
   if (aDim==mDim)
   {
       FinaleRec(anIndex,aMode);
       return;
   }
   for (int aK=0 ; aK<mSz(aDim) ; aK++)
   {
      anIndex(aDim) = aK;
      ExploreRec(anIndex,aDim+1,aMode);
   }
}


template <class Type>  
   cBenchImNDim<Type>::cBenchImNDim():
      mSz (1,eModeInitImage::eMIA_V1),
      mIm ()
{
    int aNbElem =   1+  RandUnif_N(10000);

    std::vector<int> aVCoord;
    while (aNbElem>0)
    {
        int aCoord = std::min(aNbElem+1,1 + round_ni(RandUnif_N(10)));
        aVCoord.push_back(aCoord);
        aNbElem /=  aCoord;
    }
    // Favorize 2D vect because we can check with 2D Image
    cPt2di aSz2= cPt2di(10+RandUnif_N(50),10+RandUnif_N(50));
    if (RandUnif_0_1() < 0.3)
    {
       aVCoord = std::vector<int>({aSz2.x(),aSz2.y()});
    }
    mDim = aVCoord.size();
    // it may happen that Dim was 2 before forcing, so must assure cohereence
    if (mDim==2)
    {
        // min value for interpol bilin
        for (auto & aV : aVCoord)
            aV = std::max(3,aV);
        // coherence
        aSz2=cPt2di::FromStdVector(aVCoord);
    }
    mSz =  tIndex(mDim);


    for (int aDim=0 ; aDim <mDim ; aDim++)
    {
         mSz(aDim) = aVCoord.at(aDim);
    }

    mIm.Resize(mSz);

   // Mode 0 set 0
    tIndex anIndex(mDim);
    ExploreRec(anIndex,0,0);
    AssertAllValEq(0);

   // Mode 1 check is 0 and set 1
    ExploreRec(anIndex,0,1);
    AssertAllValEq(1);

   // Mode 2 set Func
    ExploreRec(anIndex,0,2);
   // Mode 3 check is  Func
    ExploreRec(anIndex,0,3);

    if (mDim==2)
    {
        cIm2D<Type> aIm2(aSz2);
        cDataIm2D<Type> & aDIm2 = aIm2.DIm();
        for (const auto & aP : aDIm2)
        {
           Type aVal =  RandUnif_C();
           aDIm2.SetV(aP,aVal);
           mIm.SetV(aP.ToVect(),aVal);
        }
        for (int aK=0 ; aK< 100 ; aK++)
        {
             double aX = std::min(aSz2.x() * RandUnif_0_1(),aSz2.x()-1.01);
             double aY = std::min(aSz2.y() * RandUnif_0_1(),aSz2.y()-1.01);
             cPt2df aP2R(aX,aY);
             double aV1 = aDIm2.GetVBL(ToR(aP2R));
             double aV2 = mIm.GetNLinearVal(aP2R.ToVect());
             bool Ok0 = true;
             if ((RelativeDifference(aV1,aV2,&Ok0)>=1e-3)  && (std::abs(aV1-aV2)>1e-5))
             {
                   StdOut() << "ddddd " << aV1 << " " << aV2 << Ok0 << " " << RelativeDifference(aV1,aV2,&Ok0) << "\n";
                   MMVII_INTERNAL_ASSERT_bench(false,"Interpol Im N DIM");
             }

             double aVal = RandUnif_0_1();
             aDIm2.AddVBL(ToR(aP2R),aVal);
             mIm.AddNLinearVal(aP2R.ToVect(),aVal);
        }
        for (const auto & aP : aDIm2)
        {
           Type aV1= aDIm2.GetV(aP);
           Type aV2= mIm.GetV(aP.ToVect());
           MMVII_INTERNAL_ASSERT_bench(std::abs(aV1-aV2)<1e-3,"Interpol Im N DIM");
        }
    }
// template <class Type>  void   cDataGenDimTypedIm<Type>::AddNLinearVal(const tRIndex& aRI,const double & aVal) 
}

template <class Type>  
   cBenchImNDim<Type>::~cBenchImNDim()
{
}

void BenchImNDim()
{
   for (int aK=0 ; aK <100 ; aK++)
   {
       cBenchImNDim<tREAL4> aBR4;
       cBenchImNDim<tREAL8> aBR8;
   }
}



#define INSTANTIATE_GEN_DIM_TYPEDIM(TYPE)\
template  class cDataGenDimTypedIm<TYPE>;

INSTANTIATE_GEN_DIM_TYPEDIM(tINT4)
INSTANTIATE_GEN_DIM_TYPEDIM(tREAL4)
INSTANTIATE_GEN_DIM_TYPEDIM(tREAL8)
/*
INSTANTIATE_GEN_DIM_TYPEDIM(tINT2)
INSTANTIATE_GEN_DIM_TYPEDIM(tINT4)

INSTANTIATE_GEN_DIM_TYPEDIM(tU_INT1)
INSTANTIATE_GEN_DIM_TYPEDIM(tU_INT2)
INSTANTIATE_GEN_DIM_TYPEDIM(tU_INT4)


INSTANTIATE_GEN_DIM_TYPEDIM(tREAL4)
INSTANTIATE_GEN_DIM_TYPEDIM(tREAL8)
INSTANTIATE_GEN_DIM_TYPEDIM(tREAL16)
*/


};
