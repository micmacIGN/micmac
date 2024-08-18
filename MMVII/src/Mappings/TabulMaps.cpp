// #include "MMVII_PCSens.h"
// #include "MMVII_ImageInfoExtract.h"
// #include "MMVII_ExtractLines.h"
#include "MMVII_Mappings.h"
#include "MMVII_Interpolators.h"



namespace MMVII
{

/* **************************************************** */
/*                                                      */
/*              cTabulatMap2D_Id                        */
/*                                                      */
/* **************************************************** */


template <class Type>
   cTabulatMap2D_Id<Type>::cTabulatMap2D_Id(tIm anImX,tIm anImY,cDiffInterpolator1D * anInt) :
        mImX        (anImX),
        mDImX       (&mImX.DIm()),
        mImY        (anImY),
        mDImY       (&mImY.DIm()),
        mInt        (anInt),
        mNbIterInv  (10),
        mEpsInv     (1e-3)
{
        mDImX->AssertSameArea(*mDImY);
}

template <class Type>  cPt2dr   cTabulatMap2D_Id<Type>::Value(const cPt2dr & aPt) const
{
     cPt2dr aDelta;
     if (mDImX->InsideInterpolator(*mInt,aPt))
     {
         aDelta.x() =  mDImX->GetValueInterpol(*mInt,aPt) ;
         aDelta.y() =  mDImY->GetValueInterpol(*mInt,aPt) ;
     }
     else if (mDImX->InsideBL(aPt))
     {
         aDelta.x() =  mDImX->GetVBL(aPt) ;
         aDelta.y() =  mDImY->GetVBL(aPt) ;
     }
     else
     {
        cPt2di  aPP = mDImX->Proj(ToI(aPt));
        aDelta.x() =  mDImX->GetV(aPP) ;
        aDelta.y() =  mDImY->GetV(aPP) ;
     }

     return aPt +  aDelta;
}

template <class Type>  cPt2dr   cTabulatMap2D_Id<Type>::Inverse(const cPt2dr & aPt) const
{
        return InvertQuasiTrans(aPt,aPt,mEpsInv,mNbIterInv);
}
template <class Type>  tREAL8   cTabulatMap2D_Id<Type>::EpsInv() const {return mEpsInv; }


template class cTabulatMap2D_Id<tREAL4>;



/* **************************************************** */
/*                                                      */
/*              cTabulMap                               */
/*                                                      */
/* **************************************************** */

template <const int DimIn,const int DimOut> typename cTabulMap<DimIn,DimOut>::tPtIn 
     cTabulMap<DimIn,DimOut>::Pix2In(const tPtIn & aPt) const {return mP0In + MulCByC(aPt,mMulPix2In);}

template <const int DimIn,const int DimOut> typename cTabulMap<DimIn,DimOut>::tPtIn 
     cTabulMap<DimIn,DimOut>::Pix2In(const tPix & aPt) const {return Pix2In(ToR(aPt));}

template <const int DimIn,const int DimOut> typename cTabulMap<DimIn,DimOut>::tPtIn 
     cTabulMap<DimIn,DimOut>::In2Pix(const tPtIn & aPt) const {return MulCByC(aPt-mP0In,mMulIn2Pix);}

template <const int DimIn,const int DimOut> 
      cTabulMap<DimIn,DimOut>::cTabulMap(const tMap & aMap,const tBoxIn & aBoxIn,const tPix & aSz) :
          mP0In            (aBoxIn.P0()),
          mMulPix2In       (DivCByC(aBoxIn.Sz(),ToR(aSz))),
          mMulIn2Pix       (DivCByC(ToR(aSz),aBoxIn.Sz())),
          mBoxOutTabuled   (aMap.Value(aBoxIn.P0()),aMap.Value(aBoxIn.P1()))
{
   // Allocate the images 
   for (int aD=0 ;  aD<DimOut ; aD++)
        mVIms.push_back(tDIm::AllocIm(aSz+tPix::PCste(1)));

   //  fill the grids with values of mapping
   for (const auto & aPix : *(mVIms.at(0))  )
   {
       tPtIn aPixIn = Pix2In(aPix);  //  Pix ->  Init space
       tPtOut aPixOut = aMap.Value(aPixIn);  // Value of mapping in init space
       for (int aD=0 ; aD<DimOut ; aD++)
          mVIms.at(aD)->VD_SetV(aPix,aPixOut[aD]);  // Put value at grids for each dim
   }

   // compute the bounding box, add one pixel of margins
   {
      cTplBoxOfPts<tREAL8,DimOut> aRes;  // Box of Pts to compute

      cPixBox<DimIn> aBoxPix(*(mVIms.at(0)));   // Box of pixel
      cBorderPixBox<DimIn> aBorder( aBoxPix.Dilate(1),1);  // iterator on border + 1 Pixel margin

      for (const auto & aPix : aBorder)
      {
          aRes.Add(aMap.Value(Pix2In(aPix)));  // update box of pixel
      }
      mBoxOutTabuled = aRes.CurBox(); // memorize result
   }
}

template <const int DimIn,const int DimOut> 
   const typename cTabulMap<DimIn,DimOut>::tBoxOut&   cTabulMap<DimIn,DimOut>::BoxOutTabuled() const
{
   return mBoxOutTabuled;
}

template <const int DimIn,const int DimOut> 
      cTabulMap<DimIn,DimOut>::~cTabulMap()
{
   DeleteAllAndClear(mVIms);
}

template <const int DimIn,const int DimOut> 
    typename  cTabulMap<DimIn,DimOut>::tPtOut  cTabulMap<DimIn,DimOut>::Value(const tPtIn & aPtIn) const 
{
    tPtOut aRes;

    tPtIn aPix = In2Pix(aPtIn);

    for (int aD=0 ; aD<DimOut; aD++)
        aRes[aD] =  mVIms.at(aD)->GetVBL(aPix);

   return aRes;
}

template <const int DimIn,const int DimOut> bool cTabulMap<DimIn,DimOut>::OkValue(const tPtIn & aPt) const
{
      return mVIms.at(0)->InsideBL(In2Pix(aPt));
}


/* **************************************************** */
/*                                                      */
/*              cTabuMapInv                             */
/*                                                      */
/* **************************************************** */
cTabuMapInv<2>* AllocTabulDUD(cPerspCamIntrCalib * aCalib,int aNb);

template <const int Dim> typename cTabuMapInv<Dim>::tPt  cTabuMapInv<Dim>::Value  (const tPt & aPt) const
{
   return mTabulMapDir->Value(aPt);
}

template <const int Dim> typename cTabuMapInv<Dim>::tPt  cTabuMapInv<Dim>::Inverse  (const tPt & aPt) const
{
   return mTabulMapInv->Value(aPt);
}


template <const int Dim> cTabuMapInv<Dim>::cTabuMapInv(const tMap & aMap,const cTplBox<tREAL8,Dim> & aBox,const tPix & aSz):
     mTabulMapDir (new tTabuMap(aMap,aBox,aSz)),
     mTabulMapInv (nullptr)
{
    cDataInvertOfMapping<tREAL8,Dim> aMapInv(&aMap,false); // false=no adopt
    mTabulMapInv = new tTabuMap(aMapInv,mTabulMapDir->BoxOutTabuled(),aSz+tPix::PCste(2));
}

template <const int Dim> cTabuMapInv<Dim>::~cTabuMapInv() 
{
    delete mTabulMapDir;
    delete mTabulMapInv;
}

template <const int Dim> bool cTabuMapInv<Dim>::OkDirect(const tPt & aPt) const {return mTabulMapDir->OkValue(aPt);}
template <const int Dim> bool cTabuMapInv<Dim>::OkInverse(const tPt & aPt) const {return mTabulMapInv->OkValue(aPt);}

   // ================ INSTANCIATION ===========================

template  class cTabulMap<1,1>;
template  class cTabulMap<2,2>;
template  class cTabulMap<2,3>;
template  class cTabulMap<3,3>;

template  class cTabuMapInv<2>;

};
