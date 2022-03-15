#include "include/MMVII_all.h"

// #include <Eigen/Dense>

namespace MMVII
{


/// Sigma of convol sqrt(A^2+B^2)
double SomSigm(double  aS1,double aS2) {return sqrt(Square(aS1)+Square(aS2));}
/// "Inverse" of SomSigm  sqrt(A^2-B^2)
double DifSigm(double  aS1,double aS2) 
{
   MMVII_INTERNAL_ASSERT_tiny(aS1>=aS2,"DifSigm");
   return sqrt(Square(aS1)-Square(aS2));
}




/** We use a class essentially to be able to use typedef, most method will be static */
template <class Type> class cLinearFilter
{
    public :
      // in comments, note @ for convolution
      typedef Type                                  tVal;
      typedef typename tNumTrait<Type>::tBase       tBase;
      typedef typename tNumTrait<Type>::tFloatAssoc tFl;

      static void FilterExp
                  (
                      bool Normalize,  // If true, limit size effect and Cste => Cste (else lowe close to border)
                      cDataIm2D<Type> &,
                      int aNbIter,
                      const cRect2 &,
                      const tFl &aFx,
                      const  tFl &aFy
                  );
};


/**
     Convolution of aLineIm in place by expononential filter "F^|x|", use a buffer
*/
template <class TBuf,class TIm,class TFact,class TyNorm> void  
   OneLineFilterExp
   (
       int aNbIter,
       TBuf *aDBuf,
       TIm* aLineIm,
       TFact aFact,
       int anX0,
       int anX1,
       TyNorm * aDataNorm 
   )
{
   for (int aKIter=0 ; aKIter<aNbIter ; aKIter++)
   {
      aDBuf[anX0] = 0;
      // left to right ,at end  aDBuf  contains "I  @ (Fx^|x| * x<0)"
      for (int anX = anX0+1; anX<anX1 ; anX++)
      {
          aDBuf[anX] =  aFact *(aDBuf[anX-1] + aLineIm[anX-1]);
      }
      // right to left, full convolution
      for (int anX = anX1-2; anX>=anX0 ; anX--)
      {
          aLineIm[anX]  +=(TIm)(aFact*aLineIm[anX+1] );  // now Line contains "I @  (Fx^|x| * x>=0)
          aLineIm[anX+1]+=(TIm)(aDBuf[anX+1] ); // now contains   I  @ Fx^|x|
      }
   }
   if (aDataNorm)
   {
      for (int anX= anX0 ; anX<anX1 ; anX++)
         aLineIm[anX] /= aDataNorm[anX];
   }
}

template <class Type> Type * ImNormal(int aNbIter,cDataIm1D<Type> & aRes,Type aFact,int aX0,int aX1)
{
   aRes.Resize(cPt1di(aX0),cPt1di(aX1));
   aRes.InitCste(1.0);

   cIm1D<Type>  aBuf(aX0,aX1);
   OneLineFilterExp(aNbIter,aBuf.DIm().ExtractRawData1D(),aRes.ExtractRawData1D(),aFact,aX0,aX1,(float *)nullptr);
 
   return aRes.ExtractRawData1D();
}

template <class Type> void  cLinearFilter<Type>::FilterExp
                            (
                                 bool Normalise,
                                 cDataIm2D<Type> & aIm,
                                 int   aNbIter,
                                 const cRect2 & aRect,
                                 const tFl &aFx,
                                 const  tFl &aFy
                            )
{
   MMVII_INTERNAL_ASSERT_strong(aRect.IncludedIn(aIm),"cLinearFilter Rect is outside");

   // Local copy to have quick access
   Type  **  aDIm = aIm.ExtractRawData2D();
   int anX0 = aRect.P0().x();
   int anY0 = aRect.P0().y();
   int anX1 = aRect.P1().x();
   int anY1 = aRect.P1().y();

   // Create a buf  to avoid sides effect
   // int aNbBuf = Norm1(aRect.Sz());
   cIm1D<tBase> aImBuf(Norm1(aRect.Sz()));
   cIm1D<tVal> aBufDupCol(aRect.Sz().y());
   
   cIm1D<tFl>  aImNormal(Norm1(aRect.Sz()));
   // tFl * aBufNorm = nullptr;

       //  Filter the lines , I @  Fx^|x| 
   if (aFx != 0)
   {
      tBase * aDBuf = aImBuf.DIm().RawDataLin()-anX0;
      tFl * aDataNorm = Normalise ? ImNormal(aNbIter,aImNormal.DIm(),aFx,anX0,anX1) : nullptr;
      
      for (int anY=anY0 ; anY<anY1 ; anY++)
      {
           OneLineFilterExp(aNbIter,aDBuf,aDIm[anY],aFx,anX0,anX1,aDataNorm);
      }
   }
       //  Filter the Column , I @  Fy^|y| 
   if (aFy != 0)
   {
       tBase * aDBuf = aImBuf.DIm().RawDataLin()-anY0;
       tVal  * aDupCol = aBufDupCol.DIm().RawDataLin()-anY0;
       tFl * aDataNorm = Normalise ? ImNormal(aNbIter,aImNormal.DIm(),aFy,anY0,anY1) : nullptr;
       for (int anX=anX0 ; anX<anX1 ; anX++)
       {
          // Transferate Column X in line buf aDupCol
          for (int anY=anY0 ; anY<anY1 ; anY++)
          {
             aDupCol[anY] = aDIm[anY][anX];
          }
          // Filter dup col
          OneLineFilterExp(aNbIter,aDBuf,aDupCol,aFy,anY0,anY1,aDataNorm);
          // Inverse transfer
          for (int anY=anY0 ; anY<anY1 ; anY++)
          {
             aDIm[anY][anX] = aDupCol[anY] ;
          }
       }
   }
}

template <class Type>
void  ExponentialFilter(bool Normalise,cDataIm2D<Type> & aDIm,int  aNbIt,const cRect2 & aR2,double Fx,double Fy)
{
   cLinearFilter<Type>::FilterExp(Normalise,aDIm,aNbIt,aR2,Fx,Fy);
}

template <class Type> 
void  ExponentialFilter(cDataIm2D<Type> & aIm,int   aNbIter,double aFact)
{
    ExponentialFilter(true,aIm,aNbIter,aIm,aFact,aFact);
}

template <class Type> 
void  ExpFilterOfStdDev(cDataIm2D<Type> & aIm,int   aNbIter,double aStdDev)
{
     ExponentialFilter(aIm,aNbIter,FactExpFromSigma2(Square(aStdDev)/aNbIter));
}

template <class Type> 
void  ExpFilterOfStdDev(cDataIm2D<Type> & aImOut,const cDataIm2D<Type> & aImIn,int   aNbIter,double aStdDev)
{
     aImIn.DupIn(aImOut);
     ExponentialFilter(aImOut,aNbIter,FactExpFromSigma2(Square(aStdDev)/aNbIter));
}

/* ========================== */
/*       cImGrad              */
/* ========================== */


template <class Type> cImGrad<Type>::cImGrad(const cIm2D<Type> & aGx,const cIm2D<Type> &  aGy) :
    mGx (aGx),
    mGy (aGy)
{
}
template <class Type> cImGrad<Type>::   cImGrad(const cPt2di & aSz) :
     cImGrad<Type>(cIm2D<Type>(aSz),cIm2D<Type>(aSz))
{
}

template <class Type> cImGrad<Type>::cImGrad(const cIm2D<Type> & aImIn) : 
   cImGrad<Type>(aImIn.DIm().Sz()) 
{
}


/* ========================== */
/*     cDataGenUnTypedIm      */
/* ========================== */


#define MACRO_INSTANTIATE_ExpoFilter(Type)\
template  class cImGrad<Type>;\
template  class cLinearFilter<Type>;\
template void ExponentialFilter(bool,cDataIm2D<Type> &,int,const cRect2 &,double,double);\
template void  ExponentialFilter(cDataIm2D<Type> & aIm,int   aNbIter,double aFact);\
template void  ExpFilterOfStdDev(cDataIm2D<Type> & aIm,int   aNbIter,double aStdDev);\
template void  ExpFilterOfStdDev(cDataIm2D<Type> & aIm,const cDataIm2D<Type> & aImIn,int   aNbIter,double aStdDev);\

MACRO_INSTANTIATE_ExpoFilter(tREAL4);
MACRO_INSTANTIATE_ExpoFilter(tREAL8);
MACRO_INSTANTIATE_ExpoFilter(tREAL16);
MACRO_INSTANTIATE_ExpoFilter(tINT4);
MACRO_INSTANTIATE_ExpoFilter(tINT2);
MACRO_INSTANTIATE_ExpoFilter(tINT1);

/*
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
*/



};
