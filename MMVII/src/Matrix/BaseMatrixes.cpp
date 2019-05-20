#include "include/MMVII_all.h"

namespace MMVII
{


/* ========================== */
/*          cDenseVect        */
/* ========================== */

template <class Type> cDenseVect<Type>::cDenseVect(int aSz,eModeInitImage aModeInit) :
   mIm  (aSz,nullptr,aModeInit)
{
}

template <class Type> double cDenseVect<Type>::L1Dist(const cDenseVect<Type> & aV) const
{
   return mIm.DIm().L1Dist(aV.mIm.DIm());
}
template <class Type> double cDenseVect<Type>::L2Dist(const cDenseVect<Type> & aV) const
{
   return mIm.DIm().L2Dist(aV.mIm.DIm());
}
// double L2Dist(const cDenseVect<Type> & aV) const;

template <class Type> Type*       cDenseVect<Type>::RawData()       {return DIm().RawDataLin();}
template <class Type> const Type* cDenseVect<Type>::RawData() const {return DIm().RawDataLin();}

// const Type * RawData() const;



/* ========================== */
/*          cMatrix       */
/* ========================== */
cMatrix::cMatrix(int aX,int aY) :
   cRect2(cPt2di(0,0),cPt2di(aX,aY))
{
}

cMatrix::~cMatrix() 
{
}
         // ============  Mul Col ========================

template <class Type> static tMatrElem TplMulColElem(int aY,const cMatrix & aMat,const cDenseVect<Type> & aVIn)
{
    aMat.TplCheckSizeX(aVIn);

    tMatrElem aRes = 0.0;
    for (int aX=0 ; aX<aMat.Sz().x() ; aX++)
        aRes += aVIn(aX) * aMat.V_GetElem(aX,aY);

    return aRes;
}


template <class Type> static void TplMulCol(cDenseVect<Type> & aVOut,const cMatrix & aMat,const cDenseVect<Type> & aVIn) 
{
   aMat.TplCheckSizeYandX(aVOut,aVIn);

   if (&aVOut==&aVIn) // Will see later if we handle this case
   {
       MMVII_INTERNAL_ASSERT_strong(false,"Aliasing in TplMulCol")
   }

   for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
   {
       aVOut(aY) = TplMulColElem(aY,aMat,aVIn);
   }
}

template <class Type> static cDenseVect<Type> TplMulCol(const cMatrix & aMat,const cDenseVect<Type> & aVIn) 
{
    cDenseVect<Type> aRes(aMat.Sz().y());
    TplMulCol(aRes,aMat,aVIn);

    return aRes;
}

template <class Type> static void TplReadColInPlace(const cMatrix & aMat,int aX,cDenseVect<Type>& aV)
{
    aMat.TplCheckSizeY(aV);
    for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
        aV(aY) = aMat.V_GetElem(aX,aY);
}

template <class Type> static void TplWriteCol(cMatrix & aMat,int aX,const cDenseVect<Type>& aV)
{
    aMat.TplCheckSizeY(aV);
    for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
        aMat.V_SetElem(aX,aY,aV(aY)) ;
}





// virtual void WriteCol(int aY,const cDenseVect<tREAL4>&) ;


         // ============  Mul Line ========================

template <class Type> static tMatrElem TplMulLineElem(int aX,const cMatrix & aMat,const cDenseVect<Type> & aVIn)
{
    aMat.TplCheckSizeY(aVIn);
    tMatrElem aRes = 0.0;
    for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
        aRes += aVIn(aY) * aMat.V_GetElem(aX,aY);

    return aRes;
}


template <class Type> static void TplMulLine(cDenseVect<Type> & aVOut,const cMatrix & aMat,const cDenseVect<Type> & aVIn)  
{
   aMat.TplCheckSizeYandX(aVIn,aVOut);

   if (&aVOut==&aVIn) // Will see later if we handle this case
   {
       MMVII_INTERNAL_ASSERT_strong(false,"Aliasing in TplMulCol")
   }

   for (int aX=0 ; aX<aMat.Sz().x() ; aX++)
   {
       aVOut(aX) = TplMulLineElem(aX,aMat,aVIn);
   }
}

template <class Type> static cDenseVect<Type> TplMulLine(const cMatrix & aMat,const cDenseVect<Type> & aVIn) 
{
    cDenseVect<Type> aRes(aMat.Sz().x());
    TplMulLine(aRes,aMat,aVIn);

    return aRes;
}

template <class Type> static void TplReadLineInPlace(const cMatrix & aMat,int aY,cDenseVect<Type>& aV)
{
    aMat.TplCheckSizeX(aV);
    for (int aX=0 ; aX<aMat.Sz().x() ; aX++)
        aV(aX) = aMat.V_GetElem(aX,aY);
}

template <class Type> static void TplWriteLine(cMatrix & aMat,int aY,const cDenseVect<Type>& aV)
{
    aMat.TplCheckSizeX(aV);
    for (int aX=0 ; aX<aMat.Sz().x() ; aX++)
        aMat.V_SetElem(aX,aY,aV(aX)) ;
}



     // Virtuals tREAL4

void cMatrix::MulColInPlace(cDenseVect<tREAL4> & aOut,const cDenseVect<tREAL4> & aIn) const
{
    TplMulCol(aOut,*this,aIn);
}
cDenseVect<tREAL4> cMatrix::MulCol(const cDenseVect<tREAL4> & aIn) const
{
    return TplMulCol(*this,aIn);
}
tMatrElem cMatrix::MulColElem(int aY,const cDenseVect<tREAL4> & aIn) const
{
    return TplMulColElem(aY,*this,aIn);
}
void cMatrix::MulLineInPlace(cDenseVect<tREAL4> & aOut,const cDenseVect<tREAL4> & aIn) const
{
    TplMulLine(aOut,*this,aIn);
}
cDenseVect<tREAL4> cMatrix::MulLine(const cDenseVect<tREAL4> & aIn) const
{
    return TplMulLine(*this,aIn);
}
tMatrElem cMatrix::MulLineElem(int aX,const cDenseVect<tREAL4> & aIn) const
{
    return TplMulLineElem(aX,*this,aIn);
}

void cMatrix::ReadColInPlace(int aX,cDenseVect<tREAL4>& aV)  const {TplReadColInPlace(*this,aX,aV);}
void cMatrix::WriteCol(int aX,const cDenseVect<tREAL4>& aV)        {TplWriteCol(*this,aX,aV);}
void cMatrix::ReadLineInPlace(int aY,cDenseVect<tREAL4>& aV) const {TplReadLineInPlace(*this,aY,aV);}
void cMatrix::WriteLine(int aY,const cDenseVect<tREAL4>& aV)       {TplWriteLine(*this,aY,aV);}


     // Virtuals tREAL8
void cMatrix::MulColInPlace(cDenseVect<tREAL8> & aOut,const cDenseVect<tREAL8> & aIn) const
{
    TplMulCol(aOut,*this,aIn);
}
cDenseVect<tREAL8> cMatrix::MulCol(const cDenseVect<tREAL8> & aIn) const
{
    return TplMulCol(*this,aIn);
}
tMatrElem cMatrix::MulColElem( int aY,const cDenseVect<tREAL8> & aIn) const
{
    return TplMulColElem(aY,*this,aIn);
}
void cMatrix::MulLineInPlace(cDenseVect<tREAL8> & aOut,const cDenseVect<tREAL8> & aIn) const
{
    TplMulLine(aOut,*this,aIn);
}
cDenseVect<tREAL8> cMatrix::MulLine(const cDenseVect<tREAL8> & aIn) const
{
    return TplMulLine(*this,aIn);
}
tMatrElem cMatrix::MulLineElem(int aX,const cDenseVect<tREAL8> & aIn) const
{
    return TplMulLineElem(aX,*this,aIn);
}

void cMatrix::ReadColInPlace(int aX,cDenseVect<tREAL8>& aV)  const {TplReadColInPlace(*this,aX,aV);}
void cMatrix::WriteCol(int aX,const cDenseVect<tREAL8>& aV)        {TplWriteCol(*this,aX,aV);}
void cMatrix::ReadLineInPlace(int aY,cDenseVect<tREAL8>& aV) const {TplReadLineInPlace(*this,aY,aV);}
void cMatrix::WriteLine(int aY,const cDenseVect<tREAL8>& aV)       {TplWriteLine(*this,aY,aV);}

     // Virtuals tREAL16
void cMatrix::MulColInPlace(cDenseVect<tREAL16> & aOut,const cDenseVect<tREAL16> & aIn) const
{
    TplMulCol(aOut,*this,aIn);
}
cDenseVect<tREAL16> cMatrix::MulCol(const cDenseVect<tREAL16> & aIn) const
{
    return TplMulCol(*this,aIn);
}
tMatrElem cMatrix::MulColElem(int aY,const cDenseVect<tREAL16> & aIn) const
{
    return TplMulColElem(aY,*this,aIn);
}
void cMatrix::MulLineInPlace(cDenseVect<tREAL16> & aOut,const cDenseVect<tREAL16> & aIn) const
{
    TplMulLine(aOut,*this,aIn);
}
cDenseVect<tREAL16> cMatrix::MulLine(const cDenseVect<tREAL16> & aIn) const
{
    return TplMulLine(*this,aIn);
}
tMatrElem cMatrix::MulLineElem(int aX,const cDenseVect<tREAL16> & aIn) const
{
    return TplMulLineElem(aX,*this,aIn);
}

void cMatrix::ReadColInPlace(int aX,cDenseVect<tREAL16>& aV)  const {TplReadColInPlace(*this,aX,aV);}
void cMatrix::WriteCol(int aX,const cDenseVect<tREAL16>& aV)        {TplWriteCol(*this,aX,aV);}
void cMatrix::ReadLineInPlace(int aY,cDenseVect<tREAL16>& aV) const {TplReadLineInPlace(*this,aY,aV);}
void cMatrix::WriteLine(int aY,const cDenseVect<tREAL16>& aV)       {TplWriteLine(*this,aY,aV);}

     //    ===   MulMat ====


void cMatrix::MatMulInPlace(const cMatrix & aM1,const cMatrix & aM2)
{
   CheckSizeMulInPlace(aM1,aM2);
   cDenseVect<tREAL16> aLine(Sz().x());

   for (int aY= 0 ; aY< Sz().y() ; aY++)
   {
       for (int aX= 0 ; aX< Sz().x() ; aX++)
       {
           tREAL16 aVal = 0.0;
           for (int aK=0 ; aK<aM1.Sz().x() ; aK++)
              aVal += aM1.V_GetElem(aK,aY) * aM2.V_GetElem(aX,aK);
           V_SetElem(aX,aY,aVal);
       }
   }
}

template <class Type> cDenseVect<Type> operator * (const cDenseVect<Type> & aLine,const cMatrix& aMat)
{
   return aMat.MulLine(aLine);
}

template <class Type> cDenseVect<Type> operator * (const cMatrix& aMat,const cDenseVect<Type> & aCol)
{
   return aMat.MulCol(aCol);
}



/* ===================================================== */
/* ===================================================== */
/* ===================================================== */

#define INSTANTIATE_BASE_MATRICES(Type)\
template  class  cDenseVect<Type>;\
template  cDenseVect<Type> operator * (const cDenseVect<Type> & aLine,const cMatrix& aMat);\
template  cDenseVect<Type> operator * (const cMatrix& aMat,const cDenseVect<Type> & aCol);\



INSTANTIATE_BASE_MATRICES(tREAL4)
INSTANTIATE_BASE_MATRICES(tREAL8)
INSTANTIATE_BASE_MATRICES(tREAL16)

};


/* ========================== */
/*          cMatrix           */
/* ========================== */

/*
class cMatrix
{
    public :
         typedef cMatrix tDM;

         tDM & Mat()  {return *(mPtr);}
         const cMatrix & Mat() const  {return *(mPtr);}
    protected :
         cMatrix(tDM *);
    private :

         std::shared_ptr<tDM>  mSPtr;  ///< shared pointer to real image
         tDM *                 mPtr;

};

cMatrix::cMatrix(tDM * aPtr) :
   mSPtr  (aPtr),
   mPtr   (mSPtr.get())
{
}
*/
