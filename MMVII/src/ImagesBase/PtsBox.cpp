#include "include/MMVII_all.h"

// #include <Eigen/Dense>

namespace MMVII
{

/* ========================== */
/*          ::                */
/* ========================== */

/// Computation to get the point we have at end of iterating a rectangle
template <const int Dim> cPtxd<int,Dim> CalPEnd(const cPtxd<int,Dim> & aP0,const cPtxd<int,Dim> & aP1)
{
    cPtxd<int,Dim> aRes = aP0;
    aRes[Dim-1] = aP1[Dim-1] ;
    return aRes;
}


/* ========================== */
/*   cBorderPixBoxIterator    */
/* ========================== */

template <const int Dim>  
  cBorderPixBoxIterator<Dim>::cBorderPixBoxIterator(tBPB & aBPB,const  tPt & aP0) :
      cPixBoxIterator<Dim>  (aBPB.PB(),aP0),
      mBPB                  (&aBPB)
{
}

template <const int Dim>  
  cBorderPixBoxIterator<Dim>  & cBorderPixBoxIterator<Dim>::operator ++(int)
{
   return ++(*this);
}


template <const int Dim>  
  cBorderPixBoxIterator<Dim>  & cBorderPixBoxIterator<Dim>::operator ++()
{
    //cPixBoxIterator<Dim>::operator ++ ();
    tPBI::operator ++ ();
 
    mBPB->IncrPt(tPBI::mPCur);
    return *this;
}



/* ========================== */
/*      cBorderPixBox         */
/* ========================== */


template <const int Dim>  
  cBorderPixBox<Dim>::cBorderPixBox(const tPB & aPB,const tPt & aSz) :
    mPB     (aPB),
    mSz     (aSz),
    mBoxInt (mPB.Dilate(-mSz)),
    mX0     (mBoxInt.P0().x()),
    mX1     (mBoxInt.P1().x()),
    mBegin  (*this,mPB.P0()),
    mEnd    (*this,CalPEnd(mPB.P0(),mPB.P1()))
{
}
template <const int Dim>  
  cBorderPixBox<Dim>::cBorderPixBox(const tPB & aPB,int aSz) :
      cBorderPixBox<Dim>(aPB,tPt::PCste(aSz))
{
}

template <const int Dim>  
  cBorderPixBox<Dim>::cBorderPixBox(const cBorderPixBox<Dim> & aBPB) :
    cBorderPixBox<Dim>(mPB,mSz)
{
}

template <const int Dim>  cPixBox<Dim> &   cBorderPixBox<Dim>::PB() {return mPB;}


template <const int Dim>  void cBorderPixBox<Dim>::IncrPt(tPt & aP)
{
   if  ((aP.x()==mX0) && (mBoxInt.Inside(aP)))
      aP.x() = mX1;
     
}




/* ========================== */
/*          cPtxd             */
/* ========================== */

int NbPixVign(const int & aVign){return 1+2*aVign;}

template <const int Dim> int NbPixVign(const cPtxd<int,Dim> & aVign)
{
   int aRes = NbPixVign(aVign[0]);
   for (int aD=1 ;aD<Dim ; aD++)
       aRes *= NbPixVign(aVign[aD]);
   return aRes;
}




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
/*
static std::string  TestErHandler;
void TestBenchRectObjError(const std::string & aType,const std::string &  aMes,const char * aFile,int aLine)
{
   TestErHandler = aMes;
}
*/

/* ========================== */
/*          cPixBox           */
/* ========================== */


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


template <const int Dim> cPixBox<Dim>  cPixBox<Dim>::BoxWindow(int aSz)
{
    return cPixBox<Dim>(-cPtxd<int,Dim>::PCste(aSz),cPtxd<int,Dim>::PCste(aSz+1));
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

template <const int Dim> int cPixBox<Dim>::Interiority(const int  aCoord,int aD) const
{
   return std::min(aCoord-tBox::mP0[aD],tBox::mP1[aD]-1-aCoord);
}

template <const int Dim> int cPixBox<Dim>::Interiority(const tPt &  aP,int aD) const
{
   return Interiority(aP[aD],aD);
}

template <const int Dim> int cPixBox<Dim>::Interiority(const tPt& aP  ) const
{
   int aRes = Interiority(aP,0);
   for (int aD=1 ; aD<Dim ; aD++)
       aRes = std::min(aRes,Interiority(aP,aD));

   return aRes;
}

template <const int Dim> int cPixBox<Dim>::WinInteriority(const tPt& aP,const tPt& aWin,int aD) const
{
   return Interiority(aP,aD)-aWin[aD];
}

template <const int Dim> cBorderPixBox<Dim>  cPixBox<Dim>::Border(int aSz) const
{
   return  cBorderPixBox<Dim>(*this,aSz);
}


/* ========================== */
/*       cParseBoxInOut       */
/* ========================== */


template <const int Dim> cParseBoxInOut<Dim>::cParseBoxInOut(const tBox & aBoxGlob,const tBox & aBoxIndexe) :
    mBoxGlob (aBoxGlob),
    mBoxIndex (aBoxIndexe)
{
}
        // static tThis  CreateFromMem(const tBox&, double AvalaibleMem);

template <const int Dim> cParseBoxInOut<Dim> cParseBoxInOut<Dim>::CreateFromSize(const tBox & aBox,const tPt & aSzTile)
{
   return  tThis(aBox,tBox(tPt::PCste(0),CByC2P(aBox.Sz(),aSzTile,DivSup<int>)));
}
template <const int Dim> cParseBoxInOut<Dim> cParseBoxInOut<Dim>::CreateFromSizeCste(const tBox & aBox,int aSz)
{
    return CreateFromSize(aBox,tPt::PCste(aSz));
}

template <const int Dim> cParseBoxInOut<Dim> cParseBoxInOut<Dim>::CreateFromSzMem(const tBox & aBox,double aSzMem)
{
    return CreateFromSizeCste(aBox,round_ni(pow(aSzMem,1/double(Dim))));
}


template <const int Dim>  const cPixBox<Dim> &  cParseBoxInOut<Dim>::BoxIndex() const
{
      return  mBoxIndex;
}

template <const int Dim>  cPtxd<int,Dim>  cParseBoxInOut<Dim>::Index2Glob(const tPt & anIndex) const
{
      return  CByC1P(mBoxGlob.FromNormaliseCoord(mBoxIndex.ToNormaliseCoord(anIndex)),round_ni);
}

template <const int Dim>  cPixBox<Dim> cParseBoxInOut<Dim>::BoxOut(const tPt & anIndex) const
{
      return  cPixBox<Dim> 
              (
                   Index2Glob(anIndex),
                   Index2Glob(anIndex+tPt::PCste(1))
              );
}

template <const int Dim>  cPixBox<Dim> cParseBoxInOut<Dim>::BoxIn(const tPt & anIndex,const tPt& aDil) const
{
   return mBoxGlob.Inter(BoxOut(anIndex).Dilate(aDil));
}

template <const int Dim>  cPixBox<Dim> cParseBoxInOut<Dim>::BoxIn(const tPt & anIndex,int aDil) const
{
   return mBoxGlob.Inter(BoxOut(anIndex).Dilate(tPt::PCste(aDil)));
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
          // make coherent but empty
          if (mSz[aK] <0)
          {
             mP1[aK] = mP0[aK];
             mSz[aK] = 0;
          }
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


template <class Type,const int Dim> cTplBox<Type,Dim>  cTplBox<Type,Dim>::Inter(const tBox & aBox)const
{
  return tBox(PtSupEq(mP0,aBox.mP0),PtInfEq(mP1,aBox.mP1),true);
}

template <class Type,const int Dim> cTplBox<Type,Dim>  cTplBox<Type,Dim>::Dilate(const tPt & aPt) const
{
   return tBox(mP0-aPt,mP1+aPt);
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
        aRes[aK] = mP0[aK] + tBaseNumTrait<Type>::RoundNearestToType(mSz[aK]*aPN[aK]);
    }
    return aRes;
    // return Proj(aRes);
}


template <class Type,const int Dim> cPtxd<double,Dim>  cTplBox<Type,Dim>::ToNormaliseCoord(const cPtxd<Type,Dim> & aP) const 
{
    cPtxd<double,Dim> aRes;
    for (int aK=0 ; aK<Dim ; aK++)
    {
        aRes[aK] = (aP[aK]-mP0[aK]) / double(mSz[aK]);
    }
    return aRes;
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
   return Proj(FromNormaliseCoord(RandomNormalised()));
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
        // int aI0 = round_down(aX0*mSz[aK]);
        // int aI1 = round_down(aX1*mSz[aK]);
        Type aI0 =  tBaseNumTrait<Type>::RoundNearestToType(aX0*mSz[aK]);
        Type aI1 =  tBaseNumTrait<Type>::RoundNearestToType(aX1*mSz[aK]);
        aI1 = std::min((mP1[aK]-1),std::max(aI1,aI0+1));
        aI0  = std::max((mP0[aK]),std::min(aI0,aI1-1));
        aP0[aK] = aI0;
        aP1[aK] = aI1;

    }
    return cTplBox<Type,Dim>(aP0,aP1);
}



#define MACRO_INSTATIATE_PRECT_DIM(DIM)\
template class cBorderPixBoxIterator<DIM>;\
template class cBorderPixBox<DIM>;\
template class cTplBox<tINT4,DIM>;\
template  class cParseBoxInOut<DIM>;\
template class cTplBox<tREAL8,DIM>;\
template class cPixBox<DIM>;\
template  int NbPixVign(const cPtxd<int,DIM> & aVign);\
template class cDataGenUnTypedIm<DIM>;\
template <> const cPixBox<DIM> cPixBox<DIM>::TheEmptyBox(cPtxd<int,DIM>::PCste(0),cPtxd<int,DIM>::PCste(0),true);



MACRO_INSTATIATE_PRECT_DIM(1)
MACRO_INSTATIATE_PRECT_DIM(2)
MACRO_INSTATIATE_PRECT_DIM(3)



};
