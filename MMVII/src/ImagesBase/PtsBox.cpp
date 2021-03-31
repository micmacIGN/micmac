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

// static const int VeryBig = 1e9;
// const cPt2di  ThePSupImage( VeryBig, VeryBig);
// const cPt2di  ThePInfImage(-VeryBig,-VeryBig);


cPt2di  TAB4Corner[4] = {{1,1},{-1,1},{-1,-1},{1,-1}};

template <class Type,const int Dim> cPtxd<Type,Dim>  cPtxd<Type,Dim>::PCste(const Type & aVal)
{
   cPtxd<Type,Dim> aRes;
   for (int aK=0 ; aK<Dim; aK++)
       aRes.mCoords[aK]= aVal;
   return aRes;
}

template <class Type,const int Dim> cPtxd<Type,Dim>  cPtxd<Type,Dim>::PRand()
{
   cPtxd<Type,Dim> aRes;
   for (int aK=0 ; aK<Dim; aK++)
       aRes.mCoords[aK]= tNumTrait<Type>::RandomValue();
   return aRes;
}

template <class Type,const int Dim> cPtxd<Type,Dim>  cPtxd<Type,Dim>::PRandC()
{
   cPtxd<Type,Dim> aRes;
   for (int aK=0 ; aK<Dim; aK++)
       aRes.mCoords[aK]= RandUnif_C();
   return aRes;
}

template <class Type,const int Dim> cPtxd<Type,Dim>  cPtxd<Type,Dim>::PRandUnit()
{
   cPtxd<Type,Dim> aRes = PRandC();
   while (NormInf(aRes)<1e-2)
        aRes = PRandC();
   return VUnit(aRes);
}

template <class Type,const int Dim> cPtxd<Type,Dim>  
      cPtxd<Type,Dim>::PRandUnitDiff(const cPtxd<Type,Dim>& aP0,const Type & aDist)
{
   cPtxd<Type,Dim> aRes = PRandUnit();
   while (NormInf(aRes-aP0)<aDist)
        aRes = PRandUnit();
   return aRes;
}





template <class Type,const int Dim> double NormK(const cPtxd<Type,Dim> & aPt,double anExp) 
{
   double aRes = pow(std::abs(aPt[0]),anExp);
   for (int aD=1 ; aD<Dim; aD++)
      aRes += pow(std::abs(aPt[aD]),anExp);
   return pow(aRes,1/anExp);
}

template <class Type,const int Dim> double Norm2(const cPtxd<Type,Dim> & aPt)
{
   double aRes = Square(aPt[0]);
   for (int aD=1 ; aD<Dim; aD++)
      aRes += Square(aPt[aD]);
   return sqrt(aRes);
}

template <class Type,const int Dim> Type Norm1(const cPtxd<Type,Dim> & aPt)
{
   Type aRes = std::abs(aPt[0]);
   for (int aD=1 ; aD<Dim; aD++)
      aRes += std::abs(aPt[aD]);
   return aRes;
}

template <class Type,const int Dim> Type NormInf(const cPtxd<Type,Dim> & aPt)
{
   Type aRes = std::abs(aPt[0]);
   for (int aD=1 ; aD<Dim; aD++)
      aRes = std::max(aRes,std::abs(aPt[aD]));
   return aRes;
}

template <class T,const int Dim>  
   typename  tNumTrait<T>::tBig Scal(const cPtxd<T,Dim> &aP1,const cPtxd<T,Dim> & aP2)
{
   typename tNumTrait<T>::tBig  aRes = aP1[0]*aP2[0];
   for (int aD=1 ; aD<Dim; aD++)
      aRes +=  aP1[aD]*aP2[aD];
   return aRes;
}

template <class T,const int Dim>  T Cos(const cPtxd<T,Dim> &aP1,const cPtxd<T,Dim> & aP2)
{
   return T(Scal(aP1,aP2)) / (Norm2(aP1)*Norm2(aP2));
}


template <class Type,const int Dim> std::ostream & operator << (std::ostream & OS,const cPtxd<Type,Dim> &aP)
{
    OS << "[" << aP.x();
    for (int aD=1; aD<Dim; aD++)
       OS << "," << aP[aD];
    OS << "]" ;
    return OS;
}

template<class T,const int Dim> cPtxd<T,Dim>  VUnit(const cPtxd<T,Dim> & aP)
{
   return aP / T(Norm2(aP));  // Check by 0 is made in operator
}



/*
template <class Type> std::ostream & operator << (std::ostream & OS,const cPtxd<Type,1> &aP)
{ return  OS << "[" << aP.x() << "]"; }
template <class Type> std::ostream & operator << (std::ostream & OS,const cPtxd<Type,2> &aP)
{ return  OS << "[" << aP.x() << "," << aP.y() << "]"; }
template <class Type> std::ostream & operator << (std::ostream & OS,const cPtxd<Type,3> &aP)
{ return  OS << "[" << aP.x() << "," << aP.y() << "," << aP.z()<< "]"; }
template <class Type> std::ostream & operator << (std::ostream & OS,const cPtxd<Type,4> &aP)
{ return  OS << "[" << aP.x() << "," << aP.y() << "," << aP.z() << "," << aP.t() << "]"; }
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


template <const int Dim> cPixBox<Dim>  cPixBox<Dim>::BoxWindow(const tPt & aC,int aSz)
{
    return cPixBox<Dim>(aC-cPtxd<int,Dim>::PCste(aSz),aC+cPtxd<int,Dim>::PCste(aSz+1));
}

template <const int Dim> cPixBox<Dim>  cPixBox<Dim>::BoxWindow(int aSz)
{
    return BoxWindow(cPtxd<int,Dim>::PCste(0),aSz);
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

template <const int Dim> bool  cPixBox<Dim>::SignalAtFrequence(const tPt & anIndex,double aFreq) const
{
   return MMVII::SignalAtFrequence(IndexeLinear(anIndex),aFreq,this->mNbElem-1);
}


template <const int Dim> cPtxd<int,Dim>  cPixBox<Dim>::FromIndexeLinear(tINT8  anIndexe) const
{
   tPt aRes;
   for (int aK=0 ; aK<Dim ; aK++)
   {
      aRes[aK]  = tBox::mP0[aK] + anIndexe % tBox::mSz[aK];
      anIndexe /= tBox::mSz[aK];
   }
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

template <const int Dim> cPtxd<int,Dim>  cPixBox<Dim>::CircNormProj(const tPt & aPt) const
{
    cPtxd<int,Dim>  aRes;
    for (int aD=0 ; aD<Dim ; aD++)
       aRes[aD] = tBox::mP0[aD] + mod(aPt[aD]-tBox::mP0[aD],tBox::mSz[aD]);
    return aRes;
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

template <class Type,const int Dim>   
   cTplBox<Type,Dim>::cTplBox
   (
       const cPtxd<Type,Dim> & aSz,
       bool AllowEmpty
   ) :
     cTplBox<Type,Dim>(tPt::PCste(0),aSz,AllowEmpty)
{
}


template <class Type,const int Dim> bool  cTplBox<Type,Dim>::IsEmpty() const
{
   return mNbElem == 0;
}

template <class Type,const int Dim> cTplBox<Type,Dim>  cTplBox<Type,Dim>::Empty()
{
   return  cTplBox<Type,Dim>(tPt::PCste(0),true);
}

template <class Type,const int Dim> cTplBox<Type,Dim>  cTplBox<Type,Dim>::Inter(const tBox & aBox)const
{
  return tBox(PtSupEq(mP0,aBox.mP0),PtInfEq(mP1,aBox.mP1),true);
}

template <class Type,const int Dim> cTplBox<Type,Dim>  cTplBox<Type,Dim>::Dilate(const tPt & aPt) const
{
   return tBox(mP0-aPt,mP1+aPt);
}


template <class Type,const int Dim> cTplBox<Type,Dim>  cTplBox<Type,Dim>::Dilate(const Type & aVal) const
{
   return Dilate(tPt::PCste(aVal));
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

cBox2dr ToR(const cBox2di & aBox)
{
   return cBox2dr(ToR(aBox.P0()),ToR(aBox.P1()));
}

cBox2di ToI(const cBox2dr & aBox)
{
    return cBox2di(Pt_round_down(aBox.P0()),Pt_round_up(aBox.P1()));
}

cBox2dr operator * (const cBox2dr & aBox,double aScale)
{
    return cBox2dr(aBox.P0()*aScale,aBox.P1()*aScale);
}



/* ========================== */
/*       cTpxBoxOfPts         */
/* ========================== */

template <class Type,const int Dim>   cTplBoxOfPts<Type,Dim>::cTplBoxOfPts() :
   mNbPts (0),
   mP0    (tPt::PCste(0)),
   mP1    (tPt::PCste(0))
{
}

template <class Type,const int Dim>  int  cTplBoxOfPts<Type,Dim>::NbPts() const {return mNbPts;}

template <class Type,const int Dim>  const cPtxd<Type,Dim> &  cTplBoxOfPts<Type,Dim>::P0() const
{
   MMVII_INTERNAL_ASSERT_medium(mNbPts,"cTplBoxOfPts<Type,Dim>::P0()")
   return  mP0;
}

template <class Type,const int Dim>  const cPtxd<Type,Dim> &  cTplBoxOfPts<Type,Dim>::P1() const
{
   MMVII_INTERNAL_ASSERT_medium(mNbPts,"cTplBoxOfPts<Type,Dim>::P1()")
   return  mP1;
}

template <class Type,const int Dim>  cTplBox<Type,Dim>  cTplBoxOfPts<Type,Dim>::CurBox() const
{
    return  cTplBox<Type,Dim>(mP0,mP1);
}

template <class Type,const int Dim>  void  cTplBoxOfPts<Type,Dim>::Add(const tPt & aP)
{
   if (mNbPts==0)
   {
      mP0 = aP;
      mP1 = aP;
   }
   else
   {
      SetInfEq(mP0,aP);
      SetSupEq(mP1,aP);
   }
   mNbPts++;
}

template <class Type,const int Dim>  cPtxd<int,Dim> Pt_round_down(const cPtxd<Type,Dim>  aP)
{
   return ICByC1P(aP,round_down);
}
template <class Type,const int Dim>  cPtxd<int,Dim> Pt_round_up(const cPtxd<Type,Dim>  aP)
{
   return ICByC1P(aP,round_up);
}
template <class Type,const int Dim>  cPtxd<int,Dim> Pt_round_ni(const cPtxd<Type,Dim>  aP)
{
   return ICByC1P(aP,round_ni);
}


/* ========================== */
/*       INSTANTIATION        */
/* ========================== */


#define MACRO_INSTATIATE_PTXD(TYPE,DIM)\
template  std::ostream & operator << (std::ostream & OS,const cPtxd<TYPE,DIM> &aP);\
template  cPtxd<TYPE,DIM> cPtxd<TYPE,DIM>::PCste(const TYPE&);\
template  cPtxd<TYPE,DIM> cPtxd<TYPE,DIM>::PRand();\
template  cPtxd<TYPE,DIM> cPtxd<TYPE,DIM>::PRandC();\
template  cPtxd<TYPE,DIM> cPtxd<TYPE,DIM>::PRandUnit();\
template  cPtxd<TYPE,DIM>  cPtxd<TYPE,DIM>::PRandUnitDiff(const cPtxd<TYPE,DIM>& ,const TYPE&);\
template  double NormK(const cPtxd<TYPE,DIM> & aPt,double anExp);\
template  double Norm2(const cPtxd<TYPE,DIM> & aPt);\
template  TYPE Norm1(const cPtxd<TYPE,DIM> & aPt);\
template  TYPE NormInf(const cPtxd<TYPE,DIM> & aPt);\
template  typename  tNumTrait<TYPE>::tBig Scal(const cPtxd<TYPE,DIM> &,const cPtxd<TYPE,DIM> &);\
template  TYPE Cos(const cPtxd<TYPE,DIM> &,const cPtxd<TYPE,DIM> &);\
template  cPtxd<TYPE,DIM>  VUnit(const cPtxd<TYPE,DIM> & aP);

// template  cPtxd<TYPE,DIM>  PCste(const DIM & aVal);

#define MACRO_INSTATIATE_POINT(DIM)\
MACRO_INSTATIATE_PTXD(tINT4,DIM)\
MACRO_INSTATIATE_PTXD(tREAL4,DIM)\
MACRO_INSTATIATE_PTXD(tREAL8,DIM)\
MACRO_INSTATIATE_PTXD(tREAL16,DIM)

#define MACRO_INSTATIATE_PRECT_DIM(DIM)\
MACRO_INSTATIATE_POINT(DIM)\
template cPtxd<int,DIM> Pt_round_down(const cPtxd<double,DIM>  aP);\
template cPtxd<int,DIM> Pt_round_up(const cPtxd<double,DIM>  aP);\
template cPtxd<int,DIM> Pt_round_ni(const cPtxd<double,DIM>  aP);\
template class cBorderPixBoxIterator<DIM>;\
template class cBorderPixBox<DIM>;\
template class cTplBox<tINT4,DIM>;\
template class cTplBoxOfPts<tINT4,DIM>;\
template  class cParseBoxInOut<DIM>;\
template class cTplBox<tREAL8,DIM>;\
template class cTplBoxOfPts<tREAL8,DIM>;\
template class cPixBox<DIM>;\
template  int NbPixVign(const cPtxd<int,DIM> & aVign);\
template class cDataGenUnTypedIm<DIM>;\
template <> const cPixBox<DIM> cPixBox<DIM>::TheEmptyBox(cPtxd<int,DIM>::PCste(0),cPtxd<int,DIM>::PCste(0),true);


/*
void F()
{
   cPtxd<tINT4,2>::T_PCste(2);
}
*/

MACRO_INSTATIATE_PRECT_DIM(1)
MACRO_INSTATIATE_PRECT_DIM(2)
MACRO_INSTATIATE_PRECT_DIM(3)
MACRO_INSTATIATE_POINT(4)



};
