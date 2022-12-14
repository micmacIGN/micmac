#include "MMVII_Images.h"
#include "MMVII_Geom2D.h"
#include "MMVII_MMV1Compat.h"

namespace MMVII
{

/* ========================== */
/*        cSegment            */
/* ========================== */

template <class Type,const int Dim> cSegment<Type,Dim>::cSegment(const tPt& aP1,const tPt& aP2) :
   mP1  (aP1),
   mP2  (aP2)
{
    MMVII_INTERNAL_ASSERT_tiny(mP1!=mP2,"Identic point in segment");
}

template <class Type,const int Dim> void cSegment<Type,Dim>::CompileFoncLinear
                              (Type & aVal,tPt & aVec,const Type &aV1,const Type & aV2) const
{
	// return aV1 + (aV2-aV1) * Scal(mTgt,aP-this->mP1) / mN2;
    tPt aV12 =  (mP2-mP1) ;
    aVec  =   aV12 * Type((aV2-aV1) /SqN2(aV12)) ;
    aVal = aV1  - Scal(aVec,mP1);
}

/* ========================== */
/*    cSegmentCompiled        */
/* ========================== */

template <class Type,const int Dim> cSegmentCompiled<Type,Dim>::cSegmentCompiled(const tPt& aP1,const tPt& aP2) :
    cSegment<Type,Dim>(aP1,aP2),
    mN2     (Norm2(aP2-aP1)),
    mTgt    ((aP2-aP1)/mN2)
{
}

template <class Type,const int Dim> cPtxd<Type,Dim>  cSegmentCompiled<Type,Dim>::Proj(const tPt & aPt) const
{
     return this->mP1 + mTgt * Type(Scal(mTgt,aPt-this->mP1)) ;
}

template <class Type,const int Dim> Type  cSegmentCompiled<Type,Dim>::Dist(const tPt & aPt) const
{
	return Norm2(aPt-Proj(aPt));
}
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

template <class Type> double  SpecAbsSurfParalogram(const cPtxd<Type,2> & aP1,const cPtxd<Type,2> & aP2)
{
    return std::abs(aP1 ^ aP2) ;
}
template <class Type>  double SpecAbsSurfParalogram(const  cPtxd<Type,3> & aP1,const  cPtxd<Type,3> & aP2)
{
    return Norm2(aP1 ^ aP2);
}

template <class Type,const int Dim>  Type AbsSurfParalogram(const cPtxd<Type,Dim>& aP1,const cPtxd<Type,Dim>& aP2)
{
    return SpecAbsSurfParalogram(aP1,aP2);
}

template <class T>   cPtxd<T,3> TP3z0  (const cPtxd<T,2> & aPt)
{
    return cPtxd<T,3>(aPt.x(),aPt.y(),0);
}
template <class T>   cPtxd<T,2> Proj  (const cPtxd<T,3> & aPt)
{
    return cPtxd<T,2>(aPt.x(),aPt.y());
}


template <class Type,const int Dim> cPtxd<Type,Dim> cPtxd<Type,Dim>::FromStdVector(const std::vector<Type>& aV)
{
   cPtxd<Type,Dim> aRes;
   MMVII_INTERNAL_ASSERT_tiny(aV.size()==Dim,"Bad size in Vec/Pt");
   for (int aK=0 ; aK<Dim ; aK++)
        aRes.mCoords[aK]  = aV.at(aK);

   return aRes;
}

template <class T,const int Dim> cPtxd<tREAL8,Dim> Barry(const std::vector<cPtxd<T,Dim> > & aVPts)
{
   MMVII_INTERNAL_ASSERT_tiny((!aVPts.empty()),"Bad size in Vec/Pt");

   cPtxd<tREAL8,Dim> aRes = ToR(aVPts[0]);
   for (int aK=1 ; aK<int(aVPts.size()) ; aK++)
      aRes += ToR(aVPts[aK]);

   return aRes/ tREAL8(aVPts.size());
}



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

template <class Type,const int Dim> cPtxd<Type,Dim>  cPtxd<Type,Dim>::FromPtInt(const cPtxd<int,Dim> & aPInt)
{
   cPtxd<Type,Dim> aRes;
   for (int aK=0 ; aK<Dim; aK++)
       aRes.mCoords[aK]= aPInt[aK];
   return aRes;
}

template <class Type,const int Dim> cPtxd<Type,Dim>  cPtxd<Type,Dim>::FromPtR(const cPtxd<tREAL8,Dim> & aPtR)
{
   cPtxd<Type,Dim> aRes;
   for (int aK=0 ; aK<Dim; aK++)
       aRes.mCoords[aK]= aPtR[aK];
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

template <class Type,const int Dim> cPtxd<Type,Dim>  cPtxd<Type,Dim>::PRandInSphere()
{
   cPtxd<Type,Dim> aRes = PRandC();
   while (Norm2(aRes)>1.0)
        aRes = PRandC();
   return aRes;
}

template <class Type,const int Dim> cPtxd<Type,Dim>  
      cPtxd<Type,Dim>::PRandUnitNonAligned(const cPtxd<Type,Dim>& aP0,const Type & aDist)
{
   cPtxd<Type,Dim> aRes = PRandUnit();
   while ((NormInf(aRes-aP0)<aDist) || (NormInf(aRes+aP0)<aDist) )
        aRes = PRandUnit();
   return aRes;
}


template <class Type,const int Dim> cPtxd<Type,Dim>  
      cPtxd<Type,Dim>::PRandUnitDiff(const cPtxd<Type,Dim>& aP0,const Type & aDist)
{
   cPtxd<Type,Dim> aRes = PRandUnit();
   while (NormInf(aRes-aP0)<aDist)
        aRes = PRandUnit();
   return aRes;
}


template <class Type,const int Dim> 
   typename cPtxd<Type,Dim>::tBigNum cPtxd<Type,Dim>::MinSqN2(const std::vector<tPt> & aVecPts,bool SVP) const
{
   if (aVecPts.empty())
   {
       MMVII_INTERNAL_ASSERT_medium(SVP,"MinSqN2 on empty vect, no def");
       return -1;
   }
   tBigNum aRes = SqN2(aVecPts[0]-*this);
   for (size_t  aKPts=1 ; aKPts<aVecPts.size() ; aKPts++)
       aRes = std::min(aRes, SqN2(aVecPts[aKPts]-*this));
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

template <class Type,const int Dim> Type MinAbsCoord(const cPtxd<Type,Dim> & aPt)
{
   Type aRes = std::abs(aPt[0]);
   for (int aD=1 ; aD<Dim; aD++)
      aRes = std::min(aRes,std::abs(aPt[aD]));
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

template <class T,const int Dim>  
   typename  tNumTrait<T>::tBig MulCoord(const cPtxd<T,Dim> &aPt)
{
   typename tNumTrait<T>::tBig  aRes = aPt[0];
   for (int aD=1 ; aD<Dim; aD++)
      aRes *=  aPt[aD];
   return aRes;
}





template <class T,const int Dim>  T Cos(const cPtxd<T,Dim> &aP1,const cPtxd<T,Dim> & aP2)
{
   return SafeDiv(T(Scal(aP1,aP2)) , T(Norm2(aP1)*Norm2(aP2)));
}
template <class T,const int Dim>  T AbsAngle(const cPtxd<T,Dim> &aP1,const cPtxd<T,Dim> & aP2)
{
   T aCos = Cos(aP1,aP2);
   MMVII_INTERNAL_ASSERT_tiny((aCos>=1)&&(aCos<=-1),"AbsAngle cosinus out range");
   return acos(aCos);
}

template <class T,const int Dim>  T AbsAngleTrnk(const cPtxd<T,Dim> &aP1,const cPtxd<T,Dim> & aP2)
{
   T aCos = std::max(T(-1.0),std::min(T(1.0),Cos(aP1,aP2)));
   return acos(aCos);
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

template <class Type,const int DimOut,const int DimIn> cPtxd<Type,DimOut> CastDim(const cPtxd<Type,DimIn> & aPt)
{
    MMVII_INTERNAL_ASSERT_tiny(DimIn==DimOut,"CastDim : different dim");

    return  cPtxd<Type,DimOut>(aPt.PtRawData());
   
}



template <const int Dim>  class cAllocNeighourhood
{

   public :
      typedef  cPtxd<int,Dim>    tPt;
      typedef  std::vector<tPt>  tVecPt;
      typedef  std::vector<tVecPt>  tVVPt;

      static  const tVVPt & AllocTabGrowNeigh(int aDMax)
      {
           static  tVVPt aRes;;
           if (int(aRes.size()) > aDMax) return aRes;

           aRes = tVVPt(1+aDMax,tVecPt());
           
           for (const auto & aP :  cPixBox(tPt::PCste(-aDMax),tPt::PCste(aDMax+1)))
               aRes.at(NormInf(aP)).push_back(aP);

           return aRes;
      }


      static const tVecPt &  Alloc(int aNbPix)
      {
// StdOut() <<  "----------------======================\n";
            static  std::vector<tVecPt> aBufRes(Dim);
            MMVII_INTERNAL_ASSERT_tiny((aNbPix>0)&&(aNbPix<=Dim),"Bad Nb in neighbourhood");

            // If alreay computed all fine
            tVecPt & aRes = aBufRes[aNbPix-1];
            if (! aRes.empty()) return aRes;

            // Usee full neighboor
            cPixBox<Dim> aPB(tPt::PCste(-1),tPt::PCste(2));
            for (const auto & aP : aPB)
            {
                int aN = Norm1(aP);
                if ((aN>0) && (aN<=aNbPix))
                {
                    aRes.push_back(aP);
                    // StdOut() << aP << "\n";
                }
            }

            return aRes;
      }
};

template <const int Dim>  const std::vector<cPtxd<int,Dim>> & AllocNeighbourhood(int aNbVois)
{
   return  cAllocNeighourhood<Dim>::Alloc(aNbVois);
}


template <const int Dim>  const std::vector<std::vector<cPtxd<int,Dim>>> & TabGrowNeigh(int aDistMax)
{
   return  cAllocNeighourhood<Dim>::AllocTabGrowNeigh(aDistMax);
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

template <const int Dim> cPixBox<Dim>  cPixBox<Dim>::BoxWindow(const tPt & aC,const tPt & aSz)
{
    return cPixBox<Dim>(aC-aSz,aC+aSz+tPt::PCste(1));
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

template <const int Dim> tINT8  cPixBox<Dim>::IndexeUnorderedPair(const tPt&aP1,const tPt&aP2) const
{
    tINT8 aI1 = IndexeLinear(aP1);  // num of P1 in Box
    tINT8 aI2 = IndexeLinear(aP2);  // num of P2 in Box
    OrderMinMax(aI1,aI2);           // to assure P1,P2 = P2,P1
    return  aI1*this->NbElem() + aI2;     // make a unique index
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


template <const int Dim>  cPixBox<Dim> cParseBoxInOut<Dim>::BoxOutLoc(const tPt & anIndex,const tPt& aDil) const
{
   tBox aBoxIn  =  BoxIn(anIndex,aDil);
   tBox aBoxOut =  BoxOut(anIndex);
   return tBox(aBoxOut.P0() -  aBoxIn.P0(), aBoxOut.P1() -  aBoxIn.P0());
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

template <class Type,const int Dim>
   cTplBox<Type,Dim>  cTplBox<Type,Dim>::BoxCste(Type aVal)
{
   return cTplBox<Type,Dim>(tPt::PCste(-aVal),tPt::PCste(aVal));
}

template <class Type,const int Dim>
   cTplBox<Type,Dim>  cTplBox<Type,Dim>::BigBox()
{
     return  BoxCste(tNumTrait<Type>::MaxValue());
}


template <class Type,const int Dim> bool  cTplBox<Type,Dim>::IsEmpty() const
{
   return mNbElem == 0;
}

template <class Type,const int Dim> cTplBox<Type,Dim>  cTplBox<Type,Dim>::Empty()
{
   return  cTplBox<Type,Dim>(tPt::PCste(0),true);
}

template <class Type,const int Dim> 
   cTplBox<Type,Dim>  cTplBox<Type,Dim>::FromVect(const tPt * aBegin,const tPt * aEnd,bool AllowEmpty)
{
    return cTplBoxOfPts<Type,Dim>::FromVect(aBegin,aEnd).CurBox(AllowEmpty);
}

template <class Type,const int Dim> 
   cTplBox<Type,Dim>  cTplBox<Type,Dim>::FromVect(const std::vector<tPt>& aVect,bool AllowEmpty)
{
    return cTplBoxOfPts<Type,Dim>::FromVect(aVect).CurBox(AllowEmpty);
}

template <class Type,const int Dim> cTplBox<Type,Dim>  cTplBox<Type,Dim>::Inter(const tBox & aBox)const
{
  return tBox(PtSupEq(mP0,aBox.mP0),PtInfEq(mP1,aBox.mP1),true);
}

template <class Type,const int Dim> cTplBox<Type,Dim>  cTplBox<Type,Dim>::Sup(const tBox & aB2)const
{
  if (IsEmpty() && aB2.IsEmpty()) 
     return  Empty();
  if (IsEmpty())     return aB2;
  if (aB2.IsEmpty()) return *this;

  return tBox(PtInfEq(mP0,aB2.mP0),PtSupEq(mP1,aB2.mP1),true);
}

template <class Type,const int Dim> Type cTplBox<Type,Dim>::Insideness(const tPt & aP) const
{
     Type  aRes = std::min(aP[0]-mP0[0],mP1[0]-aP[0]);

     for (int aD=1 ; aD<Dim ; aD++)
         UpdateMin(aRes,std::min(aP[aD]-mP0[aD],mP1[aD]-aP[aD]));

     return aRes;
}



template <class Type,const int Dim> cTplBox<Type,Dim>  cTplBox<Type,Dim>::Dilate(const tPt & aPt) const
{
   return tBox(mP0-aPt,mP1+aPt);
}


template <class Type,const int Dim> cTplBox<Type,Dim>  cTplBox<Type,Dim>::Dilate(const Type & aVal) const
{
   return Dilate(tPt::PCste(aVal));
}

template <class Type,const int Dim> cTplBox<Type,Dim>  cTplBox<Type,Dim>::ScaleCentered(const Type & aVal) const
{
   tPt aMil = (mP0+mP1)/Type(2.0);
   tPt anAmpl = (mP1-mP0)* Type(aVal/2.0);
   return tBox(aMil-anAmpl,aMil+anAmpl);
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

template <class Type,const int Dim> size_t  cTplBox<Type,Dim>::NbFlagCorner() {return 1<<Dim;}

template <class Type,const int Dim> cPtxd<Type,Dim>  cTplBox<Type,Dim>::CornerOfFlag(size_t aFlag,const tPt &aP0,const tPt &aP1) 
{
   tPt aRes;
   for (size_t aD=0 ; aD<Dim ; aD++)
   {
       aRes[aD] = (aFlag & (1<<aD)) ? aP0[aD] : aP1[aD];
   }
   return aRes;
}

template <class Type,const int Dim> cPtxd<Type,Dim>  cTplBox<Type,Dim>::CornerOfFlag(size_t aFlag) const
{
     return CornerOfFlag(aFlag,mP0,mP1);
}

template <class Type,const int Dim> void  cTplBox<Type,Dim>::Corners(tCorner & aRes,const tPt &aP0,const tPt &aP1) 
{
    for (size_t aFlag=0; aFlag<NbFlagCorner()  ; aFlag++)
    {
        aRes[aFlag] = CornerOfFlag(aFlag,aP0,aP1);
    }
}

template <class Type,const int Dim> void  cTplBox<Type,Dim>::Corners(tCorner & aRes) const
{
     Corners(aRes,mP0,mP1);
}



template <class Type,const int Dim> Type  cTplBox<Type,Dim>::DistMax2Corners(const tPt& aPt) const
{
    Type aRes = SqN2(aPt-CornerOfFlag(0));
    for (size_t aFlag=1; aFlag<NbFlagCorner()  ; aFlag++)
    {
        UpdateMax(aRes,(Type)SqN2(aPt-CornerOfFlag(aFlag)));
    }
    return std::sqrt(aRes);
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
/*
   cPtxd<double,Dim> aP0 = RandomNormalised();
   cPtxd<Type,Dim>  aP1 = FromNormaliseCoord(aP0);
   cPtxd<Type,Dim> aP2 = Proj(aP1);
StdOut() << "BOX " << mP0 << " " << mP1 << "\n";
StdOut() <<  aP0 << aP1 << aP2 << "\n"; getchar();
   return aP2;
*/
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

/// Tricky but ToR in cTplBox<tREAL8,Dim> cannot be understood as I want
template <class Type,const int Dim>  cPtxd<tREAL8,Dim> GlobToR(const cPtxd<Type,Dim> &aV){return ToR(aV);}


template <class Type,const int Dim>  cTplBox<tREAL8,Dim> cTplBox<Type,Dim>::ToR() const
{
   return cTplBox<tREAL8,Dim>(GlobToR(mP0),GlobToR(mP1));
}

template <class Type,const int Dim>  cTplBox<tINT4,Dim> cTplBox<Type,Dim>::ToI() const
{
   return cTplBox<tINT4,Dim>(Pt_round_down(mP0),Pt_round_up(mP1));
}


cBox2dr operator * (const cBox2dr & aBox,double aScale)
{
    return cBox2dr(aBox.P0()*aScale,aBox.P1()*aScale);
}

//        x0,y1        x1,y1
//        x0,y0        x1,y0     

template <class Type> 
    void CornersTrigo(typename cTplBox<Type,2>::tCorner & aRes,const  cTplBox<Type,2>& aBox)
{
   aRes[0] = cPtxd<Type,2>(aBox.P1().x(),aBox.P1().y());
   aRes[1] = cPtxd<Type,2>(aBox.P0().x(),aBox.P1().y());
   aRes[2] = cPtxd<Type,2>(aBox.P0().x(),aBox.P0().y());
   aRes[3] = cPtxd<Type,2>(aBox.P1().x(),aBox.P0().y());
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



template <class Type,const int Dim> 
   cTplBoxOfPts<Type,Dim>  cTplBoxOfPts<Type,Dim>::FromVect(const tPt * aBegin,const tPt * aEnd)
{
   cTplBoxOfPts<Type,Dim> aRes;
   for (const auto * aPtrP=aBegin; aPtrP<aEnd ; aPtrP++)
       aRes.Add(*aPtrP);
   return  aRes;
}
template <class Type,const int Dim> 
   cTplBoxOfPts<Type,Dim>  cTplBoxOfPts<Type,Dim>::FromVect(const std::vector<tPt>& aVect)
{
   return FromVect(aVect.data(),aVect.data()+aVect.size());
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

template <class Type,const int Dim>  cTplBox<Type,Dim>  cTplBoxOfPts<Type,Dim>::CurBox(bool AllowEmpty) const
{
    return  cTplBox<Type,Dim>(mP0,mP1,AllowEmpty);
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

template <class Type,const int Dim>  cPtxd<int,Dim> Pt_round_down(const cPtxd<Type,Dim> & aP)
{
   return ICByC1P(aP,round_down);
}
template <class Type,const int Dim>  cPtxd<int,Dim> Pt_round_up(const cPtxd<Type,Dim>&  aP)
{
   return ICByC1P(aP,round_up);
}
template <class Type,const int Dim>  cPtxd<int,Dim> Pt_round_ni(const cPtxd<Type,Dim>&  aP)
{
   return ICByC1P(aP,round_ni);
}


template <class Type> bool WindInside4BL(const cBox2di & aBox,const cPtxd<Type,2> & aPt,const  cPt2di & aSzW)
{
   return
	   (aPt.x() >= aBox.P0().x() + aSzW.x())
       &&  (aPt.y() >= aBox.P0().y() + aSzW.y())
       &&  (aPt.x() <  aBox.P1().x() - aSzW.x()-1)
       &&  (aPt.y() <  aBox.P1().y() - aSzW.y()-1) ;
}

template <class Type> bool WindInside(const cBox2di & aBox,const cPt2di & aPt,const  cPt2di & aSzW)
{
   return
	   (aPt.x() >= aBox.P0().x() + aSzW.x())
       &&  (aPt.y() >= aBox.P0().y() + aSzW.y())
       &&  (aPt.x() <  aBox.P1().x() - aSzW.x())
       &&  (aPt.y() <  aBox.P1().y() - aSzW.y()) ;
}

template <class Type> bool WindInside(const cBox2di & aBox,const cPt2di & aPt,const int  & aSz)
{
    return WindInside(aBox,aPt,cPt2di(aSz,aSz));
}

/*
template <const int Dim>  cTplBox<tREAL8,Dim> ToR(const  cTplBox<int,Dim> & aBox)
{
	 return cTplBox<tREAL8,Dim>(ToR(aBox.P0()),ToR(aBox.P1()));
}

template <const int Dim>  cTplBox<int,Dim> ToI(const  cTplBox<tREAL8,Dim> & aBox)
{
	 return cTplBox<int,Dim>(ToI(aBox.P0()),ToI(aBox.P1()));
}
*/

/* ========================== */
/*       INSTANTIATION        */
/* ========================== */


	/*
template   cTplBox<tREAL8,2> ToR(const  cTplBox<int,2> & aBox);
template   cTplBox<tREAL8,3> ToR(const  cTplBox<int,3> & aBox);
template   cTplBox<int,2> ToI(const  cTplBox<tREAL8,2> & aBox);
template   cTplBox<int,3> ToI(const  cTplBox<tREAL8,3> & aBox);
*/




#define INSTANTIATE_ABS_SURF(TYPE)\
template  cPtxd<TYPE,3> TP3z0  (const cPtxd<TYPE,2> & aPt);\
template  cPtxd<TYPE,2> Proj  (const cPtxd<TYPE,3> & aPt);\
template  TYPE AbsSurfParalogram(const cPtxd<TYPE,2>& aP1,const cPtxd<TYPE,2>& aP2);\
template  TYPE AbsSurfParalogram(const cPtxd<TYPE,3>& aP1,const cPtxd<TYPE,3>& aP2);


INSTANTIATE_ABS_SURF(tINT4)
INSTANTIATE_ABS_SURF(tREAL4)
INSTANTIATE_ABS_SURF(tREAL8)
INSTANTIATE_ABS_SURF(tREAL16)

#define INSTANTIATE_SEGM_TYPE(TYPE,DIM)\
template class cSegment<TYPE,DIM>;\
template class cSegmentCompiled<TYPE,DIM>;

#define INSTANTIATE_SEGM(DIM)\
INSTANTIATE_SEGM_TYPE(tREAL4,DIM)\
INSTANTIATE_SEGM_TYPE(tREAL8,DIM)\
INSTANTIATE_SEGM_TYPE(tREAL16,DIM)

INSTANTIATE_SEGM(1)
INSTANTIATE_SEGM(2)
INSTANTIATE_SEGM(3)

template void CornersTrigo(typename cTplBox<tREAL8,2>::tCorner & aRes,const  cTplBox<tREAL8,2>&);
template void CornersTrigo(typename cTplBox<tINT4,2>::tCorner & aRes,const  cTplBox<tINT4,2>&);

template  bool WindInside4BL(const cBox2di & aBox,const cPtxd<tINT4,2> & aPt,const  cPt2di & aSzW);
template  bool WindInside4BL(const cBox2di & aBox,const cPtxd<tREAL8,2> & aPt,const  cPt2di & aSzW);

#define MACRO_INSTATIATE_PTXD_2DIM(TYPE,DIMIN,DIMOUT)\
template  cPtxd<TYPE,DIMOUT> CastDim<TYPE,DIMOUT,DIMIN>(const cPtxd<TYPE,DIMIN> & aPt);

#define MACRO_INSTATIATE_PTXD(TYPE,DIM)\
MACRO_INSTATIATE_PTXD_2DIM(TYPE,DIM,1);\
MACRO_INSTATIATE_PTXD_2DIM(TYPE,DIM,2);\
MACRO_INSTATIATE_PTXD_2DIM(TYPE,DIM,3);\
MACRO_INSTATIATE_PTXD_2DIM(TYPE,DIM,4);\
template  cPtxd<tREAL8,DIM> Barry(const std::vector<cPtxd<TYPE,DIM> > & aVPts);\
template  std::ostream & operator << (std::ostream & OS,const cPtxd<TYPE,DIM> &aP);\
template  cPtxd<TYPE,DIM> cPtxd<TYPE,DIM>::PCste(const TYPE&);\
template  cPtxd<TYPE,DIM> cPtxd<TYPE,DIM>::FromStdVector(const std::vector<TYPE>&);\
template  cPtxd<TYPE,DIM> cPtxd<TYPE,DIM>::PRand();\
template  cPtxd<TYPE,DIM> cPtxd<TYPE,DIM>::PRandC();\
template  cPtxd<TYPE,DIM> cPtxd<TYPE,DIM>::PRandUnit();\
template  cPtxd<TYPE,DIM> cPtxd<TYPE,DIM>::PRandInSphere();\
template typename cPtxd<TYPE,DIM>::tBigNum cPtxd<TYPE,DIM>::MinSqN2(const std::vector<tPt> &,bool SVP) const;\
template  cPtxd<TYPE,DIM>  cPtxd<TYPE,DIM>::PRandUnitDiff(const cPtxd<TYPE,DIM>& ,const TYPE&);\
template  cPtxd<TYPE,DIM>  cPtxd<TYPE,DIM>::PRandUnitNonAligned(const cPtxd<TYPE,DIM>& ,const TYPE&);\
template  double NormK(const cPtxd<TYPE,DIM> & aPt,double anExp);\
template  double Norm2(const cPtxd<TYPE,DIM> & aPt);\
template  TYPE Norm1(const cPtxd<TYPE,DIM> & aPt);\
template  TYPE NormInf(const cPtxd<TYPE,DIM> & aPt);\
template  TYPE MinAbsCoord(const cPtxd<TYPE,DIM> & aPt);\
template  typename  tNumTrait<TYPE>::tBig Scal(const cPtxd<TYPE,DIM> &,const cPtxd<TYPE,DIM> &);\
template  typename  tNumTrait<TYPE>::tBig MulCoord(const cPtxd<TYPE,DIM> &);\
template  TYPE Cos(const cPtxd<TYPE,DIM> &,const cPtxd<TYPE,DIM> &);\
template  TYPE AbsAngle(const cPtxd<TYPE,DIM> &,const cPtxd<TYPE,DIM> &);\
template  TYPE AbsAngleTrnk(const cPtxd<TYPE,DIM> &,const cPtxd<TYPE,DIM> &);\
template  cPtxd<TYPE,DIM>  VUnit(const cPtxd<TYPE,DIM> & aP);\
template  cPtxd<TYPE,DIM>  cPtxd<TYPE,DIM>::FromPtInt(const cPtxd<int,DIM> & aPInt);\
template  cPtxd<TYPE,DIM>  cPtxd<TYPE,DIM>::FromPtR(const cPtxd<tREAL8,DIM> & aPInt);

// template  cPtxd<TYPE,DIM>  PCste(const DIM & aVal);

#define MACRO_INSTATIATE_POINT(DIM)\
MACRO_INSTATIATE_PTXD(tINT4,DIM)\
MACRO_INSTATIATE_PTXD(tREAL4,DIM)\
MACRO_INSTATIATE_PTXD(tREAL8,DIM)\
MACRO_INSTATIATE_PTXD(tREAL16,DIM)


#define MACRO_INSTATIATE_ROUNDPT(TYPE,DIM)\
template cPtxd<int,DIM> Pt_round_down(const cPtxd<TYPE,DIM>&  aP);\
template cPtxd<int,DIM> Pt_round_up(const cPtxd<TYPE,DIM>&  aP);\
template cPtxd<int,DIM> Pt_round_ni(const cPtxd<TYPE,DIM>&  aP);\


#define MACRO_INSTATIATE_PRECT_DIM(DIM)\
MACRO_INSTATIATE_POINT(DIM)\
template const std::vector<std::vector<cPtxd<int,DIM>>> & TabGrowNeigh(int);\
template const std::vector<cPtxd<int,DIM>> & AllocNeighbourhood(int);\
MACRO_INSTATIATE_ROUNDPT(tINT4,DIM)\
MACRO_INSTATIATE_ROUNDPT(tREAL4,DIM)\
MACRO_INSTATIATE_ROUNDPT(tREAL8,DIM)\
MACRO_INSTATIATE_ROUNDPT(tREAL16,DIM)\
template class cBorderPixBoxIterator<DIM>;\
template class cBorderPixBox<DIM>;\
template class cTplBox<tINT4,DIM>;\
template class cTplBoxOfPts<tINT4,DIM>;\
template  class cParseBoxInOut<DIM>;\
template class cTplBox<tREAL4,DIM>;\
template class cTplBox<tREAL8,DIM>;\
template class cTplBox<tREAL16,DIM>;\
template class cTplBoxOfPts<tREAL4,DIM>;\
template class cTplBoxOfPts<tREAL8,DIM>;\
template class cTplBoxOfPts<tREAL16,DIM>;\
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
