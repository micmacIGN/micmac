#include "CodedTarget.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "include/MMVII_Tpl_Images.h"


namespace MMVII
{


/* ================================================= */
/*                                                   */
/*               cFilterDCT                          */
/*                                                   */
/* ================================================= */

template <class Type>  
   cFilterDCT<Type>::cFilterDCT
   (
          bool        isCumul,  // is it efficient as cumul filter
          eDCTFilters aModeF,   // type of filter
          tIm anIm,             // first image (give the size)
          bool IsSym,           // Is it symetric
          double aR0,           // R min
          double aR1,           // R Max
          double aThickN        // thickness use in "min value mode"
   ) :
      mIsCumul (isCumul),
      mModeF   (aModeF),
      mIm      (anIm),
      mDIm     (anIm.DIm()),
      mSz      (mDIm.Sz()),
      mIsSym   (IsSym),
      mR0      (aR0),
      mR1      (aR1),
      mThickN  (aThickN),
      mIVois   (SortedVectOfRadius(aR0,aR1,IsSym))   // sorted by ray neigboor, with o w/o symetric pixel
{
   // compute real neigboor
   for (const auto & aPix : mIVois)
       mRVois.push_back(ToR(aPix));  

   mRhoEnd = Norm2(mRVois.back());

   //  compute the series of intervall,  
   int aK0=0;
   int aK1=aK0;
   if (mIsCumul)
      mVK0K1.push_back(cPt2di(0,0));
   IncrK1(aK1,aK0);
   mVK0K1.push_back(cPt2di(aK0,aK1));
   // StdOut() << "KKKK " << aK0 << " " << aK1 << " \n";

   while (aK1<int(mIVois.size()))
   {
       IncrK0(aK0);
       IncrK1(aK1,aK0);
       mVK0K1.push_back(cPt2di(aK0,aK1));
   }

}

template <class Type>  cFilterDCT<Type>::~cFilterDCT()
{
}

template <class Type> void cFilterDCT<Type>::UpdateSelected(cNS_CodedTarget::cDCT & aDC)  const
{
}


template <class Type>   void cFilterDCT<Type>::IncrK0(int & aK0)
{
     double aEpsilon=1e-3;

     double  aRhoMax =  Norm2(mIVois[aK0]) + aEpsilon;
     if (! mIsCumul)
     {
        aRhoMax = std::min(aRhoMax +mThickN,mRhoEnd-mThickN+2*aEpsilon);
     }
     while  (  (aK0<int(mIVois.size()))  &&  (Norm2(mRVois[aK0])<aRhoMax)  )
     {
           aK0++;
     }
}

template <class Type>   void cFilterDCT<Type>::IncrK1(int & aK1,const int & aK0)
{
    int aRhoMax = round_up(Square(  Norm2(mIVois[aK0]) + mThickN));

    while ( (aK1<int(mIVois.size()))  && (SqN2(mIVois[aK1]) <= aRhoMax))
          aK1++;
}

template <class Type> double cFilterDCT<Type>::ComputeVal(const cPt2dr & aP)
{
    mCurC = aP;
    Reset() ;
    for (const auto & aNeigh : mRVois)
        Add(1.0,aNeigh);

    return Compute();
}
//static bool BUGF = false;

template <class Type> double cFilterDCT<Type>::ComputeValMaxCrown(const cPt2dr & aP,const double & aThreshold)
{
    mCurC = aP;
    double aVMax = -1;

    if (mIsCumul)
    {
       Reset() ;

       for (int aKPt=0 ; (aKPt+1)<int(mVK0K1.size()) ; aKPt++)
       {
           const cPt2di & aKPrec = mVK0K1.at(aKPt);
           const cPt2di & aKNext = mVK0K1.at(aKPt+1);
           int aK0Prec = aKPrec.x();
           int aK0Next = aKNext.x();
           int aK1Prec = aKPrec.y();
           int aK1Next = aKNext.y();

           for (int aK=aK1Prec ; aK<aK1Next ; aK++)
               Add(1.0,mRVois[aK]);
           for (int aK=aK0Prec ; aK<aK0Next ; aK++)
               Add(-1.0,mRVois[aK]);
           UpdateMax(aVMax,Compute());
           if (aVMax> aThreshold)  
              return aVMax;
       }
    }
    else
    {
       for (int aKPt=0 ; aKPt<int(mVK0K1.size()) ; aKPt++)
       {
           Reset() ;
           int aK0 = mVK0K1.at(aKPt).x();
           int aK1 = mVK0K1.at(aKPt).y();

           for (int aK=aK0 ; aK<aK1 ; aK++)
               Add(1.0,mRVois[aK]);
           UpdateMax(aVMax,Compute());
           if (aVMax> aThreshold)  
              return aVMax;
       }
    }

    return aVMax;
}

template <class Type> cIm2D<Type> cFilterDCT<Type>::ComputeIm()
{
    cIm2D<Type> aImRes(mSz,nullptr,eModeInitImage::eMIA_V1);
    cDataIm2D<Type> & aDRes = aImRes.DIm();

    int aD = round_up(mR1)+1;

    for (const auto & aPix : cRect2(mDIm.Dilate(-aD)))
    {
        aDRes.SetV(aPix,ComputeVal(ToR(aPix)));
    }


    return aImRes;
}

template <class Type> cIm2D<Type> cFilterDCT<Type>::ComputeImMaxCrown(const double& aThreshold)
{
    cIm2D<Type> aImRes(mSz,nullptr,eModeInitImage::eMIA_V1);
    cDataIm2D<Type> & aDRes = aImRes.DIm();

    int aD = round_up(mR1)+1;

    for (const auto & aPix : cRect2(mDIm.Dilate(-aD)))
    {
        aDRes.SetV(aPix,ComputeValMaxCrown(ToR(aPix),aThreshold));
    }

    return aImRes;
}

template <class Type>  eDCTFilters cFilterDCT<Type>::ModeF() const {return mModeF;}

/* ================================================= */
/*                                                   */
/*               cSymFilterCT                        */
/*                                                   */
/* ================================================= */

/** Class for defining the symetry filter 

    See  cSymMeasure  in "include/MMVII_Matrix.h"
*/

template <class Type>  class  cSymFilterCT : public cFilterDCT<Type>
{
    public :
           typedef cIm2D<Type>     tIm;
           typedef cDataIm2D<Type> tDIm;

           cSymFilterCT(tIm anIm,double aR0,double aR1,double aEpsilon);

    private  :
          /// method used when a new neighboor arrive
          void Add(const Type & aWeight,const cPt2dr & aNeigh)   override;
          /// compute the value once a pixel is arrived
          double Compute()           override;
          /// Reset the computation once we begin a new pixel
          void Reset()               override;

          cSymMeasure<Type> SM() const;
          double mEpsilon ; ///< used to not devide by zero when constant distribution (max of stddev an Eps)
          cSymMeasure<Type> mSM; ///<  measure the symetry
};

template <class Type>  void  cSymFilterCT<Type>::Add(const Type & aW,const cPt2dr & aNeigh)
{
     Type aV1 = this->mDIm.GetVBL(this->mCurC+aNeigh);  // compute value of Neigh
     Type aV2 = this->mDIm.GetVBL(this->mCurC-aNeigh);  // compute value of its symetric
     mSM.Add(aW,aV1,aV2);   // Add thi pair of value to the symetric
}


template <class Type>  void  cSymFilterCT<Type>::Reset()
{
          mSM = cSymMeasure<Type>();  // reinitialise the symetry object
}

template <class Type>  double  cSymFilterCT<Type>::Compute() 
{
     return mSM.Sym(mEpsilon);  
}

template <class Type>  cSymFilterCT<Type>::cSymFilterCT(tIm anIm,double aR0,double aR1,double aEpsilon) :
    cFilterDCT<Type>
    (
           true,  // I am efficient in cumulation
           eDCTFilters::eSym,  // this is my nature
           anIm,   // first (an only here) image I am workin
           true,   // I am symetruc
           aR0,    // minimal ray of crown
           aR1     // max ray of crown
    ),
    mEpsilon (aEpsilon)
{
}

template <class Type>  cSymMeasure<Type> cSymFilterCT<Type>::SM() const
{
  return mSM;
}


/*
template<class TypeEl> cIm2D<TypeEl> ImScalab(const  cDataIm2D<TypeEl> & aDImIn,double aR0,double Epsilon)
{
    std::vector<cPt2di>  aVectVois = SortedVectOfRadius(aR0,aR1,true);

    // aVectVois = GetPts_Circle(cPt2dr(0,0),aR0,true);  StdOut() << "SYMMMM\n";

    int aD = round_up(aR1);
    cPt2di aPW(aD,aD);

    cPt2di aSz = aDImIn.Sz();
    cIm2D<TypeEl> aImOut(aSz,nullptr,eModeInitImage::eMIA_V1);
    cDataIm2D<TypeEl> & aDImOut = aImOut.DIm();

    for (const auto & aP : cRect2(aPW,aSz-aPW))
    {
          cSymMeasure<float> aSM;
          for (const auto & aV  : aVectVois)
	  {
		  TypeEl aV1 = aDImIn.GetV(aP+aV);
		  TypeEl aV2 = aDImIn.GetV(aP-aV);
		  aSM.Add(aV1,aV2);
	  }
	  aDImOut.SetV(aP,aSM.Sym(Epsilon));
    }

    return aImOut;
}
*/



/* ================================================== */
/*                                                    */
/*               BINARITY                             */
/*                                                    */
/* ================================================== */

template <class Type>  class  cBinFilterCT : public cFilterDCT<Type>
{
    public :
           typedef cIm2D<Type>     tIm;
           typedef cDataIm2D<Type> tDIm;

           cBinFilterCT(tIm anIm,double aR0,double aR1);

           void UpdateSelected(cNS_CodedTarget::cDCT & aDC) const override;

    private  :
          void Reset()               override;
          void Add(const Type & aWeight,const cPt2dr & aNeigh)   override;
          double Compute()           override;

          std::vector<float> mVVal;
          float mVBlack ;
          float mVWhite ;
};

template<class Type> void  cBinFilterCT<Type>::UpdateSelected(cNS_CodedTarget::cDCT & aDC) const 
{
    aDC.mVBlack = mVBlack;
    aDC.mVWhite = mVWhite;
}

template<class Type> void cBinFilterCT<Type>::Add(const Type & aWeight,const cPt2dr & aNeigh) 
{
     mVVal.push_back(this->mDIm.GetV(ToI(this->mCurC+aNeigh)));
}

template<class Type> void cBinFilterCT<Type>::Reset()
{
     mVVal.clear();
}

template<class Type> double  cBinFilterCT<Type>::Compute()
{
    double aPRop = 0.15;
    std::sort(mVVal.begin(),mVVal.end());
    int aNb = mVVal.size();

    int aIndBlack = round_ni(aPRop * aNb);
    int aIndWhite =  aNb-1- aIndBlack;
    mVBlack = mVVal[aIndBlack];
    mVWhite = mVVal[aIndWhite];
    float aVThreshold = (mVBlack+mVWhite)/2.0;

    int aNbBlack = 0;
    double aSumDifBlack = 0;
    double aSumDifWhite = 0;
    for (const auto & aV : mVVal)
    {
        bool IsBLack = (aV <aVThreshold ) ;
        float aVRef = (IsBLack ) ?  mVBlack : mVWhite;
        float aDif = std::abs(aVRef-aV);
        if (IsBLack)
        {
            aNbBlack++;
            aSumDifBlack += aDif;
        }
        else
        {
            aSumDifWhite += aDif;
        }
    }
    double aValue=1.0;
    int aNbWhite = aNb -aNbBlack;
    if ((aNbBlack!=0) && (aNbWhite!= 0)  && (mVBlack!=mVWhite))
    {
        double aDifMoy  =  aSumDifBlack/ aNbBlack + aSumDifWhite/aNbWhite;
        aValue = aDifMoy  / (mVWhite - mVBlack);
    }
	  
    return aValue;
}

template <class Type>  cBinFilterCT<Type>::cBinFilterCT(tIm anIm,double aR0,double aR1) :
    cFilterDCT<Type>(false,eDCTFilters::eBin,anIm,false,aR0,aR1)
{
}

template<class TypeEl> cIm2D<TypeEl> ImBinarity(cIm2D<TypeEl>  aImIn,double aR0,double aR1)
{
    cBinFilterCT<TypeEl> aBinF(aImIn,aR0,aR1);
    return  aBinF.ComputeIm();
}

/* ================================================== */
/*                                                    */
/*               STARITY                              */
/*                                                    */
/* ================================================== */

template <class Type>  class  cRadFilterCT : public cFilterDCT<Type>
{
    public :
           typedef cIm2D<Type>     tIm;
           typedef cDataIm2D<Type> tDIm;
           typedef cImGrad<Type>   tImGr;

           cRadFilterCT(const  tImGr&,double aR0,double aR1,double aEpsilon);

    private  :
          void Reset()               override;
          void Add(const Type & aWeight,const cPt2dr & aNeigh)   override;
          double Compute()           override;

	  tImGr                mGrad;
	  tDIm &               mGX;
	  tDIm &               mGY;
          double               mEpsilon ;
          double               mSomScal ;
          double               mSomG2;
};

template <class Type>  cRadFilterCT<Type>::cRadFilterCT(const  tImGr& aImGr ,double aR0,double aR1,double aEpsilon) :
    cFilterDCT<Type>(true,eDCTFilters::eRad,aImGr.mGx,false,aR0,aR1),
    mGrad    (aImGr),
    mGX      (mGrad.mGx.DIm()),
    mGY      (mGrad.mGy.DIm()),
    mEpsilon (aEpsilon)
{
}

template <class Type>  void  cRadFilterCT<Type>::Reset()
{
    mSomScal = 0;
    mSomG2  = 0;
} 

template <class Type> void cRadFilterCT<Type>::Add(const Type & aWeight,const cPt2dr & aNeigh)   
{
    cPt2dr aDir = VUnit(aNeigh);
    cPt2dr aPt = this->mCurC + aNeigh;

    cPt2dr aGrad(mGX.GetVBL(aPt),mGY.GetVBL(aPt));
    mSomScal += aWeight*Square(Scal(aGrad,aDir));
    mSomG2   += aWeight*SqN2(aGrad);
}

template <class Type> double cRadFilterCT<Type>::Compute()
{
    return std::max(mSomScal,mEpsilon) / std::max(mSomG2,mEpsilon);
}



/** This filter caracetrize how an image is symetric arround  each pixel; cpmpute som diff arround
 * oposite pixel, normamlized by standard deviation (so contrast invriant)
*/

/*
class  cSymetricityCalc
{
      public :
          cSymetricityCalc(double aR0,double aR1,double Epsilon);
      protected :
};
*/

       /* ================================================== */
       /*               SYMETRY                              */
       /* ================================================== */

template<class TypeEl> cIm2D<TypeEl> ImSymMin(cIm2D<TypeEl>  aImIn,double aR0,double aR1,double Epsilon)
{
    cSymFilterCT<TypeEl> aSymF(aImIn,aR0,aR1,Epsilon);
    return  aSymF.ComputeImMaxCrown(10);
}


template<class TypeEl> void CheckSymetricity
                           (
                                cDataIm2D<TypeEl> & aDIm2Check,
                                cIm2D<TypeEl>  aImIn,
                                double aR0,
                                double aR1,
                                double Epsilon
                           )
{     
    cDataIm2D<TypeEl> & aDImIn = aImIn.DIm();
    cPt2di aSz = aDImIn.Sz();
    int aD = round_up(aR1);
    cPt2di aPW(aD+1,aD+1);

    StdOut() << "Begin cmp  low/high level\n";
    {
       cSymFilterCT<TypeEl> aSymF(aImIn,aR0,aR1,Epsilon);
       cIm2D<TypeEl>  anI2 = aSymF.ComputeIm();
       cDataIm2D<TypeEl>& aDI2 = anI2.DIm();
       for (const auto & aPix : cRect2(aPW,aSz-aPW))
       {
           TypeEl aV1 = aDIm2Check.GetV(aPix);
           TypeEl aV2 = aDI2.GetV(aPix);
           if (std::abs(aV1-aV2) > 1e-5)
           {
               StdOut() << "Diiiff = " <<aV1 -  aV2  << " PIX= " << aPix << "\n";
               // getchar();
           }
       }
    }
    StdOut() << "end computation sym\n";
}

template<class TypeEl> cIm2D<TypeEl> ImSymetricity(bool doCheck,cIm2D<TypeEl>  aImIn,double aR0,double aR1,double Epsilon)
{
    cDataIm2D<TypeEl> & aDImIn = aImIn.DIm();
    const TypeEl* aData = aDImIn.RawDataLin();

    std::vector<int>  aVNeighLine;
    for (const auto &  aPix :VectOfRadius(aR0,aR1,true))
       aVNeighLine.push_back(aDImIn.IndexeLinear(aPix));


    int aD = round_up(aR1);
    cPt2di aPW(aD+1,aD+1);

    cPt2di aSz = aDImIn.Sz();
    cIm2D<TypeEl> aImOut(aSz,nullptr,eModeInitImage::eMIA_V1);
    cDataIm2D<TypeEl> & aDImOut = aImOut.DIm();

    StdOut() << "Begin computation low level\n";
    for (const auto & aPix : cRect2(aPW,aSz-aPW))
    {
          cSymMeasure<float> aSM;
          int aInd = aDImIn.IndexeLinear(aPix);

          for (const auto & aNI  : aVNeighLine)
	  {
		  aSM.Add(aData[aInd+aNI],aData[aInd-aNI]);
          }

          double aVal = aSM.Sym(Epsilon);
	  aDImOut.SetV(aPix,aVal);
    }
    StdOut() << "End computation sym low level\n";
    if (doCheck)
    {
       CheckSymetricity(aDImOut,aImIn,aR0,aR1,Epsilon);
    }

    return aImOut;
}



/* ======================================== */
/*           ALLOCATOR                      */
/* ======================================== */

template <class Type>  
    cFilterDCT<Type> * cFilterDCT<Type>::AllocSym(tIm anIm,double aR0,double aR1,double aEpsilon)
{
    return new  cSymFilterCT<Type>(anIm,aR0,aR1,aEpsilon);
}

template <class Type>  
    cFilterDCT<Type> * cFilterDCT<Type>::AllocBin(tIm anIm,double aR0,double aR1)
{
    return new  cBinFilterCT<Type>(anIm,aR0,aR1);
}

template <class Type>  
    cFilterDCT<Type> * cFilterDCT<Type>::AllocRad(const tImGr & aImGr,double aR0,double aR1,double aEpsilon)
{
    return new  cRadFilterCT<Type>(aImGr,aR0,aR1,aEpsilon);
}



#define INSTANTIATE_FILTER_TARGET(TYPE)\
template cIm2D<TYPE> ImBinarity(cIm2D<TYPE>  aDIm,double aR0,double aR1);\
template cIm2D<TYPE> ImSymetricity(bool doCheck,cIm2D<TYPE>  aDImIn,double aR0,double aR1,double Epsilon);\
template cIm2D<TYPE> ImSymMin(cIm2D<TYPE>  aImIn,double aR0,double aR1,double Epsilon);

INSTANTIATE_FILTER_TARGET(tREAL4)

template class  cRadFilterCT<tREAL4>;
template class  cBinFilterCT<tREAL4>;
template class  cSymFilterCT<tREAL4>;
template class  cFilterDCT<tREAL4>;




};
