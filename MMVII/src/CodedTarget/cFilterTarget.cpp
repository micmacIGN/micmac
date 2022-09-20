#include "CodedTarget.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "include/MMVII_Tpl_Images.h"
// #include "../include/MMVII_2Include_Serial_Tpl.h"


namespace MMVII
{



/* =================================== */
/*            cParam1FilterDCT         */
/* =================================== */

void cParam1FilterDCT::AddData(const cAuxAr2007 & anAux)
{
     MMVII::AddData(cAuxAr2007("R0",anAux),mR0);
     MMVII::AddData(cAuxAr2007("R1",anAux),mR1);
     MMVII::AddData(cAuxAr2007("ThickN",anAux),mThickN);
}

cParam1FilterDCT::cParam1FilterDCT() :
    mR0     (0.4),
    mR1     (0.8),
    mThickN (1.5)
{
}
double   cParam1FilterDCT::R0() const {return mR0;}
double   cParam1FilterDCT::R1() const {return mR1;}
double   cParam1FilterDCT::ThickN() const {return mThickN;}


/* =================================== */
/*           cParamFilterDCT_Bin       */
/* =================================== */

bool cParamFilterDCT_Bin::IsSym() const {return false;}
bool cParamFilterDCT_Bin::IsCumul() const {return false;}
eDCTFilters cParamFilterDCT_Bin::ModeF() const {return eDCTFilters::eBin;}

cParamFilterDCT_Bin::cParamFilterDCT_Bin() :
     cParam1FilterDCT(),
     mPropBW (0.15)
{
}

void cParamFilterDCT_Bin::AddData(const cAuxAr2007 & anAux)
{
     cParam1FilterDCT::AddData(cAuxAr2007("Glob",anAux));
     MMVII::AddData(cAuxAr2007("PropBW",anAux),mPropBW);
}


/* =================================== */
/*           cParamFilterDCT_Sym       */
/* =================================== */

bool cParamFilterDCT_Sym::IsSym() const {return true;}
bool cParamFilterDCT_Sym::IsCumul() const {return true;}
eDCTFilters cParamFilterDCT_Sym::ModeF() const {return eDCTFilters::eSym;}

cParamFilterDCT_Sym::cParamFilterDCT_Sym() :
     cParam1FilterDCT(),
     mEpsilon (1.0)
{
}

double cParamFilterDCT_Sym::Epsilon() const {return mEpsilon;}
void cParamFilterDCT_Sym::AddData(const cAuxAr2007 & anAux)
{
     cParam1FilterDCT::AddData(cAuxAr2007("Glob",anAux));
}

/* =================================== */
/*           cParamFilterDCT_Rad       */
/* =================================== */

bool cParamFilterDCT_Rad::IsSym()   const {return false;}
bool cParamFilterDCT_Rad::IsCumul() const {return true;}
eDCTFilters cParamFilterDCT_Rad::ModeF() const {return eDCTFilters::eRad;}

cParamFilterDCT_Rad::cParamFilterDCT_Rad() :
     cParam1FilterDCT(),
     mEpsilon (2.0)
{
}

double cParamFilterDCT_Rad::Epsilon() const {return mEpsilon;}

void cParamFilterDCT_Rad::AddData(const cAuxAr2007 & anAux)
{
     cParam1FilterDCT::AddData(cAuxAr2007("Glob",anAux));
}

/* =================================== */
/*           cParamAllFilterDCT        */
/* =================================== */

cParamAllFilterDCT::cParamAllFilterDCT() :
	mRGlob (10.0)
{
}

void cParamAllFilterDCT::AddData(const cAuxAr2007 & anAux)
{
     mBin.AddData(cAuxAr2007("Bin",anAux));
     mSym.AddData(cAuxAr2007("Sym",anAux));
     mRad.AddData(cAuxAr2007("Rad",anAux));
     MMVII::AddData(cAuxAr2007("RGlob",anAux),mRGlob);
}
double   cParamAllFilterDCT::RGlob() const {return mRGlob;}

const cParamFilterDCT_Bin & cParamAllFilterDCT::Bin() const {return mBin;}
const cParamFilterDCT_Sym & cParamAllFilterDCT::Sym() const {return mSym;}
const cParamFilterDCT_Rad & cParamAllFilterDCT::Rad() const {return mRad;}

void AddData(const  cAuxAr2007 & anAux,cParamAllFilterDCT & aParam)
{
   aParam.AddData(anAux);
}

void TestParamTarg()
{
   cParamAllFilterDCT aParam;
   SaveInFile(aParam,"toto.xml");
}


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
      mIVois   (SortedVectOfRadius(aR0,aR1,mIsSym))   // sorted by ray neigboor, with o w/o symetric pixel
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

template <class Type>
   cFilterDCT<Type>::cFilterDCT(tIm anIm,const  cParamAllFilterDCT & aGlob,const cParam1FilterDCT * aSpecif) :
	   cFilterDCT<Type>
           (
                  aSpecif->IsCumul(),
                  aSpecif->ModeF(),   // type of filter
                  anIm,               // first image (give the size)
                  aSpecif->IsSym(),
                  aSpecif->R0() * aGlob.RGlob(),
                  aSpecif->R1() * aGlob.RGlob(),
                  aSpecif->ThickN()
           )
{
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
           cSymFilterCT(tIm anIm,const cParamAllFilterDCT &);

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
template <class Type>  cSymFilterCT<Type>::cSymFilterCT(tIm anIm,const cParamAllFilterDCT & aParam) :
    cFilterDCT<Type>(anIm,aParam,&(aParam.Sym())),
    mEpsilon (aParam.Sym().Epsilon())
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

           // cBinFilterCT(tIm anIm,double aR0,double aR1);
           cBinFilterCT(tIm anIm,const cParamAllFilterDCT &);

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


template <class Type>  cBinFilterCT<Type>::cBinFilterCT(tIm anIm,const cParamAllFilterDCT & aParam) :
    cFilterDCT<Type>(anIm,aParam,&(aParam.Bin()))
{
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

           cRadFilterCT(const  tImGr&,const cParamAllFilterDCT & aParam);
           // cRadFilterCT(const  tImGr&,double aR0,double aR1,double aEpsilon);

    private  :
          void Reset()               override;
          void Add(const Type & aWeight,const cPt2dr & aNeigh)   override;
          double Compute()           override;

	  const cParamFilterDCT_Rad* mParRad;
	  tImGr                mGrad;
	  tDIm &               mGX;
	  tDIm &               mGY;
          double               mSomScal ;
          double               mSomG2;
};

template <class Type>  cRadFilterCT<Type>::cRadFilterCT(const  tImGr& aImGr ,const cParamAllFilterDCT & aParam) :
    cFilterDCT<Type>(aImGr.mGx,aParam,&(aParam.Rad())),
    mParRad  (&(aParam.Rad())),
    mGrad    (aImGr),
    mGX      (mGrad.mGx.DIm()),
    mGY      (mGrad.mGy.DIm())
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
    return std::max(mSomScal,mParRad->Epsilon()) / std::max(mSomG2,mParRad->Epsilon());
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
    cFilterDCT<Type> * cFilterDCT<Type>::AllocSym(tIm anIm,const cParamAllFilterDCT & aParam)
{
    return new  cSymFilterCT<Type>(anIm,aParam);
}

template <class Type>
    cFilterDCT<Type> * cFilterDCT<Type>::AllocBin(tIm anIm,const cParamAllFilterDCT & aParam)
{
    return new  cBinFilterCT<Type>(anIm,aParam);
}

template <class Type>
    cFilterDCT<Type> * cFilterDCT<Type>::AllocRad(const tImGr & aImGr,const cParamAllFilterDCT & aParam)
{
    return new  cRadFilterCT<Type>(aImGr,aParam);
}


// template cIm2D<TYPE> ImBinarity(cIm2D<TYPE>  aDIm,double aR0,double aR1);

#define INSTANTIATE_FILTER_TARGET(TYPE)\
template cIm2D<TYPE> ImSymetricity(bool doCheck,cIm2D<TYPE>  aDImIn,double aR0,double aR1,double Epsilon);\
template cIm2D<TYPE> ImSymMin(cIm2D<TYPE>  aImIn,double aR0,double aR1,double Epsilon);

INSTANTIATE_FILTER_TARGET(tREAL4)

template class  cRadFilterCT<tREAL4>;
template class  cBinFilterCT<tREAL4>;
template class  cSymFilterCT<tREAL4>;
template class  cFilterDCT<tREAL4>;




};
