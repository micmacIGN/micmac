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
          bool        isCumul,
          eDCTFilters aModeF,
          tIm anIm,
          bool IsSym,
          double aR0,
          double aR1,
          double aThickN
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
      mIVois   (SortedVectOfRadius(aR0,aR1,IsSym))
{
   for (const auto & aPix : mIVois)
       mRVois.push_back(ToR(aPix));

   mRhoEnd = Norm2(mRVois.back());

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
       // StdOut() << "KKKK " << aK0 << " " << aK1 << " \n";
   }

   // StdOut() << "LLLLLLLLLLLLLl \n";
   // getchar();
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
	/*
static int aCpt=0; aCpt ++;
StdOut() <<  "CCCC  " << aCpt <<  " Cum " << mIsCumul << "\n";
BUGF = (aCpt==  9137);
*/

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

template <class Type>  class  cSymFilterCT : public cFilterDCT<Type>
{
    public :
           typedef cIm2D<Type>     tIm;
           typedef cDataIm2D<Type> tDIm;

           cSymFilterCT(tIm anIm,double aR0,double aR1,double aEpsilon);

    private  :
          void Reset()               override;
          void Add(const Type & aWeight,const cPt2dr & aNeigh)   override;
          double Compute()           override;

          cSymMeasure<Type> SM() const;
          double mEpsilon ;
          cSymMeasure<Type> mSM;
};


template <class Type>  void  cSymFilterCT<Type>::Reset()
{
          mSM = cSymMeasure<Type>();
}

template <class Type>  void  cSymFilterCT<Type>::Add(const Type & aW,const cPt2dr & aNeigh)
{
     Type aV1 = this->mDIm.GetVBL(this->mCurC+aNeigh);
     Type aV2 = this->mDIm.GetVBL(this->mCurC-aNeigh);
     mSM.Add(aW,aV1,aV2);
}
template <class Type>  double  cSymFilterCT<Type>::Compute() 
{
     return mSM.Sym(mEpsilon);
}

template <class Type>  cSymFilterCT<Type>::cSymFilterCT(tIm anIm,double aR0,double aR1,double aEpsilon) :
    cFilterDCT<Type>(true,eDCTFilters::eSym,anIm,true,aR0,aR1),
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



       /* ================================================== */
       /*               SCALITY                              */
       /* ================================================== */

/*
template <class Type>  class cComputeScaleInv
{
       public :
           cComputeScaleInv(cIm2D<Type> aIm1,double aScale,double aR0,double aR1);
       private :
           
           std::vector<cPt2di>  mVois;
};


template<class TypeEl> double ScalInv
                              (
                                  const  cDataIm2D<TypeEl> & aDIm1,
                                  const  cDataIm2D<TypeEl> & aDIm2,
                                  const  cPt2dr & aPt,
                                  double aScale,
                                  const std::vector<cPt2di>  & aV0,
                                  double aR0,
                                  double Epsilon
                              )
{
    cMatIner2Var<double> aMat;
    double aMinCorrel = 1e-4;
    for (const auto & aVInt  : aVectVois)
    {
           cPt2dr aV1 = ToR(aVInt);
           cPt2dr aV2 = aV1 * aScale ;
           TypeEl aVal1 = aDIm1.GetVBL(aPt+aV1);
           TypeEl aVal2 = aDIm2.GetVBL(aPt+aV2);
    }
}
*/

template <class Type>  class cExtractDir
{
     public :
         typedef cIm2D<Type>     tIm;
         typedef cDataIm2D<Type> tDIm;
	 typedef cNS_CodedTarget::cDCT tDCT;
         typedef std::vector<cPt2dr> tVDir;

         cExtractDir(tIm anIm,double aRho0,double aRho1,double aRh0Init);
         bool  DoExtract(tDCT &) ;
     public :

	 double Score(const cPt2dr & ,double aTeta); 
	 cPt2dr OptimScore(const cPt2dr & ,double aStepTeta); 


          tIm     mIm;
          tDIm&   mDIm;
          float   mRho0;
          float   mRho1;
          float   mRho0Init;

	  tResFlux                mPtsCrown;
          std::vector<tResFlux>   mVCircles;
          std::vector<tVDir>      mVDIrC;
          float                   mVThrs ;
	  tDCT *                  mPDCT;
       // (SortedVectOfRadius(aR0,aR1,IsSym))
};

template <class Type>  cExtractDir<Type>::cExtractDir(tIm anIm,double aRho0,double aRho1,double aRho0Init) :
     mIm        (anIm),
     mDIm       (mIm.DIm()),
     mRho0      (aRho0),
     mRho1      (aRho1),
     mRho0Init  (aRho0Init),
     mPtsCrown  (SortedVectOfRadius(mRho0,mRho1))
{
    for (double aRho = aRho0Init ; aRho<aRho1 ; aRho++)
    {
         mVCircles.push_back(GetPts_Circle(cPt2dr(0,0),aRho,true));
         mVDIrC.push_back(tVDir());

         for (const auto& aPix :  mVCircles.back())
         {
               mVDIrC.back().push_back(VUnit(ToR(aPix)));
         }
    }
}

template <class Type>  double cExtractDir<Type>::Score(const cPt2dr & aDInit,double aDeltaTeta) 
{
    cPt2dr aDir = aDInit * FromPolar(1.0,aDeltaTeta);
    double aSomDiff = 0.0;
    double aStepRho = 1.0;
    int aNb=0;

    for (double aRho =mRho0 ; aRho<mRho1; aRho+=aStepRho)
    {
         float aV1 = mDIm.GetVBL(mPDCT->mPt+aRho* aDir);
         float aV2 = mDIm.GetVBL(mPDCT->mPt-aRho* aDir);

	 aSomDiff += std::abs(aV1-mVThrs) + std::abs(aV2-mVThrs);
	 aNb += 2;
    }

    return aSomDiff /aNb;
}

template <class Type>  cPt2dr cExtractDir<Type>::OptimScore(const cPt2dr & aDir,double aStepTeta)
{
    cWhitchMin<int,double>  aWMin(0,1e10);

    for (int aK=-1; aK<=1 ; aK++)
        aWMin.Add(aK,Score(aDir,aStepTeta*aK));

    aStepTeta *= aWMin.IndexExtre();

    if (aStepTeta==0) return aDir;

    double aScore = aWMin.ValExtre();
    double aScorePrec = 2*aScore;
    int aKTeta = 1;

    while(aScore < aScorePrec)
    {
	 aScorePrec = aScore;
	 aScore= Score(aDir,aStepTeta*(aKTeta+1));
	 aKTeta++;
    }
    aKTeta--;

    return aDir * FromPolar(1.0,aKTeta*aStepTeta);

}

double TestDir(const cNS_CodedTarget::cGeomSimDCT & aGT,const cNS_CodedTarget::cDCT  &aDCT)
{
    cPt2dr anEl1 = VUnit(aGT.mCornEl1-aGT.mC) ;
    cPt2dr anEl2 = VUnit(aGT.mCornEl2-aGT.mC) ;

    //StdOut()<<  (anEl1^anEl2) <<  " " <<  (aDCT.mDirC1 ^aDCT.mDirC2) << "\n";

    if (Scal(anEl2,aDCT.mDirC2) < 0)
    {
        anEl1 = - anEl1;
        anEl2 = - anEl2;
    }

    cPt2dr aD1 = anEl1 / aDCT.mDirC1;
    cPt2dr aD2 = anEl2 / aDCT.mDirC2;


    double aSc =  (std::abs(ToPolar(aD1).y()) + std::abs( ToPolar(aD2).y())) /  2.0 ;

    return aSc;
}


template <class Type>  bool cExtractDir<Type>::DoExtract(cNS_CodedTarget::cDCT & aDCT) 
{
     mPDCT = & aDCT;
     std::vector<float>  aVVals;
     std::vector<bool>   aVIsW;
     cPt2di aC= ToI(aDCT.mPt);
     mVThrs = (aDCT.mVBlack+aDCT.mVWhite)/2.0;

     
     cPt2dr aSomDir[2] = {{0,0},{0,0}};

     for (int aKC=0 ; aKC<int(mVCircles.size()) ; aKC++)
     {
         const auto & aCircle  = mVCircles[aKC];
         const auto & aVDir    = mVDIrC[aKC];
         int aNbInC = aCircle.size();
         aVVals.clear();
         aVIsW.clear();
         for (const auto & aPt : aCircle)
         {
             float aVal  = mDIm.GetV(aC+aPt);
             aVVals.push_back(aVal);
             aVIsW.push_back(aVal>mVThrs);
         }
         int aCpt = 0;
         for (int  aKp=0 ; aKp<aNbInC ; aKp++)
         {
             int aKp1 = (aKp+1)%aNbInC;
             if (aVIsW[aKp] != aVIsW[aKp1])
             {
                 aCpt++;
                 cPt2dr aP1  = aVDir[aKp];
                 cPt2dr aP2  = aVDir[aKp1];
                 double aV1 = aVVals[aKp];
                 double aV2 = aVVals[aKp1];
                 cPt2dr aDir =   (aP1 *(aV2-mVThrs) + aP2 * (mVThrs-aV1)) / (aV2-aV1);
                 if (SqN2(aDir)==0) return false;
                 aDir = VUnit(aDir);
                 aDir = aDir * aDir;  // make a tensor of it => double its angle
                 aSomDir[aVIsW[aKp]] += aDir;
             }
         }
         if (aCpt!=4 )  return false;
     }

     for (auto & aDir : aSomDir)
     {
         aDir = ToPolar(aDir,0.0);
         aDir = FromPolar(1.0,aDir.y()/2.0);
     }
     aDCT.mDirC1 = aSomDir[1];
     aDCT.mDirC2 = aSomDir[0];

     // As each directio is up to Pi, make it oriented
     if ( (aDCT.mDirC1^aDCT.mDirC2) < 0)
     {
         aDCT.mDirC2 = -aDCT.mDirC2;
     }

     aDCT.mDirC1 =  OptimScore(aDCT.mDirC1,1e-3);
     aDCT.mDirC2 =  OptimScore(aDCT.mDirC2,1e-3);

     cAffin2D  aInit2Loc(aDCT.mPt,aDCT.mDirC1,aDCT.mDirC2);
     cAffin2D  aLoc2Init = aInit2Loc.MapInverse();

     cSegment2DCompiled aSeg1(aDCT.mPt,aDCT.mDirC1);
     cSegment2DCompiled aSeg2(aDCT.mPt,aDCT.mDirC2);

     FakeUseIt(aLoc2Init);


     double aSomWeight = 0.0;
     double aSomWEc     = 0.0;
     for (const auto & aPCr : mPtsCrown)
     {
          cPt2di aIPix = aPCr+aDCT.Pix();
          cPt2dr aRPix = ToR(aIPix);
          cPt2dr aRPixInit = aLoc2Init.Value(aRPix);

	  float aVal = mDIm.GetV(aIPix);
	  bool isW = ((aRPixInit.x()>=0) != (aRPixInit.y()>=0) );
	  float aValTheo = isW ?  aDCT.mVWhite : aDCT.mVBlack;

	  double aWeight = 1.0;
	  double aD1 =  aSeg1.Dist(aRPix);
	  double aD2 =  aSeg2.Dist(aRPix);
	  aWeight = std::min(   std::min(aD1,1.0), std::min(aD2,1.0));

	  aSomWeight += aWeight;
	  aSomWEc +=  aWeight * std::abs(aValTheo-aVal);
     }

     aSomWEc /= aSomWeight;
     double aDev = aDCT.mVWhite - aDCT.mVBlack;


     if (aDCT.mGT)
        StdOut() << "Difff=" <<    aSomWEc / aDev  << "   "  << (aDCT.mGT ? "++" : "--") << "\n"; 
     if (aDCT.mGT)
     {
	     /*
        double aSc1 = TestDir(*aDCT.mGT,aDCT);
        double aSc2 = TestDir(*aDCT.mGT,aDCT);
        StdOut() << " * ScDirs=  " << aSc1  << " " << aSc2 << "\n";
     */
     }
     return true;
}


template class cExtractDir<tREAL4>;

bool TestDirDCT(cNS_CodedTarget::cDCT & aDCT,cIm2D<tREAL4> anIm)
{
    cExtractDir<tREAL4>  anED(anIm,3,8,5);
    bool Ok = anED.DoExtract(aDCT);

    return Ok;

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
