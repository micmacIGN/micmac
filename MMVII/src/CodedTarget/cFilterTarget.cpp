#include "CodedTarget.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "include/MMVII_Tpl_Images.h"


namespace MMVII
{

/* ================================================= */
/*               cFilterDCT                          */
/* ================================================= */

template <class Type>  
   cFilterDCT<Type>::cFilterDCT
   (
          bool        isCumul,
          eDCTFilters aTypeF,
          tIm anIm,
          bool IsSym,
          double aR0,
          double aR1,
          eDCTFilters aMode,
          double aThickN
   ) :
      mIsCumul (isCumul),
      mTypeF   (aTypeF),
      mIm      (anIm),
      mDIm     (anIm.DIm()),
      mSz      (mDIm.Sz()),
      mIsSym   (IsSym),
      mR0      (aR0),
      mR1      (aR1),
      mThickN  (aThickN),
      mMode    (aMode),
      mIVois   (SortedVectOfRadius(aR0,aR1,IsSym))
{
   for (const auto & aPix : mIVois)
       mRVois.push_back(ToR(aPix));

   mRhoEnd = Norm2(mRVois.back());

   int aK0=0;
   int aK1=aK0;
   IncrK1(aK1,aK0);
   mVK0K1.push_back(cPt2di(0,0));
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

template <class Type> double cFilterDCT<Type>::ComputeValMaxCrown(const cPt2dr & aP,const double & aThreshold)
{
    mCurC = aP;
    Reset() ;

    double aVMax = -1;
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
        if (! mIsCumul)
           Reset();
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

template <class Type>  eDCTFilters cFilterDCT<Type>::TypeF() const {return mTypeF;}

/* ================================================= */
/*               cSymFilterCT                        */
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
    cFilterDCT<Type>(true,eDCTFilters::eSym,anIm,true,aR0,aR1,eDCTFilters::eSym),
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
       /*               BINARITY                             */
       /* ================================================== */

template<class TypeEl> 
   double IndBinarity(const  cDataIm2D<TypeEl> & aDIm,const cPt2di & aP0,const std::vector<cPt2di> & aVectVois)
{
    double aPRop = 0.15;
    std::vector<float> aVVal;
    for (int aKV=0; aKV<int(aVectVois.size()) ; aKV++)
    {
         cPt2di aPV = aP0 + aVectVois[aKV];
         aVVal.push_back(aDIm.GetV(aPV));
    }
    std::sort(aVVal.begin(),aVVal.end());
    int aNb = aVVal.size();

    int aIndBlack = round_ni(aPRop * aNb);
    int aIndWhite =  aNb-1- aIndBlack;
    float aVBlack = aVVal[aIndBlack];
    float aVWhite = aVVal[aIndWhite];
    float aVThreshold = (aVBlack+aVWhite)/2.0;

    int aNbBlack = 0;
    double aSumDifBlack = 0;
    double aSumDifWhite = 0;
    for (const auto & aV : aVVal)
    {
              bool IsBLack = (aV <aVThreshold ) ;
              float aVRef = (IsBLack ) ?  aVBlack : aVWhite;
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
    if ((aNbBlack!=0) && (aNbWhite!= 0)  && (aVBlack!=aVWhite))
    {
              double aDifMoy  =  aSumDifBlack/ aNbBlack + aSumDifWhite/aNbWhite;
              aValue = aDifMoy  / (aVWhite - aVBlack);
    }
	  
    return aValue;
}

template<class TypeEl> cIm2D<TypeEl> ImBinarity(const  cDataIm2D<TypeEl> & aDIm,double aR0,double aR1,double Epsilon)
{
    std::vector<cPt2di>  aVectVois = VectOfRadius(aR0,aR1,false);
    // aVectVois = GetPts_Circle(cPt2dr(0,0),aR0,true);  

    int aD = round_up(aR1);
    cPt2di aPW(aD,aD);


    cPt2di aSz = aDIm.Sz();
    cIm2D<TypeEl> aImOut(aSz,nullptr,eModeInitImage::eMIA_V1);
    cDataIm2D<TypeEl> & aDImOut = aImOut.DIm();

    // double aPRop = 0.15;
    for (const auto & aP0 : cRect2(aPW,aSz-aPW))
    {
        double aValue = IndBinarity(aDIm,aP0,aVectVois);
        aDImOut.SetV(aP0,aValue);
    }

    return aImOut;
}

       /* ================================================== */
       /*               STARITY                              */
       /* ================================================== */

std::vector<cPt2dr> VecDir(const  std::vector<cPt2di>&  aVectVois)
{
    std::vector<cPt2dr>  aVecDir;
    for (const auto & aPV : aVectVois)
    {
           aVecDir.push_back(VUnit(ToR(aPV)));
    }
    return aVecDir;
}


template<class TypeEl> double Starity
                              (
                                  const  cImGrad<TypeEl> & aImGrad,
                                  const cPt2dr & aP0,
                                  const  std::vector<cPt2di>&  aVectVois ,
                                  const  std::vector<cPt2dr>&  aVecDir,
                                  double Epsilon
                              )
{
    const cDataIm2D<TypeEl> & aIGx = aImGrad.mGx.DIm();
    const cDataIm2D<TypeEl> & aIGy = aImGrad.mGy.DIm();

    double aSomScal = 0;
    double aSomG2= 0;
    for (int aKV=0; aKV<int(aVectVois.size()) ; aKV++)
    {
         cPt2dr aVR =  ToR(aVectVois[aKV]);
         cPt2dr aPV = aP0 + aVR;
         cPt2dr aGrad(aIGx.GetVBL(aPV),aIGy.GetVBL(aPV));

         aSomScal += Square(Scal(aGrad,aVecDir[aKV]));
         aSomG2 += SqN2(aGrad);

    }
    double aValue = (aSomScal+ Epsilon) / (aSomG2+Epsilon);
 
    return aValue;
}


template<class TypeEl> cIm2D<TypeEl> ImStarity(const  cImGrad<TypeEl> & aImGrad,double aR0,double aR1,double Epsilon)
{
    std::vector<cPt2di>  aVectVois = VectOfRadius(aR0,aR1,false);
    // aVectVois = GetPts_Circle(cPt2dr(0,0),aR0,true);  
    std::vector<cPt2dr>  aVecDir = VecDir(aVectVois);

    int aDist = round_up(aR1)+2;
    cPt2di aPW(aDist,aDist);

    const cDataIm2D<TypeEl> & aIGx = aImGrad.mGx.DIm();
    // const cDataIm2D<TypeEl> & aIGy = aImGrad.mGy.DIm();

    cPt2di aSz = aIGx.Sz();
    cIm2D<TypeEl> aImOut(aSz,nullptr,eModeInitImage::eMIA_V1);
    cDataIm2D<TypeEl> & aDImOut = aImOut.DIm();

    for (const auto & aP0 : cRect2(aPW,aSz-aPW))
    {
         double aValue = Starity(aImGrad,ToR(aP0),aVectVois,aVecDir,Epsilon);
/*
          double aSomScal = 0;
          double aSomG2= 0;
          for (int aKV=0; aKV<int(aVectVois.size()) ; aKV++)
	  {
		  cPt2di aPV = aP0 + aVectVois[aKV];
		  cPt2dr aGrad(aIGx.GetV(aPV),aIGy.GetV(aPV));

		  aSomScal += Square(Scal(aGrad,aVecDir[aKV]));
		  aSomG2 += SqN2(aGrad);

		  // StdOut () << " GR=" << aGrad << " S=" << Square(Scal(aGrad,aVecDir[aKV])) << " G2=" << Norm2(aGrad) << "\n";
	  }
	  double aValue = (aSomScal+ Epsilon) / (aSomG2+Epsilon);
*/
	  aDImOut.SetV(aP0,aValue);
    }

    return aImOut;
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

template <class Type>  class cComputeScaleInv
{
       public :
           cComputeScaleInv(cIm2D<Type> aIm1,double aScale,double aR0,double aR1);
       private :
           
           std::vector<cPt2di>  mVois;
};


/*
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

/* ======================================== */
/*           ALLOCATOR                      */
/* ======================================== */

template <class Type>  
    cFilterDCT<Type> * cFilterDCT<Type>::AllocSym(tIm anIm,double aR0,double aR1,double aEpsilon)
{
    return new  cSymFilterCT<Type>(anIm,aR0,aR1,aEpsilon);
}

#define INSTANTIATE_FILTER_TARGET(TYPE)\
template double Starity(const cImGrad<TYPE> &,const cPt2dr &,const std::vector<cPt2di>&,const  std::vector<cPt2dr>&,double Epsilon);\
template cIm2D<TYPE> ImBinarity(const  cDataIm2D<TYPE> & aDIm,double aR0,double aR1,double Epsilon);\
template cIm2D<TYPE> ImStarity(const  cImGrad<TYPE> & aImGrad,double aR0,double aR1,double Epsilon);\
template cIm2D<TYPE> ImSymetricity(bool doCheck,cIm2D<TYPE>  aDImIn,double aR0,double aR1,double Epsilon);\
template cIm2D<TYPE> ImSymMin(cIm2D<TYPE>  aImIn,double aR0,double aR1,double Epsilon);

INSTANTIATE_FILTER_TARGET(tREAL4)
template class  cSymFilterCT<tREAL4>;
template class  cFilterDCT<tREAL4>;


};
