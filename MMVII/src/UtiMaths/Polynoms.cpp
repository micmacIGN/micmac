#include "MMVII_Matrix.h"
#include "MMVII_Ptxd.h"
#include "MMVII_Stringifier.h"
#include "MMVII_Geom2D.h"


#include "V1VII.h"

namespace MMVII
{

/**
    Class for extraction of roots of polynoms using "Companion matrix method"

  See  https://en.wikipedia.org/wiki/Companion_matrix
*/


template <class Type> class cEigenPolynRoots
{
    public :
       
        typedef cDenseMatrix<Type>             tMatComp;
        typedef cPtxd<Type,2>                  tCompl;

        cEigenPolynRoots(const cPolynom<Type> &,Type aEps,int aNbIterMax) ;
        bool  RootIsReal(const tCompl & ,std::string * sayWhy=nullptr);

        const std::vector<Type> &    RealRoots() const {return mRR;}
        const std::vector<tCompl> &  ComplexRoots() const {return mCR;}
        const tMatComp & CompM() const             {return mCompM;}

        tCompl  Refine(const tCompl & aV0,Type  aEps,int aNbIter) const;

    private :
        static Type PolRelAccuracy() ;
        static Type PolAbsAccuracy() ;
        static Type ComplRelAccuracy() ;
        static Type ComplAbsAccuracy() ;

        cPolynom<Type>     mPol;
        cPolynom<Type>     mDPol;
        size_t             mDeg;
        size_t             mSzMat;   /// to avoid size 0
        tMatComp           mCompM;   ///< companion matrix
        std::vector<Type>  mRR;      ///< Real roots
        std::vector<tCompl>  mCR;      ///< Real roots

};


template <class Type> cEigenPolynRoots<Type>::cEigenPolynRoots(const cPolynom<Type> & aPol,Type  aEps,int aNbIter)  :
    mPol   (aPol),
    mDPol  (aPol.Deriv()),
    mDeg   (mPol.Degree()),
    mSzMat (std::max((size_t)1,mDeg)),
    mCompM (mSzMat,mSzMat,eModeInitImage::eMIA_Null)
{
    if (mDeg==0)
       return;

    // fill the diagonal up the principal diag
    for (size_t aK = 0; aK < mDeg-1; ++aK) 
    {
        mCompM.SetElem(aK+1, aK,1); 
    }

    const Type & aHighCoeff = mPol[mDeg];
    // Fill last line with normalized coeff
    for (size_t aK = 0; aK < mDeg; ++aK) 
    {
        mCompM.SetElem(aK, mDeg - 1, -mPol[aK] / aHighCoeff);
    }

    cResulEigenDecomp<Type> aRED =  mCompM.Eigen_Decomposition() ;


    for (size_t aK = 0; aK < mDeg; ++aK) 
    {
        tCompl aCR(aRED.mEigenVal_R(aK),aRED.mEigenVal_I(aK));

        aCR = Refine(aCR,aEps*PolAbsAccuracy(),aNbIter);
        mCR.push_back(aCR);

        if (RootIsReal(aCR))
           mRR.push_back(aCR.x());
    }
    std::sort(mRR.begin(),mRR.end());
}

template <class Type> cPtxd<Type,2>  cEigenPolynRoots<Type>::Refine(const tCompl & aVal0,Type  aEps,int aNbIter) const
{
    Type aSqEps = Square(aEps);
    tCompl aLastVal = aVal0;
    tCompl aLastEval =   mPol.Value(aLastVal);
    Type   aLastSqN2 = SqN2(aLastEval);

    for (int aKIt=0  ; aKIt<aNbIter ; aKIt++)
    {
        tCompl aDeriv = mDPol.Value(aLastVal);
        if (IsNotNull(aDeriv))
        {
            tCompl aNewVal = aLastVal - aLastEval/aDeriv;
            tCompl aNewEval = mPol.Value(aNewVal);
            Type aNewSqN2 = SqN2(aNewEval);
            if (aNewSqN2 < aSqEps)
               return aNewVal;
            else if (aNewSqN2<aLastSqN2)
            {

                aLastVal = aNewVal;
                aLastEval = aNewEval;
                aLastSqN2 = aNewSqN2;
            }
            else
                return aLastVal;
        }
        else
            return aLastVal;
    }
    return aLastVal;
}

    //     tCompl  Refine(const tCompl & aV0,int aNbIter=5);



/**  Also the question seems pretty basic, it becomes more complicated due to numericall approximation */

template <>  tREAL4   cEigenPolynRoots<tREAL4>::PolRelAccuracy() {return 1e-5;}
template <>  tREAL8   cEigenPolynRoots<tREAL8>::PolRelAccuracy() {return 1e-9;}
template <>  tREAL16  cEigenPolynRoots<tREAL16>::PolRelAccuracy(){return 1e-11;}

template <>  tREAL4   cEigenPolynRoots<tREAL4>::ComplRelAccuracy()  {return 1e-8;}
template <>  tREAL8   cEigenPolynRoots<tREAL8>::ComplRelAccuracy()  {return 1e-8;}
template <>  tREAL16  cEigenPolynRoots<tREAL16>::ComplRelAccuracy() {return 1e-8;}

template <>  tREAL4   cEigenPolynRoots<tREAL4>::PolAbsAccuracy() {return 0.01;}
template <>  tREAL8   cEigenPolynRoots<tREAL8>::PolAbsAccuracy() {return 1e-5;}
template <>  tREAL16  cEigenPolynRoots<tREAL16>::PolAbsAccuracy() {return 1e-7;}
        //static Type ComplRelAccuracy() ;
        //static Type ComplAbsAccuracy() ;


template <class Type> bool cEigenPolynRoots<Type>::RootIsReal(const tCompl & aC,std::string * sayWhy)
{

   // [1]  Test is "aC" is a real number 
   Type C_i =aC.y();

   // [1.1]  if absolute value of imaginary part is "big" it's not
   if (std::abs(C_i) > 1e-5)
   {
      if (sayWhy)
         *sayWhy =  "ABS REAL COMPLEX=" + ToStr(std::abs(C_i));
      return false;
   }

   Type C_r =aC.x();
   // [1.1]  if relative imaginary part is "big"
   if (std::abs(C_i) > ComplRelAccuracy() * (std::abs(C_r)+1e-5))
   {
      if (sayWhy)
         *sayWhy =  "RELAT REAL COMPLEX=" + ToStr(std::abs(C_i)/(std::abs(C_r)+1e-5));
      return false;
   }

   // [2]  Test 
   Type aAbsVP = std::abs(mPol.Value(C_r));
   // [2.1]  if absolute value of polynom is big
   if (aAbsVP > PolAbsAccuracy())
   {
      if (sayWhy)
         *sayWhy =  "ABS VALUE POL " + ToStr(aAbsVP);
      return false;
   }

   Type aAVA = mPol.AbsValue(C_r);
   // [2.1]  if absolute value of polynom is big relatively to norm 
   if (aAbsVP > PolRelAccuracy() * (aAVA+1e-5))
   {
      if (sayWhy)
         *sayWhy =  "RELATIVE VALUE POL " + ToStr(aAbsVP/(aAVA+1e-5));
      return false;
   }

   if (sayWhy)
         *sayWhy =  "is real";

    return true;
}

template class cEigenPolynRoots<tREAL4>;
template class cEigenPolynRoots<tREAL8>;
template class cEigenPolynRoots<tREAL16>;


template<class Type> void My_Roots(const  cPolynom<Type> & aPol1) 
{
     // 4 => low accuracy
     // 16 => low timing
     // As we just want to see that there is no regression for standard double
     if (sizeof(Type) != 8) return;


// StdOut() << "DDDD " << aPol1.Degree() << "\n";
      // (X2+1)(X-1) = X3-X2+X-1
      int aNb=300;
      // vector<double> aCoeffs1 = {-1,1,-1,1,5,-2,0.12};
      // cPolynom<tREAL8>  aPol1(aCoeffs1);

      cAutoTimerSegm aTimeEigen(GlobAppTS(),"Eigen");
      for (int aK=0 ; aK<aNb ; aK++)
      {
          cEigenPolynRoots<Type> aEPR(aPol1,1e-3,10);
      }

      cAutoTimerSegm aTimeV1(GlobAppTS(),"V1");
      for (int aK=0 ; aK<aNb ; aK++)
      {
            aPol1.RealRoots(1e-20,60);
      }
      cAutoTimerSegm aTimeOthers(GlobAppTS(),"Others");

      cEigenPolynRoots<Type> aEPR(aPol1,1e-3,10);
      auto aV2 = aEPR.RealRoots();
      auto aV1 = aPol1.RealRoots(1e-20,60);
      if  (aV1.size() != aV2.size())
      {
          StdOut()  << " SZzzzZ= "  << aV1.size() << " " << aV2.size() << " SIZOFTYPE=" << sizeof(Type) << "\n";
          StdOut() << aV1  << aV2 << "\n";
          StdOut() << "Coeffs=" << aPol1.VCoeffs() << "\n";
          StdOut() << "V1=" << aV1 << "\n";
          StdOut() << "V2=" << aV2 << "\n";
          for (const auto & aC : aEPR.ComplexRoots())
          {
              std::string strWhy;
              bool isR = aEPR.RootIsReal(aC,&strWhy);
              StdOut() << "R=" << isR << " C=" << aC  << " W=" << strWhy << "\n";
          }

          StdOut() << " ------------------  MAT  ---------------------\n";
          StdOut() << aEPR.CompM() << "\n";
getchar();
      }
      // (X2+1)(X-1) = X3-X2+X-1
      // vector<double> coeffs = {-1,1,-1,1};
}


    // return V1RealRoots(mVCoeffs,aTol,ItMax);


template <class Type> std::vector<Type>  V2RealRoots(const cPolynom<Type> &  aPol, Type aTol,int aNbMaxIter)
{
    cEigenPolynRoots<Type> aEPR(aPol,aTol,aNbMaxIter);

    return aEPR.RealRoots();
}


/* ************************************************************************ */
/*                                                                          */
/*                       cPolynom                                           */
/*                                                                          */
/* ************************************************************************ */

     // ===========    constructor =========================

template <class Type> cPolynom<Type>::cPolynom(const tCoeffs & aVCoeffs) :
	mVCoeffs (aVCoeffs)
{
	// MMVII_INTERNAL_ASSERT_tiny(!mVCoeffs.empty(),"Empty polynom not handled");
}
template <class Type> cPolynom<Type>::cPolynom(const cPolynom<Type> & aPol) :
	cPolynom<Type>(aPol.mVCoeffs)
{
}
template <class Type> cPolynom<Type>::cPolynom(size_t aDegree) :
	cPolynom<Type>(tCoeffs(aDegree+1,0.0))
{
}
     // ===========    static constructor =========================

template <class Type>  cPolynom<Type> cPolynom<Type>::D0(const Type &aCste) {return cPolynom<Type>(tCoeffs{aCste});}
template <class Type>  cPolynom<Type> cPolynom<Type>::D1FromRoot(const Type &aRoot) {return cPolynom<Type>(tCoeffs{-aRoot,1.0});}
template <class Type>  cPolynom<Type> cPolynom<Type>::D2NoRoot(const Type & aVMin,const Type &aArgmin)
{
     cPolynom<Type> aRes = D1FromRoot(aArgmin);

     aRes =  aRes * aRes + D0(std::abs(aVMin));

     if (aVMin<0)
        aRes = aRes * -1.0;

     return aRes;
}

template <class Type>  cPolynom<Type> cPolynom<Type>::Monom(size_t aDegre)
{
     cPolynom<Type> aRes(aDegre);

     aRes.mVCoeffs.at(aDegre) = 1.0;
     return aRes;
}

template <class Type> const typename cPolynom<Type>::tCoeffs& cPolynom<Type>::VCoeffs() const {return mVCoeffs;}

template <class Type> 
   cPolynom<Type>  cPolynom<Type>::RandomPolyg(std::vector<Type> & aVRoots,int aNbRoot,int aNbNoRoot,Type aAmpl,Type MinDistRel)
{
	aVRoots.clear();
	std::vector<Type> aVSing;
	// number of singular point roots + NoRoots
	int aNbSing = aNbRoot + aNbNoRoot;

	// generate  singular values, no precaution
	for (int aK=0; aK<aNbSing ; aK++)
	{
            aVSing.push_back(aAmpl*RandUnif_C());
	}

	// make singular value separate
	std::sort(aVSing.begin(),aVSing.end());
	for (int aK=0; aK<aNbSing ; aK++)
	{
            Type aCorrec = aK-(aNbSing-1.0)/2.0;
            aCorrec /=  std::max(Type(1.0),Type(aNbSing));
	    aCorrec  *=  MinDistRel * aAmpl;

            aVSing[aK] += aCorrec;
	}

	cRandKAmongN  aSelRoot(aNbRoot,aNbSing);
	cPolynom<Type> aRes(tCoeffs{Type(10.0*RandUnif_C_NotNull(1e-2))});
	for (const auto & aRoot : aVSing)
	{
            if (aSelRoot.GetNext())
	    {
               aVRoots.push_back(aRoot);
               aRes = aRes * D1FromRoot(aRoot);
	    }
	    else
	    {
               Type aVMin(10.0*RandUnif_C_NotNull(1e-2));
               aRes = aRes * D2NoRoot(aVMin,aRoot);
	    }
	}

	return aRes;
}

template <class Type>  cPolynom<Type>  cPolynom<Type>::RandomPolyg(size_t aDeg,const Type & aAmpl)
{
    cPolynom<Type> aRes(aDeg);

    for (size_t aD=0 ; aD<=aDeg ; aD++)
        aRes.mVCoeffs.at(aD) =   RandUnif_C() * std::pow(aAmpl,-aD);

    return aRes;
}

template <class Type> std::vector<Type> cPolynom<Type>::RealRoots(const Type & aTol,int ItMax) const
{
//  StdOut() << "RealRootsRealRootsRealRootsRealRootsRealRootsRealRootsRealRootsRealRoots \n"; getchar();
    // return V1RealRoots(mVCoeffs,aTol,ItMax);
    return V2RealRoots(*this,aTol,ItMax);
}

     // ===========    others =========================

template <class Type> size_t cPolynom<Type>::Degree() const { return mVCoeffs.size() - 1; }


template <class Type> Type cPolynom<Type>::Value(const Type & aVal) const
{
    Type aResult = 0.0;
    Type aPowV   = 1.0;
    for (const auto & aCoef : mVCoeffs)
    {
         aResult +=  aCoef * aPowV;
	 aPowV *= aVal;
    }
    return aResult;
}

template <class Type> cPtxd<Type,2> cPolynom<Type>::Value(const tCompl & aVal) const
{
    tCompl aResult (0.0,0.0);
    tCompl aPowV   (1.0,0.0);
    for (const auto & aCoef : mVCoeffs)
    {
         aResult +=  aCoef * aPowV;
	 aPowV = aVal * aPowV;
    }
    return aResult;
}





template <class Type> Type cPolynom<Type>::AbsValue(const Type & aVal) const
{
    Type aResult = 0.0;
    Type aPowV   = 1.0;
    for (const auto & aCoef : mVCoeffs)
    {
         aResult +=  std::abs(aCoef * aPowV);
	 aPowV *= aVal;
    }
    return aResult;
}





template <class Type> cPolynom<Type>  cPolynom<Type>::operator * (const cPolynom<Type> & aP2) const
{
     cPolynom<Type> aRes(Degree() + aP2.Degree());

     for (size_t aK2 =0 ; aK2<aP2.mVCoeffs.size() ; aK2++)
     {
         const Type & aV2 = aP2[aK2];
         for (size_t aK1 =0 ; aK1<mVCoeffs.size() ; aK1++)
	 {
             aRes[aK1+aK2] += mVCoeffs[aK1] * aV2;
	 }
     }

     return aRes;
}

template <class Type> cPolynom<Type>  cPolynom<Type>::operator + (const cPolynom<Type> & aP2) const
{
      bool DMaxThis = Degree() > aP2.Degree();
      const cPolynom<Type> & aPMax =  (DMaxThis ? *this : aP2);
      const cPolynom<Type> & aPMin =  (DMaxThis ? aP2 : *this);

      cPolynom<Type>  aRes(aPMax);

      for (size_t aDMin=0 ; aDMin<aPMin.mVCoeffs.size() ; aDMin++)
          aRes[aDMin] += aPMin[aDMin];

      return aRes;
}

template <class Type> cPolynom<Type> &  cPolynom<Type>::operator += (const cPolynom<Type> & aP2) 
{
     *this = *this + aP2;
     return *this;
}

template <class Type> cPolynom<Type>  cPolynom<Type>::operator - (const cPolynom<Type> & aP2) const
{
      cPolynom<Type> aRes(std::max(Degree(),aP2.Degree()));

      for (size_t aDeg=0 ; aDeg<aRes.mVCoeffs.size() ; aDeg++)
          aRes[aDeg] = this->KthDef(aDeg) - aP2.KthDef(aDeg);

      return aRes;
}

template<class Type> cPolynom<Type>  cPolynom<Type>::Deriv() const
{
   std::vector<Type> aVCD;  // Vector Coeff Derivates
   for (size_t aDeg=1 ; aDeg<mVCoeffs.size() ; aDeg++)
       aVCD.push_back(mVCoeffs[aDeg]*aDeg);

   return cPolynom<Type>(aVCD);
}


template <class Type> cPolynom<Type>  cPolynom<Type>::operator * (const Type & aMul) const
{
     cPolynom<Type> aRes(*this);

     for (auto & aCoeff :aRes.mVCoeffs)
         aCoeff *= aMul;

     return aRes;
}

template <class Type,const int Dim> cPolynom<Type> PolSqN(const cPtxd<Type,Dim>& aVC,const cPtxd<Type,Dim>& aVL)
{
	return cPolynom<Type>
		({
		       static_cast<Type>(SqN2(aVC)), 
		       static_cast<Type>(2*Scal(aVC,aVL)), 
		       static_cast<Type>(SqN2(aVL))
                });
}



template<class Type> void TplBenchPolynome()
{
     cPolynom<Type> aPol1({3,2,1});  // 3 + 2 X + 1 X2 
     
     //   Elementary test on valus
     MMVII_INTERNAL_ASSERT_bench(std::abs(aPol1.Value(1)-6)<1e-10,"Polyn 000 ");
     MMVII_INTERNAL_ASSERT_bench(std::abs(aPol1.Value(-1)-2)<1e-10,"Polyn 000 ");
     MMVII_INTERNAL_ASSERT_bench(std::abs(aPol1.Value(0)-3)<1e-10,"Polyn 000 ");

     cPolynom<Type> aPol2({-5,4,-3,2,-1});

     // Use different operator  @  to chek  (P@Q) (a) = P(a) @ Q(a)
     cPolynom<Type> aP1mul2 = aPol1 * aPol2;
     cPolynom<Type> aP2mul1 = aPol2 * aPol1;

     cPolynom<Type> aP1p2 = aPol1 + aPol2;
     cPolynom<Type> aP2p1 = aPol2 + aPol1;

     cPolynom<Type> aP1min2 = aPol1 - aPol2;
     cPolynom<Type> aP2min1 = aPol2 - aPol1;

      // check two derivatives
     cPolynom<Type> aDerP1P2_A = aP1mul2.Deriv();
     cPolynom<Type> aDerP1P2_B = aPol1.Deriv() * aPol2 + aPol1*aPol2.Deriv();


     Type aEps = tElemNumTrait<Type> ::Accuracy();

     for (int aK=0 ; aK< 20 ; aK++)
     {
         Type aV = RandUnif_C();
	 Type aChekMul  = aPol1.Value(aV) * aPol2.Value(aV);
	 Type aChekP  = aPol1.Value(aV) + aPol2.Value(aV);
	 Type aChekMin  = aPol1.Value(aV) - aPol2.Value(aV);

         Type aDerA =  aDerP1P2_A.Value(aV) ;
         Type aDerB =  aDerP1P2_B.Value(aV) ;
         MMVII_INTERNAL_ASSERT_bench(std::abs(aDerA-aDerB) <aEps,"Polyn  mul");

         MMVII_INTERNAL_ASSERT_bench(std::abs(RelativeSafeDifference(aChekMul,aP1mul2.Value(aV)))<aEps,"Polyn  mul");
         MMVII_INTERNAL_ASSERT_bench(std::abs(RelativeSafeDifference(aChekMul,aP2mul1.Value(aV)))<aEps,"Polyn  mul");
         MMVII_INTERNAL_ASSERT_bench(std::abs(RelativeSafeDifference(aChekP,aP2p1.Value(aV)))<aEps,"Polyn  mul");
         MMVII_INTERNAL_ASSERT_bench(std::abs(RelativeSafeDifference(aChekP,aP1p2.Value(aV)))<aEps,"Polyn  mul");

         MMVII_INTERNAL_ASSERT_bench(std::abs(RelativeSafeDifference(aChekMin,aP1min2.Value(aV)))<aEps,"Polyn  mul");
         MMVII_INTERNAL_ASSERT_bench(std::abs(RelativeSafeDifference(-aChekMin,aP2min1.Value(aV)))<aEps,"Polyn  mul");
     }

     for (int aK=0 ; aK< 600 ; aK++)
     {
         std::vector<Type>  aVRootsGen;
	 Type aAmpl = 10*RandUnif_NotNull(1e-2);

         // cPolynom<Type> aPol = cPolynom<Type>::RandomPolyg(aVRootsGen,(int)RandUnif_N(6),(int)RandUnif_N(2),aAmpl,Type(1e-3));
	 //
	 //   To make it completly sure that test is ok fix easy parameter
	 //      not too low degree
	 //      => very separable roots
	 //      =>  probable instability with closed roots, to investigate (maybe option with polynom division ?)
	 //
	 int aNbRoot = RandUnif_N(5);
	 int aNbNoRoot = std::min((4-aNbRoot)/2,round_ni(RandUnif_N(3)));
         cPolynom<Type> aPol = cPolynom<Type>::RandomPolyg(aVRootsGen,aNbRoot,aNbNoRoot,aAmpl,Type(9e-2));

         std::vector<Type>  aVRootsCalc  = aPol.RealRoots(1e-20,60);
	 if (aVRootsGen.size()  != aVRootsCalc.size())
	 {
	     StdOut() <<  "VEEE  " << aVRootsGen.size()  << " " << aVRootsCalc.size()<< std::endl;
             MMVII_INTERNAL_ASSERT_bench(false,"roots size check");
	 }

	 for (size_t aK=0 ; aK<aVRootsCalc.size() ; aK++)
	 {
             Type aDif = RelativeSafeDifference(aVRootsGen[aK],aVRootsCalc[aK]);
             MMVII_INTERNAL_ASSERT_bench(aDif<aEps,"roots size check");
	 }
         // My_Roots(aPol);

     }
}

typedef cPolynom<cPolynom<tREAL8>>  tPolXY;

void Bench_Pol2Var()
{
    size_t aDx = 3;
    size_t aDy = 4;

    tPolXY aPol(aDy);
    for (size_t dy =0 ; dy<= aDy ; dy++)
        aPol[dy] = cPolynom<tREAL8>::RandomPolyg(aDx,1.0);

}



void BenchPolynome(cParamExeBench & aParam)
{
    if (! aParam.NewBench("Polynom")) return;

    // TestPolynEigen();

    // TplBenchPolynome<tREAL4>();  // =>  with eigen , impossible to have always acceptable accuracy
    TplBenchPolynome<tREAL8>();
    TplBenchPolynome<tREAL16>();

    Bench_Pol2Var();

    aParam.EndBench();
}


void JoPoly()
{
    tPolXY aPol(5);
    aPol+aPol*aPol;
    aPol.Deriv();
     
}
/*
*/

#define INSTANTIATE_PolSqN(TYPE,DIM)\
template cPolynom<TYPE> PolSqN(const cPtxd<TYPE,DIM>& aVecCste,const cPtxd<TYPE,DIM>& aVecLin);


INSTANTIATE_PolSqN(tREAL4,3);
INSTANTIATE_PolSqN(tREAL8,3);
INSTANTIATE_PolSqN(tREAL16,3);
template class cPolynom<tREAL4>;
template class cPolynom<tREAL8>;
template class cPolynom<tREAL16>;

};

