
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include "cMMVII_Appli.h"
#include "V1VII.h"
#include <Eigen/Dense>
// #include "unsupported/Eigen/Polynomials"

//#if defined(__GNUC__) && !defined(__clang__)
// #  pragma GCC diagnostic pop
// #endif


    // #include "include/MMVII_nums.h"
    // #include "include/MMVII_Bench.h"

//#include <iostream>
//#include <vector>

using namespace Eigen;
using namespace std;

namespace MMVII
{

/**
    Class for extraction of roots of polynoms using "Companion matrix method"

  See  https://en.wikipedia.org/wiki/Companion_matrix
*/


template <class Type> class cEigenPolynRoots
{
    public :
        cEigenPolynRoots(const cPolynom<Type> &) ;
        const std::vector<Type> & RealRoots() const;
        const VectorXcd  & ComplexRoots() const;
        bool  RootIsReal(const std::complex<tREAL8> &,std::string * sayWhy=nullptr);
        const MatrixXd & CompM() const;
    private :

        cPolynom<Type>     mPol;
        cPolynom<Type>     mDPol;
        size_t             mDeg;
        MatrixXd           mCompM;   ///< companion matrix
        VectorXcd          mCEV;     ///< Complex eigen values
        std::vector<Type>  mRR;      ///< Real roots

};


template <class Type> cEigenPolynRoots<Type>::cEigenPolynRoots(const cPolynom<Type> & aPol)  :
    mPol   (aPol),
    mDPol  (aPol.Deriv()),
    mDeg   (mPol.Degree()),
    mCompM (mDeg,mDeg)
{

    if (mDeg==0)
       return;

    for (size_t aI = 0; aI < mDeg; ++aI) 
        for (size_t aJ = 0; aJ < mDeg; ++aJ) 
            mCompM(aI,aJ) = 0;

    // fill the diagonal up the principal diag
    for (size_t aK = 0; aK < mDeg-1; ++aK) 
    {
        mCompM(aK+1, aK ) = 1; 
    }

    const Type & aHighCoeff = mPol[mDeg];
    // Fill last line with normalized coeff
    for (size_t aK = 0; aK < mDeg; ++aK) 
    {
        mCompM(aK, mDeg - 1) = -mPol[aK] / aHighCoeff;
    }

    EigenSolver<MatrixXd> aSolver(mCompM);
    mCEV = aSolver.eigenvalues() ;

   for (const auto & aC : mCEV)
       if (RootIsReal(aC))
          mRR.push_back(aC.real());
   std::sort(mRR.begin(),mRR.end());
}

template <class Type> const std::vector<Type> & cEigenPolynRoots<Type>::RealRoots() const {return mRR;}

template <class Type> const VectorXcd  & cEigenPolynRoots<Type>::ComplexRoots() const {return mCEV;}

template <class Type> const MatrixXd & cEigenPolynRoots<Type>::CompM() const {return mCompM;}


/**  Also the question seems pretty basic, it becomes more complicated due to numericall approximation */

template <class Type> bool cEigenPolynRoots<Type>::RootIsReal(const std::complex<tREAL8> & aC,std::string * sayWhy)
{
   // [1]  Test is "aC" is a real number 

   tREAL8 C_i = aC.imag();
   // [1.1]  if absolute value of imaginary part is "big" it's not
   if (std::abs(C_i) > 1e-5)
   {
      if (sayWhy)
         *sayWhy =  "ABS REAL COMPLEX=" + ToStr(std::abs(C_i));
      return false;
   }

   tREAL8 C_r = aC.real();
   // [1.1]  if relative imaginary part is "big"
   if (std::abs(C_i) > 1e-8 * (std::abs(C_r)+1e-5))
   {
      if (sayWhy)
         *sayWhy =  "RELAT REAL COMPLEX=" + ToStr(std::abs(C_i)/(std::abs(C_r)+1e-5));
      return false;
   }

   // [2]  Test 
   tREAL8 aAbsVP = std::abs(mPol.Value(C_r));
   // [2.1]  if absolute value of polynom is big
   if (aAbsVP > 1e-5)
   {
      if (sayWhy)
         *sayWhy =  "ABS VALUE POL " + ToStr(aAbsVP);
      return false;
   }

   tREAL8 aAVA = mPol.AbsValue(C_r);
   // [2.1]  if absolute value of polynom is big relatively to norm 
   if (aAbsVP > 1e-8 * (aAVA+1e-5))
   {
      if (sayWhy)
         *sayWhy =  "RELATIVE VALUE POL " + ToStr(aAbsVP/(aAVA+1e-5));
      return false;
   }

   if (sayWhy)
         *sayWhy =  "is real";

    return true;
}


template class cEigenPolynRoots<tREAL8>;

template<class Type> void My_Roots(const  cPolynom<Type> & aPol1) 
{
}
template <> void My_Roots(const  cPolynom<tREAL8> & aPol1) 
{
// StdOut() << "DDDD " << aPol1.Degree() << "\n";
      // (X2+1)(X-1) = X3-X2+X-1
      int aNb=300;
      // vector<double> aCoeffs1 = {-1,1,-1,1,5,-2,0.12};
      // cPolynom<tREAL8>  aPol1(aCoeffs1);

      cAutoTimerSegm aTimeEigen(GlobAppTS(),"Eigen");
      for (int aK=0 ; aK<aNb ; aK++)
      {
          cEigenPolynRoots<tREAL8> aEPR(aPol1);
      }

      cAutoTimerSegm aTimeV1(GlobAppTS(),"V1");
      for (int aK=0 ; aK<aNb ; aK++)
      {
            aPol1.RealRoots(1e-20,60);
      }
      cAutoTimerSegm aTimeOthers(GlobAppTS(),"Others");

      cEigenPolynRoots<tREAL8> aEPR(aPol1);
      auto aV2 = aEPR.RealRoots();
      auto aV1 = aPol1.RealRoots(1e-20,60);
      StdOut()  << " SZzzzZ= "  << aV1.size() << " " << aV2.size() << "\n";
      if  (aV1.size() != aV2.size())
      {
          StdOut() << "Coeffs=" << aPol1.VCoeffs() << "\n";
          StdOut() << "V1=" << aV1 << "\n";
          StdOut() << "V2=" << aV2 << "\n";
          for (const auto & aC : aEPR.ComplexRoots())
          {
              std::string strWhy;
              bool isR = aEPR.RootIsReal(aC,&strWhy);
              StdOut() << "R=" << isR << " C=" << aC  << " W=" << strWhy << "\n";
          }

          StdOut() << "DET=" << aEPR.CompM().determinant()    << "\n";
          StdOut() << " ------------------  MAT  ---------------------\n";
          StdOut() << aEPR.CompM() << "\n";
getchar();
      }
      else
         StdOut() << aV1  << aV2 << "\n";
      // (X2+1)(X-1) = X3-X2+X-1
      // vector<double> coeffs = {-1,1,-1,1};
}


#if (0)

/*
MatrixXd createCompanionMatrix(const vector<double>& coeffs) {
StdOut() << "BEGIN createCompanionMatrixcreateCompanionMatrixcreateCompanionMatrix\n";
    int n = coeffs.size();
    MatrixXd companionMatrix(n - 1, n - 1);

StdOut() << "LLLLL " << __LINE__ << "\n";

    // Remplir la matrice compagnon
    // for (int i = 0; i < n - 1; ++i) {
    for (int i = 0; i < n - 2; ++i) {
        companionMatrix(i+1, i ) = 1; // Remplir la diagonale au-dessus de la diagonale principale
    }
StdOut() << "LLLLL " << __LINE__ << "\n";

    // Remplir la dernière colonne de la matrice avec les coefficients du polynôme
    for (int i = 0; i < n - 1; ++i) {
        // companionMatrix(i, n - 1) = -coeffs[i] / coeffs[n - 1];
        companionMatrix(i, n - 2) = -coeffs[i] / coeffs[n - 1];
    }

StdOut() << "END createCompanionMatrixcreateCompanionMatrixcreateCompanionMatrix\n";
    return companionMatrix;
}

// Fonction principale
int My_Roots() {
    // (X2+1)(X-1) = X3-X2+X-1
    // Exemple de coefficients d'un polynôme : x^3 - 6x^2 + 11x - 6
    //  (x-1) (x2+ 5x  +6)
    // vector<double> coeffs = {1, -6, 11, -6};
    vector<double> coeffs = {1,-1,1,-1};



    // Créer la matrice compagnon
    MatrixXd companionMatrix = createCompanionMatrix(coeffs);

    std::cout << "mmmm " <<  companionMatrix << "\n";

    // Calculer les valeurs propres (racines du polynôme)
    EigenSolver<MatrixXd> solver(companionMatrix);
    std::cout << "eeeee " <<  solver.eigenvalues() << "\n";

    // const auto& theEigenV = solver.eigenvalues();
    VectorXcd eivals = solver.eigenvalues();
    std::cout << "EEEEEE " << eivals << "\n";

    VectorXd R_roots = solver.eigenvalues().real();
    VectorXd I_roots = solver.eigenvalues().imag();

    for (int i = 0; i < R_roots.size(); ++i) {
        cout << "R=" << R_roots[i]  << " I=" << I_roots[i] << endl;
    }
*/

/*
    VectorXd roots = solver.eigenvalues().real();

    // Afficher les racines
    cout << "Les racines du polynôme sont : " << endl;
    for (int i = 0; i < roots.size(); ++i) {
        cout << roots[i] << endl;
    }
*/

    return 0;
}
#endif

/*
 */




#if (0)
void TestPolynEigen()
{
   // cEigenMMVIIPoly  aPoly({-1,0,3});
   std::vector<tREAL8>  aPoly {-1,0,3};
   // bool  hasArealRoot;
   // tREAL8 aS = greatestRealRoot(hasArealRoot);

   StdOut() <<  "EEE:V= " << Eigen::poly_eval (aPoly,2) << "\n";

   // polynomialsolver<float,Dynamic>( internal::random<int>(9,13));
   PolynomialSolver<tREAL8,10>  aSolver;

   aSolver.compute(aPoly);
   // aSolver.roots();
   // FakeUseIt(aSolver);

}
#endif


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

template <class Type> std::vector<Type> cPolynom<Type>::RealRoots(const Type & aTol,int ItMax) const
{
//  StdOut() << "RealRootsRealRootsRealRootsRealRootsRealRootsRealRootsRealRootsRealRoots \n"; getchar();
    return V1RealRoots(mVCoeffs,aTol,ItMax);
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
     cPolynom<Type> aPol1({3,2,1});
     
     MMVII_INTERNAL_ASSERT_bench(std::abs(aPol1.Value(1)-6)<1e-10,"Polyn 000 ");
     MMVII_INTERNAL_ASSERT_bench(std::abs(aPol1.Value(-1)-2)<1e-10,"Polyn 000 ");
     MMVII_INTERNAL_ASSERT_bench(std::abs(aPol1.Value(0)-3)<1e-10,"Polyn 000 ");

     cPolynom<Type> aPol2({-5,4,-3,2,-1});

     cPolynom<Type> aP1mul2 = aPol1 * aPol2;
     cPolynom<Type> aP2mul1 = aPol2 * aPol1;

     cPolynom<Type> aP1p2 = aPol1 + aPol2;
     cPolynom<Type> aP2p1 = aPol2 + aPol1;

     cPolynom<Type> aP1min2 = aPol1 - aPol2;
     cPolynom<Type> aP2min1 = aPol2 - aPol1;

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
         My_Roots(aPol);

     }
}

void BenchPolynome(cParamExeBench & aParam)
{
    if (! aParam.NewBench("Polynom")) return;

    // TestPolynEigen();

    TplBenchPolynome<tREAL4>();
    TplBenchPolynome<tREAL8>();
    TplBenchPolynome<tREAL16>();


    aParam.EndBench();
}


#define INSTANTIATE_PolSqN(TYPE,DIM)\
template cPolynom<TYPE> PolSqN(const cPtxd<TYPE,DIM>& aVecCste,const cPtxd<TYPE,DIM>& aVecLin);


INSTANTIATE_PolSqN(tREAL4,3);
INSTANTIATE_PolSqN(tREAL8,3);
INSTANTIATE_PolSqN(tREAL16,3);
template class cPolynom<tREAL4>;
template class cPolynom<tREAL8>;
template class cPolynom<tREAL16>;

};

