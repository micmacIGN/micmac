#include "include/cMMVII_Appli.h"
#include "include/V1VII.h"


/*
    // #include "include/MMVII_nums.h"
    // #include "include/MMVII_Bench.h"
    // #include <boost/math/tools/polynomial.hpp>
     //#include "Eigen/unsupported/Eigen/Polynomials"
 */


namespace MMVII
{

	/*
template <class Type> class  cPolynom
{
	public :
           typedef std::vector<Type>  tCoeffs;
	   cPolynom(const tCoeffs &);
	   cPolynom(const cPolynom &);
	   cPolynom(size_t aDegre);
	   size_t  Degree() const;

           static cPolynom<Type>  D0(const Type &aCste);      ///< constant polynom degre 0
           static cPolynom<Type>  D1FromRoot(const Type &aRoot);    ///< degre 1 polynom with aRoot
           static cPolynom<Type>  D2NoRoot(const Type & aVMin,const Type &aArgmin);  ///< +- (|V| + (x-a) ^2) , 

           static cPolynom<Type>  RandomPolyg(int aDegree,Type & anAmpl);
	   ///  Generate random polygo from its randomly generated roots => test for 
           static cPolynom<Type>  RandomPolyg(std::vector<Type> & aVRoots,int aNbRoot,int aNbNoRoot,Type Interv,Type MinDist);

	   Type  Value(const Type & aVal) const;

	   cPolynom<Type> operator * (const cPolynom<Type> & aP2) const;
	   cPolynom<Type> operator + (const cPolynom<Type> & aP2) const;
	   cPolynom<Type> operator * (const  Type & aVal) const;
	   std::vector<Type> RealRoots(const Type & aTol,int ItMax);


	   Type&   operator [] (size_t aK) {return mVCoeffs[aK];}
	   const Type&   operator [] (size_t aK) const {return mVCoeffs[aK];}
           const tCoeffs &  VCoeffs() const;

	private :
           tCoeffs  mVCoeffs;
};
*/


/* ************************************************************************ */
/*                                                                          */
/*                       cPolynom                                           */
/*                                                                          */
/* ************************************************************************ */

     // ===========    constructor =========================

template <class Type> cPolynom<Type>::cPolynom(const tCoeffs & aVCoeffs) :
	mVCoeffs (aVCoeffs)
{
	MMVII_INTERNAL_ASSERT_tiny(!mVCoeffs.empty(),"Empty polynom not handled");
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

template <class Type> std::vector<Type> cPolynom<Type>::RealRoots(const Type & aTol,int ItMax)
{
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

template <class Type> cPolynom<Type>  cPolynom<Type>::operator * (const Type & aMul) const
{
     cPolynom<Type> aRes(*this);

     for (auto & aCoeff :aRes.mVCoeffs)
         aCoeff *= aMul;

     return aRes;
}


template<class Type> void TplBenchPolynome()
{
     StdOut() << " XXX TplBenchPolynome \n";
     cPolynom<Type> aPol1({3,2,1});
     
     MMVII_INTERNAL_ASSERT_bench(std::abs(aPol1.Value(1)-6)<1e-10,"Polyn 000 ");
     MMVII_INTERNAL_ASSERT_bench(std::abs(aPol1.Value(-1)-2)<1e-10,"Polyn 000 ");
     MMVII_INTERNAL_ASSERT_bench(std::abs(aPol1.Value(0)-3)<1e-10,"Polyn 000 ");

     cPolynom<Type> aPol2({-5,4,-3,2,-1});

     cPolynom<Type> aP1m2 = aPol1 * aPol2;
     cPolynom<Type> aP2m1 = aPol2 * aPol1;

     cPolynom<Type> aP1p2 = aPol1 + aPol2;
     cPolynom<Type> aP2p1 = aPol2 + aPol1;

     Type aEps = tElemNumTrait<Type> ::Accuracy();

     for (int aK=0 ; aK< 20 ; aK++)
     {
         Type aV = RandUnif_C();
	 Type aChekM  = aPol1.Value(aV) * aPol2.Value(aV);
	 Type aChekP  = aPol1.Value(aV) + aPol2.Value(aV);

         MMVII_INTERNAL_ASSERT_bench(std::abs(RelativeSafeDifference(aChekM,aP1m2.Value(aV)))<aEps,"Polyn  mul");
         MMVII_INTERNAL_ASSERT_bench(std::abs(RelativeSafeDifference(aChekM,aP2m1.Value(aV)))<aEps,"Polyn  mul");
         MMVII_INTERNAL_ASSERT_bench(std::abs(RelativeSafeDifference(aChekP,aP2p1.Value(aV)))<aEps,"Polyn  mul");
         MMVII_INTERNAL_ASSERT_bench(std::abs(RelativeSafeDifference(aChekP,aP1p2.Value(aV)))<aEps,"Polyn  mul");
     }

     for (int aK=0 ; aK< 200 ; aK++)
     {
         std::vector<Type>  aVRootsGen;
	 Type aAmpl = 10*RandUnif_NotNull(1e-2);
         cPolynom<Type> aPol = cPolynom<Type>::RandomPolyg(aVRootsGen,(int)RandUnif_N(6),(int)RandUnif_N(2),aAmpl,Type(1e-3));

         std::vector<Type>  aVRootsCalc  = aPol.RealRoots(1e-10,30);
	 if (aVRootsGen.size()  != aVRootsCalc.size())
	 {
	     StdOut() <<  "VEEE  " << aVRootsGen.size()  << " " << aVRootsCalc.size()<< "\n";
             MMVII_INTERNAL_ASSERT_bench(false,"roots size check");
	 }

	 for (size_t aK=0 ; aK<aVRootsCalc.size() ; aK++)
	 {
             Type aDif = RelativeSafeDifference(aVRootsGen[aK],aVRootsCalc[aK]);
             MMVII_INTERNAL_ASSERT_bench(aDif<aEps,"roots size check");
	 }

     }
}

void BenchPolynome(cParamExeBench & aParam)
{
    if (! aParam.NewBench("Polynom")) return;

    TplBenchPolynome<tREAL4>();
    TplBenchPolynome<tREAL8>();
    TplBenchPolynome<tREAL16>();


    aParam.EndBench();
}




template class cPolynom<tREAL4>;
template class cPolynom<tREAL8>;
template class cPolynom<tREAL16>;

};

