#include "MMVII_SysSurR.h"
#include "MMVII_Tpl_Images.h"


namespace MMVII
{
/*
  Big must be set equal to any very large double precision constant.
  It's value here is appropriate for the Sun.
*/
#define BIG 1.0e37

	/*
	  stdlib.h - used to define the return value of malloc() and free().
	*/

#include <cstdlib>


template <class Type> class   cMemo1LinearEq
{
     public :
        cMemo1LinearEq(const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS);

        Type              mW;
	cDenseVect<Type>  mCoeff;
	Type              mRHS;


};



template <class Type>  class cLinearMemoEq  : public cLinearOverCstrSys<Type>
{
	public :
             typedef cMemo1LinearEq<Type> tMemEq;
             /// 
	     void SpecificAddObservation(const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS) override;
             /// Add  aPds (  aCoeff .X = aRHS) , version sparse
             void SpecificAddObservation(const Type& aWeight,const cSparseVect<Type> & aCoeff,const Type &  aRHS) override;

	     ///  Virtual method , specify the parameters (coeff ...)
             Type Residual(const cDenseVect<Type> & aVect,const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS) const override;

	     Type Residual(const cDenseVect<Type> & aVect,const tMemEq&) const;
	     ///  Specific method , use the memorized equation
	     Type Residual(const cDenseVect<Type> & aVect) const;

             void Reset()  override;
	     cLinearMemoEq(size_t aNbVar);

	protected :
             std::list<tMemEq>  mLEq;

};

template <class Type>  class cCraig_Barrodale_Roberts_l1 : public cLinearMemoEq<Type>
{
	public :
	    static constexpr Type mTol=1e-8;

            typedef tINT4   INT;
	    cDenseVect<Type>  Solve() override;
	    cCraig_Barrodale_Roberts_l1(size_t aNbVar);

	    static void Bench();
	    void Bench1Sol(const cDenseVect<Type> & aSol);

	private :
            void LowLevelSolve(INT m,INT  n,Type * a,Type *b,Type toler,Type * x,Type * e);

};


/* ********************************************************* */
/*                                                           */
/*              cMemo1LinearEq                               */
/*                                                           */
/* ********************************************************* */

template <class Type>  cMemo1LinearEq<Type>::cMemo1LinearEq(const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS) :
   mW      (aWeight),
   mCoeff  (aCoeff.Dup()),
   mRHS    (aRHS)
{
}

/* ********************************************************* */
/*                                                           */
/*              cLinearMemoEq                                */
/*                                                           */
/* ********************************************************* */

template <class Type>  
    cLinearMemoEq<Type>::cLinearMemoEq(size_t aNbVar):
       cLinearOverCstrSys<Type> (aNbVar)
{
}

template <class Type>  
   Type cLinearMemoEq<Type>::Residual(const cDenseVect<Type> & aVect,const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS) const
{
     return  std::abs(aWeight*(aVect.DotProduct(aCoeff)-aRHS));
}

template <class Type> Type cLinearMemoEq<Type>::Residual(const cDenseVect<Type> & aVect,const tMemEq & anEq ) const
{
   return  Residual(aVect,anEq.mW,anEq.mCoeff,anEq.mRHS);
}

template <class Type> Type cLinearMemoEq<Type>::Residual(const cDenseVect<Type> & aVect) const
{
     Type aSom =0.0;
     for (const auto & anEq : this->mLEq)
         aSom +=  Residual(aVect,anEq);

     return aSom;
}

template <class Type>  
    void cLinearMemoEq<Type>::SpecificAddObservation(const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS)
{
    mLEq.push_back(tMemEq(aWeight,aCoeff,aRHS));
}

template <class Type>  
    void cLinearMemoEq<Type>::SpecificAddObservation(const Type& aWeight,const cSparseVect<Type> & aCoeff,const Type &  aRHS)
{
     SpecificAddObservation(aWeight,cDenseVect<Type>(aCoeff,this->mNbVar),aRHS);
}

template <class Type>  
    void cLinearMemoEq<Type>::Reset()  
{
    mLEq.clear();
}



/* ********************************************************* */
/*                                                           */
/*              cCraig_Barrodale_Roberts_l1                  */
/*                                                           */
/* ********************************************************* */

template <class Type>  
   cCraig_Barrodale_Roberts_l1<Type>::cCraig_Barrodale_Roberts_l1(size_t aNbVar) :
	cLinearMemoEq<Type>(aNbVar)
{
}

template <class Type>  cLinearOverCstrSys<Type> *  AllocL1_Barrodale(size_t aNbVar)
{
     return new cCraig_Barrodale_Roberts_l1<Type>(aNbVar);
}

template <class Type>  cDenseVect<Type> cCraig_Barrodale_Roberts_l1<Type>::Solve()
{

    cDenseVect<Type> aRes(this->mNbVar);
    cDenseVect<Type> aErr(this->mLEq.size());

   // Im2D_REAL8  A(_NbVar+2,_NbForm+2);
    cIm2D<Type>  aImA(cPt2di(this->mNbVar+2,this->mLEq.size()+2));
    cIm1D<Type>  aVecB(this->mLEq.size());

    cDataIm2D<Type> & aDA =   aImA.DIm();
    cDataIm1D<Type> & aDB =   aVecB.DIm();
    
    int aKEq = 0;
    for (const auto & anEq : this->mLEq)
    {
        for (int aKV=0 ; aKV<this->mNbVar ; aKV++)
	{
            aDA.SetV(cPt2di(aKV,aKEq),anEq.mCoeff(aKV) * anEq.mW);
	}
	aDB.SetV(aKEq,anEq.mRHS * anEq.mW);
        aKEq++;
    }
    /*
        Type              mW;
	cDenseVect<Type>  mCoeff;
	Type              mRHS;
	*/

    LowLevelSolve(this->mLEq.size(),this->mNbVar,aDA.RawDataLin(),aDB.RawDataLin(),mTol,aRes.RawData(),aErr.RawData());

    return aRes;
}

template <class Type> void cCraig_Barrodale_Roberts_l1<Type>::Bench1Sol(const cDenseVect<Type> & aSol)
{
     std::vector<Type> aVRes;
     for (const auto & anEq : this->mLEq)
         aVRes.push_back(this->Residual(aSol,anEq));

     std::sort(aVRes.begin(),aVRes.end());

     //StdOut() <<  "RESS " << aVRes.at(this->mNbVar-1) << " " << aVRes.at(this->mNbVar) << std::endl;
     // A first condition is that at least NB VAR are out
     MMVII_DEV_WARNING("Replace assert 'aVRes.at(this->mNbVar-1)<1e-5' with '1e-4' because of clang");
     MMVII_INTERNAL_ASSERT_bench(aVRes.at(this->mNbVar-1)<1e-4,"Bench1Sol");

     tREAL8 aScoreS = this->Residual(aSol);

     tREAL8 aMinDif = 1e10;
     for (int aK=0 ; aK< 100*this->mNbVar ; aK++)
     {
          cDenseVect<Type> aNewV = aSol + 0.01*cDenseVect<Type>::RanGenerate(this->mNbVar);

          tREAL8 aScoreN = this->Residual(aNewV);

	  UpdateMin(aMinDif,aScoreN - aScoreS);
     }
     //  Check it is really a minimum
     MMVII_INTERNAL_ASSERT_bench(aMinDif>-1e-5,"Bench1Sol");
}

template <class Type> void cCraig_Barrodale_Roberts_l1<Type>::Bench()
{
    for (int aTimes=0 ; aTimes<10 ;  aTimes++)
    {
        for (int aDim=1 ; aDim<10 ; aDim++)
        {
            int aNbEq = 3 + aDim * 3;
            cCraig_Barrodale_Roberts_l1 aSys(aDim);
            for (int aKEq = 0 ; aKEq<aNbEq ; aKEq++)
            {
                auto v1 = RandInInterval(0.1,1.0);
                auto v2 = cDenseVect<Type>::RanGenerate(aDim);
                auto v3 = RandInInterval(-10.0,10.0);
                aSys.PublicAddObservation(v1,v2,v3); // use variable to force evaluation order
            }
            cDenseVect<Type>  aVec = aSys.Solve();
            aSys.Bench1Sol(aVec);
        }
    }
}

void BenchL1Solver(cParamExeBench & aParam)
{
    if (! aParam.NewBench("L1Solver")) return;

    cCraig_Barrodale_Roberts_l1<tREAL4>::Bench();
    cCraig_Barrodale_Roberts_l1<tREAL8>::Bench();
    cCraig_Barrodale_Roberts_l1<tREAL16>::Bench();

    aParam.EndBench();
}

	    // static void Bench();
	    // void Bench1Sol(const cDenseVect<Type> & aSol);

// template <class tREAL> void Elise_Craig_Barrodale_Roberts_l1(INT m,INT  n,tREAL * a,tREAL *b,tREAL toler,tREAL * x,tREAL * e)
template <class Type> void cCraig_Barrodale_Roberts_l1<Type>::LowLevelSolve(INT m,INT  n,Type * a,Type *b,Type toler,Type * x,Type * e)
{
	/*
	  This algorithm uses a modification of the simplex method
	  to calculate an L1 solution to an over-determined system
	  of linear equations.

	  input:  n = number of unknowns.
		  m = number of equations (m >= n).
		  a = two dimensional double precision array of size
			m+2 by n+2.  On entry, the coefficients of the
			matrix must be stored in the first m rows and
			n columns of a.
		  b = one dimensional double precision array of length
			m.  On entry, b must contain the right hand
			side of the equations.
		  toler = a small positive tolerance.  Empirical
				evidence suggests toler=10**(-d*2/3)
				where d represents the number of
				decimal digits of accuracy available.
				Essentially, the routine regards any
				quantity as zero unless its magnitude
				exceeds toler.  In particular, the
				routine will not pivot on any number
				whose magnitude is less than toler.

	  output:  a = the first m rows and n columns are garbage on
			output.
		   a[m][n] = the minimum sum of the absolute values of
				the residuals.
		   a[m][n+1] = the rank of the matrix of coefficients.
		   a[m+1][n] = e-xit code with values:
				0 - optimal solution which is probably
					non-unique.
				1 - unique optimal solution.
				2 - calculations terminated prematurely
					due to rounding errors.
		   a[m+1][n+1] = number of simplex iterations performed.
		   b = garbage on output.
		   x = one dimensional double precision array of length
			n.  On e-xit, this array contains a solution to
			the L1 problem.
		   e = one dimensional double precision array of size n.
			On e-xit, this array contains the residuals in
			the equations.

	  Reference:  I. Barrodale and F. D. Roberts, "Algorithm 478:
		Solution of an Overdetermined System of Equations in the
		L1 norm," CACM 17, 1974, pp. 319-320.

	  Programmer:  R. J. Craig
		       kat3@ihgp.ih.lucent.com
		       (630) 979-1822
		       Lucent Technologies
		       2000 Naperville Rd.
		       Room:  1A-365
		       Napeville, IL. 60566-7033
	*/

	Type min, max, *ptr, *ptr1, *ptr2, *ptr3;
	Type  d=0, pivot;
	long double sum;
	int out, *s, m1, n1, m2, n2, i, j, kr, kl, kount, in, k, l;
	bool stage, test;

    out = in = -123456; test = false;  // Warb initialize

	/* initialization */

	m1 = m + 1;
	n1 = n + 1;
	m2 = m + 2;
	n2 = n + 2;
	s = (int *)malloc(m*sizeof(int));
	for (j=0, ptr=a+m1*n2; j<n; j++, ptr++) {
		*ptr = (Type)(j+1);
		x[j] = 0.0;
	}
	for (i=0, ptr=a+n1, ptr1=a+n, ptr2=a; i<m; i++, ptr += n2,
		ptr1 += n2, ptr2 += n2) {
		*ptr = n + i + 1;
		*ptr1 = b[i];
		if (b[i] <= 0.0)
			for (j=0, ptr3=ptr2; j<n2; j++, ptr3++)
				*ptr3 *= -1.0;
		e[i] = 0.0;
	}

	/* compute marginal costs */

	for (j=0, ptr=a; j<n1; j++, ptr++) {
		sum = (long double)0.0;
		for (i=0, ptr1=ptr; i<m; i++, ptr1 += n2) {
			sum += (long double)(*ptr1);
		}
		*(a+m*n2+j) = (Type)sum;
	}

	stage = true;
	kount = -1;
	kr = kl = 0;
loop1:	if (stage) {

		/*
		   Stage I:
			Determine the vector to enter the basis.
		*/
	
		max = -1.0;
		for (j=kr, ptr=a+m1*n2+kr, ptr1=a+m*n2+kr; j<n; j++, ptr++,
			ptr1++) {
			if (fabs(*ptr)<=(Type)n && (d=fabs(*ptr1))>max) {
				max = d;
				in = j;
			}
		}
		if (*(a+m*n2+in) < 0.0)
			for (i=0, ptr=a+in; i<m2; i++, ptr += n2)
				*ptr *= -1.0;
	} else {

		/*
		   Stage II
			Determine the vector to enter the basis.
		*/

		max = -BIG;
		for (j=kr, ptr=a+m*n2+kr; j<n; j++, ptr++) {
			d = *ptr;
			if (d >= 0.0 && d > max) {
				max = d;
				in = j;
			} else if (d <= -2.0) {
				d = -d - 2.0;
				if (d > max) {
					max = d;
					in = j;
				}
			}
		}
		if (max <= toler) {
			l = kl - 1;
			for (i=0, ptr=a+n, ptr1=a+kr; i<=l; i++, ptr += n2,
				ptr1 += n2)
				if (*ptr < 0.0)
					for (j=kr, ptr2=ptr1; j<n2; j++, ptr2++)
						*ptr2 *= -1.0;
			*(a+m1*n2+n) = 0.0;
			if (kr == 0) {
				for (j=0, ptr=a+m*n2; j<n; j++, ptr++) {
					d = fabs(*ptr);
					if (d <= toler || 2.0-d <= toler)
						goto end;
				}
				*(a+m1*n2+n) = 1.0;
			}
			goto end;
		} else if (*(a+m*n2+in) <= 0.0) {
			for (i=0, ptr=a+in; i<m2; i++, ptr += n2)
				*ptr *= -1.0;
			*(a+m*n2+in) -= 2.0;
		}
	}

	/* Determine the vector to leave the basis */

	for (i=kl, k = -1, ptr=a+kl*n2+in, ptr1=a+kl*n2+n; i<m; i++,
		ptr += n2, ptr1 += n2) {
		if (*ptr > toler) {
			k++;
			b[k] = (*ptr1)/(*ptr);
			s[k] = i;
			test = true;
		}
	}
loop2:	if (k <= -1)
		test = false;
	else {
		min = BIG;
		for (i=0; i<=k; i++)
			if (b[i] < min) {
				j = i;
				min = b[i];
				out = s[i];
			}
		b[j] = b[k];
		s[j] = s[k];
		k--;
	}

	/* check for linear dependence in stage I */

	if (!test && stage) {
		for (i=0, ptr=a+kr, ptr1=a+in; i<m2; i++, ptr += n2,
			ptr1 += n2) {
			d = *ptr;
			*ptr = *ptr1;
			*ptr1 = d;
		}
		kr++;
	} else if (!test) {
		*(a+m1*n2+n) = 2.0;
		goto end;
	} else {
		pivot = *(a+out*n2+in);
		if (*(a+m*n2+in) - pivot - pivot > toler) {
			for (j=kr, ptr=a+out*n2+kr, ptr1=a+m*n2+kr; j<n1; j++,
				ptr++, ptr1++) {
				d = *ptr;
				*ptr1 -= 2.0*d;
				*ptr = -d;
			}
			*(a+out*n2+n1) *= -1.0;
			goto loop2;
		} else {

			/* pivot on a[out][in]. */

			for (j=kr, ptr=a+out*n2+kr; j<n1; j++, ptr++)
				if (j != in)
					*ptr /= pivot;
			for (i=0, ptr=a+in, ptr1=a; i<m1; i++,
				ptr += n2, ptr1 += n2)
				if (i != out) {
					d = *ptr;
					for (j=kr, ptr2=ptr1+kr,
						ptr3=a+out*n2+kr; j<n1; j++,
						ptr2++, ptr3++)
						if (j != in)
							*ptr2 -= d*(*ptr3);
				}
			for (i=0, ptr=a+in; i<m1; i++, ptr += n2)
				if (i != out)
					*ptr /= -pivot;
			*(a+out*n2+in) = 1.0/pivot;
			d = *(a+out*n2+n1);
			*(a+out*n2+n1) = *(a+m1*n2+in);
			*(a+m1*n2+in) = d;
			kount++;
			if (stage) {

				/* interchange rows in stage I */

				kl++;
				for (j=kr,ptr=a+out*n2+kr,ptr1=a+kount*n2+kr;
					j<n2; j++, ptr++, ptr1++) {
					d = *ptr;
					*ptr = *ptr1;
					*ptr1 = d;
				}
			}
		}
	}
	if (kount + kr == n-1)
		stage = false;
	goto loop1;

	/* prepare for final return */

end:	for (i=0, ptr=a+n1, ptr1=a+n; i<m; i++, ptr += n2, ptr1 += n2) {
		k = (int)(*ptr);
		d = *ptr1;
		if (k < 0) {
			k = -k;
			d = -d;
		}
		k--;
		if (i < kl)
			x[k] = d;
		else {
			k -= n;
			e[k] = d;
		}
	}
	*(a+m1*n2+n1) = (Type)(kount+1);
	*(a+m*n2+n1) = (Type)(n1-kr -1);
	for (i=kl, ptr=a+kl*n2+n, sum=(long double)0.0; i<m; i++, ptr += n2)
		sum += (long double)(*ptr);
	*(a+m*n2+n) = (long double)sum;
	free((char *)s);
}

#define INSTANTIATE_L1(TYPE)\
template class cMemo1LinearEq<TYPE>;\
template class cLinearMemoEq<TYPE>;\
template class cCraig_Barrodale_Roberts_l1<TYPE>;\
template  cLinearOverCstrSys<TYPE> *  AllocL1_Barrodale(size_t aNbVar);

INSTANTIATE_L1(tREAL4)
INSTANTIATE_L1(tREAL8)
INSTANTIATE_L1(tREAL16)


};



