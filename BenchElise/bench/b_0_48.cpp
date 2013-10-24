/*eLiSe06/05/99
  
     Copyright (C) 1999 Marc PIERROT DESEILLIGNY

   eLiSe : Elements of a Linux Image Software Environment

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  Author: Marc PIERROT DESEILLIGNY    IGN/MATIS  
Internet: Marc.Pierrot-Deseilligny@ign.fr
   Phone: (33) 01 43 98 81 28
eLiSe06/05/99*/


#include "StdAfx.h"
#include "bench.h"

extern void Craig_etal_L1 
            ( 
                   Im2D_REAL8  A,Im1D_REAL8  B,REAL TOLER,
                   Im1D_REAL8  SOL, Im1D_REAL8  RESIDU
            );


bool Optim_L1FormLin::ExploreChVARBov
     (
        ElFilo<INT> & SubSet,
        REAL	    & sc_min,
        INT kv
     )
{
     if (! SubSet[kv])
        return false;
     
     bool got = false;
     INT Kmin = -1;
     SubSet[kv] = 0;

     for (INT kf =0; kf<_NbForm ; kf++)
     {
         if ((kf != kv) && (!SubSet[kf]))
         {
            SubSet[kf] = 1;
            REAL sc = score(SubSet);
            if (sc < sc_min)
            {
               got = true;
               Kmin = kf;
               sc_min = sc;
               _BestSol = _Sol;
            }
            SubSet[kf] = 0;
         }
     }

     if (got)
     {
        SubSet[Kmin] = 1;
     }
     else
        SubSet[kv] = 1;

     return got;
}

Optim_L1FormLin Optim_L1FormLin::RandOLF(INT NbVar,INT NbForm,INT Nb0)
{
   ElMatrix<REAL> M(NbVar+1,NbForm);
   for (INT l=0 ; l < NbForm ; l++)
   {
       for (INT c=0 ; c < NbVar ; c++ )
           M(c,l) = NRrandom3()-0.5;
       M(NbVar,l) = 50*(NRrandom3()-0.5);
       if (l < Nb0)
          M(NbVar,l) = 0;
   }

   return Optim_L1FormLin(M);
}




void Optim_L1FormLin::MinCombin
     (
        ElFilo<INT> & CurSubset,
        ElFilo<INT> & BestSet,
        REAL &        ScMin,
        INT           NbVarPos,
        INT           CurVarPos
     )
{

     if (NbVarPos == _NbVar)
     {
        REAL sc = score(CurSubset);
        if (sc < ScMin)
        {
             ScMin = sc;
             copy_on(BestSet,CurSubset);
        }
        return;
     }
     if (CurVarPos == _NbForm)
        return;

     CurSubset[CurVarPos] = 1;
     MinCombin(CurSubset,BestSet,ScMin,NbVarPos+1,CurVarPos+1);
     CurSubset[CurVarPos] = 0;
     MinCombin(CurSubset,BestSet,ScMin,NbVarPos  ,CurVarPos+1);
}

REAL Optim_L1FormLin::MinCombin()
{
    ElFilo<INT> CurSubset,BestSet;
    for (INT k=0; k<_NbForm ; k++)
        CurSubset.pushlast(0);
    REAL ScMin = 1e60;
    MinCombin(CurSubset,BestSet,ScMin,0,0);
    return score(BestSet);
}

void  Optim_L1FormLin::BenchCombin(REAL ScS)
{
   if (_bench_comb_made)
      return;
   _bench_comb_made = true;

   REAL ScComb = MinCombin();

   BENCH_ASSERT(ScS - ScComb <= epsilon);
}

void Optim_L1FormLin::BenchRand(INT NbVar,INT NbForm,INT NbTest,bool Comb)
{
   Optim_L1FormLin OLF = RandOLF(NbVar,NbForm);

   ElMatrix<REAL>  Sol = OLF.Solve();
   REAL ScS = OLF.score(Sol);


   for (;NbTest>=0; NbTest --)
   {
      REAL eps = ElMax(1e-3,0.2*ElSquare(NRrandom3()));
      ElMatrix<REAL>  D = Sol;
      for (INT k=0 ; k<NbVar ; k++)
          D(0,k) += eps * (NRrandom3()-0.5);

       REAL sd = OLF.score(D);

       if (ScS> sd) 
       {
          if (Comb)
              OLF.BenchCombin(ScS);
          cout << ScS << " " << sd 
               << " " << ((ScS-sd)/eps) 
               << " " << eps << "\n";
          BENCH_ASSERT(ElAbs((ScS-sd)/eps) < 1e-7);
       }
   }
}




void Optim_L1FormLin::BenchRandComb(INT NbVar,INT NbForm)
{
   Optim_L1FormLin OLF = RandOLF(NbVar,NbForm);

   ElMatrix<REAL>  Sol = OLF.Solve();
   REAL ScS = OLF.score(Sol);
   OLF.BenchCombin(ScS);
}


void Optim_L1FormLin::SubsetOfFlags(ElFilo<INT> & Subset,INT flag)
{
     Subset.clear();
     for (INT k=0; k<_NbForm ; k++)
         if (flag & (1 << k))
            Subset.pushlast(1);
         else
            Subset.pushlast(0);
     
}


void Optim_L1FormLin::CombinConjMinLoc
     (
        ElFilo<REAL>&  dic,
        ElFilo<INT> &  Subset,
        ElFilo<INT> &  FlagPos,
        INT            FlagSubset,
        INT            NbVarPos,
        INT            CurVarPos
     )
{
/*
     show_flag(FlagSubset);
     cout << " " << NbVarPos << " " << CurVarPos 
          << " [N= " << _NbVar << " M = " << _NbForm << "]"
          << "\n";
*/
     

     if (NbVarPos == _NbVar)
     {
        FlagPos.pushlast(FlagSubset);
        SubsetOfFlags(Subset,FlagSubset);
        dic[FlagSubset] = score(Subset);

        return;
     }
     if (CurVarPos == _NbForm)
        return;

     CombinConjMinLoc(dic,Subset,FlagPos,FlagSubset | (1<<CurVarPos),NbVarPos+1,CurVarPos+1);
     CombinConjMinLoc(dic,Subset,FlagPos,FlagSubset                 ,NbVarPos  ,CurVarPos+1);
}

void Optim_L1FormLin::show_flag(INT flag)
{
     for (INT k=0; k<_NbForm ; k++)
        cout << ((flag & (1<<k))  ? "+" : "-") ;
}


REAL Optim_L1FormLin::TestNeighConjMinLoc
     (
        INT            FlagSubset,
        ElFilo<REAL>&  dic
     )
{
      REAL res = 1e80;

      for (INT k1 =0; k1< _NbForm ; k1++)
      {
          if (FlagSubset & (1<<k1))
          {
              for (INT k2 =0; k2< _NbForm ; k2++)
              {
                   if (! (FlagSubset & (1<<k2)))
                   {
                        INT NewF = (FlagSubset & (~(1<<k1))) | (1<< k2);
                        res = ElMin(res,dic[NewF]);
/*
                        cout << "  [-]" << k1  << " [+]" << k2 << " : ";
                        show_flag(NewF);
                        cout << "\n";
*/
                   }
              }
          }
      }
      return res;
}


static INT CPT = 0;

void Optim_L1FormLin::CombinConjMinLoc
     (
        ElFilo<REAL>&  dic,
        ElFilo<INT> &  Subset,
        ElFilo<INT> &  FlagPos
     )
{
    CPT ++;
    INT NbF = (1<<_NbForm);
    for (INT k=dic.nb() ;k<NbF ; k++)
        dic.pushlast(1e20);

    FlagPos.clear();

    CombinConjMinLoc(dic,Subset,FlagPos,0,0,0);

    INT NbMin0 =0;
    INT NbMin1 =0;

    REAL M0 = 1e80,M1 = -1e80;

	{
    for (INT k=0 ; k< FlagPos.nb(); k++)
    {


        REAL VN = TestNeighConjMinLoc(FlagPos[k],dic);
        REAL V0 = dic[FlagPos[k]];

        if (V0 < VN)
           NbMin0 ++;
        if (V0 <=  VN)
        {
           NbMin1 ++;
           ElSetMin(M0,V0);
           ElSetMax(M1,V0);
        }

/*
        {
          show_flag(FlagPos[k]);
          cout << " : " << dic[FlagPos[k]];
          if (V0 <= VN) 
             cout << " ** ";
          cout << "\n";
        }
*/
    }
	}

/*
    if ((NbMin0!=1) || (NbMin1!=1))
        cout << NbMin0 << " " << NbMin1 << "\n";
*/

    BENCH_ASSERT(NbMin0 <= 1);
    BENCH_ASSERT(NbMin1 >= 1);

   cout << "MINS = " << M0 << " " << M1 << "\n";
   BENCH_ASSERT(ElAbs(M0-M1)<epsilon);


   {
    for (INT k=0 ; k< FlagPos.nb(); k++)
        dic[FlagPos[k]] = 1e20;

   }
}

void Optim_L1FormLin::CombinConjMinLoc
     (
           INT NbVar,
           INT NbForm,
           ElFilo<REAL> & TheDic,
           ElFilo<INT>  & TheSubset,
           ElFilo<INT>  & TheFlagPos,
           INT            NB0
     )
{
   Optim_L1FormLin OLF = RandOLF(NbVar,NbForm,NB0);
   OLF.CombinConjMinLoc(TheDic,TheSubset,TheFlagPos);
}



void Optim_L1FormLin::CombinConjMinLoc()
{
     ElFilo<REAL> TheDic     ;
     ElFilo<INT>  TheSubset  ;
     ElFilo<INT>  TheFlagPos ;

	 {
     for (INT NB=0; false ; NB++)
     {
        CombinConjMinLoc(2,4,TheDic,TheSubset,TheFlagPos,3);
/*
         for (INT F=0; F<30 ; F++)
             for (INT N=2; N <= 4 ; N++)
                 for (INT M= N +1; M<10; M++)
                 {
                     CombinConjMinLoc(N,M,TheDic,TheSubset,TheFlagPos,N+1);
                     CombinConjMinLoc(N,M,TheDic,TheSubset,TheFlagPos,N+2);
                     CombinConjMinLoc(N,M,TheDic,TheSubset,TheFlagPos,N+3);
                 }

         for (INT F=0; F<10 ; F++)
cout << NB << "\n";
*/
     }
	 }

	 {
     for (INT NB=0; false ; NB++)
     {
         CombinConjMinLoc(5,15+NB%5,TheDic,TheSubset,TheFlagPos);
         for (INT F=0; F<30 ; F++)
             for (INT N=2; N <= 4 ; N++)
                 for (INT M= N +1; M<10; M++)
                     CombinConjMinLoc(N,M,TheDic,TheSubset,TheFlagPos);

		{
         for (INT F=0; F<10 ; F++)
         {
             CombinConjMinLoc(2,20,TheDic,TheSubset,TheFlagPos);
             CombinConjMinLoc(3,20,TheDic,TheSubset,TheFlagPos);
         }
		}
		{
         for (INT F=0; F<4 ; F++)
              CombinConjMinLoc(4,20,TheDic,TheSubset,TheFlagPos);
		}


         cout << NB << " ; Tot = " << CPT << "\n";
     }
	 }
}




void Optim_L1FormLin::BenchRandComb()
{
     for (INT NB=0; NB < 200 ; NB++)
     {
         for (INT N=2; N <= 4 ; N++)
         {
             for (INT M= N +1; M<10; M++)
                 BenchRandComb(N,M);
         }
     }
}



void Optim_L1FormLin::bench()
{
     bench_craig();
     CombinConjMinLoc();
     BenchRandComb();


     BenchRand(20,100,1000,false);
     BenchRand(6,500,1000,false);

     for (INT k=0 ; k<5 ; k++)
         BenchRand(6,50,100,false);

	 {
    for (INT k=0; k<3 ; k++)
    {
        BenchRand(4,10,10000,true);
        BenchRand(2,20,1000,true);
        BenchRand(3,20,500,true);
        BenchRand(4,20,500,true);
        BenchRand(5,15,500,true);
    }
	 }
    cout << "end Optim_L1FormLin::bench()\n";
}


//++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++
//  BENCH BENCH BENCH BENCH BENCH BENCH BENCH
//  BENCH BENCH BENCH BENCH BENCH BENCH BENCH
//  BENCH BENCH BENCH BENCH BENCH BENCH BENCH
//++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++

void BenchGaussjPrec(INT N,INT M)
{
    GaussjPrec  GP(N,M);

    for (INT y=0; y<N ; y++)
    {
        for (INT x=0; x<N ; x++)
            GP.M()(x,y) = NRrandom3()-0.5 + 2 * N * (x==y);
		{
        for (INT x=0; x<M ; x++)
            GP.b()(x,y) = NRrandom3();
		}
    }

    bool OK = GP.init_rec();
    BENCH_ASSERT(OK);
	{
    for (INT y=0; y<N ; y++)
        for (INT x=0; x<N ; x++)
            GP.M()(x,y) +=  (NRrandom3()-0.5) * N * 5e-2;
	}

    for (INT k=0 ; k<10 ; k++)
    {
      GP.amelior_sol();
    }
    BENCH_ASSERT(GP.ecart()<epsilon);
}

void BenchGaussjPrec()
{
    for (INT N = 1 ; N < 20 ; N++)
        for (INT M = 1 ; M < 30 ; M+=5)
            BenchGaussjPrec(N,M);
}




void bench_optim_1()
{

   Optim_L1FormLin::bench();
   BenchGaussjPrec();
    cout << "OK  Optim_L1FormLin::bench  BenchGaussjPrec\n";
}


/*********************************************************/
/*********************************************************/
/*********************************************************/
/*********            CRAIG                       ********/
/*********************************************************/
/*********************************************************/
/*********************************************************/

/*
  Big must be set equal to any very large double precision constant.
  It's value here is appropriate for the Sun.
*/
#define BIG 1.0e37

	/*
	  stdlib.h - used to define the return value of malloc() and free().
	*/

#include <stdlib.h>

#if (0)

void Craig_Barrodale_Roberts_l1(INT m,INT  n,REAL * a,REAL *b,REAL toler,REAL * x,REAL * e)
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
		   a[m+1][n] = exit code with values:
				0 - optimal solution which is probably
					non-unique.
				1 - unique optimal solution.
				2 - calculations terminated prematurely
					due to rounding errors.
		   a[m+1][n+1] = number of simplex iterations performed.
		   b = garbage on output.
		   x = one dimensional double precision array of length
			n.  On exit, this array contains a solution to
			the L1 problem.
		   e = one dimensional double precision array of size n.
			On exit, this array contains the residuals in
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

	double min, max, *ptr, *ptr1, *ptr2, *ptr3;
	double  d=0, pivot;
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
		*ptr = (double)(j+1);
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
		*(a+m*n2+j) = (double)sum;
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
			if (fabs(*ptr)<=(double)n && (d=fabs(*ptr1))>max) {
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
	*(a+m1*n2+n1) = (double)(kount+1);
	*(a+m*n2+n1) = (double)(n1-kr -1);
	for (i=kl, ptr=a+kl*n2+n, sum=(long double)0.0; i<m; i++, ptr += n2)
		sum += (long double)(*ptr);
	*(a+m*n2+n) = (long double)sum;
	free((char *)s);
}


void Craig_etal_L1
     (
             Im2D_REAL8  A,
             Im1D_REAL8  B,
             REAL        TOLER,
             Im1D_REAL8  SOL,
             Im1D_REAL8  RESIDU
     )
{
    INT n = SOL.tx();
    INT m = B.tx();

    BENCH_ASSERT
    (
           (A.tx() == n+2)
        && (A.ty() == m+2)
        && (B.tx() == m)
        && (SOL.tx() == n)
        && (RESIDU.tx() == m)
    );
    Craig_Barrodale_Roberts_l1
    (
        m,n,
        A.data_lin(),
        B.data(),
        TOLER,
        SOL.data(),
        RESIDU.data()
    );
}

#endif

void Optim_L1FormLin::One_bench_craig()
{
/*

  ElMatrix<REAL> Smpd = MpdSolve();
  ElMatrix<REAL> SBar = BarrodaleSolve();

  for (INT v=0; v<_NbVar ; v++)
  {
       cout <<  SBar(0,v) << " " 
            << Smpd(0,v) <<  " " 
            << (SBar(0,v)-Smpd(0,v)) *1e10 <<"\n";
  }
*/
  ElTimer t;
  BarrodaleSolve();

   cout << "Time = " << t.sval() << " " << _NbVar << " " << _NbForm << "\n";
}

void Optim_L1FormLin::rand_bench_craig(INT N,INT M)
{
     Optim_L1FormLin OLF = RandOLF(N,M);
     OLF.One_bench_craig();
}

void Optim_L1FormLin::bench_craig()
{

     // rand_bench_craig(2,10000);


     rand_bench_craig(2,10);

     rand_bench_craig(5,1000);
     rand_bench_craig(2,1000);

     // rand_bench_craig(25,100);
/*
     for (INT k= 3 ; k< 60 ; k++)
         rand_bench_craig(k,5000);
*/
}







