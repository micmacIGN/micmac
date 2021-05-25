#include "bool.h"

/*
  Big must be set equal to any very large double precision constant.
  It's value here is appropriate for the Sun.
*/
#define BIG 1.0e37

	/*
	  stdlib.h - used to define the return value of malloc() and free().
	*/

#include <stdlib.h>

void
l1(m, n, a, b, toler, x, e)
int m, n;
double *a, *b, toler, *x, *e;
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
	double fabs(), d, pivot;
	long double sum;
	int out, *s, m1, n1, m2, n2, i, j, kr, kl, kount, in, k, l;
	boolean stage, test;

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
