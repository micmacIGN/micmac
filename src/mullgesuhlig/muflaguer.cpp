/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr

   
    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in 
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte 
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
/* ------------------------- MODULE flaguer.c ----------------------- */

/***********************************************************************
*                                                                      *
* Compute all real roots of a polynomial with real coefficients        *
* ---------------------------------------------------------------      *
* with the method of Laguerre                                          *
* ---------------------------                                          *
*                                                                      *
* exported function:                                                   *
*   - laguerre():  Laguerre's method for finding all real roots of a   *
*                  real polynomial                                     *
*                                                                      *
* Programming language: ANSI C                                         *
* Compiler:             Borland C++ 2.0                                *
* Computer:             IBM PS/2 70 mit 80387                          *
* Author:               Gisela Engeln-Muellges                         *
* Adaption:             Juergen Dietel, Rechenzentrum, RWTH Aachen     *
* Source:               FORTRAN source code                            *
* Date:                 10. 9. 1992                                    *
*                                                                      *
***********************************************************************/


#include "basis.h"      /* for   REAL, ZERO, FOUR, FABS, SQRT, TWO,   */
                        /*       MACH_EPS, copy_vector, SIGN          */
#include "vmblock.h"    /* for   vminit, vmalloc, VEKTOR, vmcomplete, */
                        /*       vmfree                               */
#include "flaguer.h"    /* for   laguerre                             */



/* ------------------------------------------------------------------ */

static void abdiv                              /* factor off one root */
/*.IX{abdiv}*/
                 (
                  int  n,      /* degree of the polynomial ...........*/
                  REAL a[],    /* old and new polynomial .............*/
                  REAL x0      /* root to be factored off ............*/
                 )

/***********************************************************************
* Compute the coefficients of the lower degree polynomial pab of degree*
* n-1 from the given polynomial p of degree n.                         *
*                                                                      *
* Input parameters:                                                    *
* =================                                                    *
* n     degree of the polynomial p                                     *
* a     [0..n] vector of coefficienten of the polynomial p with        *
*           p(x)  =  a[0] + a[1] * x + ... + a[n] * x^n                *
* x0    root of p; p is divided by the linear factor  x - x0           *
*                                                                      *
* Output parameters:                                                   *
* =================                                                    *
* a     [0..n] vector with the new coefficients of pab:                *
*           pab(x)  =  a[1] + a[2] * x + ... + a[n] * x^(n-1)          *
*           ( p(x) = pab(x) * (x - x0) )                               *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* REAL                                                                 *
***********************************************************************/

{
  REAL tmp;                      /* intermediate sum in Horner scheme */

  for (tmp = a[n--]; n >= 0; n--)
    a[n] = tmp = tmp * x0 + a[n];
}



/* ------------------------------------------------------------------ */

static void horner2      /* Horner scheme with derivatives p, p', p'' */
/*.IX{horner2}*/
                   (
                    int  n,       /* degree of p .................... */
                    REAL a[],     /* Polynomial coefficients (ascend.)*/
                    REAL x,       /* point of evaluation .............*/
                    REAL *p0,     /* p(x) ............................*/
                    REAL *p1,     /* p'(x) ...........................*/
                    REAL *p2,     /* p''(x) ..........................*/
                    REAL hilf[]   /* remainder polynomials ...........*/
                   )

/***********************************************************************
* Compute the functional value and that of the first and second        *
* derivative of a given polynomial p of degree n.                      *
*                                                                      *
* Input parameters:                                                    *
* =================                                                    *
* n     degree of  p                                                   *
* a     [0..n] vector of coefficients of p:                            *
*           p(x)  =  a[0] + a[1] * x + ... + a[n] * x^n                *
* x     location where p, p', p'' are to be computed                   *
*                                                                      *
* Output parameters:                                                   *
* =================                                                    *
* hilf  [0..n-1] auxiliary vector with hilf[0], hilf[1], hilf[2] con-  *
*       taining the values of p(x), p'(x), p''(x) and whose entries at *
*       positions 3, ..., (n-1) contain intermediate results from the  *
*       Horner scheme                                                  *
* p0    p(x)                                                           *
* p1    p'(x)                                                          *
* p2    p''(x)                                                         *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* REAL, copy_vector                                                    *
***********************************************************************/

{
  int  i,                   /* loop counter for the Horner scheme     */
       k;                   /* counter for derivative                 */
  REAL tmp;                 /* intermediate sum in the Horner scheme  */


  copy_vector(hilf, a, n);

  for (k = 0; k < 3; k++)                   /* compute kth derivative */
    for (tmp = a[n], i = n - 1; i >= k; i--)
      hilf[i] = tmp = tmp * x + hilf[i];

  *p0 =       hilf[0];
  *p1 =       hilf[1];
  *p2 = TWO * hilf[2];
}



/* ------------------------------------------------------------------ */

static int quaglei           /* real solution of a quadratic equation */
/*.IX{quaglei}*/
                  (
                   REAL a[],      /* coefficients (ascending order) ..*/
                   REAL eps,      /* machine constant ................*/
                   REAL *x1,      /* 1st root ........................*/
                   REAL *x2       /* 2nd root ........................*/
                  )               /* error code ......................*/

/***********************************************************************
* Solve the quadratic equation                                         *
*              a[2] * x^2 + a[1] * x + a[0]  =  0                      *
* for a[2] not zero, provided it has only real roots                   *
*                                                                      *
* Input parameters:                                                    *
* =================                                                    *
* a     [0..2] vector with the three real of the quadratic equation    *
* eps   Error bound for recognizing a numerical zero                   *
*                                                                      *
* Output parameters:                                                   *
* =================                                                    *
* x1 \  the two real roots                                             *
* x2 /  of the quadratic equation                                      *
*                                                                      *
* Function value:                                                      *
* ==============                                                       *
* Error code.                                                          *
* = 0: all is ok                                                       *
* = 1: a[2] = 0, or there are complex roots.                           *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* REAL, ZERO, FOUR, FABS, SQRT, TWO                                    *
***********************************************************************/

{
  REAL diskr,            /* Discriminant of the quadratic equation    */
       wurzel;           /* square root of the discriminant           */


  if (a[2] == ZERO)                  /* not a quadratic equation?     */
    return 1;

  diskr = a[1] * a[1] - FOUR * a[2] * a[0];

  if (FABS(diskr) <= eps)   /* modulus of the discriminant too small? */
    diskr = ZERO;               /* set discrimainant = 0              */
  if (diskr < ZERO)             /* discriminant negative?             */
    return 1;                   /* eror: complex roots!               */

  wurzel = SQRT(diskr);

  if (a[1] < ZERO)                              /* compute the real   */
    *x1 = (-a[1] + wurzel) / (TWO * a[2]);      /* roots without      */
  else                                          /* encountering       */
    *x1 = (-a[1] - wurzel) / (TWO * a[2]);      /* cancellation       */

  if (*x1 == ZERO)
    *x2 = -a[1] / a[2];
  else
    *x2 = a[0] / (*x1 * a[2]);


  return 0;
}



/* ------------------------------------------------------------------ */
/*.BA*/

int laguerre             /* real polynomial roots, Method of Laguerre */
/*.IX{laguerre}*/
            (
             int  n,         /* Polynomial degree (>= 3) .............*/
             REAL a[],       /* Polynomial coefficients (ascending) ..*/
             REAL abserr,    /* absolute error bound .................*/
             REAL relerr,    /* relative error bound .................*/
             int  maxit,     /* maximal number of iterations .........*/
             REAL x[],       /* real roots ...........................*/
             int  Giter[],    /* Iterations per root ..................*/
             int  *nulanz    /* Number of found roots ................*/
            )                /* Error code ...........................*/

/***********************************************************************
* Compute all roots of a real polynomial p of degree n that has only   *
* roots by using the method of Laguerre.                               *
.BE*)
*                                                                      *
* Input parameters:                                                    *
* =================                                                    *
* n        degree of the polynomial p (>= 3)                           *
* a        [0..n] vector of coefficients of p :                        *
*              p(x)  =  a[0] + a[1] * x + ... + a[n] * x^n             *
* abserr\  Error bounds which must be nonnegative.                     *
* relerr/  Their sum must exceed zero. The following mixed test is used*
*          where x1 and x2 are successive approximations:              *
*              |x1 - x2|  <=  |x2| * relerr + abserr.                  *
*          For relerr = 0, we test the absolute error, for abserr = 0  *
*          we test for the relative error.                             *
*          The input values for abserr and/or relerr are used only if  *
*          they exceed four times the machine constant; otherwise      *
*          the offending error bound is set internally to this value.  *
* maxit    maximal number of iterations per root (maxit >= 1)          *
*                                                                      *
* Output parameters:                                                   *
* =================                                                    *
* x        [0..n-1] vector with the n real roots of p                  *
* iter     [0..n-1] vector containing the number of iterations for     *
*          each root: iter[i] contains this number for x[i].           *
* nulanz   Number of roots found                                       *
*                                                                      *
* Return value :                                                       *
* ==============                                                       *
* Error code.                                                          *
* = 0: All roots have been found.                                      *
* = 1: inadmissable input parameters abserr, relerr, maxit or n        *
* = 2: The maximal number of iterations maxit has been reached. The    *
*      roots found thus far are in x[0],...,x[nulanz-1].               *
* = 3: The value of S in Laguerre's method is negative, so the square  *
*      cannot be found over the reals.                                 *
* = 4: When computing the last two roots with the function quaglei()   *
*      there are difficulties: the remainder equation is not quadratic *
*      or it has complex conjugate roots.                              *
* = 5: inadequate memory space                                         *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* REAL, ZERO, vminit, vmalloc, VEKTOR, vmcomplete, vmfree, TWO,        *
* MACH_EPS, copy_vector, horner2, FABS, SQRT, SIGN, abdiv, quaglei     *
.BA*)
***********************************************************************/
/*.BE*/

{
  REAL *p_akt,     /* [0..nakt] vector of coefficients of the actual  */
                   /* polynomial, resulting from factoring out known  */
                   /* roots.                                          */
                   /* After factoring out one linear factor for each  */
                   /* root, the pointer p_akt is raised by one in     */
                   /* order to address the remainder polynomial as    */
                   /* the previous one.                               */
       x_akt,      /* new approximation for the actual root           */
       p0,         /* Polynomial value at x_akt (actual)              */
       p1,         /* derivative of the actual polynomial at x_akt    */
       p2,         /* 2nd derivative at x_akt                         */
       S,          /* (nakt-1) * ((nakt-1) * p1*p1 - nakt * p0*p2)    */
       WurzelS,    /* Square root of S                                */
       diff,       /* Step size between two successive root           */
                   /* approximation                                   */
       nenner,     /* deniminator in the step size                    */
       x1, x2,     /* roots of the final quadratic equation           */
       eps;        /* machine constant used to correct error bounds   */
                   /* abserr and relerr or to recognize a numerical   */
                   /* zero                                            */
  int  nakt,       /* degree of the actual remainder polynomial       */
       iter_akt,   /* iteration counter for the actual root           */
       i;          /* loop counter for the  n-2 roots of the          */
                   /* Laguerre iteration                              */
  void *vmblock;   /* List of dynamically allocated vectors           */


  /* -- check abserr, relerr, maxit and n --------------------------- */

  if (abserr < ZERO || relerr < ZERO || abserr + relerr <= ZERO ||
      maxit < 1 || n <= 2)
    return 1;


  /* ------------- allocate a dynamic vector ------------------------ */

  vmblock = vminit();
  p_akt = (REAL *)vmalloc(vmblock, VEKTOR, n + 1, 0);
  if (! vmcomplete(vmblock))
    return 5;


  /* --------- adjust error bounds if necessary --------------------- */

  eps = TWO * MACH_EPS;
  if (relerr == ZERO && abserr < eps)
    abserr = eps;
  else if (abserr == ZERO && relerr < eps)
    relerr = eps;
  else
  {
    if (abserr < eps)
      abserr = eps;
    if (relerr < eps)
      relerr = eps;
  }


  /* ---------- loop to compute n-2 rootsN -------------------------- */

  copy_vector(p_akt,                  /* start with the given         */
              a, n + 1);              /* polynomial as the actual one */

  for (*nulanz = 0, nakt = n, i = 0; i < n - 2; i++, nakt--, p_akt++)
  {

    x_akt    = ZERO;     /* initialize starting value for the root    */
    iter_akt = 0;        /* and iteration counter                     */

    do                 /* start iteration to find the subsequent root */
    {
      if (iter_akt >= maxit)                     /* too many steps?   */
      {
        vmfree(vmblock);
        return 2;                                /* return eror       */
      }

      iter_akt++;                          /* count iterations        */
      horner2(nakt, p_akt, x_akt,          /* evaluate polynomial and */
              &p0, &p1, &p2, x + i);       /* derivative at x_akt     */

      S = (nakt - 1) * ((nakt - 1) * p1 * p1 - nakt * p0 * p2);
      if (FABS(S) <= eps)
        S = ZERO;
      if (S < ZERO)                                  /* is S negative?*/
      {
        vmfree(vmblock);
        return 3;                                    /* return error  */
      }
      WurzelS = SQRT(S);
      WurzelS = SIGN(WurzelS, p1);   /* choose the sign of the square */
                                     /* of S so that the denominator  */
                                     /* becomes as large as possible  */

      nenner = p1 + WurzelS;
      if (FABS(nenner) < eps)                /* denominator too small?*/
        nenner = SIGN(eps, nenner);          /* assign it the value   */
                                             /* of eps                */

      diff  =  nakt * p0 / nenner;          /* compute step size and  */
      x_akt -= diff;                        /* apply to  x_akt        */

    }
    while (FABS(diff) >                    /* accuracy not reached?   */
           FABS(x_akt) * relerr + abserr);


    x[i]    = x_akt;              /* store the root and the step size */
    if (Giter) Giter[i] = iter_akt; /* and up root counter */
    ++*nulanz;

    abdiv(nakt, p_akt, x_akt);          /* form the remainder         */
                                        /* polynomial                 */
  }                       /* End of the loop for the first n-2 roots  */


  /* -------- the two remaining roots are found from the quadratic -- */
  /* -------- remainder equation ------------------------------------ */
  /* -------- p_akt[2] * x^2 + p_akt[1] * x + p_akt[0]  =  0 -------- */

  if (quaglei(p_akt, eps, &x1, &x2))                       /* Error? */
  {
    vmfree(vmblock);
    return 4;
  }

  x[n - 2]    =  x1;                     /* store the final two roots */
  x[n - 1]    =  x2;                     /* and up counter            */
  if (Giter)
  {
     Giter[n - 2] =  0;
     Giter[n - 1] =  0;
  }
  *nulanz     += 2;


  vmfree(vmblock);
  return 0;
}

/* ------------------------- END flaguer.c -------------------------- */

int laguerre (int  n,double  a[],double  x[])
{
    int nb=0;
    int res = laguerre(n,a,1e-5,1e-7,10,x,(int *)0,&nb);
	printf("RES LAG = %d\n",res);
    return nb;
}              

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
