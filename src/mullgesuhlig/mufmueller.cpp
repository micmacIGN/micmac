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
/*.BA*/
/*.KA{C 3}{Roots of Polynomials}
          {Roots of Polynomials}*/
/*.FE{C 3.3.2}{Muller's Method}{Muller's Method}*/

/*.BE*/
/* ------------------------ MODULE fmueller.c ----------------------- */



#include "basis.h"
#include "u_proto.h"

//#include "StdAfx.h"

#define SABS(A,B) (ABS(A) + ABS(B))     /* Sum of the absolute values */

#define ITERMAX 50000                 /* maximal number of iterations */
                                        /* increase for polynomials   */
                                        /* of large degree            */
#define START (REAL)0.125          /* Starting value for skaleit = 0  */

/*.BA*/

int mueller             /* Mueller's method for real polynomials .....*/
/*.IX{mueller}*/
            (
             int   n,             /* degree of the polynomial ........*/
             REAL  a[],           /* vector of coefficients ..........*/
             int   scaleit,       /* Scaling control .................*/
             REAL  zreal[],       /* Real part of the solution .......*/
             REAL  zimag[]        /* Imaginary part of the solution ..*/
            )
/*====================================================================*
 *                                                                    *
 *  muller determines all real and complex roots of a polynomial P    *
 *  of degree n given as                                              *
 *                     n             n-1                              *
 *      P(x) = a[n] * x  + a[n-1] * x    + ... + a[1] * x + a[0],     *
 *                                                                    *
 *  with  a[i], i=0, ..., n, real.                                    *
 *                                                                    *
 *  The starting values for Mueller's method are initially set up     *
 *  via the constant START = 0.125. This value is based on experience,*
 *  other values can be assigned to START as desired.                 *
 *                                                                    *
 *====================================================================*
.BE*)
 *                                                                    *
 *   Applications:                                                    *
 *   ============                                                     *
 *      Find roots of arbitrary polynomials with real coefficients.   *
 *      Multiple roots usually are located inside a small circle      *
 *      around the true root. The true multiple root is approximately *
 *      at the mean of the computed approximate solutions in the      *
 *      small circle.                                                 *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *   Literature:                                                      *
 *   ==========                                                       *
 *      Mueller, D.E., A method for solving algebraic equations       *
 *      using an automatic computer, Math. Tables Aids Comp. 10,      *
 *      p. 208-251, (1956).                                           *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *   Input parameters:                                                *
 *   ================                                                 *
 *      n        degree of the polynomial ( >= 1 )    int    n;       *
 *      a        Vector of polynomial coefficients    REAL   a[];     *
 *               ( a[0], ..., a[n] )                                  *
 *      scaleit  = 0, no scaling desired              int    scaleit; *
 *               != 0 automatic scaling                               *
 *                                                                    *
 *   Output parameters:                                               *
 *   =================                                                *
 *      zreal    real parts of the roots,          REAL   zreal[];    *
 *               zreal[0],..,zreal[n-1]                               *
 *      zimag    zimag[0],..,zimag[n-1],           REAL   zreal[];    *
 *               the imaginary parts of the computed roots            *
 *                                                                    *
 *   Return value:                                                    *
 *   ============                                                     *
 *      = 0      all is ok                                            *
 *      = 1      Unadmissable input                                   *
 *      = 2      Maximal number of iterations ITERMAX exceeded        *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *   Functions used:                                                  *
 *   ==============                                                   *
 *                                                                    *
 *     fmval():     Evaluates the funftion value of the current       *
 *                  polynomial                                        *
 *     quadsolv():  Solves a quadratic polynomial with complex        *
 *                  coefficients                                      *
 *                                                                    *
 *   From the  C-library: pow()                                       *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *   Constants used: MACH_EPS, EPSROOT, EPSQUAD, ITERMAX, START       *
 *   ==============                                                   *
 *                                                                    *
 *====================================================================*/
{
  /*register*/ int i;
  int iu, iter;

  REAL  p, q, temp, scale, start, zrealn,
        x0real, x0imag, x1real, x1imag, x2real,x2imag,
        f0r,    f0i,    f1r,    f1i,    f2r,   f2i,
        h1real, h1imag, h2real, h2imag, hnr,   hni,
        fdr,    fdi,    fd1r,   fd1i,   fd2r,  fd2i,
        fd3r,   fd3i,
        b1r,    b1i,
        pot;

  if (zreal == NULL || zimag == NULL || a == NULL) return (1);

  for (i = 0; i < n; i++)
  {
    zreal[i] = a[i];                        /*  copy a to zreal       */
    zimag[i] = ZERO;                        /*  set zimag equal to 0  */
  }

  if ((n <= 0) || (ABS(a[n]) <= ZERO))
    return (1);                             /* wrong input parameter  */

  scale = ZERO;              /* Scale polynomial, if ( scaleit != 0 ) */

  if (scaleit != 0)          /* scale                                 */
  {                          /*               a[i]  1/(n-i)           */
    p = ABS(a[n]);           /*  =  max{ ABS( ---- )    ,i=0,...,n-1} */
    for (i = 0; i < n; i++)  /*               a[n]                    */
      if (zreal[i] != ZERO)
      {
        zreal[i] /= p;
        pot = POW (ABS(zreal[i]), ONE / (REAL) (n - i));
        scale = max (scale, pot);
      }

    zrealn = a[n] / p;                        /* zrealn = +/-1        */

                             /*                      n-i              */
    if ( scale != ONE &&     /* a[i] = a[i] / ( scale    ), i=0..n-1  */
         scale != ZERO )
      for (p = ONE, i = n - 1; i >= 0; i--)
      {
        p *= scale;
        zreal[i] /= p;
      }
  }    /* end if (scaleit.. */
  else
  {
    scale = ONE;
    zrealn = a[n];
  }

  iu = 0;

  do
  {   /*  Muellerverfahren bis iu == n-1  */

    while (ABS(zreal[iu]) < EPSQUAD)          /* zero solution of the */
    {                                         /* remainder polynomial */
      zreal[iu] = zimag[iu] = ZERO;
      iu++;
      if (iu == n - 1) break;
    }

    if (iu >= n - 1)                       /* If iu == n-1 --> Done   */
    {
      zreal[n-1] *= -scale / zrealn;
      zimag[n-1] = ZERO;
      return (0);
    }

    if (scaleit)                           /* If scaling, recompute   */
    {                                      /* starting value          */
      for (start = ZERO, i = n - 1; i >= iu; i--)
        start = max (start, ABS(zreal[i]));

      start /= (REAL)128.0;            /* All roots lie inside the    */
    }                                  /* circle with center (0,0)    */
    else                               /* and radius                  */
      start = START;                   /* r = 1 + max{ABS(a[i]),i=..} */

    iter = 0;                     /* initialize iteration counter     */

    x0real = -start;              /* Starting values for Mueller      */
    x0imag = ZERO;

    x1real = start;
    x1imag = ZERO;

    x2real = ZERO;
    x2imag = ZERO;

    h1real = x1real - x0real; h1imag = ZERO;    /*  h1 = x1 - x0      */
    h2real = x2real - x1real; h2imag = ZERO;    /*  h2 = x2 - x1      */

    f0r = zrealn;   f0i = ZERO;      /* corresponding function values */
    f1r = f0r;      f1i = ZERO;

    for (i = n; i > iu; )
    {
      f0r = f0r * x0real + zreal[--i];
      f1r = f1r * x1real + zreal[i];
    }

    f2r = zreal[iu];
    f2i = ZERO;

    fd1r = (f1r - f0r) / h1real;      /* 1st divided difference       */
    fd1i = ZERO;                      /* fd1 = (f1 - f0) / h1         */

    do /* Mueller-Iteration */
    {
      if ( SABS(f0r,f0i) < EPSQUAD             /* Starting value is a */
           || SABS(f1r,f1i) < EPSQUAD )        /* good approximation  */
      {
        x1real = x0real;
        x1imag = x0imag;

        f2r = f0r;
        f2i = f0i;
        break;
      }
                                      /* 1st divided difference       */
                                      /* fd2 = (f2 - f1) / h2         */
      temp = h2real * h2real + h2imag * h2imag;
      fdr = f2r - f1r;
      fdi = f2i - f1i;

      fd2r = ( fdr * h2real + fdi * h2imag ) / temp;
      fd2i = ( fdi * h2real - fdr * h2imag ) / temp;

      fdr = fd2r - fd1r;             /* 2nd divided difference        */
      fdi = fd2i - fd1i;             /* fd3 = (fd2 - fd1) / (h1 + h2) */

      hnr = h1real + h2real; hni = h1imag + h2imag;
      temp = hnr * hnr + hni * hni;
      fd3r = ( fdr * hnr + fdi * hni ) / temp;
      fd3i = ( fdi * hnr - fdr * hni ) / temp;

      b1r = h2real * fd3r - h2imag * fd3i + fd2r;       /*  h2 * f3   */
      b1i = h2real * fd3i + h2imag * fd3r + fd2i;

      h1real = h2real;                 /* latest correction, store    */
      h1imag = h2imag;
                                       /* compute new correction      */
      if ( (fd3r != ZERO) || (fd3i != ZERO) ||
           (b1r != ZERO)  || (b1i != ZERO)     )
        quadsolv (fd3r, fd3i, b1r, b1i, f2r, f2i, &h2real, &h2imag);
      else
      {
        h2real = HALF;
        h2imag = ZERO;
      }

      x1real =  x2real;                /* store old solution          */
      x1imag =  x2imag;
      x2real += h2real;                /* compute new solution:       */
      x2imag += h2imag;                /* x2 = x2 + h2                */

      f1r  = f2r;                      /* update function values      */
      f1i  = f2i;
      fd1r = fd2r;
      fd1i = fd2i;

      fmval (n, iu, zreal, zrealn, x2real, x2imag, &f2r, &f2i);

                           /* Avoid ineffective directions and        */
                           /* overflow                                */
      i = 0;
      while (SABS(f2r,f2i) > n * SABS(f1r,f1i))
      {
                                        /* watch for underflow        */
        if (i > 10) break;
        else
          i++;

        h2real *= HALF;                 /* half h; update x2,f2      */
        h2imag *= HALF;

        x2real -= h2real;
        x2imag -= h2imag;

        fmval (n, iu, zreal, zrealn, x2real, x2imag, &f2r, &f2i);
      }

      iter++;
      if (iter > ITERMAX) return (2);     /* ITERMAX exceeded         */

    }                                     /* end Muller iteration     */

    while ( (SABS(f2r,f2i) > EPSQUAD) &&
             (SABS(h2real,h2imag) > MACH_EPS * SABS(x2real,x2imag)) );

    if (SABS(f1r,f1i) < SABS(f2r,f2i)) /* choose better approximation */
    {
      x2real = x1real;
      x2imag = x1imag;
    }

    if (ABS(x2imag) > EPSROOT * ABS(x2real))
    {                                     /* Factor off a complex     */
                                          /* root and its conjugate   */
      p  = x2real + x2real;
      q  = -x2real * x2real - x2imag * x2imag;

      zreal[n-1] += p * zrealn;
      zreal[n-2] += p * zreal[n-1] + q * zrealn;

      for (i = n - 3; i > iu + 1; i--)
        zreal[i] += p * zreal[i+1] + q * zreal[i+2];

      x2real *= scale;
      x2imag *= scale;

      zreal [iu+1] =  x2real;
      zimag [iu+1] =  x2imag;
      zreal [iu]   =  x2real;
      zimag [iu]   = -x2imag;
      iu += 2;                              /* reduce degree by 2     */
    }
    else
    {
      zreal[n-1] += zrealn * x2real;        /* Factor off a real      */
                                            /* root                   */
      for (i = n - 2; i > iu; i--)
        zreal[i] += zreal[i+1] * x2real;

      zreal[iu] = x2real * scale;
      zimag[iu] = ZERO;
      iu++;                                 /* reduce degree by 1     */
    }
  }
  while (iu < n);                           /* End of Muller's method */

  return (0);
}


void fmval              /* (Complex) polynomial value ................*/
/*.IX{fmval}*/
           (
            int     n,            /* Maximal degree present ..........*/
            int     iu,           /* Lowest degree present ...........*/
            REAL    zre[],        /* Koefficients ....................*/
            REAL    zren,         /* Leading coefficient .............*/
            REAL    xre,          /* Real part of x ..................*/
            REAL    xim,          /* Imaginary part of x .............*/
            REAL *  fre,          /* Real part of the function value .*/
            REAL *  fim           /* Imaginary part of function value */
           )
/*====================================================================*
 *                                                                    *
 *  fmval evaluates the value of a polynomial of degree n-iu          *
 *  with real coefficients zre[iu], ..., zre[n-1], zren at the        *
 *  complex point (xre, xim).                                         *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  Input parameters:                                                 *
 *  ================                                                  *
 *      zre    Vector of coefficients           REAL   zre[];         *
 *      zren   leading coefficient              REAL   zren;          *
 *      xre    Real part of x                   REAL   xre;           *
 *      xim    Imaginary part of x              REAL   xim;           *
 *                                                                    *
 *  Output parameters:                                                *
 *  =================                                                 *
 *      fre    Real part of the polynomial value        REAL   *fre:  *
 *      fim    Imaginary part of the polynomial value   REAL   *fim;  *
 *                                                                    *
 *====================================================================*/
{
  /*register*/ int i;
  REAL     tmp;

  *fre = zren;
  *fim = ZERO;

  if (xim == ZERO)            /* x and the polynomial value are real  */
    for (i = n; i > iu; )
      *fre = *fre * xre + zre[--i];
  else
    for (i = n; i > iu; )   /* x and the polynomial value are complex */
    {
      tmp  = *fre;
      *fre = *fre * xre - *fim * xim + zre[--i];
      *fim = tmp * xim + xre * *fim;
    }
}

/* ------------------------- END fmueller.c ------------------------- */


void mueller ( int   n, REAL  a[], REAL  zreal[], REAL  zimag[])
{
    int res = mueller(n,a,0,zreal,zimag);
    if (res !=0)
    {
        printf("Error %d in Muller\n",res);
		exit(-1);
        //ElEXIT(-1,"mueller");
    }
}



// ./usr/lib/gcc-lib/i386-redhat-linux/egcs-2.90.29/include/float.h                


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
