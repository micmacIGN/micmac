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
/* ------------------------- MODULE basis.c ------------------------- */

#include "disable_msvc_warnings.h"

/***********************************************************************
*                                                                      *
* Basic fundtions: Definitions file                                    *
* ---------------------------------                                    *
*                                                                      *
* Programming language: ANSI C                                         *
* Compiler:             Turbo C 2.0                                    *
* Computer:             IBM PS/2 70 with 80387                         *
* Author:               Juergen Dietel, Computer Center, RWTH Aachen   *
* Date:                 8.12.1992                                      *
*                                                                      *
***********************************************************************/


#include "basis.h"  /*  for  NULL, freopen, stdout, fprintf, stderr,  */
                    /*       stdin, SQRT, EXP, sqrt                   */
                    /*       MACH_EPS, POSMAX, epsquad, maxroot, pi,  */
                    /*       ATAN, sqr, umleiten, readln, intervall,  */
                    /*       horner, norm_max, skalprod, copy_vector, */
                    /*       REAL, ONE, TWO, FOUR, ZERO, HALF, FABS,  */
                    /*       boolean, basis, mach_eps, epsroot,       */
                    /*       exp_1, posmin, sqrtlong, comdiv, comabs, */
                    /*       quadsolv, SetVec, CopyVec,               */
                    /*       ReadVec, WriteVec,                       */
                    /*       SetMat, CopyMat, ReadMat,                */
                    /*       WriteMat, WriteHead, WriteEnd, LogError, */
                    /*       fgetc, SWAP                              */



/*--------------------------------------------------------------------*/

int basis(void) /* find basis used for computer number representation */
/*.IX{basis}*/

/***********************************************************************
* Find the basis for representing numbers on the computer, if not      *
* already done.                                                        *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* REAL, ONE, TWO                                                       *
***********************************************************************/

{
  REAL x,
       eins,
       b;

  x = eins = b = ONE;

  while ((x + eins) - x == eins)
    x *= TWO;
  while ((x + b) == x)
    b *= TWO;


  return (int)((x + b) - x);
}



/*--------------------------------------------------------------------*/

static int groesser1(REAL x)         /* aux function for mach_eps() ..*/

/***********************************************************************
* Aux function for mach_eps() (in order to circumvent certain compiler *
* optimizations): return whethter the given x is greater than one.     *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* REAL, ONE                                                            *
***********************************************************************/

{
  return x > ONE;
}



/*--------------------------------------------------------------------*/

REAL mach_eps(void)                 /* Compute machine constant ......*/
/*.IX{mach\unt eps}*/

/***********************************************************************
* Compute the machine constant if not already done.                    *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* REAL, boolean, FALSE, ONE, HALF, TWO, TRUE                           *
***********************************************************************/

{
  static REAL    epsilon;
  static boolean schon_berechnet = FALSE;

  if (! schon_berechnet)
  {
    for (epsilon = ONE; groesser1(ONE + epsilon); )
      epsilon *= HALF;
    epsilon         *= TWO;
    schon_berechnet  = TRUE;
  }

  return epsilon;
}


/*--------------------------------------------------------------------*/

REAL epsroot(void)  /* Compute square root of the machine constant ...*/
/*.IX{epsroot}*/

/***********************************************************************
* Compute square root of the machine constant, if not already done.    *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* REAL, boolean, FALSE, TRUE, SQRT, MACH_EPS                           *
***********************************************************************/

{
  static REAL    save_mach_eps_root;
  static boolean schon_berechnet     = FALSE;

  if (! schon_berechnet)
    schon_berechnet    = TRUE,
    save_mach_eps_root = SQRT(MACH_EPS);

  return save_mach_eps_root;
}


/*--------------------------------------------------------------------*/

REAL epsquad(void)      /* Find the machine constant squared .........*/
/*.IX{epsquad}*/

/***********************************************************************
* Compute the square of the machine constant, if not already done.     *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* REAL, boolean, FALSE, TRUE, MACH_EPS                                 *
***********************************************************************/

{
  static REAL    save_mach_eps_quad;
  static boolean schon_berechnet     = FALSE;

  if (! schon_berechnet)
    schon_berechnet    = TRUE,
    save_mach_eps_quad = MACH_EPS * MACH_EPS;

  return save_mach_eps_quad;
}


/*--------------------------------------------------------------------*/

REAL maxroot(void)    /* Root of the largest representable number ....*/
/*.IX{maxroot}*/

/***********************************************************************
* Compute the square root of the largest machine number 2 ^ (MAX_EXP/2)*
* if not already done                                                  *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* REAL, boolean, FALSE, TRUE, SQRT, POSMAX                             *
***********************************************************************/

{
  static REAL       save_maxroot;
  static boolean    schon_berechnet = FALSE;
  REAL              faktor;
  unsigned long int n;

  if (! schon_berechnet)
  {
    save_maxroot = ONE;
    faktor       = TWO;
    for (n = MAX_EXP / 2; n > 1; n /= 2, faktor *= faktor)
      if (n % 2 != 0)
        save_maxroot *= faktor;
    save_maxroot    *= faktor;
    schon_berechnet  = TRUE;
  }

  return save_maxroot;
}


/*--------------------------------------------------------------------*/

REAL posmin(void)  /* Compute smallest positive floating point number */
/*.IX{posmin}*/

/***********************************************************************
* Find the smallest positive floating point number, if not already done*
* The algorithm halves the number one untilthe process becomes         *
* stationary or equal to zero. In order to avoid an infinite loop, the *
* number of halvings is limited to  32767.                             *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* REAL, boolean, FALSE, ONE, TWO, ZERO, HALF, TRUE                     *
***********************************************************************/

{
  static REAL    y;  /* after completion of loop : smallest floating  */
                     /* point number of the loop;                     */
                     /* inside the loop : 2 * y (to compare with  y)  */
  REAL           x;
  int            i;  /* counter of halvings                           */
  static boolean schon_berechnet = FALSE;

  if (! schon_berechnet)
  {
    for (i = 0, x = ONE, y = TWO; x != ZERO && x != y && i < 32767; i++)
      y =  x,
      x *= HALF;
    schon_berechnet = TRUE;
  }

  return y;
}


/*--------------------------------------------------------------------*/

REAL pi(void)                           /* Compute pi ................*/
/*.IX{pi}*/

/***********************************************************************
* Compute  PI = 3.14 ..., if not already done.                         *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* REAL, boolean, FALSE, TRUE, FOUR, ATAN                               *
***********************************************************************/

{
  static REAL    save_pi;
  static boolean schon_berechnet = FALSE;

  if (! schon_berechnet)
    schon_berechnet = TRUE,
    save_pi         = FOUR * ATAN(ONE);

  return save_pi;
}


/*--------------------------------------------------------------------*/

REAL exp_1(void)                     /* Compute e ....................*/
/*.IX{exp\unt 1}*/

/***********************************************************************
* Compute e = 2.71 ..., if not already done.                           *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* REAL, boolean, FALSE, TRUE, EXP, ONE                                 *
***********************************************************************/

{
  static REAL    save_exp_1;
  static boolean schon_berechnet = FALSE;

  if (! schon_berechnet)
    schon_berechnet = TRUE,
    save_exp_1      = EXP(ONE);

  return save_exp_1;
}


/*--------------------------------------------------------------------*/

REAL Musqr(REAL x)                    /* square a floating point number */
/*.IX{sqr}*/

/***********************************************************************
* Compute the square of a floating point number.                       *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* REAL                                                                 *
***********************************************************************/
{
  return x * x;
}


/*--------------------------------------------------------------------*/

void fehler_melden   /* Write error messages to stdout and stderr ....*/
/*.IX{fehler\unt melden}*/
                  (
                   char text[],          /* error description ........*/
                   int  fehlernummer,    /* Number of error ..........*/
                   char dateiname[],     /* file with error  .........*/
                   int  zeilennummer     /* file name, row number ....*/
                  )

/***********************************************************************
* Record an error message with the name of the generating file and row *
* number, where the error was found, possibly with the error number of *
* the just called function (if  fehlernummer > 0).                     *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* sprintf, fprintf, stderr, printf                                     *
***********************************************************************/

{
  char meldung[200];

  if (fehlernummer == 0)
    sprintf(meldung, "\n%s, Row %d: %s!\n",
                     dateiname, zeilennummer, text);
  else
    sprintf(meldung, "\n%s, Row %d: Error %d in %s!\n",
                     dateiname, zeilennummer, fehlernummer, text);

  fprintf(stderr, "%s", meldung);
  printf("%s", meldung);
}


/*--------------------------------------------------------------------*/

int umleiten            /* Perhaps redirect stdin or stdout to a file */
/*.IX{umleiten}*/
            (
             int argc,       /* number of arguments in command line ..*/
             char *argv[]    /* Vector of arguments ..................*/
            )                /* error code ...........................*/

/***********************************************************************
* Assign an input/output file to the standard stdin and stdout files.  *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* freopen, stdout, NULL, fprintf, stderr, stdin                        *
***********************************************************************/

{
  if (argc >= 3)                           /* at least 2 arguments ?  */
    if (freopen(argv[2], "w", stdout) == NULL)    /* open output file */
    {
      fprintf(stderr, "Error opening %s!\n", argv[2]);
      return 1;
    }
  if (argc >= 2)                           /* at least one argument ? */
    if (freopen(argv[1], "r", stdin) == NULL)      /* open input file */
    {
      fprintf(stderr, "Error in opening %s!\n", argv[1]);
      return 2;
    }

  return 0;
}


/*--------------------------------------------------------------------*/

void readln(void)              /* Skip the remainder of line in stdin */
/*.IX{readln}*/

/***********************************************************************
* Skip the remainder of the line in stdin including the line separator *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* fgetc, EOF                                                           *
***********************************************************************/

{
  int c;

  while ((c = fgetc(stdin)) != '\n' && c != EOF)
    ;
}



/*--------------------------------------------------------------------*/

void getline          /* Read one line from stdin ....................*/
/*.IX{getline}*/
            (
             char kette[],    /* Vector with the read text ...........*/
             int limit        /* maximal length of kette .............*/
            )

/***********************************************************************
* Read one line from stdin into kette, at most, however, limit-1       *
* characters. The remainder of the line, including the line separator, *
* is skipped. limit is the maximal length of kette. kette always       *
* terminates with the zero byte so that there is only room for limit-1 *
* characters in kette.                                                 *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* fgetc, EOF                                                           *
***********************************************************************/

{
  int c;

  for (c = 0; --limit >= 1 && (c = fgetc(stdin)) != '\n' && c != EOF; )
    *kette++ = (char)c;
  *kette = '\0';                       /* terminate with zero byte    */
  while (c != '\n' && c != EOF)        /* skip over remainder in line */
    c = fgetc(stdin);
}


/*--------------------------------------------------------------------*/

int intervall    /* Find the number for a value inside a partition ...*/
/*.IX{intervall}*/
             (
              int n,         /* lenght of partition ..................*/
              REAL xwert,    /* number whose interval index is wanted */
              REAL x[]       /* partition ............................*/
             )               /* Index for xwert ......................*/

/***********************************************************************
* For a given interval partition x[i], i = 0,1,...,n with real         *
* monotonically increasing values for the x[i], we compute the index   *
* ix for which x[ix] <= xwert < x[ix+1] holds.                         *
* If xwert < x[0] or xwert >= x[n-1], we set ix = 0 or ix = n-1.       *
* Thus ix has the return value of between 0 and n-1.                   *
* This is a standard function for use with spline evaluations.         *
*                                                                      *
* Input parameters:                                                    *
* =================                                                    *
* n:     Index of final node of partition                              *
* xwert: value whose index is desired                                  *
* x:     [0..n] vector with the partition                              *
*                                                                      *
* Return value :                                                       *
* ==============                                                       *
* desired index ix for xwert                                           *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* REAL                                                                 *
***********************************************************************/

{
  int ix,
      m;

  for (ix = 0; m = (ix + n) >> 1, m != ix; )
    if (xwert < x[m])
      n = m;
    else
      ix = m;

  return ix;
}


/*--------------------------------------------------------------------*/

REAL horner        /* Horner scheme for polynomial evaluations .......*/
/*.IX{horner}*/
           (
            int n,                         /* Polynomial degree ......*/
            REAL a[],                      /* Polynomial coefficients */
            REAL x                         /* place of evaluation ....*/
           )                               /* Polynomial value at x ..*/

/***********************************************************************
* Evaluate a polynomial P :                                            *
*       P(x)  =  a[0] + a[1] * x + a[2] * x^2 + ... + a[n] * x^n       *
* using the Horner scheme.                                             *
*                                                                      *
* Input parameters:                                                    *
* =================                                                    *
* n: degree of polynomial                                              *
* a: [0..n] coefficient vector for polynomial                          *
* x: place of evaluation                                               *
*                                                                      *
* Return value :                                                       *
* ==============                                                       *
* P(x)                                                                 *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* REAL                                                                 *
***********************************************************************/

{
  REAL summe;

  for (summe = a[n], n--; n >= 0; n--)
    summe = summe * x + a[n];

  return summe;
}


/*--------------------------------------------------------------------*/

REAL norm_max      /* Find the maximum norm of a REAL vector .........*/
/*.IX{norm\unt max}*/
             (
              REAL vektor[],               /* vector .................*/
              int  n                       /* length of vector .......*/
             )                             /* Maximum norm ...........*/

/***********************************************************************
* Return the maximum norm of a [0..n-1] vector  v.                     *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* REAL, FABS, ZERO                                                     *
***********************************************************************/

{
  REAL norm,                                             /* local max */
       betrag;                            /* magnitude of a component */

  for (n--, norm = ZERO; n >= 0; n--, vektor++)
    if ((betrag = FABS(*vektor)) > norm)
      norm = betrag;

  return norm;
}



/* ------------------------------------------------------------------ */

REAL skalprod           /* standard scalar product of two REAL vectors*/
/*.IX{skalprod}*/
        (
         REAL v[],                 /* 1st vector .....................*/
         REAL w[],                 /* 2nd vector .....................*/
         int  n                    /* vector length...................*/
        )                          /* scalar product .................*/

/***********************************************************************
* compute the scalar product   v[0] * w[0] + ... + v[n-1] * w[n-1]  of *
* the two [0..n-1] vectors v and w                                     *
*                                                                      *
* Global names used:                                                   *
* ==================                                                   *
* REAL, ZERO                                                           *
***********************************************************************/

{
  REAL skalarprodukt;

  for (skalarprodukt = ZERO; n-- != 0; )
    skalarprodukt += (*v++) * (*w++);

  return skalarprodukt;
}



/* ------------------------------------------------------------------ */

void copy_vector        /* copy a REAL vector ........................*/
/*.IX{copy\unt vector}*/
                (
                 REAL ziel[],            /* copied vector ............*/
                 REAL quelle[],          /* original vector ..........*/
                 int  n                  /* length of vector .........*/
                )

/***********************************************************************
* copy the n elements of the vector quelle into the vector ziel.       *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* REAL                                                                 *
***********************************************************************/

{
  for (n--; n >= 0; n--)
    *ziel++ = *quelle++;
}



/* -------------------- Albert Becker's functions ------------------- */


static char Separator[] =
"--------------------------------------------------------------------";

long double sqrtlong (long double x)
/*.IX{sqrtlong}*/
/*====================================================================*
 *                                                                    *
 *  Double precision square root                                      *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *   Input parameter:                                                 *
 *   ================                                                 *
 *      x        long double x;                                       *
 *               number whose square root is needed                   *
 *                                                                    *
 *   Return value :                                                   *
 *   =============                                                    *
 *               double precision square root of x                    *
 *====================================================================*/
{
   long double y;
   long double yold;
   int i;

   y = (long double) sqrt ((double) (x));
   for (i = 0; i < 10; i++)
   {
     if (y == 0.0L) return 0.0L;
     yold = y;
     y = (y + x / y) * 0.5L;
     if (ABS (y - yold) <= ABS (y) * MACH_EPS) break;
   }
   return y;
}

int comdiv              /* Complex division ..........................*/
/*.IX{comdiv}*/
           (
            REAL   ar,            /* Real part of numerator ..........*/
            REAL   ai,            /* Imaginary part of numerator .....*/
            REAL   br,            /* Real part of denominator ........*/
            REAL   bi,            /* Imaginary part of denominator ...*/
            REAL * cr,            /* Real part of quotient ...........*/
            REAL * ci             /* Imaginary part of quotient ......*/
           )
/*====================================================================*
 *                                                                    *
 *  Complex division  c = a / b                                       *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *   Input parameters:                                                *
 *   ================                                                 *
 *      ar,ai    REAL   ar, ai;                                       *
 *               Real, imaginary parts of numerator                   *
 *      br,bi    REAL   br, bi;                                       *
 *               Real, imaginary parts of denominator                 *
 *                                                                    *
 *   Output parameters:                                               *
 *   ==================                                               *
 *      cr,ci    REAL   *cr, *ci;                                     *
 *               Real , imaginary parts of the quotient               *
 *                                                                    *
 *   Return value :                                                   *
 *   =============                                                    *
 *      = 0      ok                                                   *
 *      = 1      division by 0                                        *
 *                                                                    *
 *   Macro used :     ABS                                             *
 *   ============                                                     *
 *                                                                    *
 *====================================================================*/
{
  REAL tmp;

  if (br == ZERO && bi == ZERO) return (1);

  if (ABS (br) > ABS (bi))
  {
    tmp  = bi / br;
    br   = tmp * bi + br;
    *cr  = (ar + tmp * ai) / br;
    *ci  = (ai - tmp * ar) / br;
  }
  else
  {
    tmp  = br / bi;
    bi   = tmp * br + bi;
    *cr  = (tmp * ar + ai) / bi;
    *ci  = (tmp * ai - ar) / bi;
 }

 return (0);
}


REAL comabs             /* Complex absolute value ....................*/
/*.IX{comabs}*/
              (
               REAL  ar,          /* Real part .......................*/
               REAL  ai           /* Imaginary part ..................*/
              )
/*====================================================================*
 *                                                                    *
 *  Complex absolute value of   a                                     *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *   Input parameters:                                                *
 *   ================                                                 *
 *      ar,ai    REAL   ar, ai;                                       *
 *               Real, imaginary parts of  a                          *
 *                                                                    *
 *   Return value :                                                   *
 *   =============                                                    *
 *      Absolute value of a                                           *
 *                                                                    *
 *   Macros used :    SQRT, ABS, SWAP                                 *
 *   =============                                                    *
 *                                                                    *
 *====================================================================*/
{
  if (ar == ZERO && ai == ZERO) return (ZERO);

  ar = ABS (ar);
  ai = ABS (ai);

  if (ai > ar)                                  /* Switch  ai and ar */
    SWAP (REAL, ai, ar)

  return ((ai == ZERO) ? (ar) : (ar * SQRT (ONE + ai / ar * ai / ar)));
}


void quadsolv           /* Complex quadratic equation ................*/
/*.IX{quadsolv}*/
             (
               REAL    ar,        /* second degree coefficient .......*/
               REAL    ai,
               REAL    br,        /* linear coefficient ..............*/
               REAL    bi,
               REAL    cr,        /* polynomial constant .............*/
               REAL    ci,
               REAL *  tr,        /* solution ........................*/
               REAL *  ti
             )
/*====================================================================*
 *                                                                    *
 *  Compute the least magnitude solution of the quadratic equation    *
 *  a * t**2 + b * t + c = 0. Here a, b, c and t are complex.         *
 *                                         2                          *
 *  Formeula used: t = 2c / (-b +/- sqrt (b  - 4ac)).                 *
 *  This formula is valid for a=0 .                                   *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  Input parameters:                                                 *
 *  ================                                                  *
 *      ar, ai   coefficient of t**2             REAL   ar, ai;       *
 *      br, bi   coefficient of t                REAL   br, bi;       *
 *      cr, ci   constant term                   REAL   cr, ci;       *
 *                                                                    *
 *  Output parameter:                                                 *
 *  ================                                                  *
 *      tr, ti   complex solution of minimal magnitude                *
 *                                               REAL   *tr, *ti;     *
 *                                                                    *
 *  Macro used :     SQRT                                             *
 *  ============                                                      *
 *                                                                    *
 *====================================================================*/
{
  REAL pr, pi, qr, qi, h;

  pr = br * br - bi * bi;
  pi = TWO * br * bi;                       /*  p = b * b             */

  qr = ar * cr - ai * ci;
  qi = ar * ci + ai * cr;                   /*  q = a * c             */

  pr = pr - (REAL)4.0 * qr;
  pi = pi - (REAL)4.0 * qi;                 /* p = b * b - 4 * a * c  */

  h  = SQRT (pr * pr + pi * pi);            /* q = sqrt (p)           */

  qr = h + pr;
  if (qr > ZERO)
    qr = SQRT (qr * HALF);
  else
    qr = ZERO;

  qi = h - pr;
  if (qi > ZERO)
    qi = SQRT (qi * HALF);
  else
    qi = ZERO;

  if (pi < ZERO) qi = -qi;

  h = qr * br + qi * bi;     /* p = -b +/- q, choose sign for large  */
                             /* magnitude  p                         */
  if (h > ZERO)
  {
    qr = -qr;
    qi = -qi;
  }

  pr = qr - br;
  pi = qi - bi;
  h = pr * pr + pi * pi;                      /* t = (2 * c) / p      */

  if (h == ZERO)
  {
    *tr = ZERO;
    *ti = ZERO;
  }
  else
  {
    *tr = TWO * (cr * pr + ci * pi) / h;
    *ti = TWO * (ci * pr - cr * pi) / h;
  }
}



void SetVec (int n, REAL x[], REAL val)
/*.IX{SetVec}*/
/*====================================================================*
 *                                                                    *
 *  Initialize a vector of length n with constant elements.           *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  Input parameters:                                                 *
 *  ================                                                  *
 *      n        int n;                                               *
 *               length of vector                                     *
 *      x        REAL x[];                                            *
 *               vector                                               *
 *      val      value of constant                                    *
 *                                                                    *
 *   Output parameter:                                                *
 *   ================                                                 *
 *      x        vector with n entries equal to val                   *
 *                                                                    *
 *====================================================================*/
{
  int i;

  for (i = 0; i < n; i++)
    x[i] = val;
}



void CopyVec (int n, REAL source[], REAL dest[])
/*.IX{CopyVec}*/
/*====================================================================*
 *                                                                    *
 *  Copy the vector  source of length  n ont the vector  dest .       *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  Input parameters:                                                 *
 *  ================                                                  *
 *                                                                    *
 *      n        int n;                                               *
 *               length of vector                                     *
 *      source   REAL source[];                                       *
 *               original vector                                      *
 *      dest     REAL dest[];                                         *
 *               Vector to be copied to                               *
 *                                                                    *
 *   Ausgabeparameter:                                                *
 *   ================                                                 *
 *      dest     same as above                                        *
 *                                                                    *
 *   ATTENTION: no storage is allocated for dest here.                *
 *                                                                    *
 *====================================================================*/
{
  int i;

  for (i = 0; i < n; i++)
    dest[i] = source[i];
}


int ReadVec (int n, REAL x[])
/*.IX{ReadVec}*/
/*====================================================================*
 *                                                                    *
 *  Read vector x of length n from stdin.                             *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  INput parameters:                                                 *
 *  ================                                                  *
 *                                                                    *
 *      n        int n;                                               *
 *               length of vector                                     *
 *      x        REAL x[];                                            *
 *               vector                                               *
 *                                                                    *
 *   Output parameter:                                                *
 *   ================                                                 *
 *      x        read vector                                          *
 *                                                                    *
 *   Attention: no storage is allocated for x here.                   *
 *                                                                    *
 *====================================================================*/
{
  int i;
  double tmp;

  for (i = 0; i < n; i++)
  {
    if (scanf (FORMAT_IN, &tmp) <= 0) return (-1);
    x[i] = (REAL) tmp;
  }

  return (0);
}


int WriteVec (int n, REAL x[])
/*.IX{WriteVec}*/
/*====================================================================*
 *                                                                    *
 *  Put out vector of length x to  stdout.                            *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  Input parameters:                                                 *
 *  ================                                                  *
 *                                                                    *
 *      n        int n;                                               *
 *               lenght of vector                                     *
 *      x        REAL x[];                                            *
 *               vector                                               *
 *                                                                    *
 *   Return value :                                                   *
 *   =============                                                    *
 *      =  0     All ok                                               *
 *      = -1     Error putting out to stdout                          *
 *                                                                    *
 *====================================================================*/
{
  int i;

  for (i = 0; i < n; i++)
    if (printf (FORMAT_126LF, x[i]) <= 0) return (-1);
  if (printf ("\n") <= 0) return (-1);

  return 0;
}



void SetMat (int m, int n, REAL * a[], REAL val)
/*.IX{SetMat}*/
/*====================================================================*
 *                                                                    *
 *  Initialize an m x n matrix with a constant value val .            *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  Input parameters:                                                 *
 *  ================                                                  *
 *      m        int m; ( m > 0 )                                     *
 *               row number of matrix                                 *
 *      n        int n; ( n > 0 )                                     *
 *               column number of matrix                              *
 *      a        REAL * a[];                                          *
 *               matrix                                               *
 *      val      constant value                                       *
 *                                                                    *
 *   Output parameter:                                                *
 *   ================                                                 *
 *      a        matrix with constant value val in every position     *
 *                                                                    *
 *====================================================================*/
{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      a[i][j] = val;
}


void CopyMat (int m, int n, REAL * source[], REAL * dest[])
/*.IX{CopyMat}*/
/*====================================================================*
 *                                                                    *
 *  Copy the m x n matrix source to the  m x n matrix dest.           *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  Input parameters:                                                 *
 *  ================                                                  *
 *      m        int m; ( m > 0 )                                     *
 *               numnber of rows of matrix                            *
 *      n        int n; ( n > 0 )                                     *
 *               number of columns of matrix                          *
 *      source   REAL * source[];                                     *
 *               matrix                                               *
 *      dest     REAL * dest[];                                       *
 *               matrix to be copied to                               *
 *                                                                    *
 *   Output parameter:                                                *
 *   ================                                                 *
 *      dest     same as above                                        *
 *                                                                    *
 *   ATTENTION : WE do not allocate storage for dest here.            *
 *                                                                    *
 *====================================================================*/
{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      dest[i][j] = source[i][j];
}


int ReadMat (int m, int n, REAL * a[])
/*.IX{ReadMat}*/
/*====================================================================*
 *                                                                    *
 *  Read an m x n matrix from stdin.                                  *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  Input parameters :                                                *
 *  ==================                                                *
 *      m        int m; ( m > 0 )                                     *
 *               number of rows of matrix                             *
 *      n        int n; ( n > 0 )                                     *
 *               column number of  matrix                             *
 *      a        REAL * a[];                                          *
 *               matrix                                               *
 *                                                                    *
 *   Output parameter:                                                *
 *   ================                                                 *
 *      a        matrix                                               *
 *                                                                    *
 *   ATTENTION : WE do not allocate storage for a here.               *
 *                                                                    *
 *====================================================================*/
{
  int i, j;
  double x;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
    {
      if (scanf (FORMAT_IN, &x) <= 0) return (-1);
      a[i][j] = (REAL) x;
    }

  return (0);
}


int WriteMat (int m, int n, REAL * a[])
/*.IX{WriteMat}*/
/*====================================================================*
 *                                                                    *
 *  Put out an m x n matrix in stdout .                               *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  Input parameters:                                                 *
 *  ================                                                  *
 *      m        int m; ( m > 0 )                                     *
 *               row number of matrix                                 *
 *      n        int n; ( n > 0 )                                     *
 *               column number of matrix                              *
 *      a        REAL * a[];                                          *
 *               matrix                                               *
 *                                                                    *
 *   Return value :                                                   *
 *   =============                                                    *
 *      =  0      all put out                                         *
 *      = -1      Error writing onto stdout                           *
 *                                                                    *
 *====================================================================*/
{
  int i, j;

  if (printf ("\n") <= 0) return (-1);

  for (i = 0; i < m; i++)
  {
    for (j = 0; j < n; j++)
      if (printf (FORMAT_126LF, a[i][j]) <= 0) return (-1);

    if (printf ("\n") <= 0) return (-1);
  }
  if (printf ("\n") <= 0) return (-1);

  return (0);
}


int WriteHead (char * string)
/*.IX{WriteHead}*/
/*====================================================================*
 *                                                                    *
 *  Put out header with text in string in stdout.                     *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  Input parameters:                                                 *
 *  ================                                                  *
 *      string   char *string;                                        *
 *               text of headertext (last byte is a 0)                *
 *                                                                    *
 *   Return value :                                                   *
 *   =============                                                    *
 *      =  0      All ok                                              *
 *      = -1      Error writing onto stdout                           *
 *      = -2      Invalid text for header                             *
 *                                                                    *
 *====================================================================*/
{
  if (string == NULL) return (-2);

  if (printf ("\n%s\n%s\n%s\n\n", Separator, string, Separator) <= 0)
    return (-1);

  return 0;
}


int WriteEnd ()
/*.IX{WriteEnd}*/
/*====================================================================*
 *                                                                    *
 *  Put out end of writing onto  stdout.                              *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *   Return value :                                                   *
 *   =============                                                    *
 *      =  0      All ok                                              *
 *      = -1      error writing onto stdout                           *
 *                                                                    *
 *====================================================================*/
{
  if (printf ("\n%s\n\n", Separator) <= 0) return (-1);
  return 0;
}


void LogError (char * string, int rc, char * file, int line)
/*.IX{LogError}*/
/*====================================================================*
 *                                                                    *
 *  Put out error message onto  stdout.                               *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  Input parameters:                                                 *
 *  ================                                                  *
 *      string   char *string;                                        *
 *               text of error massage (final byte is 0)              *
 *      rc       int rc;                                              *
 *               error code                                           *
 *      file     char *file;                                          *
 *               name of C file in which error was encountered        *
 *      line     int line;                                            *
 *               line number of C file with error                     *
 *                                                                    *
 *====================================================================*/
{
  if (string == NULL)
  {
    printf ("Unknown ERROR in file %s at line %d\n", file, line);
    WriteEnd ();
    return;
  }

  if (rc == 0)
    printf ("ERROR: %s, File %s, Line %d\n", string, file, line);
  else
    printf ("ERROR: %s, rc = %d, File %s, Line %d\n",
             string, rc, file, line);

  WriteEnd ();
  return;
}

/* --------------------------- END basis.c -------------------------- */

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
