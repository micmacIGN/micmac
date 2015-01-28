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
/*.KA{C 0}{Auxiliary Declarations}{Auxiliary Declarations}*/
/*.FE{C 0.1}
     {Basic Declarations and Definitions}
     {Basic Declarations and Definitions}*/

/* ----------------------- DECLARATIONS basis.h --------------------- */

/***********************************************************************
*                                                                      *
* Basic functions: Declarations file (with types and macros)           *
* ----------------------------------------------------------           *
*                                                                      *
* Programming language: ANSI C                                         *
* Compiler:             Turbo C 2.0                                    *
* Computer:             IBM PS/2 70 with 80387                         *
* Author:               Juergen Dietel, Computer Center, RWTH Aachen   *
* Date:                 9.30.1992                                      *
*                                                                      *
***********************************************************************/



/***********************************************************************
* Make preparations in case this declarations file is included more    *
* than once in a source code                                           *
***********************************************************************/

#ifndef BASIS_H_INCLUDED
#define BASIS_H_INCLUDED



/***********************************************************************
* Include a few other declarations file here.                          *
* We include the standard declarations files here mainly so that only  *
* this file needs to be included in the C modules of the Numerical     *
* Library in order to have access to all standard names and variables. *
* Moreover modifications for nonstandard compilers may then be limited *
* to this file only.                                                   *
***********************************************************************/

#include <stdio.h>    /*  for  NULL, printf, scanf, fprintf, stderr,  */
                      /*       freopen, stdin, stdout, Fopen, Fclose, */
                      /*       Fclose, fseek, SEEK_END, SEEK_SET,     */
                      /*       ftell, fwrite, fread, size_t, getchar  */
#include <stdlib.h>   /*  for  abs                                    */

#ifdef __EMX__            /* emx 0.9b with GNU CC 2.7.2?              */
#if __GNUC__ == 2         /* To prevent a compiler crash in           */
#if __GNUC_MINOR__ == 7   /* `15\gax.c' while compiling with LDOUBLE, */
#define EMX09B            /* by defining the macro _WITH_UNDERSCORE   */
#define _WITH_UNDERSCORE  /* the special variants of the mathematical */
#define fabsl _fabsl      /* functions for `long double' in `math.h'  */
#define sqrtl _sqrtl      /* are activated and their use further      */
#define powl  _powl       /* below in shape of the macros FABS, SQRT, */
#define sinl  _sinl       /* ... is prepared here; which seems quite  */
#define cosl  _cosl       /* useful even without the affair with the  */
#define expl  _expl       /* crashing compiler.                       */
#define logl  _logl
#define atanl _atanl
#define acosl _acosl
#define coshl _coshl
#endif
#endif
#endif

#include <math.h>     /*  for  fabs, sqrt, pow, exp, sin, cos, log,   */
                      /*       atan, acos                             */
#include <float.h>    /*  for  DBL_EPSILON, DBL_MAX                   */
#ifdef sun            /*  for  GNU CC under SunOS                     */
#include <unistd.h>   /*  for  SEEK_END                               */
#else
#ifdef amigados       /*  for  GNU CC on an Amiga                     */
#include <unistd.h>   /*  for  SEEK_END                               */
#endif
#endif
#ifndef SEEK_END      /* SEEK_END not yet defined (such as for        */
                      /* GNU CC 2.1 for an i386 MS-DOS computer)?     */
#include <unistd.h>   /*  for  SEEK_END                               */
#endif



/***********************************************************************
* Predefine the desired precision for floating point arithmetic:       *
* If the macro FLOAT has been defined, we compute in single precision  *
* (data type float), for LDOUBLE we compute in maximal precision       *
* (data type long double), otherwise we use double precision (data type*
* double). LDOUBLE only adds precision on those compilers for which    *
* long double is different from double such as on Turbo C, but not on  *
* QuickC compilers for example.                                        *
* For input and output of floating numbers the macros LZS (length      *
* prefix for scanf()) and LZP (length prefix for printf()) should be   *
* used (obviously two different ones), since according to the ANSI C   *
* Standard the output of double values should be done without the      *
* length declaration "l" in contrast to the input where this "l" is    *
* needed.                                                              *
* NOTE:    If the compiler used does not know the macros FLT_MAX,      *
* ====     LDBL_MAX, DBL_MAX or FLT_MAX_EXP, LDBL_MAX_EXP, DBL_MAX_EXP *
*          (which can usually be found in float.h), the user must      *
*          insert suitable values at the locations marked with !!!!.   *
***********************************************************************/

#ifdef FLOAT                     /* single precision ? ...............*/

typedef float     REAL;          /* Standard floating point type float*/
/*.IX{REAL}*/
typedef double    LONG_REAL;     /* more precise floating point type  */
/*.IX{LONG\unt REAL}*/

#ifdef FLT_EPSILON               /* ANSI C compiler ? ................*/
#define MACH_EPS  (REAL)FLT_EPSILON
/*.IX{MACH\unt EPS}*/
#else                            /* not an ANSI C compiler ? .........*/
#define MACH_EPS  mach_eps()     /* machine constant .................*/
#endif

#ifdef FLT_MAX_EXP               /* ANSI C compiler? .................*/
#define MAX_EXP   FLT_MAX_EXP    /* Binary exponent of POSMAX ........*/
/*.IX{MAX\unt EXP}*/
#else                            /* not an ANSI C compiler ? .........*/
#define MAX_EXP   128            /* make adjustments here !!!! .......*/
#endif

#ifdef FLT_MAX                   /* ANSI C compiler? ................ */
#define POSMAX    (REAL)FLT_MAX  /* largest floating point number ... */
/*.IX{POS\unt MAX}*/
#else                            /* not an ANSI C compiler ? ........ */
#define POSMAX    1e38f          /* adjust here !!!! ................ */
#endif

#ifdef FLT_MIN                   /* ANSI C compiler? .................*/
#define POSMIN    (REAL)FLT_MIN  /* smallest positive floating point  */
/*.IX{POS\unt MIN}*/
                                 /* number ...........................*/
#else                            /* not an ANSI C compiler ? .........*/
#define POSMIN    posmin()
#endif

#define LZS       ""             /* length prefix for formatted       */
/*.IX{LZS}*/
                                 /* input of floating point numbers ..*/
#define LZP       ""             /* length prefix for formatted       */
/*.IX{LZP}*/
                                 /* output of floating point numbers .*/


#else
#ifdef LDOUBLE                   /* maximal precision ? ..............*/

typedef long double  REAL;       /* Standard floating point type      */
                                 /* long double                       */
/*.IX{REAL}*/
typedef long double  LONG_REAL;  /* "more precise" floating point type*/
/*.IX{LONG\unt REAL}*/
#define LONG_DOUBLE_USED

#ifdef LDBL_EPSILON              /* ANSI C compiler? .................*/
#define MACH_EPS  (REAL)LDBL_EPSILON
/*.IX{MACH\unt EPS}*/
#else                            /* not an ANSI C compiler ? .........*/
#define MACH_EPS  mach_eps()     /* machine constant .................*/
#endif

#ifdef LDBL_MAX_EXP              /* ANSI C compiler? .................*/
#define MAX_EXP   LDBL_MAX_EXP   /* Binary exponent of POSMAX ........*/
/*.IX{MAX\unt EXP}*/
#else                            /* not an ANSI C compiler ? .........*/
#define MAX_EXP   1023           /* adjust here !!!! .................*/
#endif

#ifdef LDBL_MAX                  /* ANSI C compiler? .................*/
#define POSMAX    (REAL)LDBL_MAX /* largest floating point number ....*/
/*.IX{POS\unt MAX}*/
#else                            /* not an ANSI C compiler ? .........*/
#define POSMAX    1e100l         /* adjust here !!!! .................*/
#endif

#ifdef LDBL_MIN                  /* ANSI C compiler? .................*/
#define POSMIN    (REAL)LDBL_MIN /* smallest positive floating point  */
/*.IX{POS\unt MIN}*/                /* number ...........................*/
#else                            /* not an  ANSI C compiler ? ........*/
#define POSMIN    posmin()
#endif

#define LZS       "L"            /* length prefix for formatted       */
/*.IX{LZS}*/
                                 /* input of floating point numbers ..*/
#define LZP       "L"            /* length prefix for formatted       */
/*.IX{LZP}*/
                                 /* output of floating point numbers .*/


#else                            /* double precision ? ...............*/

typedef double       REAL;       /* Standard floating point type      */
                                 /* double                            */
/*.IX{REAL}*/
typedef long double  LONG_REAL;  /* more precise floating point type  */
/*.IX{LONG\unt REAL}*/

#ifdef DBL_EPSILON               /* ANSI C compiler? .................*/
#define MACH_EPS  (REAL)DBL_EPSILON
/*.IX{MACH\unt EPS}*/
#else                            /* not an ANSI C compiler ? .........*/
#define MACH_EPS  mach_eps()     /* machine constant .................*/
#endif


#undef  MACH_EPS
#define MACH_EPS (DBL_EPSILON * 1000)

#ifdef DBL_MAX_EXP               /* ANSI C compiler? .................*/
#define MAX_EXP   DBL_MAX_EXP    /* Binary exponent of POSMAX ........*/
/*.IX{MAX\unt EXP}*/
#else                            /* not an ANSI C compiler ? .........*/
#define MAX_EXP   1023           /* adjust here !!!! .................*/
#endif

#ifdef DBL_MAX                   /* ANSI C compiler? .................*/
#define POSMAX    (REAL)DBL_MAX  /* largest floating point number ....*/
/*.IX{POS\unt MAX}*/
#else                            /* not an ANSI C compiler ? .........*/
#define POSMAX    1e100          /* adjust here !!!! .................*/
#endif

#ifdef DBL_MIN                   /* ANSI C compiler? .................*/
#ifdef __BORLANDC__
#if __BORLANDC__ <= 0x0200       /* Borland C++ 2.0 for DOS? .........*/
                                           /* because the value       */
#define POSMIN    2.2250738585072017E-308  /* 2.2250738585072014E-308 */
#else                                      /* from `float.h' is       */
                                           /* regarded as zero!       */
#define POSMIN    DBL_MIN        /* smallest positive floating point  */
#endif                           /* number ...........................*/
#else
#define POSMIN    DBL_MIN        /* smallest positive floating point  */
#endif
/*.IX{POS\unt MIN}*/                /* number ...........................*/
#else                            /* not an ANSI C compiler ? .........*/
#define POSMIN    posmin()
#endif

#define LZS       "l"            /* length prefix for formatted       */
/*.IX{LZS}*/
                                 /* input of floating point numbers ..*/
#define LZP       ""             /* length prefix for formatted       */
/*.IX{LZP}*/
                                 /* output of floating point numbers .*/
#endif
#endif



/***********************************************************************
* declare several important data types                                 *
***********************************************************************/

#undef boolean
#undef FALSE
#undef TRUE
typedef enum {FALSE, TRUE} boolean;
/*.IX{FALSE}*/
/*.IX{TRUE}*/
/*.IX{boolean}*/

/* function pointer types for approximation in chapter 8 .............*/
typedef REAL (*ansatzfnk) (int i, REAL x);
/*.IX{ansatzfnk}*/
typedef REAL (*approxfnk) (REAL c[], REAL x);
/*.IX{approxfnk}*/
typedef void (*ableitfnk) (REAL x, REAL c[], REAL *d);
/*.IX{ableitfnk}*/

/* Type of function, which evaluates the right hand side of an .......*/
/* explicit ordinary differential equation  y' = f(x,y)        .......*/
typedef REAL (*dglfnk)(REAL x, REAL y);
/*.IX{dglfnk}*/


/* Type of function, which evaluates the right hand side .............*/
/* of a system of differential equations  y' = f(x,y)    .............*/
typedef void (*dglsysfnk)(REAL x, REAL y[], REAL f[]);
/*.IX{dglsysfnk}*/

/* Type of function, which computes the boundary value r(ya, yb) .....*/
/* of a two-point first order boundary value problem             .....*/
typedef void (*rndbedfnk)(REAL ya[], REAL yb[], REAL r[]);
/*.IX{rndbedfnk}*/

/* enumeration type for classification of error codes that are  ......*/
/* returned from most of the functions that realize a numerical ......*/
/* method                                                       ......*/
typedef enum { KEIN_FEHLER, WARNUNG, UNBEKANNT, FATAL } fehler_t;
/*.IX{KEIN\unt FEHLER}*/
/*.IX{WARNUNG}*/
/*.IX{UNBEKANNT}*/
/*.IX{FATAL}*/
/*.IX{fehler\unt t}*/

typedef REAL abl_mat1[4][2];     /* used to evaluate splinefunctions  */
/*.IX{abl\unt mat1}*/
typedef REAL abl_mat2[6][2];     /* in spliwert ......................*/
/*.IX{abl\unt mat2}*/

typedef REAL mat4x4[4][4];       /* type for bicubic splines          */

/*--------------------------------------------------------------------*
 * Type declarations by Albert Becker                                 *
 *--------------------------------------------------------------------*/

 /* Real functions ...................................................*/
 typedef REAL (* REALFCT)  (REAL);
/*.IX{REALFCT}*/

 /* Real multi-dimensional functions .................................*/
 typedef int (* FNFCT)  (int, REAL [], REAL []);
/*.IX{FNFCT}*/

 /* Functions for finding the Jacobi matrix ..........................*/
 typedef int (* JACOFCT)  (int, REAL [], REAL * []);
/*.IX{JACOFCT}*/



/***********************************************************************
* define sevaral important macros                                      *
* NOTE:    Borland C++ offers floating point standard functions        *
*          suitable for long double type from version 3.0 on such as   *
*          expl() instead of exp(), sinl() instead of sin() etc. But   *
*          Borland C++ 3.0 does not seem to define a macro that lets   *
*          the user differentiate it from  Borland C++ 2.0 and below,  *
*          the user is forced to do so himself: If using Borland C++   *
*          3.0 and long double the user should define the macro BC3    *
*          before compiling. Only in this case will the new maximally  *
*          precise floating point functions be used.                   *
***********************************************************************/

#define BASIS     basis()         /* Basis of number representation   */
/*.IX{BASIS}*/
#define EPSROOT   epsroot()       /* square root of MACH_EPS          */
/*.IX{EPSROOT}*/
#define EPSQUAD   epsquad()       /* square of MACH_EPS               */
/*.IX{EPSQUAD}*/
#define MAXROOT   maxroot()       /* square root of the largest       */
/*.IX{MAXROOT}*/
                                  /* floating point number            */
#ifndef PI
#define PI        pi()            /* pi = 3.14...                     */
/*.IX{PI}*/
#endif
#define EXP_1     exp_1()         /* e = 2.71...                      */
/*.IX{EXP\unt 1}*/

#define ZERO      (REAL)0.0       /* declare names for recurring      */
/*.IX{ZERO}*/
#define ONE       (REAL)1.0       /* floating point numbers           */
/*.IX{ONE}*/
#define TWO       (REAL)2.0
/*.IX{TWO}*/
#define THREE     (REAL)3.0
/*.IX{THREE}*/
#define FOUR      (REAL)4.0
/*.IX{FOUR}*/
#define FIVE      (REAL)5.0
/*.IX{FIVE}*/
#define SIX       (REAL)6.0
/*.IX{SIX}*/
#define EIGHT     (REAL)8.0
/*.IX{EIGHT}*/
#define NINE      (REAL)9.0
/*.IX{NINE}*/
#define TEN       (REAL)10.0
/*.IX{TEN}*/
#define HALF      (REAL)0.5
/*.IX{HALF}*/

#ifdef __BORLANDC__
#if __BORLANDC__ >= 0x0400          /* a BC distinguishable from BC2  */
                                    /* and with long double functions */
                                    /* (at least Borland C++ 3.1 or   */
                                    /* Borland C++ 1.00 for OS/2)?    */
#define BC3                         /* define `BC3' automatically     */
#endif
#endif

#ifdef _MSC_VER
#if _MSC_VER     >= 0x0258          /* a MC distinguishable from QC2  */
                                    /* and with long double functions */
                                    /* (at least Microsoft C 6.00A)?  */
#define MC6                         /* define MC6 automatically       */
#endif
#endif

#if defined(LDOUBLE) &&                     /* Borland C++ 3.0 or    */\
    (defined(BC3) || defined(MC6) ||        /* Microsoft C 6.0 or    */\
     defined(EMX09B))                       /* emx 0.9b (GCC272) with */
                                            /* maximal precision?     */
#define FABS(x)    fabsl((x))               /* use the long double    */
/*.IX{FABS}*/
#define SQRT(x)    sqrtl((x))               /* versions of the basic  */
/*.IX{SQRT}*/
#define POW(x, y)  powl((x), (y))           /* floating point         */
/*.IX{POW}*/
#define SIN(x)     sinl((x))                /* functions              */
/*.IX{SIN}*/
#define COS(x)     cosl((x))
/*.IX{COS}*/
#define EXP(x)     expl((x))
/*.IX{EXP}*/
#define LOG(x)     logl((x))
/*.IX{LOG}*/
#define ATAN(x)    atanl((x))
/*.IX{ATAN}*/
#define ACOS(x)    acosl((x))
/*.IX{ACOS}*/
#define COSH(x)    coshl((x))
/*.IX{COSH}*/

#else                                       /* less precision or not a*/
                                            /* BC3 and not a MC6 ?    */
#define FABS(x)    (REAL)fabs((double)(x))  /* declare names of basic */
#ifdef LONG_DOUBLE_USED                     /* floating point         */
#define SQRT(x)    sqrtlong((x))            /* functions that can be  */
/*.IX{SQRT}*/
#else                                       /* used in each of the    */
#define SQRT(x)    (REAL)sqrt((double)(x))  /* three precisions       */
#endif
#define POW(x, y)  (REAL)pow((double)(x), \
/*.IX{POW}*/                              \
                             (double)(y))
#define SIN(x)     (REAL)sin((double)(x))
/*.IX{SIN}*/
#define COS(x)     (REAL)cos((double)(x))
/*.IX{COS}*/
#define EXP(x)     (REAL)exp((double)(x))
/*.IX{EXP}*/
#define LOG(x)     (REAL)log((double)(x))
/*.IX{LOG}*/
#define ATAN(x)    (REAL)atan((double)(x))
/*.IX{ATAN}*/
#define ACOS(x)    (REAL)acos((double)(x))
/*.IX{ACOS}*/
#define COSH(x)    (REAL)cosh((double)(x))
/*.IX{COSH}*/
#endif

#undef sign
#undef min
#undef max
#define sign(x, y) (((y) < ZERO) ? -FABS(x) :     /* |x| times     */  \
/*.IX{sign}*/                                                          \
                                    FABS(x))      /* sign of y     */
#define min(a, b)        (((a) < (b)) ? (a) : (b))
/*.IX{min}*/
#define max(a, b)        (((a) > (b)) ? (a) : (b))
/*.IX{max}*/
#define SWAP(typ, a, b)                    /* swap two objects of   */ \
/*.IX{SWAP}*/                                                          \
  { typ temp; temp = a; a = b; b = temp; } /* arbitrary type        */

/* ------------------ Macros by Albert Becker ----------------------- */
#define ABS(X) (((X) >= ZERO) ? (X) : -(X))    /* Absolute value of X */
/*.IX{ABS}*/
#define SIGN(X,Y) \
/*.IX{SIGN}*/     \
             (((Y) < ZERO) ? -ABS(X) : ABS(X))    /* sign of Y times  */
                                                  /* ABS(X)           */
#define SQR(X) ((X) * (X))                     /* square of X         */
/*.IX{SQR}*/

#define FORMAT_IN      "%lg"               /* Input format for  REAL  */
/*.IX{FORMAT\unt IN}*/
#define FORMAT_LF      "% " LZP"f "         /* Format l for  REAL      */
/*.IX{FORMAT\unt LF}*/
#define FORMAT_126LF   "% 12.6" LZP"f "     /* Format 12.6f for  REAL  */
/*.IX{FORMAT\unt 126LF}*/
#define FORMAT_2010LF  "% 20.10" LZP"f "    /* Format 20.10f for  REAL */
/*.IX{FORMAT\unt 2010LF}*/
#define FORMAT_2016LF  "% 20.16" LZP"f "    /* Format 20.16f for REAL  */
/*.IX{FORMAT\unt 2016LF}*/
#define FORMAT_LE      "% " LZP"e "         /* Format e for REAL       */
/*.IX{FORMAT\unt LE}*/
#define FORMAT_2016LE  "% 20.16" LZP"e "    /* Format 20.16e for REAL  */
/*.IX{FORMAT\unt 2016LE}*/



/***********************************************************************
* declare all external functions defined in basis.c                    *
***********************************************************************/

int basis(void);             /* find basis for number representation  */

REAL mach_eps(void);         /* find machine constant                 */

REAL epsroot(void);          /* find square root of machine constant  */

REAL epsquad(void);          /* find square of machine constant       */

REAL maxroot(void);   /* Root of the largest representable number ....*/

REAL posmin(void);           /* find smallst positive floating point  */
                             /* number                                */

REAL pi(void);               /* find  pi                              */

REAL exp_1(void);            /* find e                                */

REAL sqr(REAL x);            /* square a floating point number        */

void fehler_melden   /* Write error messages to stdout and stderr ....*/
                  (
                   char text[],          /* error description ........*/
                   int  fehlernummer,    /* Number of error ..........*/
                   char dateiname[],     /* file with error  .........*/
                   int  zeilennummer     /* file name, row number ....*/
                  );

int umleiten            /* Perhaps redirect stdin or stdout to a file */
            (
             int argc,       /* number of arguments in command line ..*/
             char *argv[]    /* Vector of arguments ..................*/
            );               /* error code ...........................*/

void readln(void);             /* Skip the remainder of line in stdin */

void getline          /* Read one line from stdin ....................*/
            (
             char kette[],    /* Vector with the read text ...........*/
             int limit        /* maximal length of kette .............*/
            );

int intervall    /* Find the number for a value inside a partition ...*/
             (
              int n,         /* lenght of partition ..................*/
              REAL xwert,    /* number whose interval index is wanted */
              REAL x[]       /* partition ............................*/
             );              /* Index for xwert ......................*/

REAL horner        /* Horner scheme for polynomial evaluations .......*/
           (
            int n,                         /* Polynomial degree ......*/
            REAL a[],                      /* Polynomial coefficients */
            REAL x                         /* place of evaluation ....*/
           );                              /* Polynomial value at x ..*/

REAL norm_max      /* Find the maximum norm of a REAL vector .........*/
             (
              REAL vektor[],               /* vector .................*/
              int  n                       /* length of vector .......*/
             );                            /* Maximum norm ...........*/

REAL skalprod           /* standard scalar product of two REAL vectors*/
        (
         REAL v[],                 /* 1st vector .....................*/
         REAL w[],                 /* 2nd vector .....................*/
         int  n                    /* vector length...................*/
        );                         /* scalar product .................*/

void copy_vector        /* copy a REAL vector ........................*/
                (
                 REAL ziel[],            /* copied vector ............*/
                 REAL quelle[],          /* original vector ..........*/
                 int  n                  /* length of vector .........*/
                );


/*--------------------------------------------------------------------*
 * Basic functions chapter 1 (by Albert Becker) ......................*
 *--------------------------------------------------------------------*/

long double sqrtlong  (long double x);

int comdiv              /* Complex division ..........................*/
           (
            REAL   ar,            /* Real part of numerator ..........*/
            REAL   ai,            /* Imaginary part of numerator .....*/
            REAL   br,            /* Real part of denominator ........*/
            REAL   bi,            /* Imaginary part of denominator ...*/
            REAL * cr,            /* Real part of quotient ...........*/
            REAL * ci             /* Imaginary part of quotient ......*/
           );

REAL comabs             /* Complex absolute value ....................*/
              (
               REAL  ar,          /* Real part .......................*/
               REAL  ai           /* Imaginary part ..................*/
              );

void quadsolv           /* Complex quadratic equation ................*/
             (
               REAL    ar,        /* second degree coefficient .......*/
               REAL    ai,
               REAL    br,        /* linear coefficient ..............*/
               REAL    bi,
               REAL    cr,        /* polynomial constant .............*/
               REAL    ci,
               REAL *  tr,        /* solution ........................*/
               REAL *  ti
             );

void SetVec             /* initialize vector .........................*/
           (int n, REAL x[], REAL val);

void CopyVec            /* copy vector ...............................*/
            (int n, REAL source[], REAL dest[]);

int ReadVec             /* read vector from stdin ....................*/
           (int n, REAL x[]);

int WriteVec            /* write vector to stdout ....................*/
            (int n, REAL x[]);

void SetMat             /* initialize matrix .........................*/
           (int m, int n, REAL * a[], REAL val);

void CopyMat            /* copy matrix ...............................*/
            (int m, int n, REAL * source[], REAL * dest[]);

int ReadMat             /* read matrix from stdin ....................*/
           (int m, int n, REAL * a[]);

int WriteMat            /* write matrix to stdout ....................*/
            (int m, int n, REAL * mat[]);

int WriteHead  (char *s);         /* write header to stdout ..........*/

int WriteEnd  (void);             /* write separator to stdout .......*/

void LogError           /* write error message to stdout .............*/
             (char *s, int rc, char *file, int line);



#endif

/* -------------------------- END basis.h --------------------------- */

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe �  
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
