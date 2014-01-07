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
/* --------------------- DECLARATIONS u_proto.h --------------------- */

/*--------------------------------------------------------------------*
 * Include file with declarations for all Library functions.          *
 *                                                                    *
 *--------------------------------------------------------------------*/

#ifndef U_PROTO_H_INCLUDED

/* Secure against multiple inclusion */

#define U_PROTO_H_INCLUDED


/*--------------------------------------------------------------------*
 * Declare all external library functions                             *
 *--------------------------------------------------------------------*/


/*--------------------------------------------------------------------*
 * P 2  Nonlinear equations in one variable ..........................*
 *--------------------------------------------------------------------*/

int newton              /* Newton method in one dimension    .........*/
           (
            REALFCT   fct,        /* Function ........................*/
            REALFCT   fderv,      /* 1st derivative ..................*/
            REAL *    x,          /* Starting value / solution .......*/
            REAL *    fval,       /* Functional value at the root ... */
            int  *    iter        /* Number of iterations ............*/
           );

int newpoly              /* Newton method for polynomials  ...........*/
            (
             int       n,         /* degree of polynomial ............*/
             REAL      coeff[],   /* vector of coefficients ..........*/
             REAL *    x,         /* Starting value / solution .......*/
             REAL *    fval,      /* Functional value at x ...........*/
             int  *    iter       /* Number of iterations ............*/
            );

int polval                /* Newton method for polynomials ...........*/
           (
            int     n,            /* degree of the polynomial ........*/
            REAL    coeff[],      /* Vector of coefficients ..........*/
            REAL    x,            /* Value for evaluation ............*/
            REAL *  val,          /* value of the polynomial at x ....*/
            REAL *  dval          /* 1st derivative at x .............*/
           );

int newmod              /* Modified Newton Method ....................*/
           (
            REALFCT   fct,        /* Function ........................*/
            REALFCT   fderv1,     /* 1st derivative ..................*/
            REALFCT   fderv2,     /* 2nd derivative ..................*/
            REAL *    x,          /* Starting value / solution .......*/
            REAL *    fval,       /* Functional value at x ...........*/
            int *     iter,       /* Number of iterations ............*/
            int *     mul         /* Multiplicity of the root ........*/
           );

int pegasus             /* Pegasus Method    .........................*/
            (
             REALFCT   fct,       /* Function ........................*/
             REAL *    x1,        /* Starting value 1 ................*/
             REAL *    x2,        /* Starting value 2 / solution .....*/
             REAL *    f2,        /* Function value at x2 ............*/
             int *     iter       /* Number of iterations ............*/
            );

int roots               /* Pegasus, Anderson-Bjoerck-King method    ..*/
          (
           int       method,      /* method    .......................*/
           REALFCT   fct,         /* function ........................*/
           int       quadex,      /* quadratic extrapolation   .......*/
           REAL *    x1,          /* Starting value 1 ................*/
           REAL *    x2,          /* Starting value 2 / solution .....*/
           REAL *    fx2,         /* Function value at x2 ............*/
           int *     iter         /* Iteration number.................*/
          );


/*--------------------------------------------------------------------*
 * P 3  Roots of polynomials .........................................*
 *--------------------------------------------------------------------*/

int mueller             /* Mueller's method for real polynomials .....*/
            (
             int   n,             /* degree of the polynomial ........*/
             REAL  a[],           /* vector of coefficients ..........*/
             int   scaleit,       /* Scaling control .................*/
             REAL  zreal[],       /* Real part of the solution .......*/
             REAL  zimag[]        /* Imaginary part of the solution ..*/
            );

void fmval              /* (Complex) polynomial value ................*/
           (
            int     n,            /* Maximal degree present ..........*/
            int     iu,           /* Lowest degree present ...........*/
            REAL    zre[],        /* Koefficients ....................*/
            REAL    zren,         /* Leading coefficient .............*/
            REAL    xre,          /* Real part of x ..................*/
            REAL    xim,          /* Imaginary part of x .............*/
            REAL *  fre,          /* Real part of the function value .*/
            REAL *  fim           /* Imaginary part of function value */
           );

int bauhub              /* Bauhuber's method for complex polynomials .*/
           (
            int    real,          /* Are the coefficients real ? .....*/
            int    scale,         /* Scaling ? .......................*/
            int     n,            /* degree of polynomail ............*/
            REAL  ar[],           /* Real parts of coefficients ......*/
            REAL  ai[],           /* Imaginary parts, coefficients ...*/
            REAL  rootr[],        /* Real parts of roots .............*/
            REAL  rooti[],        /* Imaginary parts of roots ........*/
            REAL  absf[]          /* Absolute value of function values*/
           );


/*--------------------------------------------------------------------*
 * P 4  Direct methods for solving systems of linear equations .......*
 *--------------------------------------------------------------------*/

int gauss            /* Gauss algorithm for solving linear equations .*/
          (
           int     mod,           /* Modus: 0, 1, 2, 3 ...............*/
           int     n,             /* Dimension of matrix .............*/
           REAL *  mat[],         /* Input matrix ....................*/
           REAL *  lumat[],       /* LU decomposition ................*/
           int     perm[],        /* row remutation vector ...........*/
           REAL    b[],           /* right hand side .................*/
           REAL    x[],           /* solution of the system ..........*/
           int *   signd          /* sign of the permutation .........*/
          );

int gaudec              /* Gauss decomposition .......................*/
           (
            int     n,            /* size of matrix ..................*/
            REAL *  mat[],        /* Input matrix ....................*/
            REAL *  lumat[],      /* matrix decomposition ............*/
            int     perm[],       /* row interchanges ................*/
            int *   signd         /* sign of perm ....................*/
           );

int gausol              /* Gauss solution ............................*/
           (
            int     n,            /* size of matrix ..................*/
            REAL *  lumat[],      /* decomposed matrix (LU) ..........*/
            int     perm[],       /* row permutation vector ..........*/
            REAL    b[],          /* Right hand side .................*/
            REAL    x[]           /* solution ........................*/
           );

int gausoli              /* Gauss with iterative refinement ..........*/
            (
             int     n,            /* Dimension of matrix ............*/
             REAL *  mat[],        /* original matrix ................*/
             REAL *  lumat[],      /* LU decomposition ...............*/
             int     perm[],       /* row interchange vector .........*/
             REAL    b[],          /* Right hand side ................*/
             REAL    x[]           /* solution .......................*/
            );

int mgauss                     /* Gauss for multiple right hand sides */
           (
            int     n,            /* Dimension of system .............*/
            int     k,            /* number of right hand sides ......*/
            REAL *  mat[],        /* original matrix .................*/
            REAL *  rmat[]        /* Right hand sides/solutions ......*/
           );

REAL det                /* Determinant  ..............................*/
           (
            int     n,            /* Dimension of the matrix .........*/
            REAL *  mat[]         /* matrix ..........................*/
           );

int choly               /* Cholesky Method ...........................*/
          (
           int     mod,           /* Modus: 0, 1, 2 ..................*/
           int     n,             /* Dimension of matrix .............*/
           REAL *  mat[],         /* matrix ..........................*/
           REAL    b[],           /* Right hand side of system .......*/
           REAL    x[]            /* solution vector .................*/
          );

int chodec              /* Cholesky decomposition ....................*/
           (
            int     n,            /* size of matrix ..................*/
            REAL *  mat[]         /* input matrix/Cholesky factor ....*/
           );

int chosol              /* Cholesky solver ...........................*/
           (
            int     n,            /* Dimension of matrix .............*/
            REAL *  lmat[],       /* Cholesky matrix .................*/
            REAL    b[],          /* Right hand side of system .......*/
            REAL    x[]           /* solution vector .................*/
           );

int pivot               /* Find the matrix inverse (Exchange steps) ..*/
          (
           int     n,             /* size of matrix ..................*/
           REAL *  mat[],         /* input matrix ....................*/
           REAL *  inv[],         /* its inverse .....................*/
           REAL *  s,             /* Check sum .......................*/
           REAL *  cond           /* condition number ................*/
          );

int trdiag              /* Tridiagonal linear systems ................*/
           (
            int     n,            /* size of system matrix ...........*/
            REAL    lower[],      /* lower co-diagonal ...............*/
            REAL    diag[],       /* Diagonal ........................*/
            REAL    upper[],      /* upper co-diagonal ...............*/
            REAL    b[],          /* Right hand side / solution ......*/
            int     rep           /* rep = 0, 1 ......................*/
           );

int tzdiag              /* cyclic tridiagonal linear systems .........*/
           (
            int   n,              /* size of matrix ..................*/
            REAL  lower[],        /* Sub-diagonal ....................*/
            REAL  diag[],         /* Diagonal ........................*/
            REAL  upper[],        /* Super-diagonal ..................*/
            REAL  lowrow[],       /* row below .......................*/
            REAL  ricol[],        /* column to the right .............*/
            REAL  b[],            /* right hand side, or solution ....*/
            int   rep             /* rep = 0, 1 ......................*/
           );

int diag5               /* 5 diagonal linear systems .................*/
          (
           int   mod,             /* Modus: 0, 1, 2 ..................*/
           int   n,               /* size of matrix ..................*/
           REAL  ld2[],           /* 2. lower co-diagonal ............*/
           REAL  ld1[],           /* 1. lower co-diagonal ............*/
           REAL  d[],             /* main diagonal ...................*/
           REAL  ud1[],           /* 1. upper co-diagonal ............*/
           REAL  ud2[],           /* 2. upper co-diagonal ............*/
           REAL  b[]              /* right hand side/solution ........*/
          );

int diag5dec            /* LU factorization of a 5 diagonal matrix ...*/
             (
              int   n,            /* size of matrix  .................*/
              REAL  ld2[],        /* 2. lower co-diagonal ............*/
              REAL  ld1[],        /* 1. lower co-diagonal ............*/
              REAL  d[],          /* main diagonal ...................*/
              REAL  ud1[],        /* 1. upper co-diagonal ............*/
              REAL  ud2[]         /* 2. upper co-diagonal ............*/
             );

int diag5sol            /* solve a five diagonal linear system .......*/
             (
              int   n,            /* size of matrix ..................*/
              REAL  ld2[],        /* 2. lower co-diagonal ............*/
              REAL  ld1[],        /* 1. lower co-diagonal ............*/
              REAL  d[],          /* main diagonal ...................*/
              REAL  ud1[],        /* 1. upper co-diagonal ............*/
              REAL  ud2[],        /* 2. upper co-diagonal ............*/
              REAL  b[]           /* right hand side/solution ........*/
             );

int diag5pd             /* 5 diagonal symmetric strongly nonsingular .*/
            (
             int   mod,           /* Modus: 0, 1, 2 ..................*/
             int   n,             /* # matrix rows ...................*/
             REAL  d[],           /* main diagonal ...................*/
             REAL  ud1[],         /* first co-diagonal ...............*/
             REAL  ud2[],         /* second co-diagonal ..............*/
             REAL  b[]            /* Right hand side .................*/
            );

int diag5pddec       /* Factor 5 diagonal strongly nonsingular matrix */
               (
                int   n,          /* # Matrix rows ...................*/
                REAL  d[],        /* main diagonal ...................*/
                REAL  ud1[],      /* 1. co-diagonal ..................*/
                REAL  ud2[]       /* 2. co-diagonal ..................*/
               );

int diag5pdsol          /* Solve systems for 5 diagonal symmetric m. .*/
               (
                int   n,          /* size of matrix ..................*/
                REAL  d[],        /* main diagonal ...................*/
                REAL  ud1[],      /* 1. co-diagonal ..................*/
                REAL  ud2[],      /* 2. co-diagonal ..................*/
                REAL  b[]         /* Right hand side .................*/
               );

int pack                /* condense a row ............................*/
         (
          int     n,              /* size of matrix ..................*/
          int     ld,             /* number of lower co-diagonals ....*/
          int     ud,             /* number of upper co-diagonals ....*/
          int     no,             /* row index .......................*/
          REAL    row[],          /* original row ....................*/
          REAL    prow[]          /* condensed row ...................*/
         );

int unpack              /* uncondense row ............................*/
         (
          int     n,              /* size of matrix ..................*/
          int     ld,             /* number of lower co-diagonals ....*/
          int     ud,             /* number of upper co-diagonals ....*/
          int     no,             /* row index .......................*/
          REAL    row[],          /* uncondensed row .................*/
          REAL    prow[]          /* condensed row ...................*/
         );

int band                /* Linear systems with banded matrices .......*/
         (
          int    mod,             /* Modus: 0, 1, 2 ..................*/
          int    n,               /* size of system ..................*/
          int    ld,              /* # of lower co-diagonals .........*/
          int    ud,              /* # of upper co-diagonals .........*/
          REAL * pmat[],          /* condensed input matrix ..........*/
          REAL   b[],             /* right hand side .................*/
          int    perm[],          /* row permutation vector ..........*/
          int *  signd            /* sign of perm ....................*/
         );

int banddec             /* Factor a banded matrix ....................*/
            (
             int    n,            /* size of system ..................*/
             int    ld,           /* # of lower co-diagonals .........*/
             int    ud,           /* # of upper co-diagonals .........*/
             REAL * pmat[],       /* condensed input matrix ..........*/
             int    perm[],       /* row permutation vector ..........*/
             int *  signd         /* sign of perm ....................*/
            );

int bandsol             /* Solve a banded system .....................*/
            (
             int    n,            /* size of system ..................*/
             int    ld,           /* # of lower co-diagonals .........*/
             int    ud,           /* # of upper co-diagonals .........*/
             REAL * pmat[],       /* condensed input matrix ..........*/
             REAL   b[],          /* right hand side .................*/
             int    perm[]        /* row permutation vector ..........*/
            );

int bando               /* Linear banded system without using pivots .*/
          (
          int    mod,             /* Modus: 0, 1, 2 ..................*/
          int    n,               /* size of system ..................*/
          int    ld,              /* # of lower co-diagonals .........*/
          int    ud,              /* # of upper co-diagonals .........*/
          REAL * pmat[],          /* condensed input matrix ..........*/
          REAL   b[]              /* right hand side .................*/
         );

int banodec             /* Decompose a banded matrix .................*/
            (
             int    n,            /* size of system ..................*/
             int    ld,           /* # of lower co-diagonals .........*/
             int    ud,           /* # of upper co-diagonals .........*/
             REAL * pmat[]        /* condensed input matrix ..........*/
            );

int banosol             /* Solve a banded system .....................*/
            (
             int    n,            /* size of system ..................*/
             int    ld,           /* # of lower co-diagonals .........*/
             int    ud,           /* # of upper co-diagonals .........*/
             REAL * pmat[],       /* condensed input matrix ..........*/
             REAL   b[]           /* right hand side .................*/
            );

int house               /* Householder Method ........................*/
          (
           int     m,             /* # of rows .......................*/
           int     n,             /* # of columns ....................*/
           REAL *  mat[],         /* Input matrix ....................*/
           REAL    b[]            /* righ thand side/solution ........*/
          );

int mhouse              /* Householder method for m right hand sides .*/
           (
            int     m,            /* # of rows .......................*/
            int     n,            /* # of columns ....................*/
            int     k,            /* # right hand sides ..............*/
            REAL *  mat[],        /* matrix ..........................*/
            REAL *  xmat[]        /* Right hand sides/solutions ......*/
           );

REAL hcond              /* Hadamard condition number .................*/
             (
              int     n,          /* size of matrix ..................*/
              REAL *  mat[]       /* matrix ..........................*/
             );

REAL ccond              /* Conditions estimate according to Cline ....*/
             (
              int     n,          /* Dimension of matrix .............*/
              REAL *  mat[]       /* matrix ..........................*/
             );

REAL fcond              /* Condition estimate of Forsythe/Moler ......*/
             (
              int     n,          /* Dimension of matrix .............*/
              REAL *  mat[]       /* matrix ..........................*/
             );


/*--------------------------------------------------------------------*
 * P 5  Iterative methods for linear equations .......................*
 *--------------------------------------------------------------------*/

int seidel              /* Gauss Seidel Method with relaxation .......*/
           (
            int     crit,         /* crit = 0, 1, 2, 3 ...............*/
            int     n,            /* size of matrix ..................*/
            REAL *  mat[],        /* matrix ..........................*/
            REAL    b[],          /* Right hand side .................*/
            REAL    omega,        /* Relaxaktion coefficient .........*/
            REAL    x[],          /* solution ........................*/
            REAL    residu[],     /* Residuum vector .................*/
            int *   iter          /* # of iterations .................*/
           );


/*--------------------------------------------------------------------*
 * P 6  Systems of nonlinear equations ...............................*
 *--------------------------------------------------------------------*/

int newt                 /* Multidimensional Newton method ...........*/
         (
          int       n,            /* size of system ..................*/
          REAL      x[],          /* Starting/solution vector ........*/
          FNFCT     fct,          /* Function ........................*/
          JACOFCT   jaco,         /* Function for Jacobi matrix ......*/
          int       kmax,         /* Maximal number of damped steps ..*/
          int       prim,         /* Maximal number of basic steps ...*/
          char *    pfile,        /* Name of the protocol file .......*/
          REAL      fvalue[],     /* Function value at solution ......*/
          int *     iter,         /* number of iteration steps .......*/
          REAL      eps           /* error bound .....................*/
         );


/*--------------------------------------------------------------------*
 * P 7  Eigenvalues and eigenvectors of matrices .....................*
 *--------------------------------------------------------------------*/

int mises               /* Vector iteration for max modulus eigenvalue*/
          (
           int     n,             /* Dimension of matrix .............*/
           REAL *  mat[],         /* matrix ..........................*/
           REAL    x[],           /* Eigenvector .....................*/
           REAL *  ew             /* maximum modulus eigenvalue ......*/
          );

int eigen               /* Compute all evalues/evectors of a matrix ..*/
          (
           int     vec,           /* switch for computing evectors ...*/
           int     ortho,         /* orthogonal Hessenberg reduction? */
           int     ev_norm,       /* normalize Eigenvectors? .........*/
           int     n,             /* size of matrix ..................*/
           REAL *  mat[],         /* input matrix ....................*/
           REAL *  eivec[],       /* Eigenvectors ....................*/
           REAL    valre[],       /* real parts of eigenvalues .......*/
           REAL    valim[],       /* imaginary parts of eigenvalues ..*/
           int     cnt[]          /* Iteration counter ...............*/
          );

#endif

/* ------------------------- END u_proto.h -------------------------- */

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
