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
/* ------------------------ MODULE vmblock.c ------------------------ */

/***********************************************************************
*                                                                      *
* Management of a set of dynamically allocated vectors and matrices    *
* -----------------------------------------------------------------    *
*                                                                      *
* Idea:   In many subroutines of this Numerical Library dynamically    *
*         allocated vectors and matrices are constantly being used.    *
*         This leads to the recurring problem that there is not enough *
*         memory for all of the needed vectors and matrices so that    *
*         the memory already allocated must be freed and the lack of   *
*         memory must be dealt with. This is a very laborious task     *
*         and can lead to recurring errors.                            *
*         This module is designed to simplify the storage management   *
*         significantly:                                               *
*         It can manage all contingent memory allocation for vectors   *
*         and matrices in one linear list. To handle such a list, the  *
*         user is provided the following four functions:               *
*                                                                      *
*         - vminit()     which creates an empty vector/matrix list and *
*                        returns an untyped list pointer to be used by *
*                        the following three functions,                *
*                                                                      *
*         - vmalloc()    which allocates memory for a new vector or a  *
*                        new matrix, inserts its address into the list *
*                        and returns this address,                     *
*                                                                      *
*         - vmcomplete() to verify that all prior memory allocations   *
*                        in the list have been successful, and         *
*                                                                      *
*         - vmfree()     which frees all of the memory taken by the    *
*                        vector/matrix list.                           *
*                                                                      *
*         Moreover the seven macros                                    *
*                                                                      *
*         - VEKTOR  (for REAL vectors),                                *
*         - VVEKTOR (for arbitrary vectors),                           *
*         - MATRIX  (for REAL matrices),                               *
*         - IMATRIX (for int matrices),                                *
*         - MMATRIX (for matrices of 4x4 matrices),                    *
*         - UMATRIX (for lower triangular matrices of type REAL) and   *
*         - PMATRIX (for matrices of points in R3)                     *
*                                                                      *
*         are exported which allow the user to select the kind of data *
*         structure when calling vmalloc().                            *
*                                                                      *
*         Attention: 1. The memory taken by a vector/matrix list must  *
*                       only be freed using vmfree()!                  *
*                    2. vmfree() only frees the complete memory        *
*                       belonging to one list and therefore cannot be  *
*                       applied to just one vector or matrix of the    *
*                       list!                                          *
*                                                                      *
* Usage:  The user specifies a void pointer which is initialized       *
*         by calling vminit() and which gives the only valid entry to  *
*         the vector/matrix list.                                      *
*         Using this pointer vectors and matrices can now be allocated *
*         dynamically via vmalloc().                                   *
*         Once all storage needs have been satisfied, one should use   *
*         vmcomplete() to verify that they all were successful and to  *
*         react on a possible lack of memory.                          *
*         If the contents of a list is no longer needed, we recommend  *
*         to free its space by calling vmfree().                       *
*         Example:                                                     *
*             ...                                                      *
*             void *vmblock;    /+ start of the vector/matrix list +/  *
*             REAL *vektor1;    /+ REAL vector with n elements     +/  *
*             int  *vektor2;    /+ int vector with n elements      +/  *
*             REAL **matrix1;   /+ Matrix with m rows, n columns   +/  *
*             int  **matrix2;   /+ Matrix with m rows, n columns   +/  *
*             mat4x4 **mmat;    /+ matrix with m*n elements of     +/  *
*                               /+ type `mat4x4' (16 REAL values)  +/  *
*             REAL **umatrix;   /+ lower triangular (n,n) matrix   +/  *
*             REAL ***pmatrix;  /+ matrix with m*n points in R3    +/  *
*             ...                                                      *
*             vmblock = vminit();                                      *
*             vektor1 = (REAL *)vmalloc(vmblock, VEKTOR,  n, 0);       *
*             vektor2 = (int *) vmalloc(vmblock, VVEKTOR, n,           *
*                                       sizeof(int));                  *
*             ...                                                      *
*             matrix1 = (REAL **)  vmalloc(vmblock, MATRIX,  m, n);    *
*             matrix2 = (int  **)  vmalloc(vmblock, IMATRIX, m, n);    *
*             mmat    = (mat4x4 **)vmalloc(vmblock, MMATRIX, m, n);    *
*             umatrix = (REAL ***) vmalloc(vmblock, UMATRIX, m, 0);    *
*             pmatrix = (REAL ***) vmalloc(vmblock, PMATRIX, m, n);    *
*             ...                                                      *
*             if (! vmcomplete(vmblock))  /+ in parts unsuccessful? +/ *
*             {                                                        *
*               vmfree(vmblock);          /+ free memory in list    +/ *
*               return 99;                /+ report error           +/ *
*             }                                                        *
*             ...                                                      *
*             vmfree(vmblock);                                         *
*             ...                                                      *
*                                                                      *
* Programming language: ANSI C                                         *
* Compiler:             Borland C++ 2.0                                *
* Computer:             IBM PS/2 70 with 80387                         *
* Author:               Juergen Dietel, Computer Center, RWTH Aachen   *
* Date:                 9.10.1992                                      *
*                                                                      *
***********************************************************************/


#include "basis.h"     /*  for  size_t, NULL, malloc, free, calloc,   */
                       /*       boolean, FALSE, TRUE, REAL, mat4x4    */
#include "vmblock.h"   /*  for  vmalloc, vmcomplete, vmfree, vminit,  */
                       /*       VEKTOR, VVEKTOR, MATRIX, IMATRIX,     */
                       /*       MMATRIX, UMATRIX, PMATRIX             */



/*--------------------------------------------------------------------*/

typedef struct VML          /* Element of a vector/matrix list        */
{
  void       *vmzeiger;     /* pointer to the vector or matrix        */
  int        typ;           /* kind of pointer: vector or matrix      */
                            /* (possible values: VEKTOR, VVEKTOR,     */
                            /*                   MATRIX, IMATRIX,     */
                            /*                   MMATRIX, UMATRIX,    */
                            /*                   PMATRIX)             */
  size_t     groesse;       /* in the anchor element: the flag that   */
                            /* indicates failed memory allocations;   */
                            /* otherwise not used except for matrices */
                            /* where `groesse' is "abused" to save    */
                            /* the number of rows                     */
  size_t     spalten;       /* number of columns of matrices of       */
                            /* points in R3                           */
  struct VML *naechst;      /* pointer to next element in the list    */
} vmltyp;
/*.IX{vmltyp}*/

#define VMALLOC  (vmltyp *)malloc(sizeof(vmltyp)) /* allocate memory  */
/*.IX{VMALLOC}*/
                                                  /* for a new        */
                                                  /* element of the   */
                                                  /* list             */

#define LISTE    ((vmltyp *)vmblock)              /* for abbreviation */
/*.IX{LISTE}*/
#define MAGIC    410                              /* used to mark a   */
/*.IX{MAGIC}*/
                                                  /* valid anchor     */
                                                  /* element          */



/*--------------------------------------------------------------------*/

void *vminit         /* create an empty vector/matrix list ...........*/
/*.IX{vminit}*/
        (
         void
        )                       /* address of list ...................*/

/***********************************************************************
* Generate an empty vector/matrix list. Such a list consists of a      *
* anchor element, which is being used only to hold the `out of memory' *
* flag and a magic number that is used for plausibility checks.        *
* The return value here is the address of the anchor element or - in   *
* case of error - the value NULL.                                      *
* For subsequent calls of vmalloc(), vmcomplete() and vmfree() we use  *
* the component `typ' of the anchor element for the magic value which  *
* designates a proper anchor element in order to be able to check      *
* whether the supplied untyped pointer in fact points to a             *
* vector/matrix list.                                                  *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* vmltyp, VMALLOC, MAGIC, NULL, malloc                                 *
***********************************************************************/

{
  vmltyp *liste;             /* pointer to anchor element of list     */


  if ((liste = VMALLOC) == NULL)  /* allocate memory for the anchor   */
    return NULL;                  /* unsuccessful? => report error    */
  liste->vmzeiger = NULL;         /* to make vmfree() error free      */
  liste->typ      = MAGIC;        /* mark a valid anchor element      */
  liste->groesse  = 0;            /* no lack of memory as yet         */
  liste->naechst  = NULL;         /* no next element                  */


  return (void *)liste;
}



/*--------------------------------------------------------------------*/

static void matfree  /* free memory of a dynamic matrix ..............*/
/*.IX{matfree}*/
        (
         void   **matrix,       /* [0..m-1,0..] matrix ...............*/
         size_t m               /* number of rows of matrix ..........*/
        )

/***********************************************************************
* free the memory of a matrix with m rows as allocated in matmalloc()  *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* size_t, NULL, free                                                   *
***********************************************************************/

{
#ifdef FAST_ALLOC
  void *tmp;                    /* smallest row address               */

#endif
  if (matrix != NULL)           /* matrix exists?                     */
  {
#ifndef FAST_ALLOC              /* safe, but expensive allocation?    */
    while (m != 0)              /* free memory of matrix elements     */
      free(matrix[--m]);        /* row by row                         */
#else                           /* more economical allocation method? */
                                /* (assumes linear address space)     */
    for (tmp = matrix[0]; m != 0; )  /* find pointer with smallest    */
      if (matrix[--m] < tmp)         /* address (necessary because of */
        tmp = matrix[m];             /* possible permutation!)        */
    free(tmp);                  /* free memory of all matrix elements */
                                /* at once                            */
#endif
    free(matrix);               /* free all row pointers              */
  }
}



/*--------------------------------------------------------------------*/

/***********************************************************************
* Allocate memory for a rectangular [0..m-1,0..n-1] matrix with        *
* elements of type `typ' and store the starting address of the matrix  *
* in `mat', if successful; store NULL else. We form a new pointer to   *
* the start of each row of the matrix, which contains n elements.      *
* Lack of memory causes the part of the matrix already allocated to be *
* freed.                                                               *
* If before compilation of this file the macro FAST_ALLOC was defined, *
* there are still m row pointers used, but (following an idea of       *
* Albert Becker) the memory of the m*n matrix elements is allocated in *
* one piece into which the row pointers are directed.                  *
* According to this, matfree() contains a FAST_ALLOC part as well,     *
* where one has to pay attention to the fact that the row pointers     *
* could have been permuted since the allocation of the matrix.         *
* If a lower triangular matrix is needed (umat != 0), the value n is   *
* ignored (because the matrix is quadratic) and memory for only        *
* m*(m+1)/2 REAL values is allocated (apart from the row pointers).    *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* size_t, NULL, calloc, matfree                                        *
***********************************************************************/

#ifndef FAST_ALLOC              /* safe, but expensive allocation?    */
#define matmalloc(mat, m, n, typ, umat)                                \
/*.IX{matmalloc}*/                                                     \
                                                                       \
{                                                                      \
  size_t j,                               /* current row index     */  \
         k;                               /* elements in row j     */  \
                                                                       \
  if ((mat = (typ **)calloc((m), sizeof(typ *))) != NULL)              \
    for (j = 0; j < (m); j++)                                          \
    {                                                                  \
      k = (umat) ? (j + 1) : (n);                                      \
      if ((((typ **)mat)[j] = (typ *)calloc(k, sizeof(typ))) == NULL)  \
      {                                                                \
        matfree((void **)(mat), j);                                    \
        mat = NULL;                                                    \
        break;                                                         \
      }                                                                \
    }                                                                  \
}
#else                           /* more economical allocation method? */
                                /* (assumes linear address space)     */
#define matmalloc(mat, m, n, typ, umat)                                \
/*.IX{matmalloc}*/                                                     \
                                                                       \
{                                                                      \
  typ    *tmp;  /* address of the contingent area of memory where   */ \
                /* all memory elements reside                       */ \
  size_t j,     /* current row index                                */ \
         k,     /* index for `tmp' to the j. row (value: j*n)       */ \
         l;     /* size of memory space: full (m*n elements) or     */ \
                /* lower triangular (m*(m+1)/2 elements) matrix     */ \
                                                                       \
  if ((mat = (typ **)calloc((m), sizeof(typ *))) != NULL)              \
  {                                                                    \
    l = (umat) ? (((m) * ((m) + 1)) / 2) : ((m) * (n));                \
    if ((tmp = (typ *)calloc(l, sizeof(typ))) != NULL)                 \
      for (j = k = 0; j < (m); j++)                                    \
        ((typ **)mat)[j]  = tmp + k,                                   \
        k                += (umat) ? (j + 1) : (n);                    \
    else                                                               \
    {                                                                  \
      free(mat);                                                       \
      mat = NULL;                                                      \
    }                                                                  \
  }                                                                    \
}
#endif



/*--------------------------------------------------------------------*/

static void pmatfree  /* free memory of a matrix of R3 points ........*/
/*.IX{pmatfree}*/
        (
         void   ***matrix,      /* [0..m-1,0..n-1] matrix of points ..*/
         size_t m,              /* number of rows of matrix ..........*/
         size_t n               /* number of columns of matrix .......*/
        )

/***********************************************************************
* free a matrix with m rows and n columns as allocated in pmatmalloc() *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* size_t, NULL, free, matfree                                          *
***********************************************************************/

{
  if (matrix != NULL)              /* matrix exists?                  */
  {
    while (m != 0)                 /* free memory of matrix elements  */
      matfree(matrix[--m], n);     /* row by row                      */
    free(matrix);                  /* free row pointers               */
  }
}



/*--------------------------------------------------------------------*/

static REAL ***pmatmalloc   /* allocate memory for a matrix of points */
/*.IX{pmatmalloc}*/
        (
         size_t m,              /* number of rows of matrix ..........*/
         size_t n               /* number of columns of matrix .......*/
        )                       /* address of matrix .................*/

/***********************************************************************
* Allocate memory for a [0..m-1,0..n-1,0..2] matrix with REAL elements *
* and return its starting address, if successful; return NULL else. We *
* form a new pointer to the start of each row of the matrix.           *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* size_t, REAL, NULL, calloc, pmatfree, matmalloc                      *
***********************************************************************/

{
  REAL   ***matrix;                      /* pointer to row vectors    */
  size_t i;                              /* current row index         */


  matrix = (REAL ***)                    /* one pointer for each      */
           calloc(m, sizeof(*matrix));   /* of the m rows             */

  if (matrix == NULL)                    /* lack of memory?           */
    return NULL;                         /* report this lack          */

  for (i = 0; i < m; i++)                /* allocate one (n,3) matrix */
  {                                      /* for each row pointer      */
    matmalloc(matrix[i], n, 3, REAL, 0);
    if (matrix[i] == NULL)               /* lack of memory?           */
    {
      pmatfree((void ***)matrix, i, 3);  /* free (n,3) matrices       */
                                         /* already allocated         */
      return NULL;                       /* report lack of memory     */
    }
  }


  return matrix;
}



/*--------------------------------------------------------------------*/

void *vmalloc        /* create a dynamic vector or matrix ............*/
/*.IX{vmalloc}*/
        (
         void   *vmblock,       /* address of a vector/matrix list ...*/
         int    typ,            /* kind of vector or matrix ..........*/
         size_t zeilen,         /* length (vector) or number of rows .*/
         size_t spalten         /* number of columns or element size .*/
        )                       /* address of the created object .....*/

/***********************************************************************
* Create an element according to `typ' (vector or matrix), whose size  *
* is determined by the parameters `zeilen' and `spalten'. This object  *
* is inserted into the linear list starting at `vmblock'.              *
* The address of the new vector or matrix is returned.                 *
* For a REAL vector (kind VEKTOR) the parameter `zeilen' contains its  *
* length, `spalten' is not used. For arbitrary vectors (kind VVEKTOR)  *
* the parameter `spalten' must contain the size of one vector element. *
* For a full matrix (kind MATRIX, IMATRIX, MMATRIX or PMATRIX) the     *
* parameter `zeilen' contains the number of rows, while `spalten'      *
* contains the number of columns of the matrix. For a (quadratic)      *
* lower triangular matrix (kind UMATRIX) `zeilen' contains the number  *
* of rows resp. columns of the matrix.                                 *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* vmltyp, VMALLOC, LISTE, MAGIC, matmalloc, pmatmalloc, REAL, VEKTOR,  *
* VVEKTOR, MATRIX, IMATRIX, MMATRIX, UMATRIX, PMATRIX, NULL, size_t,   *
* malloc, calloc, mat4x4, matmalloc                                    *
***********************************************************************/

{
  vmltyp *element;                  /* pointer to new element in list */


  if (LISTE      == NULL   ||       /* invalid list?                  */
      LISTE->typ != MAGIC)          /* invalid starting element?      */
    return NULL;                    /* report error                   */


  if ((element = VMALLOC) == NULL)  /* ask for a new element          */
  {                                 /* no success? =>                 */
    LISTE->groesse = 1;             /* indicate lack of memory        */
    return NULL;                    /* report error                   */
  }

  switch (typ)         /* allocate memory for the desired data        */
  {                    /* structure (vector or matrix) and record its */
                       /* address in the new list element             */

    case VEKTOR:          /* ---------- REAL vector?       ---------- */
      element->vmzeiger = calloc(zeilen, sizeof(REAL));
      break;

    case VVEKTOR:         /* ---------- arbitrary vector?  ---------- */
      element->vmzeiger = calloc(zeilen, spalten);
      break;

    case MATRIX:          /* ---------- REAL matrix?       ---------- */
      matmalloc(element->vmzeiger, zeilen, spalten, REAL, 0);
      element->groesse  = zeilen;      /* put row number into         */
      break;                           /* `groesse' for vmfree()      */

    case IMATRIX:         /* ---------- int matrix?        ---------- */
      matmalloc(element->vmzeiger, zeilen, spalten, int, 0);
      element->groesse  = zeilen;      /* put row number into         */
      break;                           /* `groesse' for vmfree()      */

    case MMATRIX:         /* ---------- mat4x4 matrix?     ---------- */
      matmalloc(element->vmzeiger, zeilen, spalten, mat4x4, 0);
      element->groesse  = zeilen;      /* put row number into         */
      break;                           /* `groesse' for vmfree()      */

    case UMATRIX:         /* ---------- untere Dreiecksmatrix? ------ */
      matmalloc(element->vmzeiger, zeilen, 0, mat4x4, 1);
      element->groesse  = zeilen;      /* put row number into         */
      break;                           /* `groesse' for vmfree()      */

    case PMATRIX:         /* ---------- matrix with points in R3? --- */
      element->vmzeiger = (void *)pmatmalloc(zeilen, spalten);
      element->groesse  = zeilen;      /* put row number into         */
      element->spalten  = spalten;     /* `groesse' and column number */
      break;                           /* into `spalten' for vmfree() */

    default:              /* ---- invalid data type?   -------------  */
      element->vmzeiger = NULL;        /* record zero pointer         */
  }

  if (element->vmzeiger == NULL)       /* no memory for the object?   */
    LISTE->groesse = 1;                /* Let's note that down.       */

  element->typ = typ;                  /* note kind of data structure */
                                       /* in the list element         */
  element->naechst = LISTE->naechst;   /* insert new element before   */
                                       /* the first existing element, */

  LISTE->naechst = element;            /* but behind the anchor       */
                                       /* element                     */

  return element->vmzeiger;            /* return new vector/matrix    */
}                                      /* address                     */



/*--------------------------------------------------------------------*/

boolean vmcomplete   /* check vector/matrix list for lack of memory ..*/
/*.IX{vmcomplete}*/
        (
         void *vmblock          /* address of list ...................*/
        )                       /* no lack of memory? ................*/

/***********************************************************************
* Here just the negated value of the flag in the anchor element is     *
* returned which belongs to the vector/matrix list represented by      *
* `vmblock'. Thus this functions reports whether all memory            *
* allocations in the list have been successful (TRUE) or not (FALSE).  *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* LISTE                                                                *
***********************************************************************/

{
  return LISTE->groesse ? FALSE : TRUE;
}



/*--------------------------------------------------------------------*/

void vmfree          /* free the memory for a vektor/matrix list .....*/
/*.IX{vmfree}*/
        (
         void *vmblock          /* address of list ...................*/
        )

/***********************************************************************
* free all dynamic memory consumed by the list beginning at `vmblock'  *
*                                                                      *
* global names used:                                                   *
* ==================                                                   *
* vmltyp, LISTE, MAGIC, matfree, pmatfree, VEKTOR, VVEKTOR, MATRIX,    *
* IMATRIX, MMATRIX, UMATRIX, PMATRIX, NULL, free                       *
***********************************************************************/

{
  vmltyp *hilf;                  /* aux variable for value of pointer */


  if (LISTE == NULL)                     /* invalid list?             */
    return;                              /* do nothing                */

  if (LISTE->typ != MAGIC)               /* invalid anchor element?   */
    return;                              /* do nothing again          */


  for ( ; LISTE != NULL; vmblock = (void *)hilf)
  {

    switch (LISTE->typ)
    {
      case VEKTOR:
      case VVEKTOR: if (LISTE->vmzeiger != NULL)
                      free(LISTE->vmzeiger);
                    break;
      case MATRIX:
      case IMATRIX:
      case MMATRIX:
      case UMATRIX: matfree((void **)LISTE->vmzeiger,
                            LISTE->groesse);
                    break;
      case PMATRIX: pmatfree((void ***)LISTE->vmzeiger,
                             LISTE->groesse, LISTE->spalten);
    }

    hilf = LISTE->naechst;               /* save pointer to successor */
    free(LISTE);                         /* free list element         */
  }
}

/* ------------------------- END vmblock.c -------------------------- */

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
