/* ========================================================================= */
/* === AMD demo main program =============================================== */
/* ========================================================================= */

/* ------------------------------------------------------------------------- */
/* AMD Copyright (c) by Timothy A. Davis,				     */
/* Patrick R. Amestoy, and Iain S. Duff.  See ../README.txt for License.     */
/* email: davis at cise.ufl.edu    CISE Department, Univ. of Florida.        */
/* web: http://www.cise.ufl.edu/research/sparse/amd                          */
/* ------------------------------------------------------------------------- */

/* A simple C main program that illustrates the use of the ANSI C interface
 * to AMD.
 */

#include "StdAfx.h"


cAMD_Interf::cAMD_Interf(int aN) :
   mNb (aN),
   mV  (aN)
{
    // std::cout << "AMD::NB= " << mV.size() << "\n";
}


void cAMD_Interf::VerifN(int aN) const
{
    ELISE_ASSERT((aN>=0) && (aN<mNb) , "cAMD_Interf::VerifN");
}


void cAMD_Interf::AddArc(int aN1,int aN2,bool VerifDup)
{
    VerifN(aN1);
    VerifN(aN2);
    if (aN1 > aN2)
       ElSwap(aN1,aN2);

    if (VerifDup)
    {
        if (BoolFind(mV[aN1],aN2))
           return;
    }
    mV[aN1].push_back(aN2);
    mV[aN2].push_back(aN1);
}

static void  ShowMat(Im2D_Bits<1> aM)
{
   for (int anY =0 ; anY < aM.ty() ; anY++)
   {
      for (int anX =0 ; anX < aM.tx() ; anX++)
      {
          std::cout << (aM.get(anX,anY) ? "#" : "+") ;
      }
      std::cout << "\n";
   }
}

std::vector<int> cAMD_Interf::DoRank(bool show) 
{
   Im2D_Bits<1> aM(1,1);
   if (show)
      aM = Im2D_Bits<1>(mNb,mNb,0);

   std::vector<int> Ap,Ai,P;
   Ap.push_back(0);
   for (int aK=0 ; aK<mNb ; aK++)
   {
       std::vector<int> & aVK = mV[aK];
       std::sort(aVK.begin(),aVK.end());
       aVK.erase(std::unique(aVK.begin(),aVK.end()),aVK.end());

      for (int aI=0;aI<int(aVK.size()) ; aI++)
      {
          Ai.push_back(aVK[aI]);
          if (show)
          {
              aM.set(aK,aVK[aI],1);
          }
          // std::cout << " " << aVK[aI] ;
      }
      Ap.push_back(int(Ap.back() + aVK.size()));
      P.push_back(0);
      // std::cout << "\n";

   }



   double Control [AMD_CONTROL] , Info [AMD_INFO] ;

   amd_defaults (Control) ;
   // amd_control  (Control) ;

   int result;
   result = amd_order(mNb,VData(Ap),VData(Ai), VData(P),Control,Info);

   ELISE_ASSERT(result==0,"amd_order");
   //std::cout << "RES = "  << result << "\n";

   std::vector<int> aRes(mNb,-1);
   for (int aK=0 ; aK<mNb ; aK++)
   {
       aRes[P[aK]] = aK;
   }
   if (show)
   {
      for (int aK=0 ; aK<mNb ; aK++)
      {
   
      }
      std::cout << "\n";
   }
    // result = amd_order (n, Ap, Ai, P, Control, Info) ;

   if (show)
   {
       Im2D_Bits<1> aM2(mNb,mNb);
       for (int aX=0 ; aX<mNb ; aX++)
       {
           for (int aY=0 ; aY<mNb ; aY++)
           {
                 aM2.set(aRes[aX],aRes[aY],aM.get(aX,aY));
                // aM2.set(aX,aY,aM.get(P[aX],P[aY]));
           }
       }
       ShowMat(aM);
       std::cout << "\n";
       ShowMat(aM2);
   }
   return aRes;
}



int amd_demo_1 (void)
{
    /* The symmetric can_24 Harwell/Boeing matrix, including upper and lower
     * triangular parts, and the diagonal entries.  Note that this matrix is
     * 0-based, with row and column indices in the range 0 to n-1. */
    int n = 24, nz,
    Ap [ ] = { 0, 9, 15, 21, 27, 33, 39, 48, 57, 61, 70, 76, 82, 88, 94, 100,
	106, 110, 119, 128, 137, 143, 152, 156, 160 },
    Ai [ ] = {
	/* column  0: */    0, 5, 6, 12, 13, 17, 18, 19, 21,
	/* column  1: */    1, 8, 9, 13, 14, 17,
	/* column  2: */    2, 6, 11, 20, 21, 22,
	/* column  3: */    3, 7, 10, 15, 18, 19,
	/* column  4: */    4, 7, 9, 14, 15, 16,
	/* column  5: */    0, 5, 6, 12, 13, 17,
	/* column  6: */    0, 2, 5, 6, 11, 12, 19, 21, 23,
	/* column  7: */    3, 4, 7, 9, 14, 15, 16, 17, 18,
	/* column  8: */    1, 8, 9, 14,
	/* column  9: */    1, 4, 7, 8, 9, 13, 14, 17, 18,
	/* column 10: */    3, 10, 18, 19, 20, 21,
	/* column 11: */    2, 6, 11, 12, 21, 23,
	/* column 12: */    0, 5, 6, 11, 12, 23,
	/* column 13: */    0, 1, 5, 9, 13, 17,
	/* column 14: */    1, 4, 7, 8, 9, 14,
	/* column 15: */    3, 4, 7, 15, 16, 18,
	/* column 16: */    4, 7, 15, 16,
	/* column 17: */    0, 1, 5, 7, 9, 13, 17, 18, 19,
	/* column 18: */    0, 3, 7, 9, 10, 15, 17, 18, 19,
	/* column 19: */    0, 3, 6, 10, 17, 18, 19, 20, 21,
	/* column 20: */    2, 10, 19, 20, 21, 22,
	/* column 21: */    0, 2, 6, 10, 11, 19, 20, 21, 22,
	/* column 22: */    2, 20, 21, 22,
	/* column 23: */    6, 11, 12, 23 } ;

    int P [24], Pinv [24], i, j, k, jnew, p, inew, result ;
    double Control [AMD_CONTROL], Info [AMD_INFO] ;
    char A [24][24] ;

    /* here is an example of how to use AMD_VERSION.  This code will work in
     * any version of AMD. */
#if defined(AMD_VERSION) && (AMD_VERSION >= AMD_VERSION_CODE(1,2))
    printf ("AMD version %d.%d, date: %s\n", AMD_MAIN_VERSION, AMD_SUB_VERSION,
	    AMD_DATE) ;
#else
    printf ("AMD version: 1.1 or earlier\n") ;
#endif

    printf ("AMD demo, with the 24-by-24 Harwell/Boeing matrix, can_24:\n") ;

    /* get the default parameters, and print them */
    amd_defaults (Control) ;
    amd_control  (Control) ;

    /* print the input matrix */
    nz = Ap [n] ;
    printf ("\nInput matrix:  %d-by-%d, with %d entries.\n"
	   "   Note that for a symmetric matrix such as this one, only the\n"
	   "   strictly lower or upper triangular parts would need to be\n"
	   "   passed to AMD, since AMD computes the ordering of A+A'.  The\n"
	   "   diagonal entries are also not needed, since AMD ignores them.\n"
	   , n, n, nz) ;
    for (j = 0 ; j < n ; j++)
    {
	printf ("\nColumn: %d, number of entries: %d, with row indices in"
		" Ai [%d ... %d]:\n    row indices:",
		j, Ap [j+1] - Ap [j], Ap [j], Ap [j+1]-1) ;
	for (p = Ap [j] ; p < Ap [j+1] ; p++)
	{
	    i = Ai [p] ;
	    printf (" %d", i) ;
	}
	printf ("\n") ;
    }

    /* print a character plot of the input matrix.  This is only reasonable
     * because the matrix is small. */
    printf ("\nPlot of input matrix pattern:\n") ;
    for (j = 0 ; j < n ; j++)
    {
	for (i = 0 ; i < n ; i++) A [i][j] = '.' ;
	for (p = Ap [j] ; p < Ap [j+1] ; p++)
	{
	    i = Ai [p] ;
	    A [i][j] = 'X' ;
	}
    }
    printf ("    ") ;
    for (j = 0 ; j < n ; j++) printf (" %1d", j % 10) ;
    printf ("\n") ;
    for (i = 0 ; i < n ; i++)
    {
	printf ("%2d: ", i) ;
	for (j = 0 ; j < n ; j++)
	{
	    printf (" %c", A [i][j]) ;
	}
	printf ("\n") ;
    }

    /* order the matrix */
    result = amd_order (n, Ap, Ai, P, Control, Info) ;
    printf ("return value from amd_order: %d (should be %d)\n",
	result, AMD_OK) ;

    /* print the statistics */
    amd_info (Info) ;

    if (result != AMD_OK)
    {
	printf ("AMD failed\n") ;
	ElEXIT (1,"AMD DEMO") ;
    }

    /* print the permutation vector, P, and compute the inverse permutation */
    printf ("Permutation vector:\n") ;
    for (k = 0 ; k < n ; k++)
    {
	/* row/column j is the kth row/column in the permuted matrix */
	j = P [k] ;
	Pinv [j] = k ;
	printf (" %2d", j) ;
    }
    printf ("\n\n") ;

    printf ("Inverse permutation vector:\n") ;
    for (j = 0 ; j < n ; j++)
    {
	k = Pinv [j] ;
	printf (" %2d", k) ;
    }
    printf ("\n\n") ;

    /* print a character plot of the permuted matrix. */
    printf ("\nPlot of permuted matrix pattern:\n") ;
    for (jnew = 0 ; jnew < n ; jnew++)
    {
	j = P [jnew] ;
	for (inew = 0 ; inew < n ; inew++) A [inew][jnew] = '.' ;
	for (p = Ap [j] ; p < Ap [j+1] ; p++)
	{
	    inew = Pinv [Ai [p]] ;
	    A [inew][jnew] = 'X' ;
	}
    }
    printf ("    ") ;
    for (j = 0 ; j < n ; j++) printf (" %1d", j % 10) ;
    printf ("\n") ;
    for (i = 0 ; i < n ; i++)
    {
	printf ("%2d: ", i) ;
	for (j = 0 ; j < n ; j++)
	{
	    printf (" %c", A [i][j]) ;
	}
	printf ("\n") ;
    }

    return (0) ;
}
