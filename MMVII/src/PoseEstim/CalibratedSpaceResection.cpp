#include "MMVII_Ptxd.h"


/**
   \file CalibratedSpaceResection.cpp

   \brief file for implementing the space resection algorithm in calibrated case

 */

/*
      3 direction   A,B,C   we assume ||A|| = ||B|| = ||C|| = 1
      We parametrize 3 point on the bundle by 2 parameters b & c:
           PA  = A  (arbitrarily we fix on this bundle)
	   PB  = B(1+b)
	   PC  = C(1+c)
      This parametrization is made to be more numerically stable when 
     the bundle are close to each others (b & c small)

     We know  GA,GB,GC  the ground point

     (1) eliminate b

     |PA-PB|^2    |GA-GB|^2
      -----   =  ---------  = rA
     |PA-PC|^2    |GA-GC|^2



    ((1+b)B-A)^2 = ra ((1+c)C-A)^2
     (bB + AB) ^2 = ra (cC + AC) 
     b^2 + 2AB.B b  + (AB^2 -ra (c^2+ 2AC.C +AC^2)) =0
     b^2 + 2AB.B b + P(c) =0 

     b =  - AB.B +E SQRT(AB.B -P(c)^2)   E in {-1,+1}
     b =  B1 + E SQRT(B1-P(c)^2) = B1 +E S(Q(c))

     (2) resolve C
     |PC-PB|^2    |GC-GB|^2
      -----   =  ---------  = rc
     |PA-PC|^2    |GC-GA|^2


     ((1+c)C - (1+b)B)^2 = ((1+c)C -A)^2 rc
     (BC + cC -bB) ^2 = rc (AC + cC) ^2
     rc (AC + cC) ^2 = (BC +cC - (B1 + E *  S(Q)) B)^2 =   ((BC-B1B) +c C - E S(Q) B) ^2

     rc (AC + cC) ^2  - (BC-B1B)^2  - B2 Q = 2 E (BC-B1B).B S(Q)
       


*/

namespace MMVII
{

	/*
struct  cPair3D3D
{
     public :
          cPair2D3D(const cPt2dr &,const cPt3dr &);
          cPt2dr mP2;
          cPt3dr mP3;
};
*/


template <class Type>  class cElemSpaceResection
{
      public :
           typedef cPtxd<Type,3>   tP3;
           typedef cPolynom<Type>  tPol;


	   tP3 ToPt(const tPt3dr &  aP) {return tP3(aP.x(),aP.y(),aP.z());}

	   // All points in are in REAL8, only intermediar computation is eventually on REAL16
           cElemSpaceResection
           (
	        const tPt3dr & aDirBundlA,
	        const tPt3dr & aDirBundlB,
	        const tPt3dr & aDirBundlC,
	        const tPt3dr & aPGroundA,
	        const tPt3dr & aPGroundB,
	        const tPt3dr & aPGroundC
	   );
	   
	   // copy bundles, are unitary
	   tP3  bA;
	   tP3  bB;
	   tP3  bC;

	   tP3  bVAB;      ///<  Vector  A -> B
	   tP3  bVAC;      ///<  Vector  A -> C
	   Type bDistAB2;  ///<  ||A-B||^2 
	   Type bABsB;     ///<  (A->B).B


	   // copy of ground point coordinates, local precision
	   tP3  gA;
	   tP3  gB;
	   tP3  gC;


	   //  Square Distance between  ground points 
	   Type  gD2AB;
	   Type  gD2AC;
	   Type  gD2BC;
	   //  ratio of dist
	   Type  rABC;  ///<   mD2AB / mD2AC
	   Type  rCBA;  ///<   mD2CB / mD2CA
};

template <class Type> 
   cElemSpaceResection<Type>::cElemSpaceResection
   (
       const tPt3dr & aDirBundlA,
       const tPt3dr & aDirBundlB,
       const tPt3dr & aDirBundlC,
       const tPt3dr & aPGroundA,
       const tPt3dr & aPGroundB,
       const tPt3dr & aPGroundC
   ) :
        bA     (VUnit(ToPt(aDirBundlA))),
        bB     (VUnit(ToPt(aDirBundlB))),
        bC     (VUnit(ToPt(aDirBundlC))),

        bVAB      (bB - bA),
        bVAC      (bC - bA),
	bDistAB2  (SqN2(bVAB)),
	bABsB     (Scal(bVAB,bB)),

        gA (ToPt(aPGroundA)),
        gB (ToPt(aPGroundB)),
        gC (ToPt(aPGroundC)),

	gD2AB (SqN2(gA-gB)),
	gD2AC (SqN2(gA-gC)),
	gD2BC (SqN2(gB-gC)),

	rABC  (gD2AB/gD2AC),
	rCBA  (gD2BC/gD2AC)
{
/*
      3 direction  of bundles  A,B,C   we have made ||A|| = ||B|| = ||C|| = 1
      We parametrize 3 point on the bundle by 2 parameters b & c:
           PA  = A  (arbitrarily we fix on this bundle)
	   PB  = B(1+b)
	   PC  = C(1+c)
      This parametrization is made to be more numerically stable when 
     the bundle are close to each others which is a current case (b & c small)
*/


/*  ===============  (1) eliminate b  =====================
     We have a conservation of ratio of distance :

     |PA-PB|^2    |GA-GB|^2
      -----   =  ---------  = rABC
     |PA-PC|^2    |GA-GC|^2


    ((1+b)B-A)^2 = rABC ((1+c)C-A)^2
     (bB + AB) ^2 = rABC (cC + AC) ^2

     b^2 + 2AB.B b  + (AB^2 -rABC(cC + AC)^2  ) =0
   # P(c) =  (AB^2 -rABC (cC + AC)^2)
     b^2 + 2AB.B b + P(c) =0   => 

     b =  - AB.B +E SQRT(AB.B -P(c))   E in {-1,+1}
   # Q(c) = AB.B -P(c)
     b =  - AB.B + E S(Q(c))
*/


     // P(c) =  (AB^2 -rABC (cC + AC)^2)
    cPolynom<Type> aPc =  tPol::D0(bDistAB2) -  PolSqN(bVAC,bC) *rABC ;
    // Q(c) = AB.B -P(c)^2
    cPolynom<Type> aQc =  tPol::D0(bABsB) - aPc;

  //  Now we can eliminate b using :   b =  - AB.B + E S(Q(c))   E in {-1,1} 
/* ======================== (2) resolve c =====================
     |PC-PB|^2    |GC-GB|^2
      -----   =  ---------  = rCBA
     |PA-PC|^2    |GC-GA|^2


     ((1+c)C - (1+b)B)^2 = ((1+c)C -A)^2 rCBA
     (BC + cC -bB) ^2 = rCBA (AC + cC) ^2
     rCBA (AC + cC) ^2 = (BC +cC - (B1 + E *  S(Q)) B)^2 =   ((BC-B1B) +c C - E S(Q) B) ^2

     rc (AC + cC) ^2  - (BC-B1B)^2  - B2 Q = 2 E (BC-B1B).B S(Q)
*/
       

    for (Type E : {-1.0,1.0})
    {
        FakeUseIt(E);
    }

    FakeUseIt(aQc);
}


template class cElemSpaceResection<tREAL8>;
template class cElemSpaceResection<tREAL16>;





}; // MMVII

