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


     (2) resolve C

    ((1+b)B-A)^2 = ra ((1+c)C-A)^2
     (bB + AB) ^2 = ra (cC + AC) 
     b^2 + 2AB.B b  + (AB^2 -ra (c^2+ 2AC.C +AC^2)) =0
     b^2 + 2AB.B b + P(c) =0 

     b =  - AB.B +E SQRT(AB.B -P(c)^2)   E in {-1,+1}
     b =  B1 + E SQRT(B1-P(c)^2) = B1 +E S(Q(c))

     |PC-PB|^2    |GC-GB|^2
      -----   =  ---------  = rc
     |PA-PC|^2    |GA-GC|^2


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
	   tP3  mBdlA;
	   tP3  mBdlB;
	   tP3  mBdlC;

	   tP3  mBdlVAB;
	   Type mBdlDistAB2;


	   // copy of ground point coordinates, local precision
	   tP3  mGrA;
	   tP3  mGrB;
	   tP3  mGrC;


	   //  Square Distance between  ground points 
	   Type  mD2AB;
	   Type  mD2AC;
	   Type  mD2BC;
	   //  ratio of dist
	   Type  rABC;  ///<   mD2AB / mD2AC
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
        mBdlA (VUnit(ToPt(aDirBundlA))),
        mBdlB (VUnit(ToPt(aDirBundlB))),
        mBdlC (VUnit(ToPt(aDirBundlC))),

	mBdlVAB  (mBdlB-mBdlA),

        mGrA (ToPt(aPGroundA)),
        mGrB (ToPt(aPGroundB)),
        mGrC (ToPt(aPGroundC)),

	mD2AB (SqN2(mGrA-mGrB)),
	mD2AC (SqN2(mGrA-mGrC)),
	mD2BC (SqN2(mGrB-mGrC)),

	rABC  (mD2AB/mD2AC)
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

     b^2 + 2AB.B b  + (AB^2 -rABC (c^2+ 2AC.C +AC^2)) =0
     b^2 + 2AB.B b + P(c) =0   => aPc

     b =  - AB.B +E SQRT(AB.B -P(c)^2)   E in {-1,+1}
     b =  B1 + E SQRT(B1-P(c)^2) = B1 +E S(Q(c))
*/

    cPolynom<Type> aPc =  tPol::D0(1);


    FakeUseIt(aPc);
}


template class cElemSpaceResection<tREAL8>;
template class cElemSpaceResection<tREAL16>;





}; // MMVII

