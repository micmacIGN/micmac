#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"


/**
   \file CalibratedSpaceResection.cpp

   \brief file for implementing the space resection algorithm in calibrated case

 */

namespace MMVII
{

template <class Type>  class cElemSpaceResection
{
      public :
           typedef cPtxd<Type,3>   tP3;
           typedef cPolynom<Type>  tPol;
           typedef cPt2dr          tPairBC;

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

	   std::list<tPairBC>  ComputeBC() const;
	   static void OneTestCorrectness();

       private :

	   tP3 ToPt(const tPt3dr &  aP) {return tP3(aP.x(),aP.y(),aP.z());}
	   
	   // copy bundles, are unitary
	   tP3  A;
	   tP3  B;
	   tP3  C;

	   tP3  AB;      ///<  Vector  A -> B
	   tP3  AC;      ///<  Vector  A -> C
	   tP3  BC;      ///<  Vector  A -> C
	   Type abb;     ///<  (A->B).B

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
        A     (VUnit(ToPt(aDirBundlA))),
        B     (VUnit(ToPt(aDirBundlB))),
        C     (VUnit(ToPt(aDirBundlC))),

        AB      (B - A),
        AC      (C - A),
        BC      (C - B),
	abb     (Scal(AB,B)),

        gA (ToPt(aPGroundA)),
        gB (ToPt(aPGroundB)),
        gC (ToPt(aPGroundC)),

	gD2AB (SqN2(gA-gB)),
	gD2AC (SqN2(gA-gC)),
	gD2BC (SqN2(gB-gC)),

	rABC  (gD2AB/gD2AC),
	rCBA  (gD2BC/gD2AC)
{
}


template <class Type> std::list<cPt2dr>  cElemSpaceResection<Type>::ComputeBC() const
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
     b^2 + 2AB.B b + P(c) =0   =  (B+AB.B)^2 - (AB.B^2 -P(c))
     2nd degre equation in b 
     b =  - AB.B +E SQRT(AB.B^2  -P(c))   E in {-1,+1}
   # Q(c) = AB.B^2 -P(c)
     b =  - AB.B + E S(Q(c))
*/


     // P(c) =  (AB^2 -rABC (cC + AC)^2)
    tPol  aPol_AC_C =  PolSqN(AC,C);
    cPolynom<Type> aPc =  tPol::D0(SqN2(AB)) -  aPol_AC_C *rABC ;
    // Q(c) = AB.B^2 -P(c)
    cPolynom<Type> aQc =  tPol::D0(Square(abb)) - aPc;


  //  Now we can eliminate b using :   b =  - AB.B + E S(Q(c))   E in {-1,1} 
/* ======================== (2) resolve c =====================
    2nd conservation  of ratio
     |PC-PB|^2    |GC-GB|^2
      -----   =  ---------  = rCBA
     |PA-PC|^2    |GC-GA|^2


     ((1+c)C - (1+b)B)^2 = ((1+c)C -A)^2 rCBA
     (BC + cC -bB) ^2 = rCBA (AC + cC) ^2
     rCBA (AC + cC) ^2 = (BC +cC - (-AB.B + E *  S(Q)) B)^2 =   ((BC + AB.B B +c C)  - E S(Q) B) ^2

     rCBA (AC + cC) ^2 = (BC + AB.B B +c C)^2 -2 (BC + AB.B B +c C) .B E S(Q) + Q B^2
     B^2 = 1
     rCBA (AC + cC) ^2 - (BC + AB.B B +c C)^2  - Q  =  -2 (BC.B + AB.B  +c C.B) E S(Q)

                      
                       R(c) = -2E L(c) S(Q)
		       R^2(c) = 4 L(c)^2 Q(c)

*/
    /*
    {
	    Type b = 0.2; FakeUseIt(b);
	    Type c = 0.4;
	    Type Q = aQc.Value(c);
	    ///StdOut() << "REAL bc=" << b << " " << c<< "\n";
	    for (Type E : {-1,1})
	    {
	       Type b = -abb  + E * std::sqrt(Q);
	       StdOut() << "E=" << E  << " B=" << b << "\n";
	    }
	    StdOut() << "______________________________\n";
            // (BC + cC -bB) ^2 = rCBA (AC + cC) ^2
	    //  OK StdOut () << "CHK0 " <<  SqN2(BC+c*C-b*B) - rCBA * SqN2(AC+c*C) << "\n";

	    for (Type E : {1})
	    {
            // rCBA (AC + cC) ^2 = (BC +cC - (-AB.B + E *  S(Q)) B)^2 =   ((BC + AB.B B +c C)  - E S(Q) B) ^2
	    // OK : StdOut() << "CHEE " << E << " " << rCBA * SqN2(AC+c*C) -SqN2((BC+abb*B+c*C) - E * std::sqrt(Q) *B) << "\n";

            //  rCBA (AC + cC) ^2 - (BC + AB.B B +c C)^2  - Q  =  -2 (BC.B + AB.B  +c C.B) E S(Q)
	    //StdOut() << "CHEE " << E << " " 
            //		 << rCBA * SqN2(AC+c*C)  -SqN2(BC+abb*B+c*C) -Q +2 *(Scal(BC,B)+abb + c *Scal(C,B)) * E * std::sqrt(Q) << "\n";

            tPol  aLc ({Scal(BC,B)+abb,Scal(B,C)});
	    StdOut() << "CHEE " << E << " " 
		    << rCBA * aPol_AC_C.Value(c) -PolSqN(BC +  abb*B  ,C).Value(c) -Q +2 *aLc.Value(c) * E * std::sqrt(Q) << "\n";
	    StdOut() << "CHEE " << E << " " 
		    << rCBA * aPol_AC_C.Value(c) -PolSqN(BC +  abb*B  ,C).Value(c) -Q +2 *(Scal(BC,B)+abb + c *Scal(C,B)) * E * std::sqrt(Q) << "\n";
	    }
	    StdOut() << "______________________________\n";
    }
*/
       
    tPol  aRc =   aPol_AC_C *rCBA  -  aQc  -  PolSqN(BC +  abb*B  ,C);
    tPol  aLc ({Scal(BC,B)+abb,Scal(B,C)});
    tPol aSolver = Square(aRc) - aQc * Square(aLc) * 4;
    std::vector<Type> aVRoots = aSolver.RealRoots (1e-20,30);


    std::list<tPairBC> aRes;
    for (Type c : aVRoots)
    {
        for (Type E : {-1.0,1.0})
        {
	    Type Q =  aQc.Value(c);
	    if (Q>=0)
	    {
	        Type b =  -abb + E * std::sqrt(Q);

		tP3 PA = A;
		tP3 PB = (1+b)  * B;
		tP3 PC = (1+c)  * C;

		Type aCheckABC =  SqN2(PA-PB)/SqN2(PA-PC) - rABC;
		Type aCheckCBA =  SqN2(PC-PB)/SqN2(PC-PA) - rCBA;

		if (  (std::abs(aCheckABC)< 1e-5)  && (std::abs(aCheckCBA)< 1e-5) )
		{
                   aRes.push_back(tPairBC(b,c));
		   StdOut()  <<  "bc " << b << " " << c << " " << aCheckABC << " " << aCheckCBA << "\n";
		}
	    }
        }
    }

    return aRes;

}

template <class Type> void  cElemSpaceResection<Type>::OneTestCorrectness()
{
   cPt3dr A (1,0,0);
   cPt3dr B = VUnit(cPt3dr(1,0.1,0));
   cPt3dr C = VUnit(cPt3dr(1,0,0.2));

   cElemSpaceResection<tREAL8> anESR(A,B,C, A,B*1.2,C*1.4);
}

void TestResec()
{
   cElemSpaceResection<tREAL8>::OneTestCorrectness();
   cElemSpaceResection<tREAL16>::OneTestCorrectness();
   StdOut()<< "RESEC : DOOOOOnnnne \n"; getchar();
}

template class cElemSpaceResection<tREAL8>;
template class cElemSpaceResection<tREAL16>;





}; // MMVII

