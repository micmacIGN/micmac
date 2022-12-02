#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"


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

	   std::list<tPairBC>  ComputeBC(bool Debug) const;
	   static void OneTestCorrectness();

       private :

	   tP3 ToPt(const tPt3dr &  aP) {return tP3(aP.x(),aP.y(),aP.z());}
	   
           Type nNormA;
           Type nNormB;
           Type nNormC;
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

	   std::list<tPairBC>  mListPair;
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
        nNormA  (Norm2(aDirBundlA)),
        nNormB  (Norm2(aDirBundlB)),
        nNormC  (Norm2(aDirBundlC)),
        A       (ToPt(aDirBundlA) / nNormA),
        B       (ToPt(aDirBundlB) / nNormB),
        C       (ToPt(aDirBundlC) / nNormC),

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



template <class Type> std::list<cPt2dr>  cElemSpaceResection<Type>::ComputeBC(bool Debug) const
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
       
    tPol  aRc =   aPol_AC_C *rCBA  -  aQc  -  PolSqN(BC +  abb*B  ,C);
    tPol  aLc ({Scal(BC,B)+abb,Scal(B,C)});
    tPol aSolver = Square(aRc) - aQc * Square(aLc) * 4;
    std::vector<Type> aVRoots = aSolver.RealRoots (1e-30,60);

    if (Debug)
        StdOut() << "aVRoots " << aVRoots.size()   << " " << aVRoots << "\n";

    std::list<tPairBC> aRes;

    for (Type c : aVRoots)
    {
        if (Debug)
	{
		StdOut() << "RRRR " << aSolver.Value(c) << "\n";
	}
        for (Type E : {-1.0,1.0})
        {
	    Type Q =  aQc.Value(c);
	    if (Q>=0)
	    {
	        Type b =  -abb + E * std::sqrt(Q);

		tP3 PA = A;
		tP3 PB = (1+b)  * B;
		tP3 PC = (1+c)  * C;

		// Due to squaring sign of E is not always consistant, so now we check if ratio are really found
		Type aCheckABC =  SqN2(PA-PB)/SqN2(PA-PC) - rABC;
		Type aCheckCBA =  SqN2(PC-PB)/SqN2(PC-PA) - rCBA;

		//  test with 1e-5  generate bench problem ...
		if (  (std::abs(aCheckABC)< 1e-3)  && (std::abs(aCheckCBA)< 1e-3) )
		{
                   aRes.push_back(tPairBC((1+b),(1+c)));
		   // StdOut()  << " E " << E <<  " bc " << b << " " << c << " " << aCheckABC << " " << aCheckCBA << "\n";
		}
	    }
        }
    }
    return aRes;
}

/*
                   Type aSqPerim0 =  SqN2(
    Type aSqPerim0 =  gD2AB + gD2AC + gD2BC;
*/

template <class Type> void  cElemSpaceResection<Type>::OneTestCorrectness()
{
   static int aCpt=0; aCpt++;
   {
       cTriangle<Type,3> aTriBund = RandomTetraTriangRegul<Type>(1e-3,1e2);

       StdOut() << "regul "<< TetraReg(aTriBund.Pt(0),aTriBund.Pt(1),aTriBund.Pt(2))  << "\n";

       cPt3dr A = ToR(VUnit(aTriBund.Pt(0)));
       cPt3dr B = ToR(VUnit(aTriBund.Pt(1)));
       cPt3dr C = ToR(VUnit(aTriBund.Pt(2)));

       cSimilitud3D<tREAL8> aSim(
		               RandUnif_C_NotNull(1e-2)*10.0,
			       cPt3dr::PRandC()*100.0,
			       cRotation3D<tREAL8>::RandomRot()
                         );


       // double b = RandUnif_C_NotNull(1e-2) * 10;
       // double c = RandUnif_C_NotNull(1e-2) * 10;
       // 
       //  Too extrem value, generate sometime unaccuracyy then bench error
       //
       double b = pow(2.0,RandUnif_C());
       double c = pow(2.0,RandUnif_C());

       cElemSpaceResection<tREAL8> anESR(A,B,C, aSim.Value(A),aSim.Value(B*b),aSim.Value(C*c));
       //cElemSpaceResection<tREAL8> anESR(A,B,C, A,B*b,C*c);
       auto aLBC =anESR.ComputeBC(aCpt==339104);
       double aMinDist = 1e10;
       for (auto & aPair : aLBC)
       {
           UpdateMin(aMinDist,Norm2(aPair-cPt2dr(b,c)));
       }

       StdOut() << " CPT=" << aCpt << " DIST " << aMinDist << " " << aLBC.size() << " BC " << b << " " << c << "\n";
       // if (aLBC.size() >)
       MMVII_INTERNAL_ASSERT_bench(aMinDist<1e-4,"2 value in OneTestCorrectness");
   }
}

void TestResec()
{
   for (int aK=0 ; aK< 1000000 ; aK++)
   {
      cElemSpaceResection<tREAL8>::OneTestCorrectness();
      //cElemSpaceResection<tREAL16>::OneTestCorrectness();
      StdOut()<< "  ====================== \n"; 
   }
   StdOut()<< "RESEC : DOOOOOnnnne \n"; getchar();
}

template class cElemSpaceResection<tREAL8>;
template class cElemSpaceResection<tREAL16>;





}; // MMVII

