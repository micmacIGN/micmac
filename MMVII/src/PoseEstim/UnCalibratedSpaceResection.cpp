#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"


/**
   \file CalibratedSpaceResection.cpp

   \brief file for implementing the space resection algorithm in calibrated case

 */

namespace MMVII
{

/**
 *
 * Class for solving the "11 parameter" equation, AKA uncalibrated resection
 */
template <class Type>  class cUncalibSpaceRessection
{
      public :
           cUncalibSpaceRessection
           (
	       const cSet2D3D &
	   );

       private :
};



#if (0)

template <class Type> 
   cElemSpaceResection<Type>::cElemSpaceResection
   (
       const tTri & aTriB,
       const tTri & aTriG
   ) :
        nNormA  (Norm2(aTriB.Pt(0))),
        nNormB  (Norm2(aTriB.Pt(1))),
        nNormC  (Norm2(aTriB.Pt(2))),
        A       (aTriB.Pt(0) / nNormA),
        B       (aTriB.Pt(1) / nNormB),
        C       (aTriB.Pt(2) / nNormC),

        AB      (B - A),
        AC      (C - A),
        BC      (C - B),
	abb     (Scal(AB,B)),

	mTriG   (aTriG),
        gA (aTriG.Pt(0)),
        gB (aTriG.Pt(1)),
        gC (aTriG.Pt(2)),

	gD2AB (SqN2(mTriG.KVect(0))),
	gD2AC (SqN2(mTriG.KVect(2))),
	gD2BC (SqN2(mTriG.KVect(1))),
        mSqPerimG ( gD2AB + gD2AC + gD2BC),

	rABC  (gD2AB/gD2AC),
	rCBA  (gD2BC/gD2AC)
{
}



template <class Type> std::list<cPtxd<Type,3>>  cElemSpaceResection<Type>::ComputeBC() const
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

    std::list<tResBC> aRes;

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

                Type aD2AB =  SqN2(PA-PB);
                Type aD2AC =  SqN2(PA-PC);
                Type aD2BC =  SqN2(PB-PC);

		// Due to squaring sign of E is not always consistant, so now we check if ratio are really found
		Type aCheckABC =  aD2AB/aD2AC - rABC;
		Type aCheckCBA =  aD2BC/aD2AC - rCBA;

		//  test with 1e-5  generate bench problem ...
		if (  (std::abs(aCheckABC)< 1e-3)  && (std::abs(aCheckCBA)< 1e-3) )
		{
                   Type aSqPerim = aD2AB + aD2AC + aD2BC;
                   aRes.push_back(tResBC((1+b),(1+c),std::sqrt(mSqPerimG/aSqPerim)));
		   // StdOut()  << " E " << E <<  " bc " << b << " " << c << " " << aCheckABC << " " << aCheckCBA << "\n";
		}
	    }
        }
    }
    return aRes;
}

template <class Type> cTriangle<Type,3>  cElemSpaceResection<Type>::BC2LocCoord(const tResBC & aRBC) const 
{
     const Type & b =  aRBC.x();
     const Type & c =  aRBC.y();
     const Type & aMul = aRBC.z();

     return  cTriangle<Type,3>(aMul*A,(aMul*b)*B,(aMul*c)*C);
}

template <class Type> cIsometry3D<Type> cElemSpaceResection<Type>::BC2Pose(const tResBC & aRBC) const 
{
     cTriangle<Type,3> aTri = BC2LocCoord(aRBC);

     return cIsometry3D<Type>::FromTriInAndOut(0,aTri,0,mTriG);
}


template <class Type> void  cElemSpaceResection<Type>::OneTestCorrectness()
{
   static int aCpt=0; aCpt++;
   {
       // generate 3 bundle not too degenared => 0,P0,P1,P2 cot coplanar
       cTriangle<Type,3> aTriBund = RandomTetraTriangRegul<Type>(1e-3,1e2);

       //   Generate b &c ;  Too extrem value =>  unaccuracyy bench ; not : RandUnif_C_NotNull(1e-2) * 10
       Type b = pow(2.0,RandUnif_C());
       Type c = pow(2.0,RandUnif_C());

       // comput A,B,C  with  ratio given by b,c and A unitary
       cPtxd<Type,3> A = VUnit(aTriBund.Pt(0));
       cPtxd<Type,3> B = VUnit(aTriBund.Pt(1))*b;
       cPtxd<Type,3> C = VUnit(aTriBund.Pt(2))*c;

       //  put them anywhere and with any ratio using a random similitud
       cSimilitud3D<Type> aSim(
		               static_cast<Type>(RandUnif_C_NotNull(1e-2)*10.0),
			       cPtxd<Type,3>::PRandC()*static_cast<Type>(100.0),
			       cRotation3D<Type>::RandomRot()
                         );
       cTriangle<Type,3> aTriG(aSim.Value(A),aSim.Value(B),aSim.Value(C));

       //  Now see that we can recover b & c
       cElemSpaceResection<Type> anESR(aTriBund,aTriG);
       auto aLBC = anESR.ComputeBC();  //list of b,c,Perimeter

       cWhichMin<cPtxd<Type,3>,Type> aWMin(cPtxd<Type,3>(0,0,0),1e10);  // will extract b,c closest to ours
       for (auto & aTripl : aLBC)
       {
           aWMin.Add(aTripl,std::abs(aTripl.x()-b)+std::abs(aTripl.y()-c));
       }
       MMVII_INTERNAL_ASSERT_bench(aWMin.ValExtre()<1e-4,"2 value in OneTestCorrectness");  // is it close enough


       //  Now see that if can recover local coord from b,c
       cTriangle<Type,3>  aTriComp = anESR.BC2LocCoord(aWMin.IndexExtre());
       for (auto aK : {0,1,2})
       {
              //  Test the triangle Local and Ground are isometric
             double aDif = RelativeSafeDifference(Norm2(aTriG.KVect(aK)),Norm2(aTriComp.KVect(aK))) ;
             MMVII_INTERNAL_ASSERT_bench(aDif<1e-4,"Local coord in OneTestCorrectness");
              //  Test the Local coordinate are aligned on bundles
             double aAngle = AbsAngleTrnk(aTriBund.Pt(aK),aTriComp.Pt(aK))  ;
             MMVII_INTERNAL_ASSERT_bench(aAngle<1e-4,"Local coord in OneTestCorrectness");
       }
       //  Now see that if can recover local pose from b,c
       cIsometry3D<Type>  aPose = anESR.BC2Pose(aWMin.IndexExtre());
       for (auto aK : {0,1,2})
       {
           // check that  Bundle is colinear to Pose^-1 (PGround)
           cPtxd<Type,3>  aPLoc=  aPose.Inverse(aTriG.Pt(aK));
           Type aAngle = AbsAngleTrnk(aPLoc,aTriBund.Pt(aK));
           MMVII_INTERNAL_ASSERT_bench(aAngle<1e-4,"Pose in OneTestCorrectness");
       }
   }
}
template class cElemSpaceResection<tREAL8>;
template class cElemSpaceResection<tREAL16>;

void BenchUnCalibResection()
{
   for (int aK=0 ; aK< 1000 ; aK++)
   {
      cElemSpaceResection<tREAL8>::OneTestCorrectness();
      cElemSpaceResection<tREAL16>::OneTestCorrectness();
   }
}


void BenchPoseEstim(cParamExeBench & aParam)
{
   if (! aParam.NewBench("PoseEstim")) return;

   BenchUnCalibResection();
   aParam.EndBench();
}



#endif


}; // MMVII

