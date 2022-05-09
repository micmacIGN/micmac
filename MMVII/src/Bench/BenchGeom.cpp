#include "include/MMVII_all.h"
//#include "include/MMVII_Tpl_Images.h"

namespace MMVII
{

template<class Type> void TplBenchRotation3D(cParamExeBench & aParam)
{
   int aNbTest = std::min(10000,300*(1+aParam.Level()));
   for (int aKTest=0 ; aKTest<aNbTest ; aKTest++)
   {
       cPtxd<Type,3> aP0 = cPtxd<Type,3>::PRandUnit();
       {
          // Compute a Normal Repair completing 1 vect
          cRotation3D<Type> aRP0 = cRotation3D<Type>::CompleteRON(aP0);
          MMVII_INTERNAL_ASSERT_bench(aRP0.Mat().Unitarity()<1e-5,"Complete RON 1 Vect"); // Its a rot
          MMVII_INTERNAL_ASSERT_bench(Norm1( aP0-aRP0.AxeI())<1e-5,"Complete RON 1 Vect"); //Its axe is P0
       }

       // Compute a Normal Repair completing 2 vect
       cPtxd<Type,3> aP1  =  cPtxd<Type,3>::PRandUnitDiff(aP0,1e-2);
       cRotation3D<Type> aRP01 = cRotation3D<Type>::CompleteRON(aP0,aP1);
       {
          MMVII_INTERNAL_ASSERT_bench(aRP01.Mat().Unitarity()<1e-5,"Complete RON 1 Vect"); // Its a rot
          MMVII_INTERNAL_ASSERT_bench(Norm1( aP0-aRP01.AxeI())<1e-5,"Complete RON 1 Vect"); //Its axe is P0
          MMVII_INTERNAL_ASSERT_bench(std::abs(Scal(aP1,aRP01.AxeK()))<1e-5,"Complete RON 1 Vect"); // Orthog to P1
       }


       // Compute the  given rotation from a given   Axe/Angle
       Type aTeta = RandUnif_C_NotNull(1e-3)*3;//Avoid close pi,(pb modulo 2pi), Small Teta (Axe not accurate)
       cRotation3D<Type>  aRAxe = cRotation3D<Type>::RotFromAxe(aP0,aTeta);
       // Test rotation and Axe
       {
         MMVII_INTERNAL_ASSERT_bench(aRAxe.Mat().Unitarity()<1e-5,"Complete RON 1 Vect"); // Its a rot
         cPtxd<Type,3> aImAxe = aRAxe.Value(aP0);
         MMVII_INTERNAL_ASSERT_bench(Norm2(aP0-aImAxe)<1e-5,"Complete RON 1 Vect"); // P0 is its axe
       }

       // Test teta
       {
           cPtxd<Type,3> aJ = aRP01.AxeJ();  // we complete the Axe with any ortog repair
           cPtxd<Type,3> aK = aRP01.AxeK();

           cPtxd<Type,3> aJA = aRAxe.Value(aJ);
           Type aDif = std::abs(Cos(aJ,aJA)-cos(aTeta)) +  std::abs(Cos(aK,aJA)-sin(aTeta));
           MMVII_INTERNAL_ASSERT_bench(aDif<1e-5,"Rotation from Axe"); // P0 is its axe
       }
       // Test extract Axe from rot
       {
           cPtxd<Type,3> aAxe2;
           Type aTeta2;
           aRAxe.ExtractAxe(aAxe2,aTeta2);
           // Axes & teta are defined up to a sign
           if (Scal(aAxe2,aP0) < 0)
           {
               aAxe2 =  aAxe2 * Type(-1);
               aTeta2 = - aTeta2;
           }
           Type AccAngle = tNumTrait<Type>::Accuracy() * 1e-3;
           Type AccAxe =  AccAngle / std::abs(aTeta) ;  // The smallest teta the less accurate axes

           MMVII_INTERNAL_ASSERT_bench(Norm2(aAxe2-aP0)<AccAxe,"Axe from Rot"); // P0 is its axe
           MMVII_INTERNAL_ASSERT_bench(std::abs(aTeta- aTeta2)<AccAngle,"Angle from Rot"); // P0 is its axe
       }

       // Bench quaternions
       {
           cPtxd<Type,4> aP1 = cPtxd<Type,4>::PRandUnit();
           {
              //  check order on computing neutral element
              cPtxd<Type,4> aQId = cPtxd<Type,4>(1,0,0,0);
              MMVII_INTERNAL_ASSERT_bench(Norm2(aP1-aQId*aP1)<1e-5,"Quat assoc"); // Check (1,0,0,0) is neutral
           }

           {
               //  check associativity
               cPtxd<Type,4> aP2 = cPtxd<Type,4>::PRandUnit();
               cPtxd<Type,4> aP3 = cPtxd<Type,4>::PRandUnit();

               cPtxd<Type,4> aP12_3 = (aP1*aP2) * aP3;
               cPtxd<Type,4> aP1_23 = aP1*(aP2 * aP3);
               MMVII_INTERNAL_ASSERT_bench(Norm2(aP12_3-aP1_23)<1e-5,"Quat assoc"); // Is it associative
           }

           // Matr to Rot
           cDenseMatrix<Type> aM1 =Quat2MatrRot(aP1);
           MMVII_INTERNAL_ASSERT_bench(aM1.Unitarity()<1e-5,"Quat assoc"); // Check its a rotation

           // Check morphism  Quat -> Rot
           {
              cPtxd<Type,4> aP2 = cPtxd<Type,4>::PRandUnit();
              cDenseMatrix<Type> aMp1p2 = aM1 * Quat2MatrRot(aP2);
              cDenseMatrix<Type> aMp12 =  Quat2MatrRot(aP1*aP2);
              MMVII_INTERNAL_ASSERT_bench(aMp1p2.DIm().L2Dist(aMp12.DIm())<1e-5,"Morphism Quat/Rot");
           }

           // Check "pseudo invertibility",  in fact Quat -> Rot is a proj  
           //    Quat -> Rot ->Quat  does not give identity
           {
              cPtxd<Type,4> aPM1 =MatrRot2Quat(aM1);
              cDenseMatrix<Type> aMPM1 =Quat2MatrRot(aPM1);
              MMVII_INTERNAL_ASSERT_bench(std::abs(Norm2(aPM1)-1)<1e-5,"Morphism Quat/Rot"); // PM1 is unit
              // MMVII_INTERNAL_ASSERT_bench(NormInf(aP1-aPM1)<1e-5,"??"); DOES NOT WORK

           }
           //    Rot -> Quat -> Rot   do  give identity
           {
                cDenseMatrix<Type> aM1 = cRotation3D<Type>::RandomRot().Mat();
                cPtxd<Type,4> aPM1 =MatrRot2Quat(aM1);
                cDenseMatrix<Type> aMPM1 =Quat2MatrRot(aPM1);
                Type aDist =  aM1.DIm().L2Dist(aMPM1.DIm()) ;
                MMVII_INTERNAL_ASSERT_bench(aDist<1e-5,"Rot->Quat->Rot"); // Inversion this way
           }
       }
   }
}

void BenchRotation3D(cParamExeBench & aParam)
{
    TplBenchRotation3D<tREAL4 >(aParam);
    TplBenchRotation3D<tREAL8 >(aParam);
    TplBenchRotation3D<tREAL16>(aParam);
}

template<class Type> void TplBenchIsometrie(cParamExeBench & aParam)
{
    Type aEps = tElemNumTrait<Type>::Accuracy();
    for (int aKCpt=0 ; aKCpt<100 ; aKCpt++)
    {
        int aK1 =RandUnif_N(3);
        int aK2 =RandUnif_N(3);
	cTriangle<Type,3>  aT1 = cTriangle<Type,3>::RandomTri(10.0);
	cTriangle<Type,3>  aT2 = cTriangle<Type,3>::RandomTri(10.0);

        cIsometry3D<Type>  aIsom = cIsometry3D<Type>::FromTriInAndOut(aK1,aT1,aK2,aT2);
        cSimilitud3D<Type> aSim = cSimilitud3D<Type>::FromTriInAndOut(aK1,aT1,aK2,aT2);

	// Check image of point
	MMVII_INTERNAL_ASSERT_bench(Norm2(aIsom.Value(aT1.Pt(aK1)) -aT2.Pt(aK2)) <aEps,"FromTriInAndOut p1 !!");
	MMVII_INTERNAL_ASSERT_bench(Norm2(aSim.Value(aT1.Pt(aK1)) -aT2.Pt(aK2)) <aEps,"FromTriInAndOut p1 !!");

	// Check image of vect1
	{
	     cPtxd<Type,3> aV1 = aIsom.Rot().Value(aT1.KVect(aK1));
	     cPtxd<Type,3> aV2 = aT2.KVect(aK2);
	     MMVII_INTERNAL_ASSERT_bench( std::abs(Cos(aV1,aV2)-1)  <aEps,"Isom:FromTriInAndOut v2");

	      aV1 = aSim.Value(aT1.Pt((aK1+1)%3)) - aSim.Value(aT1.Pt(aK1));
	     MMVII_INTERNAL_ASSERT_bench( Norm2(aV1  -aV2)  <aEps,"Sim:FromTriInAndOut v2");
	}
	// Check Normals
	{
	    cPtxd<Type,3> aN1 = aIsom.Rot().Value(NormalUnit(aT1));
	    cPtxd<Type,3> aN2 = NormalUnit(aT2);
	
	    MMVII_INTERNAL_ASSERT_bench( Norm2(aN1 - aN2)  <aEps,"FromTriInAndOut Normals");

	    aN1 = aSim.Rot().Value(NormalUnit(aT1));
	    MMVII_INTERNAL_ASSERT_bench( Norm2(aN1 - aN2)  <aEps,"FromTriInAndOut Normals");
	}



	cPtxd<Type,3> aP1 = cPtxd<Type,3>::PRandC() * Type(20.0);
	cPtxd<Type,3> aQ1 = cPtxd<Type,3>::PRandC() * Type(20.0);
	Type aD1 = Norm2(aP1 - aQ1) ;
	cPtxd<Type,3> aP2 = aIsom.Value(aP1);
	cPtxd<Type,3> aQ2 = aIsom.Value(aQ1);
	Type aD2 = Norm2(aP2 - aQ2) ;

	// Is it isometric
	MMVII_INTERNAL_ASSERT_bench(std::abs(aD1 - aD2) <aEps,"cIsometry3D is not isometric !!");

	// Check Value/Inverse
	cPtxd<Type,3> aP3 = aIsom.Inverse(aP2);
	MMVII_INTERNAL_ASSERT_bench(Norm2(aP1 - aP3)<aEps,"cIsometry3D Value/Inverse");
	aP3 = aSim.Inverse(aSim.Value(aP1));
	MMVII_INTERNAL_ASSERT_bench(Norm2(aP1 - aP3)<aEps,"cIsometry3D Value/Inverse");


	// Check  I o I-1 = Id
	cIsometry3D<Type> aIsoI = aIsom.MapInverse();
	aP3 = aIsoI.Value(aP2);
	MMVII_INTERNAL_ASSERT_bench(Norm2(aP1 - aP3)<aEps,"cIsometry3D MapInverse");

	cSimilitud3D<Type> aSimI = aSim.MapInverse();
	aP3 = aSimI.Value(aSim.Value(aP1));
	MMVII_INTERNAL_ASSERT_bench(Norm2(aP1 - aP3)<aEps,"Sim MapInverse");
    }
    // StdOut() << "=======================\n";
    // =========== Test  ext3d of 2D similitude ===========
    for (int aKCpt=0 ; aKCpt<10000 ; aKCpt++)
    {

	 // Parameter of rand 2D sim
         cPtxd<Type,2> aSc = cPtxd<Type,2>::PRandUnitDiff(cPtxd<Type,2>(0,0),1e-1) * Type(3.0);
         cPtxd<Type,2> aTr = cPtxd<Type,2>::PRandC() * Type(10.0);

	 //   2 DSIm and its 3D Ext
	 cSim2D<Type>       aS2 (aTr,aSc);
	 cSimilitud3D<Type> aS3 = aS2.Ext3D();

	 //  Random 3D point and its 2D projection
	 cPtxd<Type,3> aP3 = cPtxd<Type,3>::PRandC() * Type(10.0);
	 cPtxd<Type,2> aP2 = Proj(aP3);

	 //  Check  S3D(P3) ~ S2D(P2)
	 cPtxd<Type,3> aQ3 = aS3.Value(aP3);
	 cPtxd<Type,2> aQ2 = aS2.Value(aP2);
	 MMVII_INTERNAL_ASSERT_bench(Norm2( Proj(aQ3)  -aQ2)<aEps,"Sim MapInverse");

    }
    // =========== Test  ext3d of 2D similitude ===========
    for (int aKCpt=0 ; aKCpt<10000 ; aKCpt++)
    {
	    //  generatere randomly :  triangle , point inside,  2D segment
	 cTriangle<Type,3> aTri = cTriangle<Type,3>::RandomTri(100.0,1e-2);
	 int aK = round_down(RandUnif_N(3));
         cPtxd<Type,2> aP1 = cPtxd<Type,2>::PRandC() * Type(10.0);
         cPtxd<Type,2> aP2 = aP1+ cPtxd<Type,2>::PRandUnitDiff(cPtxd<Type,2>(0,0),1e-1) * Type(3.0);

	   // compute sim matching tri to  seg
         cSimilitud3D<Type> aSim =  cSimilitud3D<Type>::FromTriInAndSeg(aP1,aP2,aK,aTri);

	   // Test Sim(T(K)) -> P1  ;    Sim(T(K+1)) -> P2  ;  Sim(T(K+1))  is on plane ;  normal orientations OK
	 MMVII_INTERNAL_ASSERT_bench( Norm2(TP3z0(aP1)  - aSim.Value(aTri.Pt(aK)))<aEps,"Simil Tri3D  P1");
	 MMVII_INTERNAL_ASSERT_bench( Norm2(TP3z0(aP2)  - aSim.Value(aTri.Pt((aK+1)%3)))<aEps,"Simil Tri3D  P1");
	 MMVII_INTERNAL_ASSERT_bench( std::abs(aSim.Value(aTri.Pt((aK+1)%2)).z()) <aEps,"Simil Tri3D  P1");

	 cPtxd<Type,3> aN = NormalUnit(aTri);
	 cPtxd<Type,3> aImN = aSim.Value(aN) - aSim.Value(cPtxd<Type,3>(0,0,0)) ;  // Image of normal as vector
	 cPtxd<Type,3> aVecK(0,0,1); // we dont have aImN==aVK because scaling, BUT colinear and same orient, test cosinus then

	 MMVII_INTERNAL_ASSERT_bench( std::abs(Cos(aImN,aVecK) -1) <aEps,"Simil Tri3D  P1");
    }
}




void BenchIsometrie(cParamExeBench & aParam)
{
    TplBenchIsometrie<tREAL4 >(aParam);
    TplBenchIsometrie<tREAL8 >(aParam);
    TplBenchIsometrie<tREAL16>(aParam);
}

/* ========================== */
/*          BenchGlobImage    */
/* ========================== */


void BenchGeom(cParamExeBench & aParam)
{
    if (! aParam.NewBench("Geom")) return;

    BenchIsometrie(aParam);
    BenchRotation3D(aParam);

    aParam.EndBench();
}




};
