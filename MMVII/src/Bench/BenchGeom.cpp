#include "MMVII_Geom2D.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_PCSens.h"

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
       cPtxd<Type,3> aP1  =  cPtxd<Type,3>::PRandUnitNonAligned(aP0,1e-2);
       cRotation3D<Type> aRP01 = cRotation3D<Type>::CompleteRON(aP0,aP1);
       {
          // Type anAcc =  tElemNumTrait<Type>::Accuracy();
          Type aU= aRP01.Mat().Unitarity();
          // StdOut() << "UUUUU " << aU  << " " << Scal(aP0,aP1) << std::endl;
          MMVII_INTERNAL_ASSERT_bench(aU<1e-5,"Complete RON 1 Vect"); // Its a rot
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
   for (int aKTest=0 ; aKTest<20 ; aKTest++)
   {
       {
           cPtxd<Type,3> aW = cPtxd<Type,3>::PRandC() *Type(3.0);
	   cDenseMatrix<Type> aMW = MatProdVect(aW);
           cPtxd<Type,3> aP = cPtxd<Type,3>::PRandC() *Type(10.0);
	   cPtxd<Type,3> aQ1 = aW ^aP;
	   cPtxd<Type,3> aQ2 =  aMW * aP;
	   Type aD = Norm2(aQ1-aQ2);

           MMVII_INTERNAL_ASSERT_bench(aD<1e-5,"Mat ProdVect"); // Inversion this way
           cRotation3D<Type>  aR = cRotation3D<Type>::RotFromAxiator(aW);

	   // will comput exp(MW) by  (1+MW/N) ^N  for N big
           int aNbPow2 = 15;
	   // init by MatExp = 1+MW/2^N
	   cDenseMatrix<Type>  aMatExp = cDenseMatrix<Type>::Identity(3) + aMW *Type(1.0/(1<<aNbPow2));
	   for (int aK=0 ; aK<aNbPow2 ; aK++)  // quick pow 2^N by iterative square
               aMatExp = aMatExp * aMatExp;

	   aD =  aR.Mat().DIm().L2Dist(aMatExp.DIm());
	   // Low accuracy required becaus if NPow2 too big numerical error, too small formula wrong ...
           MMVII_INTERNAL_ASSERT_bench(aD<1e-2,"Mat ProdVect"); 
       }
   }

   double aSomD=0;
   for (int aKTest=0 ; aKTest<20 ; aKTest++)
   {
	// generate WPK, with caution to have cos phi not to close to 0
       auto v1 = RandUnif_C()*20;
       auto v2 = RandUnif_C()*1.5;
       auto v3 = RandUnif_C()*20;
        cPtxd<Type,3>  aWPK(v1,v2,v3);

	// now force to PI/2 and -PI/2 sometime
	if (aKTest%3!=1)
	{
            aWPK.y() = (M_PI/2.0) * (aKTest%3 -1) + RandUnif_C()*1e-4;
	}

	cRotation3D<Type>  aR0 = cRotation3D<Type>::RotFromWPK(aWPK);
	aWPK = aR0.ToWPK();
	cRotation3D<Type>  aR1 = cRotation3D<Type>::RotFromWPK(aWPK);

	Type aD = aR0.Mat().DIm().L2Dist(aR1.Mat().DIm());
	aSomD += aD;
        MMVII_INTERNAL_ASSERT_bench(aD<1e-3,"Omega Phi Kapa"); 


	aR0 = cRotation3D<Type>::RotFromYPR(aWPK);
	aWPK = aR0.ToYPR();
	aR1 = cRotation3D<Type>::RotFromYPR(aWPK);
	aD = aR0.Mat().DIm().L2Dist(aR1.Mat().DIm());
        MMVII_INTERNAL_ASSERT_bench(aD<1e-2,"Omega Phi Kapa"); 
	// StdOut() << "DDDD " << aD << std::endl;
   }
   // StdOut() << "============================" << std::endl;
}

void BenchRotation3DReal8()
{
    tREAL8 aEps = 0.2;
    for (int aKT=0 ; aKT<100 ; aKT++)
    {
         cRotation3D<tREAL8>  aR0 = cRotation3D<tREAL8>::RandomRot();
         cRotation3D<tREAL8>  aRTarget = aR0*cRotation3D<tREAL8>::RandomRot(aEps);

         cPoseWithUK  aPUK(tPoseR(cPt3dr(0,0,0),aR0));

         for (int aKIter=0 ; aKIter<5; aKIter++)
         {
             cPt3dr  W = aPUK.ValAxiatorFixRot(aRTarget);
             aPUK.Omega() = W;
             aPUK.OnUpdate();
         }
         tREAL8 aD =  aRTarget.Mat().L2Dist(aPUK.Pose().Rot().Mat()) ;
	 MMVII_INTERNAL_ASSERT_bench(aD <1e-10,"FromTriInAndOut p1 !!");
    }
}

void BenchRotation3D(cParamExeBench & aParam)
{
    BenchRotation3DReal8();
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
    // StdOut() << "=======================" << std::endl;
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

	 //  StdOut() << aSim.Value(aTri.Pt(aK)) <<   aSim.Value(aTri.Pt((aK+1)%3)) << std::endl; 

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

template <class tMap,class TypeEl> void TplBenchMap2D(const tMap & aMap,const tMap & aMap2,TypeEl *)
{
	auto aP1 = tMap::tPt::PRandC();
	auto aP2 = aMap.Value(aP1);
	auto aQ1 = aMap.Inverse(aP2);

	TypeEl aD = Norm2(aP1-aQ1) /tElemNumTrait<TypeEl>::Accuracy();


	MMVII_INTERNAL_ASSERT_bench(aD<1e-2,"MapInv");

	tMap aMapI =  aMap.MapInverse();
	aQ1 = aMapI.Value(aP2);
	aD = Norm2(aP1-aQ1) /tElemNumTrait<TypeEl>::Accuracy();
	if (aD>=1e-2)
	{
		// StdOut() << "DDDelta " <<  aMap.Delta() << " accc : " << tElemNumTrait<TypeEl>::Accuracy() << std::endl;
		// StdOut() << "Tr " <<  aMap.Tr() << " Vx " << aMap.VX() << " VY " << aMap.VY() << std::endl;
		StdOut() << "DDD " <<  aD << std::endl;
	    MMVII_INTERNAL_ASSERT_bench(aD<1e-2,"MapInv");
	}

	auto aP3 = aMap2.Value(aP2);
	tMap aMap12 = aMap2 * aMap;
	auto aQ3 = aMap12.Value(aP1);
	aD = Norm2(aP3-aQ3) /tElemNumTrait<TypeEl>::Accuracy();
	MMVII_INTERNAL_ASSERT_bench(aD<1e-2,"MapInv");

        tMap aIdent;
        auto aR1 = aIdent.Value(aP1);
	aD = Norm2(aP1-aR1) /tElemNumTrait<TypeEl>::Accuracy();
	MMVII_INTERNAL_ASSERT_bench(aD<1e-2,"MapIdent");
}




template <class tMap,class TypeEl> void TplBenchMap2D_LSQ(TypeEl *)
{
     bool IsHomogr =  (tMap::Name() == "Homogr2D");

     int aNbPts = (tMap::NbDOF+1)/2;
     std::vector<cPtxd<TypeEl,2> > aVIn =  RandomPtsOnCircle<TypeEl>(aNbPts);
     std::vector<cPtxd<TypeEl,2> > aVOut;
 
     // Generate some random point on the circle, not degenerated =>  Put a top level in .h
     for (int aK=0 ; aK<aNbPts ; aK++)
     {
          aVOut.push_back(cPtxd<TypeEl,2>::PRand());
     }

     if (IsHomogr)
     {
       auto aPair = RandomPtsHomgr<TypeEl>();
       aVIn  = aPair.first;
       aVOut = aPair.second;
     }

     tMap aMap =  tMap::StdGlobEstimate(aVIn,aVOut);

     if (tMap::NbDOF%2) // in this case match cannot be perfect "naturally", not enoug DOF, must cheat
     {
         aVOut.clear();
         for (int aK=0 ; aK<aNbPts ; aK++)
         {
              aVOut.push_back(aMap.Value(aVIn[aK]));
         }
         aMap =  tMap::StdGlobEstimate(aVIn,aVOut);
     }
     typename tMap::tTabMin aTabIn;
     typename tMap::tTabMin aTabOut;

     for (int aK=0 ; aK<int(aVIn.size()); aK++)
     {
          TypeEl anEr = Norm2(aVOut[aK] - aMap.Value(aVIn[aK]));
	  anEr /= tElemNumTrait<TypeEl>::Accuracy();
          // Very leniant with homography ....
         
          TypeEl aDiv=std::min(TypeEl(1.0),Square(aMap.Divisor(aVIn[aK])));
          if ((aDiv>1e-10) && (anEr*aDiv>=1e-2))
          {
               StdOut()  << "Diivv " << aMap.Divisor(aVIn[aK])  << " DD=" << aDiv  << " E=" << anEr << std::endl;
	       MMVII_INTERNAL_ASSERT_bench(false,"Least Sq Estimat 4 Mapping");
          }
          aTabIn[aK] = aVIn[aK];
          aTabOut[aK] = aVOut[aK];
    }


    // Test  estimation from a minimum of samples
    aMap =  tMap::FromMinimalSamples(aTabIn,aTabOut);

    for (int aK=0 ; aK<int(aVIn.size()); aK++)
    {
         TypeEl anEr = Norm2(aVOut[aK] - aMap.Value(aVIn[aK]));
         anEr /= tElemNumTrait<TypeEl>::Accuracy();
         MMVII_INTERNAL_ASSERT_bench(anEr<1e-2,"Least Sq Estimat 4 Mapping");
    }


    // Test ransac
     {
         // Generate a set with perfect match and a subset of noisy match
         // perfact match are created with previous map
         aVIn.clear();
         aVOut.clear();
         int aNbPts = 50;  
         int aNbBad = 20;
         if (IsHomogr) aNbBad = 5;
         cRandKAmongN aSelBad(aNbBad,aNbPts);
         for (int aK=0 ; aK<aNbPts ; aK++)
         {
             cPtxd<TypeEl,2> aPIn = cPtxd<TypeEl,2>::PRandC();
             cPtxd<TypeEl,2> aPOut =  aMap.Value(aPIn);
             if (aSelBad.GetNext())
                aPIn = aPIn + cPtxd<TypeEl,2>::PRandC()*TypeEl(0.1);
             aVIn.push_back (aPIn);
             aVOut.push_back(aPOut);
         }
         // Estimate match by ransac
         tMap aMapRS = aMap.RansacL1Estimate(aVIn,aVOut,200);

         //  Map should be equal to inital value, test this by action on points
         for (int aK=0 ; aK<aNbPts ; aK++)
         {
             TypeEl anEr =  Norm2(aMap.Value(aVIn[aK])-aMapRS.Value(aVIn[aK])) ;
             anEr /= tElemNumTrait<TypeEl>::Accuracy();

             TypeEl aDiv = std::min(Square( aMapRS.Divisor(aVIn[aK])),Square(aMap.Divisor(aVIn[aK]) ));
             aDiv = std::min(TypeEl(1.0),aDiv);

             if ((aDiv>1e-10) && ((anEr*aDiv)>=1e-3))
             {
                  TypeEl aEps = 1e-3;
                  cPtxd<TypeEl,2> aDx(aEps,0);
                  StdOut() <<  "erRRR = " << anEr << std::endl;

             }
         }
      }
}


template <class tMap,class TypeEl> void TplBenchMap2D_NonLinear(const tMap & aMap0,const tMap &aPerturb,TypeEl *)
{
    // Generate point with noises ,
    // 2/3 are perfect correspondance
    // 1/3 are noise with 0.1 ampl
    int aNbPts = 50;
    std::vector<cPtxd<TypeEl,2> > aVIn ;      // cloud point generated in [0,1] ^2
    std::vector<cPtxd<TypeEl,2> > aVOutNoise; // their noisy corresp
    std::vector<cPtxd<TypeEl,2> > aVOutRef;   // their perfect corresp

    // generate the points
    for (int aK=0 ; aK<aNbPts ; aK++)
    {
         cPtxd<TypeEl,2>  aPIn = cPtxd<TypeEl,2>::PRandC();
         cPtxd<TypeEl,2>  aPOut = aMap0.Value(aPIn);
         cPtxd<TypeEl,2>  aPNoise = aPOut +  cPtxd<TypeEl,2>::PRandC() * TypeEl(0.1);

         if (aK%3==0)
            aPNoise =  cPtxd<TypeEl,2>::PRandC() * TypeEl(2);

         aVIn.push_back(aPIn);
         aVOutRef.push_back(aPOut);
         aVOutNoise.push_back(aPNoise);
    }

    // ransac estimation, perturbate it (else we woul get good answer initially)
    tMap aMap = tMap::RansacL1Estimate(aVIn, aVOutNoise,300) *  aPerturb;

    TypeEl aRes (0);
    TypeEl aResMin=10;
    for (int aKIter=0 ; aKIter<10 ; aKIter++)
    {
        aMap = aMap.LeastSquareRefine(aVIn,aVOutRef,&aRes);
        // StdOut() << "      RESIDUAL=" << aRes << std::endl;
        aResMin= std::min(aRes,aResMin);
    }
    aRes /= tElemNumTrait<TypeEl>::Accuracy();

    //StdOut() << "RESIDUAL=" << aRes << " " << aResMin << std::endl;

    MMVII_INTERNAL_ASSERT_bench(aResMin<1e-5,"Ransac  Estimat 4 Mapping");
    // Dont understand why sometimes it grows back after initial decrease, to see later ...
    MMVII_INTERNAL_ASSERT_Unresolved(aRes<1e-3,"Ransac  Estimat 4 Mapping");
    // StdOut() << "Hhhhhhhhhhhhh " << std::endl; getchar();
}

template <class Type> void TplElBenchMap2D()
{
   auto v1 = cRot2D<Type>::RandomRot(5);
   auto v2 = cPtxd<Type,2>::PRandC()*Type(0.3);
   auto v3 = Type(RandUnif_C()*0.2);
   TplBenchMap2D_NonLinear
   (
         v1,
         cRot2D<Type>(v2, v3),
         (Type*)nullptr
   );

   TplBenchMap2D_LSQ<cHomogr2D<Type>>((Type*)nullptr);
   TplBenchMap2D_LSQ<cRot2D<Type>>((Type*)nullptr);
   TplBenchMap2D_LSQ<cAffin2D<Type>>((Type*)nullptr);
   TplBenchMap2D_LSQ<cSim2D<Type>>((Type*)nullptr);
   TplBenchMap2D_LSQ<cHomot2D<Type>>((Type*)nullptr);


   auto v4 = cAffin2D<Type>::AllocRandom(1e-1);
   auto v5 = cAffin2D<Type>::AllocRandom(1e-1);
   TplBenchMap2D(v4,v5,(Type*)nullptr);
   auto v6 = cSim2D<Type>::RandomSimInv(5,2,1e-1);
   auto v7 = cSim2D<Type>::RandomSimInv(3,4,1e-1);
   TplBenchMap2D(v6,v7,(Type*)nullptr);
   auto v8 = cHomot2D<Type>::RandomHomotInv(5,2,1e-1);
   auto v9 = cHomot2D<Type>::RandomHomotInv(3,4,1e-1);
   TplBenchMap2D(v8,v9,(Type*)nullptr);
   auto v10 = cRot2D<Type>::RandomRot(5);
   auto v11 = cRot2D<Type>::RandomRot(3);
   TplBenchMap2D(v10,v11,(Type*)nullptr);

   cHomogr2D<Type> aHgrId =  RandomMapId<cHomogr2D<Type>>(0.1);
   cHomogr2D<Type> aHgGlob =  cHomogr2D<Type>::AllocRandom(2.0);
   TplBenchMap2D(aHgGlob,aHgrId,(Type*)nullptr);

/*
*/

}

void  BenchMap2D()
{
   for(int aK=0 ;aK<100; aK++)
   {
       TplElBenchMap2D<tREAL8>();
       TplElBenchMap2D<tREAL16>();
       TplElBenchMap2D<tREAL4>();
   }
}

/* ========================== */
/*          BenchGlobImage    */
/* ========================== */
void BenchPlane3D();
void BenchHomogr2D();


void BenchGeom(cParamExeBench & aParam)
{
    if (! aParam.NewBench("Geom")) return;

    BenchSampleQuat();

    BenchHomogr2D();

    cEllipse::BenchEllispe();

    BenchIsometrie(aParam);
    BenchRotation3D(aParam);
    BenchMap2D();
    BenchPlane3D();

    aParam.EndBench();
}




};
