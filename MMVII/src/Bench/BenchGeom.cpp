#include "include/MMVII_all.h"

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
         cPtxd<Type,3> aImAxe = aRAxe.Direct(aP0);
         MMVII_INTERNAL_ASSERT_bench(Norm2(aP0-aImAxe)<1e-5,"Complete RON 1 Vect"); // P0 is its axe
       }

       // Test teta
       {
           cPtxd<Type,3> aJ = aRP01.AxeJ();  // we complete the Axe with any ortog repair
           cPtxd<Type,3> aK = aRP01.AxeK();

           cPtxd<Type,3> aJA = aRAxe.Direct(aJ);
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
   }
}

void BenchRotation3D(cParamExeBench & aParam)
{
    TplBenchRotation3D<tREAL4 >(aParam);
    TplBenchRotation3D<tREAL8 >(aParam);
    TplBenchRotation3D<tREAL16>(aParam);
}

/* ========================== */
/*          BenchGlobImage    */
/* ========================== */


void BenchGeom(cParamExeBench & aParam)
{
    if (! aParam.NewBench("Geom")) return;

    BenchRotation3D(aParam);

    aParam.EndBench();
}




};
