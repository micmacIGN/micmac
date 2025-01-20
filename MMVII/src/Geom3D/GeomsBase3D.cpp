#include "MMVII_SysSurR.h"
#include "MMVII_Geom3D.h"

namespace MMVII
{

/*  *********************************************************** */
/*                                                              */
/*                  tSeg3dr                                     */
/*                                                              */
/*  *********************************************************** */

/*   Seg = Mil,V     N = V1 ^ V2
 *
 *   Mil2 =  Mil1 + A V1 + B N + C V2
 *
 *                (       ) (A)
 *   Mil2-Mil1 =  (V1 V2 N) (B)
 *                (       ) (C)
 *
 *   I =  Mil1 + aV1 +  B/2 N
 *
 */

cPt3dr  BundleInters(cPt3dr & aABC,const tSeg3dr & aSeg1,const tSeg3dr & aSeg2,tREAL8 aW12)
{
   cPt3dr  aV1   = aSeg1.V12();
   cPt3dr  aV2   = aSeg2.V12();
   cPt3dr  aNorm = aV1 ^ aV2;

   cDenseMatrix<tREAL8> aMat =  M3x3FromCol(aV1,-aV2,aNorm);
   aABC = SolveCol(aMat,aSeg2.P1()-aSeg1.P1());

   return aSeg1.P1() + aV1 * aABC.x() + aNorm * (aABC.z() * (1.0-aW12));
}

cPt3dr  BundleInters(const tSeg3dr & aSeg1,const tSeg3dr & aSeg2,tREAL8 aW12)
{
   cPt3dr  ABC ;
   return BundleInters(ABC,aSeg1,aSeg2,aW12);
}

/*
 *   "ONR"= Orthognal Normalised Repair 
 *
 *    Algo 4 Bundle inter, for each seg Sk,  let 
 *
 *       *  uk be a normal vector, and Pk a point on SK
 *
 *    I we complete the base with vk,wk such that (uk,vk,wk) is a ONR, we can write the least square equations "Q in Sk " as :
 *
 *       (Pk-Q) . vk=0   ; (Pk-Q).wk = 0
 *    
 *    So the square resiual R2 is
 *
 *    R2  =   Sum  ( (Pk-Q) . vk ^2 + (Pk-Q).wk )
 *        =   Sum  ( (Pk-Q) (tvk vk + twk wk) (Pk-Q) )
 *
 *    But as its a ONR  we have  tukuk +  tvk vk + twk wk = Id  , let call :
 *
 *      Ok = Id- tukuk = tvk vk + twk wk 
 *
 *   R2 = Sum(  tQ Ok Q  -2 Ok Pk + Cte)
 *
 *
 *
 */



cPt3dr  BundleInters(const std::vector<tSeg3dr> & aVSeg,const std::vector<tREAL8> * aVWeight)
{
     MMVII_INTERNAL_ASSERT_tiny(aVSeg.size()>=2,"Not enough seg in BundleInters");
     cDenseMatrix<tREAL8>  aDM(3,eModeInitImage::eMIA_Null);
     cPt3dr aRHS(0,0,0);

     int aNbWNN = 0;

     for (size_t aKSeg=0 ; aKSeg<aVSeg.size() ; aKSeg++)
     {
         tREAL8 aW = aVWeight ?  aVWeight->at(aKSeg)  : 1.0 ;

	 const cPt3dr& aP1  =  aVSeg[aKSeg].P1();
	 tREAL8 x1 = aP1.x();
	 tREAL8 y1 = aP1.y();
	 tREAL8 z1 = aP1.z();
	 const cPt3dr& aP2  =  aVSeg[aKSeg].P2();
	 cPt3dr aUk = VUnit(aP2-aP1);

	 tREAL8 aXu = aUk.x();
	 tREAL8 aYu = aUk.y();
	 tREAL8 aZu = aUk.z();

	 tREAL8 a00 = 1 - Square(aXu);
	 tREAL8 a10 = - aXu*aYu;
	 tREAL8 a20 = - aXu*aZu;
	 tREAL8 a11 = 1 - Square(aYu);
	 tREAL8 a12 = - aYu*aZu;
	 tREAL8 a22 = 1 - Square(aZu);

	 aDM.AddElem(0,0,aW*a00);
	 aDM.AddElem(1,1,aW*a11);
	 aDM.AddElem(2,2,aW*a22);

	 aDM.AddElem(1,0,aW*a10);
	 aDM.AddElem(2,0,aW*a20);
	 aDM.AddElem(2,1,aW*a12);

	 /*
	 aDM.AddElem(0,1,aW*a10);
	 aDM.AddElem(0,2,aW*a20);
	 aDM.AddElem(1,2,aW*a12);
	 */

	 aRHS += cPt3dr
		 (
		      aW*(a00 * x1 + a10*y1 + a20 * z1),
		      aW*(a10 * x1 + a11*y1 + a12 * z1),
		      aW*(a20 * x1 + a12*y1 + a22 * z1)
		 );

         if (aW>0) aNbWNN++;
     }
     MMVII_INTERNAL_ASSERT_bench(aNbWNN>=2,"Not enough segs in BundleInters");

     aDM.SelfSymetrizeBottom();
     return SolveCol(aDM,aRHS);
}

cPt3dr  L1_BundleInters(const std::vector<tSeg3dr> & aVSeg,int NbSegCompl,const std::vector<tREAL8> * aVWeight)
{
    return cPt3dr::Dummy(); // TODO
}




cPt3dr  RobustBundleInters(const std::vector<tSeg3dr> & aVSeg)
{
     if (aVSeg.size() == 2)
        return BundleInters(aVSeg[0],aVSeg[1],0.5);

     std::vector<cSegmentCompiled<tREAL8,3>> aVSC;

     for (const auto & aSeg : aVSeg)
        aVSC.push_back(cSegmentCompiled<tREAL8,3>(aSeg.P1(),aSeg.P2()));

     cWhichMin<cPt3dr,tREAL8>  aWMin;


     for (size_t aKSeg1=0 ; aKSeg1<aVSeg.size() ; aKSeg1++)
     {
         for (size_t aKSeg2=aKSeg1+1 ; aKSeg2<aVSeg.size() ; aKSeg2++)
         {
             cPt3dr aInt = BundleInters(aVSeg[aKSeg1],aVSeg[aKSeg2],0.5);

             tREAL8 aSum =0;
             for (size_t  aKSeg3=0 ;  aKSeg3<aVSeg.size() ; aKSeg3++)
             {
                 if ((aKSeg3!=aKSeg1) && (aKSeg3!=aKSeg2))
                 {
                    aSum += aVSC[aKSeg3].Dist(aInt);
                 }
             }
             aWMin.Add(aInt,aSum);
         }
     }

     return aWMin.IndexExtre();
}

cPt3dr  BundleFixZ(const tSeg3dr & aSeg,const tREAL8 & aZ)
{
    const cPt3dr & aP1 = aSeg.P1();
    cPt3dr aV12 = aSeg.V12();
    return  aP1 +  aV12 *  ((aZ - aP1.z()) /aV12.z());
}


tPoseR RobustIsometry(const std::vector<cPt3dr> & aPtsA, const std::vector<cPt3dr> & aPtsB)
{
    MMVII_INTERNAL_ASSERT_strong(aPtsA.size()>2, "Isometry initialization needs at least 3 points")
    MMVII_INTERNAL_ASSERT_strong(aPtsA.size()==aPtsB.size(), "Isometry initialization needs two sets with same number of points")
    if (aPtsA.size() == 3)
        return tPoseR::FromTriInAndOut(0, {aPtsA[0], aPtsA[1], aPtsA[2]}, 0, {aPtsB[0], aPtsB[1], aPtsB[2]});

    // find the best 3 points
    cWhichMin<tPoseR,tREAL8>  aIsoWMin(tPoseR(), INFINITY);
    for (unsigned int i = 0; i < aPtsA.size()-2; ++i)
        for (unsigned int j = i+1; j < aPtsA.size()-1; ++j)
            for (unsigned int k = j+1; k < aPtsA.size(); ++k)
            {
                tTri3dr aTriA = cTriangle(aPtsA[i], aPtsA[j], aPtsA[k]);
                tTri3dr aTriB = cTriangle(aPtsB[i], aPtsB[j], aPtsB[k]);
                auto anIso = tPoseR::FromTriInAndOut(0, aTriA, 0, aTriB);
                tREAL8 score = 0.;
                for (unsigned int l = 0; l < aPtsA.size(); ++l)
                {
                    score += SqN2(anIso.Value(aPtsA[l])-aPtsB[l]);
                }
                aIsoWMin.Add(anIso, score);
            }
    return aIsoWMin.IndexExtre();
}

/*  *********************************************************** */
/*                                                              */
/*                  cPlan3D                                     */
/*                                                              */
/*  *********************************************************** */


cPlane3D::cPlane3D(const cPt3dr & aP0,const cPt3dr& aAxeI , const cPt3dr& aAxeJ) :
     mP0(aP0)
{
    cRotation3D<tREAL8> aRot = cRotation3D<tREAL8>::CompleteRON(aAxeI,aAxeJ);

    mAxeI = aRot.AxeI();
    mAxeJ = aRot.AxeJ();
    mAxeK = aRot.AxeK();
}

cPlane3D cPlane3D::FromP0And2V(const cPt3dr & aP0,const cPt3dr& aAxeI , const cPt3dr& aAxeJ) 
{
    return cPlane3D(aP0,aAxeI,aAxeJ);
}

cPlane3D cPlane3D::From3Point(const cPt3dr & aP0,const cPt3dr & aP1,const cPt3dr & aP2)
{
	return cPlane3D(aP0,aP1-aP0,aP2-aP0);
}

tREAL8  cPlane3D::Dist(const cPt3dr & aPt) const
{
    return std::abs(Scal(mAxeK,aPt-mP0));
}

tREAL8 cPlane3D::AvgDist(const std::vector<cPt3dr> & aVPts) const
{
      tREAL8 aSom=0.0;

      for (const auto & aPt : aVPts)
          aSom += Dist(aPt);

      return SafeDiv(aSom,tREAL8(aVPts.size()));
}

tREAL8 cPlane3D::MaxDist(const std::vector<cPt3dr> & aVPts) const
{
     tREAL8 aMaxD = 0.0;

     for (const auto & aPt : aVPts)
          UpdateMax(aMaxD,Dist(aPt));

     return aMaxD;
}

static const cPt3di NoTriplet(-1,-1,-1);

std::pair<cPt3di,tREAL8> cPlane3D::IndexRansacEstimate(const std::vector<cPt3dr> & aVPts,bool AvgOrMax,int aNbTest,tREAL8 aRegulMinTri)
{
     cWhichMin<cPt3di,tREAL8> aWM(NoTriplet,1e30);

     std::vector<cSetIExtension>  aSet3I; // Set of triple of indexes

     if (aNbTest<0) 
        aNbTest = 1000000000;

     GenRanQsubCardKAmongN(aSet3I,aNbTest,3,aVPts.size());

     for (const auto & a3I : aSet3I)
     {
         cPt3di anInd(a3I.mElems.at(0),a3I.mElems.at(1),a3I.mElems.at(2));
         cPt3dr aP0 = aVPts.at(anInd.x());
         cPt3dr aP1 = aVPts.at(anInd.y());
         cPt3dr aP2 = aVPts.at(anInd.z());

	 tTri3dr aTri(aP0,aP1,aP2);
	 if (aTri.Regularity() > aRegulMinTri)
	 {
            cPlane3D aPlane = cPlane3D::From3Point(aP0,aP1,aP2);

	    tREAL8 aD = AvgOrMax ? aPlane.AvgDist(aVPts) : aPlane.MaxDist(aVPts);
	    aWM.Add(anInd,aD);
	 }
     }

     return std::pair(aWM.IndexExtre(),aWM.ValExtre());
}

std::pair<cPlane3D,tREAL8> cPlane3D::RansacEstimate(const std::vector<cPt3dr> & aVPts,bool AvgOrMax,int aNbTest,tREAL8 aRegulMinTri)
{
   auto [anInd,aCost]  = IndexRansacEstimate(aVPts,AvgOrMax,aNbTest,aRegulMinTri);

   MMVII_INTERNAL_ASSERT_tiny(anInd!=NoTriplet,"No Triplet found in cPlane3D::RansacEstimate");

   return std::pair<cPlane3D,tREAL8>(cPlane3D::From3Point(aVPts.at(anInd.x()),aVPts.at(anInd.y()),aVPts.at(anInd.z())),aCost);

}



// void GenRanQsubCardKAmongN(std::vector<cSetIExtension> & aRes,int aQ,int aK,int aN)

cPlane3D cPlane3D::FromPtAndNormal(const cPt3dr & aP0,const cPt3dr& aNormal)
{
   cRotation3D<tREAL8> aRep = cRotation3D<tREAL8>::CompleteRON(aNormal);

   //  Rep =  I(=Normal) , J, K
   return cPlane3D(aP0,aRep.AxeJ(),aRep.AxeK());

   //return cPlane3D(aP0,aRep.AxeI(),aRep.AxeJ());
}

const cPt3dr& cPlane3D::AxeI() const {return mAxeI;}
const cPt3dr& cPlane3D::AxeJ() const {return mAxeJ;}
const cPt3dr& cPlane3D::AxeK() const {return mAxeK;}
const cPt3dr& cPlane3D::P0()   const {return mP0;  }

cPt3dr  cPlane3D::ToLocCoord(const cPt3dr & aPGlob) const
{
     cPt3dr aVect = aPGlob-mP0;
     return cPt3dr (Scal(mAxeI,aVect), Scal(mAxeJ,aVect), Scal(mAxeK,aVect));
}

cPt3dr  cPlane3D::FromCoordLoc(const cPt3dr & aP) const
{
    return mP0 + mAxeI*aP.x() + mAxeJ*aP.y() + mAxeK*aP.z();
}

cPt3dr  cPlane3D::Inter(const cPt3dr&aP0,const cPt3dr&aP1) const
{
     cPt3dr aVect = aP1-aP0;
     tREAL8 aS1  = Scal(mAxeK,aP1-mP0);
     tREAL8 aS01 = Scal(mAxeK,aVect);
    
     //  Scal(mAxeK,aP1+t*aVect -mP0) = 0
     //  t = - Scal(aP1-mP0,aK)  / Scal (aVect,aK)

     return  aP1 -  aVect*(aS1/aS01);
}
cPt3dr  cPlane3D::Inter(const tSeg3dr&aSeg) const {return Inter(aSeg.P1(),aSeg.P2());}


std::vector<cPt3dr>  cPlane3D::RandParam()
{
    cPt3dr aP0 = cPt3dr::PRandC() * 100.0;

    cPt3dr  aI =  cPt3dr::PRandUnit() ;
    cPt3dr  aJ =  cPt3dr::PRandUnitDiff(aI) ;

    auto v1 = aI*RandInInterval(0.1,2.0);
    auto v2 = aJ*RandInInterval(0.1,2.0);
    return std::vector<cPt3dr>{aP0,v1,v2};
}

/**  Compute the direction of intersection, algorithm :
 *
 *      - compute the quadratic form  E(Pt) =  Sum(D^2(Plk,Pt))
 *      - search the minimum on unity sphere
 *      - the minimum is reached for the eigen vector corresponding to the lowest eigen-value
 */


cPt3dr cPlane3D::DirInterPlane(const std::vector<const cPlane3D*>& aVPlanes,int aSzMin)
{
    MMVII_INTERNAL_ASSERT_tiny((int)aVPlanes.size()>=aSzMin,"DirInterPlane not enough planes");

    // compute the quadratic form
    cStrStat2<tREAL8>  aCov(3);
    for (const auto & aPlanePtr : aVPlanes)
    {
        aCov.Add(aPlanePtr->AxeK().ToVect());
    }

    // extract the diagonalisation
    const cResulSymEigenValue<tREAL8> & aResE = aCov.DoEigen();

    // extract the eigen-vector corresponding to lowest eigen value (they are in growing order)
    cPt3dr aRes;
    GetCol(aRes,aResE.EigenVectors(),0);

    return aRes;
}

cPt3dr cPlane3D::DirInterPlane(const std::vector<cPlane3D>& aVPlanes,int aSzMin)
{
    return DirInterPlane(VecObj2VecPtr(aVPlanes),aSzMin);
}
/**  Compute the line of intersection of N Plane , Method :
 *
 *    - 1 compute the direction "DIR"
 *    - 2 compute "the" point
 *
 *    The problem arrise when the direction of intersection is almost perfect, in this
 *    case the position is undetermined.  This is obviously the case when N=2.  This here
 *    where the stabilizer is used, it fix with some weight, the position on  "DIR" to be
 *    closed to average of P0() of all plane.
 */

tSeg3dr  cPlane3D::InterPlane(const std::vector<const cPlane3D*>& aVPlanes,int aSzMin,tREAL8 aWeightStabRel)
{
   cPt3dr aDir = DirInterPlane(aVPlanes,aSzMin);
   int aNbPl = aVPlanes.size();

   if (aNbPl<=1)
   {
       cPt3dr aP0 = aVPlanes.empty() ? cPt3dr::PRandC() : aVPlanes.at(0)->P0();
       return tSeg3dr(aP0,aP0+aDir);
   }

   cLeasSqtAA<tREAL8> aSys(3);

   cPt3dr aAvgP0(0,0,0);
   tREAL8 aSomDPl = 0;
   for (const auto & aPlanePtr : aVPlanes)
   {
       const cPt3dr & aK = aPlanePtr->AxeK();
       aSys.PublicAddObservation(1.0,aK.ToVect(),Scal(aPlanePtr->P0(),aK));
       aAvgP0 += aPlanePtr->P0();
       aSomDPl +=  std::abs(Scal(aK,aDir));
   }

   if (aWeightStabRel != 0)
   {
       aAvgP0 = aAvgP0 / tREAL8(aNbPl);
       aSomDPl /=  aNbPl;

       tREAL8 aWeightStab = aWeightStabRel / (aSomDPl+aWeightStabRel);
       aSys.PublicAddObservation(aWeightStab,aDir.ToVect(),Scal(aDir,aAvgP0));
   }

   cPt3dr aP0 = cPt3dr::FromVect(aSys.PublicSolve());
   return tSeg3dr(aP0,aP0+aDir);
}

tSeg3dr  cPlane3D::InterPlane(const std::vector<cPlane3D>& aVPlanes,int aSzMin,tREAL8 aWeightStabRel)
{
    return InterPlane(VecObj2VecPtr(aVPlanes),aSzMin,aWeightStabRel);
}


void BenchPlaneInter()
{
    for (int aKPlane=0 ; aKPlane<100 ; aKPlane++)
    {
        cPt3dr aAxeSym  =  cPt3dr::PRandUnit();
        tRotR aR = tRotR::CompleteRON(aAxeSym );
        int aNbTeta = 0 + (aKPlane%7);
        tREAL8 aTeta0 = RandUnif_C() *10.0;
        std::vector<cPlane3D> aVPlanes;

        cPt3dr aP0 = cPt3dr::PRandC();

        tREAL8 aEps = 0.1 * RandUnif_C_NotNull(0.1);
       // force a perfect intersec to test stabilization
        if ( (aNbTeta>2) && (aKPlane%3==0))
            aEps = 0;

        for (int aKTeta=0 ; aKTeta<aNbTeta ; aKTeta++)
        {
            tREAL8 aTeta = aTeta0 + (2*M_PI*aKTeta) / aNbTeta;
            cPt3dr aPNorm = aR.Value(cPt3dr(aEps,std::cos(aTeta),std::sin(aTeta)));

            aVPlanes.push_back(cPlane3D::FromPtAndNormal(aP0,aPNorm));
        }
        cPt3dr  aDirInter = cPlane3D::DirInterPlane(aVPlanes,0);
        tSeg3dr aLineInter = cPlane3D::InterPlane(aVPlanes,0);

        // Juste check that dir are identic
        tREAL8 aDifDir = Norm2(aDirInter- aLineInter.V12());
        MMVII_INTERNAL_ASSERT_bench(aDifDir<1e-10,"Diff of direction in interplane");

        if (Cos(aDirInter,aAxeSym)<0)
           aAxeSym = - aAxeSym;

        for (const auto & aPl : aVPlanes)
        {
               tREAL8 aVCos = std::abs(Cos(aDirInter,aPl.AxeK()));
               tREAL8 aZ1 = std::abs(aPl.ToLocCoord(aLineInter.P1()).z());
               tREAL8 aZ2 = std::abs(aPl.ToLocCoord(aLineInter.P2()).z());
               if (aNbTeta<=2)
               {
                  // if NbPlane is 2, then the intersection is perfect => Test that the intersection is orthog to normal
                  MMVII_INTERNAL_ASSERT_bench(aVCos<1e-5,"Dir Inter Plane, Case 2");
		  // Test point P1 and P2  belongs to all plane
                  MMVII_INTERNAL_ASSERT_bench((aZ1<1e-5) && (aZ2<1e-5),"Z-Inter Plane, Case <= 2");
               }
               else
               {
                  // just test that inter is not  perfect
                  // StdOut() << "CCCc" << aVCos << " " << aDirInter << aPl.AxeK() << "\n";

                  if (aEps>0)
                  {
                      MMVII_INTERNAL_ASSERT_bench(aVCos>std::abs(aEps/2.0),"Dir Inter Plane, Case > 2");
                  }
		  // Test point P1 belongs to all plane
                  MMVII_INTERNAL_ASSERT_bench((aZ1<1e-5) ,"Z-Inter Plane, Case 2");
               }
        }
        if (aNbTeta>2)
        {
            tREAL8 aDist =  Norm2(aDirInter - aAxeSym);
            MMVII_INTERNAL_ASSERT_bench(aDist<1e-5,"Dir Inter Plane, Case > 2");
        }

    }
}


void BenchPlane3D()
{
    BenchPlaneInter();

    for  (int aK=0 ;aK<100 ;aK++)
    {
         std::vector<cPt3dr>  aVP = cPlane3D::RandParam();
         cPlane3D aPlane = cPlane3D::FromP0And2V(aVP[0],aVP[1],aVP[2]);
	 MMVII_INTERNAL_ASSERT_bench(Norm2(aPlane.ToLocCoord(aVP[0])) < 1e-9,"BenchPlane3D");
	 MMVII_INTERNAL_ASSERT_bench(std::abs(aPlane.ToLocCoord(aVP[0]+aVP[1]).z())<1e-5,"BenchPlane3D");
	 MMVII_INTERNAL_ASSERT_bench(std::abs(aPlane.ToLocCoord(aVP[0]+aVP[2]).z())<1e-5,"BenchPlane3D");

         cPt3dr aP0 = cPt3dr::PRandC() * 100.0;
         auto v1 = RandUnif_C();
         auto v2 = RandUnif_C();
         auto v3 = RandUnif_C_NotNull(0.1);
         cPt3dr aP1 = aP0 +  aPlane.AxeI() * v1 + aPlane.AxeJ() * v2 +  aPlane.AxeK()  * v3;

	 cPt3dr aPI = aPlane.Inter(aP0,aP1);
	 cPt3dr aPI2 = aPlane.Inter(tSeg3dr(aP0,aP1));
	 MMVII_INTERNAL_ASSERT_bench(Norm2(aPI-aPI2)<1e-9,"BenchPlane3D");
	 MMVII_INTERNAL_ASSERT_bench(std::abs(aPlane.ToLocCoord(aPI).z())<1e-5,"BenchPlane3D");

	 cSegmentCompiled<tREAL8,3> aSeg(aP0,aP1);
	 MMVII_INTERNAL_ASSERT_bench(aSeg.Dist(aPI)<1e-5,"BenchPlane3D");

	 MMVII_INTERNAL_ASSERT_bench(Norm2(aP0 -aPlane.ToLocCoord(aPlane.FromCoordLoc(aP0)))<1e-5,"BenchPlane3D");
	 MMVII_INTERNAL_ASSERT_bench(Norm2(aP0 -aPlane.FromCoordLoc(aPlane.ToLocCoord(aP0)))<1e-5,"BenchPlane3D");
    }

    for  (int aK=0 ;aK<100 ;aK++)
    {
       cPt3dr  aV1 = cPt3dr::PRandUnit();
       cPt3dr  aV2 = cPt3dr::PRandUnitNonAligned(aV1);
       aV1 = aV1 * RandInInterval(0.1,2.0);
       aV2 = aV2 * RandInInterval(0.1,2.0);

       cPt3dr  aP1 = cPt3dr::PRandC() * 10.0;
       cPt3dr  aP2 = cPt3dr::PRandC() * 10.0;

       cSegmentCompiled<tREAL8,3> aSeg1(aP1,aP1+aV1);
       cSegmentCompiled<tREAL8,3> aSeg2(aP2,aP2+aV2);

       cPt3dr  aPI = BundleInters(aSeg1,aSeg2,0.5);

       cPt3dr aProj1 = aSeg1.Proj(aPI);
       cPt3dr aProj2 = aSeg2.Proj(aPI);

       MMVII_INTERNAL_ASSERT_bench(std::abs(Cos(aSeg1.V12(),aProj1-aProj2)) <1e-5,"BundleInters");
       MMVII_INTERNAL_ASSERT_bench(std::abs(Cos(aSeg2.V12(),aProj1-aProj2)) <1e-5,"BundleInters");
       MMVII_INTERNAL_ASSERT_bench(Norm2(aPI-(aProj1+aProj2)/2.0)<1e-5,"BundleInters");

       std::vector<tSeg3dr> aVSeg{aSeg1,aSeg2};
       cPt3dr aPIVec =   BundleInters(aVSeg);

       //  StdOut() << "NnnNnn " <<  Norm2(aPI - aPIVec) << std::endl;
       MMVII_INTERNAL_ASSERT_bench(Norm2(aPI - aPIVec) <1e-5,"BundleInters");
    }
    
}

/*  *********************************************************** */
/*                                                              */
/*  *********************************************************** */

tREAL8 L2_DegenerateIndex(const std::vector<cPt3dr> & aVPt,size_t aNumEigV)
{
    cStrStat2<tREAL8>  aStat(3);
    for (const auto & aP3 : aVPt)
            aStat.Add(aP3.ToVect());
    aStat.Normalise();
    const cDenseVect<tREAL8> anEV = aStat.DoEigen().EigenValues() ;
    // theoretically eigenvalues can't be negavive => treat <0 as =0
    if (anEV(2)<=0)  return 0.0;
    if (anEV(aNumEigV)<0)  return 0.0;

    return Sqrt(SafeDiv(anEV(aNumEigV),anEV(2)));
}

tREAL8 L2_PlanarityIndex(const std::vector<cPt3dr> & aVPt) 
{ 
    tREAL8 aNbP = aVPt.size() ;
    if (aNbP <=3) return 0.0;

    return L2_DegenerateIndex(aVPt,0) * (aNbP / (aNbP-3.0));
}


tREAL8 L2_LinearityIndex(const std::vector<cPt3dr> & aVPt) 
{ 
    tREAL8 aNbP = aVPt.size() ;
    if (aNbP <=2) return 0.0;

    return L2_DegenerateIndex(aVPt,1) *  (aNbP / (aNbP-2.0));
}





template<class T> cPtxd<T,3>  PFromNumAxe(int aNum)
{
   static const cDenseMatrix<T> anId3x3(3,3,eModeInitImage::eMIA_MatrixId);
   return cPtxd<T,3>::Col(anId3x3,aNum);
}

template<class T> cDenseMatrix<T> MatFromCols(const cPtxd<T,3>&aP0,const cPtxd<T,3>&aP1,const cPtxd<T,3>&aP2)
{
   cDenseMatrix<T> aRes(3,3);

   SetCol(aRes,0,aP0);
   SetCol(aRes,1,aP1);
   SetCol(aRes,2,aP2);

   return aRes;
}

template<class T> cDenseMatrix<T> MatFromLines(const cPtxd<T,3>&aP0,const cPtxd<T,3>&aP1,const cPtxd<T,3>&aP2)
{
   cDenseMatrix<T> aRes(3,3);

   SetLine(0,aRes,aP0);
   SetLine(1,aRes,aP1);
   SetLine(2,aRes,aP2);

   return aRes;
}

/*
    (X1)   (X2)      Y1*Z2 - Z1*Y2     ( 0   -Z1    Y1)   (X2) 
    (Y1) ^ (Y2) =    Z1*X2 - X1*Z2  =  ( Z1    0   -X1) * (Y2)
    (Z1)   (Z2)      X1*Y2 - Y1*X2     (-Y1    X1    0)   (Z2)
 
*/

template<class T> cDenseMatrix<T> MatProdVect(const cPtxd<T,3>& W)
{
	return MatFromLines<T>
               (
	          cPtxd<T,3>(  0    , -W.z() ,  W.y() ),
	          cPtxd<T,3>( W.z() ,   0    , -W.x() ),
	          cPtxd<T,3>(-W.y() ,  W.x() ,   0    )
	       );
}


/*
template <class T>  cPtxd<T,3> operator ^ (const cPtxd<T,3> & aP1,const cPtxd<T,3> & aP2)
{
   return cPtxd<T,3>
          (
               aP1.y() * aP2.z() -aP1.z()*aP2.y(),
               aP1.z() * aP2.x() -aP1.x()*aP2.z(),
               aP1.x() * aP2.y() -aP1.y()*aP2.x()
          );
}
*/

template<class T> cPtxd<T,3>  VOrthog(const cPtxd<T,3> & aP)
{
   // we make a vect product with any vector, just avoid one too colinear  to P
   // test I and J, as P cannot be colinear to both, its sufficient 
   // (i.e : we are sur to maintain the biggest of x, y and z)
   if (std::abs(aP.x()) > std::abs(aP.y()))
      return cPtxd<T,3>( aP.z(), 0, -aP.x());

  return cPtxd<T,3>(0,aP.z(),-aP.y());
}

template <class T>  T  TetraReg (const cPtxd<T,3> & aP1,const cPtxd<T,3> & aP2,const cPtxd<T,3> & aP3,const T& FactEps)
{
     cPtxd<T,3> aCDG = (aP1+aP2+aP3) /static_cast<T>(3.0);

     T aSqDist = (SqN2(aCDG) + SqN2(aCDG-aP1) + SqN2(aCDG-aP2) + SqN2(aCDG-aP3)) / static_cast<T>(4.0);
     T aCoeffNorm = std::pow(aSqDist,3.0/2.0);  // 1/2 for D2  3->volume

     return Determinant(aP1,aP2,aP3) / (aCoeffNorm +  std::numeric_limits<T>::epsilon()*FactEps);
}

template <class T>  T  TetraReg (const cPtxd<T,3> & aP1,const cPtxd<T,3> & aP2,const cPtxd<T,3> & aP3,const cPtxd<T,3> & aP4,const T& FactEps)
{
    return TetraReg(aP2-aP1,aP3-aP1,aP4-aP1,FactEps);
}

template <class T>  T  Determinant (const cPtxd<T,3> & aP1,const cPtxd<T,3> & aP2,const cPtxd<T,3> & aP3)
{
	return Scal(aP1,aP2^aP3);
}

template<class Type>  cTriangle<Type,3> RandomTriang(Type aAmpl)
{
      auto v1 = cPtxd<Type,3>::PRandC()*aAmpl;
      auto v2 = cPtxd<Type,3>::PRandC()*aAmpl;
      auto v3 = cPtxd<Type,3>::PRandC()*aAmpl;
      return cTriangle<Type,3>(v1,v2,v3);
}

template<class Type>  cTriangle<Type,3> RandomTriangRegul(Type aRegulMin,Type aAmpl)
{
    for (;;)
    {
        cTriangle<Type,3> aT = RandomTriang(aAmpl);
	if (aT.Regularity()> aRegulMin)
           return aT;
    }
    return RandomTriang(static_cast<Type>(0.0)); // Not sur its mandatory to have a return here
}

template<class Type>  cTriangle<Type,3> RandomTetraTriangRegul(Type aRegulMin,Type aAmpl)
{
    for (;;)
    {
        cTriangle<Type,3> aT = RandomTriang(aAmpl);
	if (TetraReg(aT.Pt(0),aT.Pt(1),aT.Pt(2)) > aRegulMin)
           return aT;
    }
    return RandomTriang(static_cast<Type>(0.0)); // Not sur its mandatory to have a return here
}




/* ========================== */
/*          ::                */
/* ========================== */

//template cPtxd<int,3>  operator ^ (const cPtxd<int,3> & aP1,const cPtxd<int,3> & aP2);
//template cPtxd<TYPE,3>  operator ^ (const cPtxd<TYPE,3> & aP1,const cPtxd<TYPE,3> & aP2);

#define MACRO_INSTATIATE_PTXD(TYPE)\
template  cTriangle<TYPE,3> RandomTriang(TYPE aRegulMin);\
template  cTriangle<TYPE,3> RandomTriangRegul(TYPE aRegulMin,TYPE aAmpl);\
template  cTriangle<TYPE,3> RandomTetraTriangRegul(TYPE aRegulMin,TYPE aAmpl);\
template TYPE  Determinant (const cPtxd<TYPE,3> & aP1,const cPtxd<TYPE,3> & aP2,const cPtxd<TYPE,3> & aP3);\
template TYPE  TetraReg (const cPtxd<TYPE,3> & aP1,const cPtxd<TYPE,3> & aP2,const cPtxd<TYPE,3> & aP3,const TYPE&);\
template TYPE  TetraReg (const cPtxd<TYPE,3> & aP1,const cPtxd<TYPE,3> & aP2,const cPtxd<TYPE,3> & aP3,const cPtxd<TYPE,3> & aP4,const TYPE&);\
template cDenseMatrix<TYPE> MatProdVect(const cPtxd<TYPE,3>& W);\
template cDenseMatrix<TYPE> MatFromCols(const cPtxd<TYPE,3>&,const cPtxd<TYPE,3>&,const cPtxd<TYPE,3>&);\
template cDenseMatrix<TYPE> MatFromLines(const cPtxd<TYPE,3>&,const cPtxd<TYPE,3>&,const cPtxd<TYPE,3>&);\
template cPtxd<TYPE,3>  PFromNumAxe(int aNum);\
template cPtxd<TYPE,3>  VOrthog(const cPtxd<TYPE,3> & aP);


MACRO_INSTATIATE_PTXD(tREAL4)
MACRO_INSTATIATE_PTXD(tREAL8)
MACRO_INSTATIATE_PTXD(tREAL16)



};
