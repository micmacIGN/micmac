#include "MMVII_PoseRel.h"
#include "MMVII_Tpl_Images.h"


/* We have the image formula w/o distorsion:


*/

namespace MMVII
{

bool BUGME = false;

/**
 *  Class for computing pose from a scene assumed to be planar
 */


/*
 *   If the scene is plane, the correspondance between homologous point is an homography.
 *  We can write :
 *
 *            a x1/z1 + b y1/z1 +c                      d x1/z1 + e y1/z1 +f
 *    x2/z2= --------------------------       y2/z2= --------------------------
 *            g x1/z1 + h y1/z1 +i                     g x1/z1 + h y1/z1 +i
 *
 *  Or :
 *     x2    (a  b  c) (x1 )           L1.p1
 *     y2 ~  (d  e  f) (y1 )    p2=    L2.p1   of  p2 = H p1 where p1 is the homography matric
 *     z2    (g  h  i) (z1 )           L3.p1
 *
 *
 *  When estimating the homography H, for adding the equation in a sytem we use  cPS_CompPose::SetPt with :
 *
 *      (ax1 + by1 + cz1) z2 - (gx1 + h y1 i z1) x2 = 0     L1.P1 z2 - L3.P1 x2 = 0
 *      (dx1 + ey1 + fz1) z2 - (gx1 + h y1 i z1) y2 = 0     L2.P1 z2 - L3.P1 y2 = 0
 *
 *   For computing epipolar repair, there is an ambiguity by a common rotation arround X-axe.  In the
 *   planar case, we define the special epipolar repair such that the Y-axe belong to the plane . In this
 *   repair, the plane has equation :
 *           P :  Z = c + k X
 *    Let :
 *      #  p1= (x1 y1 1) a point of image 1
 *      #  the bundle B= L P1 intersect plane P in point Q1 at value of parameter L = c/(1-k x1)
 *      #  let B be the base and Q2 be coordinate of Q1 in repair of image 2, we have
 *         Q2= (B + Lx1,  Ly1 , L) that is projected on image 2 at p2 = (x1+B/L,y1,1) = (ax1+b,y,1)
 *         with a =1- k/c  b=B/c
 *      # so in epipolar repair we have :  (x2)        (a 0 b) (x1)
 *                                         (y2)   =    (0 1 0) (y1)  or  p2 = Q(a,b) p1   [1]
 *                                         (1 )        (0 0 1) (z1)
 *    So let R1,R2 be the rotation of special epipolar repair, we have :
 *           R2 p2 = Q(a,b) R1 p1  , comparing with p2 = H p1
 *    The estimation of R2 and R1 is equivalent to solve equation :
 *
 *               tRE2 Q(a,b) RE1 = Lambda  H [2]
 *
 *    Where H is known and Lambda is used because all matrix, point are define up to scale (projective).
 *    Note that the problem is well posed we have 8 unknown a,b,Lambda and 3 for each rotation.
 *
 *    To solve this equation, we will use the following step :
 *        # make a SVD of H = tRh2 D Rh1 with D=[l1,l2,l3]
 *        # estimate lambda, a and b using some invariant on singular values such
 *            Q(a,b) = tRq1 D/Lamda Rq2
 *        # make a svd of  Q(a,b)
 *        # then     Q(a,b) = tRq1 D/Lamda Rq2
 *
 *    And finally :
 *         tRE2 tRq1 D/Lamda Rq2 RE1 = tRh2 D Rh1
 *    So
 *        ...
 *
 *    For estimating a,b,Lambda=L, we use :
 *      # equality of determinant in [2]  gives :
 *
 *               a = L^3 det H = L^3 l1l2l3  [P1]
 *
 *      # compute M tM for both side we have :
 *
 *                                   (a 0 b)   (a 0 0)     (a^2 + b^2      0      b)
 *          tRE2 Q(a,b) Q(a,b) RE2 = (0 1 0) * (0 1 0) =   (    0          1      0)
 *                                   (0 0 1)   (b 0 1)     (    b          0      1)
 *        as trace is invariant by changing base (tr(A-1BA) = tr(B)) we have :
 *
 *          a^2 + b^2 + 2 =  L^2 Tr(H t H) = L^2 (l1^2 + l2^2 + l3^2) [P2]
 *
 *      # and as tr( (M tM) t(MtM) ) it the frobenius norm of M tM an we have, and still base invariant :
 *
 *         (a^2+b^2) ^2 + 2 b^2 +2 = L4^(l1^4+l2^4+l3 ^4)  [P3]
 *
 *     So syntheicaly we have the set of equation :
 *
 *        [P1] a                       = L^3 D
 *        [P2] a^2 + b^2 + 2           = L^2 Tr2
 *        [P3] (a^2+b^2) ^2 + 2 b^2 +2 = L^4 Tr4
 *
 *     By substitution in [P3] we get a polynomial equation in L
 *     (a^2 + b^2) comes from [P2] and b^2 from [P1] & [P2]) :
 *         (L^2Tr^2 -2) ^2 + 2(L^2 Tr2-2 -L^6 D^2)   +2 = L^4 Tr4
 *     This 6-degree in L can be solved as 3 degree i n L2
 *     Once we have values of L, we use [P1] to have value of a, and [P2] to have values of b.
 *
 */


/**
 * @brief cPS_CompPose::cPS_CompPose
 * @param aSetCple
 */

class cPS_CompPose
{
   public :
        cPS_CompPose( cSetHomogCpleDir &,bool ModeBench=false);

   private :
        static void SetPt( cDenseVect<tREAL8>&,size_t aIndex,const cPt3dr&,tREAL8 aMul);

};

void cPS_CompPose::SetPt( cDenseVect<tREAL8>& aVect,size_t aIndex,const cPt3dr& aPt,tREAL8 aMul)
{
    for (size_t aK=0 ; aK<3 ; aK++)
        aVect(aIndex+aK) = aPt[aK] * aMul;
}

/*
 * @brief cPS_CompPose::cPS_CompPose
 * @param aSetCple
 */


cPS_CompPose::cPS_CompPose(cSetHomogCpleDir & aSetCple,bool ModeBench)
{
    const std::vector<cPt3dr>& aV1 = aSetCple.VDir1();
    const std::vector<cPt3dr>& aV2 = aSetCple.VDir2();

    //  [1]   Estimate the homography
    cDenseVect<tREAL8> aVect(9);
    cLeasSqtAA<tREAL8> aSys(9);

    for (size_t aKP=0 ; aKP<aV1.size() ; aKP++)
    {
        const cPt3dr & aP1 = aV1.at(aKP);
        const cPt3dr & aP2 = aV2.at(aKP);

        //  Add the equation L1.P1 z2 - L3.P1 x2 = 0
        SetPt(aVect,0,aP1,aP2.z());
        SetPt(aVect,3,aP1,0);
        SetPt(aVect,6,aP1,-aP2.x());
        aSys.PublicAddObservation(1.0,aVect,0.0);

        // Add the equation  L2.P1 z2 - L3.P1 y2 = 0
        SetPt(aVect,0,aP1,0);
        SetPt(aVect,3,aP1,aP2.z());
        SetPt(aVect,6,aP1,-aP2.y());
        aSys.PublicAddObservation(1.0,aVect,0.0);
    }
    aSys.AddObsFixVar(tREAL8(aV1.size()),8,1.0);
    cDenseVect<tREAL8> aSol = aSys.PublicSolve();
    cDenseMatrix<tREAL8> aMatH = Vect2MatEss(aSol);

    if (ModeBench)
    {
       for (size_t aKP=0 ; aKP<aV1.size() ; aKP++)
       {
           const cPt3dr & aP1 = aV1.at(aKP);
           const cPt3dr & aP2 = aV2.at(aKP);

           tREAL8 aDif = Norm2(VUnit(aMatH * aP1) - aP2);

           MMVII_INTERNAL_ASSERT_bench(aDif<1e-7,"cPS_CompPose :: Matrix");
          // StdOut()  << " * DiiIiff=  " <<aDif << "\n";
       }
    }


    // [2]  Estimate the paramater a,b,L

     //  [2.1] make a SVD, maybe not optimal for invariant, btw will be used later
     cResulSVDDecomp<tREAL8> aSvdH = aMatH.SVD();
     cDenseVect<tREAL8>      aSingV = aSvdH.SingularValues();

     // [2.2]  extract invariant
              // ------- singular values
     tREAL8 aL1 = aSingV(0);
     tREAL8 aL2 = aSingV(1);
     tREAL8 aL3 = aSingV(2);
              // -------- square of singular values
     tREAL8 aSqL1 = Square(aL1);
     tREAL8 aSqL2 = Square(aL2);
     tREAL8 aSqL3 = Square(aL3);
             // --- Determinant,  Trace of MtM, trace of MtM^2
     tREAL8 aDet = aL1*aL2*aL3 ;
     tREAL8 aTr2 = aSqL1+aSqL2+aSqL3;
     tREAL8 aTr4 = Square(aSqL1)+Square(aSqL2)+Square(aSqL3);

     // some basic check, compare values from singular with direct computation
     if (ModeBench)
     {
         cDenseMatrix<tREAL8> aH_tH = aMatH * aMatH.Transpose();

         if (0)
         {
             StdOut()  << "DDDD= " << aDet - aMatH.Det() << "\n";
             StdOut()  << "tTr2= " << aTr2 - aH_tH.Trace() << "\n";
             StdOut()  << "tTr4= " << aTr4 - aH_tH.DIm().SqL2Norm(false) << "\n";
             StdOut() << "\n";
         }

         MMVII_INTERNAL_ASSERT_bench(std::abs(aDet - aMatH.Det())<1e-7,  "cPS_CompPose :: Det");
         MMVII_INTERNAL_ASSERT_bench(std::abs(aTr2 - aH_tH.Trace())<1e-7,"cPS_CompPose :: Tr2");
         MMVII_INTERNAL_ASSERT_bench(std::abs(aTr4 - aH_tH.DIm().SqL2Norm(false))<1e-7,"cPS_CompPose :: Tr4")
     }

     //     (L^2Tr^2 -2) ^2 + 2(L^2 Tr2-2 -L^6 D^2)   +2 = L^4 Tr4
     //  2 D^2 LL^3 + Tr4 LL^2   - 2 Tr2 LL +4 -    2   - 4 +4 Tr2 LL -Tr2^2 LL^2
     //  2 D^2 LL^3  + (Tr4-Tr2^2) LL^2  +  2 Tr2 LL - 2

     // Extract polyno
     std::vector<tREAL8> aVCoef {-2.0,2.0*aTr2,aTr4-Square(aTr2),2.0*Square(aDet)};
     cPolynom<tREAL8> aPolL(aVCoef);
     std::vector<tREAL8> aLSol = aPolL.RealRoots(1e-7,10);

     StdOut()  << "LAMBDAS= " << aLSol  << " SV=" << aSingV << "\n";

     for (const auto & aL2 : aLSol)
     {
         if (aL2>-1e-6)
         {
             tREAL8 aAbsLambda = std::sqrt(std::max(0.0,aL2));
             int aS0Lambda = (aAbsLambda >0) ? -1 : 1;
             for (int aSignL=aS0Lambda ; aSignL<=1 ; aSignL+=2)
             {
                 tREAL8 aLambda = aAbsLambda * aSignL;
                //  [P1] a  = L^3 D
                 tREAL8 aA = Cube(aLambda) * aDet;
                 //  a^2 + b^2 + 2           = L^2 Tr2
                 tREAL8 aB2 = Square(aLambda)* aTr2 -Square(aA)-2.0;
                 if (aB2>-1e-6)
                 {
                     tREAL8 aAbsB = std::sqrt(std::max(0.0,aB2));
                     int aS0B = (aAbsB >0) ? -1 : 1;
                     for (int aSignB=aS0B ; aSignB<=1 ; aSignB+=2)
                     {
                         tREAL8 aB = aAbsB * aSignB;

                         cDenseMatrix<tREAL8>  aQab =  M3x3FromLines
                                                       (
                                                          cPt3dr(aA,0,aB) / aLambda,
                                                          cPt3dr(0,1,0) / aLambda,
                                                          cPt3dr(0,0,1) / aLambda
                                                       );

                         cResulSVDDecomp<tREAL8> aSvdQab = aQab.SVD();
                         cDenseVect<tREAL8>      aSingVQab = aSvdQab.SingularValues();
                         StdOut()<< " SINGV " << aSingVQab
                                 << " DetUab=" << aSvdQab.MatU().Det() << " "<< aSvdH.MatU().Det()
                                 << " DetVab=" << aSvdQab.MatV().Det() << " "<< aSvdH.MatV().Det()
                                 << "\n";
                     }
                 }
             }
         }
     }
     StdOut() << " ----------------------------------------- \n";
   // getchar();

}



/* ************************************** */
/*                                        */
/*            cCamSimul                   */
/*                                        */
/* ************************************** */


cCamSimul::cCamSimul() :
   mCenterGround (10.0,5.0,20.0),
   mProfMin      (10.0),
   mProfMax      (20.0),
   mBsHMin       (0.1),
   mBsHMax       (0.5),
   mRandInterK   (0.1)
{
}

cCamSimul::~cCamSimul()
{
    DeleteAllAndClear(mListCam);
    DeleteAllAndClear(mListCalib);
}

bool cCamSimul::ValidateCenter(const cPt3dr & aP2) const
{ 
    if (mListCam.empty()) return true;

    tREAL8 aTetaMin = 1e10;
    cPt3dr aV20 = aP2 - mCenterGround;
    for (const auto & aPtr : mListCam)
    {
         cPt3dr aV10 = aPtr->Center() - mCenterGround;
         UpdateMin(aTetaMin,AbsAngleTrnk(aV10,aV20));
    }
    return  (aTetaMin>mBsHMin) && (aTetaMin<mBsHMax);
}

cPt3dr  cCamSimul::GenCenterWOCstr(bool SubVert) const
{
    // Case "sub-vertical" we generate a point above mCenterGround
    //   * the delta in x and y is in an interval {
    if (SubVert)
    {
        auto v1 = RandUnif_C();
        auto v2 = RandUnif_C();
        auto v3 = RandInInterval(mProfMin,mProfMax);
        return    mCenterGround  + cPt3dr(v1,v2,1.0) * v3;
    }

    //
    auto v1 = cPt3dr::PRandUnit();
    auto v2 = RandInInterval(mProfMin,mProfMax);
    return mCenterGround + v1 * v2;
}


cPt3dr  cCamSimul::GenValideCenter(bool SubVert) const
{
   cPt3dr aRes = GenCenterWOCstr(SubVert);
   while (! ValidateCenter(aRes))
          aRes = GenCenterWOCstr(SubVert);

  // MMVII_INTERNAL_ASSERT_strong(!SubVert,"GenValideCenter");
  // StdOut() << "GenValideCenterGenValideCenter " << SubVert << "\n";
   return aRes;
}


void cCamSimul::AddCam(cPerspCamIntrCalib * aPC,bool SubVert)
{
      cPt3dr aNewC = GenValideCenter(SubVert);

      cPt3dr aK = VUnit(mCenterGround - aNewC);
      cPt3dr aI = cPt3dr::PRandUnitNonAligned(aK,1e-2);
      cPt3dr aJ = VUnit(aK ^aI);
      aI = aJ ^aK;


      cRotation3D<tREAL8> aRot(M3x3FromCol(aI,aJ,aK),false);

      aNewC += cPt3dr::PRandC() * mRandInterK;
      cIsometry3D<tREAL8> aPose(aNewC,aRot);

      mListCam.push_back(new cSensorCamPC("Test",aPose,aPC));
}

void cCamSimul::AddCam(eProjPC aProj,bool SubVert)
{
    // 1 => means Deg of direct dist is 2 (dir inverse is 5,1,1)
    cPerspCamIntrCalib * aCalib = cPerspCamIntrCalib::RandomCalib(aProj,1);

    mListCalib.push_back(aCalib);
    AddCam(aCalib,SubVert);
}

cCamSimul * cCamSimul::Alloc2VIewTerrestrial(eProjPC aProj1,eProjPC aProj2,bool SubVert)
{
   cCamSimul * aRes = new cCamSimul();

   aRes->AddCam(aProj1,SubVert);
   aRes->AddCam(aProj2,SubVert);

   return aRes;
}

void cCamSimul::TestCam(cSensorCamPC * aCam) const
{
	StdOut() << "CC " << aCam->Center()  << " CG=" << mCenterGround << std::endl;

cPt3dr aV = aCam->Center() - mCenterGround;

StdOut()  << " I " << Cos(aV,aCam->AxeI())
          << " J " << Cos(aV,aCam->AxeI())
          << " K " << Cos(aV,aCam->AxeK())
	  << " \n";

	StdOut() << "Vis " <<  aCam->IsVisible(mCenterGround) << std::endl;
}

void cCamSimul::BenchPoseRel2Cam
     (
        cTimerSegm * aTS,
        bool         PerfInter,
        bool         isSubVert,
        bool         isPlanar
     )
{
    thread_local static int aCpt=0;
    /// cLinearOverCstrSys<tREAL8> *  aSysL1 = AllocL1_Barrodale<tREAL8>(9);
    // cLinearOverCstrSys<tREAL8> *  aSysL1 = new cLeasSqtAA<tREAL8>(9);
    cLeasSqtAA<tREAL8> aSysL2(9);

    thread_local static int aCptPbL1 = 0;

    for (int aK1=0 ; aK1<(int)eProjPC::eNbVals ; aK1++)
    {
        for (int aK2=0 ; aK2<(int)eProjPC::eNbVals ; aK2++)
        {
            cAutoTimerSegm aTSSim(aTS,"CreateSimul");
            aCpt++;
            cCamSimul * aCamSim = cCamSimul::Alloc2VIewTerrestrial(eProjPC(aK1),eProjPC(aK2),isSubVert);

            // we want to test robustness in perfect degenerate & close to degenertae
            if (PerfInter)
               aCamSim->mRandInterK = 0.0;

            // Generate 2 cams 
            cSensorCamPC * aCam1 = aCamSim->mListCam.at(0);
            cSensorCamPC * aCam2 = aCamSim->mListCam.at(1);

            // generate  perfect homologous point
            cSetHomogCpleIm aSetH;
            size_t aNbPts = 40;
          ///  StdOut() << "Innnn IssssPllannnn " << isPlanar << "\n";

           for (size_t aKP=0 ; aKP<aNbPts ; aKP++)
           {
               // StdOut() << " Planaaarr " << isPlanar << " K=" << aKP << "\n";
               cHomogCpleIm aCple =  isPlanar                                                     ?
                                     aCam1->RandomVisibleCple(aCamSim->mCenterGround.z(),*aCam2)  :
                                     aCam1->RandomVisibleCple(*aCam2)                             ;
               aSetH.Add(aCple);
           }

      ///     StdOut() << "Ouut IssssPllannnn " << isPlanar << "\n";

            // Make 3D direction of points
            cSetHomogCpleDir aSetD (aSetH,*(aCam1->InternalCalib()),*(aCam2->InternalCalib()));

            cAutoTimerSegm aTSGetMax(aTS,"GetMaxK");
         //   StdOut() << "Ouut IssssPllannnn " << isPlanar << "\n";

            if (isPlanar )
            {
                 cPS_CompPose aPsC(aSetD,true);
            }
            else
            {
                   int aKMax =  MatEss_GetKMax(aSetD,1e-6);

                  // These point where axe k almost intersect, the z1z2 term of mat ess is probably small
                  // and must not be KMax
                   MMVII_INTERNAL_ASSERT_bench(aKMax!=8,"cComputeMatEssential::GetKMax");

                // Now test that residual is ~ 0 on these perfect points
                 cAutoTimerSegm aTSL2(aTS,"L2");
                 cMatEssential aMatEL2(aSetD,aSysL2,aKMax);

                 {
                     cIsometry3D<tREAL8>  aPRel =  aCam1->RelativePose(*aCam2);
                     // When we give aPRel
                     aMatEL2.ComputePose(aSetD,&aPRel);
                 }
                 MMVII_INTERNAL_ASSERT_bench(aMatEL2.AvgCost(aSetD,1.0)<1e-5,"Avg cost ");

                cAutoTimerSegm aTSL1(aTS,"L1");
                cLinearOverCstrSys<tREAL8> *  aSysL1 = AllocL1_Barrodale<tREAL8>(9);
                cMatEssential aMatEL1(aSetD,*aSysL1,aKMax);
                MMVII_INTERNAL_ASSERT_bench(aMatEL1.AvgCost(aSetD,1.0)<1e-5,"Avg cost ");

               for (int aK=0 ; aK<4 ; aK++)
                    aSetD.GenerateRandomOutLayer(0.1);

                cMatEssential aMatNoise(aSetD,*aSysL1,aKMax);

                delete aSysL1;
	    
               if (0)
               {
                  StdOut() << "Cpt=" << aCpt
                 << " Cost95= "  << aMatNoise.KthCost(aSetD,0.95)
                   << " Cost80= "  << aMatNoise.KthCost(aSetD,0.70)
                 << " KMax= "  << aKMax
                  // << " Disp= "  << aSetD.Disp()
                  << "\n";
                 MMVII_INTERNAL_ASSERT_bench(aMatNoise.KthCost(aSetD,0.70) <1e-5,"Kth cost ");
                  MMVII_INTERNAL_ASSERT_bench(aMatNoise.KthCost(aSetD,0.95) >1e-2,"Kth cost ");
               }

             // We test if the residual at 70% is almost 0 (with 4/40 outlayers)
             if (aMatNoise.KthCost(aSetD,0.70)>1e-5)
                  aCptPbL1++;
        }

	    if (BUGME) 
	    {

           StdOut() << "***************************************************" << std::endl;
                getchar();
	    }


            delete aCamSim;
        }
    }
    StdOut() << "CPT " << aCptPbL1  << "  " << aCpt << std::endl;

}


}; // MMVII




