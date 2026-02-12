#include "MMVII_PoseRel.h"
#include "MMVII_Tpl_Images.h"


/* We have the image formula w/o distorsion:


*/

namespace MMVII
{


class cPSC_PB  //< cPS_CompPose Param Bench
{
     public :
       std::string mMsg;
       bool        mModeEpip;
       bool        mWithGT;

       //cSetHomogCpleDir mSetDir;

       cPSC_PB(const std::string & aMsg,bool aModeEpip = false) :
           mMsg      (aMsg),
           mModeEpip (aModeEpip),
           mWithGT   (false)
       {
       }
};


/**
 * @brief cPS_CompPose::cPS_CompPose -> Planar Scene Compute Pose
 * @param aSetCple
 */


class cPS_CompPose
{
   public :
        cPS_CompPose( cSetHomogCpleDir &,const  cPSC_PB * = nullptr);

        typedef std::pair<cPSC_PB,cSetHomogCpleDir>  tResSimul;

        /** generate pair 3D of direction that can correpond to coherent camera, can be more
         "extremate" than with camera, can simulate for example two side of plane.
         */
         static tResSimul SimulateDirAny(tREAL8 aEps,tREAL8 aDistPlane,bool isSameSide); // Steep of plane, may be adjusted


         /** generate pair of direction the correspond to camera already in epipolar config, more for
             debugin than for check */
         static cSetHomogCpleDir SimulateDirEpip
                                 (
                                      tREAL8 aZ,           // Altitude of both camera
                                      tREAL8 aSteepPlane,  // Z = X * Steep
                                      tREAL8 aRho,         // Circle of random plane
                                      bool   BaseIsXP1     // is the base (1,0,0) or (-1,0,0) ?
                                 );
   private :

        ///  Compute the 3D homog matrix
        static cDenseMatrix<tREAL8>   ComputeMatHom3D(cSetHomogCpleDir &,const  cPSC_PB *);

        bool TestOneHypoth(cResulSVDDecomp<tREAL8>& aSvdH,const cPt3dr&ABL,int SignB,const cPt3dr & aSignD);

        /// generate a point of view by randomizing Theta
       // static    cPt3dr RandPointOfView(const cPt2dr & aRhoZ);
        static    cPt3dr RandPointOfView(int aSign,tREAL8 aDPlane);


        /// Utilitary : transferate "Pt" in "Vect" at offset "Index" muliplied by "Mul"
        static void SetPt( cDenseVect<tREAL8>& aVect,size_t aIndex,const cPt3dr& aPt,tREAL8 aMul);


        const std::vector<cPt3dr>* mCurV1 ;
        const std::vector<cPt3dr>* mCurV2 ;
        size_t                     mCurNbPts;
        const  cPSC_PB *           mCurParam;

        // parameters extract of homog matrix , used to compute polynome
        tREAL8                     mHM_Det;
        tREAL8                     mHM_Tr2;
        tREAL8                     mHM_Tr4;

        size_t                     mNbSolTested; ///< number of sol we try
        size_t                     mNbSolOk;     ///< number of sol possible at end
};




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
 *     Once we have QabL = Q(a,b)/L :
 *
 *              Qabl = tRq2  D  Rq1
 *              H =  tRh2 D Rh1
 *
 *
 */


/*
cPt3dr cPS_CompPose::RandPointOfView(const cPt2dr & aRhoZ)
{
    return Cyl2Cart(cPt3dr(aRhoZ.x(),RandUnif_Angle(),aRhoZ.y()));
}
*/



cSetHomogCpleDir cPS_CompPose::SimulateDirEpip
                        (
                             tREAL8 aZ,           // Altitude of both camera
                             tREAL8 aSteep,  // Z = X * Steep
                             tREAL8 aRho,
                             bool   BaseIsXP1     // is the base (1,0,0) or (-1,0,0) ?
                        )
{
     std::vector<cPt3dr>  aVDir1; // store directions 1
     std::vector<cPt3dr>  aVDir2; // store directions 2
     // Is C2 in X+ or X- relatively to C1
     tREAL8 aDx = BaseIsXP1 ? 0.5 : - 0.5;

     cPt3dr aC1(-aDx,0.0,aZ);  // Center of Im1
     cPt3dr  aC2( aDx,0.0,aZ); // Center of Im2

     //  Check that the point are  on the same side of the plane and far enough
     {
         tREAL8 aDz1 = aC1.z()-aSteep*aC1.x();
         tREAL8 aDz2 = aC2.z()-aSteep*aC2.x();
         MMVII_INTERNAL_ASSERT_bench (std::abs(aDz1)>aRho/100.0,"Bad Param in SimulateDirEpip");
         MMVII_INTERNAL_ASSERT_bench (std::abs(aDz2)>aRho/100.0,"Bad Param in SimulateDirEpip");
         MMVII_INTERNAL_ASSERT_bench ((aDz1>0)==(aDz2>0),"Bad Param in SimulateDirEpip");

     }

     // Generate a given number of point +- random
     for (int aK=0 ; aK<100 ; aK++)
     {
         // the plane component of point are generate in the circle unit
         cPt2dr aPPlane =cPt2dr::PRandInSphere()*aRho;

         // The 9 first point are deterministic, A- for debuging B- assuring th
         if (aK<9)
         {
             aPPlane = ToR(FreemanV9[aK])* aRho/2.0;
         }

         // from plani generate point in 3d plane
         cPt3dr aPGround(aPPlane.x(),aPPlane.y(),aSteep*aPPlane.x());

         // Add the 2 directions
         aVDir1.push_back(VUnit(aPGround-aC1));
         aVDir2.push_back(VUnit(aPGround-aC2));

         if (aK==0)
         {
             StdOut() << " V1V2" << aVDir1.back() << aVDir2.back() << "\n";
         }
     }

     return cSetHomogCpleDir(aVDir1,aVDir2);
}


/**
 *  Generate random center using spherical cordinates
 *
 */

cPt3dr cPS_CompPose::RandPointOfView(int aSign,tREAL8 aDPlane)
{
    // any value  in [0,2PI]
    tREAL8 aTeta =  RandUnif_Angle();
    // any value, far enough from equator
    tREAL8 aPhi = (1.0-std::sqrt(RandInInterval(0.0,1.0))) * (M_PI/2.0);
    // value in [0.1,20] with non uniform distrib, privilegiat low values
    tREAL8 aRho =  std::pow(RandUnif_0_1(),4.0) * 20.0;

    cPt3dr aP = spher2cart(cPt3dr(aTeta,aPhi,aRho));
    aP.z() = std::max(aP.z(),aDPlane) * aSign;

    return aP;
}


/**
 * @brief cPS_CompPose::SimulateDirAny
 *    Generate pairs of bundle for planar estimation in (possibly) complicate situtation.  Method :
 *
 *    [1]  Consider  plane P of equation Z = 0 generate random ground points (X,Y,0) in a circle unit
 *    [2]  Generate random view point C1,C2 :
 *             # assure  d(C1,C2) > Epsilon
 *             # and C1,C2 not in P
 *             # sign control
 *        This can generate some extreme case
 *
 * @param aSteepZ
 * @return
 */

cPS_CompPose::tResSimul cPS_CompPose::SimulateDirAny(tREAL8 aEpsilon,tREAL8 aDistPlane,bool SameSide)
{
    tREAL8 aSign = SameSide ? 1 : -1;

    cPt3dr aC1 = RandPointOfView(1,aDistPlane);
    cPt3dr aC2 = aC1;
    while (Norm2(aC1-aC2)<aEpsilon)
         aC2 = RandPointOfView(aSign,aDistPlane);


    tRotR aR1 = tRotR::RandomRot();       // Global rotation applied to dir 1
    tRotR aR2 = tRotR::RandomRot();       // Global rotation applied to dir 2

    std::vector<cPt3dr>  aVDir1; // store directions 1
    std::vector<cPt3dr>  aVDir2; // store directions 2
    for (int aK=0 ; aK<100 ; aK++)
    {
        cPt2dr aPPlane = cPt2dr::PRandInSphere();
        // the planer comp of point are generate in the circle unit
        cPt3dr aPGround = TP3z(aPPlane, 0.0);

        aVDir1.push_back(aR1.Value(VUnit(aPGround-aC1)));
        aVDir2.push_back(aR2.Value(VUnit(aPGround-aC2)));
    }

    cPSC_PB aPSC("Any",false);
    return tResSimul(aPSC,cSetHomogCpleDir(aVDir1,aVDir2));
}



void cPS_CompPose::SetPt( cDenseVect<tREAL8>& aVect,size_t aIndex,const cPt3dr& aPt,tREAL8 aMul)
{
    for (size_t aK=0 ; aK<3 ; aK++)
        aVect(aIndex+aK) = aPt[aK] * aMul;
}

/*
 * @brief cPS_CompPose::cPS_CompPose
 * @param aSetCple
 */



cDenseMatrix<tREAL8>
    cPS_CompPose::ComputeMatHom3D(cSetHomogCpleDir &aSetCple,const  cPSC_PB * aCurParam)
{
    const std::vector<cPt3dr>* aCurV1   = & aSetCple.VDir1() ;
    const std::vector<cPt3dr>* aCurV2   = & aSetCple.VDir2() ;
    size_t aCurNbPts = aCurV1->size();

     //  [1]   Estimate the homography
     cDenseVect<tREAL8> aVect(9);
     cLeasSqtAA<tREAL8> aSys(9); // least square to estimate parameters

     for (size_t aKP=0 ; aKP<aCurNbPts ; aKP++)
     {
         const cPt3dr & aP1 = aCurV1->at(aKP);
         const cPt3dr & aP2 = aCurV2->at(aKP);

         //  Add the equation L1.P1 z2 - L3.P1 x2 = 0
         SetPt(aVect,0,aP1,aP2.z());
         SetPt(aVect,3,aP1,0); // 0 => reset, P1 will be unused
         SetPt(aVect,6,aP1,-aP2.x());
         aSys.PublicAddObservation(1.0,aVect,0.0);

         // Add the equation  L2.P1 z2 - L3.P1 y2 = 0
         SetPt(aVect,0,aP1,0);  // reset, P1 will be unused
         SetPt(aVect,3,aP1,aP2.z());
         SetPt(aVect,6,aP1,-aP2.y());
         aSys.PublicAddObservation(1.0,aVect,0.0);
     }
     // Fix last value as matrix is up to a scale
     aSys.AddObsFixVar(tREAL8(aCurNbPts),8,1.0);
     cDenseVect<tREAL8> aSol = aSys.PublicSolve();
     cDenseMatrix<tREAL8> aMatH = Vect2MatEss(aSol);

     if (aCurParam)
     {
        for (size_t aKP=0 ; aKP<aCurNbPts ; aKP++)
        {
            const cPt3dr & aP1 = aCurV1->at(aKP);
            const cPt3dr & aP2 = aCurV2->at(aKP);

            // As point are projective P and -P are equivalent : use line-angle
            tREAL8 anAng = AbsLineAngleTrnk(aMatH*aP1,aP2);
            MMVII_INTERNAL_ASSERT_bench(anAng<1e-7,"cPS_CompPose :: Matrix");

            // This test can no longer sucess with data simulated by direction (more chalenging)
            if (0)
            {
                tREAL8 aDif = Norm2(VUnit(aMatH * aP1) - aP2);
                MMVII_INTERNAL_ASSERT_bench(aDif<1e-7,"cPS_CompPose :: Matrix");
            }
          //  StdOut()  << " * DiiIiff=  " << AbsLineAngleTrnk(aMatH*aP1,aP2) << "\n";
        }
     }
     // Not real reason to do that, just to supress an arbitray degree of freedom
     if (aMatH.Det() < 0)
         aMatH = - aMatH;
     return aMatH;
}

/*
    const std::vector<cPt3dr>* mCurV1 = aSetCple.VDir1();
    const std::vector<cPt3dr>* mCurV2 = aSetCple.VDir1();
 */

bool cPS_CompPose::TestOneHypoth
     (
        cResulSVDDecomp<tREAL8>& aSvdH,  // SVD of 3D-Homograhy
        const cPt3dr & anABL,            // Value for A,B and lamda
        int   aSignBase,                 // Sign of base
        const cPt3dr & aSignDiag         // sign of diagonal
     )
{
    mNbSolTested++;

    static int aCptSol=0;  // Debug counter
    aCptSol++;
    //StdOut() << " CPTSOL=" << aCptSol << "\n";
    //bool Debug = (aCptSol==395);

    cPt3dr aBase(aSignBase,0,0); // vector of base in repair of Im1
    // Homography 1->2 in epipolar repair
    cDenseMatrix<tREAL8>  aQab =  M3x3FromLines
                                  (
                                         cPt3dr(anABL.x(),0,anABL.y()) / anABL.z(),
                                         cPt3dr(0,1,0)                 / anABL.z(),
                                         cPt3dr(0,0,1)                 / anABL.z()
                                  );


    cResulSVDDecomp<tREAL8> aSvd_Qab = aQab.SVD(true);
    cDenseVect<tREAL8>      aSingVQab = aSvd_Qab.SingularValues();

    cDenseMatrix<tREAL8> aMatTransfo
                = cDenseMatrix<tREAL8>::Diag(cDenseVect<tREAL8>(aSignDiag.ToStdVector()));

    //  Qab =  Uq Diag tVq ; btw, with M=aMatTransfo, we have  M Diag M = Diag , M =Mt
    // so we can write Qab =  Uq MM  Diag MM tVq = UqM  Diag t(VqM)
    // and  (UqM,VqM) are another valide svd decompisition of Qab for the same diagonal

    cDenseMatrix<tREAL8>  aQU =  aSvd_Qab.MatU() * aMatTransfo;
    cDenseMatrix<tREAL8>  aQV =  aSvd_Qab.MatV() * aMatTransfo;
    if (mCurParam)
    {

        tREAL8 aDist = aSvdH.SingularValues().L2Dist(aSingVQab) / (1+std::sqrt(mHM_Tr2));
        MMVII_INTERNAL_ASSERT_bench(aDist<1e-7,"cPS_CompPose dist sing val");

        // to be honest, I cannot prove formally this assertion, btw
        // it happens to be true even with very randomized condition
        MMVII_INTERNAL_ASSERT_bench( aQU.Det()>0," SvdQab.MatU");
        MMVII_INTERNAL_ASSERT_bench( aQV.Det()>0," SvdQab.MatV");
        MMVII_INTERNAL_ASSERT_bench( aSvdH.MatU().Det()>0," aSvdH.MatU");
        MMVII_INTERNAL_ASSERT_bench( aSvdH.MatV().Det()>0," aSvdH.MatV");
    }


 /*                  H= Uh Diag t Vh
 *             v ------------------------> u
 *             |                          |
 *        Ev  \ /                        \ / Eu
 *             |                          |
 *             ve -----------------------> ue
 *                   Qab = Uq Diag t Vq
 *
 *     We can write ue from v with 2 different path:
 *        ue = Eu(u) = Eu Uh Diag tVh (v)
 *        ue = Uq Diag tVq Ev (v)
 *     Then :
 *            Eu Uh Diag tVH = Uq Diag tVq Ev
 *     As Diag are equals we can identify both side of orthognal matrices and get :
 *          Eu = Uq tUh
 *          Ev = Vq tVh
 */


    tRotR  aRE1(aQV *  aSvdH.MatV().Transpose(),false);
    tRotR  aRE2(aQU *  aSvdH.MatU().Transpose(),false);

    tREAL8 aNbM1 = 0.0; FakeUseIt(aNbM1);
    tREAL8 aNbM2 = 0.0; FakeUseIt(aNbM2);
    std::vector<cPt3dr> aVecPt;
    int aNbBundleParal = 0;

     for (size_t aK=0 ; aK<mCurNbPts ; aK++)
     {
         cPt3dr aPE1 = aRE1.Value(mCurV1->at(aK));
         cPt3dr aPE2 = aRE2.Value(mCurV2->at(aK));

         // mesaure if the bundles are parallel
         tREAL8 aDistDir = DistDirLine(aPE1,aPE2);

         // if not paral an intersection can be computed
         if (aDistDir>1e-6)
         {
            tSegComp3dr aSeg1(cPt3dr(0,0,0),aPE1);
            tSegComp3dr aSeg2(aBase,aBase+aPE2);

            cPt3dr aCoeffI;
            cPt3dr anInter = BundleInters(aCoeffI,aSeg1,aSeg2);
            aVecPt.push_back(anInter);

            //tREAL8 aWeight = 1.0 / (1.0+std::abs(aCoeffI.z()));
            if (aCoeffI.x() < 0)
                aNbM1 += 1;
            if (aCoeffI.y() < 0)
                 aNbM2 += 1;


            if (mCurParam)
            {
               tREAL8 aDist = std::abs(aCoeffI.z()) / (1.0+Square(aCoeffI.x())+Square(aCoeffI.y()));
               tREAL8 aD1 = aSeg1.Dist(anInter);
               tREAL8 aD2 = aSeg2.Dist(anInter);
               if ((aDist>1e-6) || (Norm2(aCoeffI)>1e10))
               {
                  StdOut() << " D1D2 " << aD1 << " " << aD2  << aCoeffI << " " << aDist
                                <<  " CPT=" << aCptSol << " Kpt=" << aK << "\n";
                  StdOut() << " Msg=" <<  mCurParam->mMsg  << "\n";
                  if (Norm2(aCoeffI)>1e10)
                  {

                     StdOut() << " COEFFI=" << aCoeffI << " I=" << anInter
                              <<  " CPT=" << aCptSol << " K=" << aK <<"\n";

                     StdOut() << " V1=" << mCurV1->at(aK)<< " V2="<< mCurV2->at(aK)
                              << " E1=" << aPE1 << " E2=" << aPE2 << "\n";

                     getchar();
                  }
                  if (aDist>1e-6)
                  {
                     MMVII_INTERNAL_ASSERT_bench(false,"PlanEpip : dist bund");
                  }
               }
            }
         }
         else
         {
             aNbBundleParal++;
         }
     }

     if (aVecPt.size()<3)
     {
         if (mCurParam)
         {
            StdOut() << "    ****** NB BunPar " << aNbBundleParal << " on " << mCurNbPts << "\n";
         }
         return false;
     }

     auto [aPlane,aRes] =  cPlane3D::RansacEstimate(aVecPt,true,100);

     cPt3dr aC0 = aPlane.ToLocCoord(cPt3dr(0,0,0));
     cPt3dr aC1   = aPlane.ToLocCoord(aBase);
     bool isZP0 = aC0.z() > 0;
     bool isZP1 = aC1.z() > 0;

     bool   isSameSide = (isZP0==isZP1) ; FakeUseIt(isSameSide);


     tREAL8 aScorePos = (aNbM1+aNbM2) / (2.0* mCurV1->size());


     if (mCurParam)
     {
         if (true) //  aScorePos<1e-4)
         {

             StdOut()  <<  "PooOs:= "
                        << " SignB= " << aSignBase  << " SD=" <<  aSignDiag
                        <<  ((aScorePos<1e-4)  ? " *** " : "     ")
                         <<  " " << (isSameSide ? "==" : "!!")
                          <<  ( isZP0 ? "+" : "-")
                           <<  ( isZP1 ? "+" : "-") << " "
                            << aNbM1 << " " << aNbM2
                            <<  " N=" << aPlane.AxeK().y()
                             << " On: " << aScorePos << " CPT=" << aCptSol ;
             StdOut() << "\n";


             if (mCurParam->mModeEpip && (aScorePos<1e-4))
             {
                 /*
                 for (int aY=0 ; aY<3 ; aY++)
                 {
                     StdOut() <<  "                      ";
                     PP_1Line_MatRot(aRE1.Mat(),aY);
                     StdOut() << "   |||   ";
                     PP_1Line_MatRot(aRE2.Mat(),aY);
                     StdOut()  << "\n";
                 }*/
              //   StdOut() << " VVV=" << aVecPt[0] << "\n";

//                 tPoseR aP1()
           //      cPt3dr aPE1 = aRE1.Value(mCurV1->at(aK));
           //     cPt3dr aPE2 = aRE2.Value(mCurV2->at(aK));


                  tPoseR aPC1toE1(cPt3dr(0,0,0),aRE1);
                  tPoseR aPE1toE2(aBase,tRotR::Identity());
                  tPoseR aPC2toE2(cPt3dr(0,0,0),aRE2);

                  //                  C2 <-E2               E2 <-E1    E1 <- C1
                  tPoseR aPC1toC2 =aPC2toE2.MapInverse() * aPE1toE2 * aPC1toE1;


                   StdOut() << " ######## TR12=" << aPC1toC2.Tr()  << aSignDiag << "####### \n";

                 // StdOut()  << aRE1.AxeI() <<  aRE1.AxeJ() << aRE1.AxeK() << "\n";
                 // StdOut()  << aRE2.AxeI() <<  aRE2.AxeJ() << aRE2.AxeK() << "\n";


             }
             mNbSolOk++;
             return true;
         }
         else
         {
             // StdOut()  << " Nbumber Minus -> " << aNbM1  << " " << aNbM2 << "\n";
         }

    }
     return false;
}

cPS_CompPose::cPS_CompPose(cSetHomogCpleDir & aSetCple,const  cPSC_PB * aBenchParam) :
    mNbSolTested (0),
    mNbSolOk     (0)
{
    // const std::vector<cPt3dr>& aV1 = aSetCple.VDir1();
   // const std::vector<cPt3dr>& aV2 = aSetCple.VDir2();
    cDenseMatrix<tREAL8> aMatH = ComputeMatHom3D(aSetCple,aBenchParam); // aV1,aV2,aParamBench);

    mCurV1    = & aSetCple.VDir1() ;
    mCurV2    = & aSetCple.VDir2() ;
    mCurNbPts = mCurV1->size();
    mCurParam = aBenchParam;

    // [2]  Estimate the paramater a,b,L

     //  [2.1] make a SVD, Eigen value used for invariant,  maybe not optimal , btw will be used later
     cResulSVDDecomp<tREAL8> aSvdH = aMatH.SVD(true);
     cDenseVect<tREAL8>      aSingV = aSvdH.SingularValues();

     // [2.2]  compute invariant
              // ------- singular values
     tREAL8 aL1 = aSingV(0);
     tREAL8 aL2 = aSingV(1);
     tREAL8 aL3 = aSingV(2);
              // -------- square of singular values
     tREAL8 aSqL1 = Square(aL1);
     tREAL8 aSqL2 = Square(aL2);
     tREAL8 aSqL3 = Square(aL3);
             // --- Determinant,  Trace of MtM, trace of MtM^2
      mHM_Det = aL1*aL2*aL3 ;
      mHM_Tr2 = aSqL1+aSqL2+aSqL3;
      mHM_Tr4 = Square(aSqL1)+Square(aSqL2)+Square(aSqL3);

     // some basic check, compare values from singular with direct computation
     if (mCurParam)
     {
         cDenseMatrix<tREAL8> aH_tH = aMatH * aMatH.Transpose();

         if (0)
         {
             StdOut()  << "DDDD= " << mHM_Det - aMatH.Det() << "\n";
             StdOut()  << "tTr2= " << mHM_Tr2 - aH_tH.Trace() << "\n";
             StdOut()  << "tTr4= " << mHM_Tr4 - aH_tH.DIm().SqL2Norm(false) << "\n";
             StdOut() << "\n";
         }
         StdOut() << " SINGV " << aSingV << "\n";

         // For determinant of projective, sign is undefined
         MMVII_INTERNAL_ASSERT_bench(RelativeDifference(mHM_Det, std::abs(aMatH.Det()))<1e-7,  "cPS_CompPose :: Det");
         MMVII_INTERNAL_ASSERT_bench(RelativeDifference(mHM_Tr2, aH_tH.Trace())<1e-7,"cPS_CompPose :: Tr2");
         MMVII_INTERNAL_ASSERT_bench(RelativeDifference(mHM_Tr4,aH_tH.DIm().SqL2Norm(false))<1e-7,"cPS_CompPose:Tr4");
         //RelativeDifference()
     }

     //     (L^2Tr^2 -2) ^2 + 2(L^2 Tr2-2 -L^6 D^2)   +2 = L^4 Tr4
     //  2 D^2 LL^3 + Tr4 LL^2   - 2 Tr2 LL +4 -    2   - 4 +4 Tr2 LL -Tr2^2 LL^2
     //  2 D^2 LL^3  + (Tr4-Tr2^2) LL^2  +  2 Tr2 LL - 2

     // [2.3]  compute polynom and roots to have values of lambda^2
     std::vector<tREAL8> aVCoef {-2.0,2.0*mHM_Tr2,mHM_Tr4-Square(mHM_Tr2),2.0*Square(mHM_Det)};
     cPolynom<tREAL8> aPolL(aVCoef);
     std::vector<tREAL8> aLSol = aPolL.RealRoots(1e-7,10);

     // [2.4]  extract    a,b and Lambda
     std::vector<cPt3dr> aVABL;   // vector store values of a,b,Lambda
     for (const auto & aL2 : aLSol)
     {
         if (aL2>-1e-6) // if root positive (up to epslion)
         {
             tREAL8 aAbsLambda = std::sqrt(std::max(0.0,aL2)); // aboslute value
             int aS0Lambda = (aAbsLambda >0) ? -1 : 1; // if 0 no need to explore 2
             for (int aSignL=aS0Lambda ; aSignL<=1 ; aSignL+=2)
             {
                 tREAL8 aLambda = aAbsLambda * aSignL;  // signed lambda
                 tREAL8 aA = Cube(aLambda) * mHM_Det; //  compute a : [P1] a  = L^3 D
                 // compute b^2 :  a^2 + b^2 + 2           = L^2 Tr2
                 tREAL8 aB2 = Square(aLambda)* mHM_Tr2 -Square(aA)-2.0;
                 if (aB2>-1e-6) // if B2 is positive up to epsilon
                 {
                     tREAL8 aAbsB = std::sqrt(std::max(0.0,aB2));
                     int aS0B = (aAbsB >0) ? -1 : 1;
                     for (int aSignB=aS0B ; aSignB<=1 ; aSignB+=2)
                     {
                         tREAL8 aB = aAbsB * aSignB;
                         aVABL.push_back(cPt3dr(aA,aB,aLambda));
                     }
                 }
             }
         }
     }


     //std::vector<cPt3dr> aVPtsSign{{1,1,1},{1,-1,-1},{-1,1,-1},{-1,-1,1}};
     std::vector<cPt3dr> aVPtsSign{{1,1,1}};

     int aNbSol= 0;
     for (int aSignBase =-1 ; aSignBase<=1 ; aSignBase+=2)
     {
         for (const auto & anABL : aVABL)
         {
              for (const auto & aPtSignDiag : aVPtsSign)
              {
                  aNbSol += TestOneHypoth
                  (
                      aSvdH,
                      anABL,
                      aSignBase,
                      aPtSignDiag
                  );
              }
         }
    }

    StdOut() << "------------ " <<  aNbSol << " -------------------------------------------------\n";
     if (aNbSol == 0) getchar();
    if (0&& mCurParam)
    {
         StdOut() << "----------------------------------------" << mCurParam->mMsg << mCurParam->mMsg  << mCurParam->mMsg  <<" \n";
         if (0&&UserIsMPD())
            getchar();
    }

}


void BenchMEP_Coplan()
{

    if (1)
    {
        /* In this test we generate random bundle
         */
       for (int aNbTest=0 ; aNbTest < 100 ; aNbTest++)
       {
            for (int aSign =  1 ; aSign<= 1 ; aSign+=2)
            {
                /*
                tREAL8 aRho1 = RandUnif_0_1() * 2.0;
                tREAL8 aRho2 = RandUnif_0_1() * 2.0;
                tREAL8 aZ1 = 0.1 + RandUnif_0_1() * 2.0;
                tREAL8 aZ2 = aSign*(0.1 + RandUnif_0_1() * 2.0);
*/
                cPS_CompPose::tResSimul aRes = cPS_CompPose::SimulateDirAny(1e-2,0.1,true);
               // std::string aMsg = (aSign>0) ? "++++++++++" : "-----------" ;
                 // cPSC_PB aParam((aSign>0) ? "++++++++++" : "-----------",true);
                cPS_CompPose aPsC(aRes.second,&aRes.first);
           }

            // getchar();
       }
    }

    if (0)
    {
        // Z  Steep  Rho Sens

       std::vector<std::vector<tREAL8>>  aVConf
                                         {
           {-10.0, 0.0,1.0,1.0},
           {-10.0, 0.2,1.0,1.0},
           {-10.0,-0.2,1.0,1.0},
                                               {-1.0, 0.0,1.0,1.0},
                                               {-1.0, 0.1,1.0,1.0},
                                               {-1.0,-0.1,1.0,1.0},

                                               {-1.0, 0.0,1.0,0.0},
                                               {-1.0, 0.1,1.0,0.0},
                                               {-1.0,-0.1,1.0,0.0},

                                               {1.0,0.1,1.0,1.0},
                                             // {1.0,0.,1.0,0.0},
                                              {1.0,0.1,1.0,1.0},
                                              {2.0,1.0,10.0,1.0},
                                              {1.0,0.2,1.0,0}
                                         };
       for (const auto& aV : aVConf)
       {
           cSetHomogCpleDir aSetCple=cPS_CompPose::SimulateDirEpip(aV.at(0),aV.at(1),aV.at(2),aV.at(3)!=0);

           cPSC_PB aParam("EpipInit:"+ToStr(aV),true);
           cPS_CompPose aPsC(aSetCple,&aParam);
           getchar();
       }
    }
    /*
    cSetHomogCpleDir cPS_CompPose::SimulateDirEpip
                            (
                                 tREAL8 aZ,           // Altitude of both camera
                                 tREAL8 aSteep,  // Z = X * Steep
                                 bool   BaseIsXP1     // is the base (1,0,0) or (-1,0,0) ?
                            )
                            */
   //StdOut() << "CPT " << aCptPbL1  << "  " << aCpt << std::endl;

}


}; // MMVII




