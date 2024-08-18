#include "MMVII_PoseRel.h"
#include "MMVII_Tpl_Images.h"


/* We have the image formula w/o distorsion:


*/

namespace MMVII
{



/* ************************************** */
/*                                        */
/*         cSetHomogCpleDir               */
/*                                        */
/* ************************************** */


cSetHomogCpleDir::cSetHomogCpleDir(const cSetHomogCpleIm & aSetH,const cPerspCamIntrCalib & aCal1,const cPerspCamIntrCalib & aCal2) :
    mR1ToInit (tRot::Identity()),
    mR2ToInit (tRot::Identity())
{
     for (const auto & aCplH : aSetH.SetH() )
     {
         mVDir1.push_back(VUnit(aCal1.DirBundle(aCplH.mP1)));
         mVDir2.push_back(VUnit(aCal2.DirBundle(aCplH.mP2)));
     }
}

const std::vector<cPt3dr>& cSetHomogCpleDir::VDir1() const {return mVDir1;}
const std::vector<cPt3dr>& cSetHomogCpleDir::VDir2() const {return mVDir2;}

void  cSetHomogCpleDir::NormalizeRot(cRotation3D<tREAL8>&  aRot ,std::vector<cPt3dr> & aVPts)
{
      cPt3dr aP = Centroid(aVPts);
      tRot  aRKAB  = tRot::CompleteRON(aP);

      tRot  aRepairABK(aRKAB.AxeJ(),aRKAB.AxeK(),aRKAB.AxeI(),false);

      AddRot(aRepairABK.MapInverse() , aRot,aVPts);
}

void  cSetHomogCpleDir::AddRot(const tRot& aNewRot,tRot& aRotAccum,std::vector<cPt3dr> & aVPts)
{
     for (auto & aPt : aVPts)
         aPt  = aNewRot.Value(aPt);

     aRotAccum =  aRotAccum * aNewRot.MapInverse();
}

void cSetHomogCpleDir::NormalizeRot()
{
     NormalizeRot(mR1ToInit,mVDir1);
     NormalizeRot(mR2ToInit,mVDir2);
}

void cSetHomogCpleDir::GenerateRandomOutLayer(double aAmpl)
{
    std::vector<cPt3dr> & aV = HeadOrTail() ?  mVDir1 : mVDir2;
    cPt3dr & aDir = aV.at(RandUnif_N(aV.size()));
    aDir = VUnit(aDir+cPt3dr::PRandC() * aAmpl);
}

void cSetHomogCpleDir::RandomizeRot()
{
    AddRot(tRot::RandomRot(),mR1ToInit,mVDir1);
    AddRot(tRot::RandomRot(),mR2ToInit,mVDir2);
}

void cSetHomogCpleDir::Show() const
{
     cPt3dr aS1(0,0,0),aS2(0,0,0);
     tREAL8  aSomDif = 0;
     for (size_t aK=0 ; aK<mVDir1.size() ; aK++)
     {
        
         // StdOut() << " Dir="  << mVDir1[aK]  << mVDir2[aK] << std::endl;
	 aS1 += mVDir1[aK];
	 aS2 += mVDir2[aK];
	 aSomDif += Norm2(mVDir1[aK]-mVDir2[aK]);
     }
     StdOut() << " ** AVG-Dirs="  << VUnit(aS1)  <<  VUnit(aS2)  << " DIFS=" << aSomDif/mVDir1.size() << std::endl;
}

/* ************************************** */
/*                                        */
/*             MMVII                      */
/*                                        */
/* ************************************** */

/*
 *              (a b c) (x2)
 *              (d e f) (y2)
 *   (x1 y1 z1) (g h i) (z2)
 */

void SetVectMatEss(cDenseVect<tREAL8> & aVect,const cPt3dr& aP1,const cPt3dr& aP2)
{
     aVect(0) = aP1.x() *  aP2.x();
     aVect(1) = aP1.x() *  aP2.y();
     aVect(2) = aP1.x() *  aP2.z();

     aVect(3) = aP1.y() *  aP2.x();
     aVect(4) = aP1.y() *  aP2.y();
     aVect(5) = aP1.y() *  aP2.z();

     aVect(6) = aP1.z() *  aP2.x();
     aVect(7) = aP1.z() *  aP2.y();
     aVect(8) = aP1.z() *  aP2.z();
}

cDenseMatrix<tREAL8> Vect2MatEss(const cDenseVect<tREAL8> & aSol)
{
    cDenseMatrix<tREAL8> aMat  (cPt2di(3,3));

    SetLine(0,aMat,cPt3dr(aSol(0),aSol(1),aSol(2)));
    SetLine(1,aMat,cPt3dr(aSol(3),aSol(4),aSol(5)));
    SetLine(2,aMat,cPt3dr(aSol(6),aSol(7),aSol(8)));

    return aMat;
}

void  MatEssAddEquations(const cSetHomogCpleDir & aSetD,cLinearOverCstrSys<tREAL8> & aSys)
{
     cDenseVect<tREAL8>   aVect(9);
     const std::vector<cPt3dr>&  aVD1 = aSetD.VDir1() ;
     const std::vector<cPt3dr>&  aVD2 = aSetD.VDir2() ;
     for (size_t aKP=0 ; aKP<aVD1.size() ; aKP++)
     {
         SetVectMatEss(aVect,aVD1[aKP],aVD2[aKP]);
	 aSys.PublicAddObservation(1.0,aVect,0.0);
     }
}

/**  For essential matrix as the equations are purely linear,  it is necessary to add
 *  some arbitray constraint to make the system solvable.
 *
 *   The obvious choice is to select one arbitray constraint Xk  and to add Xk=1,
 *   often it's m22 that is fix to 1.  But the problem is that, if we are unlucky,
 *   the natural solution for variable Xk  is Xk=0, so fixing Xk=1 lead to have
 *   infinite value for other.  More generally is natural solution lead to Xk very small,
 *   the system is unstable.
 *
 *   One could argue that this is a very pessimistic assumption, but Murphy's law is 
 *   not ambiguous, if this can happen, it will happen, and much more often than you expect ...
 *
 *   Ideally, we should fix Xk=1 for the variable havind the biggest value, but the problem
 *   if that if dont know this value ...
 *
 *   The implementation try to guess it, using a not fast, but not so slow and, hopefully robust, approach. 
 *   It test all the possible variable :
 *
 *       - for each test, we add the equation Xk=1
 *       - also to stabilize the system we add, with a very small weight  Xj=0 for the other variables
 *       - we compute the solution of the system Sk, normalized by Sk=Sk/||S||inf
 *
 *  Each |Sk| may give a indication of what is the biggest variable, but is biased as it fix Xk=1 and Xj=0 (even with
 *  a vey small weight).
 *  So to have a fair indicator, we compute  S = Sum(|Sk)
 *
 *  A bit heavy ... but not so slow, when we have many points as the longest part, computation of
 *  covariance matrix is done only once .
 *
 */

int   MatEss_GetKMax(const cSetHomogCpleDir & aSetD,tREAL8 aWeightStab,bool Show)
{
    size_t aNbEq =  aSetD.VDir1().size();

    // 1- compute a standard sys
    cLeasSqtAA<tREAL8> aSysOri(9);
    MatEssAddEquations(aSetD,aSysOri);


    // 2- Now try all possible var-fix and accumulate solution
    cDenseVect<tREAL8> aSum(9, eModeInitImage::eMIA_Null);
    for (int aKFix1=0 ; aKFix1<9 ; aKFix1++)
    {
        cLeasSqtAA<tREAL8> aSys =aSysOri.Dup();
        aSys.AddObsFixVar(aNbEq,aKFix1,1.0);

        // 2- Add a "small" weight to 0  be sure the syst will be well condtionned
         for (int aKFix0=0 ; aKFix0<9 ; aKFix0++)
         {
             if (aKFix0 != aKFix1)
                 aSys.AddObsFixVar(aNbEq*aWeightStab,aKFix0,0.0);
         }
        
        cDenseVect<tREAL8> aSol = aSys.Solve();

        tREAL8 aSInf = aSol.LInfNorm();
        for (int aK=0 ; aK<9 ; aK++)
        {
            tREAL8 aNewV = std::abs(aSol(aK) /aSInf);
            aSum(aK) += aNewV;
            if (Show)
                StdOut() << ToStr(aNewV*1000,4) << " ";
        }
        if (Show)
           StdOut() << std::endl;
    }
    if (Show)
    {
       Vect2MatEss(aSum).Show();
       getchar();
    }
    cWhichMax<int,tREAL8> aWMax(-1,0.0);
    for (int aK=0 ; aK<9 ; aK++)
    {
        aWMax.Add(aK,aSum(aK));
    }

    return aWMax.IndexExtre();
}

/* ************************************** */
/*                                        */
/*         cMatEssential                  */
/*                                        */
/* ************************************** */

cMatEssential::cMatEssential(const cSetHomogCpleDir & aSetD,cLinearOverCstrSys<tREAL8> & aSys,int aKFix) :
    mMat  (cPt2di(3,3))
{
     aSys.Reset();
     cDenseVect<tREAL8> aVect(9);

     const std::vector<cPt3dr>&  aVD1 = aSetD.VDir1() ;
     const std::vector<cPt3dr>&  aVD2 = aSetD.VDir2() ;
     for (size_t aKP=0 ; aKP<aVD1.size() ; aKP++)
     {
         MMVII::SetVectMatEss(aVect,aVD1[aKP],aVD2[aKP]);
	 aSys.PublicAddObservation(1.0,aVect,0.0);
     }
     aSys.AddObsFixVar(aVD1.size(),aKFix,1.0);
     cDenseVect<tREAL8> aSol = aSys.Solve();

     SetLine(0,mMat,cPt3dr(aSol(0),aSol(1),aSol(2)));
     SetLine(1,mMat,cPt3dr(aSol(3),aSol(4),aSol(5)));
     SetLine(2,mMat,cPt3dr(aSol(6),aSol(7),aSol(8)));
}


tREAL8  cMatEssential::Cost(const  cPt3dr & aP1,const  cPt3dr &aP2,const tREAL8 & aSigma) const
{
   //  tP1 Mat P2 =0  
   cPt3dr aQ1 = VUnit(aP1 * mMat); // Q1 is orthognal to plane supposed to  contain P2
   cPt3dr aQ2 = VUnit(mMat * aP2); // Q2 is orthognal to plane supposed to  contain P1
					 
   // P1 and P2 are (hopefully) already unitary, so no need to normalize them
   tREAL8 aD = (std::abs(Scal(aQ1,aP2)) + std::abs(Scal(aP1,aQ2))  ) / 2.0;

   if (false)
   {
	   StdOut() << "DDd==" << aD  << aP1 << " " << aP2 << " p1Mp2=" <<   Scal(aP1,mMat * aP2)  
	   << " ML2=" <<mMat.L2Dist(cDenseMatrix<tREAL8>(3, eModeInitImage::eMIA_Null))<< "\n";
   }
   if (aSigma<=0) 
      return aD;

   return (aD*aSigma) / (aD+aSigma);
}

void cMatEssential::Show(const cSetHomogCpleDir& aSetD) const
{

    for (int aY=0 ; aY<3 ; aY++)
    {
        for (int aX=0 ; aX<3 ; aX++)
        {
		StdOut() << " " <<  FixDigToStr(1000*mMat.GetElem(aX,aY),8) ;
        }
	StdOut() << std::endl;
    }
    StdOut() << "     Cost=" << AvgCost(aSetD,1.0) << std::endl;
    cResulSVDDecomp<tREAL8>  aRSVD =  mMat.SVD();
    cDenseVect<tREAL8>       aVP = aRSVD.SingularValues();
    StdOut() <<  "EIGEN-VAL " << aVP(0) << " " << aVP(1) << " " << aVP(2) << std::endl;
    StdOut() << "============================================" << std::endl;
}

tREAL8  cMatEssential::AvgCost(const  cSetHomogCpleDir & aSetD,const tREAL8 & aSigma) const
{
   tREAL8 aSom = 0.0;
   const std::vector<cPt3dr>&  aVD1 = aSetD.VDir1() ;
   const std::vector<cPt3dr>&  aVD2 = aSetD.VDir2() ;
   for (size_t aKP=0 ; aKP<aVD1.size() ; aKP++)
       aSom += Cost(aVD1[aKP],aVD2[aKP],aSigma);

   return aSom / aVD1.size();
}

tREAL8  cMatEssential::KthCost(const  cSetHomogCpleDir & aSetD,tREAL8  aProp) const
{
  std::vector<tREAL8> aVRes;
   const std::vector<cPt3dr>&  aVD1 = aSetD.VDir1() ;
   const std::vector<cPt3dr>&  aVD2 = aSetD.VDir2() ;
   for (size_t aKP=0 ; aKP<aVD1.size() ; aKP++)
       aVRes.push_back(Cost(aVD1[aKP],aVD2[aKP],0.0));

   return Cst_KthVal(aVRes,aProp);
}



cMatEssential::tPose  cMatEssential::ComputePose(const cSetHomogCpleDir & aHom,tPose * aRef) const
{
    /*  We have EssM = U D tV , and due to eigen convention
             1 0 0
        D =  0 1 0      
             0 0 0

        Let "SwXZ" be matrix swaping xz, and "aRot" the rotation arround x, we have :
        We can write 
            M =  U D tV  =  U (SwXZ SwXZ) D (SwXZ aRot) t(aSwXZ aRot) tV
           =   (U SwXZ)  (SwXZ D SwXZ aRot)  t( V aSwXZ R)

        And  Sw D Sw aRot is what we want :
                              0 0  0
	   (Sw D Sw aRot) =   0 0 -1
	                      0 1  0
        
        Now  we must take into account sign ambiguity  in SVD , let S be any of the direct
	diagonal matrix with "1/-1" , we have also 

            M =  U SDS tV  =  US  D  t(VS) = 
           =   (U S SwXZ)  (SwXZ D SwXZ aRot)  t( V S aSwXZ R)

    */
    // static matrix will never be freed
    cMemManager::SetActiveMemoryCount(false);
    static tMat aMSwapXZ = M3x3FromLines(cPt3dr(0,0,1),cPt3dr(0,1,0 ),cPt3dr(1,0,0));
    static tMat aMRot90    = M3x3FromLines(cPt3dr(1,0,0),cPt3dr(0,0,-1),cPt3dr(0,1,0));

    static tMat aMId      = M3x3FromLines(cPt3dr(1,0,0),cPt3dr(0, 1,0),cPt3dr(0,0, 1));
    static tMat aMSymX    = M3x3FromLines(cPt3dr(1,0,0),cPt3dr(0,-1,0),cPt3dr(0,0,-1));
    static tMat aMSymY    = M3x3FromLines(cPt3dr(-1,0,0),cPt3dr(0, 1,0),cPt3dr(0,0,-1));
    static tMat aMSymZ    = M3x3FromLines(cPt3dr(-1,0,0),cPt3dr(0,-1,0),cPt3dr(0,0,1));
    static tMat aSVD0 = M3x3FromLines(cPt3dr(1,0,0),cPt3dr(0,1,0 ),cPt3dr(0,0,0));

    static tMat aSwR    = aMSwapXZ * aMRot90;
    cMemManager::SetActiveMemoryCount(true);


    cResulSVDDecomp<tREAL8> aSVD = mMat.SVD();

    tMat aMatU0 = aSVD.MatU() * aMSwapXZ;
    tMat aMatV0 = aSVD.MatV() * aSwR;

    // we want matrix to be direct and btw multiply by cste will not change coplanarity
    aMatU0.SetDirectBySign();
    aMatV0.SetDirectBySign();

    // test assumption on eigen dec + some matrix multiplication
    const auto & aEV = aSVD.SingularValues();

    if (aRef!=nullptr)
    {
       // eigen convention waranty >=0 
       for (int aK=0 ; aK<3 ; aK++)
           MMVII_INTERNAL_ASSERT_bench(aEV(aK) >= 0.0,"Matric organ in MatEss ");
      
       static bool First = true;
       // eigen waranty decreasing order (see bench on matrix)
       tREAL8  aDif  = std::abs((aEV(0)-aEV(1)) / (aEV(0)+aEV(1)));
       tREAL8  aZero = std::abs( aEV(2)         / (aEV(0)+aEV(1)));

       MMVII_INTERNAL_ASSERT_bench(aDif<1e-4,"Matric organ in MatEss ");
       MMVII_INTERNAL_ASSERT_bench(aZero<1e-4,"Matric organ in MatEss ");

       // Matrix that should result from a perefct eigen decomposition due to eigen convention
       // Matrix we want in current epipolar formalization
       tMat aSVD1 = M3x3FromLines(cPt3dr(0,0,0),cPt3dr(0,0,-1 ),cPt3dr(0,1,0));

       // test the multiplication we do to have matrix we want
       if (First)
       {
          tMat aTest = aMSwapXZ * aSVD0 * aMSwapXZ * aMRot90;
          MMVII_INTERNAL_ASSERT_bench(aTest.L2Dist(aSVD1)==0,"Matrix organization in MatEss ");
          First = false;
       }
    }

    cWhichMax<tPose,int> aBestPose(tPose::Identity(),-1);
    int aNb11 = 0; // number of test where have 100% ins the good direction
    for (int aKOri=0 ; aKOri< 2 ; aKOri++)
    {
       tMat aMatSign = (aKOri==1) ? aMId : aMSymX;
       tMat aMatU = aMatU0 * aMatSign ;
       tMat aMatV = aMatV0 ;
       // the matrix correspond to y1z2-y2z1 modified by signs
       tMat aSVD1 =  aMatSign * aMSwapXZ  * aSVD0 * aMSwapXZ * aMRot90;
       for (int aSignPt= -1 ; aSignPt<=1 ;  aSignPt+=2)
       {

          // test we can rerbuilt the matrix up to a scaling factor
          if (aRef!=nullptr)
          {

              // from the 2 matrix and the standar epip we reconstitue M
              tMat aRconst = aMatU *  aSVD1 * aMatV.Transpose();
              //  there is a scaling factor + an undefined sign, to we test both
              tREAL8 aDif1 = mMat.L2Dist( aRconst *  aEV(0));
              tREAL8 aDif2 = mMat.L2Dist( aRconst * (-aEV(0)));
              MMVII_INTERNAL_ASSERT_bench(std::min(aDif1,aDif2)<1e-5,"Matric Reconstution in EssMat");
	  }


          size_t aNbP = aHom.VDir1().size();
          size_t aNbPU = 0;
          size_t aNbPV = 0;

	  cPt3dr aPU(0,0,0); //Image center for first cam
	  cPt3dr aPV(aSignPt,0,0); // Image center for second cam

          for (size_t aKP=0 ; aKP<aNbP ; aKP++)
          {
	     // Direction in local repair of each camea
             cPt3dr  aDirU0  =  aHom.VDir1().at(aKP);
             cPt3dr  aDirV0  =  aHom.VDir2().at(aKP) ;

	     // Direction in epipolar repair
	     cPt3dr aDirU = aDirU0 * aMatU;
	     cPt3dr aDirV = aDirV0 * aMatV ;

             // In Bench mode test that bundle complies with essential equation, in repair init and epip
	     if (aRef)
	     {
                MMVII_INTERNAL_ASSERT_bench(std::abs(Scal(aDirU0,mMat*aDirV0))<1e-4 ,"Ess scal MatInit ");
                MMVII_INTERNAL_ASSERT_bench(std::abs(Scal(aDirU,aSVD1*aDirV))<1e-4 ,"Ess scal Mat Rec   ");
	     }

	     // bundles in epipolar repair
	     tSeg3dr aSegU(aPU,aPU+aDirU);
	     tSeg3dr aSegV(aPV,aPV+aDirV);

	     cPt3dr ABC;
	     BundleInters(ABC,aSegU,aSegV);
	     // on ground truth the bundle intersects perfectly
	     if ((aRef!=nullptr) && (std::abs(ABC.z())>=1e-4))
	     {
                // this bad intersection can in fact occur when bundle are sub-parallel
                MMVII_INTERNAL_ASSERT_bench( 1.0- std::abs(Cos(aDirU,aDirV)) <1e-3 ,"Ess Bundle Inter  ");
	     }
	     aNbPU += ABC.x() > 0;
	     aNbPV += ABC.y() > 0 ;
          }

	  aNb11 += (aNbPU==aNbP) && (aNbPV==aNbP);
	  tPose aSol(aMatU * aPV,cRotation3D<tREAL8>(aMatU * aMatV.Transpose(),false));
	  aBestPose.Add(aSol,aNbPU+aNbPV);
       }
    }

    if (aRef)
    {
       // with perfect data should have 1 and only 1 combination with perfect orient
       MMVII_INTERNAL_ASSERT_bench(aNb11==1 ,"Mat ess : number comb in good dir   ");

       const tPose& aPose =  aBestPose.IndexExtre();

       MMVII_INTERNAL_ASSERT_bench(aPose.Rot().Mat().L2Dist(aRef->Rot().Mat())<1e-4,"Disr Rot in Matess");
       MMVII_INTERNAL_ASSERT_bench(Norm2(VUnit(aRef->Tr()) -aPose.Tr())<1e-4,"Disr Tr in Matess");
    }

    return aBestPose.IndexExtre();
}

void Bench_MatEss(cParamExeBench & aParam)
{
    if (! aParam.NewBench("MatEss")) return;

    cTimerSegm * aTS = nullptr;
    if (aParam.Show())
    {
       aTS = new cTimerSegm(&cMMVII_Appli::CurrentAppli());
    }  
    for (int aNb=0 ; aNb<1 ; aNb++)
    {
        cCamSimul::BenchMatEss(aTS,false);
        cCamSimul::BenchMatEss(aTS,true);
    }

    delete aTS;
    aParam.EndBench();
}

}; // MMVII

