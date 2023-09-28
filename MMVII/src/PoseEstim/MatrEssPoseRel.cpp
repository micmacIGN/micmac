#include "MMVII_PCSens.h"
#include "MMVII_Tpl_Images.h"
// #include "MMVII_BundleAdj.h"


/* We have the image formula w/o distorsion:


*/

namespace MMVII
{

class cSetHomogCpleDir;  // store tie point as pairs of homologous directions
class cMatEssential;     // class for representing a matt ess, double rep :  vector 9 + matrix 3x3
class cCamSimul ;        //  Class for simulating position of set camera having visibility domain in common


// Vect being "a-plat" representation of matrix M, put in Vect the constraint P1 M P2
void SetVectMatEss(cDenseVect<tREAL8> & aVect,const cPt3dr& aP1,const cPt3dr& aP2);  
// transformate linear rep in a matrix
cDenseMatrix<tREAL8> Vect2MatEss(const cDenseVect<tREAL8> & aSol);
									    
//  Add in Sys all the constraint relative to direction of Sys
void  MatEssAddEquations(const cSetHomogCpleDir & aSetD,cLinearOverCstrSys<tREAL8> & aSys);


bool BUGME = false;

/**   Class for storing homologous point as unitary direction of bundles (one made
 *     its no longer needed to access to the internal calibration + the coordonates are bounded).
 *
 *     Optionnaly , for normalization, the point can be turned from a rotation 
 */

class cSetHomogCpleDir
{
     public :
        typedef cRotation3D<tREAL8> tRot;

	///  Create from image homologue + internal calibration
        cSetHomogCpleDir(const cSetHomogCpleIm &,const cPerspCamIntrCalib &,const cPerspCamIntrCalib &);

	/// make both normalization so that bundles are +- centered onK
	void NormalizeRot();

        const std::vector<cPt3dr>& VDir1() const; ///< Accessor
        const std::vector<cPt3dr>& VDir2() const; ///< Accessor
	void Show() const;

	  
	/// 4 bench :  Randomize the rotation 
	void RandomizeRot();
	///  4 bench : Put in current data an outlayer, in Dir1 or Dir2 (randomly), at random position
        void GenerateRandomOutLayer(double aAmpl);

     private :
	/// make one normalization so that bundles are +- centered onK
	void  NormalizeRot(tRot&,std::vector<cPt3dr> &);
	/// Transormate bundle and accuumlate to memorize transformation
	void  AddRot(const tRot&,tRot&,std::vector<cPt3dr> &);

	std::vector<cPt3dr>  mVDir1;
	std::vector<cPt3dr>  mVDir2;

	tRot                 mR1ToInit;
	tRot                 mR2ToInit;
};

/** Class to represent the result of essential matrix computation,
 * essentially 3x3 matrix and an eq
 */

class cMatEssential
{
    public :
        /** Constructor use a set of 3D homologous dir, a solver (L1/L2) , and the number, in [0,8] of the variable
	    that is arbirtrarily fixed */
        cMatEssential(const cSetHomogCpleDir &,cLinearOverCstrSys<tREAL8> & aSys,int aKFix);

        ///  Sigma attenuates big error  E*S / (E+S)  => ~E in 0  , bound to S at infty, if S<=0  return E
	tREAL8  Cost(const  cPt3dr & aP1,const  cPt3dr &aP2,const tREAL8 & aSigma) const;

	/// No more than average of Cost
	tREAL8  AvgCost(const  cSetHomogCpleDir &,const tREAL8 & aSigma) const;

	/// No more than 
	tREAL8  KthCost(const  cSetHomogCpleDir &,tREAL8 aProp) const;

	void Show(const cSetHomogCpleDir &) const;

        cMatEssential(const  cDenseMatrix<tREAL8>& aMat);

         void DoSVD() const;
 
    private :
        cDenseMatrix<tREAL8> mMat; /// The Ess  matrix itself
};



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
        
         // StdOut() << " Dir="  << mVDir1[aK]  << mVDir2[aK] << "\n";
	 aS1 += mVDir1[aK];
	 aS2 += mVDir2[aK];
	 aSomDif += Norm2(mVDir1[aK]-mVDir2[aK]);
     }
     StdOut() << " ** AVG-Dirs="  << VUnit(aS1)  <<  VUnit(aS2)  << " DIFS=" << aSomDif/mVDir1.size() << "\n";
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
	 aSys.AddObservation(1.0,aVect,0.0);
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

int   GetKMax(const cSetHomogCpleDir & aSetD,tREAL8 aWeightStab,bool Show=false)
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
           StdOut() << "\n";
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
	 aSys.AddObservation(1.0,aVect,0.0);
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
	StdOut() << "\n";
    }
    StdOut() << "     Cost=" << AvgCost(aSetD,1.0) << "\n";
    cResulSVDDecomp<tREAL8>  aRSVD =  mMat.SVD();
    cDenseVect<tREAL8>       aVP = aRSVD.SingularValues();
    StdOut() <<  "EIGEN-VAL " << aVP(0) << " " << aVP(1) << " " << aVP(2) << "\n";
    StdOut() << "============================================\n";
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


void cMatEssential::DoSVD() const
{
    cResulSVDDecomp<tREAL8> aSVD = mMat.SVD();

    const auto & aEV = aSVD.SingularValues();
    cWhichMin<int,tREAL8> aMinAbsInd(-1,1e30);

    for (int aK=0 ; aK<3 ; aK++)
    {
        aMinAbsInd.Add(aK,std::abs(aEV(aK)));
    }
    


    StdOut() << "KM= " << aMinAbsInd.IndexExtre() << "\n";
    StdOut() << "EV=" << aEV(0) << " " << aEV(1) << " " << aEV(2) << "\n";
    StdOut() << "Det=" << aSVD.MatU().Det() << " " << aSVD.MatV().Det() << "\n";
    StdOut() << "\n";
}

/* ************************************** */
/*                                        */
/*            cCamSimul                   */
/*                                        */
/* ************************************** */

class cCamSimul : public cMemCheck
{
   public :
      static cCamSimul * Alloc2VIewTerrestrial(eProjPC aProj1,eProjPC aProj2,bool SubVert);

      ~cCamSimul();

      cPt3dr mCenterGround;
     
      //   Geometry of acquisition
      tREAL8 mProfMin;
      tREAL8 mProfMax;
      tREAL8 mBsHMin;
      tREAL8 mBsHMax;
      tREAL8 mRandInterK;

      static void BenchMatEss(cTimerSegm * aTS,bool PerfInter);

      void TestCam(cSensorCamPC * aCam) const;
   private :
      void AddCam(cPerspCamIntrCalib *,bool SubVert);
      void AddCam(eProjPC aProj1,bool SubVert);

      cCamSimul();
      ///  is the new center sufficiently far, but not too much
      bool ValidateCenter(const cPt3dr & aP) const;

      ///  Generatea new valide point
      cPt3dr  GenValideCenter(bool SubVert) const;
      /// Generate a point w/o constraint
      cPt3dr  GenAnyCenter(bool SubVert) const;

      std::vector<cSensorCamPC *>         mListCam;
      std::vector<cPerspCamIntrCalib *>   mListCalib;


      // cSetHomogCpleIm
};


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

cPt3dr  cCamSimul::GenAnyCenter(bool SubVert) const
{
    if (SubVert)
       return    mCenterGround 
	       + cPt3dr(RandUnif_C()/mProfMax,RandUnif_C()/mProfMax,1.0) * RandInInterval(mProfMin,mProfMax);

    return mCenterGround + cPt3dr::PRandUnit() * RandInInterval(mProfMin,mProfMax);
}


cPt3dr  cCamSimul::GenValideCenter(bool SubVert) const
{
   cPt3dr aRes = GenAnyCenter(SubVert);
   while (! ValidateCenter(aRes))
          aRes = GenAnyCenter(SubVert);
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
	StdOut() << "CC " << aCam->Center()  << " CG=" << mCenterGround << "\n";

cPt3dr aV = aCam->Center() - mCenterGround;

StdOut()  << " I " << Cos(aV,aCam->AxeI())
          << " J " << Cos(aV,aCam->AxeI())
          << " K " << Cos(aV,aCam->AxeK())
	  << " \n";

	StdOut() << "Vis " <<  aCam->IsVisible(mCenterGround) << "\n";
}

void cCamSimul::BenchMatEss
     (
        cTimerSegm * aTS,
        bool         PerfInter
     )
{
    static int aCpt=0;
    /// cLinearOverCstrSys<tREAL8> *  aSysL1 = AllocL1_Barrodale<tREAL8>(9);
    // cLinearOverCstrSys<tREAL8> *  aSysL1 = new cLeasSqtAA<tREAL8>(9);
    cLeasSqtAA<tREAL8> aSysL2(9);

    static int aCptPbL1 = 0;

    for (int aK1=0 ; aK1<(int)eProjPC::eNbVals ; aK1++)
    {
        for (int aK2=0 ; aK2<(int)eProjPC::eNbVals ; aK2++)
        {
            cAutoTimerSegm aTSSim(aTS,"CreateSimul");
            aCpt++;
            cCamSimul * aCamSim = cCamSimul::Alloc2VIewTerrestrial(eProjPC(aK1),eProjPC(aK2),false);

            // we want to test robustness in perfect degenerate & close to degenertae
            if (PerfInter)
               aCamSim->mRandInterK = 0.0;

            // Generate 2 cams 
	    cSensorCamPC * aCam1 = aCamSim->mListCam.at(0);
	    cSensorCamPC * aCam2 = aCamSim->mListCam.at(1);

            // generate  perfect homologous point
	    cSetHomogCpleIm aSetH;
	    size_t aNbPts = 40;
	    for (size_t aKP=0 ; aKP<aNbPts ; aKP++)
	    {
		 aSetH.Add(aCam1->RandomVisibleCple(*aCam2));
	    }

            // Make 3D direction of points
	    cSetHomogCpleDir aSetD (aSetH,*(aCam1->InternalCalib()),*(aCam2->InternalCalib()));

            cAutoTimerSegm aTSGetMax(aTS,"GetMaxK");
            int aKMax =  GetKMax(aSetD,1e-6);

            // These point where axe k almost intersect, the z1z2 term of mat ess is probably small
            // and must not be KMax
            MMVII_INTERNAL_ASSERT_bench(aKMax!=8,"cComputeMatEssential::GetKMax");
            
            // Now test that residual is ~ 0 on these perfect points
            cAutoTimerSegm aTSL2(aTS,"L2");
            cMatEssential aMatEL2(aSetD,aSysL2,aKMax);

            {
                cIsometry3D<tREAL8>  aPRel =  aCam1->RelativePose(*aCam2);
                StdOut() << "TR-REL " << aPRel.Tr() << "\n";
                aMatEL2.DoSVD();
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


	    if (BUGME) 
	    {
/*
                StdOut()  <<  "----------------KkkkkkkkkkkkMmmmmmmax \n";
                // cComputeMatEssential aCME;
                // aCME.GetKMax(aSetD,1e-6);

                aSetD.Show();
                aMatE.Show(aSetD);


	        aSetD.NormalizeRot();
                aSetD.Show();
                cMatEssential aMNorm(aSetD,*aSysL1,0);
		aMNorm.Show(aSetD);

	        aSetD.RandomizeRot();
                aSetD.Show();
                cMatEssential aMRand(aSetD,*aSysL1,0);
		aMRand.Show(aSetD);
*/
		/*
		*/

		StdOut() << "***************************************************\n";
                getchar();
	    }

	// void NormalizeRot();
	/// Randomize the rotation , for bench
	// void RandomizeRot();


            //TestRotMatEss(aSetD,aMatE,aSysL1);

            delete aCamSim;
        }
    }
    StdOut() << "CPT " << aCptPbL1  << " " << aCpt << "\n";

    // delete aSysL1;
}

#if (0)
/**  Check the impact of rotation on ess matrix
 */
void TestRotMatEss(const cSetHomogCpleDir & aSetD,const cMatEssential aMatE,cLinearOverCstrSys<tREAL8> * aSysL1)
{
    cSetHomogCpleDir aSetBis;
    cRotation3D<tREAL8>  aRot = cRotation3D<tREAL8>::RandomRot();
    for (const auto & aCple : aSetD.mSetD)
    {
           aSetBis.mSetD.push_back(cHomogCpleDir(aRot.Value(aCple.mP1),aRot.Value(aCple.mP2)));
    }
    cMatEssential aMat2(aSetBis,*aSysL1);
    aMatE.Show();
    StdOut() << "\n";
    aMat2.Show();
    getchar();
}
#endif


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

   /* ********************************************************** */
   /*                                                            */
   /*                 cAppliBundlAdj                             */
   /*                                                            */
   /* ********************************************************** */

class cAppli_MatEss : public cMMVII_Appli
{
     public :
        cAppli_MatEss(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :
        cPhotogrammetricProject   mPhProj;
	std::string               mSpecImIn;
};

cCollecSpecArg2007 & cAppli_MatEss::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
              << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
              <<  mPhProj.DPOrient().ArgDirInMand()
           ;
}



}; // MMVII




