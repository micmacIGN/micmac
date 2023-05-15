#include "MMVII_PCSens.h"
// #include "MMVII_BundleAdj.h"


/* We have the image formula w/o distorsion:


*/

namespace MMVII
{

bool BUGME = false;

/* ************************************** */
/*                                        */
/*         cSetHomogCpleDir               */
/*                                        */
/* ************************************** */

class cSetHomogCpleDir
{
     public :
        typedef cRotation3D<tREAL8> tRot;

	///  Create from image homologue + internal calibration
        cSetHomogCpleDir(const cSetHomogCpleIm &,const cPerspCamIntrCalib &,const cPerspCamIntrCalib &);

	/// make both normalization so that bundles are +- centered onK
	void NormalizeRot();
	/// Randomize the rotation , for bench
	void RandomizeRot();

        const std::vector<cPt3dr>& VDir1() const; ///< Accessor
        const std::vector<cPt3dr>& VDir2() const; ///< Accessor


	void Show() const;

     private :
	/// make one normalization so that bundles are +- centered onK
	void  NormalizeRot(tRot&,std::vector<cPt3dr> &);
	/// Transormate bundle and accuumlate to memorize transformation
	void  AddRot(const tRot&,tRot&,std::vector<cPt3dr> &);

	tRot                 mR1ToInit;
	std::vector<cPt3dr>  mVDir1;
	tRot                 mR2ToInit;
	std::vector<cPt3dr>  mVDir2;
};


cSetHomogCpleDir::cSetHomogCpleDir(const cSetHomogCpleIm & aSetH,const cPerspCamIntrCalib & aCal1,const cPerspCamIntrCalib & aCal2) :
    mR1ToInit (tRot::Identity()),
    mR2ToInit (tRot::Identity())
{
     for (const auto & aCplH : aSetH.mSetH)
     {
         mVDir1.push_back(VUnit(aCal1.DirBundle(aCplH.mP1)));
         mVDir2.push_back(VUnit(aCal2.DirBundle(aCplH.mP2)));
     }
}

const std::vector<cPt3dr>& cSetHomogCpleDir::VDir1() const {return mVDir1;}
const std::vector<cPt3dr>& cSetHomogCpleDir::VDir2() const {return mVDir2;}

void  cSetHomogCpleDir::NormalizeRot(cRotation3D<tREAL8>&  aRot ,std::vector<cPt3dr> & aVPts)
{
   
      // cPt3dr aP = cComputeCentroids<std::vector<cPt3dr> >::StdRobustCentroid(aVPts,0.5,2);
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
/*
 *              (a b c) (x2)
 *              (d e f) (y2)
 *   (x1 y1 z1) (g h i) (z2)
 */

/* ************************************** */
/*                                        */
/*         cMatEssential                  */
/*                                        */
/* ************************************** */

class cMatEssential
{
    public :
        cMatEssential(const cSetHomogCpleDir &,cLinearOverCstrSys<tREAL8> & aSys,int aKFix);
        ///  Sigma attenuates big error  E*S / (E+S)  => ~E in 0  , bound to S at infty
	tREAL8  Cost(const  cPt3dr & aP1,const  cPt3dr &aP2,const tREAL8 & aSigma) const;
	tREAL8  AvgCost(const  cSetHomogCpleDir &,const tREAL8 & aSigma) const;

	void Show(const cSetHomogCpleDir &) const;

	void SetVectMatEss(const cPt3dr& aP1,const cPt3dr& aP2);
    private :
        cDenseVect<tREAL8>   mVect;
        cDenseMatrix<tREAL8> mMat;
};


cMatEssential::cMatEssential(const cSetHomogCpleDir & aSetD,cLinearOverCstrSys<tREAL8> & aSys,int aKFix) :
    mVect (9), 
    mMat  (cPt2di(3,3))
{
     aSys.Reset();

     const std::vector<cPt3dr>&  aVD1 = aSetD.VDir1() ;
     const std::vector<cPt3dr>&  aVD2 = aSetD.VDir2() ;
     for (size_t aKP=0 ; aKP<aVD1.size() ; aKP++)
     {
         SetVectMatEss(aVD1[aKP],aVD2[aKP]);
	 aSys.AddObservation(1.0,mVect,0.0);
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
   cPt3dr aQ1 = VUnit(aP1 * mMat); // Q1 is orthognal to plane containing P2
   cPt3dr aQ2 = VUnit(mMat * aP2); // Q1 is orthognal to plane containing P2
					 //
   tREAL8 aD = (std::abs(Scal(aQ1,aP2)) + std::abs(Scal(aP1,aQ2))  ) / 2.0;

   if (false)
   {
	   StdOut() << "DDd==" << aD  << aP1 << " " << aP2 << " p1Mp2=" <<   Scal(aP1,mMat * aP2)  
	   << " ML2=" <<mMat.L2Dist(cDenseMatrix<tREAL8>(3, eModeInitImage::eMIA_Null))<< "\n";
   }
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

void cMatEssential:: SetVectMatEss(const cPt3dr& aP1,const cPt3dr& aP2)
{
     mVect(0) = aP1.x() *  aP2.x();
     mVect(1) = aP1.x() *  aP2.y();
     mVect(2) = aP1.x() *  aP2.z();

     mVect(3) = aP1.y() *  aP2.x();
     mVect(4) = aP1.y() *  aP2.y();
     mVect(5) = aP1.y() *  aP2.z();

     mVect(6) = aP1.z() *  aP2.x();
     mVect(7) = aP1.z() *  aP2.y();
     mVect(8) = aP1.z() *  aP2.z();
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


      static void BenchMatEss();

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
   mBsHMax       (0.5)
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

void cCamSimul::BenchMatEss()
{
    static int aCpt=0;
    /// cLinearOverCstrSys<tREAL8> *  aSysL1 = AllocL1_Barrodale<tREAL8>(9);
    cLinearOverCstrSys<tREAL8> *  aSysL1 = new cLeasSqtAA<tREAL8>(9);

    for (int aK1=0 ; aK1<(int)eProjPC::eNbVals ; aK1++)
    {
        for (int aK2=0 ; aK2<(int)eProjPC::eNbVals ; aK2++)
        {
             aCpt++;
BUGME = true;
// BUGME = (aCpt<=3);
            cCamSimul * aCamSim = cCamSimul::Alloc2VIewTerrestrial(eProjPC(aK1),eProjPC(aK2),false);
            // cCamSimul * aCamSim = cCamSimul::Alloc2VIewTerrestrial(eProjPC::eStenope,eProjPC::eStenope,false);

	    cSensorCamPC * aCam1 = aCamSim->mListCam.at(0);
	    cSensorCamPC * aCam2 = aCamSim->mListCam.at(1);


	    // aCamSim->TestCam(aCam1);
	    // aCamSim->TestCam(aCam2);
	    // StdOut() <<  aCam1->IsVisible(aCamSim->mCenterGround)  << " " << aCam2->IsVisible(aCamSim->mCenterGround) << "\n";
	    // StdOut() << aCam1->AxeK() << " " <<  aCam2->AxeK() << "\n";

	    cSetHomogCpleIm aSetH;
	    size_t aNbPts = 20;
	    for (size_t aKP=0 ; aKP<aNbPts ; aKP++)
	    {
		 aSetH.mSetH.push_back(aCam1->RandomVisibleCple(*aCam2));
	    }

	    cSetHomogCpleDir aSetD (aSetH,*(aCam1->InternalCalib()),*(aCam2->InternalCalib()));
            cMatEssential aMatE(aSetD,*aSysL1,0);
	    
	    StdOut() << "Cpt=" << aCpt 
		     << " Cost= "  << aMatE.AvgCost(aSetD,1.0)  
		     // << " Disp= "  << aSetD.Disp()
		    << "\n"; 

	    if (BUGME) 
	    {
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

    delete aSysL1;
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

    for (int aNb=0 ; aNb<1 ; aNb++)
        cCamSimul::BenchMatEss();

    aParam.EndBench();
}

}; // MMVII




