#include "MMVII_PCSens.h"
// #include "MMVII_BundleAdj.h"


/* We have the image formula w/o distorsion:


*/

namespace MMVII
{

bool BUGME = false;



/*
 *              (a b c) (x2)
 *              (d e f) (y2)
 *   (x1 y1 z1) (g h i) (z2)
 */
class cMatEssential
{
    public :
        cMatEssential(const cSetHomogCpleDir &,cLinearOverCstrSys<tREAL8> & aSys);
        ///  Sigma attenuates big error  E*S / (E+S)  => ~E in 0  , bound to S at infty
	tREAL8  Cost(const  cHomogCpleDir &,const tREAL8 & aSigma) const;
	tREAL8  AvgCost(const  cSetHomogCpleDir &,const tREAL8 & aSigma) const;

    private :
        cDenseMatrix<tREAL8> mMat;
};

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
    cLinearOverCstrSys<tREAL8> *  aSysL1 = AllocL1_Barrodale<tREAL8>(8);
    // cLinearOverCstrSys<tREAL8> *  aSysL1 = new cLeasSqtAA<tREAL8>(8);

    for (int aK1=0 ; aK1<(int)eProjPC::eNbVals ; aK1++)
    {
        for (int aK2=0 ; aK2<(int)eProjPC::eNbVals ; aK2++)
        {
aCpt++;
BUGME = (aCpt<=3);
            // cCamSimul * aCamSim = cCamSimul::Alloc2VIewTerrestrial(eProjPC(aK1),eProjPC(aK2),false);
            cCamSimul * aCamSim = cCamSimul::Alloc2VIewTerrestrial(eProjPC::eStenope,eProjPC::eStenope,false);

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

            cMatEssential aMatE(aSetD,*aSysL1);
	    
	    StdOut() << aCpt << " BenchMatEssBenchMatEss "  << aMatE.AvgCost(aSetD,1.0)  << "\n"; 
	    if (BUGME) {getchar();}

            delete aCamSim;
        }
    }

    delete aSysL1;
}

void Bench_MatEss(cParamExeBench & aParam)
{
    if (! aParam.NewBench("MatEss")) return;

    cCamSimul::BenchMatEss();

    aParam.EndBench();
}

/* ************************************** */
/*                                        */
/*         cMatEssential               */
/*                                        */
/* ************************************** */

cMatEssential::cMatEssential(const cSetHomogCpleDir & aSetD,cLinearOverCstrSys<tREAL8> & aSys) :
    mMat  (cPt2di(3,3))
{
     aSys.Reset();
     cDenseVect<tREAL8> aVect(8);
     tREAL8 aRHS;
     for (const auto & aCple : aSetD.mSetD)
     {
         aCple.SetVectMatEss(aVect,aRHS);
	 aSys.AddObservation(1.0,aVect,aRHS);
     }
     cDenseVect<tREAL8> aSol = aSys.Solve();

     SetLine(0,mMat,cPt3dr(aSol(0),aSol(1),aSol(2)));
     SetLine(1,mMat,cPt3dr(aSol(3),aSol(4),aSol(5)));
     SetLine(2,mMat,cPt3dr(aSol(6),aSol(7),    1.0));
}

tREAL8  cMatEssential::Cost(const  cHomogCpleDir & aCple,const tREAL8 & aSigma) const
{
   //  tP1 Mat P2 =0  
   cPt3dr aQ1 = VUnit(aCple.mP1 * mMat); // Q1 is orthognal to plane containing P2
   cPt3dr aQ2 = VUnit(mMat * aCple.mP2); // Q1 is orthognal to plane containing P2
					 //
   tREAL8 aD = (std::abs(Scal(aQ1,aCple.mP2)) + std::abs(Scal(aCple.mP1,aQ2))  ) / 2.0;

   if (BUGME)
   {
	   StdOut() << "DDd==" << aD  << aCple.mP1 << " " << aCple.mP2 << " p1Mp2=" <<   Scal(aCple.mP1,mMat * aCple.mP2)  
	   << " ML2=" <<mMat.L2Dist(cDenseMatrix<tREAL8>(3, eModeInitImage::eMIA_Null))<< "\n";
   }
   return (aD*aSigma) / (aD+aSigma);
}

tREAL8  cMatEssential::AvgCost(const  cSetHomogCpleDir & aSetD,const tREAL8 & aSigma) const
{
   tREAL8 aSom = 0.0;
   for (const auto & aCple : aSetD.mSetD)
       aSom += Cost(aCple,aSigma);

   return aSom / aSetD.mSetD.size();
}


}; // MMVII




