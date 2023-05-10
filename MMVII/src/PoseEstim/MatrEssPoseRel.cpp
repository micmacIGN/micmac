#include "MMVII_PCSens.h"
// #include "MMVII_BundleAdj.h"


/* We have the image formula w/o distorsion:


*/

namespace MMVII
{

class cHomogCpleIm
{
      public :
           cHomogCpleIm(const cPt2dr &,const cPt2dr &);
           cPt2dr  mP1;
           cPt2dr  mP2;
};

class cSetHomogCpleIm
{
      public :
        std::vector<cHomogCpleIm>  mSetH;
};

class cHomogCpleDir
{
      public :
           cHomogCpleDir(const cPt3dr & aP1,const cPt3dr & aP2);
           void SetVectMatEss(cDenseVect<tREAL8> &,tREAL8 & aRHS) const;
           cPt3dr  mP1;
           cPt3dr  mP2;
};


class cSetHomogCpleDir
{
      public :
        std::vector<cHomogCpleDir>  mSetD;
	cSetHomogCpleDir(const cSetHomogCpleIm &,const cPerspCamIntrCalib &,const cPerspCamIntrCalib &);
};


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
      cRotation3D<tREAL8> aRot= cRotation3D<tREAL8>::CompleteRON(aK,aI);

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

void cCamSimul::BenchMatEss()
{
    for (int aK1=0 ; aK1<(int)eProjPC::eNbVals ; aK1++)
    {
        for (int aK2=0 ; aK2<(int)eProjPC::eNbVals ; aK2++)
        {
            cCamSimul * aCamSim = cCamSimul::Alloc2VIewTerrestrial(eProjPC(aK1),eProjPC(aK2),false);

            delete aCamSim;
        }
    }
}

void Bench_MatEss(cParamExeBench & aParam)
{
    if (! aParam.NewBench("MatEss")) return;

    cCamSimul::BenchMatEss();

    aParam.EndBench();
}

/* ************************************** */
/*                                        */
/*            cHomogCpleDir               */
/*                                        */
/* ************************************** */

cHomogCpleDir::cHomogCpleDir(const cPt3dr & aP1,const cPt3dr & aP2) :
   mP1  (VUnit(aP1)),
   mP2  (VUnit(aP2))
{
}

void cHomogCpleDir::SetVectMatEss(cDenseVect<tREAL8> &aVect,tREAL8 & aRHS) const
{
         aVect(0) = mP1.x() *  mP2.x();
         aVect(1) = mP1.x() *  mP2.y();
         aVect(2) = mP1.x() *  mP2.z();

         aVect(3) = mP1.y() *  mP2.x();
         aVect(4) = mP1.y() *  mP2.y();
         aVect(5) = mP1.y() *  mP2.z();

         aVect(6) = mP1.z() *  mP2.x();
         aVect(7) = mP1.z() *  mP2.y();
         aRHS    = -mP1.z() *  mP2.z();
}

/* ************************************** */
/*                                        */
/*         cSetHomogCpleDir               */
/*                                        */
/* ************************************** */

cSetHomogCpleDir::cSetHomogCpleDir
(
     const cSetHomogCpleIm &    aSetH,
     const cPerspCamIntrCalib & aCal1,
     const cPerspCamIntrCalib & aCal2
) 
{
     for (const auto & aCplH : aSetH.mSetH)
     {
         cPt3dr aP1 =  aCal1.DirBundle(aCplH.mP1);
         cPt3dr aP2 =  aCal2.DirBundle(aCplH.mP2);
	 mSetD.push_back(cHomogCpleDir(aP1,aP2));
     }
}

/* ************************************** */
/*                                        */
/*         cSetHomogCpleDir               */
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




