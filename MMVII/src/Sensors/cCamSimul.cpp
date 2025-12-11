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



class cPS_CompPose
{
   public :
        cPS_CompPose( cSetHomogCpleDir &);

   private :
        static void SetPt( cDenseVect<tREAL8>&,size_t aIndex,const cPt3dr&,tREAL8 aMul){}

};

cPS_CompPose::cPS_CompPose( cSetHomogCpleDir & aSetCple)
{
    const std::vector<cPt3dr>& aV1 = aSetCple.VDir1();
    const std::vector<cPt3dr>& aV2 = aSetCple.VDir2();

    cDenseVect<tREAL8> aVect(9);
    cLeasSqtAA<tREAL8> aSys(9);

    for (size_t aKP=0 ; aKP<aV1.size() ; aKP++)
    {
        const cPt3dr & aP1 = aV1.at(aKP);
        const cPt3dr & aP2 = aV2.at(aKP);

        SetPt(aVect,0,aP1,aP2.z());
        SetPt(aVect,3,aP1,0);
        SetPt(aVect,6,aP1,-aP2.x());
        aSys.PublicAddObservation(1.0,aVect,0.0);

        SetPt(aVect,0,aP1,0);
        SetPt(aVect,3,aP1,aP2.z());
        SetPt(aVect,6,aP1,-aP2.y());
        aSys.PublicAddObservation(1.0,aVect,0.0);
    }
    aSys.AddObsFixVar(tREAL8(aV1.size()),8,1.0);
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
               StdOut() << " Planaaarr " << isPlanar << " K=" << aKP << "\n";
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
                 cPS_CompPose aPsC(aSetD);
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




