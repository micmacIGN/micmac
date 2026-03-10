#include "MMVII_PoseRel.h"
#include "MMVII_Tpl_Images.h"
#include "../Graphs/ArboTriplets.h"

#include <unordered_set>

/* We have the image formula w/o distorsion:


*/

namespace MMVII
{

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


void cCamSimul::AddCam(cPerspCamIntrCalib * aPC,bool SubVert,std::string Name)
{
      cPt3dr aNewC = GenValideCenter(SubVert);

      //  Axe K will point to the center of the scene
      cPt3dr aK = VUnit(mCenterGround - aNewC);
      // generate random I orthog to K
      cPt3dr aI = cPt3dr::PRandUnitNonAligned(aK,1e-2);
      // complete
      cPt3dr aJ = VUnit(aK ^aI);
      aI = aJ ^aK; // Just in case

      // we have now a rotation
      cRotation3D<tREAL8> aRot(M3x3FromCol(aI,aJ,aK),false);

      // if we add a small noise to not have a perfect intersec
      aNewC += cPt3dr::PRandC() * mRandInterK;
      // now we have a pose
      cIsometry3D<tREAL8> aPose(aNewC,aRot);

      // now we have a Cam
      mListCam.push_back(new cSensorCamPC(Name,aPose,aPC));
}

void cCamSimul::AddCam(eProjPC aProj,bool SubVert,std::string Name)
{
    // 1 => means Deg of direct dist is 2 (dir inverse is 5,1,1)
    cPerspCamIntrCalib * aCalib = cPerspCamIntrCalib::RandomCalib(aProj,1);

    mListCalib.push_back(aCalib);
    AddCam(aCalib,SubVert,Name);
}

cCamSimul * cCamSimul::Alloc2VIewTerrestrial(eProjPC aProj1,eProjPC aProj2,bool SubVert)
{
   cCamSimul * aRes = new cCamSimul();

   aRes->AddCam(aProj1,SubVert);
   aRes->AddCam(aProj2,SubVert);

   return aRes;
}

cCamSimul * cCamSimul::AllocNVIewTerrestrial(int aNb,eProjPC aProj,bool SubVert)
{
    cCamSimul * aRes = new cCamSimul();

    for (int aK=0; aK<aNb; aK++)
    {
        aRes->AddCam(aProj,SubVert,"SimCam"+ToStr(aK));
    }

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

void BenchMEP_Coplan();

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

    thread_local static int aCptPbL1 = 0; FakeUseIt(aCptPbL1);


    if (1)
    {

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
		       // To see, not sure validate any more
		       /*
                    cPSC_PB aParam("Cam");
                    cPS_CompPose aPsC(aSetD,&aParam);
		    */
               }
               else if (false)
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
                         << "\n" ;
                     MMVII_INTERNAL_ASSERT_bench(aMatNoise.KthCost(aSetD,0.70) <1e-5,"Kth cost ");
                     MMVII_INTERNAL_ASSERT_bench(aMatNoise.KthCost(aSetD,0.95) >1e-2,"Kth cost ");
                 }

                 // We test if the residual at 70% is almost 0 (with 4/40 outlayers)
                 if (aMatNoise.KthCost(aSetD,0.70)>1e-5)
                      aCptPbL1++;
               }
               delete aCamSim;
            }
        }
    }

    BenchMEP_Coplan();
}


struct TripletHash {
    std::size_t operator()(const std::array<int,3>& t) const noexcept {
        std::size_t h1 = std::hash<int>{}(t[0]);
        std::size_t h2 = std::hash<int>{}(t[1]);
        std::size_t h3 = std::hash<int>{}(t[2]);

        // hash combine
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

void Bench_HBA(cParamExeBench & aParam)
{
    if (! aParam.NewBench("HierarchBA")) return;

    cTimerSegm * aTS = nullptr;
    if (aParam.Show())
    {
        aTS = new cTimerSegm(&cMMVII_Appli::CurrentAppli());
    }

    cCamSimul::BenchHierchBA(aTS,true,false);

    delete aTS;
    aParam.EndBench();
}

void cCamSimul::BenchHierchBA(cTimerSegm * aTS,
                              bool         PerfInter,
                              bool         isSubVert)
{
    // syntethic triplets and tie points
    int aNbCam = 25;
    int aNbTri = round_ni((aNbCam-1)/2 * 5);
    int aNbHPts = 20; // per pair of images (approx)

    // tie points noise setting
    double aPtNoiseAmpl=0.5; //in pixel

    // tie points outlier settings
    double aPtOutlierAmpl=20.0;// in pixel
    double aPtOutlierRate=0.1;

    // triplet outlier settings
    double aTriOutNb = aNbTri * 0.2;
    std::vector<double> aTriOutAmpl({0.5,0.1});// [Tr,Rot]




    StdOut() << "Nb of cams=" << aNbCam << ", nb of triplets=" << aNbTri << std::endl;

    std::vector<int> aProjTest{0,1,2,3,4,5};
    for (int aK1=0 ; aK1<aProjTest.size(); aK1++) //(int)eProjPC::eNbVals
    {
        StdOut() << "Projection " << aK1 << " " << E2Str(eProjPC(aProjTest[aK1]));
        getchar();

        std::vector<std::string> aSetIm; // set of images that are part of triplet graph
        std::vector<std::pair<int,int>> aEdges; // to make sure triplets form a connected component
        std::vector<bool> aCamVisited(aNbCam,false);
        std::unordered_set<std::array<int,3>, TripletHash> aTriplets;

        std::unique_ptr<cCamSimul> aCamSim(cCamSimul::AllocNVIewTerrestrial(aNbCam,eProjPC(aProjTest[aK1]),isSubVert));

        if (0)
        {
            // dirty code to save generated poses
            cMMVII_Appli &  anApTest = cMMVII_Appli::CurrentAppli();
            cPhotogrammetricProject aPhPSave(anApTest);
            aPhPSave.FinishInit();
            const std::string aNameOriTest = "TESCIK";
            aPhPSave.DPOrient().SetDirOut(aNameOriTest);
            for (auto aC : aCamSim->mListCam)
                aPhPSave.SaveCamPC(*aC);
            getchar();
        }

        // update PhotogrammetricProjectMemory
        cMMVII_Appli &  anAp = cMMVII_Appli::CurrentAppli();

        // populate in-memory PhProj with calibrations for all simulated cameras
        cPhotogrammetricProjectMemory aMemPhProj;
        for (int aKIm=0 ; aKIm<aNbCam ; aKIm++)
        {
            aMemPhProj.AddCalib(aCamSim->mListCam.at(aKIm)->NameImage(),
                                aCamSim->mListCam.at(aKIm)->InternalCalib());
        }

        // we want to test robustness in perfect degenerate & close to degenertae
        if (PerfInter)
            aCamSim->mRandInterK = 0.0;


        // first triplet
        while (aTriplets.size()==0)
        {
            // first triplet's id
            std::vector<int> aFirstTri;
            aFirstTri = RandSet(3,aNbCam);

            std::array<int,3> t = {aFirstTri[0],aFirstTri[1],aFirstTri[2]};
            std::sort(t.begin(), t.end());

            // each node is distinct
            if (t[0] == t[1] || t[0] == t[2] || t[1] == t[2])
                continue;

            aTriplets.insert(t);

            // update variable keeping track of already visited cameras
            for (int node : t)
            {
                aCamVisited[node] = 1;
                aEdges.push_back(std::make_pair(t[0],t[1]));
                aEdges.push_back(std::make_pair(t[1],t[2]));
                aEdges.push_back(std::make_pair(t[2],t[0]));

                StdOut() << "\t **** New node " << node << std::endl;
            }
        }

        // keep generating triplets
        //   (not all nodes are forced to be visited but it's fine
        //    as long as the graph is connected ie triplet connected by an edge)
        while (aTriplets.size() < aNbTri)
        {
            // current triplet's id
            std::vector<int> aCurTri;

            // draw 2 nodes from existing nodes to ensure connectivity
            int anEdgeId = RandUnif_N(aEdges.size()-1);
            aCurTri.push_back(aEdges[anEdgeId].first);
            aCurTri.push_back(aEdges[anEdgeId].second);

            // draw one node from all nodes
            aCurTri.push_back(RandUnif_N(aNbCam-1));

            // move to array to make triplet search faster
            std::array<int,3> t = {aCurTri[0],aCurTri[1],aCurTri[2]};
            std::sort(t.begin(), t.end());

            // ensure distinctiveness
            if (t[0] == t[1] || t[0] == t[2] || t[1] == t[2])
                continue;


            // see if the triplet already exists
            auto [it, inserted] = aTriplets.insert(t);

            // if new triplet
            if (inserted)
            {
                StdOut() << "New triplet " << aTriplets.size() << ": " << t[0] << " " << t[1] << " " << t[2] << std::endl;

                for (int node : t)
                {
                    // if this camera has not been visited
                    //if (!aCamVisited[node]) //not needed and could cause trouble
                    {
                        aCamVisited[node] = 1;
                        aEdges.push_back(std::make_pair(t[0],t[1]));
                        aEdges.push_back(std::make_pair(t[1],t[2]));
                        aEdges.push_back(std::make_pair(t[2],t[0]));

                        StdOut() << "\t **** New node " << node << std::endl;

                    }
                }
            }

        }

        // update aSetIm necessary for retrieving tie points structure in photogrammetric project
        for (int aKCV=0; aKCV<aCamVisited.size(); aKCV++)
        {
            if (aCamVisited[aKCV]==true)
                aSetIm.push_back(aCamSim->mListCam[aKCV]->NameImage());
        }
        std::sort(aSetIm.begin(),aSetIm.end());

        // generate homologous points
        StdOut() << "Generate tie points" << std::endl;
        //cPhotogrammetricProjectMemory::AddMulTieP(const std::string & aNameIm,
        //                                        const cVecTiePMul & aVec)

        // local map of features that will be later used to update cPhotogrammetricProjectMemory
        std::map<std::string, cVecTiePMul> aMulTiePMap;
        for (auto aCam : aCamSim->mListCam)
        {
            aMulTiePMap[aCam->NameImage()] = cVecTiePMul();
        }

        // iterate over all triplets
        int aPtImIdx=0;
        for (auto& aT : aTriplets)
        {
            // get cams
            std::vector<cSensorCamPC *> aCams;
            aCams.push_back(aCamSim->mListCam.at(aT[0]));
            aCams.push_back(aCamSim->mListCam.at(aT[1]));
            aCams.push_back(aCamSim->mListCam.at(aT[2]));

            // all possible pairs
            int aNbKeyPtsInTri=0;
            for (int aK1Cam=0; aK1Cam<aCams.size(); aK1Cam++)
            {
                for (int aK2Cam=aK1Cam+1; aK2Cam<aCams.size(); aK2Cam++)
                {
                    // generate aNbHPts points
                    for (int aKPt=0; aKPt<aNbHPts; aKPt++)
                    {
                        cHomogCpleIm aHPair = aCams[aK1Cam]->RandomVisibleCple(*aCams[aK2Cam],1000);

                        // get 3D
                        cPt3dr aPt3D = aCams[aK1Cam]->PInterBundle(aHPair,*aCams[aK2Cam]);

                        // check if visible in 3rd image (if so, multiple pt)
                        for (int aK3Cam=0; aK3Cam<aCams.size(); aK3Cam++)
                        {
                            if ((aK3Cam!=aK1Cam) && (aK3Cam!=aK2Cam))
                            {
                                if (aCams[aK3Cam]->IsVisible(aPt3D))
                                {
                                    // save to structure
                                    cPt2dr aPt = aCams[aK3Cam]->Ground2Image(aPt3D);
                                    cPt2dr aPtDelta = cPt2dr::PRandC()*aPtNoiseAmpl; // add noise
                                    // does it still project to camera?
                                    if (aCams[aK3Cam]->IsVisibleOnImFrame(aPt+aPtDelta))
                                        aPt = aPt + aPtDelta;

                                    cTiePMul aPtCam3(aPt,aPtImIdx);
                                    aMulTiePMap[aCams[aK3Cam]->NameImage()].mVecTPM.push_back(aPtCam3);
                                    aNbKeyPtsInTri++;
                                }
                            }
                        }

                        // add noise
                        cPt2dr aPt1Delta = cPt2dr::PRandC()*aPtNoiseAmpl;
                        cPt2dr aPt2Delta = cPt2dr::PRandC()*aPtNoiseAmpl;

                        // check if visible in image with added noise
                        if (aCams[aK1Cam]->IsVisibleOnImFrame(aHPair.mP1+aPt1Delta))
                            aHPair.mP1 = aHPair.mP1 + aPt1Delta;
                        if (aCams[aK2Cam]->IsVisibleOnImFrame(aHPair.mP2+aPt2Delta))
                            aHPair.mP2 = aHPair.mP2 + aPt2Delta;

                        // save to structure
                        cTiePMul aPtCam1(aHPair.mP1,aPtImIdx);
                        cTiePMul aPtCam2(aHPair.mP2,aPtImIdx);
                        aNbKeyPtsInTri++;
                        aNbKeyPtsInTri++;

                        aMulTiePMap[aCams[aK1Cam]->NameImage()].mVecTPM.push_back(aPtCam1);
                        aMulTiePMap[aCams[aK2Cam]->NameImage()].mVecTPM.push_back(aPtCam2);

                        aPtImIdx++;
                    }
                }
            }

            // add outliers
            //   note1: must be integrated here to check if visible in frame
            //   note2: coded in a way that certain outliers can be replaced by new outliers
            //
            int aNbOutliers = aNbKeyPtsInTri*aPtOutlierRate;
            for (int aKOut=0; aKOut<aNbOutliers; aKOut++)
            {
                int aRandCam = RandUnif_N(3);
                int aNbKeyPts =  aMulTiePMap[aCams[aRandCam]->NameImage()].mVecTPM.size();

                int aRandIdx = RandUnif_N(aNbKeyPts);
                cPt2dr aPt = aMulTiePMap[aCams[aRandCam]->NameImage()].mVecTPM.at(aRandIdx).mPt;

                //outlier
                cPt2dr aDelta = cPt2dr::PRandC()*aPtOutlierAmpl;
                // make sure still visible in frame
                if (aCams[aRandCam]->IsVisibleOnImFrame(aPt+aDelta))
                    aMulTiePMap[aCams[aRandCam]->NameImage()].mVecTPM.at(aRandIdx).mPt = aPt+aDelta;

            }

        } 
        StdOut() << "DONE Generate tie points" << std::endl;

        // update map of homologous points inside the photogrammetric project
        for (auto aHStr : aMulTiePMap )
        {
            aMemPhProj.AddMulTieP(aHStr.first,aHStr.second);
        }


        std::unique_ptr<cTripletSet>  a3Set(new cTripletSet);

        // compute relative orientation of the triplets
        size_t aTriCount=0;
        for (auto & aT : aTriplets)
        {
            // generate cams
            cSensorCamPC * aCam1 = aCamSim->mListCam.at(aT[0]);
            cSensorCamPC * aCam2 = aCamSim->mListCam.at(aT[1]);
            cSensorCamPC * aCam3 = aCamSim->mListCam.at(aT[2]);

            tPoseR aPose1 = tPoseR::Identity();
            tPoseR aPose2toPose1 = aCam1->RelativePose(*aCam2);
            tPoseR aPose3toPose1 = aCam1->RelativePose(*aCam3);

            //normalise the triplet
            double dist = Norm2(aPose2toPose1.Tr());
            aPose2toPose1.Tr() = aPose2toPose1.Tr() / dist;
            aPose3toPose1.Tr() = aPose3toPose1.Tr() / dist;

            // save to triplet structure for hierarchical init
            std::vector<cView> aTViews;
            aTViews.push_back(cView(aPose1,aCam1->NameImage()));
            aTViews.push_back(cView(aPose2toPose1,aCam2->NameImage()));
            aTViews.push_back(cView(aPose3toPose1,aCam3->NameImage()));

            cTriplet aThisTri;
            aThisTri.Id() = aTriCount;
            aThisTri.PVec() = aTViews;

            a3Set->PushTriplet(aThisTri);

            aTriCount++;

        }
        // generate outliers on triplets
        for (int aKtri=0; aKtri<aTriOutNb; aKtri++)
        {
            int aRandTri = RandUnif_N(aNbTri);

            for (int aK=1; aK<3; aK++) //first camera is spared
            {
                tPose& aCurP = a3Set->Set()[aRandTri].Pose(aK).Pose();
                aCurP.Tr() = aCurP.Tr() + cPt3dr::PRandInSphere() * aTriOutAmpl[0];
                aCurP.Rot() = aCurP.Rot() * tRotR::RandomSmallElem(aTriOutAmpl[1]);
            }
        }

        // run hierarchical init
        StdOut() << "Start Hierarchical SfM" << std::endl;
        cMakeArboTriplet  aMk3(*a3Set,false,1.0,aMemPhProj,anAp);

        //aMk3.ViscPose() = mViscPose;
        aMk3.LVM() = 0.001;
        aMk3.SigmaTPt() = 50;
        aMk3.FacElim()= 1000;
        aMk3.NbIterBA() = 3;

        aMk3.InitTPtsStruct("",aSetIm);

        aMk3.MakeGraphPose();
        aMk3.InitialiseCalibs();
        aMk3.DoPoseRef();
        aMk3.MakeCnxTriplet();
        aMk3.MakeWeightingGraphTriplet();
        aMk3.ComputeArbor();

        // retrieve computed poses
        aMk3.SaveGlobSol();
        std::map<std::string, cSensorCamPC *> aSolCams = aMemPhProj.SensorMap();

        // compute similarity transformation from GT frame and computed frame
        std::vector<tPoseR> aVPoseFrameGT;
        std::vector<tPoseR> aVPoseFrameCalc;
        for (auto & aCamGT : aCamSim->mListCam)
        {

            if (aSolCams[aCamGT->NameImage()])
            {
                aVPoseFrameGT.push_back(aCamGT->Pose());
                aVPoseFrameCalc.push_back(aSolCams[aCamGT->NameImage()]->Pose());

                //StdOut() << aCamGT->Pose().Tr() << " " << aSolCams[aCamGT->NameImage()]->Pose().Tr() << std::endl;
            }
        }
        auto [aRes,aSim] = EstimateSimTransfertFromPoses(aVPoseFrameGT,aVPoseFrameCalc);


        double ErrTrTotal=0;
        double ErrRTotal=0;
        int aNbPoses=0;
        for (auto aCamGT : aCamSim->mListCam)
        {
            if (aSolCams[aCamGT->NameImage()])
            {
                tPoseR aPoseCalcInGT = TransfoPose(aSim,aSolCams[aCamGT->NameImage()]->Pose());

                double aErrTrCur  = Norm2(aCamGT->Pose().Tr() - aPoseCalcInGT.Tr());
                double aErrRotCut = aCamGT->Pose().Rot().Dist(aPoseCalcInGT.Rot());

                ErrTrTotal+=aErrTrCur;
                ErrRTotal+=aErrRotCut;
                aNbPoses++;

                StdOut() << "ErrTr=" << aErrTrCur << ", ErrR=" << aErrRotCut
                         << ", dd " << aCamGT->Pose().Tr() - aPoseCalcInGT.Tr() << std::endl;
            }

        }
        StdOut() << "ErrTrAvg=" << ErrTrTotal/aNbPoses << ", ErrRAvg=" << ErrRTotal/aNbPoses << std::endl;
        getchar();




        // print residuals

        /*
            - alow introduction of outliers on triplets
        */



    }
}


}; // MMVII




