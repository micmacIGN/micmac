#include "ArboTriplets.h"

namespace MMVII
{
   // mNbTriAnchor mNbTreeGlob mNbTree2Split 

/* ********************************************************* */
/*                                                           */
/*                     cNodeArborTriplets                    */
/*                                                           */
/* ********************************************************* */


cNodeArborTriplets::cNodeArborTriplets(cMakeArboTriplet & aMAT ,const t3G3_Tree & aTree,int aLevel,
                                       cPhotogrammetricProject & aPhProj) :
   mPhProj    (aPhProj),
   mDepth     (aLevel),
   mTree      (aTree),
   mChildren  {0,0},
   mPMAT      (&aMAT)
{
   // if the edges tree is no empty, recursively split (a tree with 1 vertex has no edge)
   if (!aTree.Edges().empty())
   {
      // [1]  extract best split
      cWhichMax<t3G3_Edge*,tREAL8> aWME; // memorize best split
      for (const auto & anE : aTree.Edges())
      {
          std::array<t3G3_Tree,2>  a2T;
          aTree.Split(a2T,anE);
          int aN1 = a2T[0].Edges().size();
          int aN2 = a2T[1].Edges().size();
          tREAL8 aBalanceRatio =  std::min(aN1,aN2) / (1.0+std::max(aN1,aN2));  // balancing ratio
          // weight taking into account the Balance Ratio and importance of balancing
          tREAL8  aWeight = aMAT.WBalance() * aBalanceRatio + (1-aMAT.WBalance()) ; 
          //update with cost taking into account intrinsiq quality & balance
          aWME.Add(anE,anE->AttrSym().mCostTree * aWeight);
      }

      // [2]  split arround best
      std::array<t3G3_Tree,2>  a2T;
      aTree.Split(a2T,aWME.IndexExtre());  // split the tree
      aMAT.CostMergeTree() += a2T.at(0).Edges().size() +  a2T.at(1).Edges().size();
      // recursively build the split node
      for (size_t aKT=0 ; aKT<a2T.size() ; aKT++)
      {
          mChildren[aKT] = new cNodeArborTriplets(aMAT,a2T.at(aKT),aLevel+1,aPhProj);
      }
   }

}

cNodeArborTriplets:: ~cNodeArborTriplets()
{
    delete mChildren[0];
    delete mChildren[1];
}



void cNodeArborTriplets::ShowPose(const std::string & aPrefix) 
{
    std::vector<int> aVPose; GetPoses(aVPose);
    StdOut() << aPrefix;
    for (int aK=0 ; aK< mDepth ; aK++)
       StdOut() << " |" ;
    StdOut() <<  aVPose << "\n";
}

void cNodeArborTriplets::ComputeResursiveSolution()
{
   MMVII_INTERNAL_ASSERT_tiny((mChildren.at(0) == nullptr) == (mChildren.at(1) == nullptr),"ComputeResursiveSolution, assert on desc");

    ShowPose("TRM");

    if (mChildren.at(0) == nullptr) // Terminal node just put the riplet
    {
       auto  aVecTri = mTree.Vertices();
       MMVII_INTERNAL_ASSERT_tiny(aVecTri.size()==1,"ComputeResursiveSolution, assert on desc");
       c3G3_AttrV & anATri = aVecTri.at(0)->Attr();
       for (size_t aK3=0 ; aK3<3 ; aK3++)
       {
	  int aNumPose = anATri.m3V.at(aK3)->Attr().mKIm;
	  tPoseR aPose = anATri.mT0->Pose(aK3).Pose();

          mLocSols.push_back(cSolLocNode(aPose,aNumPose));
       }
       MakeIndexGlob2Loc();
    }
    else  // non terminal node
    {
       // recursive call to compute
       for (auto & aChild :mChildren)
           aChild->ComputeResursiveSolution();
       // to the computation for the sol of 2 children
       MergeChildrenSol();
    }

    // compare with ground truth
    CmpWithGT();
}

void cNodeArborTriplets::finalize()
{

    if (mChildren.at(0) == nullptr)
    {}//StdOut() << "Do nothing" << std::endl;
    else
    {
        // recursive call to compute
        //for (auto & aChild :mChildren)
        //    aChild->ComputeResursiveSolution();
        // to the computation for the sol of 2 children
        MergeChildrenSol();
    }


    // compare with ground truth
    //CmpWithGT();


}

// before finalize I have to run down the tree to initialise Index
void cNodeArborTriplets::DoTerminalNode()
{
    MMVII_INTERNAL_ASSERT_tiny((mChildren.at(0) == nullptr) == (mChildren.at(1) == nullptr),"DoTerminalNode, assert on desc");

    ShowPose("TRM");

    if (mChildren.at(0) == nullptr) // Terminal node just put the riplet
    {
        auto  aVecTri = mTree.Vertices();
        MMVII_INTERNAL_ASSERT_tiny(aVecTri.size()==1,"ComputeResursiveSolution, assert on desc");
        c3G3_AttrV & anATri = aVecTri.at(0)->Attr();
        for (size_t aK3=0 ; aK3<3 ; aK3++)
        {
            int aNumPose = anATri.m3V.at(aK3)->Attr().mKIm;
            tPoseR aPose = anATri.mT0->Pose(aK3).Pose();

            mLocSols.push_back(cSolLocNode(aPose,aNumPose));
        }
        MakeIndexGlob2Loc();
    }
    else  // non terminal node
    {
        // recursive call to compute
        for (auto & aChild :mChildren)
            aChild->DoTerminalNode();//ComputeResursiveSolution();
        MakeIndexGlob2Loc();
    }
}

void cNodeArborTriplets::MakeIndexGlob2Loc()
{
    mTabGlob2LocInd = std::vector<int>(mPMAT->GOP().NbVertex(),-1);  // create a tab with all -1 
    // compute the inverse map when it exist
    for (size_t aK=0 ; aK<mLocSols.size() ; aK++)
       mTabGlob2LocInd.at(mLocSols.at(aK).mNumPose) = aK; 
}

void cNodeArborTriplets::FreeIndexSol()
{
   mTabGlob2LocInd.clear();
   mLocSols.clear();
   mRotateLS.clear();
}


cOneTripletMerge  cNodeArborTriplets::ComputeTripletLinking(const t3G3_Vertex & aVTri)
{
     cOneTripletMerge aResult;
     aResult.mNumTri = aVTri.Attr().mKT;

     cNodeArborTriplets & aN0 = *(mChildren.at(0));  // extract first child
     cNodeArborTriplets & aN1 = *(mChildren.at(1));  // extract second child

     std::vector<tPairI>  aVPair0;  // store the link  between the triplet and N0
     std::vector<tPairI>  aVPair1;  // store the link  between the triplet and N1

     // parse the 3 images of the triplet
     for (size_t aK=0 ; aK<3 ; aK++)
     {
         int aKIm = aVTri.Attr().m3V.at(aK)->Attr().mKIm; // extract the global num  of the image

         int aI0Loc =  aN0.mTabGlob2LocInd.at(aKIm); // extract it num in N0 (-1 if none)
         int aI1Loc =  aN1.mTabGlob2LocInd.at(aKIm); // extract it num in N1 (-1 if none)

         if ((aI0Loc>=0) || (aI1Loc>=0)) // if one of both exist
         {
             if ((aI0Loc>=0) && (aI1Loc>=0)) // if both exist, it's an image common to 2 children
             {
                 aResult.mVCommon.push_back(tPairI(aI0Loc,aI1Loc));
             }
             else if (aI0Loc>=0) // else if exist in N0, it can be a link, memo num in triplet & N0
             {
                 aVPair0.push_back(tPairI(aK,aI0Loc));
             }
             else // else if exist in N1, it can be a link, memo num in triplet & N1
             {
                 aVPair1.push_back(tPairI(aK,aI1Loc));
             }
         }
     }

     // transformat in pair linking the 2 nodes,  if VPair0 or VPair1 is empty; there is no links,
     // else there can be 1 or 2 links 
     for (const auto & [aK0,aI0Loc] : aVPair0)
     {
         for (const auto & [aK1,aI1Loc] : aVPair1)
         {
             aResult.mVLinkPose.push_back(tPairI(aI0Loc,aI1Loc));
             aResult.mVLinkInTri.push_back(tPairI(aK0,aK1));
         }
     }

     return aResult;
}

tPoseR  cNodeArborTriplets::PoseRelEdge(int aI0Loc,int aI1Loc) const
{
    cNodeArborTriplets & aN0 = *(mChildren.at(0));
    cNodeArborTriplets & aN1 = *(mChildren.at(1));

    // [1]  Extract the edge of the graph

    int aI0Glob = aN0.mLocSols[aI0Loc].mNumPose;  //  Global numbers
    int aI1Glob = aN1.mLocSols[aI1Loc].mNumPose;

    t3GOP_Vertex & aV0 = mPMAT->GOP().VertexOfNum(aI0Glob); // Vertices
    t3GOP_Vertex & aV1 = mPMAT->GOP().VertexOfNum(aI1Glob);

    t3GOP_Edge & aE01 =  *aV0.EdgeOfSucc(aV1);  // the corresponding edge
    
    // [2]  Extract the "Reference" pose *
    tPoseR  aResult = aE01.AttrSym().mPoseRef2to1;
    //and eventually correct from sens
    if (!aE01.IsDirInit())
       return aResult.MapInverse();

    return aResult;
}

tRotR  cNodeArborTriplets::EstimateRotTransfert
             (
                  std::vector<tREAL8>&                   aWeightR,
                  const std::vector<tPairI>&             aVPairCommon,
                  const std::vector<tPairI>&             aVPairLink2,
                  const std::vector<cOneTripletMerge> &  aVLink3
             )
{
    cAutoTimerSegm aTimerRestim(mPMAT->TimeSegm(),"RotEstim");
    // store all the individual transfer, note that as they are defined up to a scale, the translation cannot be used directly
    std::vector<tPoseR>   aVecTransf_W1_to_W0;
    cNodeArborTriplets & aN0 = *(mChildren.at(0));
    cNodeArborTriplets & aN1 = *(mChildren.at(1));

    // *****************************************************************************
    //  A :  Compute all the possible "transfer-rotation" using the 3 possibilities
    // *****************************************************************************

    // [A.1]   Extrac transfer from comon poses
    for (const auto & [aI0,aI1] : aVPairCommon)
    {
         const tPoseR  & aPI_to_W0 = aN0.mLocSols[aI0].mPose;  //  I -> W0
         const tPoseR  & aPI_to_W1 = aN1.mLocSols[aI1].mPose;  //  I -> W1
        //  Im->W0 * (Im->W1)-1 = W1->W0
         tPoseR aP_W1_to_W0 =   aPI_to_W0 * aPI_to_W1.MapInverse();
         aVecTransf_W1_to_W0.push_back(aP_W1_to_W0);
    }

    // [A.2]   Extrac transfer from edge-link (pair not included in a triplet fully include)
    for (const auto & [aI0Loc,aI1Loc] : aVPairLink2)
    {
         const tPoseR  & aPI0_to_W0 = aN0.mLocSols[aI0Loc].mPose;  //  I0 -> W0
         const tPoseR  & aPI1_to_W1 = aN1.mLocSols[aI1Loc].mPose;  //  I1 -> W1
         tPoseR  aP_I1_to_I0 = PoseRelEdge(aI0Loc,aI1Loc);

         //   (I0-> W0) * (I1->I0) * (W1->I1) =  W1->W0
         tPoseR aP_W1_to_W0  =   aPI0_to_W0 *  aP_I1_to_I0   * aPI1_to_W1.MapInverse();

         aVecTransf_W1_to_W0.push_back(aP_W1_to_W0);
    }

    // [A.3]  Extract transfer from linking-triplet fully includes in the 2 children
    for (const auto & aLnk3: aVLink3)
    {
        // initial triplet, contain the relative poses
        const cTriplet & a3 = *(mPMAT->GO3().VertexOfNum(aLnk3.mNumTri).Attr().mT0);
        for (size_t aKL=0 ; aKL< aLnk3.mVLinkPose.size() ; aKL++)  // parse the 2 links
        {
             // [3.1]  extract the realtive pose , and mapping I1 coord-> I0 Coord
             const auto & [aK0Tri,aK1Tri] = aLnk3.mVLinkInTri.at(aKL); // num in triplet , [0,1,2]
             const tPoseR  & aPI0_to_Tri = a3.Pose(aK0Tri).Pose();  // Map Coord Im -> coord triplet
             const tPoseR  & aPI1_to_Tri = a3.Pose(aK1Tri).Pose();
             //  transfer mapping "I1,coord->I0,coord"  :  (Tri->I0) *  (I1->Tri) =  I1->I0
             tPoseR aP_I1_to_I0 =  aPI0_to_Tri.MapInverse() * aPI1_to_Tri;

             // [3.2]  extract the current pose in each node
             const auto & [aI0Loc,aI1Loc] = aLnk3.mVLinkPose.at(aKL);
             const tPoseR  & aPI0_to_W0 = aN0.mLocSols[aI0Loc].mPose;  //  I0 -> W0
             const tPoseR  & aPI1_to_W1 = aN1.mLocSols[aI1Loc].mPose;  //  I1 -> W1

             // estimation of transfer mapping World1-coord -> Word0-coord
             tPoseR aP_W1_to_W0  =   aPI0_to_W0 *  aP_I1_to_I0   * aPI1_to_W1.MapInverse();
             aVecTransf_W1_to_W0.push_back(aP_W1_to_W0);
        }
    }

    // *****************************************************************************
    //  [B]   Compute the global rotation transfer
    // *****************************************************************************

    // B.1  : ---------- construct a vector of rotation --------------------
    std::vector<tRotR>  aVRot;
    std::vector<tREAL8> aVWeight;
    for (const auto & aP : aVecTransf_W1_to_W0)
    {
       aVRot.push_back(aP.Rot());
       aVWeight.push_back(1.0);
    }

    // B.2  : ---------- make a robust estimation --------------------
    //  B.2.1 robust estimator, but to limit compuation time is Nb>aSzMax => split in small pack, estimate, and aggregate
    int aSzMax = 20;
    tREAL8 aSig0 = 0.1;
    tRotR aRotEstim ;
    {

        aRotEstim = tRotR::PseudoMediane(aVRot,aSzMax);  
        //  B.2.2 make weighted averaging initializd from pseudo median d
        //  Make exactly  2 iteration with  a weighting having a L1 behaviour at infinite
        aRotEstim = tRotR::RobustAvg(aVRot,aRotEstim,{aSig0,2,0.5},2);
        //  Make exactly  iteration with 1/R2 at infinite
        aRotEstim = tRotR::RobustAvg(aVRot,aRotEstim,{aSig0,2,1.0},2,1e-4,8);
    }
  
    // B.3  : ---------- weight the data taking into account the rotation residual --------------------
    {
         aWeightR.clear();
         for (const auto & aRot : aVRot)
             aWeightR.push_back(aRot.Dist(aRotEstim));
         // this value modelize the fact that the estimation are noisy
         tREAL8 aMinD = 1e-2;
         // we add some noise to the 
         auto SoftMax2 =[](tREAL8 A,tREAL8 B) {return std::sqrt(Square(A)+Square(B));};

         tREAL8 aMedian = SoftMax2(ConstMediane(aWeightR),aMinD);
         for (auto & aW : aWeightR)
         {
             aW = SoftMax2(aMinD,aW);
             aW = 1/(1+ std::pow(aW/aMedian,2.0));
         }
    }
     

    // some msg and check with perfect data that all rotation estimation are close to each other
    StdOut()<<" COM="<< aVPairCommon <<" Liink= "<< aVPairLink2<<" NB3=" << aVLink3.size() << " NbTransPose=" << aVecTransf_W1_to_W0.size() << "\n";
    if (mPMAT->PerfectData())
    {
       for (const auto & aP : aVecTransf_W1_to_W0)
       {
            tREAL8 aD = aRotEstim.Dist(aP.Rot());
            StdOut() << "Rot-estimation on perfect data " << aD << std::endl;
            //MMVII_INTERNAL_ASSERT_bench((aD<1e-5),"Rot-estimation on perfect data");
       }
    }

    return aRotEstim;
}

void cNodeArborTriplets::AddEqLink
     (
          cLinearOverCstrSys<tREAL8> * aSys,cSetIORSNL_SameTmp<tREAL8> * aSubst,
          tREAL8 aWeight, int aKEq,
          const cPt3dr &aC0_in_W0, const cPt3dr &aC1_in_W0,
          const cPt3dr &aCTri0_in_W0, const cPt3dr &aCTri1_in_W0
     )
{
   for (int aKC=0 ; aKC<3 ; aKC++)
   {
       // Add the equation that force first center triplet to be equal to C0 after transfering with the unknown of triplet
       //  TrTri and LambdaTri that are stored at aKEq,aKEq+1,..,aKEq+3
       //   aC0_in_W0  = TrTri + LambdaTri * aCTri0_In_W0
       int aICoord = aSubst ? -(1+aKC)  : (aKEq+aKC) ;
       int aILambda = aSubst ? -4       : (aKEq+3) ;

       tVIV aSV0 {{aICoord,1.0},{aILambda,aCTri0_in_W0[aKC]}} ;

       // idem for second vertex but must take into account the the transfer W1->W0
       //   Tr + Lambda * aC1_in_W0  = TrTri + LambdaTri * aCTri1_In_W0
       tVIV aSV1 {    {aICoord,1.0},{aILambda,aCTri1_in_W0[aKC]},  // TrTri + LambdaTri * aCTri1_In_W0
                               {aKC,-1.0},{3,-aC1_in_W0[aKC]}              // - (  Tr + Lambda * aC1_in_W0)
                 };
       if (aSubst)
       {
          aSubst->AddOneLinearObs(aWeight,aSV0,aC0_in_W0[aKC]);
          aSubst->AddOneLinearObs(aWeight,aSV1,0.0);
       }
       else
       {
          aSys->PublicAddObservation(aWeight, tSV(aSV0 ),aC0_in_W0[aKC]);
          aSys->PublicAddObservation(aWeight,tSV(aSV1 ),0.0);
       }
     }
}





tSim3dR cNodeArborTriplets::EstimateSimTransfert
             (
                  const std::vector<tPairI>& aVPairCommon,
                  const std::vector<tPairI>& aVPairLink2,
                  const std::vector<cOneTripletMerge> &  aVLink3
             )
{
    cAutoTimerSegm aTimerRestim(mPMAT->TimeSegm(),"SimEstim");
    // [0]  some preliminary stuff
    // typedef   std::vector<cCplIV<tREAL8>> tVIV;
    // typedef   cSparseVect<tREAL8>         tSV;
    bool  withLnk2=true;
    bool  withLnk3=true;
    bool  withSchur = false;  // work with Schur but, surprinsingly (?) , increase computation time

    cNodeArborTriplets & aN0 = *(mChildren.at(0));
    cNodeArborTriplets & aN1 = *(mChildren.at(1));
    // estimate rotation first
    std::vector<tREAL8> aWeightR;
    tRotR aRot_W1_to_W0 = EstimateRotTransfert(aWeightR,aVPairCommon,aVPairLink2,aVLink3);


    // [1]   Make in N1 a copy of local sol that are turned of aRot_W1_to_W0
    {
         aN1.mRotateLS.clear();
         // make a pose corresponding to pure rotation, arbitrary translation because undefined
         tPoseR aPose_W1_to_W0(cPt3dr(0,0,0),aRot_W1_to_W0);
         for (const  auto & aLS :  aN1.mLocSols)
             aN1.mRotateLS.push_back(cSolLocNode(aPose_W1_to_W0*aLS.mPose,aLS.mNumPose));
    }

    // [2] initialize the solver
    int aNbUnk = 4;
    if (!withSchur)
    {
       // for each edge, there is 4 unkown to fix arbitrary scale/trans od the edge
       if (withLnk2)
          aNbUnk += aVPairLink2.size() * 4;
       // for each triplet with again 4 unknwon
       if (withLnk3)
          aNbUnk += aVLink3.size() * 4;
     }

    // Using sparse system improve a factor 1000 (!!) the speed of  solving 
    eModeSSR aModeSSR = withSchur ? eModeSSR::eSSR_LsqDense  : eModeSSR::eSSR_LsqNormSparse;
    cLinearOverCstrSys<tREAL8> * aSys = cLinearOverCstrSys<tREAL8>::AllocSSR(aModeSSR,aNbUnk);
    // cLinearOverCstrSys<tREAL8> * aSys = new cLeasSqtAA<tREAL8>(aNbUnk);
    //cLinearOverCstrSys<tREAL8> * aSys = AllocL1_Barrodale<tREAL8>(aNbUnk);
    

    int aKWeight=0;
    // [3]   Add the equation corresponding to common pose
    for (const auto & [aI0Loc,aI1Loc] : aVPairCommon)
    {
        tREAL8 aW= aWeightR.at(aKWeight++);
        cPt3dr aC0 = aN0.mLocSols[aI0Loc].mPose.Tr();  // Centre of I in W0
        cPt3dr aC1 = aN1.mRotateLS[aI1Loc].mPose.Tr(); // centre of I in W1 after rotation W0->W1

        //  C0 and C1 are two estimation of the center of the pose, they must be equal up
        //  to the global transfert (Tr,Lambda) from W1 to W0
        // 
        for (int aKC=0 ; aKC<3 ; aKC++)
        {
            //  observauion is :   aC0.{x,y,z} = Tr{x,y,z} + Lambda aC1.{x,y,z}   
            tVIV aVIV {{aKC,1.0},{3,aC1[aKC]}};   //  KC->num of Tr.{x,y,z}  ,  3 num of lambda
            aSys->PublicAddObservation(aW, tSV(aVIV),aC0[aKC]);
        }
    }
    // count the current position of equation for edge/triplet
    int aKEq = 4;
    cSetIORSNL_SameTmp<tREAL8> aSubst(std::vector<tREAL8>(4));
    cSetIORSNL_SameTmp<tREAL8> * aPtrSubst = withSchur ? &aSubst : nullptr;//  static_cast<cSetIORSNL_SameTmp<tREAL8>*>(nullptr);

    // [4]   Add the equation corresponding to edge links, for each pair we have two equation that involves
    // the unknown tranfer Edge->W0 and the global unknown W1->W0
    if (withLnk2)
    {
        for (const auto & [aI0Loc,aI1Loc] : aVPairLink2)
        {
             tREAL8 aWeight = aWeightR.at(aKWeight++);
             const tPoseR  & aPI0_to_W0 = aN0.mLocSols[aI0Loc].mPose;  //  Pose/Mappoing  I0 -> W0
             const tPoseR  & aPI1_to_W1 = aN1.mRotateLS[aI1Loc].mPose;  // Pose/Mappoing  I1 -> W1  (after rotation)

             cPt3dr aC0_in_W0  = aPI0_to_W0.Tr();  // extract centers
             cPt3dr aC1_in_W0  = aPI1_to_W1.Tr();

             tPoseR  aPI1_toTri = PoseRelEdge(aI0Loc,aI1Loc);  // pose of I1 in triplet
             tPoseR  aPI0_toTri = tPoseR::Identity();         // just to symetrize the process with PI1

             // for the center of the edge we must first align the orientation of  the pose with W0
             tRotR aR_Tri_to_W0;   // rotation Triplet/Edge  ->  W0
             {
                  // we can use I0 or I1 for this computation, the result should be equivalent -> do the average
                  tRotR  aR0_Tri_to_W0 = aPI0_to_W0.Rot() * aPI0_toTri.Rot().MapInverse();
                  tRotR  aR1_Tri_to_W0 =   aPI1_to_W1.Rot() * aPI1_toTri.Rot().MapInverse();
                  aR_Tri_to_W0 = aR0_Tri_to_W0.Centroid(aR1_Tri_to_W0);
 
                  if (mPMAT->PerfectData()) // test that in fact the 2 computation are equivalent
                  {
                      tREAL8 aDist = aR0_Tri_to_W0.Dist(aR1_Tri_to_W0);
                      StdOut() << "Rot-estimation Tri->W0 on perfect data " << aDist << std::endl;
                      //MMVII_INTERNAL_ASSERT_bench((aDist<1e-5),"Rot-estimation Tri->W0 on perfect data");
                  }
             }


             // "transer" the centers of triplet in the W0
             cPt3dr aCTri0_In_W0 = aR_Tri_to_W0.Value(aPI0_toTri.Tr());
             cPt3dr aCTri1_In_W0 = aR_Tri_to_W0.Value(aPI1_toTri.Tr());

             AddEqLink(aSys,aPtrSubst,aWeight,aKEq,aC0_in_W0,aC1_in_W0,aCTri0_In_W0,aCTri1_In_W0);
             aKEq += 4;
             if (withSchur)
                aSys->PublicAddObsWithTmpUK(*aPtrSubst);
        }
    }

    // [5]  Add the equation corresponding to triplet
    if (withLnk3)
    {
       for (const auto & aLnk3: aVLink3)
       {
           const c3G3_AttrV & anAttr = mPMAT->GO3().VertexOfNum(aLnk3.mNumTri).Attr();
           // initial triplet, contain the relative poses
           const cTriplet & a3 = *(anAttr.mT0);

            // [5.1]  Compute the transfer rotation from Tri->W0
           std::vector<tRotR>  aVEstimTriToV0;
           for (int aKIn3=0 ; aKIn3<3 ; aKIn3++)
           {
               tPoseR aPIK_to_W0;  // Pose Im-> W0
               int aNumImG = anAttr.m3V.at(aKIn3)->Attr().mKIm  ;  //  Num Image Glob
               int aLocNum0 = aN0.mTabGlob2LocInd.at(aNumImG); // Num in W0
               if (aLocNum0>=0)
                   aPIK_to_W0 = aN0.mLocSols.at(aLocNum0).mPose;
               else
               {
                   int aLocNum1 = aN1.mTabGlob2LocInd.at(aNumImG);
                   aPIK_to_W0 = aN1.mRotateLS.at(aLocNum1).mPose;
               }
               const tPoseR  & aPIk_to_Tri = a3.Pose(aKIn3).Pose();

               tPoseR  aP_Tri_to_W0 =   aPIK_to_W0 * aPIk_to_Tri.MapInverse();
               aVEstimTriToV0.push_back(aP_Tri_to_W0.Rot());
           }
           tRotR  aR_Tri_to_W0 = tRotR::Centroid(aVEstimTriToV0,{1.0,1.0,1.0});

           if (mPMAT->PerfectData())
           {
               for (int aKIn3=0 ; aKIn3<3 ; aKIn3++)
               {
                   tREAL8 aD = aR_Tri_to_W0.Dist(aVEstimTriToV0.at(aKIn3));
                   StdOut() << "Transfer Triple->W0 on perfect data " << aD << std::endl;
                   //MMVII_INTERNAL_ASSERT_bench((aD<1e-5),"Transfer Triple->W0 on perfect data");
               }
           }

           // [5.2]  now add the equation ...
           for (size_t aKL=0 ; aKL< aLnk3.mVLinkPose.size() ; aKL++)  // parse the 2 links
           {
                 tREAL8 aWeight = aWeightR.at(aKWeight++);
                 const auto & [aK0Tri,aK1Tri] = aLnk3.mVLinkInTri.at(aKL); // num in triplet , [0,1,2]
                 cPt3dr aCTri0_In_W0 = aR_Tri_to_W0.Value(a3.Pose(aK0Tri).Pose().Tr());
                 cPt3dr aCTri1_In_W0 = aR_Tri_to_W0.Value(a3.Pose(aK1Tri).Pose().Tr());

                 const auto & [aI0Loc,aI1Loc] = aLnk3.mVLinkPose.at(aKL);
                 cPt3dr  aC0_in_W0 = aN0.mLocSols.at (aI0Loc).mPose.Tr();
                 cPt3dr  aC1_in_W0 = aN1.mRotateLS.at(aI1Loc).mPose.Tr();

                 AddEqLink(aSys,aPtrSubst,aWeight,aKEq,aC0_in_W0,aC1_in_W0,aCTri0_In_W0,aCTri1_In_W0);
           }
           aKEq += 4;
           if (withSchur)
              aSys->PublicAddObsWithTmpUK(*aPtrSubst);
       }
    }

    MMVII_INTERNAL_ASSERT_bench(aKWeight==int(aWeightR.size()),"End of rotation weighting");

    cAutoTimerSegm aTimerSolveSim(mPMAT->TimeSegm(),"SolveSim");
    cDenseVect<tREAL8> aSol = aSys->PublicSolve();
    tREAL8 aLambda = aSol(3);
    cPt3dr aTr(aSol(0),aSol(1),aSol(2));

    delete aSys;

    return tSim3dR(aLambda,aTr,aRot_W1_to_W0);
}

void cNodeArborTriplets::SaveGlobSol(const std::string & aPrefix) const
{
    //tREAL8 AngConv = AngleInRad(eTyUnitAngle::eUA_degree);
    cDenseMatrix<double> aZRot(3,3,eModeInitImage::eMIA_Null);
    aZRot.SetElem(0,0,1);
    aZRot.SetElem(1,1,-1);
    aZRot.SetElem(2,2,-1);

    cPerspCamIntrCalib *  aCalib = mPhProj.InternalCalibFromStdName(mPMAT->MapI2Str(mLocSols.at(0).mNumPose));

    std::string aSaveSolG = aPrefix + "_depth_" + ToStr(mDepth) + "_" + ToStr(RandUnif_N(1000));
    cMMVII_Ofs aFile(aSaveSolG, eFileModeOut::CreateText);

    aFile.Ofs() << "#F=N X Y Z a b c d e f g h i\n";

    std::cout << std::setprecision(10);
    for (const auto & aSol :   mLocSols)
    {
        std::string aCurImName = mPMAT->MapI2Str(aSol.mNumPose);

        cRotation3D<tREAL8> aRotNew(aSol.mPose.Rot().Mat().Transpose(),false); //mmv1 convention
        cPt3dr aCNew = aSol.mPose.Tr()  * aZRot ; //mmv1 convention


        std::string aPrntTxt = aCurImName + " "
                               + ToStr(aCNew.x()) + " " + ToStr(aCNew.y()) + " " + ToStr(aCNew.z()) + " ";

        for (int aK1=0; aK1<3; aK1++)
        {
            for (int aK2=0; aK2<3; aK2++)
            {
                aPrntTxt += ToStr(aRotNew.Mat()(aK2,aK1)) + " ";
            }
        }
        aPrntTxt += "\n";
        aFile.Ofs() << aPrntTxt;

        //cIsometry3D aPose(aCNew,aRotNew);
        cSensorCamPC aCam(aCurImName,aSol.mPose,aCalib); //mmv2 convention
        mPhProj.SaveCamPC(aCam);

    }
    //////////////////////////////
    std::vector<tPoseR> aVComp;
    std::vector<tPoseR> aVGt;

    if (0)
    {
        StdOut() << "=========== GT vs Computed Pose " << std::endl;
        std::cout << std::setprecision(6);
        for (const auto & aSol :   mLocSols)
        {
            aVComp.push_back(aSol.mPose);
            aVGt.push_back(mPMAT->GOP().VertexOfNum(aSol.mNumPose).Attr().mGTRand);
        }
        auto [aRes,aSim] =  EstimateSimTransfertFromPoses(aVComp,aVGt);

        for (size_t aKP=0 ; aKP<aVComp.size() ; aKP++)
        {
            tPoseR aVGtInComp = TransfoPose(aSim,aVGt.at(aKP));
            StdOut() << aVComp[aKP].Tr() << " == " << aVGtInComp.Tr() << std::endl;
            aVComp[aKP].Rot().Mat().Show();
            aVGtInComp.Rot().Mat().Show();
            StdOut() << "===========" << std::endl;
        }
    }
}

void cNodeArborTriplets::CmpWithGT()
{
    if (! mPMAT->DoRand())  return;

    std::vector<tPoseR> aVComp;
    std::vector<tPoseR> aVGt;

    for (const auto & aSol :   mLocSols)
    {
         aVComp.push_back(aSol.mPose);
         aVGt.push_back(mPMAT->GOP().VertexOfNum(aSol.mNumPose).Attr().mGTRand);
    }
    auto [aRes,aSim] =  EstimateSimTransfertFromPoses(aVComp,aVGt);

    for (size_t aKP=0 ; aKP<aVComp.size() ; aKP++)
    {
         tREAL8 aD = aVComp.at(aKP).DistPose(TransfoPose(aSim,aVGt.at(aKP)),1.0); 
         if (mPMAT->PerfectData())
         {
            StdOut() << "====CmpWithGT==== D=" << aD << std::endl;
            //MMVII_INTERNAL_ASSERT_bench((aD<1e-5),"Sim-Transfer on perfect data");
         }
    }
}

void cNodeArborTriplets::MergeChildrenSol()
{
     cNodeArborTriplets & aN0 = *(mChildren.at(0));
     cNodeArborTriplets & aN1 = *(mChildren.at(1));

     //    ShowPose("DoMx :");
     //aN0.ShowPose("DoM0 :");
     //aN1.ShowPose("DoM1 :");

     std::vector<tPairI>              aVPairCommon;  //  Store data for vertex present in 2 children
     std::vector<tPairI>              aVPairLink2;   // store data for edges between 2 children (the 3 vertex being out)
     std::vector<cOneTripletMerge>    aVLink3;  // store triplet with 3 vertices doing the link
     std::vector<bool>               aSetIndexTri(mPMAT->GO3().NbVertex(),false);  // marqer to test triplet once
     std::vector<bool>               aSetIndComN0(aN0.mLocSols.size(),false);      // marqer to have common vertex once

     // before computing the merge, accumulate all the links;  the must be done a priori because the number of
     // unknown will depend of the link (for example on scale unknwon by triplet)

     for (const auto & aSol0 : aN0.mLocSols)  // parse 1 child, because linking triplet must belong to 2 children
     {
         const auto & aVertexPose = mPMAT->GOP().VertexOfNum(aSol0.mNumPose);  // extract vertex of pose
         const c3GOP_AttrV aAttrPose = aVertexPose.Attr(); // extracts its attributes
         for (const auto & aNumTri : aAttrPose.mTriBelongs)  // parse the triplets is belong to (avoid parse all triplet at all level)
         {
             if (!aSetIndexTri.at(aNumTri)) // avoid  exploring several time the same triplet
             {
                 aSetIndexTri.at(aNumTri) = true;  // marq it as explored
                 cOneTripletMerge  a1TM = ComputeTripletLinking(mPMAT->GO3().VertexOfNum(aNumTri)); // compute the linking

                 // memorize the common image to N0-N1
                 for (const auto & [aI0,aI1] : a1TM.mVCommon)
                 {
                     if  (! aSetIndComN0.at(aI0)) // avoid store twice the image
                     {
                         aSetIndComN0.at(aI0) = true; // marq as done,  
                         aVPairCommon.push_back(tPairI(aI0,aI1)); // store
                     }
                 }
                 // add a link if (1) there is link !  (2) there is no common pose
                 if ((! a1TM.mVLinkPose.empty())  && (a1TM.mVCommon.empty()))
                 {
                     if (a1TM.mVLinkPose.size() == 1) // if only 1 link, this a "edge link"
                     {
                        aVPairLink2.push_back(a1TM.mVLinkPose.at(0));
                     }
                     else  // else 2 link, it's triplet link
                     {
                        aVLink3.push_back(a1TM);
                     }
                 }
             }
         }
     }
     // now the same link edges can have been store many time : supress the duplicata
     {
         std::sort(aVPairLink2.begin(),aVPairLink2.end());  // sort
         auto aEndUniqueLnk2 = std::unique(aVPairLink2.begin(),aVPairLink2.end());  // put duplicata at end
         aVPairLink2.resize(aEndUniqueLnk2 - aVPairLink2.begin());  // resize
     }

     // estimate the tranfser similitude between N0 & N1
     tSim3dR  aSimTransfer = EstimateSimTransfert(aVPairCommon,aVPairLink2,aVLink3);

     // [3]   Finnaly do the merge, using N0 system as reference
        // [3.1]  Put Sol0 that are not in Sol1
     for (const auto & aSol0 : aN0.mLocSols)
     {
         if (aN1.SolOfGlobalIndex(aSol0.mNumPose)==nullptr)
         {
              mLocSols.push_back(aSol0);
         }
     }

         //  [3.1]  Put Sol1, use transfert, and separate case in Sol0 or not
     for (const auto & aSol1 : aN1.mLocSols)
     {
         tPoseR  aPoseInS0  =    TransfoPose(aSimTransfer,aSol1.mPose);  // compute Pose tranfsered in N0
         cSolLocNode * aSol0 = aN0.SolOfGlobalIndex(aSol1.mNumPose);
         if (aSol0)
         {
            // if exist in N0 also, we have two estimation, use the average
            aPoseInS0 = tPoseR::Centroid({aPoseInS0,aSol0->mPose},{1.0,1.0});

            if (mPMAT->PerfectData())  // case simulation with perfect triplets, we have almost identic pose for two solution
            {
                 tREAL8 aDist = aPoseInS0.DistPose(aSol0->mPose,1.0);
                StdOut() << "Sim-Transfer on perfect data " << aDist << std::endl;
                 //MMVII_INTERNAL_ASSERT_bench((aDist<1e-5),"Sim-Transfer on perfect data");
            }
         }

         // add the , potentially averaged, solution
         mLocSols.push_back(cSolLocNode(aPoseInS0,aSol1.mNumPose));
     }

     //aN0.SaveGlobSol("Child0Init");
     //aN1.SaveGlobSol("Child1Init");
     //SaveGlobSol("Init");

     // refine the solution with BA
     //if (!mPMAT->TPFolder().empty())
     if (mPMAT->TPtsStruct() !=nullptr)
        RefineSolution();

     //SaveGlobSol("Adj");

     //  Free some temporary memory that are  no longer necessary
     for (auto & aChild :mChildren)
     {
          aChild->FreeIndexSol();
     }
     // prepare index for merge in parent
     MakeIndexGlob2Loc();

// getchar();
}

void cNodeArborTriplets::RefineSolution()
{

    StdOut() << "RefineSolution" << std::endl;


    int aNbIter=mPMAT->NbIterBA();

    // structure storing declared unknowns of BA
    cSetInterUK_MultipeObj<tREAL8> aSetIntervUK;


    // camera parameters and colinearity equations
    std::vector<cSensorCamPC *> aVCams ;
    std::vector<cSensorImage *> aVSens ;
    std::vector<cCalculator<double> *> aVEqCol ;

    // intrinsic parameters considered the same for all images
    //StdOut() << mLocSols.at(0).mNumPose  << " " << mPMAT->MapI2Str(mLocSols.at(0).mNumPose) << std::endl;
    //cPerspCamIntrCalib *   aCal = mPhProj.InternalCalibFromStdName(mPMAT->MapI2Str(mLocSols.at(0).mNumPose),false);
    //aSetIntervUK.AddOneObj(aCal);

    // vector of all image names belonging to this tree level
    std::vector<std::string> aVNames;


    // fill in the vector of image names
    for (auto aLocPose : mLocSols)
    {
        std::string aImName = mPMAT->MapI2Str(aLocPose.mNumPose);
        aVNames.push_back(aImName);

    }

    // sort images alphbetically (and mLocSols accordingly) for AllocStdFromMTPFromFolder
    Sort2VectFirstOne(aVNames,mLocSols);

    //SaveGlobSol("Init");

    int aCamCurCount=0;
    for (auto aSol : mLocSols)
    {
        std::string aImName = mPMAT->MapI2Str(aSol.mNumPose);
        StdOut() << aSol.mNumPose << " " << aImName << std::endl;

        cIsometry3D<tREAL8> aPoseChgConv(aSol.mPose.Tr() ,aSol.mPose.Rot() );

        // store camera in a vector
        aVCams.push_back( new cSensorCamPC(aImName,aPoseChgConv,nullptr) );
        aVSens.push_back( aVCams.at(aCamCurCount) );


        // collinearity equation (calculator)
        aVEqCol.push_back( aVCams.at(aCamCurCount)->CreateEqColinearity(true,100,false) );

        // add/declare the camera as unknonwn
        aSetIntervUK.AddOneObj(aVCams.at(aCamCurCount));

        aCamCurCount++;
    }
    //getchar();


    // BA solver
    cResolSysNonLinear<tREAL8> * aSys = new cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqNormSparse,aSetIntervUK.GetVUnKnowns());


    // add viscosity on poses
    for (auto & aCam : aVCams)
    {
        if ( mPMAT->ViscPose().at(0)>0)
        { StdOut() << "RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR " << mPMAT->ViscPose().at(0)
                     << ", " << mPMAT->ViscPose().at(1) << std::endl;
            aSys->AddEqFixCurVar(*aCam,aCam->Center(),Square(1.0/mPMAT->ViscPose().at(0)));
        }
        if (mPMAT->ViscPose().at(1)>0)
        {
            aSys->AddEqFixCurVar(*aCam,aCam->Omega(),Square(1.0/mPMAT->ViscPose().at(1)));
        }
    }

    // read the tie points corresponding to your image set
    cComputeMergeMulTieP aTPts(*mPMAT->TPtsStruct(),aVNames);


    // image points weighting function
    //tREAL8 aSigAtt=mPMAT->SigmaTPt();
    //tREAL8 aFactElim=mPMAT->FacElim();
    //std::vector<tREAL8> aThrRange({aSigAtt*aFactElim,aSigAtt*5});
    //tREAL8 aDeltaThr=aThrRange[0]-aThrRange[1];

    //StdOut() << "Start BA : #Configs=" << aTPts->Pts().size() << std::endl;
    StdOut() << "---------------------- "
             << "#Images " << aVCams.size() << ", pts=" << aTPts.Pts().size() << std::endl;
    for (int aIter=0; aIter<aNbIter; aIter++)
    {

        // intersect tie-points in 3D
        for (auto & aPair : aTPts.Pts())
            MakePGroundFromBundles(aPair,aVSens); //MakePGround

        /* W(R) =
               0 if R>Thrs
               1/Sigma0^2  * (1/(1+ (R/SigmaAtt)^Exp))
        cStdWeighterResidual(tREAL8 aSGlob,tREAL8 aSigAtt,tREAL8 aThr,tREAL8 aExp); */

        //tREAL8 aThr = aDeltaThr*(1 - double(aIter)/(aNbIter-1)) + aThrRange[1];
        //StdOut() << "aThr=" << aThr << ", Start=" << aThrRange[0] << ", End=" << aThrRange[1] << std::endl;
        cStdWeighterResidual aTPtsW (0.0001,0.005,0.2,1.0);//(1.0,50.0,1000.0,1.0)  (1,aSigAtt,aThr,2);
        tREAL8 aMaxRes=0;
        tREAL8 aTotalW=0;


        int aNumAllTiePts=0;
        int aNumTPts=0;
        int aNumAll3DPts=0;
        int aNum3DPts=0;
        cWeightAv<tREAL8> aWeigthedRes;

        for (auto aAllConfigs : aTPts.Pts())
        {
            const auto & aConfig = aAllConfigs.first;
            auto & aVals = aAllConfigs.second;

            size_t aNbIm = aConfig.size();
            size_t aNbPts = aVals.mVIdPts.size();

            aNumAll3DPts+=aNbPts;

            // add to BA
            for (size_t aKPts=0; aKPts<aNbPts; aKPts++)
            {

                const cPt3dr & aP3D = aVals.mVPGround.at(aKPts);
                cSetIORSNL_SameTmp<tREAL8>  aStrSubst(aP3D.ToStdVector());
                //if ( aIter==(aNbIter-1))
                //    StdOut() << aP3D.x() << " " << aP3D.y() << " " << aP3D.z() << std::endl;

                //tREAL8 aResTotal = 0;
                size_t aNbEqAdded = 0;
                for (size_t aKIm=0; aKIm<aNbIm; aKIm++)
                {
                    size_t aKImSorted = aConfig.at(aKIm);

                    aNumAllTiePts++;

                    const cPt2dr aPIm = aVals.mVPIm.at(aKPts*aNbIm+aKIm);
                    cSensorCamPC* aCam = aVCams.at(aKImSorted);

                    // handle IsVisibleOnImFrame&IsVisible later
                    //if (aCam->IsVisibleOnImFrame(aPIm) && aCam->IsVisible(aP3D))
                    {

                        cPt2dr aResidual = aPIm - aCam->Ground2Image(aP3D);
                        //StdOut() << aPIm << " " << aCam->Ground2Image(aP3D)
                        //         << ", res=" << aResidual << std::endl;
                        tREAL8 aResNorm = Norm2(aResidual);
                        //aResTotal+=aResNorm;

                        tREAL8 aWeight = aTPtsW.SingleWOfResidual(aResidual);

                        //StdOut() << "RRRR " << aPIm << " " << aResidual << " W=" << aWeight
                        //         << " " << aCam->NameImage() << " " << aKImSorted << " " << aKIm << "\n";



                        cCalculator<double> * aEqCol =  aVEqCol.at(aKIm);
                        std::vector<double> aVObs = aPIm.ToStdVector();

                        aCam->PushOwnObsColinearity(aVObs,aP3D);

                        std::vector<int> aVIndGlob = {-1,-2,-3};  // index of unknown, temporary
                        for (auto & anObj : aCam->GetAllUK())  // now put sensor unknown
                        {
                            anObj->PushIndexes(aVIndGlob);
                        }

                        if (aWeight>0)
                        {
                            aWeigthedRes.Add(aWeight,aResNorm);
                            aSys->R_AddEq2Subst(aStrSubst,aEqCol,aVIndGlob,aVObs,aWeight);
                            aNbEqAdded++;
                            aNumTPts++;

                            aTotalW+=aWeight;
                            if (aMaxRes<aResNorm)
                                aMaxRes=aResNorm;
                        }
                    }
                }

                if (aNbEqAdded>=2)
                {
                    aSys->R_AddObsWithTmpUK(aStrSubst);
                    aNum3DPts++;
                }

            }
        }

        double aPercInliers = (aNumTPts*100)/aNumAllTiePts;
        StdOut() << "#Iter=" << aIter
                 << ", #3D points=" << aNumAll3DPts << ", #Inliers=" << aNum3DPts
                 << ", #2D obs=" << aNumTPts << ", #Inliers=" << aPercInliers << " %"
                 << ", MaxRes=" << aMaxRes << ", TotalW=" << aTotalW
                 << " Weighted Res=" << aWeigthedRes.Average() << std::endl;

        tREAL8 aLVM=0.1;
        const auto & aVectSol = aSys->SolveUpdateReset({aLVM},{},{});//
        aSetIntervUK.SetVUnKnowns(aVectSol);

        //StdOut() << " StdDevLast=" << std::sqrt(aSys->VarLastSol())
        //         << " StdDevCur=" << std::sqrt(aSys->VarCurSol()) << std::endl;


    }
    //ShowPose("===BA at tree depth: ===");
    //StdOut() << "END BA" << std::endl;

    // final pose update in the global tree structure
    aCamCurCount=0;
    for (auto aCamAdj : aVCams)
    {
        mLocSols.at(aCamCurCount).mPose.Tr() = aCamAdj->Center();
        mLocSols.at(aCamCurCount).mPose.Rot() = aCamAdj->Pose().Rot();

        aCamCurCount++;
    }
    //getchar();




    aSetIntervUK.SIUK_Reset();

    delete aSys;
   // delete aTPts;
    for (auto aECol : aVEqCol)
        delete aECol;
    for (auto aCam : aVCams)
        delete aCam;

}

void cNodeArborTriplets::RefineSolution_()
{

    //StdOut() << "RefineSolution" << std::endl;


    int aNbIter=mPMAT->NbIterBA();

    // structure storing declared unknowns of BA
    cSetInterUK_MultipeObj<tREAL8> aSetIntervUK;


    // camera parameters and colinearity equations
    std::vector<cSensorCamPC *> aVCams ;
    std::vector<cSensorImage *> aVSens ;
    std::vector<cCalculator<double> *> aVEqCol ;

    // intrinsic parameters considered the same for all images
    //StdOut() << mLocSols.at(0).mNumPose  << " " << mPMAT->MapI2Str(mLocSols.at(0).mNumPose) << std::endl;
    cPerspCamIntrCalib *   aCal = mPhProj.InternalCalibFromStdName(mPMAT->MapI2Str(mLocSols.at(0).mNumPose),false);
    aSetIntervUK.AddOneObj(aCal);

    // vector of all image names belonging to this tree level
    std::vector<std::string> aVNames;


    // fill in the vector of image names
    for (auto aLocPose : mLocSols)
    {
        std::string aImName = mPMAT->MapI2Str(aLocPose.mNumPose);
        aVNames.push_back(aImName);

    }

    // sort images alphbetically (and mLocSols accordingly) for AllocStdFromMTPFromFolder
    Sort2VectFirstOne(aVNames,mLocSols);

    //SaveGlobSol("Init");

    int aCamCurCount=0;
    for (auto aSol : mLocSols)
    {
        std::string aImName = mPMAT->MapI2Str(aSol.mNumPose);
        StdOut() << aSol.mNumPose << " " << aImName << std::endl;

        cIsometry3D<tREAL8> aPoseChgConv(aSol.mPose.Tr() ,aSol.mPose.Rot() );

        // store camera in a vector
        aVCams.push_back( new cSensorCamPC(aImName,aPoseChgConv,aCal) );
        aVSens.push_back( aVCams.at(aCamCurCount) );


        // collinearity equation (calculator)
        aVEqCol.push_back( aVCams.at(aCamCurCount)->CreateEqColinearity(true,100,false) );

        // add/declare the camera as unknonwn
        aSetIntervUK.AddOneObj(aVCams.at(aCamCurCount));

        aCamCurCount++;
    }
    //getchar();



    // BA solver
    cResolSysNonLinear<tREAL8> * aSys = new cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqNormSparse,aSetIntervUK.GetVUnKnowns());
    // freeze internal calibration
    aSys->SetFrozenFromPat(*aCal,".*",true);


    // add viscosity on poses
    for (auto & aCam : aVCams)
    {
        if ( mPMAT->ViscPose().at(0)>0)
        { StdOut() << "RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR " << mPMAT->ViscPose().at(0)
                     << ", " << mPMAT->ViscPose().at(1) << std::endl;
            aSys->AddEqFixCurVar(*aCam,aCam->Center(),Square(1.0/mPMAT->ViscPose().at(0)));
        }
        if (mPMAT->ViscPose().at(1)>0)
        {
            aSys->AddEqFixCurVar(*aCam,aCam->Omega(),Square(1.0/mPMAT->ViscPose().at(1)));
        }
    }

    // read the tie points corresponding to your image set
    cComputeMergeMulTieP aTPts(*mPMAT->TPtsStruct(),aVNames);


    // image points weighting function
    //tREAL8 aSigAtt=mPMAT->SigmaTPt();
    //tREAL8 aFactElim=mPMAT->FacElim();
    //std::vector<tREAL8> aThrRange({aSigAtt*aFactElim,aSigAtt*5});
    //tREAL8 aDeltaThr=aThrRange[0]-aThrRange[1];

    //StdOut() << "Start BA : #Configs=" << aTPts->Pts().size() << std::endl;
    StdOut() << "---------------------- "
             << "#Images " << aVCams.size() << ", pts=" << aTPts.Pts().size() << std::endl;
    for (int aIter=0; aIter<aNbIter; aIter++)
    {

        // intersect tie-points in 3D
        for (auto & aPair : aTPts.Pts())
            MakePGround(aPair,aVSens); //

        /* W(R) =
               0 if R>Thrs
               1/Sigma0^2  * (1/(1+ (R/SigmaAtt)^Exp))
        cStdWeighterResidual(tREAL8 aSGlob,tREAL8 aSigAtt,tREAL8 aThr,tREAL8 aExp); */

        //tREAL8 aThr = aDeltaThr*(1 - double(aIter)/(aNbIter-1)) + aThrRange[1];
        //StdOut() << "aThr=" << aThr << ", Start=" << aThrRange[0] << ", End=" << aThrRange[1] << std::endl;
        cStdWeighterResidual aTPtsW (1.0,50.0,1000.0,1.0);//(1,aSigAtt,aThr,2);
        tREAL8 aMaxRes=0;
        tREAL8 aTotalW=0;


        int aNumAllTiePts=0;
        int aNumTPts=0;
        int aNumAll3DPts=0;
        int aNum3DPts=0;
        cWeightAv<tREAL8> aWeigthedRes;

        for (auto aAllConfigs : aTPts.Pts())
        {
            const auto & aConfig = aAllConfigs.first;
            auto & aVals = aAllConfigs.second;

            size_t aNbIm = aConfig.size();
            size_t aNbPts = aVals.mVIdPts.size();

            aNumAll3DPts+=aNbPts;

            // add to BA
            for (size_t aKPts=0; aKPts<aNbPts; aKPts++)
            {

                const cPt3dr & aP3D = aVals.mVPGround.at(aKPts);
                cSetIORSNL_SameTmp<tREAL8>  aStrSubst(aP3D.ToStdVector());
                //if ( aIter==(aNbIter-1))
                //    StdOut() << aP3D.x() << " " << aP3D.y() << " " << aP3D.z() << std::endl;

                //tREAL8 aResTotal = 0;
                size_t aNbEqAdded = 0;
                for (size_t aKIm=0; aKIm<aNbIm; aKIm++)
                {
                    size_t aKImSorted = aConfig.at(aKIm);

                    aNumAllTiePts++;

                    const cPt2dr aPIm = aVals.mVPIm.at(aKPts*aNbIm+aKIm);
                    cSensorCamPC* aCam = aVCams.at(aKImSorted);

                    if (aCam->IsVisibleOnImFrame(aPIm) && aCam->IsVisible(aP3D))
                    {

                        cPt2dr aResidual = aPIm - aCam->Ground2Image(aP3D);
                        tREAL8 aResNorm = Norm2(aResidual);
                        //aResTotal+=aResNorm;

                        tREAL8 aWeight = aTPtsW.SingleWOfResidual(aResidual);

                        //StdOut() << "RRRR " << aPIm << " " << aResidual << " W=" << aWeight
                        //         << " " << aCam->NameImage() << " " << aKImSorted << " " << aKIm << "\n";



                        cCalculator<double> * aEqCol =  aVEqCol.at(aKIm);
                        std::vector<double> aVObs = aPIm.ToStdVector();

                        aCam->PushOwnObsColinearity(aVObs,aP3D);

                        std::vector<int> aVIndGlob = {-1,-2,-3};  // index of unknown, temporary
                        for (auto & anObj : aCam->GetAllUK())  // now put sensor unknown
                        {
                            anObj->PushIndexes(aVIndGlob);
                        }

                        if (aWeight>0)
                        {
                            aWeigthedRes.Add(aWeight,aResNorm);
                            aSys->R_AddEq2Subst(aStrSubst,aEqCol,aVIndGlob,aVObs,aWeight);
                            aNbEqAdded++;
                            aNumTPts++;

                            aTotalW+=aWeight;
                            if (aMaxRes<aResNorm)
                                aMaxRes=aResNorm;
                        }
                    }
                }

                if (aNbEqAdded>=2)
                {
                    aSys->R_AddObsWithTmpUK(aStrSubst);
                    aNum3DPts++;
                }

            }
        }

        double aPercInliers = (aNumTPts*100)/aNumAllTiePts;
        StdOut() << "#Iter=" << aIter
                 << ", #3D points=" << aNumAll3DPts << ", #Inliers=" << aNum3DPts
                 << ", #2D obs=" << aNumTPts << ", #Inliers=" << aPercInliers << " %"
                 << ", MaxRes=" << aMaxRes << ", TotalW=" << aTotalW
                 << " Weighted Res=" << aWeigthedRes.Average() << std::endl;

        tREAL8 aLVM=0.1;
        const auto & aVectSol = aSys->SolveUpdateReset({aLVM},{},{});//
        aSetIntervUK.SetVUnKnowns(aVectSol);

        //StdOut() << " StdDevLast=" << std::sqrt(aSys->VarLastSol())
        //         << " StdDevCur=" << std::sqrt(aSys->VarCurSol()) << std::endl;


    }
    //ShowPose("===BA at tree depth: ===");
    //StdOut() << "END BA" << std::endl;

    // final pose update in the global tree structure
    aCamCurCount=0;
    for (auto aCamAdj : aVCams)
    {
        mLocSols.at(aCamCurCount).mPose.Tr() = aCamAdj->Center();
        mLocSols.at(aCamCurCount).mPose.Rot() = aCamAdj->Pose().Rot();

        aCamCurCount++;
    }
    //getchar();




    aSetIntervUK.SIUK_Reset();

    delete aCal;
    delete aSys;
    // delete aTPts;
    for (auto aECol : aVEqCol)
        delete aECol;
    for (auto aCam : aVCams)
        delete aCam;

}

cSolLocNode * cNodeArborTriplets::SolOfGlobalIndex(int  anIndAbs) 
{
   int anIndRel = mTabGlob2LocInd.at(anIndAbs);
   if (anIndRel>=0)
      return  &mLocSols.at(anIndRel);


   return nullptr;
}


void cNodeArborTriplets::GetPoses(std::vector<int> & aResult)
{
   aResult.clear();
   // marqer to avoid duplicata, dont use flag because of pentotially use in parallel
   std::vector<bool> aVGot(mPMAT->GOP().NbVertex(),false);

   for (const auto & aVertexTri : mTree.Vertices()) 
   {
        auto & anAttr = aVertexTri->Attr();
	for (auto & aVertexPos : anAttr.m3V)
	{
	     int aNumPose = aVertexPos->Attr().mKIm;
             if (!aVGot.at(aNumPose))
             {
                aVGot.at(aNumPose) = true;
                aResult.push_back(aNumPose);
             }
	}
   }
   std::sort(aResult.begin(),aResult.end());
}

/* ********************************************************* */
/*                                                           */
/*                     c3GOP_AttrSym                         */
/*                                                           */
/* ********************************************************* */

void c3GOP_AttrSym::ComputePoseRef(tREAL8 aWTr)
{
   
   cWhichMin<size_t,tREAL8> aWMin;

   // Theoretically , mListP2to1 cannot be empty
   for (size_t aK1=0 ; aK1<mListP2to1.size() ; aK1++)
   {
       tREAL8 aSum = 0.0;
       for (size_t aK2=0 ; aK2<mListP2to1.size() ; aK2++)
       {
           aSum += mListP2to1.at(aK1).DistPoseRel(mListP2to1.at(aK2),aWTr);
       }
       aWMin.Add(aK1,aSum);
   }
   mPoseRef2to1 = mListP2to1.at(aWMin.IndexExtre());
}

/* ********************************************************* */
/*                                                           */
/*                     c3G3_AttrV                            */
/*                                                           */
/* ********************************************************* */

c3G3_AttrV::c3G3_AttrV (cTriplet* aT0,int aKT) :
            mT0         (aT0),
            mKT         (aKT),
            mCnxE       {0,0,0},
            m3V         {0,0,0},
            mOk         (false),
            mCostIntr   (10.0)
{
}

size_t c3G3_AttrV::GetIndexVertex(const t3GOP_Vertex* aV) const
{
      for (size_t aKV=0 ; aKV<3 ; aKV++)
          if (m3V.at(aKV) == aV)
             return aKV;

      MMVII_INTERNAL_ERROR("c3G3_AttrV::GetIndexVertex");
      return 3;
}

tREAL8 c3G3_AttrV::CostVertexCommon(const c3G3_AttrV & anAttr2,tREAL8 aWTr) const
{
   // Extract indexes that are common to both triplet
   std::vector<int> aV;
   for (size_t aKV1=0 ; aKV1<3 ; aKV1++)
   {
       for (size_t aKV2=0 ; aKV2<3 ; aKV2++)
       {
          if (m3V.at(aKV1) == anAttr2.m3V.at(aKV2))
          {
             aV.push_back(aKV1);
             aV.push_back(aKV2);
          }
       }
   }

   //  extract pose relative of B to A in triplet 1
   tPoseR  aPAtoW1 = mT0->Pose(aV.at(0)).Pose();
   tPoseR  aPBtoW1 = mT0->Pose(aV.at(2)).Pose();
   tPoseR  aW1_BtoA  = aPAtoW1.MapInverse()  *  aPBtoW1;

   //  extract pose relative of B to A in triplet 2
   tPoseR  aPAtoW2   = anAttr2.mT0->Pose(aV.at(1)).Pose();
   tPoseR  aPBtoW2   = anAttr2.mT0->Pose(aV.at(3)).Pose();
   tPoseR  aW2_BtoA  = aPAtoW2.MapInverse()  *  aPBtoW2;

   //  compute distances  in translation and rotation
   tREAL8 aDistTr  = Norm2( VUnit(aW1_BtoA.Tr()) - VUnit(aW2_BtoA.Tr()));
   tREAL8 aDistRot = aW1_BtoA.Rot().Dist(aW2_BtoA.Rot());
   tREAL8 aRes =  (aDistRot+aWTr*aDistTr) / (1+aWTr);


   if (0)
   {
       static std::set<std::vector<int>> aSV;
       aSV.insert(aV);
       StdOut() << "VvV=" << aV << " " << aSV.size() << " " << VUnit(aW1_BtoA.Tr()) - VUnit(aW2_BtoA.Tr()) << "\n";
   }
   return aRes;
}



/* ********************************************************* */
/*                                                           */
/*                     cAppli_ArboTriplets                   */
/*                                                           */
/* ********************************************************* */


cMakeArboTriplet::cMakeArboTriplet(cTripletSet & aSet3,bool doCheck,tREAL8 aWBalance, cPhotogrammetricProject & aPhProj, cMMVII_Appli & anAppli) :
   mAppli       (anAppli),
   mPhProj      (aPhProj),
   mTimeSegm    (mAppli.TimeSegm()),
   mSet3        (aSet3),
   mDoCheck     (doCheck),
   mGPoses      (), // false=> not 4 simul
   mDoRand      (false),
   mLevelRand   {0.0,0.0},
   mWeigthEdge3 {0.75,0.25,0.1},
   mPerfectData (false),
   mPerfectOri  (false),
   mArbor       (nullptr),
   mWBalance    (aWBalance),
   mWeightTr    (0.5),
   mNbEdgeP     (0),
   mNbHypP      (0),
   mNbEdgeTri   (0),
   mTPtsFolder  (""), //to be removed
   mTPtsStruct  (nullptr),
   mViscPose    ({-1,-1}),
   mSigmaTPt    (1.0),
   mFacElim     (10.0),
   mNbIterBA    (2)
{
}

cMakeArboTriplet::~cMakeArboTriplet()
{
   // TO REDO => avoide LINK-PB  4 NOW
   delete mArbor;
   delete mTPtsStruct;
}
void cMakeArboTriplet::SetRand(const std::vector<tREAL8> & aLevelRand)
{
   mDoRand = true;
   mLevelRand = aLevelRand;
   mPerfectData = (aLevelRand.at(0)==0) && ( (aLevelRand.at(1)==0));
}

void cMakeArboTriplet::ShowStat()
{
   StdOut() << "\n";
   StdOut() << " * INPUT : NbTriplet= " << mSet3.Set().size() << " NbIm=" << mMapStrI.size() << "\n";
   StdOut() << " * POSE  : NbEgde= "  << mNbEdgeP <<  " NbHypP=" << mNbHypP << "\n";
   StdOut() << " * TRI  : NbEgde= "  << mNbEdgeTri << "\n";
   StdOut() << " * TREE:  Anchor=" << mNbTriAnchor << " Tree2Split=" << mNbTree2Split << " TreeGlob=" << mNbTreeGlob << "\n";

   StdOut() << "\n";
}

void cMakeArboTriplet::SaveGlobSol() const
{

    mArbor->SaveGlobSol("");
}

void cMakeArboTriplet::InitialiseCalibs()
{
    for (size_t aKIm=0 ; aKIm<mMapStrI.size() ; aKIm++)
    {
        cPerspCamIntrCalib *   aCal = mPhProj.InternalCalibFromStdName(*mMapStrI.I2Obj(aKIm));
        FakeUseIt(aCal);
    }
}

void cMakeArboTriplet::InitTPtsStruct(const std::string& aFolder, std::vector<std::string>& aVNames)
{
    // read tie points
    mTPtsStruct = AllocStdFromMTPFromFolder(aFolder,aVNames,mPhProj,true,false,true);

    //convert tie points to bundles (i.e., normalise)
    for (auto & [aConf,aVals] : mTPtsStruct->Pts())
    {
        int NbIm = aConf.size();
        int NbPts = aVals.mVPIm.size()/NbIm;

        // check that input points are not bundles
        MMVII_INTERNAL_ASSERT_medium(aVals.mVPZ.size()==0,"Observations are already bundles");
        // resize Z
        aVals.mVPZ.resize(NbIm*NbPts);

        // for every image
        for (int aKIm=0; aKIm<NbIm; aKIm++)
        {
            cPerspCamIntrCalib *   aCal = mPhProj.InternalCalibFromStdName(aVNames[aKIm],false);

            std::vector<cPt3dr> aOutBundles;
            std::vector<cPt2dr> aInObs;


            // for every image observation
            for (int aKObs=0; aKObs<NbPts; aKObs++)
            {
                aInObs.push_back(aVals.mVPIm[aKIm*NbPts+aKObs]);
            }

            // transform point to bundle
            aCal->DirBundles(aOutBundles,aInObs);

            // update vector of observation in tie-point structure
            for (int aKObs=0; aKObs<NbPts; aKObs++)
            {
                /*StdOut() << aCal->F() << " " << aCal->PP() << ", pix "
                         << aVals.mVPIm[aKIm*NbPts+aKObs].x() << " "
                         << aVals.mVPIm[aKIm*NbPts+aKObs].y() << " "
                         << aOutBundles[aKObs].x() << " "
                         << aOutBundles[aKObs].y() << std::endl;*/
                //getchar();
                aVals.mVPIm.at(aKIm*NbPts+aKObs) = cPt2dr(aOutBundles[aKObs].x(),
                                                          aOutBundles[aKObs].y());
                aVals.mVPZ[aKIm*NbPts+aKObs] = aOutBundles[aKObs].z();
            }

            delete aCal;
        }

    }

}

void cMakeArboTriplet::MakeGraphPose()
{

   // create vertices of mGTriC  & compute map NamePose/Int (in mMapStrI)
   for (size_t aKT=0; aKT<mSet3.Set().size() ; aKT++)
   {
        cTriplet & a3 = mSet3.Set().at(aKT);
        mGTriC.NewVertex(c3G3_AttrV(&a3,aKT));
        for (size_t aK3=0 ; aK3<3 ; aK3++)
        {
             mMapStrI.Add(a3.Pose(aK3).Name(),true);
        }
   }


   // In case we want to make some test with random rot, create the "perfect" rot of each pose

   //  Add the vertex corresponding to each poses in Graph-Pose
   for (size_t aKIm=0 ; aKIm<mMapStrI.size() ; aKIm++)
   {
       tPoseR aRandGT(cPt3dr::PRandInSphere(),tRotR::RandomRot());
       mGPoses.NewVertex(c3GOP_AttrV(aKIm,aRandGT));
   }

   //  Parse all the triplet; for each triplet  Add 3 edges of Grap-Pose ;
   //  also memorize the edges in the triplet for each, mCnxE
   for (size_t aKT=0; aKT<mGTriC.NbVertex() ; aKT++)
   {
        c3G3_AttrV & aTriC =   mGTriC.VertexOfNum(aKT).Attr();
        cTriplet & a3 = *(aTriC.mT0);
        if (mDoRand)  
        {
            /*
            // in simul we must take into account that each triplet is in its own arbitrary system  W2L , Word -> Loc
            tSim3dR   aRandSim= tSim3dR::RandomSim3D(2.0,2.0);
            // parse the 3 pair of consecutive poses
            for (int aK3=0 ; aK3<3 ; aK3++)
            {
                cView&  aView = a3.Pose(aK3);
                tPoseR & aP = aView.Pose();
                // initialize local pose to ground truth
                int aInd =  mMapStrI.Obj2I(aView.Name());
                t3GOP_Vertex & aPoseV = mGPoses.VertexOfNum(aInd);
                aP = aPoseV.Attr().mGTRand;
                // Firts create small perturbations of "perfect" values of "Tr/Rot"
                cPt3dr aTr = aP.Tr() + cPt3dr::PRandInSphere() * mLevelRand.at(0);
                tRotR aRot = aP.Rot()* tRotR::RandomSmallElem(mLevelRand.at(1));
                // Now put everyting in the local system and   finally save the result
                aP = TransfoPose(aRandSim,tPoseR(aTr,aRot));
            }
*/
        }
        if (mPerfectOri)
        {
            if (0) StdOut() << "************* GT Pose " << std::endl;
            for (int aK3=0 ; aK3<3 ; aK3++)
            {
                cView&  aView = a3.Pose(aK3);
                // initialize GTRand pose to ground truth
                int aInd =  mMapStrI.Obj2I(aView.Name());
                t3GOP_Vertex & aPoseV = mGPoses.VertexOfNum(aInd);
                tPoseR & aP = aPoseV.Attr().mGTRand;

                cSensorCamPC * aCam = mPhProj.ReadCamPC(aView.Name(),true);

                aP = tPoseR(aCam->Pose().Tr(),aCam->Pose().Rot());

                if (0) StdOut() << aView.Name() << " " << aCam->Pose().Tr() << std::endl;
                if (0) aCam->Pose().Rot().Mat().Show();
                if (0) StdOut() << "*************" << std::endl;
                // mmv1 vs mmv2 : there is a transpose on Rot
                if (0) getchar();
            }
        }

        // Make some normalization on triplet center
        for (int aKIter=0 ; aKIter<1 ;aKIter++)  // Iter =2 -> for test CDG/sum ...
        {
            // compute centroid
            cPt3dr aCdg(0,0,0);
            for (size_t aK3=0 ; aK3<3 ; aK3++)
                aCdg = aCdg + a3.Pose(aK3).Pose().Tr();
            aCdg = aCdg /3.0;
            // compute standar dev
            tREAL8 aSumD = 0.0;
            for (size_t aK3=0 ; aK3<3 ; aK3++)
            {
                aSumD+= SqN2(a3.Pose(aK3).Pose().Tr()-aCdg);
            }
            aSumD = std::sqrt(aSumD/3.0);
            // normalize
            for (size_t aK3=0 ; aK3<3 ; aK3++)
            {
                cPt3dr & aTr = a3.Pose(aK3).Pose().Tr();
                aTr = (aTr-aCdg) / aSumD;
            }
            // Check if 2 iteration
            if (aKIter==1)
               StdOut() << "   CCCC=" << aCdg << " SS=" << aSumD << "\n";
        }



        for (size_t aK3=0 ; aK3<3 ; aK3++)
        {
            // extract  the 2 vertices corresponding to poses
            const cView&  aView1 = a3.Pose(aK3);
            const cView&  aView2 = a3.Pose((aK3+1)%3);
            int aI1 =  mMapStrI.Obj2I(aView1.Name());
            int aI2 =  mMapStrI.Obj2I(aView2.Name());
            t3GOP_Vertex & aV1  = mGPoses.VertexOfNum(aI1);
            t3GOP_Vertex & aV2  = mGPoses.VertexOfNum(aI2);

            aTriC.m3V[aK3] = &aV1;

            // extract relative poses 
            tPoseR  aP1toW = aView1.Pose();
            tPoseR  aP2toW = aView2.Pose();

            if (0)
            {
                StdOut() << "===extract relative poses===\n";
                StdOut() << aView1.Name() << " Tr=" << aP1toW.Tr() << std::endl;
                aP1toW.Rot().Mat().Show();
                StdOut() << aView2.Name() << " Tr=" << aP2toW.Tr() << std::endl;
                aP2toW.Rot().Mat().Show();
                StdOut() << "===END===\n";
            }

            //  Ori_2->1  = Ori_G->1  o Ori_2->G    ( PL2 -> PG -> PL1)
            tPoseR  aP2to1 = aP1toW.MapInverse()  *  aP2toW;
	 
            tRotR aR2to1 = aP2to1.Rot();          
            t3GOP_Edge * anE_12 = aV1.EdgeOfSucc(aV2,true);
            if  (anE_12==nullptr)
            {
                mNbEdgeP++;
                anE_12 = mGPoses.AddEdge(aV1,aV2,c3GOP_AttrOr(),c3GOP_AttrOr(),c3GOP_AttrSym());
            }

            // mGPoses.AddHyp(aV1,aV2,aR2to1,1.0,c3GOP_1Hyp(aKT));


            t3GOP_Edge * anE_DirInit = anE_12->EdgeInitOr();
            aTriC.mCnxE.at(aK3) = anE_DirInit;
             
            tPoseR aPoseEdge =  anE_12->IsDirInit()  ? aP2to1 : aP2to1.MapInverse();
            anE_DirInit->AttrSym().mListP2to1.push_back(aPoseEdge);
            anE_DirInit->AttrSym().mListKT.push_back(aKT);
            mNbHypP++;
        }
   }
}


void cMakeArboTriplet::DoPoseRef()
{
   for (auto & anE : mGPoses.AllEdges_DirInit())
   {
      c3GOP_AttrSym & anAttr =  anE->AttrSym();

      anAttr.ComputePoseRef(mWeightTr);
      const auto & aListP = anAttr.mListP2to1;
      if (mPerfectData)
      {
          StdOut () <<  "================================= Triplet Coherency \n";
          tPoseR aPoseRef = anAttr.mPoseRef2to1;
          for (const auto & aPose : aListP)
          {
              tREAL8 aDistP = aPoseRef.DistPoseRel(aPose,1.0);
              aPose.Rot().Mat().Show();
              if (aDistP>1e-5)
              {
                  tREAL8 aDistR = aPoseRef.Rot().Dist(aPose.Rot());
                  tREAL8 aDistC = Norm2(VUnit(aPoseRef.Tr())- VUnit(aPose.Tr()));
                  StdOut() << "*****xxxxxxx  DP= " << aDistP << " DR=" << aDistR<< " DC=" << aDistC << std::endl;
                  //getchar();
              }
              else
                  StdOut() << " 000 dist=" <<  aDistP << "\n";
              //MMVII_INTERNAL_ASSERT_bench((aDist<1e-5),"Pose reference on perfect data");
          }
      }

   }
}

void cMakeArboTriplet::MakeCnxTriplet()
{
   // Parse all triplet
   for (size_t aKT1=0; aKT1<mGTriC.NbVertex() ; aKT1++)
   {
       t3G3_Vertex & aVert1  = mGTriC.VertexOfNum(aKT1);
       auto & aTriAttr1 = aVert1.Attr();
       for (size_t aKE1=0 ; aKE1<3 ; aKE1++)  // parse the 3 pose-edges attached to the vertex
       {
           const c3GOP_AttrSym & anAttrPose = aTriAttr1.mCnxE.at(aKE1)->AttrSym(); 
           for (const size_t & aKT2 : anAttrPose.mListKT)  //  parse all the triplet contained in this edge
           {
                  if (aKT1<aKT2) // do it only one way
                  {
                     mNbEdgeTri++;
                     t3G3_Vertex & aVert2  = mGTriC.VertexOfNum(aKT2);
                     auto & aTriAttr2 = aVert2.Attr();

                     // evaluate the coherence of the 2 triplet connected by 2 pose
                     tREAL8 aCost = aTriAttr1.CostVertexCommon(aTriAttr2,0.5);
                     if (mPerfectData)
                     {
                         StdOut() << " Cost=" << aCost << std::endl;
                        //MMVII_INTERNAL_ASSERT_bench((aCost<1e-5),"Cost 3-3 on perfect data");
                     }

                     mGTriC.AddEdge
                     (
                          aVert1,aVert2,
                          c3G3_AttrOriented(),c3G3_AttrOriented(),
                          c3G3_AttrSym(aCost)
                     );
                  }
           }
       }
   }
   auto  aListCC = cAlgoCC<t3G3_Graph>::All_ConnectedComponent(mGTriC,cAlgo_ParamVG<t3G3_Graph>());
   StdOut() << "Number of connected compon Triplet-Graph= " << aListCC.size() << "\n";
   // to see later, we can probably analyse CC by CC, but it would be more complicated (already enough complexity ...)
   MMVII_INTERNAL_ASSERT_tiny(aListCC.size(),"non connected triplet-graph");
}


/*  Basic idea , the weight of a vertices is the average of the weight of edges ( mCostInit2Ori).

      But pb, because if we have outlayer, the cost of edges is bad from2 sides.  So we iterate
    and make at each step a weighting so that  previous high residula have low weight
*/

void cMakeArboTriplet::MakeWeightingGraphTriplet()
{
    int aNbIter = 5;
    tREAL8 aS666 =0;
    for (int aKIter=0 ; aKIter<aNbIter ; aKIter++)
    {
        // store the new cost 
        std::vector<tREAL8>  aVCost(mGTriC.NbVertex());
        aS666 += 0.01;  // if all cost are ~ 0

         
        for (size_t aKT=0 ; aKT<mGTriC.NbVertex() ; aKT++)  // parse all vertices
        {
            t3G3_Vertex & aVTri = mGTriC.VertexOfNum(aKT);
        
            cWeightAv<tREAL8,tREAL8> aAvg;
            for (auto & aSucc : aVTri.EdgesSucc()) // parse neighboors
            {
                tREAL8 aWeight = 1.0;
                if (aKIter!=0)
                {
                   tREAL8 aCostNeigh =  aSucc->Succ().Attr().mCostIntr;
                   aWeight = 1.0 / (1+Square(aCostNeigh/aS666));
                }
                aAvg.Add(aWeight,aSucc->AttrSym().mCostInit2Ori);
            }
            aVCost.at(aKT)  = aAvg.Average(1000.0);  // 1000=> default, there exist
        }

         
        //   transferate new cost in mCostIntr
        cWeightAv<tREAL8,tREAL8> aAvgDif;
        cWeightAv<tREAL8,tREAL8> aAvgCost;
        for (size_t aKT=0 ; aKT<mGTriC.NbVertex() ; aKT++)
        {
            c3G3_AttrV & anAttr = mGTriC.VertexOfNum(aKT).Attr();
            aAvgDif.Add(1.0,std::abs(anAttr.mCostIntr-aVCost.at(aKT)));
            aAvgCost.Add(1.0,aVCost.at(aKT));
            anAttr.mCostIntr =  aVCost.at(aKT);
        }
        StdOut() <<  "COST, Evol=" << aAvgDif.Average() << " Avg=" << aAvgCost.Average() << "\n";

        aS666 = NC_KthVal(aVCost,0.6666);
        if (aKIter == (aNbIter-1))
        {
            for (const auto & aP : {0.1,0.5,0.75,0.9})
                StdOut() << " [" << aP << ":" << NC_KthVal(aVCost,aP) << "]";
            StdOut() << "\n";
        }
    }

    // Now fix the cost of edges
    for (auto & aVTri : mGTriC.AllVertices())
    {
        for (auto & aSucc : aVTri->EdgesSucc()) // parse neighboors
        {
            tREAL8 aW1 = aVTri->Attr().mCostIntr;
            tREAL8 aW2 = aSucc->Succ().Attr().mCostIntr;
            OrderMinMax(aW1,aW2);
            // some "magic" formula
            tREAL8 aCost = aW1*mWeigthEdge3.at(1) + aW2*mWeigthEdge3.at(0) + aSucc->AttrSym().mCostInit2Ori*mWeigthEdge3.at(2);
            aSucc->AttrSym().mCostTree = aCost;
        }
    }
}

void cMakeArboTriplet::ComputeArbor()
{
   // ============= [1]   parse all triplet so that at the end, each image : ==============
   //   - know the lowest cost triplet it belongs to
   //   - know all the triplets it belongs to
   for (size_t aKTri=0; aKTri<mGTriC.NbVertex() ; aKTri++)
   {
       auto & aTriVertex  = mGTriC.VertexOfNum(aKTri);
       c3G3_AttrV & aAttrTriV  = aTriVertex.Attr();
       tREAL8 aCost = aAttrTriV.mCostIntr;
       for (int aKV=0 ; aKV<3 ; aKV++) // parse the 3 image of the triplet
       {
           c3GOP_AttrV & aAttrPose = aAttrTriV.m3V.at(aKV)->Attr();
           aAttrPose.mWMinTri.Add(aKTri,aCost);
           aAttrPose.mTriBelongs.push_back(aKTri);
       }
   }


   // ==============   [2]  Does two things: ===============================================
   //    - for a triplet store in mVPoseMin the pose that consider it as the "best one"
   //    - store in aVectTriMin the triplet for which mVPoseMin is not empty
   std::vector<t3G3_Vertex*> aVectTriMin;
   for (size_t aKPose=0 ; aKPose<mGPoses.NbVertex() ; aKPose++)
   {
       int aKTriplet = mGPoses.VertexOfNum(aKPose).Attr().mWMinTri.IndexExtre();
       std::vector<int> * aVPoseMin = & mGTriC.VertexOfNum(aKTriplet).Attr().mVPoseMin;

       // store it only the first time to avoid duplicata
       if (aVPoseMin->empty())
       {
          aVectTriMin.push_back(&mGTriC.VertexOfNum(aKTriplet));
       }
       aVPoseMin->push_back(aKPose);
   }
   mNbTriAnchor =  aVectTriMin.size() ;

  
   // ==============    [3]  Compute the minimal spaning forest of graph of triplet   ====================
   // Create a subgraph of template type cTpl_WeithingSubGr, with a lambda expression
   //   mCos2Tri has been inialize in constructor
   auto aWeighting = Tpl_WeithingSubGr(&mGTriC,[](const t3G3_Edge & anE) {return anE.AttrSym().mCostTree;});
   // auto aWeighting = Tpl_WeithingSubGr(&mGTriC,[](const auto & anE) {return anE.AttrSym().mCost2Tri;});

   cAlgoSP<t3G3_Graph>::tForest  aForest(mGTriC);
   cAlgoSP<t3G3_Graph>  anAlgo;
   anAlgo.MinimumSpanningForest(aForest,mGTriC,mGTriC.AllVertices(),aWeighting);
   
   // dont handle 4 now the multiple connected components
   MMVII_INTERNAL_ASSERT_tiny(aForest.VTrees().size()==1,"Not single forest");// litle checj
   const auto & aGlobalTree = *(aForest.VTrees().begin());
   mNbTreeGlob = aGlobalTree.Edges().size();

   
   // ==============    [4]  Compute the pruning  : =====================================================
   //       recursing supression of extremities as long as they are not anchor points   ====================

   // define the sub-graph which  edges are the edges of the GlobalTree
   cSubGraphOfEdges<t3G3_Graph>  aSubGrTree(mGTriC,aGlobalTree.Edges());
   // define the sub-graph of anchor points, i.e the triplets that are the "best one" of given pose
   cSubGraphOfVertices<t3G3_Graph>  aSubGrAnchor(mGTriC,aVectTriMin); 


   // algorithm of extremities pruning : job is done in the constructor (compute the "Extrem()")
   cAlgoPruningExtre<t3G3_Graph> aAlgoSupExtr(mGTriC,aSubGrTree,aSubGrAnchor);

   // now supress from initial edges all the "pruned" extremities
   std::vector<t3G3_Edge*>  aSetEdgeKern;
   cVG_OpBool<t3G3_Graph>::EdgesMinusVertices(aSetEdgeKern,aGlobalTree.Edges(),aAlgoSupExtr.Extrem());
   mNbTree2Split = aSetEdgeKern.size();

   if (mDoCheck)
   {
        cSubGraphOfEdges<t3G3_Graph>  aSubGrKernel(mGTriC,aSetEdgeKern);
        // [1]  Check that all the triplet minimal  are belonging to the connection stuff
        for (const auto & aTriC : aVectTriMin)
        {
            MMVII_INTERNAL_ASSERT_always(aSubGrKernel.InsideVertex(*aTriC),"Kernel doesnot contain all  anchors");
        }
        // [2]  Check that aSetEdgeKern is a tree ...
        std::list<std::vector<t3G3_Vertex *>>  allCC =  cAlgoCC<t3G3_Graph>::All_ConnectedComponent(mGTriC,aSubGrKernel);
        MMVII_INTERNAL_ASSERT_always(allCC.size()==1,"Kernel is not connected");
        const std::vector<t3G3_Vertex *>& aCC0 =  *(allCC.begin());
        MMVII_INTERNAL_ASSERT_always(aCC0.size()==(aSetEdgeKern.size()+1),"Kernel is not tree");

   }
   // ==============    [5]  compute the tree : =====================================================
   t3G3_Tree  aTreeKernel(aSetEdgeKern);
   mCostMergeTree = 0.0;
   mArbor = new cNodeArborTriplets(*this,aTreeKernel,0,mPhProj);
   StdOut() << "CostMerge " << mCostMergeTree << "\n";

   //mArbor->ComputeResursiveSolution();

   mArbor->DoTerminalNode();
   StdOut() << "END DoTerminalNode" << std::endl;
   //
   cMemManager::SetActiveMemoryCount(false);
   mAppli.SetMultiThread(true);
   TreeThreads<cNodeArborTriplets*> tp;
   tp.Exec(mArbor,mAppli.NbProcAllowed());
   mAppli.SetMultiThread(false);
   cMemManager::SetActiveMemoryCount(true);

   StdOut() << "END Exec" << std::endl;


}

cAppli_ArboTriplets::cAppli_ArboTriplets(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mNbMaxClust  (5),
    mDistClust   (0.02),
    mDoCheck     (true),
    mWBalance    (1.0),
    mPerfectData (false),
    mViscPose    ({-1,-1})
{
}

cCollecSpecArg2007 & cAppli_ArboTriplets::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
              <<  mPhProj.DPOriTriplets().ArgDirInMand("Input triplets")
           ;
}

cCollecSpecArg2007 & cAppli_ArboTriplets::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return    anArgOpt
          << AOpt2007(mNbMaxClust,"NbMaxClust","Number max of rot in 1 cluster",{eTA2007::HDV})
          << AOpt2007(mDistClust,"DistClust","Distance in clustering",{eTA2007::HDV})
          << AOpt2007(mLevelRand,"LevelRand","Level of random if simulation [Trans,Rot]", {{eTA2007::ISizeV,"[2,2]"}})
          << AOpt2007(mWeigthEdge3,"WE3","Edge Weigthing from Vertices/InitialEdge [WMax,WMin,WEdge]", {{eTA2007::ISizeV,"[3,3]"}})
          << AOpt2007(mDoCheck,"DoCheck","do some checking on result",{eTA2007::HDV,eTA2007::Tuning})
          << AOpt2007(mWBalance,"WBalance","Weight for balancing trees, 0 NONE, 1 Max",{eTA2007::HDV})
          << AOpt2007(mPerfectData,"PerfectData","Evaluate coherency of triplets with simulated poses",{eTA2007::HDV})
          << AOpt2007(mViscPose,"ViscPose","Regularization on poses for BA: [SigmaTr,SigmaRot]",{eTA2007::HDV})
          <<  mPhProj.DPOrient().ArgDirInOpt("","Ground truth input orientation directory | Use internal calibration for saving")
          <<  mPhProj.DPOrient().ArgDirOutOpt("","Global orientation output directory")
          <<  mPhProj.DPOriTriplets().ArgDirOutOpt("","Directory for dmp-save of triplet (for faster read later)")
          <<  mPhProj.DPMulTieP().ArgDirInOpt("","Input features")
   ;
}

int cAppli_ArboTriplets::Exe()
{
     mPhProj.FinishInit();

      
     cAutoTimerSegm  aATS(TimeSegm(),"Read3");
     cTripletSet *  a3Set =  mPhProj.ReadTriplets();

     if (mPhProj.DPOriTriplets().DirOutIsInit())
     {
         mPhProj.SaveTriplets(*a3Set,false);
         delete a3Set;
         return EXIT_SUCCESS;
     }
     TimeSegm().SetIndex("cMakeArboTriplet");


     cMakeArboTriplet  aMk3(*a3Set,mDoCheck,mWBalance,mPhProj,*this);
     if (IsInit(&mLevelRand))
        aMk3.SetRand(mLevelRand);
     if (IsInit(&mWeigthEdge3))
        aMk3.WeigthEdge3() = mWeigthEdge3;
     if (IsInit(&mPerfectData))
        aMk3.PerfectData() = true;
     if (mPhProj.IsOriInDirInit())
        aMk3.PerfectOri() = true;
     if (IsInit(&mViscPose))
     {
         // tie-points must be provided for BA
         std::string aFolderTpts;
         if (mPhProj.DPMulTieP().DirInIsInit())
             aFolderTpts = mPhProj.DPMulTieP().DirIn().at(0);
         else
             MMVII_INTERNAL_ASSERT_always(mPhProj.DPMulTieP().DirInIsInit(),"Features not initialised");

         aMk3.TPFolder() = aFolderTpts;
         aMk3.ViscPose() = mViscPose;
     }

     // cAutoTimerSegm aTSRead(mTimeSegm,"cMakeArboTriplet");
     TimeSegm().SetIndex("MakeGraphPose");
     aMk3.MakeGraphPose();

     TimeSegm().SetIndex("PoseRef");
     aMk3.DoPoseRef();

     TimeSegm().SetIndex("MakeCnxTriplet");
     aMk3.MakeCnxTriplet();

     TimeSegm().SetIndex("TripletWeighting");
     aMk3.MakeWeightingGraphTriplet();

     TimeSegm().SetIndex("ComputeArbor");
     aMk3.ComputeArbor();

     if (mPhProj.DPOrient().DirOutIsInit())
     {
         StdOut() << " ========== Output Global Orientation  ========== " << std::endl;
         aMk3.SaveGlobSol();
     }

     aMk3.ShowStat();

     delete a3Set;
     return EXIT_SUCCESS;
}

tMMVII_UnikPApli Alloc_ArboTriplets(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ArboTriplets(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ArboTriplet
(
     "___OriPoseArboTriplet",
      Alloc_ArboTriplets,
      "Create arborescence of triplet (internal use essentially)",
      {eApF::Ori},
      {eApDT::Ori},
      {eApDT::Orient},
      __FILE__
);
}; //  namespace MMVII


