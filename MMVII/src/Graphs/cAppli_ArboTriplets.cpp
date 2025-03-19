#include "MMVII_GraphTriplets.h"

#include "MMVII_nums.h"
#include "MMVII_util_tpl.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Tpl_GraphAlgo_SPCC.h"
#include "MMVII_Tpl_GraphStruct.h"
#include "MMVII_Tpl_GraphAlgo_EnumCycles.h"
#include "MMVII_Interpolators.h"
#include "MMVII_Sensor.h"
#include "MMVII_PoseTriplet.h"
#include "MMVII_Tpl_GraphAlgo_Group.h"
#include "MMVII_Tpl_Images.h"


namespace MMVII
{

typedef std::pair<int,int>  tPairI;

class cNodeArborTriplets;  // the hierarchical decomposition in tree of triplet
class cMakeArboTriplet;    // class for computing everything
class  cSolLocNode;        // class for storing the pose of one image inside a cNodeArborTriplets
class cOneTripletMerge;    // class for storing the topological/links between 2 Nodes before merge


///   Store the pose and an ident (int) to the image
class  cSolLocNode
{
     public :
        cSolLocNode(tPoseR aPose,int aNumPose) : mPose (aPose), mNumPose (aNumPose) {}
        tPoseR mPose;
	int    mNumPose;
};

/// Store the common images, triplet, edges ...  that allow to compute transfer similitude
class cOneTripletMerge
{
    public :

       std::vector<tPairI>  mVLinkPose;  /// num inside mLocSols of poses linked by a triplet
       std::vector<tPairI>  mVLinkInTri; /// num inside the triplet (in [0,1,2])  of the link 
       std::vector<tPairI>  mVCommon;    /// num inside mLocSols of poses common to 2 children 
       int                  mNumTri;     /// num of the triplet in the graph of triplet
};

/// store the hierarchical decomposition 
class  cNodeArborTriplets : public cMemCheck
{
    public :
        typedef cNodeArborTriplets * tNodePtr;

        cNodeArborTriplets(cMakeArboTriplet &,const t3G3_Tree &,int aLevel);
        ~cNodeArborTriplets();
       
	void Test_RecursiveMerge();

        typedef  std::pair<tPoseR,tPoseR>  tPP;
        tPoseR     P_I1_to_I0(const tPairI aPairLoc) const;

    private :
        /// return the information on the link corresponding to a tripl
        cOneTripletMerge   MakeTriplet(const t3G3_Vertex & aVTri);

        /** estimate the rotation of the similitude transfer from information on link , note tha Common and link2
            do not come from cOneTripletMerge because come merge has been done 
        */
        tSim3dR EstimateSimTransfert
                (
                  const std::vector<tPairI>& aVPairCommon,
                  const std::vector<tPairI>& aVPairLink2,
                  const std::vector<cOneTripletMerge> &  aVLink3
                );

        /** idem, but estimate the similitude (after computing the rotation) */
        tRotR EstimateRotTransfert
              (
                  const std::vector<tPairI>& aVPairCommon,
                  const std::vector<tPairI>& aVPairLink2,
                  const std::vector<cOneTripletMerge> &  aVLink3
              );


        void ShowPose(const std::string & aPrefix) ;
	void DoMerge();
        void MakeIndexSol();
        void FreeIndexSol();
        void SortSol();
	void GetPoses(std::vector<int> &); 
        
        cSolLocNode *  SolOfInd(int aNumPose) ;

	int                       mDepth;  /// Leve
        t3G3_Tree                 mTree;
	std::array<tNodePtr,2>    mChildren;
	cMakeArboTriplet*         mPMAT;
	std::vector<cSolLocNode>  mLocSols;
	std::vector<int>          mVIndexes;
};




class cMakeArboTriplet
{
     public :
         
         cMakeArboTriplet(cTripletSet & aSet3,bool doCheck,tREAL8 aWBalance);
         ~cMakeArboTriplet();

         // make the graph on pose, using triplet as 3 edges
         void MakeGraphPose();

         /// reduce eventually the number of triplet by clustering them, used in cycle computation
         void DoClustering(int aNbMax,tREAL8 aDistCluster);
         ///  compute some quality criteria on triplet by loop closing on rotation
         void DoIterCycle(int aNbC);
         /// Compute the weight each triplet
         void DoTripletWeighting();
         /// Compute the connectivity on triangles
         void MakeGraphTriC();

         /// For each edge, compute the lowest cost triplet it belongs
         void ComputeArbor();

         /// Activate the simulation mode
         void SetRand(const std::vector<tREAL8> &);


	 tREAL8 WBalance() const {return mWBalance;}
	 tREAL8 & CostMergeTree() {return mCostMergeTree;}

         t3G3_Graph &       GO3()  {return mGTriC;}
         t3GOP &            GOP()  {return mGGPoses;}

         bool  PerfectData() const {return mPerfectData;}
     private :

         int KSom(c3G3_AttrV* aTri,int aK123) const;


         cTripletSet  &          mSet3;          ///< Initial triplet structure
         bool                    mDoCheck;       ///< do checking ....
         t2MapStrInt             mMapStrI;       ///< Maping name of pose / int used to import triplet
         t3GOP                   mGGPoses;       ///< Graph of pose
         t3G3_Graph              mGTriC;         ///<  Graph of triplet

         bool                    mDoRand;         ///< Do we generate random values
	 std::vector<tREAL8>     mLevelRand;      ///< Parameters of random values [RandOnTr,RandOnRot]
         bool                    mPerfectData;    ///< Are the triplet perfect with simulated pose
         cNodeArborTriplets *    mArbor;          ///< Tree  for hierarchical  split
         tREAL8                  mWBalance;       ///<  Weighting for balance the tree
         tREAL8                  mWeightTr;       ///<  Relative weight Tranlastion vs rotation for pose distance
	 tREAL8                  mCostMergeTree;  ///<  ???? 
};

/* ********************************************************* */
/*                                                           */
/*                     cNodeArborTriplets                    */
/*                                                           */
/* ********************************************************* */

cNodeArborTriplets::cNodeArborTriplets(cMakeArboTriplet & aMAT ,const t3G3_Tree & aTree,int aLevel) :
   mDepth     (aLevel),
   mTree      (aTree),
   mChildren  {0,0},
   mPMAT      (&aMAT)
{

	/*
       StdOut() << " |" ;
   for (const auto & aV : mTree.Vertices()) 
       StdOut()  << " " <<  aV->Attr().mKT;
   StdOut() << "\n";
   */

   if (!aTree.Edges().empty())
   {
      cWhichMax<t3G3_Edge*,tREAL8> aWME;
      for (const auto & anE : aTree.Edges())
      {
          std::array<t3G3_Tree,2>  a2T;
          aTree.Split(a2T,anE);
          int aN1 = a2T[0].Edges().size();
          int aN2 = a2T[1].Edges().size();
          tREAL8 aRatio =  std::min(aN1,aN2) / (1.0+std::max(aN1,aN2));
          tREAL8  aWeight = aMAT.WBalance() * aRatio + (1-aMAT.WBalance()) ;
          aWME.Add(anE,anE->AttrSym().mCost2Tri * aWeight);
          //aWME.Add(anE, aRatio);
      }

      std::array<t3G3_Tree,2>  a2T;
      aTree.Split(a2T,aWME.IndexExtre());
      aMAT.CostMergeTree() += a2T.at(0).Edges().size() +  a2T.at(1).Edges().size();
      for (size_t aKT=0 ; aKT<a2T.size() ; aKT++)
      {
          mChildren[aKT] = new cNodeArborTriplets(aMAT,a2T.at(aKT),aLevel+1);
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
    StdOut() << aPrefix ;
    for (int aK=0 ; aK< mDepth ; aK++)
       StdOut() << " |" ;
    StdOut() <<  aVPose << "\n";
}


void cNodeArborTriplets::Test_RecursiveMerge()
{
   MMVII_INTERNAL_ASSERT_tiny((mChildren.at(0) == nullptr) == (mChildren.at(1) == nullptr),"Test_RecursiveMerge, assert on desc");

    ShowPose("TRM");

    if (mChildren.at(0) == nullptr)
    {
       // Terminal node just put the triplet
       auto  aVecTri = mTree.Vertices();
       MMVII_INTERNAL_ASSERT_tiny(aVecTri.size()==1,"Test_RecursiveMerge, assert on desc");
       c3G3_AttrV & anATri = aVecTri.at(0)->Attr();
       for (size_t aK3=0 ; aK3<3 ; aK3++)
       {
	  int aNumPose = anATri.m3V.at(aK3)->Attr().Attr().mKIm;
	  tPoseR aPose = anATri.mT0->Pose(aK3).Pose();

          mLocSols.push_back(cSolLocNode(aPose,aNumPose));
       }
       MakeIndexSol();
       //SortSol();
    }
    else
    {
       for (auto & aChild :mChildren)
           aChild->Test_RecursiveMerge();
       DoMerge();
    }
}


void cNodeArborTriplets::MakeIndexSol()
{
    mVIndexes = std::vector<int>(mPMAT->GOP().NbVertex(),-1);
    for (size_t aK=0 ; aK<mLocSols.size() ; aK++)
       mVIndexes.at(mLocSols.at(aK).mNumPose) = aK;
}

void cNodeArborTriplets::FreeIndexSol()
{
   mVIndexes.clear();
   mLocSols.clear();
}


cOneTripletMerge  cNodeArborTriplets::MakeTriplet(const t3G3_Vertex & aVTri)
{
     cOneTripletMerge aResult;
     aResult.mNumTri = aVTri.Attr().mKT;

     cNodeArborTriplets & aN0 = *(mChildren.at(0));
     cNodeArborTriplets & aN1 = *(mChildren.at(1));

     std::vector<tPairI>  aVPair0;
     std::vector<tPairI>  aVPair1;

     for (size_t aK=0 ; aK<3 ; aK++)
     {
         int aKPose = aVTri.Attr().m3V.at(aK)->Attr().Attr().mKIm;
         int aI0 =  aN0.mVIndexes.at(aKPose);
         int aI1 =  aN1.mVIndexes.at(aKPose);

         if ((aI0>=0) || (aI1>=0))
         {
             if ((aI0>=0) && (aI1>=0))
             {
                 aResult.mVCommon.push_back(tPairI(aI0,aI1));
             }
             else if (aI0>=0)
             {
                 aVPair0.push_back(tPairI(aK,aI0));
             }
             else
             {
                 aVPair1.push_back(tPairI(aK,aI1));
             }
         }
     }

     for (const auto & [aK0,aI0] : aVPair0)
     {
         for (const auto & [aK1,aI1] : aVPair1)
         {
             aResult.mVLinkPose.push_back(tPairI(aI0,aI1));
             aResult.mVLinkInTri.push_back(tPairI(aK0,aK1));
         }
     }

     return aResult;
}


tRotR  cNodeArborTriplets::EstimateRotTransfert
             (
                  const std::vector<tPairI>& aVPairCommon,
                  const std::vector<tPairI>& aVPairLink2,
                  const std::vector<cOneTripletMerge> &  aVLink3
             )
{
    std::vector<tPoseR>   aVPose;
    cNodeArborTriplets & aN0 = *(mChildren.at(0));
    cNodeArborTriplets & aN1 = *(mChildren.at(1));

    for (const auto & [aI0,aI1] : aVPairCommon)
    {
         const tPoseR  & aPI_to_W0 = aN0.mLocSols[aI0].mPose;  //  I -> W0
         const tPoseR  & aPI_to_W1 = aN1.mLocSols[aI1].mPose;  //  I -> W1
        //  Im->W0 * (Im->W1)-1 = W1->W0
         tPoseR aP_W1_to_W0 =   aPI_to_W0 * aPI_to_W1.MapInverse();
         aVPose.push_back(aP_W1_to_W0);
    }

    for (const auto & [aI0Loc,aI1Loc] : aVPairLink2)
    {
         const tPoseR  & aPI0_to_W0 = aN0.mLocSols[aI0Loc].mPose;  //  I0 -> W0
         const tPoseR  & aPI1_to_W1 = aN1.mLocSols[aI1Loc].mPose;  //  I1 -> W1
         int aI0Glob = aN0.mLocSols[aI0Loc].mNumPose;
         int aI1Glob = aN1.mLocSols[aI1Loc].mNumPose;

         t3GOP_Vertex & aV0 = mPMAT->GOP().VertexOfNum(aI0Glob);
         t3GOP_Vertex & aV1 = mPMAT->GOP().VertexOfNum(aI1Glob);

         t3GOP_Edge & aE01 =  *aV0.EdgeOfSucc(aV1);
         tPoseR  aP_I1_to_I0 = aE01.AttrSym().Attr().mPoseRef2to1;
         if (!aE01.IsDirInit())
            aP_I1_to_I0 = aP_I1_to_I0.MapInverse();

         tPoseR aP_W1_to_W0  =   aPI0_to_W0 *  aP_I1_to_I0   * aPI1_to_W1.MapInverse();

         aVPose.push_back(aP_W1_to_W0);
    }

    for (const auto & aLnk3: aVLink3)
    {
        const cTriplet & a3 = *(mPMAT->GO3().VertexOfNum(aLnk3.mNumTri).Attr().mT0);
        for (size_t aKL=0 ; aKL< aLnk3.mVLinkPose.size() ; aKL++)
        {
             const auto & [aI0Loc,aI1Loc] = aLnk3.mVLinkPose.at(aKL);
             const auto & [aK0Tri,aK1Tri] = aLnk3.mVLinkInTri.at(aKL);
             const tPoseR  & aPI0_to_W0 = aN0.mLocSols[aI0Loc].mPose;  //  I0 -> W0
             const tPoseR  & aPI1_to_W1 = aN1.mLocSols[aI1Loc].mPose;  //  I1 -> W1

             const tPoseR  & aPI0_to_Tri = a3.Pose(aK0Tri).Pose();
             const tPoseR  & aPI1_to_Tri = a3.Pose(aK1Tri).Pose();
             tPoseR aP_I1_to_I0 =  aPI0_to_Tri.MapInverse() * aPI1_to_Tri;

             tPoseR aP_W1_to_W0  =   aPI0_to_W0 *  aP_I1_to_I0   * aPI1_to_W1.MapInverse();
             aVPose.push_back(aP_W1_to_W0);
        }
    }

    std::vector<tRotR>  aVRot;
    std::vector<tREAL8> aVWeight;
    for (const auto & aP : aVPose)
    {
       aVRot.push_back(aP.Rot());
       aVWeight.push_back(1.0);
    }
    tRotR aRotEstim = tRotR::Centroid(aVRot,aVWeight);

    StdOut()<<" COM="<< aVPairCommon <<" Liink= "<< aVPairLink2<<" NB3=" << aVLink3.size() << " R=" << aVPose.size() << "\n";
    for (const auto & aP : aVPose)
    {
         tREAL8 aD = aRotEstim.Dist(aP.Rot());
         StdOut() << "------ "  << aD << " ----------------\n";
         if (mPMAT->PerfectData())
            MMVII_INTERNAL_ASSERT_bench((aD<1e-5),"Rot-estimation on perfect data");
    }

    return aRotEstim;
}


tSim3dR cNodeArborTriplets::EstimateSimTransfert
             (
                  const std::vector<tPairI>& aVPairCommon,
                  const std::vector<tPairI>& aVPairLink2,
                  const std::vector<cOneTripletMerge> &  aVLink3
             )
{
    typedef   std::vector<cCplIV<tREAL8>> tVIV;
    typedef   cSparseVect<tREAL8>         tSV;
    bool  withLnk2=true;


    // StdOut() << " COM=" << aVPairCommon <<  " Liink= " << aVPairLink2 << " NB3=" <<   aVLink3.size()  << "\n";
    tRotR aRot_W1_to_W0 = EstimateRotTransfert(aVPairCommon,aVPairLink2,aVLink3);
    // make a pose corresponding to pure rotation, arbitrary translation because undefined
    // tPoseR aPose_W1_to_W0(cPt3dr(0,0,0),aRot_W1_to_W0);

    int aNbUnk = 4;
    if (withLnk2)
       aNbUnk += aVPairLink2.size() * 4;

    cLeasSqtAA<tREAL8> aSys(aNbUnk);
    

    cNodeArborTriplets & aN0 = *(mChildren.at(0));
    cNodeArborTriplets & aN1 = *(mChildren.at(1));

    for (const auto & [aI0,aI1] : aVPairCommon)
    {
         const tPoseR  & aPI_to_W0 = aN0.mLocSols[aI0].mPose;  //  I -> W0
         const tPoseR  & aPI_to_W1 = aN1.mLocSols[aI1].mPose;  //  I -> W1

         cPt3dr aC0 = aPI_to_W0.Tr();
         cPt3dr aC1 = aRot_W1_to_W0.Value(aPI_to_W1.Tr());

         //  C0 and C1 are two estimation of the center of the pose, they must be equal up
         //  to the global transfert (Tr,Lambda) from W1 to W0
         // 
         for (int aKC=0 ; aKC<3 ; aKC++)
         {
           //  aC0 = Tr + Lambda aC1
            aSys.PublicAddObservation(1.0, tSV(tVIV {{aKC,1.0},{3,aC1[aKC]}} ),aC0[aKC]);
         }
    }
    int aKEq = 4;

    if (withLnk2)
    {
        for (const auto & [aI0Loc,aI1Loc] : aVPairLink2)
        {
             const tPoseR  & aPI0_to_W0 = aN0.mLocSols[aI0Loc].mPose;  //  I0 -> W0
             const tPoseR  & aPI1_to_W1 = aN1.mLocSols[aI1Loc].mPose;  //  I1 -> W1

             cPt3dr aC0_in_W0  = aPI0_to_W0.Tr();
             cPt3dr aC1_in_W0  = aRot_W1_to_W0.Value(aPI1_to_W1.Tr());

FakeUseIt(aC0_in_W0);
FakeUseIt(aC1_in_W0);

             int aI0Glob = aN0.mLocSols[aI0Loc].mNumPose;
             int aI1Glob = aN1.mLocSols[aI1Loc].mNumPose;
             t3GOP_Vertex & aV0 = mPMAT->GOP().VertexOfNum(aI0Glob);
             t3GOP_Vertex & aV1 = mPMAT->GOP().VertexOfNum(aI1Glob);
             t3GOP_Edge & aE01 =  *aV0.EdgeOfSucc(aV1);
             tPoseR  aPEdge_I1_to_I0 = aE01.AttrSym().Attr().mPoseRef2to1;
             if (!aE01.IsDirInit())
                aPEdge_I1_to_I0 = aPEdge_I1_to_I0.MapInverse();

             tPoseR  aPI0_toTri = tPoseR::Identity();
             tPoseR  aPI1_toTri = aPEdge_I1_to_I0;

             tRotR  aR0_Tri_to_W0 = aPI0_to_W0.Rot() * aPI0_toTri.Rot().MapInverse();
             tRotR  aR1_Tri_to_W0 =  aRot_W1_to_W0 * aPI1_to_W1.Rot() * aPI1_toTri.Rot().MapInverse();
             tRotR aR_Tri_to_W0 = aR0_Tri_to_W0.Centroid(aR1_Tri_to_W0);
 
             if (mPMAT->PerfectData())
             {
                 tREAL8 aDist = aR0_Tri_to_W0.Dist(aR1_Tri_to_W0);
                 MMVII_INTERNAL_ASSERT_bench((aDist<1e-5),"Rot-estimation Tri->W0 on perfect data");
             }

             cPt3dr aCTri0_In_W0 = aR_Tri_to_W0.Value(aPI0_toTri.Tr());
             cPt3dr aCTri1_In_W0 = aR_Tri_to_W0.Value(aPI1_toTri.Tr());

             for (int aKC=0 ; aKC<3 ; aKC++)
             {
                //   aC0_in_W0  = TrTri + LambdaTri * aCTri0_In_W0
                tVIV aSV0 {{aKEq+aKC,1.0},{aKEq+3,aCTri0_In_W0[aKC]}} ;
                aSys.PublicAddObservation(1.0, tSV(aSV0 ),aC0_in_W0[aKC]);
                //   Tr + Lambda * aC1_in_W0  = TrTri + LambdaTri * aCTri1_In_W0
                tVIV aSV1 {    {aKEq+aKC,1.0},{aKEq+3,aCTri1_In_W0[aKC]},  // TrTri + LambdaTri * aCTri1_In_W0
                               {aKC,-1.0},{3,-aC1_in_W0[aKC]}              // - (  Tr + Lambda * aC1_in_W0)
                          };
                aSys.PublicAddObservation(1.0, tSV(aSV1 ),0.0);
             }

             aKEq += 4;
        }
    }

    cDenseVect<tREAL8> aSol = aSys.PublicSolve();
    tREAL8 aLambda = aSol(3);
    cPt3dr aTr(aSol(0),aSol(1),aSol(2));


    return tSim3dR(aLambda,aTr,aRot_W1_to_W0);
}


void cNodeArborTriplets::DoMerge()
{
     cNodeArborTriplets & aN0 = *(mChildren.at(0));
     cNodeArborTriplets & aN1 = *(mChildren.at(1));

         ShowPose("DoMx :");
     aN0.ShowPose("DoM0 :");
     aN1.ShowPose("DoM1 :");

     std::vector<tPairI>  aVPairCommon;  //  Store data for vertex present in 2 children
     std::vector<tPairI>  aVPairLink2;   // store data for edges between 2 children (the 3 vertex being out)
     std::vector<bool>               aSetIndexTri(mPMAT->GO3().NbVertex(),false);  // marqer to test triplet once
     std::vector<bool>               aSetIndComN0(aN0.mLocSols.size(),false);      // marqer to have common vertex once
     std::vector<cOneTripletMerge>    aVLink3;  // store triplet with 3 vertices doing the link

     for (const auto & aSol0 : aN0.mLocSols)
     {
         const auto & aVertexPose = mPMAT->GOP().VertexOfNum(aSol0.mNumPose);
         const c3GOP_AttrV aAttrPose = aVertexPose.Attr().Attr();
         for (const auto & aNumTri : aAttrPose.mTriBelongs)
         {
             if (!aSetIndexTri.at(aNumTri))
             {

                 aSetIndexTri.at(aNumTri) = true;
                 cOneTripletMerge  a1TM = MakeTriplet(mPMAT->GO3().VertexOfNum(aNumTri));
                 for (const auto & [aI0,aI1] : a1TM.mVCommon)
                 {
                     if  (! aSetIndComN0.at(aI0))
                     {
                         aSetIndComN0.at(aI0) = true;
                         aVPairCommon.push_back(tPairI(aI0,aI1));
                     }
                 }
                 if ((! a1TM.mVLinkPose.empty())  && (a1TM.mVCommon.empty()))
                 {
                     if (a1TM.mVLinkPose.size() == 1)
                     {
                        aVPairLink2.push_back(a1TM.mVLinkPose.at(0));
                     }
                     else
                     {
                        aVLink3.push_back(a1TM);
                     }
                 }
             }
         }
     }
     {
         std::sort(aVPairLink2.begin(),aVPairLink2.end());
         auto aEndUniqueLnk2 = std::unique(aVPairLink2.begin(),aVPairLink2.end());
         aVPairLink2.resize(aEndUniqueLnk2 - aVPairLink2.begin());
     }

     tSim3dR  aSimTransfer = EstimateSimTransfert(aVPairCommon,aVPairLink2,aVLink3);

     // Put Sol0 that are not in Sol1
     for (const auto & aSol0 : aN0.mLocSols)
     {
         if (aN1.SolOfInd(aSol0.mNumPose)==nullptr)
         {
              mLocSols.push_back(aSol0);
         }
     }

     // Put Sol1, use transfert, and separate case in Sol0 or not
     for (const auto & aSol1 : aN1.mLocSols)
     {
         tPoseR  aPoseInS0  =    TransfoPose(aSimTransfer,aSol1.mPose);
         cSolLocNode * aSol0 = aN0.SolOfInd(aSol1.mNumPose);
         if (aSol0)
         {
            if (mPMAT->PerfectData())
            {
                 tREAL8 aDist = aPoseInS0.DistPose(aSol0->mPose,1.0);
                 MMVII_INTERNAL_ASSERT_bench((aDist<1e-5),"Sim-Transfer on perfect data");
            }
            aPoseInS0 = tPoseR::Centroid({aPoseInS0,aSol0->mPose},{1.0,1.0});
         }

         mLocSols.push_back(cSolLocNode(aPoseInS0,aSol1.mNumPose));
     }

/*
     StdOut() << "NP=";
     for (const auto & aSol : mLocSols)
         StdOut() << " " << aSol.mNumPose;
     StdOut() << "\n";
*/
getchar();

     // --------------------------------------
     for (auto & aChild :mChildren)
     {
          aChild->FreeIndexSol();
     }
     MakeIndexSol();
}

cSolLocNode * cNodeArborTriplets::SolOfInd(int  anIndAbs) 
{
   int anIndRel = mVIndexes.at(anIndAbs);
   if (anIndRel>=0)
      return  &mLocSols.at(anIndRel);


   return nullptr;
}


void cNodeArborTriplets::GetPoses(std::vector<int> & aResult)
{
   aResult.clear();
   size_t aFlagPose = mPMAT->GOP().AllocBitTemp();

   for (const auto & aVertexTri : mTree.Vertices()) 
   {
        auto & anAttr = aVertexTri->Attr();
	for (auto & aVertexPos : anAttr.m3V)
	{
		if (!aVertexPos->BitTo1(aFlagPose))
		{
                   aVertexPos->SetBit1(aFlagPose);
                   aResult.push_back(aVertexPos->Attr().Attr().mKIm);
		}
	}
   }
   for (const auto aNumPose : aResult)
   {
       mPMAT->GOP().VertexOfNum(aNumPose).SetBit0(aFlagPose);
   }
   mPMAT->GOP().FreeBitTemp(aFlagPose);
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
/*                     cAppli_ArboTriplets                   */
/*                                                           */
/* ********************************************************* */


cMakeArboTriplet::cMakeArboTriplet(cTripletSet & aSet3,bool doCheck,tREAL8 aWBalance) :
   mSet3        (aSet3),
   mDoCheck     (doCheck),
   mGGPoses     (), // false=> not 4 simul
   mDoRand      (false),
   mLevelRand   {0.0,0.0},
   mPerfectData (false),
   mArbor       (nullptr),
   mWBalance    (aWBalance),
   mWeightTr    (0.5)
{
}

cMakeArboTriplet::~cMakeArboTriplet()
{
   delete mArbor;
}
void cMakeArboTriplet::SetRand(const std::vector<tREAL8> & aLevelRand)
{
   mDoRand = true;
   mLevelRand = aLevelRand;
   mPerfectData = (aLevelRand.at(0)==0) && ( (aLevelRand.at(1)==0));
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
   StdOut() << "Nb3= " << mSet3.Set().size() << " NBI=" << mMapStrI.size() << "\n";


   // In case we want to make some test with random rot, create the "perfect" rot of each pose
   std::vector<tRotR> aVRandR;
   for (size_t aKR=0; aKR<mMapStrI.size() ; aKR++)
       aVRandR.push_back(tRotR::RandomRot());

   //  Add the vertex corresponding to each poses in Graph-Pose
   for (size_t aKIm=0 ; aKIm<mMapStrI.size() ; aKIm++)
   {
       tPoseR aRandGT(cPt3dr::PRandInSphere(),tRotR::RandomRot());
       mGGPoses.AddVertex(c3GOP_AttrV(aKIm,aRandGT));
   }

   //  Parse all the triplet; for each triplet  Add 3 edges of Grap-Pose ;
   //  also memorize the edges in the triplet for each, mCnxE
   for (size_t aKT=0; aKT<mGTriC.NbVertex() ; aKT++)
   {
        c3G3_AttrV & aTriC =   mGTriC.VertexOfNum(aKT).Attr();
        cTriplet & a3 = *(aTriC.mT0);
        if (mDoRand)  
	{
	    // in simul we must take into account that each triplet is in its own arbitrary system  W2L , Word -> Loc
            tSim3dR   aRandSim= tSim3dR::RandomSim3D(2.0,2.0);
            // parse the 3 pair of consecutive poses
	    for (int aK3=0 ; aK3<3 ; aK3++)
	    {
                cView&  aView = a3.Pose(aK3);
                tPoseR & aP = aView.Pose();
		// initialize local pose to ground truth
                int aInd =  mMapStrI.Obj2I(aView.Name());
		t3GOP_Vertex & aPoseV = mGGPoses.VertexOfNum(aInd);
		aP = aPoseV.Attr().Attr().mGTRand;
		// Firts create small perturbations of "perfect" values of "Tr/Rot"
		cPt3dr aTr = aP.Tr() + cPt3dr::PRandInSphere() * mLevelRand.at(0);
		tRotR aRot = aP.Rot()* tRotR::RandomSmallElem(mLevelRand.at(1));
		// Now put everyting in the local system and   finally save the result
		aP = TransfoPose(aRandSim,tPoseR(aTr,aRot));
	    }
	}
        for (size_t aK3=0 ; aK3<3 ; aK3++)
        {
            // extract  the 2 vertices corresponding to poses
            const cView&  aView1 = a3.Pose(aK3);
            const cView&  aView2 = a3.Pose((aK3+1)%3);
            int aI1 =  mMapStrI.Obj2I(aView1.Name());
            int aI2 =  mMapStrI.Obj2I(aView2.Name());
            t3GOP_Vertex & aV1  = mGGPoses.VertexOfNum(aI1);
            t3GOP_Vertex & aV2  = mGGPoses.VertexOfNum(aI2);

            aTriC.m3V[aK3] = &aV1;

            // extract relative poses 
            tPoseR  aP1toW = aView1.Pose();
            tPoseR  aP2toW = aView2.Pose();
            //  Ori_2->1  = Ori_G->1  o Ori_2->G    ( PL2 -> PG -> PL1)
            tPoseR  aP2to1 = aP1toW.MapInverse()  *  aP2toW;
	 
            tRotR aR2to1 = aP2to1.Rot();          

            mGGPoses.AddHyp(aV1,aV2,aR2to1,1.0,c3GOP_1Hyp(aKT));

            t3GOP_Edge * anE_12 = aV1.EdgeOfSucc(aV2);
            t3GOP_Edge * anE_DirInit = anE_12->EdgeInitOr();
            aTriC.mCnxE.at(aK3) = anE_DirInit;
             
            tPoseR aPoseEdge =  anE_12->IsDirInit()  ? aP2to1 : aP2to1.MapInverse();
            anE_DirInit->AttrSym().Attr().mListP2to1.push_back(aPoseEdge);
        }
   }
}


void cMakeArboTriplet::DoClustering(int aNbMax,tREAL8 aDistCluster)
{
   // call the clustering of Graph - group
   mGGPoses.DoClustering(aNbMax,aDistCluster);

   // print some stat
   int aNbH0 = 0;
   int aNbHC = 0;
   int aNbE = 0;
   for (auto & anE : mGGPoses.AllEdges_DirInit())
   {
      t3GOP_EdAttS & anAttr =  anE->AttrSym();
      aNbH0 += anAttr.ValuesInit().size();
      aNbHC += anAttr.ValuesClust().size();
      aNbE++;

      anAttr.Attr().ComputePoseRef(mWeightTr);
      const auto & aListP = anAttr.Attr().mListP2to1;
      if (!aListP.empty())
      {
          tPoseR aPoseRef = anAttr.Attr().mPoseRef2to1;
          for (const auto & aPose : aListP)
          {
              if (mPerfectData)
              {
                 tREAL8 aDist = aPoseRef.DistPoseRel(aPose,1.0);
                 MMVII_INTERNAL_ASSERT_bench((aDist<1e-5),"Pose reference on perfect data");
              }
          }
      }

      StdOut() << " NBE=" << anAttr.Attr().mListP2to1.size() << "\n";
   }

   StdOut() << "NBH/Edge , Init=" << aNbH0 /double(aNbE) << " Clustered=" << aNbHC/double(aNbE)  << "\n";
}

void cMakeArboTriplet::DoIterCycle(int aNbIter)
{
     // call the IterCycles-method of graph group, that compute quality on edges-hyp
     for (int aK=0 ; aK<aNbIter ; aK++)
     {
         mGGPoses.OneIterCycles(3,1.0,false);
     }
     StdOut() << "DONE DoIterCycle \n";

      cStdStatRes aStat;
      for (const auto & anE : mGGPoses.AllEdges_DirInit())
      {
          const  t3GOP_EdAttS & anAttr =  anE->AttrSym();
          //StdOut() << "-----------------------------------------------\n";
          for (const auto & aH : anAttr.ValuesClust())
          {
              aStat.Add(aH.mWeightedDist.Average() );
          }
       }
       if (1)
       {
           for (const auto & aProp : {0.1,0.5,0.9})
               StdOut() << " Prop=" << aProp << " Res=" << aStat.ErrAtProp(aProp) << "\n";
       }

}


void cMakeArboTriplet::DoTripletWeighting()
{
    for (size_t aKT=0 ; aKT<mGTriC.NbVertex() ; aKT++)
    {
        auto & aTriAttr = mGTriC.VertexOfNum(aKT).Attr();

        // store the weigth of 3 edges it belongs
        std::vector<tREAL8>  aVectCost;
        for (size_t aKE=0 ; aKE<3 ; aKE++)
        {
             const auto & anAttr = aTriAttr.mCnxE.at(aKE)->AttrSym(); // Attr of 1 Edges
             //  extract the hypothesis that contain this triangle
             const auto aHyp0  =  anAttr.HypOfNumSet([aKT] (const auto & anAttr) {return anAttr.mNumSet==(int)aKT;});

             if (aHyp0->IsMarked()) // if has been clustered
             {
                 const auto aHypC =  anAttr.HypCompOfH0(*aHyp0);  // extract clustered hyp associated
                 aVectCost.push_back(aHypC->mWeightedDist.Average());
             }
        }

        // compute some kind of soft-max, require cost found for the 3 edge
        if (aVectCost.size()==3)
        {
            std::sort(aVectCost.begin(),aVectCost.end()); // sort to weight by rank
            cWeightAv<tREAL8,tREAL8> aWCost;  // compute weighted cost
            tREAL8 aExp = 2.0;
            for (size_t aKP=0 ; aKP<3 ;aKP++)
                aWCost.Add(std::pow(aExp,aKP),aVectCost.at(aKP));
            aTriAttr.mOk = true;
            aTriAttr.mCostIntr = aWCost.Average() ;
        }
        else
        {
        }
    }
}

void cMakeArboTriplet::MakeGraphTriC()
{
   tREAL8 aWeighMax = 0.75;

   for (size_t aKT1=0; aKT1<mGTriC.NbVertex() ; aKT1++)
   {
       auto & aVert1  = mGTriC.VertexOfNum(aKT1);
       auto & aTriAttr1 = aVert1.Attr();
       tREAL8 aC1 = aTriAttr1.mCostIntr;
       for (size_t aKE1=0 ; aKE1<3 ; aKE1++)
       {
           const auto & anAttr = aTriAttr1.mCnxE.at(aKE1)->AttrSym(); // Attr of 1 Edges
           for (const auto & aHyp : anAttr.ValuesInit())
           {
               if (aHyp.IsMarked()) // if has been clustered
               {
                  size_t  aKT2 = aHyp.mAttr.mNumSet;
                  if (aKT1<aKT2)
                  {
                     auto & aVert2  = mGTriC.VertexOfNum(aKT2);
                     auto & aTriAttr2 = aVert2.Attr();
                     tREAL8 aC2 = aTriAttr2.mCostIntr;
                     size_t aKE2 = aTriAttr2.GetIndexVertex(aTriAttr1.m3V.at(aKE1));
                     //auto & aTriAttr1 = mGTriC.VertexOfNum(aKT1).Attr();
                     tREAL8 aCost12 = std::min(aC1,aC2)*(1-aWeighMax)+std::max(aC1,aC2)*aWeighMax;
                     mGTriC.AddEdge
                     (
                          aVert1,aVert2,
                          c3G3_AttrOriented(aKE1),c3G3_AttrOriented(aKE2),
                          c3G3_AttrSym(aCost12)
                     );
                  }
               }
           }
       }
   }
   auto  aListCC = cAlgoCC<t3G3_Graph>::All_ConnectedComponent(mGTriC,cAlgo_ParamVG<t3G3_Graph>());
   StdOut() << "Number of connected compon Triplet-Graph= " << aListCC.size() << "\n";
   // to see later, we can probably analyse CC by CC, but it would be more complicated (already enough complexity ...)
   MMVII_INTERNAL_ASSERT_tiny(aListCC.size(),"non connected triplet-graph");
}



void cMakeArboTriplet::ComputeArbor()
{

   for (size_t aKTri=0; aKTri<mGTriC.NbVertex() ; aKTri++)
   {
       auto & aTriVertex  = mGTriC.VertexOfNum(aKTri);
       c3G3_AttrV & aAttrTriV  = aTriVertex.Attr();
       tREAL8 aCost = aAttrTriV.mCostIntr;
       for (int aKV=0 ; aKV<3 ; aKV++)
       {
           c3GOP_AttrV & aAttrPose = aAttrTriV.m3V.at(aKV)->Attr().Attr();
           aAttrPose.mWMinTri.Add(aKTri,aCost);
           aAttrPose.mTriBelongs.push_back(aKTri);
       }
   }
   int aNbTripInit = 0;
   std::vector<t3G3_Vertex*> aVectTriMin;
   for (size_t aKPose=0 ; aKPose<mGGPoses.NbVertex() ; aKPose++)
   {
       int aKTriplet = mGGPoses.VertexOfNum(aKPose).Attr().Attr().mWMinTri.IndexExtre();
       std::vector<int> & aLMin = mGTriC.VertexOfNum(aKTriplet).Attr().mVPoseMin;

       if (aLMin.empty())
       {
          // mArbor.NewTerminalNode(aKTriplet);
          aNbTripInit ++;
          aVectTriMin.push_back(&mGTriC.VertexOfNum(aKTriplet));
       }
       aLMin.push_back(aKPose);
   }
   StdOut() << "NB TRIPLET=" << aNbTripInit << "\n";


   auto aWeighting = Tpl_WeithingSubGr(&mGTriC,[](const auto & anE) {return anE.AttrSym().mCost2Tri;});

   cAlgoSP<t3G3_Graph>::tForest  aForest(mGTriC);
   cAlgoSP<t3G3_Graph>  anAlgo;
   anAlgo.MinimumSpanningForest(aForest,mGTriC,mGTriC.AllVertices(),aWeighting);
   
   // dont handle 4 now
   MMVII_INTERNAL_ASSERT_tiny(aForest.VTrees().size()==1,"Not single forest");// litle checj
   const auto & aGlobalTree = *(aForest.VTrees().begin());
   StdOut() << "NB TREE=" << aForest.VTrees().size()  << " SzTree1=" << aGlobalTree.Edges().size() << "\n";


   
   cSubGraphOfEdges<t3G3_Graph>  aSubGrTree(mGTriC,aGlobalTree.Edges());
   cSubGraphOfVertices<t3G3_Graph>  aSubGrAnchor(mGTriC,aVectTriMin); 


   cAlgoPruningExtre<t3G3_Graph> aAlgoSupExtr(mGTriC,aSubGrTree,aSubGrAnchor);
   StdOut() <<  "NB EXTR SUPR=" << aAlgoSupExtr.Extrem().size() << "\n";

   std::vector<t3G3_Edge*>  aSetEdgeKern;
   cVG_OpBool<t3G3_Graph>::EdgesMinusVertices(aSetEdgeKern,aGlobalTree.Edges(),aAlgoSupExtr.Extrem());


   StdOut() <<  "NB KERNEL=" << aSetEdgeKern.size() << "\n";

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
   t3G3_Tree  aTreeKernel(aSetEdgeKern);
   mCostMergeTree = 0.0;
   mArbor = new cNodeArborTriplets(*this,aTreeKernel,0);
   StdOut() << "CostMerge " << mCostMergeTree << "\n";
   mArbor->Test_RecursiveMerge();
}





int cMakeArboTriplet::KSom(c3G3_AttrV* aTriC,int aK123) const
{
   return mMapStrI.Obj2I(aTriC->mT0->Pose(aK123).Name());
}

class cAppli_ArboTriplets : public cMMVII_Appli
{
     public :

        cAppli_ArboTriplets(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :
        cPhotogrammetricProject   mPhProj;
        int                       mNbMaxClust;
        tREAL8                    mDistClust;
        int                       mNbIterCycle;
	std::vector<tREAL8>       mLevelRand;
        bool                      mDoCheck;  
        tREAL8                    mWBalance;
};


cAppli_ArboTriplets::cAppli_ArboTriplets(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mNbMaxClust  (5),
    mDistClust   (0.02),
    mNbIterCycle (3),
    mDoCheck     (true),
    mWBalance    (1.0)
{
}

cCollecSpecArg2007 & cAppli_ArboTriplets::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
              <<  mPhProj.DPOriTriplets().ArgDirInMand("Input orientation for calibration")
           ;
}

cCollecSpecArg2007 & cAppli_ArboTriplets::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return    anArgOpt
          << AOpt2007(mNbMaxClust,"NbMaxClust","Number max of rot in 1 cluster",{eTA2007::HDV})
          << AOpt2007(mDistClust,"DistClust","Distance in clustering",{eTA2007::HDV})
          << AOpt2007(mLevelRand,"LevelRand","Level of random if simulation [Trans,Rot]", {{eTA2007::ISizeV,"[2,2]"}})
          << AOpt2007(mDoCheck,"DoCheck","do some checking on result",{eTA2007::HDV,eTA2007::Tuning})
          << AOpt2007(mWBalance,"WBalance","Weight for balancing trees, 0 NONE, 1 Max",{eTA2007::HDV})
   ;
}

int cAppli_ArboTriplets::Exe()
{
     mPhProj.FinishInit();

     cTripletSet *  a3Set =  mPhProj.ReadTriplets();
     cMakeArboTriplet  aMk3(*a3Set,mDoCheck,mWBalance);
     if (IsInit(&mLevelRand))
        aMk3.SetRand(mLevelRand);

     aMk3.MakeGraphPose();
     aMk3.DoClustering(mNbMaxClust,mDistClust);
     aMk3.DoIterCycle(mNbIterCycle);
     aMk3.DoTripletWeighting();
     aMk3.MakeGraphTriC();
     aMk3.ComputeArbor();

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


