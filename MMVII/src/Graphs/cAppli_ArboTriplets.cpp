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


namespace MMVII
{


class cMakeArboTriplet;
class cNodeArborTriplets;

class  cSolLocNode
{
     public :
        tPoseR mPose;
	int    mNumPose;
};

class  cNodeArborTriplets : public cMemCheck
{
    public :
        typedef cNodeArborTriplets * tNodePtr;

        cNodeArborTriplets(cMakeArboTriplet &,const t3G3_Tree &,int aLevel);
        ~cNodeArborTriplets();
       
	void TestMergeRot();

    private :
	void GetPoses(std::vector<int> &); 

	int                       mLevel;
        t3G3_Tree                 mTree;
	std::array<tNodePtr,2>    mChildren;
	cMakeArboTriplet*         mPMAT;
	std::vector<cSolLocNode>  mLocSols;
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

         t3G3_Graph &       GO3() ;
         t3GOP &            GOP()  {return mGGPoses;}

     private :

         int KSom(c3G3_AttrV* aTri,int aK123) const;


         cTripletSet  &          mSet3;
         bool                    mDoCheck;
         t2MapStrInt             mMapStrI;
         t3GOP                   mGGPoses;
         t3G3_Graph              mGTriC;
         bool                    mDoRand;
	 std::vector<tREAL8>     mLevelRand;
         cNodeArborTriplets *    mArbor;
         tREAL8                  mWBalance;

	 tREAL8                  mCostMergeTree;
};

/* ********************************************************* */
/*                                                           */
/*                     cNodeArborTriplets                    */
/*                                                           */
/* ********************************************************* */

cNodeArborTriplets::cNodeArborTriplets(cMakeArboTriplet & aMAT ,const t3G3_Tree & aTree,int aLevel) :
   mLevel     (aLevel),
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

void cNodeArborTriplets::TestMergeRot()
{
   MMVII_INTERNAL_ASSERT_tiny((mChildren.at(0) == nullptr) == (mChildren.at(1) == nullptr),"TestMergeRot, assert on desc");

    std::vector<int> aVPose; GetPoses(aVPose);
    for (int aK=0 ; aK< mLevel ; aK++)
       StdOut() << " |" ;
    StdOut() <<  aVPose << "\n";

    if (mChildren.at(0) == nullptr)
    {
       auto  aVecTri = mTree.Vertices();
       MMVII_INTERNAL_ASSERT_tiny(aVecTri.size()==1,"TestMergeRot, assert on desc");
       c3G3_AttrV & anATri = aVecTri.at(0)->Attr();
       for (size_t aK3=0 ; aK3<3 ; aK3++)
       {
          cSolLocNode aSol;
	  aSol.mNumPose = anATri.m3V.at(aK3)->Attr().Attr().mKIm;
	  aSol.mPose = anATri.mT0->Pose(aK3).Pose();

          mLocSols.push_back(aSol);
       }
    }
    else
    {
       for (auto & aChild :mChildren)
           aChild->TestMergeRot();
    }
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
/*                     cAppli_ArboTriplets                   */
/*                                                           */
/* ********************************************************* */


cMakeArboTriplet::cMakeArboTriplet(cTripletSet & aSet3,bool doCheck,tREAL8 aWBalance) :
   mSet3        (aSet3),
   mDoCheck     (doCheck),
   mGGPoses     (), // false=> not 4 simul
   mDoRand      (false),
   mLevelRand   {0.0,0.0},
   mArbor       (nullptr),
   mWBalance    (aWBalance)
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
	    tRotR aRandW2L = tRotR::RandomRot();
	    tREAL8 aScaleW2L = RandInInterval(0.5,1.5);
	    cPt3dr aTransW2L = cPt3dr::PRandInSphere() * 2.0;
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
		cPt3dr aTr = aP.Tr() + cPt3dr::PRandInSphere() * mLevelRand.at(1);
		tRotR aRot = aP.Rot()* tRotR::RandomSmallElem(mLevelRand.at(0));
		// Now put everyting in the local system
		aTr = aTransW2L  + aTr*aScaleW2L;
		aRot =  aRandW2L * aRot;
                // finally save the result
		aP = tPoseR(aTr,aRot);
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

            // extract relative poses (eventually adapt when simul)
	 
            tRotR  aR1toW = aView1.Pose().Rot();
            tRotR  aR2toW = aView2.Pose().Rot();
            // formula forcomputing relative pose
            //  Ori_2->1  = Ori_G->1  o Ori_2->G    ( PL2 -> PG -> PL1)
            tRotR aR2to1 = aR1toW.MapInverse() * aR2toW;

            mGGPoses.AddHyp(aV1,aV2,aR2to1,1.0,c3GOP_1Hyp(aKT));

            t3GOP_Edge * anE = aV1.EdgeOfSucc(aV2)->EdgeInitOr();
            aTriC.mCnxE.at(aK3) = anE;
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
   for (const auto & anE : mGGPoses.AllEdges_DirInit())
   {
      const  t3GOP_EdAttS & anAttr =  anE->AttrSym();
      aNbH0 += anAttr.ValuesInit().size();
      aNbHC += anAttr.ValuesClust().size();
      aNbE++;
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
       auto & aAttrTrip  = mGTriC.VertexOfNum(aKTri).Attr();
       tREAL8 aCost = aAttrTrip.mCostIntr;
       for (int aKV=0 ; aKV<3 ; aKV++)
           aAttrTrip.m3V.at(aKV)->Attr().Attr().mWMinTri.Add(aKTri,aCost);
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
   mArbor->TestMergeRot();
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
          << AOpt2007(mLevelRand,"LevelRand","Level of random if simulation [Rot,Trans]", {{eTA2007::ISizeV,"[2,2]"}})
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


