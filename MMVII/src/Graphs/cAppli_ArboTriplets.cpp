#include "MMVII_nums.h"
#include "MMVII_util_tpl.h"
// #include "MMVII_Geom2D.h"
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


/* ********************************************************* */
/*                                                           */
/*                     cAppli_ArboTriplets                   */
/*                                                           */
/* ********************************************************* */




class cMAT_1Hyp
{
   public :
        explicit cMAT_1Hyp(int aNumSet) : mNumSet (aNumSet) {}
        cMAT_1Hyp() : mNumSet(-1) {}
        int  mNumSet;
};

class cMAT_Vertex
{
    public :
         cMAT_Vertex()           {}
         cWhichMin<int,tREAL8>   mWMinTri;
};


typedef cGroupGraph<tRotR,cMAT_Vertex,cEmptyClass,cEmptyClass,cMAT_1Hyp>       tGrPoses;

typedef typename tGrPoses::tAttrS                           tAS_GGRR;
typedef typename tGrPoses::tVertex                          tV_GGRR;
typedef typename tGrPoses::tEdge                            tE_GGRR;
typedef tE_GGRR*                                         tEdgePtr;
typedef std::array<tE_GGRR*,3>                           t3E_GGRR;
typedef std::array<tV_GGRR*,3>                           t3V_GGRR;

class cAtOri_3GGRR
{
   public :
        cAtOri_3GGRR(size_t aKE) : mKE(aKE) {}
        size_t mKE;
   private :
};
 
class cAtSym_3GGRR
{
   public :
      cAtSym_3GGRR(tREAL8 aCost) : mCost2Tri(aCost) {}
      tREAL8 mCost2Tri;
   private :
};


class cTri_GGRR
{
   public :
        cTri_GGRR (const cTriplet* aT0,int aKT) :
            mT0         (aT0),
            mKT         (aKT),
            mOk         (false),
            mCostIntr   (1e10) 
        {
        }
        size_t GetIndexVertex(const tV_GGRR* aV) const
        {
             for (size_t aKV=0 ; aKV<3 ; aKV++)
                 if (m3V.at(aKV) == aV)
                    return aKV;
             MMVII_INTERNAL_ERROR("cTri_GGRR::GetIndexVertex");
             return 3;
        }

        const cTriplet*  mT0;
        int              mKT;
        t3E_GGRR         mCnxE;      // the 3 edges 
        t3V_GGRR         m3V;        // the 3 vertices 
        bool             mOk;        // for ex, not OK if on of it edges was not clustered
        tREAL8           mCostIntr;  // intrisiq cost
        std::vector<int> mVPoseMin;  // List of pose consider this a best triplet
};

typedef cVG_Graph<cTri_GGRR, cAtOri_3GGRR,cAtSym_3GGRR> tGrTriplet;






class cMakeArboTriplet
{
     public :
         
         cMakeArboTriplet(const cTripletSet & aSet3,bool doCheck);

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
         void ComputeMinTri();

         /// Activate the simulation mode
         void SetRand(tREAL8 aLevelRand);

         typedef cTri_GGRR * t3CPtr;
         typedef tGrTriplet::tEdge    tEdge3;
         typedef tGrTriplet::tVertex  tVertex3;
     private :

         int KSom(t3CPtr aTri,int aK123) const;


         const cTripletSet  &    mSet3;
         bool                    mDoCheck;
         t2MapStrInt             mMapStrI;
         tGrPoses                mGGPoses;
         tGrTriplet              mGTriC;
         bool                    mDoRand;
         tREAL8                  mLevelRand;
         //cGlobalArborBin         mArbor;
};


cMakeArboTriplet::cMakeArboTriplet(const cTripletSet & aSet3,bool doCheck) :
   mSet3      (aSet3),
   mDoCheck   (doCheck),
   mGGPoses   (false), // false=> not 4 simul
   mDoRand    (false),
   mLevelRand (0.0)
{
}

void cMakeArboTriplet::SetRand(tREAL8 aLevelRand)
{
   mDoRand = true;
   mLevelRand = aLevelRand;
}


void cMakeArboTriplet::MakeGraphPose()
{

   // create vertices of mGTriC  & compute map NamePose/Int (in mMapStrI)
   for (size_t aKT=0; aKT<mSet3.Set().size() ; aKT++)
   {
        const cTriplet & a3 = mSet3.Set().at(aKT);
        mGTriC.NewVertex(cTri_GGRR(&a3,aKT));
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
       mGGPoses.AddVertex(cMAT_Vertex());
   }

   //  Parse all the triplet; for each triplet  Add 3 edges of Grap-Pose ;
   //  also memorize the edges in the triplet for each, mCnxE
   for (size_t aKT=0; aKT<mGTriC.NbVertex() ; aKT++)
   {
        cTri_GGRR & aTriC =   mGTriC.VertexOfNum(aKT).Attr();
        const cTriplet & a3 = *(aTriC.mT0);
        // parse the 3 pair of consercutive poses
        for (size_t aK3=0 ; aK3<3 ; aK3++)
        {
            // extract  the 2 vertices corresponding to poses
            const cView&  aView1 = a3.Pose(aK3);
            const cView&  aView2 = a3.Pose((aK3+1)%3);
            int aI1 =  mMapStrI.Obj2I(aView1.Name());
            int aI2 =  mMapStrI.Obj2I(aView2.Name());
            tV_GGRR & aV1  = mGGPoses.VertexOfNum(aI1);
            tV_GGRR & aV2  = mGGPoses.VertexOfNum(aI2);

            aTriC.m3V[aK3] = &aV1;

            // extract relative poses (eventually adapt when simul)
            tRotR  aR1toW = aView1.Pose().Rot();
            tRotR  aR2toW = aView2.Pose().Rot();
            if (mDoRand)  // if random : perfect relatives poses with some randomisation
            {
               aR1toW = aVRandR.at(aI1) * tRotR::RandomSmallElem(mLevelRand);
               aR2toW = aVRandR.at(aI2) * tRotR::RandomSmallElem(mLevelRand); 
            }
            // formula forcomputing relative pose
            //  Ori_2->1  = Ori_G->1  o Ori_2->G    ( PL2 -> PG -> PL1)
            tRotR aR2to1 = aR1toW.MapInverse() * aR2toW;

            mGGPoses.AddHyp(aV1,aV2,aR2to1,1.0,cMAT_1Hyp(aKT));

            tE_GGRR * anE = aV1.EdgeOfSucc(aV2)->EdgeInitOr();
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
      const  tAS_GGRR & anAttr =  anE->AttrSym();
      aNbH0 += anAttr.ValuesInit().size();
      aNbHC += anAttr.ValuesComp().size();
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
          const  tAS_GGRR & anAttr =  anE->AttrSym();
          //StdOut() << "-----------------------------------------------\n";
          for (const auto & aH : anAttr.ValuesComp())
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
                          cAtOri_3GGRR(aKE1),cAtOri_3GGRR(aKE2),
                          cAtSym_3GGRR(aCost12)
                     );
                  }
               }
           }
       }
   }
   auto  aListCC = cAlgoCC<tGrTriplet>::All_ConnectedComponent(mGTriC,cAlgo_ParamVG<tGrTriplet>());
   StdOut() << "Number of connected compon Triplet-Graph= " << aListCC.size() << "\n";
   // to see later, we can probably analyse CC by CC, but it would be more complicated (already enough complexity ...)
   MMVII_INTERNAL_ASSERT_tiny(aListCC.size(),"non connected triplet-graph");
}

int NUMDEBUG = 9;

void cMakeArboTriplet::ComputeMinTri()
{

   for (size_t aKTri=0; aKTri<mGTriC.NbVertex() ; aKTri++)
   {
       auto & aAttrTrip  = mGTriC.VertexOfNum(aKTri).Attr();
       tREAL8 aCost = aAttrTrip.mCostIntr;
       for (int aKV=0 ; aKV<3 ; aKV++)
           aAttrTrip.m3V.at(aKV)->Attr().Attr().mWMinTri.Add(aKTri,aCost);
   }
   int aNbTripInit = 0;
   std::vector<tVertex3*> aVectTriMin;
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

   cAlgoSP<tGrTriplet>::tForest  aForest(mGTriC);
   cAlgoSP<tGrTriplet>  anAlgo;
   anAlgo.MinimumSpanningForest(aForest,mGTriC,mGTriC.AllVertices(),aWeighting);
   
   // dont handle 4 now
   MMVII_INTERNAL_ASSERT_tiny(aForest.VTrees().size()==1,"Not single forest");// litle checj
   const auto & aGlobalTree = *(aForest.VTrees().begin());
   StdOut() << "NB TREE=" << aForest.VTrees().size()  << " SzTree1=" << aGlobalTree.Edges().size() << "\n";


   
   cSubGraphOfEdges<tGrTriplet>  aSubGrTree(mGTriC,aGlobalTree.Edges());
   cSubGraphOfVertices<tGrTriplet>  aSubGrAnchor(mGTriC,aVectTriMin); 


   cAlgoPruningExtre<tGrTriplet> aAlgoSupExtr(mGTriC,aSubGrTree,aSubGrAnchor);
   StdOut() <<  "NB EXTR SUPR=" << aAlgoSupExtr.Extrem().size() << "\n";

   std::vector<tEdge3*>  aSetEdgeKern;
   cVG_OpBool<tGrTriplet>::EdgesMinusVertices(aSetEdgeKern,aGlobalTree.Edges(),aAlgoSupExtr.Extrem());


   StdOut() <<  "NB KERNEL=" << aSetEdgeKern.size() << "\n";

   if (mDoCheck)
   {
        cSubGraphOfEdges<tGrTriplet>  aSubGrKernel(mGTriC,aSetEdgeKern);
        // [1]  Check that all the triplet minimal  are belonging to the connection stuff
        for (const auto & aTriC : aVectTriMin)
        {
            MMVII_INTERNAL_ASSERT_always(aSubGrKernel.InsideVertex(*aTriC),"Kernel doesnot contain all  anchors");
        }
        // [2]  Check that aSetEdgeKern is a tree ...
        std::list<std::vector<tVertex3 *>>  allCC =  cAlgoCC<tGrTriplet>::All_ConnectedComponent(mGTriC,aSubGrKernel);
        MMVII_INTERNAL_ASSERT_always(allCC.size()==1,"Kernel is not connected");
        const std::vector<tVertex3 *>& aCC0 =  *(allCC.begin());
        MMVII_INTERNAL_ASSERT_always(aCC0.size()==(aSetEdgeKern.size()+1),"Kernel is not tree");

   }
   cVG_Tree<tGrTriplet>  aTreeKernel(aSetEdgeKern);



}





int cMakeArboTriplet::KSom(t3CPtr aTriC,int aK123) const
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
        tREAL8                    mLevelRand;
        bool                      mDoCheck;
};


cAppli_ArboTriplets::cAppli_ArboTriplets(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mNbMaxClust  (5),
    mDistClust   (0.02),
    mNbIterCycle (3),
    mDoCheck     (true)
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
          << AOpt2007(mLevelRand,"LevelRand","Level of random if simulation")
          << AOpt2007(mDoCheck,"DoCheck","do some checking on result",{eTA2007::HDV,eTA2007::Tuning})
   ;
}

int cAppli_ArboTriplets::Exe()
{
if (0)
{
   tRotR  aId = tRotR::Identity();
   tREAL8 aDMax=0;
   for (int aK=0 ; aK<100000000 ; aK++)
   {
        tRotR aR = tRotR::RandomElem();
        tREAL8 aD = aId.Dist(aR);
// StdOut() << "dddd " << aD   << " " << aDMax << "\n";
        if (aD>aDMax)
        {
            aDMax =aD;
            StdOut() << "DMAX== "  << aDMax << "\n";
            aR.Mat().Show();
        }
   }

}
     mPhProj.FinishInit();

     cTripletSet *  a3Set =  mPhProj.ReadTriplets();
     cMakeArboTriplet  aMk3(*a3Set,mDoCheck);
     if (IsInit(&mLevelRand))
        aMk3.SetRand(mLevelRand);

     aMk3.MakeGraphPose();

     aMk3.DoClustering(mNbMaxClust,mDistClust);
     aMk3.DoIterCycle(mNbIterCycle);
     aMk3.DoTripletWeighting();
     aMk3.MakeGraphTriC();
     aMk3.ComputeMinTri();

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

/*
class cGlobalArborBin;
class cNodeArbor
{
    public :
      friend class cGlobalArborBin;
      static constexpr int  NoNode = -1;
    private :
       void Recurs_GetListValues(std::vector<int>&,const std::vector<cNodeArbor> &) const;

       cNodeArbor(int aNum,int aValue);

       int                mNum;
       int                mValue;
       int                mAsc;   // mother/father: non gendered name ;-)
       int                mDesc1;  // son/daugher: non gendered name ;-)
       int                mDesc2;  // son/daugher: non gendered name ;-)
};

cNodeArbor::cNodeArbor(int aNum,int aValue) :
   mNum    (aNum),
   mValue  (aValue),
   mAsc    (NoNode),
   mDesc1  (NoNode),
   mDesc2  (NoNode)
{
}

void cNodeArbor::Recurs_GetListValues(std::vector<int>&,const std::vector<cNodeArbor> &) const;
{
   aResult.push_back(mValue);
   if (mDesc1!=NoNode)
}

class  cGlobalArborBin
{
    public :
        cNodeArbor  NewTerminalNode(int aVal);
        const std::vector<cNodeArbor> & Nodes() const {return mNodes;}

        void GetListValues(std::vector<int>&) const;
        std::vector<int>  GetListValues() const;
    private :
        std::vector<cNodeArbor>  mNodes;
};

cNodeArbor cGlobalArborBin::NewTerminalNode(int aVal)
{
    mNodes.push_back(cNodeArbor(mNodes.size(),aVal));

    return mNodes.back();
}
*/

