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



namespace MMVII
{

/*
      Classes for  graphs on group. It could work with any group, even if the concrete use is essentially on
   group of mapping . Example in photogrammetry :
          - 3D rotation , use in pose estimation is we omit the translation
          - can be easily extended to  different model of 2D mapping, approximating the mapping between ground coordinate 
            and images with different approximation (can be used for fast approximate pose with aerial/drone) :
                * homography if the scene is plane
                * 2D-Affinity, scene plane, camera un-calibrated
                * 2D Isometry, scene plane, camera calibrated, high variable
                * 2D rotatio, scene plane, camera calibrated, hights constant

   The pre-requisite on class of group is to define :

       - the multiplication  "*"
       - the inverse  "MapInverse()"
       - the neutral element   "Identity()"
       - a distance  Dist() 

      The main purpose is to compute global solution from relative solution.  Suppose we have :

           - N unkown mapping   Mk   Ek -> W
           - M known relative  m_ij  Ei -> Ej
 
       We want to compute (up to a global mapping) the Mk using the relation :

           - Mi  = Mj *  m_ij   ;      i.e  :   Mi (Pi) = Pw = Mj(Pj) = Mj(m_ij(Pi))   [#F1]

        We have the relation :

              m_jk *  m_ij = m_ij  

       If the estimation were "perferct" we would have for any cycle :

            m_da   m_cd   m_bc  m_ab  = "m_aa" = Ident

       In the most general case, we have several estimation of  m_ij note it m_ij,k
         
*/

/*
    Regarding the notation (supposing it is poses) :

        * In class Som ,  mCurValue is the mapping Cam->Word  (or column of the rotaation are vector IJK)

               Ori_L->G   Local camera coordinates   ->   Global Word coordinate
                           PG = Ori_L->G (PL)

       *  In class AttrEdge,   Ori_2->1 is pose of Cam2 relatively to Cam 1,   PL2 ->PL1

               Ori_2->1  = Ori_G->1  o Ori_2->G    ( PL2 -> PG -> PL1)

       * for composing value of relative  
               Ori_2->1 =(PL2 -> PL1)    ;   Ori_3->2 =(PL3->PL2)  ;    Ori_4->3  = (PL4->PL3)

                Ori_2->1 * Ori_3->2 * Ori_4->3 ...
*/

template <class TGroup>  
    TGroup  RandomGroupInInterval(tREAL8 aV0,tREAL8 aV1)
{
   TGroup aRes = TGroup::Identity();
   while (aRes.Dist(TGroup::Identity()) <= aV0)
        aRes = TGroup::RandomSmallElem(aV1);


   return aRes;
}


template <class TGroup>  class cGGA_EdgeOr;    //  Group-Grap Edge-Oriented Attribute
template <class TGroup>  class cGGA_EdgeSym;   //  Group-Grap Edge-Symetric Attribute
template <class TGroup>  class cGG_1HypInit;   //  1-element of group graph
template <class TGroup>  class cGG_1HypComputed;   //  1-element of group graph
template <class TGroup>  class cGGA_Vertex;    //  Group-Graph Vertex Attribute
template <class TGroup>  class cGroupGraph;    //  Group Graph


/*----------------------------------------- cGGA_EdgeOr --------------------------------------------------*/

/** Oriented attributes of graph-group, nothing to do 4 now */

template <class TGroup>  class cGGA_EdgeOr
{
     public :
};

/*----------------------------------------- cGG_1HypInit --------------------------------------------------*/

/**  Store an initial hypothesis of relative orientation , the containing several hypothesis */

template <class TGroup>  class cGG_1HypInit
{
   public :
      cGG_1HypInit(const TGroup & aG,tREAL8 aW0,int aNumSet) :
           mW0       (aW0), 
           mNumComp  (-1) , 
           mVal      (aG), 
           mNoiseSim (0),
           mNumSet   (aNumSet)
      {}

      bool IsMarked() const {return mNumComp != -1;}


      tREAL8   mW0;        // initial weight 
      int      mNumComp;    // used as temporary mark in clustering
      TGroup   mVal;       // Value of the map/group
      tREAL8   mNoiseSim;  // Noise used in/if simulation
      int      mNumSet;
};

/*----------------------------------------- cGG_1HypComputed -----------------------------------------------*/

/**  Store a "computed" hypothesis, this is an initial hypothesis + some additionnal info*/
 
template <class TGroup>  class cGG_1HypComputed : public cGG_1HypInit<TGroup>
{
    public  :
      cGG_1HypComputed(const TGroup & aG,tREAL8 aW) : 
           cGG_1HypInit<TGroup> (aG,aW,-1),
           mWeight              (aW) // weight = initial weight
      {}

      tREAL8   mWeight;
      cWeightAv<tREAL8,tREAL8> mWeightedDist;
};

/*----------------------------------------- cGGA_EdgeSym -----------------------------------------------*/

/** Store the symetric attribute of a "group-graph", essentially a set of hypothesis + some computation */

template <class TGroup>  class cGGA_EdgeSym
{
     public :
          typedef cGG_1HypInit<TGroup>       t1HypInit;
          typedef cGG_1HypComputed<TGroup>   t1HypComp;

          cGGA_EdgeSym() :  mBestH (nullptr) {}

          std::vector<t1HypComp> &  ValuesComp() {return mValuesComp;} ///< Accessor
          std::vector<t1HypInit> &  ValuesInit() {return mValuesInit;} ///< Accessor
          const std::vector<t1HypComp> &  ValuesComp() const {return mValuesComp;} ///< Accessor
          const std::vector<t1HypInit> &  ValuesInit() const {return mValuesInit;} ///< Accessor
	  /// Add 1 initial hypothesis 
          t1HypInit & Add1Value(const TGroup& aVal,tREAL8 aW,int aNumSet);
	  /// Do the clustering to reduce the number of initial hypothesis
          void DoCluster(int aNbMax,tREAL8 aDist);

          void SetBestH(t1HypComp * aBestH) {mBestH=aBestH;}
          const t1HypComp * BestH() const {return mBestH;}
         

          const t1HypInit *  HypOfNumSet(int aNum,bool SVP=false) const;
          const t1HypComp *  HypCompOfH0(const t1HypInit&,bool SVP=false) const;
     private :

          /// for a given center return a score (+- weighted count of element not marked closed enough)
          tREAL8  CenterClusterScore(const TGroup &,tREAL8 aDist) const;

          ///  Add the next best center of the cluster, return true if added something
          bool  GetNextBestCCluster(tREAL8 aThDist);

          std::vector<t1HypComp>  mValuesComp;  ///
          std::vector<t1HypInit>  mValuesInit;
          t1HypComp *             mBestH;
};


/*----------------------------------------- cGGA_Vertex -----------------------------------------------*/

/** Class for storing the attibute of a vertex of graph-group */

template <class TGroup>  class cGGA_Vertex
{
     public :
          cGGA_Vertex() 
           {
           }
          
     private :
          TGroup              mComputedValue;  // computed value
};

/*----------------------------------------- cGroupGraph -----------------------------------------------*/

/** Class for manipulating the group - graph */

template <class TGroup>  
     class cGroupGraph :
             public cVG_Graph<cGGA_Vertex<TGroup>, cGGA_EdgeOr<TGroup>,cGGA_EdgeSym<TGroup>>
{
    public :
          // -------------------------------------- typedef part ---------------------------
          typedef cGGA_Vertex<TGroup>   tAVert;
          typedef cGGA_EdgeOr<TGroup>   tAEOr;
          typedef cGGA_EdgeSym<TGroup>  tAESym;
          typedef cGG_1HypInit<TGroup>       t1HypInit;
          typedef cGG_1HypComputed<TGroup>   t1HypComp;

          typedef cVG_Graph<tAVert,tAEOr,tAESym> tGraph;
          typedef typename tGraph::tVertex       tVertex;
          typedef typename tGraph::tEdge         tEdge;
          typedef cGroupGraph<TGroup>            tGrGr;
          typedef  cAlgo_SubGr<tGrGr>            tSubGr;
          typedef  cAlgo_ParamVG<tGrGr>          tParamWG;

          typedef  cAlgoSP<tGrGr>                tAlgoSP;
          typedef  typename tAlgoSP::tForest     tForest;



          cGroupGraph(bool ForSimul);

	  /// create a vertex with no value
          tVertex &  AddVertex();

	  ///  Add a new hypothesis
          t1HypInit & AddHyp(tVertex& aN1,tVertex& aN2,const TGroup &,tREAL8 aW,int aSet);

	  /// return the value of G taking in E=v1->V2 into account that it may have been set in V2->V1
          static TGroup  ValOrient(const TGroup& aG,const tEdge & anE) { return anE.IsDirInit() ? aG : aG.MapInverse(); }

	  /// execute one iteration of adding cycles  
          void OneIterCycles(size_t aSzMaxC,tREAL8 aPow,bool Show);

	  /// Make the clustering of all hypothesis is necessary
          void DoClustering(size_t aSzMaxCluster,tREAL8 aDistClust);

	  ///  class for call-back in cycles enumeration
          class cGG_OnCycle  : public cActionOnCycle<tGrGr>
          {
             public :
                cGG_OnCycle();

                /// call back for exploring all cycles
                void OnCycle(const cAlgoEnumCycle<tGrGr> &) override;

                /// method to recusrively explorate all the combination of hypothesis of a given cycle
                void   RecursOnCycle
                       (
                            const TGroup &,  // current group accumulated
                            size_t aDepth,   // current depth
                            tREAL8 aW        // current weight accumulated
                       );

                std::vector<tEdge*>     mVCurPath; // make a local copy of current path
                std::vector<t1HypComp*> mVCurHyp;  // stack for hypothesis being explored
          };

	  class  cWeightOnBestH : public tParamWG
	  {
		  public :
                      tREAL8 WeightEdge(const    tEdge & anE) const override 
		      {
			      return anE.AttrSym().BestH()->mWeightedDist.Average();
		      }

	  };
          tREAL8 MakeMinSpanTree();

    protected :
           tGraph * mGraph;   // the graph itself
           bool     mClusterDone; // have we already done the clustering
           // std::map<std::string,tVertex*>  mMapV;  // Map Name->Vertex , for user can access by name
           int                             mNbVHist;
           bool                            mForSimul;
           int                             mCptC;
};


/* ********************************************************* */
/*                                                           */
/*                     cGGA_EdgeSym                          */
/*                                                           */
/* ********************************************************* */


template <class TGroup> tREAL8 cGGA_EdgeSym<TGroup>::CenterClusterScore(const TGroup & aG,tREAL8 aThrDist) const
{
   tREAL8 aSum = 0.0;
   for (const auto & aVal : mValuesInit)
   {
       if (! aVal.IsMarked())  // only unmarked are of interest
       {
          tREAL8 aD = aG.Dist(aVal.mVal);     // distance to the proposed center
          aSum += CubAppGaussVal(aD/aThrDist); // weighting function
       }
   }
   return aSum;
}

template <class TGroup> const cGG_1HypInit<TGroup> *  cGGA_EdgeSym<TGroup>::HypOfNumSet(int aNum,bool SVP) const
{
   for (const auto & aVH: mValuesInit)
      if (aVH.mNumSet== aNum)
         return &aVH;
   MMVII_INTERNAL_ASSERT_tiny(SVP,"HypOfNumSet cannot find");// litle checj
   return nullptr;
}

template <class TGroup> const cGG_1HypComputed<TGroup> *  cGGA_EdgeSym<TGroup>::HypCompOfH0(const t1HypInit&aH0,bool SVP) const
{
   if (!aH0.IsMarked())
   {
       MMVII_INTERNAL_ASSERT_tiny(SVP,"HypCompOfH0 cannot find");// litle checj
       return nullptr;
   }
   return  & mValuesComp.at(aH0.mNumComp);
}

template <class TGroup> bool  cGGA_EdgeSym<TGroup>::GetNextBestCCluster(tREAL8 aThrDist)
{
     /// extract the element with best score
     cWhichMax<t1HypInit*,tREAL8> aWMaxEl;
     for (auto & aV : mValuesInit)
     {
           aWMaxEl.Add(&aV,CenterClusterScore(aV.mVal,aThrDist));
     }

     // initialise 3 variable usefull for result
     TGroup aCenterClust = TGroup::Identity();   // center of putative cluster
     tREAL8 aSumNoise = 0.0;  // sum of noise of cluster (used in simulation)
     tREAL8 aSumW = 0.0;

     if (aWMaxEl.ValExtre() >0) // if an element was added
     {
        // make some iteration of refinement 
        aCenterClust = aWMaxEl.IndexExtre()->mVal;
        int aNbIter = 3;
        for (int aKIter=0 ; aKIter<aNbIter ; aKIter++)
        {
             std::vector<TGroup>  aVG;  // store the element close enough
             std::vector<tREAL8>  aVW;  // store the weight
             for (const auto & aVal : mValuesInit)
             {
                 if (! aVal.IsMarked()) // is not marked
                 {
                    tREAL8 aD = aCenterClust.Dist(aVal.mVal);
                    tREAL8 aW =  CubAppGaussVal(aD/aThrDist);
                    if (aW != 0) // is close enough
                    {
                        aVG.push_back(aVal.mVal);
                        aVW.push_back(aW);
                    }
                 }
             }
             // replace by weighted average
             if (aVW.size() > 0)
                aCenterClust = TGroup::Centroid(aVG,aVW);
             else
                aKIter = aNbIter;
        }
        // at the end mark points close enough and count theme
        for (auto & aVal : mValuesInit)
        {
            if ((! aVal.IsMarked())  && ( aCenterClust.Dist(aVal.mVal)<aThrDist))
            {
                  aVal.mNumComp = mValuesComp.size();
                  aSumNoise += aVal.mNoiseSim * aVal.mW0;
		  aSumW += aVal.mW0;
            }
        }
     }

     // Ok we got a new center, add it in "mValuesComp"
     if (aSumW!=0)
     {
          mValuesComp.push_back(t1HypComp(aCenterClust,aSumW));
          mValuesComp.back().mNoiseSim = aSumNoise/aSumW;
          return true;
     }

     return false;
}

template <class TGroup> void cGGA_EdgeSym<TGroup>::DoCluster(int aNbMax,tREAL8 aDist)
{
    mValuesComp.clear();
    if (aDist<=0)
    {
        for (auto & aH0 : mValuesInit)
        {
           aH0.mNumComp = mValuesComp.size();
           mValuesComp.push_back(t1HypComp(aH0.mVal,1.0));
        }
    }
    else
    {
        // compute a maximum of aNbMax cluster
        for (int aKNew=0 ; aKNew<aNbMax ; aKNew++)
        {
            bool Ok = GetNextBestCCluster(aDist);
            if (! Ok)
            {
                aKNew = aNbMax;
            }
        }
    }
}

template <class TGroup> cGG_1HypInit<TGroup> & cGGA_EdgeSym<TGroup>::Add1Value(const TGroup& aVal,tREAL8 aW,int aNumSet)
{
    mValuesInit.push_back(t1HypInit(aVal,aW,aNumSet));
    return mValuesInit.back();
}




/* ********************************************************* */
/*                                                           */
/*                     cGroupGraph                           */
/*                                                           */
/* ********************************************************* */



     /* ********************************************************* */
     /*                                                           */
     /*                 cGroupGraph::cGG_OnCycle                  */
     /*                                                           */
     /* ********************************************************* */

template  <class TGroup>  
  cGroupGraph<TGroup>::cGG_OnCycle::cGG_OnCycle() 
{
}

template  <class TGroup>  
   void cGroupGraph<TGroup>::cGG_OnCycle::OnCycle(const cAlgoEnumCycle<tGrGr> & anAlgo) 
{
   // store a local copy of the path
   mVCurPath = anAlgo.CurPath();
   
   // Throw recursive exploration 
   RecursOnCycle(TGroup::Identity(),0,1.0);
}


template  <class TGroup>  
   void cGroupGraph<TGroup>::cGG_OnCycle::RecursOnCycle(const TGroup & aG,size_t aDepth,tREAL8 aW)
{
    // if we have finish the path
    if (aDepth==mVCurPath.size())
    {
       // compute the distance to identity at the end of the loop, should be 0 with perfect data
       tREAL8 aDist = aG.Dist(TGroup::Identity());
       MMVII_INTERNAL_ASSERT_tiny(mVCurHyp.size()==aDepth,"RecursOnCycle -> Depth");// litle checj
       // accumulate the scoring in all hypothesis involved in the computation
       for (auto & aPtrH : mVCurHyp)
       {
           aPtrH->mWeightedDist.Add(aW,aDist);
       }
       return;
    }

    // if not finish, parse all the hypothesis of current edge
    tEdge & anE = * mVCurPath.at(aDepth);
    for (auto & anElem : anE.AttrSym().ValuesComp())
    {
        mVCurHyp.push_back(&anElem);  // push in the stack of hypothesie
        // Ori_2->1 * Ori_3->2 * Ori_4->3 ...
        TGroup aNewG = aG * ValOrient(anElem.mVal , anE);
        RecursOnCycle(aNewG,aDepth+1,aW*anElem.mWeight);
        mVCurHyp.pop_back(); // restore the stack of hypothesis
    }          
}

     /* ********************************************************* */
     /*                                                           */
     /*                     cGroupGraph                           */
     /*                                                           */
     /* ********************************************************* */

template <class TGroup>  
    cGroupGraph<TGroup>::cGroupGraph(bool forSimul) :
        mGraph        (this),
        mClusterDone  (false),
        mNbVHist      (1000),
        mForSimul     (forSimul),
        mCptC         (0)
{
}


template <class TGroup>  
    typename cGroupGraph<TGroup>::tVertex &
        cGroupGraph<TGroup>::AddVertex()
{
    // 1-check dont exist, 2-create, 3-memorize
    // MMVII_INTERNAL_ASSERT_tiny(!MapBoolFind(mMapV,aName),"cGroupGraph, name alrady exist :" +aName);
    tVertex * aV = mGraph->NewSom(tAVert());
    // mMapV[aName] = aV;
    return *aV;
}


template <class TGroup>  
    cGG_1HypInit<TGroup>& cGroupGraph<TGroup>::AddHyp(tVertex & aV1,tVertex & aV2,const TGroup& aG,tREAL8 aW,int aNumSet)
{
    //  Add edge if does not exist
    tEdge * anE = aV1.EdgeOfSucc(aV2,SVP::Yes);
    if (anE==nullptr)
       anE = mGraph->AddEdge(aV1,aV2,tAEOr(),tAEOr(),tAESym());

    //  now add the hypothesis to existing edge 
    return anE->AttrSym().Add1Value(ValOrient(aG,*anE),aW,aNumSet);
}
  

template <class TGroup>  
   void cGroupGraph<TGroup>::DoClustering(size_t aNbMaxCluster,tREAL8 aDistClust)
{
   if (! mClusterDone)
   {
       mClusterDone = true;
       for (auto & anEPtr :  mGraph->AllEdges_DirInit ())
           anEPtr->AttrSym().DoCluster(aNbMaxCluster,aDistClust);
   }
}

template <class TGroup>  
   void cGroupGraph<TGroup>::OneIterCycles(size_t aSzMaxC,tREAL8 aPow,bool Show)
{
   mCptC++;
   // cluster the values is not already done

   // Reset the weights
   for (auto & aPtrE :  mGraph->AllEdges_DirInit ())
       for (auto & aH : aPtrE->AttrSym().ValuesComp())
           aH.mWeightedDist.Reset();



   cGG_OnCycle aOnC;
   cAlgoEnumCycle<tGrGr>  aAlgoEnum(*this,aOnC,tSubGr(),aSzMaxC); 
   aAlgoEnum.ExplorateAllCycles();


   // compute the histogramm of mWeightedDist to have some normalization of costs
   cHistoCumul<tREAL8,tREAL8> aHisto(TGroup::MaxDist()*mNbVHist+1);
   for (const auto & aPtrE : mGraph->AllEdges_DirInit ())
   {
       for (const auto & anH : aPtrE->AttrSym().ValuesComp())
       {
           int aInd = round_ni(anH.mWeightedDist.Average()*mNbVHist);
           aHisto.AddV(aInd, anH.mWeightedDist.SW());
       }
   }
   aHisto.MakeCumul();

   // In simulation, if we make a visualisation of histo-2D  Weigh/Noise
   int aNbVisu = (Show ? 1000 : 1);
   cIm2D<tREAL8> aIm(cPt2di(aNbVisu,aNbVisu),nullptr,eModeInitImage::eMIA_Null);

   for (const auto & aPtrE : mGraph->AllEdges_DirInit ())
   {
       cWhichMin<t1HypComp*,tREAL8> aWHypMin;
       for (auto & anH : aPtrE->AttrSym().ValuesComp())
       {
           tREAL8 aWDAvg = anH.mWeightedDist.Average();
           if (Show)  // Acumulate histogram 2D
           {
              // tREAL8 aD = anH.mWeightedDist.Average();  // weight dist of cycles
              tREAL8 aN = anH.mNoiseSim;                // noise of simulation
              cPt2dr aPDN(aWDAvg*aNbVisu,aN*aNbVisu);       // point to accumulate
              if (aIm.DIm().InsideBL(aPDN))
              {
                 aIm.DIm().AddVBL(aPDN,100.0);
              }
           }
          
           tREAL8 aInd = round_ni(aWDAvg*mNbVHist); // index in histo
           tREAL8 aMul  = 1.0- aHisto.PropCumul(aInd) + 1.0/mNbVHist;    //  Mul = 1-Rank, + eps to avoid 0
           aMul = std::pow(aMul,aPow);                                   // pow : exagerate good rank
           anH.mWeight =  anH.mW0 * aMul;                                // weigh = Weightinit * Mul , to not forget pop init
           aWHypMin.Add(&anH,aWDAvg);
       }
       aPtrE->AttrSym().SetBestH(aWHypMin.IndexExtre());
   }
   if (Show)
   {
     aIm = aIm.GaussFilter(aNbVisu/100.0);
     aIm.DIm().ToFile("Histo-DistNoise_Cpt"+ ToStr(mCptC) +  "_SzC"+ ToStr(aSzMaxC) + "_Pow" + ToStr(aPow) +".tif");
   }
}

template <class TGroup>  
   tREAL8 cGroupGraph<TGroup>::MakeMinSpanTree()
{

    tForest aForest;
    tAlgoSP anAlgoSP;
    cWeightOnBestH aWBE;
    anAlgoSP.MinimumSpanningForest(aForest,*this,this->AllVertices(), aWBE);  // extract the forest

    // Theoreitcally this can happen, but dont want to gandle it 4 now
    MMVII_INTERNAL_ASSERT_tiny(aForest.size()==1,"cGroupGraph::MakeMinSpanTree Forest size");

    tREAL8 aMaxCost = -1.0;

    for (const auto &   [aV0,aListEdges] : aForest)
    {
        for (const auto & aPtrE : aListEdges)
        {
            UpdateMax(aMaxCost,aPtrE->AttrSym().BestH()->mNoiseSim);
            // StdOut() << "W=" << aWBE.WeightEdge(*aPtrE) << " N=" << aPtrE->AttrSym().BestH()->mNoiseSim << "\n";
	}
    }
    // StdOut() << "MAX COST = " << aMaxCost << "\n";
    return aMaxCost;
}


/* ********************************************************* */
/*                                                           */
/*                     cBench_G3                             */
/*                                                           */
/* ********************************************************* */

template <class TGroup>  class cBench_G3  // Grid-Group-Graph
{
    public :
          typedef cGroupGraph<TGroup>     tGG;
          typedef typename tGG::tVertex   tVertex;
          typedef tVertex*                tVertPtr;

          class cBG3V
          {
               public :
                  TGroup     mValRef;
                  tVertex*   mVertex;
          };

          cBench_G3
          (
              const cPt2di & aSz,int aNbMin,int aNbMax,tREAL8 aPropOutLayer,
              tREAL8 aNoiseInLayer,tREAL8 aNoiseMinOutlayer,tREAL8 aNoiseMaxOutLayer
          );

          tGG &    GG() {return mGG;}

    private :
         cBG3V & ValOfPt(const cPt2di & aPt) {return mGridVals.at(aPt.y()).at(aPt.x());}
         TGroup  RefRel_2To1(const cBG3V & aV1,const cBG3V & aV2)
         {
               //  Ori_2->1  = Ori_G->1  o Ori_2->G    ( PL2 -> PG -> PL1)
               return aV1.mValRef.MapInverse() * aV1.mValRef;
         }


          void Add1Edge(const cPt2di &  aP0,const cPt2di & aP1);

          cPt2di mSz;
          cRect2 mBox;
          tGG    mGG;
          std::vector<std::vector<cBG3V>>        mGridVals;

          int mNbMinE;
          int mNbMaxE;
          tREAL8 mPropOutLayer;
          tREAL8 mNoiseInLayer;
          tREAL8 mNoiseMinOutLayer;
          tREAL8 mNoiseMaxOutLayer;
};

template <class TGroup> 
    cBench_G3<TGroup>::cBench_G3
    (
        const cPt2di & aSz,
        int aNbMinE,int aNbMaxE,
        tREAL8 aPropOutLayer,tREAL8 aNoiseInLayer,
        tREAL8 aNoiseMinOutLayer, tREAL8 aNoiseMaxOutLayer
    ) :
       mSz                (aSz),
       mBox               (cPt2di(0,0),aSz),
       mGG                (true),
       mGridVals          (mSz.y(),std::vector<cBG3V>(mSz.x(),cBG3V())),
       mNbMinE            (aNbMinE),
       mNbMaxE            (aNbMaxE),
       mPropOutLayer      (aPropOutLayer),
       mNoiseInLayer      (aNoiseInLayer),
       mNoiseMinOutLayer  (aNoiseMinOutLayer),
       mNoiseMaxOutLayer  (aNoiseMaxOutLayer)
{
    for (const auto & aPix : mBox)
    {
        std::string aName = ToStr(aPix.x()) + "_" + ToStr(aPix.y());
        ValOfPt(aPix).mVertex = &mGG.AddVertex();
        ValOfPt(aPix).mValRef = TGroup::RandomElem();
    }

    for (const auto & aPix : mBox)
    {
          Add1Edge(aPix,aPix+cPt2di(1,0));
          Add1Edge(aPix,aPix+cPt2di(0,1));
          Add1Edge(aPix,aPix+cPt2di(1,1));
          Add1Edge(aPix,aPix+cPt2di(-1,1));
    }

// StdOut() << "Add1EdgeAdd1EdgeAdd1Edge \n"; getchar();

}




template <class TGroup> void cBench_G3<TGroup>::Add1Edge(const cPt2di &  aP0,const cPt2di & aP1)
{
   if ((!mBox.Inside(aP0)) || (!mBox.Inside(aP1)))
      return;

   const cBG3V & aVal0 = ValOfPt(aP0);
   const cBG3V & aVal1 = ValOfPt(aP1);

   TGroup aRefRel_2To1 = RefRel_2To1(aVal0,aVal1);

   int aNbAdd = RandUnif_M_N(mNbMinE,mNbMaxE);
   for (int aK=0 ; aK<aNbAdd ; aK++)
   {
        bool isInLayer = (RandUnif_0_1() > mPropOutLayer);
        TGroup aPerturb = TGroup::RandomSmallElem(mNoiseInLayer);

        if (! isInLayer)
           aPerturb= RandomGroupInInterval<TGroup>(mNoiseMinOutLayer,mNoiseMaxOutLayer);
        tREAL8 aNoise = aPerturb.Dist(TGroup::Identity());

        TGroup aRel_2To1 =  aRefRel_2To1 * aPerturb;


        cGG_1HypInit<TGroup> & aH =mGG.AddHyp(*(aVal0.mVertex),*(aVal1.mVertex),aRel_2To1,1.0,-1);
        aH.mNoiseSim = std::abs(aNoise);
   }
}

template class cBench_G3<tRotR>;

void BenchGroupGraph(cParamExeBench & aParam)
{
    if (! aParam.NewBench("GroupGraph")) return;

    if (UserIsMPD())
    {
        cBench_G3<tRotR> aBG3
                     (
                         cPt2di(52,62),
                         3,10,
                         0.2,   //  Prop Out layer
                         0.05,0.0,0.5
                     );
         aBG3.GG().DoClustering(3,0.15);

         aBG3.GG().OneIterCycles(3,1.0,true);
         aBG3.GG().OneIterCycles(3,1.0,true);
         aBG3.GG().OneIterCycles(3,1.0,true);
         aBG3.GG().OneIterCycles(3,1.0,true);
         aBG3.GG().OneIterCycles(3,1.0,true);
         aBG3.GG().OneIterCycles(3,1.0,true);
         aBG3.GG().OneIterCycles(3,1.0,true);

         aBG3.GG().MakeMinSpanTree();
     }

     if (1)
     {

         cBench_G3<tRotR> aBG3
                     (
                         cPt2di(12,22),
                         3,10,
                         0.2,   //  Prop Out layer
                         0.00,0.2,0.5
                     );
         aBG3.GG().DoClustering(3,0.15);
         aBG3.GG().OneIterCycles(3,1.0,true);
         aBG3.GG().OneIterCycles(3,1.0,true);
         aBG3.GG().OneIterCycles(3,1.0,true);

         tREAL8 aNoiseMax = aBG3.GG().MakeMinSpanTree();

         StdOut() << "NM=" << aParam.Level() << " "<< aNoiseMax << "\n";
         // It's "highly probable" that with this random generation, the min span tree will
         // use only unoised edges, but not 100% (at least I cannot prouve it) so do it only for
         // 100 first, for which, empirically it was tested to be true
         if  (aParam.Level() <100)
         {
             MMVII_INTERNAL_ASSERT_bench(aNoiseMax==0,"Group graph MakeMinSpanTree ");
         }
     }

    aParam.EndBench();
}

/*
template class cGG_1HypInit<tRotR>;
template class cGG_1HypComputed<tRotR>;
template class cGGA_EdgeSym<tRotR>;
template class cGGA_Vertex<tRotR>;
template class cGGA_EdgeOr<tRotR>;
template class cGroupGraph<tRotR>;
*/

/* ********************************************************* */
/*                                                           */
/*                     cAppli_ArboTriplets                   */
/*                                                           */
/* ********************************************************* */

typedef cGroupGraph<tRotR>       tGGRR;
typedef typename tGGRR::tVertex  tV_GGRR;
typedef typename tGGRR::tEdge    tE_GGRR;
typedef tE_GGRR*                 tEdgePtr;
typedef std::array<tE_GGRR*,3>   t3E_GGRR;

class cTri_GGRR
{
   public :
        cTri_GGRR (const cTriplet* aT0) :
            mT0         (aT0),
            mOk         (false),
            mCostIntr   (1e10) 
        {
        }

        const cTriplet*  mT0;
        t3E_GGRR         mCnxE;      // the 3 edges 
        bool             mOk;        // for ex, not OK if on of it edges was not clustered
        tREAL8           mCostIntr;  // intrisiq cost
};

class cMakeArboTriplet
{
     public :
         
         cMakeArboTriplet(const cTripletSet & aSet3);

         // make the graph on pose, using triplet as 3 edges
         void MakeGraphPose();

         /// reduce eventually the number of triplet by clustering them, used in cycle computation
         void DoClustering(int aNbMax,tREAL8 aDistCluster);
         ///  compute some quality criteria on triplet by loop closing on rotation
         void DoIterCycle(int aNbC);
         void DoTripletWeighting();

         void SetRand(tREAL8 aLevelRand);

         typedef cTri_GGRR * t3CPtr;
     private :

         int KSom(t3CPtr aTri,int aK123) const;


         const cTripletSet  &    mSet3;
         std::vector<cTri_GGRR>  mVTriC;
         t2MapStrInt             mMapStrI;
         tGGRR                   mGGPoses;
         bool                    mDoRand;
         tREAL8                  mLevelRand;
};


cMakeArboTriplet::cMakeArboTriplet(const cTripletSet & aSet3) :
   mSet3      (aSet3),
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
   // fix the size for memory opt
   mVTriC.reserve(mSet3.Set().size());

   // create mVTriC & compute map NamePose/Int in mMapStrI
   for (auto & a3 : mSet3.Set())
   {
        for (size_t aK3=0 ; aK3<3 ; aK3++)
        {
             mMapStrI.Add(a3.Pose(aK3).Name(),true);
        }
        mVTriC.push_back(cTri_GGRR(&a3));
   }
   StdOut() << "Nb3= " << mSet3.Set().size() << " NBI=" << mMapStrI.size() << "\n";


   // In case we want to make some test with random rot, create the "perfect" rot of each pose
   std::vector<tRotR> aVRandR;
   for (size_t aKR=0; aKR<mMapStrI.size() ; aKR++)
       aVRandR.push_back(tRotR::RandomRot());

   //  Add the vertex corresponding to each poses in Graph-Pose
   for (size_t aKIm=0 ; aKIm<mMapStrI.size() ; aKIm++)
   {
       mGGPoses.AddVertex();
   }

   for (size_t aKT=0; aKT<mSet3.Set().size() ; aKT++)
   {
        const cTriplet & a3 = mSet3.Set().at(aKT);
        for (size_t aK3=0 ; aK3<3 ; aK3++)
        {
            const cView&  aView1 = a3.Pose(aK3);
            const cView&  aView2 = a3.Pose((aK3+1)%3);
            int aI1 =  mMapStrI.Obj2I(aView1.Name());
            int aI2 =  mMapStrI.Obj2I(aView2.Name());

            tRotR  aR1toW = aView1.Pose().Rot();
            tRotR  aR2toW = aView2.Pose().Rot();

            if (mDoRand)
            {
               aR1toW = aVRandR.at(aI1) * tRotR::RandomSmallElem(mLevelRand);
               aR2toW = aVRandR.at(aI2) * tRotR::RandomSmallElem(mLevelRand);
            }

            //  Ori_2->1  = Ori_G->1  o Ori_2->G    ( PL2 -> PG -> PL1)
            tRotR aR2to1 = aR1toW.MapInverse() * aR2toW;

            tV_GGRR & aV1  = mGGPoses.VertexOfNum(aI1);
            tV_GGRR & aV2  = mGGPoses.VertexOfNum(aI2);

            mGGPoses.AddHyp(aV1,aV2,aR2to1,1.0,aKT);

            tE_GGRR * anE = aV1.EdgeOfSucc(aV2)->EdgeInitOr();
            mVTriC.at(aKT).mCnxE.at(aK3) = anE;
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
      const  cGGA_EdgeSym<tRotR> & anAttr =  anE->AttrSym();
      aNbH0 += anAttr.ValuesInit().size();
      aNbHC += anAttr.ValuesComp().size();
      aNbE++;
   }

   StdOut() << "NBH , Init=" << aNbH0 /double(aNbE) << " Comp=" << aNbHC/double(aNbE)  << "\n";
}

void cMakeArboTriplet::DoIterCycle(int aNbIter)
{
     // call the IterCycles-method of graph group, that compute quality on edges-hyp
     for (int aK=0 ; aK<aNbIter ; aK++)
     {
         mGGPoses.OneIterCycles(3,1.0,false);
         StdOut() << "DONE DoIterCycle \n";
     }

      cStdStatRes aStat;
      for (const auto & anE : mGGPoses.AllEdges_DirInit())
      {
          const  cGGA_EdgeSym<tRotR> & anAttr =  anE->AttrSym();
          //StdOut() << "-----------------------------------------------\n";
          for (const auto & aH : anAttr.ValuesComp())
          {
              aStat.Add(aH.mWeightedDist.Average() );
              //StdOut()  <<  "WWdd="  << aH.mWeightedDist.Average() << "\n";
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
    for (size_t aKT=0 ; aKT<mVTriC.size() ; aKT++)
    {
        std::vector<tREAL8>  aVectCost;
        for (size_t aKE=0 ; aKE<3 ; aKE++)
        {
             const auto & anAttr = mVTriC.at(aKT).mCnxE.at(aKE)->AttrSym();
             const auto aHyp0 = anAttr.HypOfNumSet(aKT);
             if (aHyp0->IsMarked())
             {
                 const auto aHypC =  anAttr.HypCompOfH0(*aHyp0);
                 aVectCost.push_back(aHypC->mWeightedDist.Average());
             }
        }
        if (aVectCost.size()==3)
        {
            std::sort(aVectCost.begin(),aVectCost.end());
            cWeightAv<tREAL8,tREAL8> aWCost;
            tREAL8 aExp = 2.0;
            for (size_t aKP=0 ; aKP<3 ;aKP++)
                aWCost.Add(std::pow(aExp,aKP),aVectCost.at(aKP));
            mVTriC.at(aKT).mOk = true;
            mVTriC.at(aKT).mCostIntr = aWCost.Average() ;
        }
        else
        {
        }
    }
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
};


cAppli_ArboTriplets::cAppli_ArboTriplets(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mNbMaxClust  (5),
    mDistClust   (0.02),
    mNbIterCycle (3)
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
   ;
}

int cAppli_ArboTriplets::Exe()
{
     mPhProj.FinishInit();

     cTripletSet *  a3Set =  mPhProj.ReadTriplets();
     cMakeArboTriplet  aMk3(*a3Set);
     if (IsInit(&mLevelRand))
        aMk3.SetRand(mLevelRand);

     aMk3.MakeGraphPose();

     aMk3.DoClustering(mNbMaxClust,mDistClust);
     aMk3.DoIterCycle(mNbIterCycle);
     aMk3.DoTripletWeighting();
     //aMk3.MakeTree();

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











};
