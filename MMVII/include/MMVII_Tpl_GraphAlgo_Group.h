#ifndef _MMVII_Tpl_GraphGroup_H_
#define _MMVII_Tpl_GraphGroup_H_

#include "MMVII_Tpl_GraphAlgo_SPCC.h"
#include "MMVII_Tpl_GraphStruct.h"
#include "MMVII_Tpl_GraphAlgo_EnumCycles.h"
#include "MMVII_Interpolators.h"

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

/** Function used in simulation, create a random elemen of group with distance to identity in
the interval [aV0,aV1] */

template <class TGroup>  
    TGroup  RandomGroupInInterval(tREAL8 aV0,tREAL8 aV1)
{
   TGroup aRes = TGroup::Identity();
   while (aRes.Dist(TGroup::Identity()) <= aV0)
        aRes = TGroup::RandomSmallElem(aV1);

   return aRes;
}

/**   cGroupGraph are a specialisation of cVG_Graph. In a group graph we want to put :

     - the data required by the group computation (essentially list of hypothesis of group in the edges)
     - any supplementary data that the user may need to add  to work after some standar computation has
       been made

     Considering for example the edge attributes of a cGroupGraph, the symetric part is represented by class 
    "cGGA_EdgeSym" . This class contains :
           - the user specific data of the edge
           - the different hypothesis of group, but each hypothesis can contains also user specific data 
             (as some info on the triplet it belongs to)
      So the cGGA_EdgeSym is templetized by 3 class :
            - the group 
            - the  used specific attribute of the edge
            - the user specifici attribute of the hypothesis
      When manipulating an edge E  of a cGGA_EdgeOr, note that :
            - E.AttrS()  return the cGGA_EdgeSym
            - E.AttrS().Attr()  will contain the users specific attribute ....
*/

template <class TGroup,class TAO>  class cGGA_EdgeOr;    ///  Group-Grap Edge-Oriented Attribute
template <class TGroup,class TAS,class TAH>  class cGGA_EdgeSym;   ///  Group-Grap Edge-Symetric Attribute
template <class TGroup,class TAH>  class cGG_1HypInit;       ///  1-element of group graph
template <class TGroup,class TAH>  class cGG_1HypClustered;   ///  1-element of group graph
template <class TGroup,class TAV>  class cGGA_Vertex;        ///  Group-Graph Vertex Attribute
template <class TGroup,class TAV,class TAO,class TAS,class TAH>  class cGroupGraph;    ///  Group Graph


/*----------------------------------------- cGGA_EdgeOr --------------------------------------------------*/

/** Oriented attributes of graph-group, nothing to do 4 now */

template <class TGroup,class TAO>  class cGGA_EdgeOr
{
     public :
          cGGA_EdgeOr(const TAO & anAttr) : mAttr(anAttr) {}

          const TAO & Attr() const {return mAttr;}
          TAO & Attr() {return mAttr;}
     private :
          TAO mAttr;
};

/*----------------------------------------- cGG_1HypInit --------------------------------------------------*/

/**  Store an initial hypothesis of relative orientation , the containing several hypothesis */

template <class TGroup,class TAH>  class cGG_1HypInit
{
   public :
      cGG_1HypInit(const TGroup & aG,tREAL8 aW0,const TAH & anAttr) :
           mW0         (aW0), 
           mNumCluster (-1) , 
           mVal        (aG), 
           mNoiseSim   (0),
           mAttr       (anAttr)
      {}

      bool IsMarked() const {return mNumCluster != -1;}


      tREAL8   mW0;         /// initial weight , for example, could take into account the number of point 
      int      mNumCluster; /// the identifier of the cluster
      TGroup   mVal;        /// Value of the map/group
      tREAL8   mNoiseSim;   /// Noise used in/if simulation
      TAH      mAttr;       /// User's specific attribute
};

/*----------------------------------------- cGG_1HypClustered -----------------------------------------------*/

/**  Store a "clustered" hypothesis, this is an initial hypothesis + some additionnal info.
   The clustering of hypothesis is used to reduce the combinatory exlposion of all cycles.
*/
 
template <class TGroup,class TAH>  class cGG_1HypClustered : public cGG_1HypInit<TGroup,TAH>
{
    public  :
      cGG_1HypClustered(const TGroup & aG,tREAL8 aW) : 
           cGG_1HypInit<TGroup,TAH> (aG,aW,TAH()),
           mWeight              (aW) // weight = initial weight
      {}

      tREAL8                      mWeight;       /// weight of the cluster
      cWeightAv<tREAL8,tREAL8>    mWeightedDist; /// weighted dist on cycle-closing
};

/*----------------------------------------- cGGA_EdgeSym -----------------------------------------------*/

/** Store the symetric attribute of a "group-graph", essentially a set of hypothesis + some computation */

template <class TGroup,class TAS,class TAH>  class cGGA_EdgeSym
{
     public :
          typedef cGG_1HypInit<TGroup,TAH>        t1HypInit;
          typedef cGG_1HypClustered<TGroup,TAH>   t1HypClust;

          cGGA_EdgeSym() :  mBestH (nullptr) {}

          std::vector<t1HypClust> &  ValuesClust() {return mValuesClust;} ///< Accessor
          std::vector<t1HypInit> &  ValuesInit() {return mValuesInit;} ///< Accessor
          const std::vector<t1HypClust> &  ValuesClust() const {return mValuesClust;} ///< Accessor
          const std::vector<t1HypInit> &  ValuesInit() const {return mValuesInit;} ///< Accessor
	  /// Add 1 initial hypothesis 
          t1HypInit & Add1Value(const TGroup& aVal,tREAL8 aW,const  TAH &);
	  /// Do the clustering to reduce the number of initial hypothesis
          void DoCluster(int aNbMax,tREAL8 aDist);

          void SetBestH(t1HypClust * aBestH) {mBestH=aBestH;} ///< fix the best hypothesis
          const t1HypClust * BestH() const {return mBestH;}   ///< accessor
         

          /// return if possible the clustered hypotheseis associated 
          const t1HypClust *  HypCompOfH0(const t1HypInit&,bool SVP=false) const;

          /// extract the hypothesis complying with a certain condition
          template <class FoncTest>  const t1HypInit *  HypOfNumSet(const FoncTest & aFTest,bool SVP=false) const
          {
             for (const auto & aVH: mValuesInit)
                if (aFTest(aVH.mAttr))
                   return &aVH;

             MMVII_INTERNAL_ASSERT_tiny(SVP,"HypOfNumSet cannot find");// litle checj
             return nullptr;
         }

          const TAS & Attr() const {return mAttr;}   ///< Accessor
          TAS & Attr() {return mAttr;}               ///< Accessor
     private :

          /// for a given center return a score (+- weighted count of element not marked closed enough)
          tREAL8  CenterClusterScore(const TGroup &,tREAL8 aDist) const;

          ///  Add the next best center of the cluster, return true if added something
          bool  GetNextBestCCluster(tREAL8 aThDist);

          std::vector<t1HypInit>   mValuesInit;  /// set of hypotethic group associated , the rough data given by user
          std::vector<t1HypClust>  mValuesClust; /// cluster of these hypothesis
          t1HypClust *             mBestH;       /// the best hypothesis, used in miminal spaning (more for simul)
          TAS                      mAttr;        /// users part
};


/*----------------------------------------- cGGA_Vertex -----------------------------------------------*/

/** Class for storing the attibute of a vertex of graph-group */

template <class TGroup,class TAV>  class cGGA_Vertex
{
     public :
          cGGA_Vertex(const TAV & anAttr)  : mAttr (anAttr) { }

           const TAV & Attr() const {return mAttr;}
           TAV & Attr() {return mAttr;}
          
     private :
          TAV                 mAttr;           ///< User's attribute
          TGroup              mComputedValue;  ///<  will store estimated value (not used 4 now...)
};

/*----------------------------------------- cGroupGraph -----------------------------------------------*/

/** Class for manipulating the group - graph */

template <class TGroup,class TAV,class TAO,class TAS,class TAH>  
     class cGroupGraph :
             public cVG_Graph<cGGA_Vertex<TGroup,TAV>, cGGA_EdgeOr<TGroup,TAO>,cGGA_EdgeSym<TGroup,TAS,TAH>>
{
    public :
          // -------------------------------------- typedef part ---------------------------
          typedef cGGA_Vertex<TGroup,TAV>        tAVert;
          typedef cGGA_EdgeOr<TGroup,TAO>        tAEOr;
          typedef cGGA_EdgeSym<TGroup,TAS,TAH>   tAESym;
          typedef cGG_1HypInit<TGroup,TAH>       t1HypInit;
          typedef cGG_1HypClustered<TGroup,TAH>   t1HypClust;

          typedef cVG_Graph<tAVert,tAEOr,tAESym> tGraph;
          typedef typename tGraph::tVertex       tVertex;
          typedef typename tGraph::tEdge         tEdge;
          typedef cGroupGraph<TGroup,TAV,TAO,TAS,TAH>    tGrGr;
          typedef  cAlgo_ParamVG<tGrGr>            tSubGr;
          typedef  cAlgo_ParamVG<tGrGr>          tParamWG;

          typedef  cAlgoSP<tGrGr>                tAlgoSP;
          typedef  typename tAlgoSP::tForest     tForest;



          cGroupGraph();

	  /// create a vertex with no value
          tVertex &  AddVertex(const TAV&);

	  ///  Add a new hypothesis
          t1HypInit & AddHyp(tVertex& aN1,tVertex& aN2,const TGroup &,tREAL8 aW,const TAH & aH);

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
                std::vector<t1HypClust*> mVCurHyp;  // stack for hypothesis being explored
          };

          /// Class for computing weight in minimal spaning tree
	  class  cWeightOnBestH : public tParamWG
	  {
		  public :
                      tREAL8 WeightEdge(const    tEdge & anE) const override 
		      {
			      return anE.AttrSym().BestH()->mWeightedDist.Average();
		      }

	  };
          /// Used in bench simul 4 now
          tREAL8 MakeMinSpanTree();

    protected :
           tGraph * mGraph;       ///< the graph itself
           bool     mClusterDone; ///< have we already done the clustering
           int      mNbVHist;     ///< Number of value for normalization- histogram
           int      mCptC;        ///< Count the iteration of cycles, used for output
};


/* ********************************************************* */
/*                                                           */
/*                     cGGA_EdgeSym                          */
/*                                                           */
/* ********************************************************* */


template <class TGroup,class TAS,class TAH> 
     tREAL8 cGGA_EdgeSym<TGroup,TAS,TAH>::CenterClusterScore(const TGroup & aG,tREAL8 aThrDist) const
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


template <class TGroup,class TAS,class TAH> 
    const cGG_1HypClustered<TGroup,TAH> *  cGGA_EdgeSym<TGroup,TAS,TAH>::HypCompOfH0(const t1HypInit&aH0,bool SVP) const
{
   if (!aH0.IsMarked())
   {
       MMVII_INTERNAL_ASSERT_tiny(SVP,"HypCompOfH0 cannot find");// litle checj
       return nullptr;
   }
   return  & mValuesClust.at(aH0.mNumCluster);
}

template <class TGroup,class TAS,class TAH> bool  cGGA_EdgeSym<TGroup,TAS,TAH>::GetNextBestCCluster(tREAL8 aThrDist)
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
                  aVal.mNumCluster = mValuesClust.size();
                  aSumNoise += aVal.mNoiseSim * aVal.mW0;
		  aSumW += aVal.mW0;
            }
        }
     }

     // Ok we got a new center, add it in "mValuesClust"
     if (aSumW!=0)
     {
          mValuesClust.push_back(t1HypClust(aCenterClust,aSumW));
          mValuesClust.back().mNoiseSim = aSumNoise/aSumW;
          return true;
     }

     return false;
}

template <class TGroup,class TAS,class TAH> void cGGA_EdgeSym<TGroup,TAS,TAH>::DoCluster(int aNbMax,tREAL8 aDist)
{
    mValuesClust.clear();
    if (aDist<=0)
    {
        // In this case make a cluster of all points
        for (auto & aH0 : mValuesInit)
        {
           aH0.mNumCluster = mValuesClust.size();
           mValuesClust.push_back(t1HypClust(aH0.mVal,1.0));
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

template <class TGroup,class TAS,class TAH> 
   cGG_1HypInit<TGroup,TAH> & cGGA_EdgeSym<TGroup,TAS,TAH>::Add1Value(const TGroup& aVal,tREAL8 aW,const TAH & anAttrH)
{
    mValuesInit.push_back(t1HypInit(aVal,aW,anAttrH));
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

template  <class TGroup,class TAV,class TAO,class TAS,class TAH>  
  cGroupGraph<TGroup,TAV,TAO,TAS,TAH>::cGG_OnCycle::cGG_OnCycle() 
{
}

template  <class TGroup,class TAV,class TAO,class TAS,class TAH>  
   void cGroupGraph<TGroup,TAV,TAO,TAS,TAH>::cGG_OnCycle::OnCycle(const cAlgoEnumCycle<tGrGr> & anAlgo) 
{
   // store a local copy of the path
   mVCurPath = anAlgo.CurPath();
   
   // Throw recursive exploration 
   RecursOnCycle(TGroup::Identity(),0,1.0);
}


template  <class TGroup,class TAV,class TAO,class TAS,class TAH>  
   void cGroupGraph<TGroup,TAV,TAO,TAS,TAH>::cGG_OnCycle::RecursOnCycle(const TGroup & aG,size_t aDepth,tREAL8 aW)
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
    for (auto & anElem : anE.AttrSym().ValuesClust())
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

template <class TGroup,class TAV,class TAO,class TAS,class TAH>  
    cGroupGraph<TGroup,TAV,TAO,TAS,TAH>::cGroupGraph() :
        mGraph        (this),
        mClusterDone  (false),
        mNbVHist      (1000),
        mCptC         (0)
{
}


template <class TGroup,class TAV,class TAO,class TAS,class TAH>  
    typename cGroupGraph<TGroup,TAV,TAO,TAS,TAH>::tVertex &
        cGroupGraph<TGroup,TAV,TAO,TAS,TAH>::AddVertex(const TAV& aTAV)
{
    tVertex * aV = mGraph->NewVertex(tAVert(aTAV));
    return *aV;
}


template <class TGroup,class TAV,class TAO,class TAS,class TAH>  
    cGG_1HypInit<TGroup,TAH>& 
       cGroupGraph<TGroup,TAV,TAO,TAS,TAH>::AddHyp(tVertex & aV1,tVertex & aV2,const TGroup& aG,tREAL8 aW,const TAH & aH)
{
    //  Add edge if does not exist
    tEdge * anE = aV1.EdgeOfSucc(aV2,SVP::Yes);
    if (anE==nullptr)
       anE = mGraph->AddEdge(aV1,aV2,tAEOr(TAO()),tAEOr(TAO()),tAESym());

    //  now add the hypothesis to existing edge 
    return anE->AttrSym().Add1Value(ValOrient(aG,*anE),aW,aH);
}
  

template <class TGroup,class TAV,class TAO,class TAS,class TAH>  
   void cGroupGraph<TGroup,TAV,TAO,TAS,TAH>::DoClustering(size_t aNbMaxCluster,tREAL8 aDistClust)
{
   if (! mClusterDone)
   {
       mClusterDone = true;
       for (auto & anEPtr :  mGraph->AllEdges_DirInit ())
           anEPtr->AttrSym().DoCluster(aNbMaxCluster,aDistClust);
   }
}

template <class TGroup,class TAV,class TAO,class TAS,class TAH>  
   void cGroupGraph<TGroup,TAV,TAO,TAS,TAH>::OneIterCycles(size_t aSzMaxC,tREAL8 aPow,bool Show)
{
   mCptC++;
   // cluster the values is not already done

   // Reset the weights
   for (auto & aPtrE :  mGraph->AllEdges_DirInit ())
       for (auto & aH : aPtrE->AttrSym().ValuesClust())
           aH.mWeightedDist.Reset();



   cGG_OnCycle aOnC;
   cAlgoEnumCycle<tGrGr>  aAlgoEnum(*this,aOnC,tSubGr(),aSzMaxC); 
   aAlgoEnum.ExplorateAllCycles();


   // compute the histogramm of mWeightedDist to have some normalization of costs
   cHistoCumul<tREAL8,tREAL8> aHisto(TGroup::MaxDist()*mNbVHist+1);
   for (const auto & aPtrE : mGraph->AllEdges_DirInit ())
   {
       for (const auto & anH : aPtrE->AttrSym().ValuesClust())
       {
// StdOut() << " DDDD=" << anH.mWeightedDist.Average() << "\n";
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
       cWhichMin<t1HypClust*,tREAL8> aWHypMin;
       for (auto & anH : aPtrE->AttrSym().ValuesClust())
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

template <class TGroup,class TAV,class TAO,class TAS,class TAH>  
   tREAL8 cGroupGraph<TGroup,TAV,TAO,TAS,TAH>::MakeMinSpanTree()
{

    tForest aForest(*this);
    tAlgoSP anAlgoSP;
    cWeightOnBestH aWBE;
    anAlgoSP.MinimumSpanningForest(aForest,*this,this->AllVertices(), aWBE);  // extract the forest

    // Theoreitcally this can happen, but dont want to gandle it 4 now
    MMVII_INTERNAL_ASSERT_tiny(aForest.VTrees().size()==1,"cGroupGraph::MakeMinSpanTree Forest size");

    tREAL8 aMaxCost = -1.0;

    for (const auto &   aTree : aForest.VTrees())
    {
        for (const auto & aPtrE : aTree.Edges())
        {
            UpdateMax(aMaxCost,aPtrE->AttrSym().BestH()->mNoiseSim);
            // StdOut() << "W=" << aWBE.WeightEdge(*aPtrE) << " N=" << aPtrE->AttrSym().BestH()->mNoiseSim << "\n";
	}
    }
    // StdOut() << "MAX COST = " << aMaxCost << "\n";
    return aMaxCost;
}

}; //  namespace MMVII

#endif // _MMVII_Tpl_GraphGroup_H_
