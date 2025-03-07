#ifndef _MMVII_Tpl_GraphAlgo_SPCC_H_
#define _MMVII_Tpl_GraphAlgo_SPCC_H_

#include "MMVII_TplHeap.h"
#include "MMVII_Tpl_GraphStruct.h"
#include "MMVII_Tpl_Graph_SubGraph.h"

namespace MMVII
{
/** \file   MMVII_Tpl_GraphAlgo_SPCC.h
    \brief  Clases for Algorithm "sortest path/minimum spanning tree/connected components"
*/

//  -------- main classes for algorithm --------

template <class TGraph>  class cAlgoSP;  //  Class for shortest path & minimum spaning tree
template <class TGraph>  class cAlgoCC;  //  Class for connected components



//  -------- helper classes ---------------------
template <class TGraph>  class cHeapParam_VG_Graph;


/* ********************************************************************************* */
/*                                                                                   */
/*                                   cAlgoSP                                         */
/*                                                                                   */
/* ********************************************************************************* */

/**   Helper class for heap in cAlgoSP, define access to index of a vertex */ 
template <class TGraph>  class cHeapParam_VG_Graph
{
    public :
       typedef typename TGraph::tVertex        tVertex;
       typedef tVertex *         tVertexPtr ;
       static void SetIndex(const tVertexPtr & aPtrV,tINT4 i) { aPtrV->mAlgoIndexHeap = i;} 
       static int  GetIndex(const tVertexPtr & aPtrV)  { return aPtrV->mAlgoIndexHeap; }
};

/**   Class for computing shortest-path and minimum-spaning-tree/forest
      All the "hard" job is done by a single method :

              Internal_MakeShortestPathGen

     The other method are simplified interface to it .
*/

template <class TGraph>   class cVG_Tree
{
      public :
	  typedef typename TGraph::tVertex        tVertex;
	  typedef typename TGraph::tEdge          tEdge;
	  typedef std::vector<tEdge*>             tSetEdges;  // tree = set of edge
	  typedef cVG_Tree<TGraph>                tTree;  // tree = set of edge
          typedef std::array<tTree,2>             t2Tree;
          friend class cAlgoSP<TGraph>;
          
          const tSetEdges & Edges() const {return mEdges;}

          /// tree with non empty set of edges
          inline cVG_Tree(const tSetEdges & aSetE);
          /// tree with 0 or 1 edge
          cVG_Tree(TGraph & aGraph,tVertex * aV0=nullptr) :
                mGraph   (&aGraph),
                mV0      (aV0)
           {
           }
           /// Default constructor required for array of tree
           cVG_Tree() : mGraph(nullptr), mV0(nullptr) {}

           void Split(t2Tree&,tEdge *);
           std::vector<tVertex*> Vertices() const {return tEdge::VerticesOfEdges(mEdges);}
      private :
           void Tree_clear()
           {
               mV0 = nullptr;
               mEdges.clear();
           }

           void SetV0(tVertex * aV0) {mV0=aV0;}
           void AddEdge(tEdge * anE) {mEdges.push_back(anE);}

           TGraph *   mGraph;
           tVertex *  mV0;
           tSetEdges  mEdges;
};






template <class TGraph>   class cVG_Forest
{
    public :
	  typedef cVG_Tree<TGraph>  tTree;
          friend class cAlgoSP<TGraph>;

          const std::list<tTree> & VTrees() const {return  mVTrees;}
          cVG_Forest(TGraph & aGraph) : mGraph(&aGraph) {}
    private :
          void Clear() {mVTrees.clear();}

          TGraph *            mGraph;
          std::list<tTree>    mVTrees;
};





template <class TGraph>   class cAlgoSP
{
     public :
          typedef  cAlgo_ParamVG<TGraph>          tParamA;
          typedef  cAlgo_ParamVG<TGraph>          tSubGr;
	  typedef typename TGraph::tVertex        tVertex;
	  typedef typename TGraph::tEdge          tEdge;
          typedef std::vector<tVertex*>           tVectVertex;

          //  -------  classes used for defining the result of spaning tree/forest --------------

	  // typedef std::pair<tVertex*,tEdge*>      tPairVE;     
	  //typedef std::vector<tPairVE>            tSetPairVE;  // tree = set of edge
	  //typedef std::vector<tEdge*>            tSetEdges;  // tree = set of edge
          // 1 elem of forest = the tree (set of edge) + the seed,  the seed is required in case of emty egde
          // to reconstruct the solution
	  //typedef std::pair<tVertex*,tSetEdges>  tTree; 
	  //typedef std::list<tTree>         tForest;   // A Forest = list of element of forest
          typedef   cVG_Tree<TGraph>         tTree;
          typedef   cVG_Forest<TGraph>       tForest;
          
          const std::vector<tVertex*> & VVertexReached() const {return mVVertexReached;}
          const std::vector<tVertex*> & VVertexOuted() const   {return mVVertexOuted;}
          /** compute the shotest path between "aBegin" and "aEnd" in graph "aGraph", the length of edges
              and the valide Vertex/Edges are indicated by "aParam".  If the path is succesfully computed
              aEnd is returned, else nullptr
              For accesing to result, the user can consult in class vertices:
                    - AlgoCost to know the length of the path
                    - BackTrackFathersPath to compute the path
          */
          tVertex * ShortestPath_A2B
                 (
                    TGraph &          aGraph,
                    tVertex &         aBegin,
                    tVertex &         aEnd,
                    const tParamA  &  aParam
                 );


           /** compute the minimum spanning tree starting from  aSeed */
           tTree  MinimumSpanninTree (TGraph & aGraph, tVertex &  aSeed, const tParamA &);
           /** idem, but avoi copy of result */
           void  MinimumSpanninTree (tTree&,TGraph & aGraph, tVertex &  aSeed, const tParamA &);

           /** compute the minimal forest (= set of tree) crossing  a set of seeds */
           void MinimumSpanningForest(tForest&,TGraph &,const tVectVertex&  aSeed, const tParamA &);

           /** generic public method for computing shortest path & trees */
           tVertex * MakeShortestPathGen
                  (
                      TGraph &              aGraph,     // the graph on which it is computed
                      bool                 aModeMinSP,  // true=shortest path, false= minimum spaning tree
                      const tVectVertex &  aVSeeds,     // set of starting points 
                      const tParamA  &     aParam,      // parameter of the computation (sub-graph/weigthing=
                      const tSubGr   &     aGoal        // goal to be reached,  will end the computation
                  );

          private :
              // clas for ordering vertex in the heap
              class cHeapCmp
              {
                 public :
                    typedef tVertex *         tVertexPtr ;
                    bool  operator () (const tVertexPtr & aV1,const tVertexPtr & aV2) 
                    {
                       return aV1->AlgoCost() < aV2->AlgoCost();
                    }
              };

              tVertex * Internal_MakeShortestPathGen
                 (
                    TGraph &             aGraph,         // graph
                    bool                 aModeMinPCC,    // Shortest Path /Min Span Tree
                    const tVectVertex &  aVSeeds,        // set of seeds
                    const tParamA   &    aParam,         // paramete (sub gr/weitgh)
                    const tSubGr   &     aGoal,          // goal ending computatio,
                    size_t               aBitReached,    // flag to mark vertices reached
                    size_t               aBitOutHeap,    // flag to mark vertices outed from heap
                    bool                 CleanAnFreeBits // to we 1-un mark the vertices 2-recycle the flag in graph
                 );

              void  Internal_MinimumSpanninTree 
                    (
                          tTree&           aTree,
                          TGraph &         aGraph, 
                          tVertex &        aSeed, 
                          const tParamA &  aParam,
                          size_t           aBitReached,
                          size_t           aBitOutHeap,
                          bool             CleanAnFreeBits
                    ); 

         std::vector<tVertex*>   mVVertexReached;  ///<  Vertex reached, cost was initialized
         std::vector<tVertex*>   mVVertexOuted;    ///< Vertex outed of the heap, cost is computed OK

};


/*   Generic method 

        aBitOutHeap  : is used to know if vertex has already been outed of heap, in this case nothing to do
                       when we reach it

        aBitReached : is used to know it a vertex has already been reached, this change if we initialize cost
                       or just update
*/

template <class TGraph>   
    typename TGraph::tVertex * cAlgoSP<TGraph>::Internal_MakeShortestPathGen
         (
              TGraph &                       aGraph,
              bool                           aModeMinPCC,
              const std::vector<tVertex*> &  aVSeeds,
              const cAlgo_ParamVG<TGraph> &  aParam,
              const cAlgo_ParamVG<TGraph>   &  aGoal,
              size_t                         aBitReached,
              size_t                         aBitOutHeap,
              bool                           CleanAnFreeBits
         )
{
    tVertex * aResult = nullptr;  // 4 now we have not reached the goal

    mVVertexReached.clear(); 
    mVVertexOuted.clear();    

    // [0]  Initialize Heap
    cHeapCmp aCompare;
    typedef cHeapParam_VG_Graph<TGraph>       tHeapParam;
    cIndexedHeap<tVertex*,cHeapCmp,tHeapParam> aHeap(aCompare);
 
    // [1]  Put in heap all the seeds  
    for (auto aPtrV : aVSeeds)
    {
        // consider only seeds in the graph, on only once
        if (aParam.InsideVertex(*aPtrV) && (!aPtrV->BitTo1(aBitReached))) 
        {
           aPtrV->SetBit1(aBitReached);   // mark that vertex has been reached
           aPtrV->mAlgoCost = 0.0;        // cost 4 empty path
           aPtrV->mAlgoFather = nullptr;  // no father for seed points
           mVVertexReached.push_back(aPtrV);    // memorize for clean
           aHeap.Push(aPtrV);             // and, of course, initialize the heap
        }
    }

    // [2]  core of the method 
    bool  GoOn= true;
    while (GoOn)
    {
        if (aHeap.IsEmpty())  // if heap is empty, end of the game
        {
            GoOn = false;
        }
        else
        {
            tVertex * aNewVOut =  aHeap.PopVal(nullptr); // get best vertex
            mVVertexOuted.push_back(aNewVOut);
            if (aGoal.InsideVertex(*aNewVOut))  // is we have reached the goal, end of the game
            {
               GoOn = false;
               aResult = aNewVOut;  //  The result is first vertex reached
            }
            else
            {
                 aNewVOut->SetBit1(aBitOutHeap);  // mark newly outed as an "outed of heap vertex"
                 for (const auto &  anEdge :  aNewVOut->EdgesSucc()) // parse all  neighbours for possible update
                 {
                      if (aParam.InsideV1AndEdgeAndSucc(*anEdge))  // consider only vertex in the sub-graph
                      {
                         tVertex & aNewVIn = anEdge->Succ();  // extract the neighouring vertex

                         if (!aNewVIn.BitTo1(aBitOutHeap))  // is vertex has not already been outed 
                         {
                            // compute cost, in mode spaning tree just the cost  of reaching a vertex by this edge
                            // is just the cost of the edge, in mode shortest path
                            tREAL8 aNewCost =  aParam.WeightEdge(*anEdge);
                            if (aModeMinPCC)   // @DIF:SP:MST
                               aNewCost += aNewVOut->mAlgoCost;
                            else
                            {
                            }

                            // if vertex was never reached, initialize its puting in the heap
                            if (!aNewVIn.BitTo1(aBitReached))
                            {
                               mVVertexReached.push_back(&aNewVIn);
                               aNewVIn.SetBit1(aBitReached);   // memorize it is reached
                               aNewVIn.mAlgoCost = aNewCost+1; // trick to make "previous" cost worst than new one
                               aHeap.Push(&aNewVIn);           // put it in the heap
                            }

                            // is new way to comput cost is betters than current 
                            if (aNewCost<aNewVIn.mAlgoCost)
                            {
                                 aNewVIn.mAlgoCost = aNewCost;   // update cost
                                 aHeap.UpDate(&aNewVIn);         // ask heap to restore the heap propert that may have been broken
                                 aNewVIn.mAlgoFather = aNewVOut; // update the link to best path
                            }
                         }
                      }
                 }
            }
        }
    }

    // [3]  clean marks & recycle flags
    if (CleanAnFreeBits)
    {
        // unmark all vertices reached
        tVertex::SetBit0(mVVertexReached,aBitReached);
        tVertex::SetBit0(mVVertexReached,aBitOutHeap);

        // recycle flag so that they can be used again
        aGraph.FreeBitTemp(aBitReached);
        aGraph.FreeBitTemp(aBitOutHeap);
    }

    return aResult;
}  

   
       /* ======================= Some interfaces for shortest path ============================== */


    /* public general interface for computing shortest path:
           - the seeds is multiple, given by extension,  it's a set of vector
           - the goal  is multiple, given by comprehension, it's sub-grap
    */


    /* public general interface for computing shortest path:
           - the seeds is multiple, given by extension,  it's a set of vector
           - the goal  is multiple, given by comprehension, it's sub-grap
    */

template <class TGraph>   
    typename TGraph::tVertex * cAlgoSP<TGraph>::MakeShortestPathGen
         (
              TGraph &                      aGraph,
              bool                          aModeMinPCC,
              const std::vector<tVertex*> & aVSeeds,
              const cAlgo_ParamVG<TGraph> & aParam,
              const cAlgo_ParamVG<TGraph>   & aGoal
         )
{
    size_t aBitReached = aGraph.AllocBitTemp();
    size_t aBitOutHeap  = aGraph.AllocBitTemp();
    tVertex * aResult  = Internal_MakeShortestPathGen
                         (
                            aGraph,aModeMinPCC,aVSeeds,aParam,aGoal,
                            aBitReached,aBitOutHeap,true
                         );

    return aResult;
}


    /* specific interface for the most common case, shortest path
       between 2 vertices  aBegin->aEnd 
    */

template <class TGraph>   
    typename TGraph::tVertex * cAlgoSP<TGraph>::ShortestPath_A2B
                 (
                    TGraph &                       aGraph,
                    tVertex &  aBegin,
                    tVertex &  aEnd,
                    const cAlgo_ParamVG<TGraph> &  aParam
                 )
{
   return
      MakeShortestPathGen
      (
            aGraph,
            true,
            {&aBegin},   // make a vector at 1 element "Begin"
            aParam,
            cAlgo_SingleSubGr<TGraph>(&aEnd)  // make a goal, the singleton "End"
      );
}

      /* =========================  Some interfaces for minimum spaning tree ======================= */

template <class TGraph>   
     void cAlgoSP<TGraph>::Internal_MinimumSpanninTree 
           (
                          tTree& aTree,
                          TGraph & aGraph, 
                          tVertex &  aSeed, 
                          const cAlgo_ParamVG<TGraph> &  aParam,
                          size_t                         aBitReached,
                          size_t                         aBitOutHeap,
                          bool                           CleanAnFreeBits
           )
{
    // write seed in Tree , 0 if the vertex does not belong to sub-graph
    //## aTree.first = aParam.InsideVertex(aSeed) ? &aSeed : nullptr;
    aTree.SetV0(aParam.InsideVertex(aSeed) ? &aSeed : nullptr);

    tVertex * aResult  = Internal_MakeShortestPathGen
                         (
                            aGraph,
                            false,  // Not Shortest Path <=> Min Span Tree
                            {&aSeed},
                            aParam,
                            cAlgo_SubGrNone<TGraph>(),  // goal is never reached, we want to go as far as possible
                            aBitReached,
                            aBitOutHeap,
                            CleanAnFreeBits
                         );
    // litle  check, with goal none, result should be none
    MMVII_INTERNAL_ASSERT_always(aResult==nullptr,"Internal_MinimumSpanninTree result!=0");

    // compute the edge that were use,
    {
       // ## tSetEdges & aSetPair = aTree.second;
       aTree.mEdges.clear();
       for (const auto aPtrV : mVVertexReached)  // parse reached point
       {
           if (aPtrV->mAlgoFather != nullptr)  // avoid seeds
              aTree.mEdges.push_back(aPtrV->EdgeOfSucc(*(aPtrV->mAlgoFather)));
       }
    }
}


/** simplified interface, case we compute the tree with a single vertex as seed */

template <class TGraph> 
     void cAlgoSP<TGraph>::MinimumSpanninTree
          (
                tTree   & aTree,
                TGraph & aGraph, 
                tVertex &  aSeed, 
                const cAlgo_ParamVG<TGraph> &  aParam
          )
{
    size_t aBitReached = aGraph.AllocBitTemp();
    size_t aBitOutHeap  = aGraph.AllocBitTemp();

     Internal_MinimumSpanninTree 
     (
          aTree,aGraph,aSeed,aParam,
          aBitReached,
          aBitOutHeap,
          true
     );
}

/** idem, but return the value (simpler but makes copy)*/
template <class TGraph> 
     typename  cAlgoSP<TGraph>::tTree cAlgoSP<TGraph>::MinimumSpanninTree
          (
                TGraph & aGraph, 
                tVertex &  aSeed, 
                const cAlgo_ParamVG<TGraph> &  aParam
          )
{
       tTree aTree(aGraph);
       MinimumSpanninTree(aTree,aGraph,aSeed,aParam);

       return aTree;
}

/**  Spaning forest, +or-  call spaning with all the seed, but must avoid duplication */

template <class TGraph> 
     void cAlgoSP<TGraph>::MinimumSpanningForest
          (
              tForest& aForest,
              TGraph & aGraph,
              const tVectVertex&  aVectSeed, 
              const tParamA & aParam
          )
{
    aForest.Clear();  // reset, just in case
    // alloc markers
    size_t aBitReached = aGraph.AllocBitTemp();  
    size_t aBitOutHeap  = aGraph.AllocBitTemp();


    for (const auto & aSeed : aVectSeed)  // parse all seed
    {
        // avoid  Vertex not in graph  AND  vertex already reached by a previous tree
        if (aParam.InsideVertex(*aSeed) && (!aSeed->BitTo1(aBitReached)))
        {
           aForest.mVTrees.push_back(tTree(aGraph,aSeed));
        
           // compute tree with the seed
           Internal_MinimumSpanninTree 
           (
                 aForest.mVTrees.back(),
                 aGraph,*aSeed,aParam,
                 aBitReached,
                 aBitOutHeap,
                 false    // No clean, because marker are used to avoid duplicatio,
           );
        }
   }

   // do the cleaning  because it was not made by Internal_MinimumSpanninTree
   for (const auto & aTree :  aForest.mVTrees)
   {
        if (aTree.mV0)
        {
            aTree.mV0->SetBit0(aBitReached);
            aTree.mV0->SetBit0(aBitOutHeap);
        }
        for (const auto & anE : aTree.mEdges)
        {
            anE->VertexInit().SetBit0(aBitReached);
            anE->VertexInit().SetBit0(aBitOutHeap);
            anE->Succ().SetBit0(aBitReached);
            anE->Succ().SetBit0(aBitOutHeap);
        }
   }

   // recycle the bits for future use
   aGraph.FreeBitTemp(aBitReached);
   aGraph.FreeBitTemp(aBitOutHeap);
}

                  

/* ********************************************************************************* */
/*                                                                                   */
/*                                   cAlgoCC                                         */
/*                                                                                   */
/* ********************************************************************************* */

/**   Class for computing connected components :
      All the "hard" job is done by a 1 or 2 method :

             essentially   Internal_ConnectedComponent
             also          Internal_Multiple_ConnectedComponent

     The other method are simplified interface to them .
*/


template <class TGraph>   class cAlgoCC
{
     public :
	  typedef typename TGraph::tVertex  tVertex;

          // return the connected component of aSeed in sub-graph defined by aParam
	  static std::vector<tVertex *>  ConnectedComponent
		                  (
                                      TGraph & aGraph,
				      tVertex& aSeed,
                                      const cAlgo_ParamVG<TGraph> & aParam
				  );

          // return all the connected component of the set defined by aVectSeed
	  static std::list<std::vector<tVertex *>>  Multiple_ConnectedComponent
		                  (
                                      TGraph & aGraph,
				      const std::vector<tVertex*> & aVectSeed,
                                      const cAlgo_ParamVG<TGraph> & aParam
				  );

          // return all the connected component of the graph 
	  static std::list<std::vector<tVertex *>>  All_ConnectedComponent
		                  (
                                      TGraph & aGraph,
                                      const cAlgo_ParamVG<TGraph> & aParam
				  );

     private :
          // compute the connected component with aBitReached as marker, do not clean marker
	  static void  Internal_ConnectedComponent
		                  (
				      tVertex* aSeed,
				      std::vector<tVertex *>& aResult,
                                      const cAlgo_ParamVG<TGraph> & aParam,
				      size_t aBitReached
				  );

          // compute multiple CC with set of seeds "aVectSeed", put it in aRes, alloc markers and do the cleaning
	  static void  Internal_Multiple_ConnectedComponent
		                  (
                                      TGraph & aGraph,
                                      std::list<std::vector<tVertex *>>  & aRes,
				      const std::vector<tVertex*> & aVectSeed,
                                      const cAlgo_ParamVG<TGraph> & aParam
				  );

};


template <class TGraph> 
     void cAlgoCC<TGraph>::Internal_ConnectedComponent
          (
              tVertex* aSeed,                        // seed of the connected component
              std::vector<tVertex *>& aResult,       // data where we write result
              const cAlgo_ParamVG<TGraph> & aParam,  // param to define sub-graph
              size_t aBitReached                     // marker used to avoid multiple visit of one  vertex
          )
{
    aResult.clear();
    // initialize by puting seed in Result
    if (aParam.InsideVertex(*aSeed) &&  (!aSeed->BitTo1(aBitReached)))
    {
       aSeed->SetBit1(aBitReached);  // mark as already visited
       aResult.push_back(aSeed);     // add to queue
    }

    size_t aIndBottom = 0;  // index of botom  of queue, where begin neighbours not explored
    while (aIndBottom != aResult.size())  // while there is neighbours to explorate
    {
       tVertex * aVCur =  aResult.at(aIndBottom);  // extract next vertex
       for (const auto &  anEdge :  aVCur->EdgesSucc())  // parse all edge V1->V2 neighbours 
       {
           if (aParam.InsideV1AndEdgeAndSucc(*anEdge))  // check they are in sub-graph
           {
              tVertex & aVNext = anEdge->Succ();    // extract V2
	      if (!  aVNext.BitTo1(aBitReached))    // check if V2 has already been visited
	      {
                  aResult.push_back(&aVNext);  // add new som
                  aVNext.SetBit1(aBitReached);  // mark it as already visited
	      }
           }
       }
       aIndBottom++;  // this one was explorated => increment
    }

}

template <class TGraph> 
     std::vector< typename TGraph::tVertex *> 
           cAlgoCC<TGraph>::ConnectedComponent
            (
                                      TGraph & aGraph,
				      tVertex& aSeed,
                                      const cAlgo_ParamVG<TGraph> & aParam
            )
{
    size_t aBitReached = aGraph.AllocBitTemp();  // alloc marker
    std::vector<tVertex*>   aResult;

    Internal_ConnectedComponent(&aSeed,aResult,aParam,aBitReached);

    // clean & free
    tVertex::SetBit0(aResult,aBitReached);
    aGraph.FreeBitTemp(aBitReached);

    return aResult;
}

template <class TGraph>
     void   cAlgoCC<TGraph>::Internal_Multiple_ConnectedComponent
            (
                 TGraph & aGraph,
                 std::list<std::vector<tVertex *>>  & aResult,
                 const std::vector<tVertex*> & aVecSeed,
                 const cAlgo_ParamVG<TGraph> & aParam
            )
{
    size_t aBitReached = aGraph.AllocBitTemp();  // alloc marker

    for (const auto & aSeed : aVecSeed) // parse all seeds
    {
	aResult.push_back(std::vector<tVertex*>());  // create a new empty component
	Internal_ConnectedComponent(aSeed,aResult.back(),aParam,aBitReached);  // fill it
        // if it was empty (already visited, or not in graph) 
	if (aResult.back().empty()) 
	    aResult.pop_back();
    }

    for (const auto & aVec : aResult)
         tVertex::SetBit0(aVec,aBitReached);

    aGraph.FreeBitTemp(aBitReached);
}


template <class TGraph> 
     std::list<std::vector< typename TGraph::tVertex *> >
           cAlgoCC<TGraph>::Multiple_ConnectedComponent
           (
                  TGraph & aGraph,
                  const std::vector<tVertex*> & aVecSeed,
                  const cAlgo_ParamVG<TGraph> & aParam
            )
{
    std::list<std::vector<tVertex*>>   aResult;
    Internal_Multiple_ConnectedComponent(aGraph,aResult,aVecSeed,aParam);
    return aResult;
}

template <class TGraph> 
     std::list<std::vector< typename TGraph::tVertex *> >
                    cAlgoCC<TGraph>::All_ConnectedComponent
		    (
                                      TGraph & aGraph,
                                      const cAlgo_ParamVG<TGraph> & aParam
                    )
{
    std::list<std::vector<tVertex*>>   aResult;
    Internal_Multiple_ConnectedComponent(aGraph,aResult,aGraph.AllVertices(),aParam);
    return aResult;
}

/* ********************************************************************************* */
/*                                                                                   */
/*                            cAlgoPruningExtre                                      */
/*                                                                                   */
/* ********************************************************************************* */

/** Class for "pruning" algorithm , i.e recursive supression of extremity with conservation
 * of anchor points */

template <class TGraph> class cAlgoPruningExtre
{
    public :
       typedef typename TGraph::tVertex tVertex;
       typedef  cAlgo_ParamVG<TGraph>     tSubGr;

       cAlgoPruningExtre(TGraph &aGraph,const tSubGr & aSubGr ,const tSubGr &aSubAnchor) ;
       const std::vector<tVertex*> & Extrem() const {return mExtrem;} ///< accesor to the result

    private :
       void  TestExtreAndSupr(tVertex *);

       TGraph &               mGraph;     ///< graph we are working on
       const tSubGr &         mSubGr;     ///< subgraph we are working on
       const tSubGr &         mAnchor;    ///< graph for anchor point that must be maintained
       size_t                 mFlagSupr;  ///< Flag to memorize supressed vertices
       std::vector<tVertex*>  mExtrem;	  ///< store the results (recursive extremity)
};

template <class TGraph> cAlgoPruningExtre<TGraph>::cAlgoPruningExtre(TGraph &aGraph,const tSubGr & aSubGr ,const tSubGr &aSubAnchor) :
    mGraph     (aGraph),
    mSubGr     (aSubGr),
    mAnchor    (aSubAnchor),
    mFlagSupr  (mGraph.AllocBitTemp())
{
   // initialize , compute all the initial extremities
   for (auto & aVertex :  aGraph.AllVertices())
   {
       TestExtreAndSupr(aVertex);
   }

   // StdOut() << "-----ITER 0----- " <<  mExtrem.size() << "\n";

   // recursive supression of extremities
   size_t aInd0 = 0;
   while (aInd0!=mExtrem.size())
   {
         for (const auto & anE : mExtrem.at(aInd0)->EdgesSucc())
	 {
             TestExtreAndSupr(&anE->Succ());
	 }
         aInd0++;
   }

   tVertex::SetBit0(mExtrem,mFlagSupr);  // unmark all the marked vertices
   mGraph.FreeBitTemp(mFlagSupr); // recycle the bit allocated
}

template <class TGraph> void  cAlgoPruningExtre<TGraph>::TestExtreAndSupr(tVertex * aVertex)
{
    // eliminate point that "structurally" cannot be extremity
    if (
             mAnchor.InsideVertex(*aVertex)     // anchor points must not be eliminated
          || (!mSubGr.InsideVertex(*aVertex))   // point oustide the graph must not be eliminated
	  || (aVertex->BitTo1(mFlagSupr))       // point already supressed must not be revisited
       )
    {
       // StdOut() << "OUT " << aVertex->Attr().mPt   <<   "\n";
       return ;
    }

    // count the number of valid remaining neighboors
    int aNbSucc = 0;
    for (const auto & anEdge : aVertex->EdgesSucc())
    {
        const tVertex & aSucc = anEdge->Succ();
        if (mSubGr.InsideV1AndEdgeAndSucc(*anEdge) && (! aSucc.BitTo1(mFlagSupr)))
           aNbSucc++;
    }
    //  extremity por single submit must be supressed
    if (aNbSucc <=1)
    {
        // StdOut() << " IN " << aVertex->Attr().mPt << " NBS=" << aNbSucc   <<   "\n";
        aVertex->SetBit1(mFlagSupr);
        mExtrem.push_back(aVertex);
    }
}

/* ********************************************************************************* */
/*                                                                                   */
/*                                cVG_Tree                                           */
/*                                                                                   */
/* ********************************************************************************* */

template <class TGraph>   cVG_Tree<TGraph>::cVG_Tree(const tSetEdges & aSetE) :
    mEdges (aSetE)
{
#if (The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_tiny )
    if (mEdges.empty())
    {
       MMVII_INTERNAL_ERROR("Empty edges in cVG_Tree");
    }
    mGraph  = &  aSetE.at(0)->Graph();
    mV0 = &  aSetE.at(0)->Succ();  // any set will be ok
    cSubGraphOfEdges<TGraph>  aSubGrTree(*mGraph,mEdges);
       
    auto aVV = tEdge::VerticesOfEdges(mEdges);
    auto allCC = cAlgoCC<TGraph>::Multiple_ConnectedComponent(*mGraph,aVV,aSubGrTree);
    MMVII_INTERNAL_ASSERT_always(allCC.size()==1,"Multi CC in tree");
    const auto & aCC0 = *allCC.begin();
    MMVII_INTERNAL_ASSERT_always(aCC0.size()==mEdges.size()+1,"cycle in tree");
#endif
}


}; // MMVII
#endif  // _MMVII_Tpl_GraphAlgo_SPCC_H_




