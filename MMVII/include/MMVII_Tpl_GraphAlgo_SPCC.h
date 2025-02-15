#ifndef _MMVII_Tpl_GraphAlgo_SPCC_H_
#define _MMVII_Tpl_GraphAlgo_SPCC_H_

#include "MMVII_TplHeap.h"
#include "MMVII_Tpl_GraphStruct.h"

namespace MMVII
{
/** \file   MMVII_Tpl_GraphAlgo_SPCC.h
    \brief  Clases for Algorithm "sortest path/minimum spanning tree/connected components"
*/

//  -------- main classes for algorithm --------

template <class TGraph>  class cAlgoSP;  //  Class for shortest path & minimum spaning tree
template <class TGraph>  class cAlgoCC;  //  Class for connected components


//  -------- class for parametrizing the algorithm : sub-graphs & weithing

template <class TGraph>  class cAlgo_SubGr;         // mother class for sub-graph
template <class TGraph>  class cAlgo_SingleSubGr ;  // singleton sub-graph
template <class TGraph>  class cAlgo_SubGrNone ;    // empty sub-braph
template <class TGraph>  class cAlgo_ParamVG ;      // parameters for algorithm

//  -------- helper classes ---------------------
template <class TGraph>  class cHeapParam_VG_Graph;


/**   interface-class for sub-graph, default  :  all edges &vertices belong to the graph */

template <class TGraph>  class cAlgo_SubGr
{
        public :
	     typedef typename TGraph::tVertex  tVertex;
	     typedef typename TGraph::tEdge    tEdge;

             virtual bool   InsideVertex(const  tVertex &) const {return true;}

             // take as parameter V1 & edge E=V1->V2 (because we cannot acces V1 from E)
             virtual bool   InsideEdge(const tVertex &,const    tEdge &) const {return true;}
             
             // method frequently used, so that user only redefines InsideEdge
             inline bool   InsideV1AndEdgeAndSucc(const tVertex & aV1,const    tEdge & anEdge) const 
             {
                    return this->InsideEdge(aV1,anEdge) && this->InsideVertex(anEdge.Succ()) && this->InsideVertex(aV1);
             }

};


/**   define a sub-graph than contain a single vertex, used for example as a goal in shortest path */
template <class TGraph>  class cAlgo_SingleSubGr : public cAlgo_SubGr<TGraph>
{
        public :
	     typedef typename TGraph::tVertex  tVertex;

             bool   InsideVertex(const  tVertex & aV) const override {return (mSingleV==&aV);}
	     cAlgo_SingleSubGr(const tVertex * aV) : mSingleV (aV) {}
        private :
	     const tVertex * mSingleV;
};

/**   define a sub-graph than contain nothing, used for example as a goal in minimal spanning tree */
template <class TGraph>  class cAlgo_SubGrNone : public cAlgo_SubGr<TGraph>
{
        public :
	     typedef typename TGraph::tVertex  tVertex;

             bool   InsideVertex(const  tVertex &) const override {return false;}
        private :
};


/**   parametrizarion of algorithm */

template <class TGraph>  class cAlgo_ParamVG : public cAlgo_SubGr<TGraph>
{
        public :
	     typedef typename TGraph::tVertex  tVertex;
	     typedef typename TGraph::tEdge    tEdge;

             virtual tREAL8 WeightEdge(const tVertex &,const    tEdge &) const {return 1.0;}

};

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

template <class TGraph>   class cAlgoSP
{
     public :
          typedef  cAlgo_ParamVG<TGraph>          tParamA;
          typedef  cAlgo_SubGr<TGraph>            tSubGr;
	  typedef typename TGraph::tVertex        tVertex;
	  typedef typename TGraph::tEdge          tEdge;
          typedef std::vector<tVertex*>           tVectVertex;

          //  -------  classes used for defining the result of spaning tree/forest --------------

	  typedef std::pair<tVertex*,tEdge*>      tPairVE;     
	  typedef std::vector<tPairVE>            tSetPairVE;  // tree = set of edge
          // 1 elem of forest = the tree (set of edge) + the seed,  the seed is required in case of emty egde
          // to reconstruct the solution
	  typedef std::pair<tVertex*,tSetPairVE>  tTree; 
	  typedef std::list<tTree>         tForest;   // A Forest = list of element of forest
          
          /** compute the shotest path between "aBegin" and "aEnd" in graph "aGraph", the length of edges
              and the valide Vertex/Edges are indicated by "aParam".  If the path is succesfully computed
              aEnd is returned, else nullptr
              For accesing to result, the user can consult in class vertices:
                    - AlgoCost to know the length of the path
                    - BackTrackFathersPath to compute the path
          */
          static tVertex * ShortestPath_A2B
                 (
                    TGraph &          aGraph,
                    tVertex &         aBegin,
                    tVertex &         aEnd,
                    const tParamA  &  aParam
                 );


           /** compute the minimum spanning tree starting from  aSeed */
           static tTree  MinimumSpanninTree (TGraph & aGraph, tVertex &  aSeed, const tParamA &);
           /** idem, but avoi copy of result */
           static void  MinimumSpanninTree (tTree&,TGraph & aGraph, tVertex &  aSeed, const tParamA &);

           /** compute the minimal forest (= set of tree) crossing  a set of seeds */
           static void MinimumSpanningForest(tForest&,TGraph &,const tVectVertex&  aSeed, const tParamA &);

           /** generic public method for computing shortest path & trees */
           static tVertex * MakeShortestPathGen
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

              static tVertex * Internal_MakeShortestPathGen
                 (
                    TGraph &             aGraph,         // graph
                    bool                 aModeMinPCC,    // Shortest Path /Min Span Tree
                    const tVectVertex &  aVSeeds,        // set of seeds
                    const tParamA   &    aParam,         // paramete (sub gr/weitgh)
                    const tSubGr   &     aGoal,          // goal ending computatio,
                    tVectVertex &        aVReached,      // store all vertex reached (out of heap or still waiting)
                    size_t               aBitReached,    // flag to mark vertices reached
                    size_t               aBitOutHeap,    // flag to mark vertices outed from heap
                    bool                 CleanAnFreeBits // to we 1-un mark the vertices 2-recycle the flag in graph
                 );

              static void  Internal_MinimumSpanninTree 
                    (
                          tTree&           aTree,
                          TGraph &         aGraph, 
                          tVertex &        aSeed, 
                          const tParamA &  aParam,
                          size_t           aBitReached,
                          size_t           aBitOutHeap,
                          bool             CleanAnFreeBits
                    ); 
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
              const cAlgo_SubGr<TGraph>   &  aGoal,
              std::vector<tVertex*> &        aVReached,
              size_t                         aBitReached,
              size_t                         aBitOutHeap,
              bool                           CleanAnFreeBits
         )
{
    tVertex * aResult = nullptr;  // 4 now we have not reached the goal

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
           aVReached.push_back(aPtrV);    // memorize for clean
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
                      if (aParam.InsideV1AndEdgeAndSucc(*aNewVOut,*anEdge))  // consider only vertex in the sub-graph
                      {
                         tVertex & aNewVIn = anEdge->Succ();  // extract the neighouring vertex

                         if (!aNewVIn.BitTo1(aBitOutHeap))  // is vertex has not already been outed 
                         {
                            // compute cost, in mode spaning tree just the cost  of reaching a vertex by this edge
                            // is just the cost of the edge, in mode shortest path
                            tREAL8 aNewCost =  aParam.WeightEdge(*aNewVOut,*anEdge);
                            if (aModeMinPCC)  
                               aNewCost += aNewVOut->mAlgoCost;
                            else
                            {
                            }

                            // if vertex was never reached, initialize its puting in the heap
                            if (!aNewVIn.BitTo1(aBitReached))
                            {
                               aVReached.push_back(&aNewVIn);  // memorize for clean
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
        tVertex::SetBit0(aVReached,aBitReached);
        tVertex::SetBit0(aVReached,aBitOutHeap);

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

template <class TGraph>   
    typename TGraph::tVertex * cAlgoSP<TGraph>::MakeShortestPathGen
         (
              TGraph &                      aGraph,
              bool                          aModeMinPCC,
              const std::vector<tVertex*> & aVSeeds,
              const cAlgo_ParamVG<TGraph> & aParam,
              const cAlgo_SubGr<TGraph>   & aGoal
         )
{
    std::vector<tVertex*>   aVReached;
    size_t aBitReached = aGraph.AllocBitTemp();
    size_t aBitOutHeap  = aGraph.AllocBitTemp();
    tVertex * aResult  = Internal_MakeShortestPathGen
                         (
                            aGraph,aModeMinPCC,aVSeeds,aParam,aGoal,
                            aVReached,aBitReached,aBitOutHeap,true
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
    aTree.first = aParam.InsideVertex(aSeed) ? &aSeed : nullptr;

    std::vector<tVertex*>   aVReached;
    tVertex * aResult  = Internal_MakeShortestPathGen
                         (
                            aGraph,
                            false,  // Not Shortest Path <=> Min Span Tree
                            {&aSeed},
                            aParam,
                            cAlgo_SubGrNone<TGraph>(),  // goal is never reached, we want to go as far as possible
                            aVReached,                  // get reached for computing pairs
                            aBitReached,
                            aBitOutHeap,
                            CleanAnFreeBits
                         );
    // litle  check, with goal none, result should be none
    MMVII_INTERNAL_ASSERT_always(aResult==nullptr,"Internal_MinimumSpanninTree result!=0");

    // compute the edge that were use,
    {
       tSetPairVE& aSetPair = aTree.second;
       aSetPair.clear();
       for (const auto aPtrV : aVReached)  // parse reached point
       {
           if (aPtrV->mAlgoFather != nullptr)  // avoid seeds
              aSetPair.push_back(tPairVE(aPtrV,aPtrV->EdgeOfSucc(*(aPtrV->mAlgoFather))));
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
       tTree aTree;
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
    aForest.clear();  // reset, just in case
    // alloc markers
    size_t aBitReached = aGraph.AllocBitTemp();  
    size_t aBitOutHeap  = aGraph.AllocBitTemp();


    for (const auto & aSeed : aVectSeed)  // parse all seed
    {
        // avoid  Vertex not in graph  AND  vertex already reached by a previous tree
        if (aParam.InsideVertex(*aSeed) && (!aSeed->BitTo1(aBitReached)))
        {
           aForest.push_back({aSeed,{}});
        
           // compute tree with the seed
           Internal_MinimumSpanninTree 
           (
                 aForest.back(),
                 aGraph,*aSeed,aParam,
                 aBitReached,
                 aBitOutHeap,
                 false    // No clean, because marker are used to avoid duplicatio,
           );
        }
   }

   // do the cleaning  because it was not made by Internal_MinimumSpanninTree
   for (const auto & [aSeed,aVPair] :  aForest)
   {
        if (aSeed)
        {
            aSeed->SetBit0(aBitReached);
            aSeed->SetBit0(aBitOutHeap);
        }
        for (const auto & [aV,anE] : aVPair)
        {
            aV->SetBit0(aBitReached);
            aV->SetBit0(aBitOutHeap);
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
           if (aParam.InsideV1AndEdgeAndSucc(*aVCur,*anEdge))  // check they are in sub-graph
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

}; // MMVII
#endif  // _MMVII_Tpl_GraphAlgo_SPCC_H_




