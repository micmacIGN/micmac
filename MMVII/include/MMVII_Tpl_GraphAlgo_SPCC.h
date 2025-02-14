#ifndef _MMVII_Tpl_GraphAlgo_SPCC_H_
#define _MMVII_Tpl_GraphAlgo_SPCC_H_

#include "MMVII_TplHeap.h"
#include "MMVII_Tpl_GraphStruct.h"

namespace MMVII
{

template <class TGraph>  class cAlgo_SubGr
{
        public :
	     typedef typename TGraph::tVertex  tVertex;
	     typedef typename TGraph::tEdge    tEdge;

             virtual bool   InsideVertex(const  tVertex &) const {return true;}
};

template <class TGraph>  class cAlgo_SingleSubGr : public cAlgo_SubGr<TGraph>
{
        public :
	     typedef typename TGraph::tVertex  tVertex;

             bool   InsideVertex(const  tVertex & aV) const override {return (mSingleV==&aV);}
	     cAlgo_SingleSubGr(const tVertex * aV) : mSingleV (aV) {}
        private :
	     const tVertex * mSingleV;
};

template <class TGraph>  class cAlgo_SubGrNone : public cAlgo_SubGr<TGraph>
{
        public :
	     typedef typename TGraph::tVertex  tVertex;

             bool   InsideVertex(const  tVertex &) const override {return false;}
        private :
};



template <class TGraph>  class cAlgo_ParamVG : public cAlgo_SubGr<TGraph>
{
        public :
	     typedef typename TGraph::tVertex  tVertex;
	     typedef typename TGraph::tEdge    tEdge;

             virtual bool   InsideEdge(const tVertex &,const    tEdge &) const {return true;}
             virtual tREAL8 WeightEdge(const tVertex &,const    tEdge &) const {return 1.0;}

             inline bool   InsideV1AndEdgeAndSucc(const tVertex & aV1,const    tEdge & anEdge) const 
             {
                    return InsideEdge(aV1,anEdge) && this->InsideVertex(anEdge.Succ()) && this->InsideVertex(aV1);
             }
};

/* ********************************************************************************* */
/*                                                                                   */
/*                                   cAlgoSP                                         */
/*                                                                                   */
/* ********************************************************************************* */

template <class TGraph>   class cAlgoSP
{
     public :
          typedef  cAlgo_ParamVG<TGraph>          tParamA;
	  typedef typename TGraph::tVertex        tVertex;
	  typedef typename TGraph::tEdge          tEdge;
          typedef std::vector<tVertex*>           tVectVertex;
	  typedef std::pair<tVertex*,tEdge*>      tPairVE;
	  typedef std::vector<tPairVE>            tSetPairVE;  //One tree
	  typedef std::pair<tVertex*,tSetPairVE>  t1ElemForest; // One elem of forest
	  typedef std::list<t1ElemForest>         tForest;
          // typedef std::vector<tVertex*,tSetPairVE>  tPVertSPVE;

          class cHeapCmp
          {
                 public :
                    typedef tVertex *         tVertexPtr ;
                    bool  operator () (const tVertexPtr & aV1,const tVertexPtr & aV2) 
                    {
                       return aV1->AlgoCost() < aV2->AlgoCost();
                    }
          };
          class cHeapParam
          {
              public :
                 typedef tVertex *         tVertexPtr ;
                 static void SetIndex(const tVertexPtr & aPtrV,tINT4 i) { aPtrV->AlgoIndexHeap() = i;} 
                 static int  GetIndex(const tVertexPtr & aPtrV)  { return aPtrV->AlgoIndexHeap(); }
          };

          static tVertex * MakeShortestPathGen
                 (
                    TGraph &                       aGraph,
                    bool                           aModeMinPCC,
                    const tVectVertex &            aVSeeds,
                    const tParamA  &               aParam,
                    const cAlgo_SubGr<TGraph>   &  aGoal
                 );

          static tVertex * ShortestPath_A2B
                 (
                    TGraph &          aGraph,
                    tVertex &         aBegin,
                    tVertex &         aEnd,
                    const tParamA  &  aParam
                 );

           static tSetPairVE  MinimumSpanninTree (TGraph & aGraph, tVertex &  aSeed, const tParamA &);
           static void  MinimumSpanninTree (tSetPairVE&,TGraph & aGraph, tVertex &  aSeed, const tParamA &);

           static void MinimumSpanningForest(tForest&,TGraph &,const tVectVertex&  aSeed, const tParamA &);

          private :
              static tVertex * Internal_MakeShortestPathGen
                 (
                    TGraph &                       aGraph,
                    bool                           aModeMinPCC,
                    const tVectVertex &            aVSeeds,
                    const cAlgo_ParamVG<TGraph> &  aParam,
                    const cAlgo_SubGr<TGraph>   &  aGoal,
                    tVectVertex &                  aVReached,
                    size_t                         aBitReached,
                    size_t                         aBitOutHeap,
                    bool                           CleanAnFreeBits
                 );

              static void  Internal_MinimumSpanninTree 
                    (
                          tSetPairVE&,
                          TGraph & aGraph, 
                          tVertex &  aSeed, 
                          const cAlgo_ParamVG<TGraph> &  aParam,
                          size_t                         aBitReached,
                          size_t                         aBitOutHeap,
                          bool                           CleanAnFreeBits
                    ); 
};


           /*   Generic method */

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
    tVertex * aResult = nullptr;


    cHeapCmp aCompare;
    cIndexedHeap<tVertex*,cHeapCmp,cHeapParam> aHeap(aCompare);
 
    for (auto aPtrV : aVSeeds)
    {
        if (aParam.InsideVertex(*aPtrV))
        {
           aPtrV->SetBit1(aBitReached);
           aPtrV->mAlgoCost = 0.0;
           aPtrV->mAlgoFather = nullptr;
           aVReached.push_back(aPtrV);
           aHeap.Push(aPtrV);
        }
    }

    bool  GoOn= true;
    while (GoOn)
    {
        if (aHeap.IsEmpty())
        {
            GoOn = false;
        }
        else
        {
            tVertex * aNewVOut =  aHeap.PopVal(nullptr);
            if (aGoal.InsideVertex(*aNewVOut))
            {
               GoOn = false;
               aResult = aNewVOut;
            }
            else
            {
                 aNewVOut->SetBit1(aBitOutHeap);
                 for (const auto &  anEdge :  aNewVOut->EdgesSucc())
                 {
                      if (aParam.InsideV1AndEdgeAndSucc(*aNewVOut,*anEdge))
                      {
                         tVertex & aNewVIn = anEdge->Succ();

                         if (!aNewVIn.BitTo1(aBitOutHeap))
                         {
                            tREAL8 aNewCost =  aParam.WeightEdge(*aNewVOut,*anEdge);
                            if (aModeMinPCC)  
                               aNewCost += aNewVOut->mAlgoCost;

                            if (!aNewVIn.BitTo1(aBitReached))
                            {
                               aVReached.push_back(&aNewVIn);
                               aNewVIn.SetBit1(aBitReached);
                               aNewVIn.mAlgoCost = aNewCost+1;
                               aHeap.Push(&aNewVIn);
                            }

                            if (aNewCost<aNewVIn.mAlgoCost)
                            {
                                 aNewVIn.mAlgoCost = aNewCost;
                                 aHeap.UpDate(&aNewVIn);
                                 aNewVIn.mAlgoFather = aNewVOut;
                            }
                         }
                      }
                 }
            }
        }
    }

    if (CleanAnFreeBits)
    {
        tVertex::SetBit0(aVReached,aBitReached);
        tVertex::SetBit0(aVReached,aBitOutHeap);

        aGraph.FreeBitTemp(aBitReached);
        aGraph.FreeBitTemp(aBitOutHeap);
    }

    return aResult;
}  

   
           /*   Some interfaces for shortest path */

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


template <class TGraph>   
    typename TGraph::tVertex * cAlgoSP<TGraph>::ShortestPath_A2B
                 (
                    TGraph &                       aGraph,
                    tVertex &  aBegin,
                    tVertex &  aEnd,
                    const cAlgo_ParamVG<TGraph> &  aParam
                 )
{
   // TGraph & aGraph = aBegin.Graph();
   // TGraph & aGraph = *aBegin.mGr;
   // TGraph * aPtrGraph = aBegin.mGr;
   // TGraph &  aGraph = * aPtrGraph;
   return
      MakeShortestPathGen
      (
            //aBegin.Graph(),
            aGraph,
            true,
            {&aBegin},
            aParam,
            cAlgo_SingleSubGr<TGraph>(&aEnd)
      );
}

           /*   Some interfaces for minimum spaning tree */

template <class TGraph>   
     void cAlgoSP<TGraph>::Internal_MinimumSpanninTree 
           (
                          tSetPairVE& aSetPair,
                          TGraph & aGraph, 
                          tVertex &  aSeed, 
                          const cAlgo_ParamVG<TGraph> &  aParam,
                          size_t                         aBitReached,
                          size_t                         aBitOutHeap,
                          bool                           CleanAnFreeBits
           )
{
    std::vector<tVertex*>   aVReached;
    tVertex * aResult  = Internal_MakeShortestPathGen
                         (
                            aGraph,
                            false,  // Not Shortest Path <=> Min Span Tree
                            {&aSeed},
                            aParam,
                            cAlgo_SubGrNone<TGraph>(),
                            aVReached,
                            aBitReached,
                            aBitOutHeap,
                            CleanAnFreeBits
                         );
    MMVII_INTERNAL_ASSERT_always(aResult==nullptr,"Internal_MinimumSpanninTree result!=0");

    aSetPair.clear();
    for (const auto aPtrV : aVReached)
    {
        if (aPtrV->mAlgoFather != nullptr)
           aSetPair.push_back(tPairVE(aPtrV,aPtrV->EdgeOfSucc(*(aPtrV->mAlgoFather))));
    }
}

template <class TGraph> 
     void cAlgoSP<TGraph>::MinimumSpanninTree
          (
                tSetPairVE& aSetPair,
                TGraph & aGraph, 
                tVertex &  aSeed, 
                const cAlgo_ParamVG<TGraph> &  aParam
          )
{
    size_t aBitReached = aGraph.AllocBitTemp();
    size_t aBitOutHeap  = aGraph.AllocBitTemp();

     Internal_MinimumSpanninTree 
     (
          aSetPair,aGraph,aSeed,aParam,
          aBitReached,
          aBitOutHeap,
          true
     );
}

template <class TGraph> 
     typename  cAlgoSP<TGraph>::tSetPairVE cAlgoSP<TGraph>::MinimumSpanninTree
          (
                TGraph & aGraph, 
                tVertex &  aSeed, 
                const cAlgo_ParamVG<TGraph> &  aParam
          )
{
       tSetPairVE aSetPair;
       MinimumSpanninTree(aSetPair,aGraph,aSeed,aParam);

       return aSetPair;
}

template <class TGraph> 
     void cAlgoSP<TGraph>::MinimumSpanningForest
          (
              tForest& aForest,
              TGraph & aGraph,
              const tVectVertex&  aVectSeed, 
              const tParamA & aParam
          )
{
    aForest.clear();
    size_t aBitReached = aGraph.AllocBitTemp();
    size_t aBitOutHeap  = aGraph.AllocBitTemp();


    for (const auto & aSeed : aVectSeed)
    {
        if (aParam.InsideVertex(*aSeed) && (!aSeed->BitTo1(aBitReached)))
        {
           aForest.push_back({aSeed,{}});
           t1ElemForest a1EF(aSeed,{}); 

           Internal_MinimumSpanninTree 
           (
                 aForest.back().second,
                 aGraph,*aSeed,aParam,
                 aBitReached,
                 aBitOutHeap,
                 false
           );
        }
   }

   for (const auto & [aSeed,aVPair] :  aForest)
   {
        aSeed->SetBit0(aBitReached);
        aSeed->SetBit0(aBitOutHeap);
        for (const auto & [aV,anE] : aVPair)
        {
            aV->SetBit0(aBitReached);
            aV->SetBit0(aBitOutHeap);
            anE->Succ().SetBit0(aBitReached);
            anE->Succ().SetBit0(aBitOutHeap);
        }
   }

   aGraph.FreeBitTemp(aBitReached);
   aGraph.FreeBitTemp(aBitOutHeap);
}

                  

/* ********************************************************************************* */
/*                                                                                   */
/*                                   cAlgoCC                                         */
/*                                                                                   */
/* ********************************************************************************* */

template <class TGraph>   class cAlgoCC
{
     public :
	  typedef typename TGraph::tVertex  tVertex;
	  static std::vector<tVertex *>  ConnectedComponent
		                  (
                                      TGraph & aGraph,
				      tVertex& aSeed,
                                      const cAlgo_ParamVG<TGraph> & aParam
				  );

	  static std::list<std::vector<tVertex *>>  Multiple_ConnectedComponent
		                  (
                                      TGraph & aGraph,
				      const std::vector<tVertex*> & aSeed,
                                      const cAlgo_ParamVG<TGraph> & aParam
				  );

	  static std::list<std::vector<tVertex *>>  All_ConnectedComponent
		                  (
                                      TGraph & aGraph,
                                      const cAlgo_ParamVG<TGraph> & aParam
				  );

     private :
	  static void  Internal_ConnectedComponent
		                  (
				      tVertex* aSeed,
				      std::vector<tVertex *>& aResult,
                                      const cAlgo_ParamVG<TGraph> & aParam,
				      size_t aFlag
				  );
	  static void  Internal_Multiple_ConnectedComponent
		                  (
                                      TGraph & aGraph,
                                      std::list<std::vector<tVertex *>>  &,
				      const std::vector<tVertex*> & aSeed,
                                      const cAlgo_ParamVG<TGraph> & aParam
				  );

};


template <class TGraph> 
     void cAlgoCC<TGraph>::Internal_ConnectedComponent
          (
              tVertex* aSeed,
              std::vector<tVertex *>& aResult,
              const cAlgo_ParamVG<TGraph> & aParam,
              size_t aFlag
          )
{
    aResult.clear();
    if (aParam.InsideVertex(*aSeed) &&  (!aSeed->BitTo1(aFlag)))
    {
       aSeed->SetBit1(aFlag);
       aResult.push_back(aSeed);
    }

    size_t aIndBottom = 0;
    while (aIndBottom != aResult.size())
    {
       tVertex * aVCur =  aResult.at(aIndBottom);
       for (const auto &  anEdge :  aVCur->EdgesSucc())
       {
           if (aParam.InsideV1AndEdgeAndSucc(*aVCur,*anEdge))
           {
              tVertex & aVNext = anEdge->Succ();
	      if (!  aVNext.BitTo1(aFlag))
	      {
                  aVNext.SetBit1(aFlag);
                  aResult.push_back(&aVNext);
	      }
           }
       }
       aIndBottom++;
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
    size_t aBitReached = aGraph.AllocBitTemp();
    std::vector<tVertex*>   aResult;

    Internal_ConnectedComponent(&aSeed,aResult,aParam,aBitReached);

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
    size_t aBitReached = aGraph.AllocBitTemp();

    for (const auto & aSeed : aVecSeed)
    {
	aResult.push_back(std::vector<tVertex*>());
	Internal_ConnectedComponent(aSeed,aResult.back(),aParam,aBitReached);
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




