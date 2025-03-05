#ifndef _MMVII_Tpl_Graph_SubGr_H_
#define _MMVII_Tpl_Graph_SubGr_H_

#include "MMVII_Tpl_GraphStruct.h"

namespace MMVII
{
/** \file   MMVII_Tpl_Graph_SubGraph.h
    \brief  Clases for specialization of cAlgo_ParamVG
*/

template <class TGraph>  class cAlgo_ParamVG
{
        public :
             typedef typename TGraph::tVertex  tVertex;
             typedef typename TGraph::tEdge    tEdge;

             virtual bool   InsideVertex(const  tVertex &) const {return true;}

             // take as parameter V1 & edge E=V1->V2 (because we cannot acces V1 from E)
             virtual bool   InsideEdge(const    tEdge &) const {return true;}
             
             // method frequently used, so that user only redefines InsideEdge
             inline bool   InsideV1AndEdgeAndSucc(const    tEdge & anEdge) const 
             {
                    return   this->InsideEdge(anEdge) 
                          && this->InsideVertex(anEdge.Succ()) 
                          && this->InsideVertex(anEdge.VertexInit());
             }

             virtual tREAL8 WeightEdge(const    tEdge &) const {return 1.0;}
};



/*
template <class TGraph>  class cAlgo_SingleSubGr ;  // singleton sub-graph
template <class TGraph>  class cAlgo_SubGrNone ;    // empty sub-braph

template <class TGraph>  class cSubGraphOfVertices; // sub graph of given set/vector of vertices
template <class TGraph>  class cSubGraphOfEdges ;   // sub graph of given set/vector of  edges
template <class TGraph>  class cAlgo_SubGrCostOver ; // sub graph of vertice having internal cost over a threshold
template <class TGraph,class TFunc>  class cTpl_WeithingSubGr ; // sub-graph of vertice computed with lambda/func 
// general sub-graphe parametrization with 3 func inside Vertex, inside edge, weight edge
template <class TGraph,class TFunc_V,class TFunc_E,class TFunc_W>  class cTpl_InsideAndWSubGr ;
*/

/**   define a sub-graph than contain a single vertex, used for example as a goal in shortest path */
template <class TGraph>  class cAlgo_SingleSubGr : public cAlgo_ParamVG<TGraph>
{
        public :
	     typedef typename TGraph::tVertex  tVertex;

             bool   InsideVertex(const  tVertex & aV) const override {return (mSingleV==&aV);}
	     cAlgo_SingleSubGr(const tVertex * aV) : mSingleV (aV) {}
        private :
	     const tVertex * mSingleV;
};

/**   define a sub-graph than contain nothing, used for example as a goal in minimal spanning tree */
template <class TGraph>  class cAlgo_SubGrNone : public cAlgo_ParamVG<TGraph>
{
        public :
	     typedef typename TGraph::tVertex  tVertex;

             bool   InsideVertex(const  tVertex &) const override {return false;}
        private :
};

/**  class for define a subgraph by a set of edges, use flag to have fast access to belonging */
template <class TGraph>  class cSubGraphOfVertices : public cAlgo_ParamVG<TGraph>
{
        public :
             typedef typename TGraph::tVertex  tVertex;
             typedef std::vector<tVertex*>     tVVPtr;

             cSubGraphOfVertices(TGraph&,const tVVPtr & aVV) ;
             ~cSubGraphOfVertices();

             bool   InsideVertex(const  tVertex & aV) const override {return aV.BitTo1(mFlagVertex);}
        private :
            TGraph&   mGr;
            tVVPtr    mVVertices;
            size_t    mFlagVertex;
};

template <class TGraph> cSubGraphOfVertices<TGraph>::cSubGraphOfVertices(TGraph& aGr,const tVVPtr & aVVertices)  :
    mGr         (aGr),
    mVVertices  (aVVertices),
    mFlagVertex (aGr.Vertex_AllocBitTemp())
{
    for (const auto& aV : mVVertices)
        aV->SetBit1(mFlagVertex);
}
template <class TGraph> cSubGraphOfVertices<TGraph>::~cSubGraphOfVertices() 
{
    for (const auto& aV : mVVertices)
        aV->SetBit0(mFlagVertex);
    mGr.Vertex_FreeBitTemp(mFlagVertex);
}

/*
template <class TGraph>  class cSubGraph_NegVertex : public cAlgo_ParamVG<TGraph>
{
    public :  
           cSubGraph_Negation(const cAlgo_ParamVG<TGraph> * aSubGrInt) :
                 mSubGrInit (aSubGrInt)
           {
           }

           bool   InsideVertex(const  tVertex & aV) const override {return ! mSubGrInit->InsideVertex(aV);}
    private  :  
           const cAlgo_ParamVG<TGraph> * mSubGrInit;
};
*/

template <class TGraph>  class cSubGraphOfNotVertices : public cAlgo_ParamVG<TGraph>
{
      public :
           typedef typename TGraph::tVertex  tVertex;
           cSubGraphOfNotVertices(TGraph& aGraph,const std::vector<tVertex*>  & aVV)  :  mSubGrInit (aGraph,aVV) {}
           bool   InsideVertex(const  tVertex & aV) const override {return ! mSubGrInit.InsideVertex(aV);}
      private :
           cSubGraphOfVertices<TGraph>  mSubGrInit;
};



/**  class for define a subgraph by a set of edges, use flag to have fast access to belonging */
template <class TGraph>  class cSubGraphOfEdges : public cAlgo_ParamVG<TGraph>
{
        public :
             typedef typename TGraph::tVertex tVertex;
             typedef typename TGraph::tEdge   tEdge;
             typedef std::vector<tEdge*>      tVEPtr;

             cSubGraphOfEdges(TGraph&,const tVEPtr & aVEdge) ;
             ~cSubGraphOfEdges();

             bool   InsideVertex(const  tVertex & aV) const override {return aV.BitTo1(mFlagVertex);}
             bool   InsideEdge(const    tEdge & anE) const override {return anE.EdgeInitOr()->BitTo1(mFlagEdge);}
        private :
            TGraph&   mGr;
            tVEPtr    mVEdges;
            size_t    mFlagVertex;
            size_t    mFlagEdge;
};

template <class TGraph> cSubGraphOfEdges<TGraph>::cSubGraphOfEdges(TGraph& aGr,const tVEPtr & aVEdge) :
    mGr         (aGr),
    mVEdges     (aVEdge),
    mFlagVertex (aGr.Vertex_AllocBitTemp()),
    mFlagEdge   (aGr.Edge_AllocBitTemp())
{
    for (const auto& anE : mVEdges)
    {
         anE->EdgeInitOr()->SetBit1(mFlagEdge);
         anE->Succ().SetBit1(mFlagVertex);
         anE->VertexInit().SetBit1(mFlagVertex);
    }
}

template <class TGraph> cSubGraphOfEdges<TGraph>::~cSubGraphOfEdges()
{
    for (const auto& anE : mVEdges)
    {
         anE->EdgeInitOr()->SetBit0(mFlagEdge);
         anE->Succ().SetBit0(mFlagVertex);
         anE->VertexInit().SetBit0(mFlagVertex);
    }
    mGr.Vertex_FreeBitTemp(mFlagVertex);
    mGr.Edge_FreeBitTemp(mFlagEdge);
}


/**   define a sub-graph that contain as goal the vertex having cost over a threshold, use to
 have it as goal, to compute the sorhtest path tree up to a distance */

template <class TGraph>  class cAlgo_SubGrCostOver : public cAlgo_ParamVG<TGraph>
{
        public :
	     typedef typename TGraph::tVertex  tVertex;

             cAlgo_SubGrCostOver(tREAL8 aThr) : mThreshold (aThr) {}
             bool   InsideVertex(const  tVertex & aV) const override {return aV.AlgoCost() > mThreshold;}
             
        private :
             tREAL8 mThreshold;
};


/** Class & function for using func or lambda to weight edges */

template <class TGraph,class TFunc>  class cTpl_WeithingSubGr : public cAlgo_ParamVG<TGraph>
{
      public :
          typedef typename TGraph::tEdge tEdge;
          cTpl_WeithingSubGr(const TFunc & aFunc) : mFunc (aFunc) {}
          tREAL8 WeightEdge(const    tEdge & anEdge) const override {return mFunc(anEdge);}

          TFunc  mFunc;
};
template <class TGraph,class TFunc> cTpl_WeithingSubGr<TGraph,TFunc> Tpl_WeithingSubGr(const TGraph *,const TFunc & aFunct)
{
   return cTpl_WeithingSubGr<TGraph,TFunc>(aFunct);
}

/** Class & function for using func or lambda to define insideness of vertices & edges */

template <class TGraph,class TFunc_V,class TFunc_E,class TFunc_W>  class cTpl_InsideAndWSubGr : public cAlgo_ParamVG<TGraph>
{
      public :
          typedef typename TGraph::tVertex tVertex;
          typedef typename TGraph::tEdge   tEdge;
          cTpl_InsideAndWSubGr
          (
              const TFunc_V & aFunc_V,
              const TFunc_E & aFunc_E ,
              const TFunc_W & aFunc_W 
          ) :
             mFunc_V (aFunc_V),
             mFunc_E (aFunc_E),
             mFunc_W (aFunc_W) 
          {
          }

          bool   InsideVertex(const  tVertex & aV) const override {return mFunc_V(aV);}
          bool   InsideEdge  (const  tEdge & anE)  const override {return mFunc_E(anE);}
          tREAL8 WeightEdge  (const  tEdge & anE)  const override {return mFunc_W(anE);}

          TFunc_V  mFunc_V;
          TFunc_E  mFunc_E;
          TFunc_W  mFunc_W;
};

template <class TGraph,class TFunc_V,class TFunc_E,class TFunc_W> 
      cTpl_InsideAndWSubGr<TGraph,TFunc_V,TFunc_E,TFunc_W> 
           Tpl_InsideAndWSubGr
           (
	         const TGraph *,
                 const TFunc_V & aFunc_V,
                 const TFunc_E & aFunc_E ,
                 const TFunc_W & aFunc_W 
           )
{
   return cTpl_InsideAndWSubGr<TGraph,TFunc_V,TFunc_E,TFunc_W>(aFunc_V,aFunc_E,aFunc_W);
}

/* ****************************************************************** */
/*                                                                    */
/*           BOOLEAN OPERATION ON VERTICES/EDGES                      */
/*                                                                    */
/* ****************************************************************** */

     //   filtering operation on edges / vertices with sub-graph
     //   "Fast boolean" operation on vertices/edges usin              

template <class TGraph>  class cVG_OpBool
{
     public :
          typedef typename TGraph::tVertex tVertex;
          typedef typename TGraph::tEdge   tEdge;
          typedef cAlgo_ParamVG<TGraph>    tParamA;
          typedef std::vector<tVertex*>    tVVertices;
          typedef std::vector<tEdge*>      tVEdges;


          static void FilterEdges(tVEdges & aVOut, const tVEdges &  aVIn,const tParamA & aParam)
          {
               aVOut.clear();
               for (const auto & anEIn : aVIn)
                   if (aParam.InsideV1AndEdgeAndSucc(*anEIn))
                      aVOut.push_back(anEIn);
          }

          static void EdgesMinusVertices(tVEdges & aVOut, const tVEdges &  aVIn,const tVVertices & aVV)
          {
                if (aVV.empty())
                   aVOut = aVIn;
                else
                {
                    cSubGraphOfNotVertices<TGraph> aSubGr(aVV.at(0)->Graph(),aVV);
                    FilterEdges(aVOut,aVIn,aSubGr);
                }
          }
};


}; // MMVII
#endif  // _MMVII_Tpl_Graph_SubGr_H_




