#ifndef _MMVII_Tpl_GraphAlgo_EnumCycle_H_
#define _MMVII_Tpl_GraphAlgo_EnumCycle_H_

#include "MMVII_Tpl_GraphStruct.h"



namespace MMVII
{


/************************************************************/
/*                                                          */
/*                   cAlgoEnumCycle                         */
/*                                                          */
/************************************************************/

template <class TGraph> class cActionOnCycle;
template <class TGraph> class cAlgoEnumCycle;


/**  Mother class for "call back" used by cAlgoEnumCycle when a cycle is detected */
template <class TGraph>   class cActionOnCycle
{
      public :
          typedef typename TGraph::tVertex        tVertex;
          typedef cAlgoEnumCycle<TGraph>          tAlgoEnum;

          virtual void OnCycle(const tAlgoEnum&) = 0;  ///< virtual methode called on cycle detection
};

/** Class for enumerating all the cycles with lengh < to given threshold */

template <class TGraph>   class cAlgoEnumCycle
{
     public :
          //---------------- typedef  section ---------------------
          typedef  cAlgoSP<TGraph>                tAlgoSP;
          typedef  cAlgo_ParamVG<TGraph>          tSubGr;
          typedef  cActionOnCycle<TGraph>         tActionC;
          typedef typename TGraph::tVertex        tVertex;
          typedef typename TGraph::tEdge          tEdge;
          typedef std::vector<tVertex*>           tVectVertex;

          /// Constr : Graph + Action On Cycle + Sub Graph it is used + Size of cycle
	  cAlgoEnumCycle(TGraph &,tActionC&,const tSubGr &,size_t aSz);
          /// destructor : free the bits allocated
	  ~cAlgoEnumCycle();

          /// Explorate all the cycles of the grah and calls the call back on each
          void ExplorateAllCycles();
          /// Explorate the cycle going through  a given edges
          void ExplorateCyclesOneEdge(tEdge & anE);

          /// Accesor to the last computed path
          const std::vector<tEdge*> &  CurPath() const {return mCurPath;}

          size_t  MaxSzCycle() const {return  mMaxSzCycle;}
	  TGraph &   Graph() {return mGraph;}

     private :
          /// Internal method for 1 edge, WithMarker=>do we mark the edge at the end for not going several time
          void ExplorateCyclesOneEdge(tEdge & anE,bool WithMarker);

          /// recursive method for exploring the cycle of 1 edge
          void RecExplCycles1Edge();

          /// Edge is valide id In graph, and not marked as already done
          bool  OkEdge(const tEdge & anE) const;

          
	  TGraph &             mGraph;  ///< The grap we are working on
	  tActionC *           mAction; ///< object for call back once we have a cycle
	  const tSubGr &       mSubGr;

	  size_t               mBitEgdeExplored;  ///< marker for edge already explored in ExplorateAllCycles
	  size_t               mBitEgdeInCurPath; ///< marker to assure that edge cannot be multiple in a path
	  size_t               mBitVertexCurSphere; ///< marker to know if distance 2 origin was computed

          tAlgoSP              mAlgoSP;     ///< Alorithm for shortest path computation
          size_t               mMaxSzCycle; ///< max size of cycle  we want to exlporate
          std::vector<tEdge*>  mCurPath;    ///< store the current computed sorthest path
          tVertex *            mFirstSom;   ///< begin of path
          
};

template <class TGraph>  
     cAlgoEnumCycle<TGraph>::cAlgoEnumCycle
     (
           TGraph &          aGraph,
	   tActionC&         anAction,
           const tSubGr &    aSubGr,
           size_t            aSz
     ) :
         mGraph              (aGraph),
         mAction             (&anAction),
         mSubGr              (aSubGr),
         mBitEgdeExplored    (mGraph.AllocBitTemp()),
         mBitEgdeInCurPath   (mGraph.AllocBitTemp()),
         mBitVertexCurSphere (mGraph.AllocBitTemp()),
         mMaxSzCycle         (aSz)
{
}

template <class TGraph>  cAlgoEnumCycle<TGraph>::~cAlgoEnumCycle()
{
   mGraph.FreeBitTemp(mBitEgdeExplored);
   mGraph.FreeBitTemp(mBitEgdeInCurPath);
   mGraph.FreeBitTemp(mBitVertexCurSphere);
}

template <class TGraph> bool  cAlgoEnumCycle<TGraph>::OkEdge(const tEdge & anE) const
{
    return      mSubGr.InsideV1AndEdgeAndSucc(anE)    // Is in sub-graph
	    &&  (! anE.SymBitTo1(mBitEgdeExplored))   // Has not already be explored in ExplorateAllCycles
	    &&  (! anE.SymBitTo1(mBitEgdeInCurPath))  // edge is not already in the path
    ;
}

template <class TGraph> void  cAlgoEnumCycle<TGraph>::ExplorateCyclesOneEdge(tEdge & anE)
{
  // Not sure if we must return quik or generate an error, exceptionnaly be the kind guy and accept it
  if (! OkEdge(anE)) return;

  ExplorateCyclesOneEdge(anE,false);
}

template <class TGraph> void  cAlgoEnumCycle<TGraph>::ExplorateCyclesOneEdge(tEdge & anE,bool WithMarker)
{
     //  ----------------[0]  Prepare the recursive exploration ----------------------

     // compute the distance to the first vertex of anE, to be able to avoid useless exploration (see @Rec:CutSphere)
     mAlgoSP.MakeShortestPathGen
     (
           mGraph,
           true,
           {&anE.VertexInit()},
           cAlgo_ParamVG<TGraph>(),
           cAlgo_SubGrCostOver<TGraph>(mMaxSzCycle/2)
     );

     mCurPath =  std::vector<tEdge*> {&anE};  // current path  begin with anE
     mFirstSom = &anE.VertexInit();           // to know when we have "close" the loop

     anE.SetBit1(mBitEgdeInCurPath);   // to avoid to go twice through  anE
     tVertex::SetBit1(mAlgoSP.VVertexOuted(),mBitVertexCurSphere); /// to know that distance was computed

     // ----------------[1]  recursive exploration itself ----------------------------------

     RecExplCycles1Edge();  

     //  ----------------[2]  "Clean" data after recursive exploration ----------------------

     anE.SetBit0(mBitEgdeInCurPath);  // clean bit "inside path"
     tVertex::SetBit0(mAlgoSP.VVertexOuted(),mBitVertexCurSphere); // clean bits in sphere

     // if required, mark anE so that edge will be avoides in all future exploration
     if (WithMarker)
        anE.SymSetBit1(mBitEgdeExplored);
}


template <class TGraph> void  cAlgoEnumCycle<TGraph>::RecExplCycles1Edge()
{

    tEdge &   aLastEdge  = *mCurPath.back();
    tVertex & aLastSom   = aLastEdge.Succ();


    // if this bit was not set, this mean we are at distance of  mFirstSom > to size of cycle, => no hope
    if (! aLastSom.BitTo1(mBitVertexCurSphere))
    {
       return;
    }

    // @Rec:CutSphere, now that we know that AlgoCost for aLastSom containt the distance to mFirstSom : we know that the cycle
    // begining by mCurPath and finishing to mFirstSom, wll have at least a length of "aLastSom.AlgoCost() +mCurPath.size()"
    if ( aLastSom.AlgoCost() +mCurPath.size() > mMaxSzCycle)
    {
       return;
    }

    if (&aLastSom==mFirstSom)  // we have close the loop !
    {
      // call the "call back" so that mAction can do anything it wants we the path
       mAction->OnCycle(*this); 
       // we return now if we dont accept loop going several time through begining, questionnable
       // return;
    }

    // obviously, nextpath will be too long ...
    if (mCurPath.size() >= mMaxSzCycle)
    {
        return;
    }

    // parse all the neighbours to explorate the pathes
    for (auto & anEdge : aLastSom.EdgesSucc())
    {
        if (OkEdge(*anEdge))  // is edge is in subgraph
	{
            //  prepare for recursive exploration
            anEdge->SymSetBit1(mBitEgdeInCurPath);
	    mCurPath.push_back(anEdge);

            // do the recursive exploration
	    RecExplCycles1Edge();

            //  restore after recursive exploration
	    mCurPath.pop_back();
            anEdge->SymSetBit0(mBitEgdeInCurPath);
	}
    }
    
}



template <class TGraph> void  cAlgoEnumCycle<TGraph>::ExplorateAllCycles()
{
    for (auto & aVPTr : mGraph.AllVertices())
    {
          if (mSubGr.InsideVertex(*aVPTr))
          {
               for (auto & anEdgePtr :  aVPTr->EdgesSucc())
               {
                   // we can restrict to IsDirInit
                   if (anEdgePtr->IsDirInit() && OkEdge(*anEdgePtr))
                   {
                         ExplorateCyclesOneEdge(*anEdgePtr,true);
                   }
               }
          }
    }
    // clean the mBitEgdeExplored
    for (const auto & anEdgePtr :  mGraph.AllEdges_DirInit())
         anEdgePtr->SymSetBit0(mBitEgdeExplored);

}


};
#endif //  _MMVII_Tpl_GraphAlgo_EnumCycle_H_

