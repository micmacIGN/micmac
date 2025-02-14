#ifndef _MMVII_Tpl_GraphStruct_H_
#define _MMVII_Tpl_GraphStruct_H_  

#include "MMVII_memory.h"
#include "MMVII_SetITpl.h"
#include "MMVII_TplHeap.h"

namespace MMVII
{

//    declaration of templates classes for Valuated-Graph (VGà , parametrised by 
//
//    - TA_Vertex : attribute  of vertex
//    - TA_Oriented : attribute  of oriented edge ( A(S1->S2) !=  A(S2->S1) )
//    - TA_Sym : attribute  of non oriented edge  :  A(S1->S2) and A(S2->S1)  are shared
//

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  class cVG_Vertex;
template <class TA_Vertex,class TA_Oriented,class TA_Sym>  class cVG_Edge;
template <class TA_Vertex,class TA_Oriented,class TA_Sym>  class cVG_Graph;
template <class TGraph>  class cAlgo_ParamVG;
template <class TGraph> class cAlgoSP;
template <class TGraph> class cAlgoCC;


/**  In adjacency graph, the neighboor of a vertex S1 are represented by a set of oriented edge S1->S2 : cVG_Edge
     An edge contain :
        * the vertex S2 it is directing to
        * the oriented attribute  
        * the shared symetric attribute (in fact mNumAttr)
        * a boolean indicating if, between the 2 oriented edges, it correspond to the direction at creation,
          used for example in group-graph, where the group G value is shared, to know if :
                    S1->S2 must be interpreted as G or G-1 
*/

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  class cVG_Edge : public cMemCheck
{
     public :
          // ------------------- typedef & friendship --------------------------------
          typedef cVG_Vertex<TA_Vertex,TA_Oriented,TA_Sym>  tVertex;
          typedef cVG_Graph<TA_Vertex,TA_Oriented,TA_Sym>   tGraph;
          typedef cVG_Edge<TA_Vertex,TA_Oriented,TA_Sym>    tEdge;
          typedef cAlgo_ParamVG<tGraph>                     tParamAlgo;
          typedef cAlgoSP<tGraph>                           tAlgoSP;
          typedef cAlgoCC<tGraph>                           tAlgoCC;

          friend tAlgoSP;
          friend tVertex;
          friend tGraph;
          template <class> friend class cAlgoSP;
          template <class> friend class cAlgoCC;

          // ------------------- method, essentially accesor --------------------------------

	  inline       tVertex & Succ() ;         ///< Accesor to "S2"
	  inline const tVertex & Succ() const ;   ///< Accesor to "S2"

	  inline TA_Oriented & AttrOriented()  {return mAttrO;}  ///<  Accessor to oriented attribute
	  inline const TA_Oriented & AttrOriented() const {return mAttrO;}  ///<  Accessor to oriented attribute

	  inline       TA_Sym & AttrSym() ;  ///< Accessor to symetric attribute
	  inline const TA_Sym & AttrSym() const ;  ///< Accessor to symetric attribute

     private :
          cVG_Edge(const tEdge&) = delete;  ///< No copy for graph structures
          inline ~cVG_Edge();
          inline cVG_Edge(tGraph*,const TA_Oriented &,int aNumSucc,int aNumAttr,bool DirInit);
          

	  tGraph *        mGr;         ///<  Graph it's belongin to
	  TA_Oriented     mAttrO;      ///<  Oriented Attribute
	  int             mNumSucc;    ///<  Num of successor ~ to a pointer 
          size_t          mNumAttr;    ///<  Num of symetric attribute ~ to a pointer
          bool            mDirInit;    ///<  Is it the 2 of edges that correspond to initial direction
};


/**  Class for representing the vertex of a graph
*/

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  class cVG_Vertex : public cMemCheck
{
     public :
          // ------------------- typedef & friendship --------------------------------
	  typedef cVG_Vertex<TA_Vertex,TA_Oriented,TA_Sym>    tVertex;
	  typedef cVG_Edge<TA_Vertex,TA_Oriented,TA_Sym>      tEdge;
          typedef cVG_Graph<TA_Vertex,TA_Oriented,TA_Sym>     tGraph;
          typedef cAlgoSP<tGraph>                             tAlgoSP;
          typedef cAlgoCC<tGraph>                             tAlgoCC;

          friend  tEdge;
          friend  tGraph;
          friend  tAlgoSP;
          template <class> friend class cAlgoSP;
          template <class> friend class cAlgoCC;

          // ---------------  Manipulating Attribute of vertex ------------------------------
          inline void              SetAttr(const TA_Vertex & anAttr)  {mAttr = anAttr;}
          inline TA_Vertex &       Attr()       {return mAttr;}
          inline const TA_Vertex & Attr() const {return mAttr;}

          // ---------------  Manipulating adjacent edges ------------- ------------------------------

                 //  For S2, return edge E such E=S1->S2, 0 if does not exist (error if not SVP)
          const tEdge * EdgeOfSucc(const tVertex  & aS2,bool SVP=false) const {return const_cast<tVertex*>(this)->EdgeOfSucc(aS2,SVP);}
          inline tEdge * EdgeOfSucc(const tVertex  & aS2,bool SVP=false) ;

          inline const  std::vector<tEdge*> & EdgesSucc()       {return mVEdges;}
          inline const  std::vector<tEdge*> & EdgesSucc() const {return mVEdges;}

          inline tREAL8  AlgoCost() const {return  mAlgoCost;}
          inline int &   AlgoIndexHeap() {return  mAlgoIndexHeap;}


	  inline void BackTrackFathersPath(std::vector<tVertex*> &) ;
	  inline std::vector<tVertex*>  BackTrackFathersPath();

          inline void SetBit1(size_t aBit) {mAlgoTmpMark.AddElem(aBit);}
          inline void SetBit0(size_t aBit) {mAlgoTmpMark.SuprElem(aBit);}
          inline bool BitTo1(size_t aBit) const {return mAlgoTmpMark.IsInside(aBit);}
          static inline void SetBit0(const std::vector<tVertex*> & aVV,size_t aBit) ;


          tGraph & Graph()              {return *mGr;}
          const tGraph & Graph() const  {return *mGr;}

     private :
          cVG_Vertex(const tVertex &) = delete;
          inline ~cVG_Vertex();
          inline cVG_Vertex(tGraph * aGr,const TA_Vertex & anAttr,int aNum) ;

          tGraph *                   mGr;             
          TA_Vertex                  mAttr;
          std::vector<tEdge*>        mVEdges;
          int                        mNum;

          cSetISingleFixed<tU_INT4>  mAlgoTmpMark;
          tVertex*                   mAlgoFather;
          int                        mAlgoIndexHeap;
          tREAL8                     mAlgoCost;
};


template <class TA_Vertex,class TA_Oriented,class TA_Sym>  class cVG_Graph : public cMemCheck
{
     public :
	  typedef cVG_Vertex<TA_Vertex,TA_Oriented,TA_Sym>        tVertex;
	  typedef cVG_Edge<TA_Vertex,TA_Oriented,TA_Sym>          tEdge;
	  typedef cVG_Graph<TA_Vertex,TA_Oriented,TA_Sym>         tGraph;
          typedef cAlgoSP<tGraph>                                 tAlgoSP;
          typedef cAlgoCC<tGraph>                                 tAlgoCC;
	     
	  friend tVertex;
	  friend tEdge;
          template <class> friend class cAlgoSP;
          template <class> friend class cAlgoCC;

          size_t NbVertex()   const {return mV_Vertices.size();}
	  tVertex &        VertexOfNum(size_t aNum)       {return *mV_Vertices.at(aNum);}
	  const tVertex &  VertexOfNum(size_t aNum) const {return *mV_Vertices.at(aNum);}

	  const std::vector<tVertex*>  &    AllVertices() const {return mV_Vertices;}
	  const std::vector<TA_Sym*>   &    AllAttrSym() const  {return mV_AttrSym;}

          inline cVG_Graph();
          inline ~cVG_Graph();

          inline tVertex * NewSom(const TA_Vertex & anAttr) ;

          inline void AddEdge(tVertex & aV1,tVertex & aV2,const TA_Oriented &A12,const TA_Oriented &A21,const TA_Sym &,bool OkExist=false);
          void AddEdge(tVertex & aV1,tVertex & aV2,const TA_Oriented &aAOr,const TA_Sym & aASym,bool OkExist) 
               {AddEdge(aV1,aV2,aAOr,aAOr,aASym,OkExist);}

          inline size_t AllocBitTemp();
          inline void   FreeBitTemp(size_t);

     protected :
	  const TA_Sym &  AttrSymOfNum(size_t aNum) const {return *mV_AttrSym.at(aNum);}
	  TA_Sym &        AttrSymOfNum(size_t aNum)       {return *mV_AttrSym.at(aNum);}

     private :
          cVG_Graph(const tGraph &) = delete;
	  std::vector<tVertex*>       mV_Vertices;
	  std::vector<TA_Sym*>        mV_AttrSym;
          cSetISingleFixed<tU_INT4>   mBitsAllocaTed;
};



/* ********************************************* */
/*                                               */
/*                 cVG_Edge                      */
/*                                               */
/* ********************************************* */

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  
    cVG_Edge<TA_Vertex,TA_Oriented,TA_Sym>::~cVG_Edge()
{
}

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  
  cVG_Edge<TA_Vertex,TA_Oriented,TA_Sym>::cVG_Edge
  (
         tGraph* aGr,
         const TA_Oriented & anAttrO,
         int aNumSucc,
         int aNumAttr,
         bool DirInit
   ) :
     mGr       (aGr),    
     mAttrO    (anAttrO),
     mNumSucc  (aNumSucc),
     mNumAttr  (aNumAttr),
     mDirInit  (DirInit)
{
}

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  
   const cVG_Vertex<TA_Vertex,TA_Oriented,TA_Sym> &  cVG_Edge<TA_Vertex,TA_Oriented,TA_Sym>::Succ()  const 
{
	return  mGr->VertexOfNum(mNumSucc);
}

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  
   cVG_Vertex<TA_Vertex,TA_Oriented,TA_Sym> &  cVG_Edge<TA_Vertex,TA_Oriented,TA_Sym>::Succ() 
{
	return  mGr->VertexOfNum(mNumSucc);
}

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  
   TA_Sym &  cVG_Edge<TA_Vertex,TA_Oriented,TA_Sym>::AttrSym() 
{
   return mGr->AttrSymOfNum(mNumAttr);
}

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  
   const TA_Sym &  cVG_Edge<TA_Vertex,TA_Oriented,TA_Sym>::AttrSym()  const
{
   return mGr->AttrSymOfNum(mNumAttr);
}

/* ********************************************* */
/*                                               */
/*                 cVG_Vertex                    */
/*                                               */
/* ********************************************* */

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  
  cVG_Edge<TA_Vertex,TA_Oriented,TA_Sym> * 
            cVG_Vertex<TA_Vertex,TA_Oriented,TA_Sym>::EdgeOfSucc(const tVertex  & aS2,bool SVP) 
{
     for (auto & anEdge : mVEdges)
         if (anEdge->mNumSucc==aS2.mNum)
            return  anEdge;

      MMVII_INTERNAL_ASSERT_tiny(SVP,"No EdgeOfSucc");
      return nullptr;
}

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  
            cVG_Vertex<TA_Vertex,TA_Oriented,TA_Sym>::~cVG_Vertex()
{
    for (auto & anE :  mVEdges)
        delete anE;
}
template <class TA_Vertex,class TA_Oriented,class TA_Sym>
   void cVG_Vertex<TA_Vertex,TA_Oriented,TA_Sym>::BackTrackFathersPath(std::vector<tVertex*> & aPath) 
{
     aPath.clear();
     tVertex * aV = this;
     while (aV!=nullptr)
     {
        aPath.push_back(aV);
	aV = aV->mAlgoFather;
     }
}
template <class TA_Vertex,class TA_Oriented,class TA_Sym>
   std::vector<cVG_Vertex<TA_Vertex,TA_Oriented,TA_Sym> *>  cVG_Vertex<TA_Vertex,TA_Oriented,TA_Sym>::BackTrackFathersPath()
{
    std::vector<tVertex*> aPath;
    BackTrackFathersPath(aPath);
    return aPath;
}

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  
  cVG_Vertex<TA_Vertex,TA_Oriented,TA_Sym>::cVG_Vertex(tGraph * aGr,const TA_Vertex & anAttr,int aNum) :
      mGr            (aGr),
      mAttr          (anAttr),
      mNum           (aNum),
      mAlgoTmpMark   (0),
      mAlgoFather    (nullptr),
      mAlgoIndexHeap (HEAP_NO_INDEX),
      mAlgoCost      ( 1e7*std::cos(220+aNum*337.88))
{
}

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  
   void cVG_Vertex<TA_Vertex,TA_Oriented,TA_Sym>::SetBit0(const std::vector<tVertex*> & aVV,size_t aBit) 
{
   for (const auto & aV : aVV)
       aV->SetBit0(aBit);
}


/* ********************************************* */
/*                                               */
/*                 cVG_Graph                     */
/*                                               */
/* ********************************************* */

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  
   cVG_Graph<TA_Vertex,TA_Oriented,TA_Sym>::cVG_Graph()   :
        mBitsAllocaTed (0)
{
}


template <class TA_Vertex,class TA_Oriented,class TA_Sym>  
   cVG_Graph<TA_Vertex,TA_Oriented,TA_Sym>::~cVG_Graph()
{
    for (auto & aPtrV : mV_Vertices)
       delete aPtrV;
    for (auto & aPtrA : mV_AttrSym)
       delete aPtrA;
}

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  
   cVG_Vertex<TA_Vertex,TA_Oriented,TA_Sym>* cVG_Graph<TA_Vertex,TA_Oriented,TA_Sym>::NewSom(const TA_Vertex & anAttr) 
{
    mV_Vertices.push_back(new tVertex(this,anAttr,mV_Vertices.size()));
    return mV_Vertices.back();
}

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  
  void cVG_Graph<TA_Vertex,TA_Oriented,TA_Sym>::AddEdge
      (
           tVertex & aV1,
           tVertex & aV2,
           const TA_Oriented &A12,
           const TA_Oriented &A21,
           const TA_Sym & aASym,
           bool  OkExist
       )
{
    // Not sure can handle this case, at least would require some precaution (duplication ...)
    MMVII_INTERNAL_ASSERT_tiny((aV1.mNum!=aV2.mNum) ,"Reflexive edge in Add edge");
    tEdge * anE12 = aV1.EdgeOfSucc(aV2,SVP::Yes);

    if (anE12)
    {
        //tEdge * anE21 = aV2.EdgeOfSucc(aV1,SVP::Yes);
        // to do later, if we accept, we must supress eddges in vertices
        // + handle dir init 
        // probably do it after implementing remove edge
        MMVII_INTERNAL_ASSERT_tiny(false,"Ok exist in AddEdge");
/*
        delete anE12;
        delete anE21;
        aNumAttr = anE12->mNumAttr;
        delete 
*/
    }
    else 
    {
         aV1.mVEdges.push_back(new tEdge(this,A12,aV2.mNum,mV_AttrSym.size(),true));
         aV2.mVEdges.push_back(new tEdge(this,A21,aV1.mNum,mV_AttrSym.size(),false));

         mV_AttrSym.push_back(new TA_Sym(aASym));
    }
}
template <class TA_Vertex,class TA_Oriented,class TA_Sym>  
    size_t  cVG_Graph<TA_Vertex,TA_Oriented,TA_Sym>::AllocBitTemp()
{
    for (size_t aBit=0 ; aBit<32 ; aBit++)
    {
          if (!mBitsAllocaTed.IsInside(aBit))
          {
               mBitsAllocaTed.AddElem(aBit);
               return aBit;
          }
    }
    MMVII_INTERNAL_ERROR("No more bits in AllocBitTemp (forgot to free ?)");
    return 0;
}

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  
    void  cVG_Graph<TA_Vertex,TA_Oriented,TA_Sym>::FreeBitTemp(size_t aBit)
{
    
    mBitsAllocaTed.SuprElem(aBit);
}

};
#endif // _MMVII_Tpl_GraphStruct_H_  



