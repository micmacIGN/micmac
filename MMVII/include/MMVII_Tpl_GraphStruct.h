#ifndef _MMVII_Tpl_GraphStruct_H_
#define _MMVII_Tpl_GraphStruct_H_  

#include "MMVII_memory.h"
#include "MMVII_SetITpl.h"
#include "MMVII_TplHeap.h"

namespace MMVII
{
/** \file   MMVII_Tpl_GraphStruct.h
    \brief   Define the 3 classes Vertex/Edge/Graph for implementation of valuated graphs
*/




//    declaration of 3 templates classes for Valuated-Graph (VGà , parametrised by 
//
//    - TA_Vertex : attribute  of vertex
//    - TA_Oriented : attribute  of oriented edge ( A(S1->S2) !=  A(S2->S1) )
//    - TA_Sym : attribute  of non oriented edge  :  A(S1->S2) and A(S2->S1)  are shared
//
template <class TA_Vertex,class TA_Oriented,class TA_Sym>  class cVG_Vertex;  // Vertex of Valuated Graph
template <class TA_Vertex,class TA_Oriented,class TA_Sym>  class cVG_Edge;    // Edge of Valuated Graph
template <class TA_Vertex,class TA_Oriented,class TA_Sym>  class cVG_Graph;   // Valuated Graph



/**   class for representing neighboors of a vertex in a valuated graph

      In adjacency graph, the neighboor of a vertex S1 are represented by a set of oriented edge S1->S2 : cVG_Edge
     An edge contain :
        * the vertex S2 it is directing to
        * the oriented attribute  
        * the shared symetric attribute (in fact mNumAttr)
        * a boolean indicating if, between the 2 oriented edges, it correspond to the direction at creation,
          used for example in group-graph, where the group G value is shared, to know if :
                    S1->S2 must be interpreted as G or G-1 
*/

static constexpr int FlagEdgeDirInit = 0;

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  class cVG_Edge : public cMemCheck
{
     public :
          // ------------------- typedef & friendship --------------------------------
          typedef cVG_Vertex<TA_Vertex,TA_Oriented,TA_Sym>  tVertex;
          typedef cVG_Graph<TA_Vertex,TA_Oriented,TA_Sym>   tGraph;
          typedef cVG_Edge<TA_Vertex,TA_Oriented,TA_Sym>    tEdge;

          friend tVertex;
          friend tGraph;

          // ------------------- method, essentially accesor --------------------------------

	  inline       tVertex & Succ() ;         ///< Accesor to "S2"
	  inline const tVertex & Succ() const ;   ///< Accesor to "S2"

	  inline TA_Oriented & AttrOriented()  {return mAttrO;}  ///<  Accessor to oriented attribute
	  inline const TA_Oriented & AttrOriented() const {return mAttrO;}  ///<  Accessor to oriented attribute
 

	  inline       TA_Sym & AttrSym() ;  ///< Accessor to symetric attribute
	  inline const TA_Sym & AttrSym() const ;  ///< Accessor to symetric attribute

          inline bool IsDirInit()  const {return  mBitMarked.IsInside(FlagEdgeDirInit);}
	  inline tEdge  * EdgeInitOr () {return IsDirInit() ? this : mEdgeInv;}
	  inline const tEdge  * EdgeInitOr() const {return IsDirInit() ? this : mEdgeInv;}

	  inline tEdge  * DirInv() {return  mEdgeInv;}
	  inline const tEdge  * DirInv() const {return  mEdgeInv;}

     private :
          cVG_Edge(const tEdge&) = delete;  ///< No copy for graph structures
          inline ~cVG_Edge();
          inline cVG_Edge(tGraph*,const TA_Oriented &,int aNumSucc,int aNumAttr,bool DirInit);
	  void SetEdgeInv(tEdge*);
          

	  tGraph *        mGr;         ///<  Graph it's belongin to
	  tEdge *         mEdgeInv;    ///<  The "invert" edge  if V1->V2 , then V2->V1
	  TA_Oriented     mAttrO;      ///<  Oriented Attribute
	  int             mNumSucc;    ///<  Num of successor ~ to a pointer 
          size_t          mNumAttr;    ///<  Num of symetric attribute ~ to a pointer
	  tSet32Bits      mBitMarked;  ///
          // bool            mDirInit;    ///<  Is it the 2 of edges that correspond to initial direction
};


/**  Class for representing the vertex of a valuated graph */

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  class cVG_Vertex : public cMemCheck
{
     public :
          // ------------------- typedef & friendship --------------------------------
	  typedef cVG_Vertex<TA_Vertex,TA_Oriented,TA_Sym>    tVertex;
	  typedef cVG_Edge<TA_Vertex,TA_Oriented,TA_Sym>      tEdge;
          typedef cVG_Graph<TA_Vertex,TA_Oriented,TA_Sym>     tGraph;

          friend  tEdge;
          friend  tGraph;

                    //  Algorithmd require acces to some internal variables
          template <class> friend class cAlgoSP;
          template <class> friend class cAlgoCC;
          template <class> friend class cHeapParam_VG_Graph;

         

          // ---------------  Manipulating Attribute of vertex ------------------------------
          inline void              SetAttr(const TA_Vertex & anAttr)  {mAttr = anAttr;}
          inline TA_Vertex &       Attr()       {return mAttr;}
          inline const TA_Vertex & Attr() const {return mAttr;}

          // ---------------  Manipulating adjacent edges ------------- ------------------------------

                 //  For S2, return edge E such E=S1->S2, 0 if does not exist (error if not SVP)
          const tEdge * EdgeOfSucc(const tVertex  & aS2,bool SVP=false) const {return const_cast<tVertex*>(this)->EdgeOfSucc(aS2,SVP);}
          inline tEdge * EdgeOfSucc(const tVertex  & aS2,bool SVP=false) ;

          inline const  std::vector<tEdge*> & EdgesSucc()       {return mVEdges;}  ///< Acessor
          inline const  std::vector<tEdge*> & EdgesSucc() const {return mVEdges;}  ///< Acessor

          // ---------------  Manipulating  data resulting from agorithm computation --------------------

          inline tREAL8  AlgoCost() const {return  mAlgoCost;}  //  Cost for readin lenght of shortest path

	  inline void BackTrackFathersPath(std::vector<tVertex*> &) ;  // read shortest path by back-tracking ling of fathers
	  inline std::vector<tVertex*>  BackTrackFathersPath();        // read shortest path by back-tracking ling of fathers

                        // Bit manipulation, CAUTION !!,  read doc not to interferate with algorithm
          inline void SetBit1(size_t aBit) {mAlgoTmpMark.AddElem(aBit);}
          inline void SetBit0(size_t aBit) {mAlgoTmpMark.SuprElem(aBit);}
          inline bool BitTo1(size_t aBit) const {return mAlgoTmpMark.IsInside(aBit);}
          static inline void SetBit0(const std::vector<tVertex*> & aVV,size_t aBit) ; // facility 4 applying SetBit0 to a vect

          // ------------------------ Miscelaneous accessor ----------------------
          tGraph & Graph()              {return *mGr;}  ///<  Accessor
          const tGraph & Graph() const  {return *mGr;}  ///<  Accessor

     private :

          cVG_Vertex(const tVertex &) = delete;  ///< Nocopy
          inline ~cVG_Vertex();   ///< Free mVEdges
          inline cVG_Vertex(tGraph * aGr,const TA_Vertex & anAttr,int aNum) ; ///< initiales with no neighboor

          tGraph *              mGr;            ///< graph it belongs to 
          TA_Vertex             mAttr;          ///< attribute of the vertex
          std::vector<tEdge*>   mVEdges;        ///< vector of link to neighboors
          int                   mNumInGr;       ///< internal numbering

                      //  ------  Data for algorithms ------------ 

          tSet32Bits            mAlgoTmpMark;       ///<  Set of 32 bits for marking vertices
          tVertex*              mAlgoFather;        ///<  link to father in shortest-path like algorithms
          int                   mAlgoIndexHeap;     ///<  index in heap in some algo
          tREAL8                mAlgoCost;          ///< cost of vertex in some algo
};


/**   Class for representing a valuated graph */

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  class cVG_Graph : public cMemCheck
{
     public :
          // ------------------- typedef & friendship --------------------------------
	  typedef cVG_Vertex<TA_Vertex,TA_Oriented,TA_Sym>        tVertex;
	  typedef cVG_Edge<TA_Vertex,TA_Oriented,TA_Sym>          tEdge;
	  typedef cVG_Graph<TA_Vertex,TA_Oriented,TA_Sym>         tGraph;
	     
	  friend tVertex;
	  friend tEdge;

          // ------------------   Different accessors
          size_t NbVertex()   const {return mV_Vertices.size();}       ///<  number total of vertices
	  tVertex &        VertexOfNum(size_t aNum)       {return *mV_Vertices.at(aNum);} ///< KTh vertex
	  const tVertex &  VertexOfNum(size_t aNum) const {return *mV_Vertices.at(aNum);} ///< KTh vertex

	  const std::vector<tVertex*>  &    AllVertices() const {return mV_Vertices;}  ///< vector of vertices (as input to some algo)
	  const std::vector<TA_Sym*>   &    AllAttrSym() const  {return mV_AttrSym;}   ///< all atributs, usefull for some global setting
           

          // ------------------   Creation of objects  (graph/vertices/edges) ----------------
          inline cVG_Graph();  ///< constructor does do a lot
          inline ~cVG_Graph();  ///< free memory

          inline tVertex * NewSom(const TA_Vertex & anAttr) ;  ///< create a vertex with given attribute

             /** create  Edges  "V1->V2" and "V2->V1" with 2 attribute oriented and 1 attribute sym
                 for now, error if already exist, to see if it must evolve */
          inline tEdge * AddEdge(tVertex & aV1,tVertex & aV2,const TA_Oriented &A12,const TA_Oriented &A21,const TA_Sym &,bool OkExist=false);

             /// create Edges, case where initially the 2 oriented attributes are equal
          tEdge * AddEdge(tVertex & aV1,tVertex & aV2,const TA_Oriented &aAOr,const TA_Sym & aASym,bool OkExist) 
               {AddEdge(aV1,aV2,aAOr,aAOr,aASym,OkExist);}

                  //------------------ Bit marker of vertices  manipulation ------------------
          inline size_t Vertex_AllocBitTemp();          ///<  alloc a bit free an return it
          inline void   Vertex_FreeBitTemp(size_t);     ///< "Recycle" a bit no longer used

          inline size_t Edge_AllocBitTemp();          ///<  alloc a bit free an return it
          inline void   Edge_FreeBitTemp(size_t);     ///< "Recycle" a bit no longer used

          // inline size_t AllocBitTemp();          ///<  alloc a bit free an return it
          // line void   FreeBitTemp(size_t);     ///< "Recycle" a bit no longer used
     private :
          cVG_Graph(const tGraph &) = delete;  ///< no copy for this type

	  const TA_Sym &  AttrSymOfNum(size_t aNum) const {return *mV_AttrSym.at(aNum);}
	  TA_Sym &        AttrSymOfNum(size_t aNum)       {return *mV_AttrSym.at(aNum);}


	  std::vector<tVertex*>       mV_Vertices;
	  std::vector<TA_Sym*>        mV_AttrSym;
          tSet32Bits                  mVertex_BitsAllocaTed;
          tSet32Bits                  mEdge_BitsAllocaTed;
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
         bool isDirInit
   ) :
     mGr       (aGr),    
     mEdgeInv  (nullptr),
     mAttrO    (anAttrO),
     mNumSucc  (aNumSucc),
     mNumAttr  (aNumAttr),
     mBitMarked (0)
     // mDirInit  (DirInit)
{
       if (isDirInit) 
           mBitMarked.AddElem(FlagEdgeDirInit);
}

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  
  void cVG_Edge<TA_Vertex,TA_Oriented,TA_Sym>::SetEdgeInv(tEdge* anInv)
{
    MMVII_INTERNAL_ASSERT_tiny(mEdgeInv==nullptr,"Multiple SetEdgeInv");
    mEdgeInv = anInv;
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
     // find if it exist the edge  V1->V2
     for (auto & anEdge : mVEdges)
         if (anEdge->mNumSucc==aS2.mNumInGr)
            return  anEdge;

      // return nullptr if allowed
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
     tVertex * aV = this;  // begin with this vertex
     while (aV!=nullptr)   // loop untill there is no longer any father
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
      mNumInGr       (aNum),
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
        mVertex_BitsAllocaTed (0),
        mEdge_BitsAllocaTed (0)
{
    mEdge_BitsAllocaTed.AddElem(FlagEdgeDirInit);
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
  cVG_Edge<TA_Vertex,TA_Oriented,TA_Sym>* cVG_Graph<TA_Vertex,TA_Oriented,TA_Sym>::AddEdge
      (
           tVertex & aV1,
           tVertex & aV2,
           const TA_Oriented &A12,
           const TA_Oriented &A21,
           const TA_Sym & aASym,
           bool  OkExist
       )
{
    // Not sure can handle this case, at least would require some precaution for the oriented part (duplication ...)
    MMVII_INTERNAL_ASSERT_tiny((aV1.mNumInGr!=aV2.mNumInGr) ,"Reflexive edge in Add edge");
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
         anE12 = new tEdge(this,A12,aV2.mNumInGr,mV_AttrSym.size(),true);  // true= dir init
         aV1.mVEdges.push_back(anE12);
	 tEdge * anE21 = new tEdge(this,A21,aV1.mNumInGr,mV_AttrSym.size(),false); // false= NOT dir init
         aV2.mVEdges.push_back(anE21); // false= NOT dir init

	 anE12->SetEdgeInv(anE21);
	 anE21->SetEdgeInv(anE12);

         mV_AttrSym.push_back(new TA_Sym(aASym));
         //anE12 = aV1.mVEdges.back();
    }
    return anE12;
}
template <class TA_Vertex,class TA_Oriented,class TA_Sym>  
    size_t  cVG_Graph<TA_Vertex,TA_Oriented,TA_Sym>::Vertex_AllocBitTemp()
{
    // look for a bit not already currently allocated
    for (size_t aBit=0 ; aBit<32 ; aBit++)
    {
          if (!mVertex_BitsAllocaTed.IsInside(aBit))
          {
               mVertex_BitsAllocaTed.AddElem(aBit);
               return aBit;
          }
    }
    //  if not : Error
    MMVII_INTERNAL_ERROR("No more bits in Vertex_AllocBitTemp (forgot to free ?)");
    return 0;
}

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  
    void  cVG_Graph<TA_Vertex,TA_Oriented,TA_Sym>::Vertex_FreeBitTemp(size_t aBit)
{
    // recycle, the bit is usable again
    mVertex_BitsAllocaTed.SuprElem(aBit);
}

};
#endif // _MMVII_Tpl_GraphStruct_H_  



