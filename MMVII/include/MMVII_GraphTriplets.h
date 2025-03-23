#ifndef _MMVII_Tpl_GraphTriplet_H_
#define _MMVII_Tpl_GraphTriplet_H_

#include "MMVII_PoseTriplet.h"
#include "MMVII_Tpl_GraphStruct.h"
#include "MMVII_Tpl_GraphAlgo_SPCC.h"
// #include "MMVII_Tpl_GraphAlgo_Group.h"



namespace MMVII
{

class c3G3_AttrOriented;
class c3G3_AttrSym;
class c3G3_AttrV;
typedef cVG_Vertex<c3G3_AttrV, c3G3_AttrOriented,c3G3_AttrSym> t3G3_Vertex;

/* ********************************************************* */
/*                                                           */
/*                     Graph of poses                        */
/*                                                           */
/* ********************************************************* */

/* The graph of pose is an instantion of cGroupGraph, wich is a specialization of cVG_Graph ...

   It require 5 attributes (for now c3GOP_AttrSym and c3GOP_AttrOr are empty)
*/


/// The attribite complementing on hypothesis of rotation
class c3GOP_1Hyp
{
   public :
        explicit c3GOP_1Hyp(int aNumSet) : mNumSet (aNumSet) {}
        c3GOP_1Hyp() : mNumSet(-1) {}

        int  mNumSet;   /// the num of set (i.e.  triplet in our case)
};

/// the attribute of vertex of 
class c3GOP_AttrV
{
    public :
         c3GOP_AttrV(int aKIm,const tPoseR& aGTRand): mKIm (aKIm), mGTRand (aGTRand)          {}

         cWhichMin<int,tREAL8>      mWMinTri;   /// for computing the triplet  with minimal cost it belongs to
	 int                        mKIm;
	 tPoseR                     mGTRand;   /// Random Ground-truth, for simul/check
         std::vector<int>           mTriBelongs;      /// Triplet it belongs to
};

/// the symetric  attibute of cGroupGraph
class c3GOP_AttrSym
{
   public :
        void ComputePoseRef(tREAL8 aWTr);

        std::vector<tPoseR>   mListP2to1;
        std::vector<size_t>   mListKT;
        /// +- the robust estimator of mListP2to1, Used when we need to have "the" pose of 1 edge
        tPoseR                mPoseRef2to1;
};

/// the oriented attribute of cGroupGraph
class c3GOP_AttrOr
{
   public :
};

typedef cVG_Graph<c3GOP_AttrV,c3GOP_AttrOr,c3GOP_AttrSym> t3GOP;

typedef typename t3GOP::tVertex                          t3GOP_Vertex;
typedef typename t3GOP::tEdge                            t3GOP_Edge;

/* ********************************************************* */
/*                                                           */
/*                     cAppli_ArboTriplets                   */
/*                                                           */
/* ********************************************************* */
typedef std::array<t3GOP_Edge*,3>                           t3E_GOP;
typedef std::array<t3GOP_Vertex*,3>                         t3V_GOP;


/// Oriented edge-attribute of graph of triplet
class c3G3_AttrOriented
{
   public :
        c3G3_AttrOriented()   {}
   private :
};

///  Symetric edge-attribute of graphe of triplet
class c3G3_AttrSym
{
   public :
      c3G3_AttrSym(tREAL8 aCost) : mCostInit2Ori (aCost) {}

      tREAL8 mCostInit2Ori;  /// Initial cost make from coherence of 2 orientations
   private :
};


///  Vertex attribute of graphe of triplet
class c3G3_AttrV
{
   public :
        c3G3_AttrV (cTriplet* aT0,int aKT) ;
        size_t GetIndexVertex(const t3GOP_Vertex* aV) const;
        tREAL8 CostVertexCommon(const c3G3_AttrV &,tREAL8 aWTr) const;

        cTriplet*        mT0;    ///< the initial triplet itself
        int              mKT;
        t3E_GOP          mCnxE;      // the 3 edges 
        t3V_GOP           m3V;        // the 3 vertices 
        bool             mOk;        // for ex, not OK if on of it edges was not clustered
        tREAL8           mCostIntr;  // intrisiq cost
        std::vector<int> mVPoseMin;  // List of pose consider this a best triplet
};

typedef cVG_Graph<c3G3_AttrV, c3G3_AttrOriented,c3G3_AttrSym> t3G3_Graph;
typedef cVG_Tree<t3G3_Graph>   t3G3_Tree;
typedef t3G3_Graph::tEdge      t3G3_Edge;
//typedef cVG_Vertex<c3G3_AttrV, c3G3_AttrOriented,c3G3_AttrSym> t3G3_Vertex;
typedef t3G3_Graph::tVertex    t3G3_Vertex;




}; //  namespace MMVII

#endif // _MMVII_Tpl_GraphTriplet_H_
