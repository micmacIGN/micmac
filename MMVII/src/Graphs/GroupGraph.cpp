#include "MMVII_nums.h"
#include "MMVII_util_tpl.h"
#include "MMVII_Geom2D.h"
#include "MMVII_Geom3D.h"

#include "MMVII_Tpl_GraphAlgo_SPCC.h"
#include "MMVII_Tpl_GraphStruct.h"

namespace MMVII
{

	/*
	  void ComputeAllNonOrCycles(int aLengthCycle);
	  void ComputeNonOrCyclesBy1Edge(const tEdge &,int aLengthCycle);
	  */

/*
    Regarding the notation (supposing it is poses) :

        * In class Som ,  mCurValue is the mapping Cam->Word  (or column of the rotaation are vector IJK)

               Ori_L->G   Local camera coordinates   ->   Global Word coordinate
                           PG = Ori_L->G (PL)

       *  In class AttrEdge,   Ori_2->1 is pose of Cam2 relatively to Cam 1,   PL2 ->PL1

               Ori_2->1  = Ori_G->1  o Ori_2->G    ( PL2 -> PG -> PL1)
*/

template <class TGroup>  class cGGA_EdgeOr;
template <class TGroup>  class cGGA_EdgeSym;
template <class TGroup>  class cGGA_Vertex;
template <class TGroup>  class cGGA_Graph;

/* ********************************************************* */
/*                                                           */
/*                     cGGA_EdgeOr                           */
/*                                                           */
/* ********************************************************* */

template <class TGroup>  class cGGA_EdgeOr
{
     public :
};

/* ********************************************************* */
/*                                                           */
/*                     cGGA_EdgeSym                          */
/*                                                           */
/* ********************************************************* */


template <class TGroup>  class cGG_1ElemSym
{
    public  :
      TGroup   mVal;
      tREAL8   mWeight;
};


template <class TGroup>  class cGGA_EdgeSym
{
     public :
          typedef cGG_1ElemSym<TGroup>  t1Elem;


          void Add1Value(const TGroup& aVal,tREAL8 aW);
         
     private :
          std::vector<t1Elem>  mValues;
};

template <class TGroup> void cGGA_EdgeSym<TGroup>::Add1Value(const TGroup& aVal,tREAL8 aW)
{
    t1Elem anElem;
    anElem.mVal = aVal;
    anElem.mWeight = aW;
    
    mValues.push_back(anElem);
}

/* ********************************************************* */
/*                                                           */
/*                     cGGA_Vertex                           */
/*                                                           */
/* ********************************************************* */


template <class TGroup>  class cGGA_Vertex
{
     public :
          cGGA_Vertex(const std::string & aName);
          
     private :
          std::string         mName;
          TGroup              mComputedValue;
};

template <class TGroup>  
    cGGA_Vertex<TGroup>::cGGA_Vertex(const std::string & aName)  :
        mName (aName)
{
}

/* ********************************************************* */
/*                                                           */
/*                     cGGA_Graph                            */
/*                                                           */
/* ********************************************************* */


template <class TGroup>  class cGroupGraph
{
    public :
          typedef cGGA_EdgeOr<TGroup>  tAEOr;
          typedef cGGA_EdgeSym<TGroup> tAESym;
          typedef cGGA_Vertex<TGroup>  tAVert;

          typedef cVG_Graph<tAVert,tAEOr,tAESym> tGraph;
          typedef typename tGraph::tVertex       tVertex;
          typedef typename tGraph::tEdge         tEdge;

          cGroupGraph();

          tVertex &  AddVertex(const std::string & aName);
          tVertex &  VertexOfName(const std::string & aName);

          void AddEdge(tVertex& aN1,tVertex& aN2,const TGroup &,tREAL8 aW);
          void AddEdge(const std::string & aN1,const std::string & aN2,const TGroup &,tREAL8 aW);

          TGroup  ValOrient(const TGroup& aG,const tEdge & anE) { return anE.IsDirInit() ? aG : aG.MapInverse(); }

    protected :
           tGraph mGraph;
           std::map<std::string,tVertex*>  mMapV;
};


template <class TGroup>  
    cGroupGraph<TGroup>::cGroupGraph() :
        mGraph ()
{
}


template <class TGroup>  
    typename cGroupGraph<TGroup>::tVertex &
        cGroupGraph<TGroup>::AddVertex(const std::string & aName) 
{
    MMVII_INTERNAL_ASSERT_tiny(!MapBoolFind(mMapV,aName),"cGroupGraph, name alrady exist :" +aName);

    tVertex * aV = mGraph.NewSom(tAVert(aName));
    mMapV[aName] = aV;
    return *aV;
}

template <class TGroup>  
    typename cGroupGraph<TGroup>::tVertex &
        cGroupGraph<TGroup>::VertexOfName(const std::string & aName) 
{
    MMVII_INTERNAL_ASSERT_tiny(MapBoolFind(mMapV,aName),"cGroupGraph, does not exist :" +aName);

    return *(mMapV[aName]);
}

template <class TGroup>  
    void cGroupGraph<TGroup>::AddEdge(tVertex & aV1,tVertex & aV2,const TGroup& aG,tREAL8 aW)
{
    tEdge * anE = aV1.EdgeOfSucc(aV2,SVP::Yes);
    if (anE==nullptr)
       anE = mGraph.AddEdge(aV1,aV2,tAEOr(),tAEOr(),tAESym());

}
  
template <class TGroup>  
   void cGroupGraph<TGroup>::AddEdge(const std::string & aN1,const std::string & aN2,const TGroup& aG,tREAL8 aW)
{
   AddEdge(VertexOfName(aN1),VertexOfName(aN2),aG,aW);
}


/* ********************************************************* */
/*                                                           */
/*                     cBench_G3                             */
/*                                                           */
/* ********************************************************* */


template class cGGA_EdgeOr<tRotR>;
template class cGGA_EdgeSym<tRotR>;
template class cGGA_Vertex<tRotR>;
template class cGroupGraph<tRotR>;



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

          cBench_G3(const cPt2di & aSz,int aNbMin,int aNbMax,tREAL8 aPropOutLayer,tREAL8 aNoiseInLayer,tREAL8 aNoiseOutLayer);

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
          tREAL8 mNoiseOutLayer;
};

template <class TGroup> 
    cBench_G3<TGroup>::cBench_G3
    (
        const cPt2di & aSz,
        int aNbMinE,int aNbMaxE,
        tREAL8 aPropOutLayer,tREAL8 aNoiseInLayer,tREAL8 aNoiseOutLayer
    ) :
       mSz  (aSz),
       mBox (cPt2di(0,0),aSz),
       mGG  (),

       mGridVals       (mSz.y(),std::vector<cBG3V>(mSz.x(),cBG3V())),
       mNbMinE         (aNbMinE),
       mNbMaxE         (aNbMinE),
       mPropOutLayer   (aPropOutLayer),
       mNoiseInLayer   (aNoiseInLayer),
       mNoiseOutLayer  (aNoiseOutLayer)
{
    for (const auto & aPix : mBox)
    {
        std::string aName = ToStr(aPix.x()) + "_" + ToStr(aPix.y());
        ValOfPt(aPix).mVertex = &mGG.AddVertex(aName);
        ValOfPt(aPix).mValRef = TGroup::RandomElem();
    }

    for (const auto & aPix : mBox)
    {
          Add1Edge(aPix,aPix+cPt2di(1,0));
          Add1Edge(aPix,aPix+cPt2di(0,1));
          Add1Edge(aPix,aPix+cPt2di(1,1));
          Add1Edge(aPix,aPix+cPt2di(-1,1));
    }

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
        tREAL8 aNoise = RandUnif_C() * (isInLayer ? mNoiseInLayer : mNoiseOutLayer);
        TGroup aRel_2To1 =  aRefRel_2To1 * TGroup::RandomSmallElem(aNoise);

        mGG.AddEdge(*(aVal0.mVertex),*(aVal1.mVertex),aRel_2To1,1.0);
   }
}


template class cBench_G3<tRotR>;

void BenchGroupGraph(cParamExeBench & aParam)
{
    if (! aParam.NewBench("GroupGraph")) return;

    cBench_G3<tRotR> aBG3(cPt2di(17,22),2,4,0.1,0.05,0.5);


    aParam.EndBench();
}




#if (0)


/*
    Regarding the notation (supposing it is poses) :

        * In class Som ,  mCurValue is the mapping Cam->Word  (or column of the rotaation are vector IJK)

               Ori_L->G   Local camera coordinates   ->   Global Word coordinate
                           PG = Ori_L->G (PL)

       *  In class AttrEdge,   Ori_2->1 is pose of Cam2 relatively to Cam 1,   PL2 ->PL1

               Ori_2->1  = Ori_G->1  o Ori_2->G    ( PL2 -> PG -> PL1)
*/

template <class TVal>  class cGrpValuatedEdge;
template <class TVal>  class cGrpValuatedAttrEdge;
template <class TVal>  class cGrpValuatedSom;
template <class TVal,class TParam>  class cGrpValuatedGraph;



/* ********************************************************* */
/*                                                           */
/*                cGrpValuateEdge                            */
/*                                                           */
/* ********************************************************* */

template <class TVal>  class cGrpValuatedEdge
{
    public :
          cGrpValuatedEdge(int aSucc,int aNumAttr,bool DirInit) ;

          size_t  mSucc;
          size_t  mNumAttr;
          bool    mDirInit;

          // The mapping transformating Coord2 to Coord1 associate to a Val
          TVal    VRel2_to_1 (const TVal & aV0) const {return  mDirInit ? aV0 : aV0.MapInverse() ;}
          // The mapping transformating Coord1 to Coord2 associate to a Val
          TVal    VRel1_to_2 (const TVal & aV0) const {return  mDirInit ? aV0.MapInverse() : aV0 ;}
};

template <class TVal>  
   cGrpValuatedEdge<TVal>::cGrpValuatedEdge(int aSucc,int aNumAttr,bool isDirInit) :
       mSucc      (aSucc),
       mNumAttr   (aNumAttr),
       mDirInit   (isDirInit)
{
}

/* ********************************************************* */
/*                                                           */
/*                cGrpValuated_OneAttrEdge                   */
/*                                                           */
/* ********************************************************* */

template <class TVal>  class cGrpValuated_OneAttrEdge
{
       public :
          cGrpValuated_OneAttrEdge(const TVal &,std::string aAttrib,tREAL8 aCost);
          TVal         mVal; 
          std::string  mAux;
          tREAL8       mCost; 
};

template <class TVal>  cGrpValuated_OneAttrEdge<TVal>::cGrpValuated_OneAttrEdge(const TVal & aVal,std::string anAux,tREAL8 aCost) :
     mVal    (aVal),
     mAux    (anAux),
     mCost   (aCost)
{
}

/* ********************************************************* */
/*                                                           */
/*                cGrpAttrEdge                               */
/*                                                           */
/* ********************************************************* */

template <class TVal>  class cGrpValuatedAttrEdge
{
    public :

          typedef  cGrpValuated_OneAttrEdge<TVal>  tOneAttr;
          typedef  std::vector<tOneAttr>           tSetAttr;

          cGrpValuatedAttrEdge();
          cGrpValuatedAttrEdge(const TVal & aValues,std::string aAttrib,tREAL8 aPrioriCost)  ;

          void   Add(const TVal & aValue,std::string aAttrib,tREAL8 aPrioriCost) ;
          
          size_t  NbValues () const {return mValues.size();}
          const tOneAttr & KthVal (size_t aK) const {return mValues.at(aK);}
          tOneAttr & KthVal (size_t aK) {return mValues.at(aK);}

          const tSetAttr & Values () const {return mValues;}
          tSetAttr & Values () {return mValues;}

          void SetIndexMinCost(int aInd) {mIndMinCost=aInd;}

          const tOneAttr & AttrMinCost() const;

	  bool IsTree() const {return mIsTree;}
          void SetIsTree(bool IsTree);
    private :
          tSetAttr          mValues;
          int               mIndMinCost;
          bool    mIsTree;
};

template <class TVal>  
    cGrpValuatedAttrEdge<TVal>::cGrpValuatedAttrEdge() :
        mIndMinCost (-1),
	mIsTree     (false)
{
}

template <class TVal>
    void cGrpValuatedAttrEdge<TVal>::SetIsTree(bool isTree)
{
  mIsTree = isTree;
}

template <class TVal>  
    cGrpValuatedAttrEdge<TVal>::cGrpValuatedAttrEdge(const TVal & aValue,std::string anAttrib,tREAL8 aPrioriCost)  :
          cGrpValuatedAttrEdge<TVal>()
{
     Add(aValue,anAttrib,aPrioriCost);
}

template <class TVal> void  cGrpValuatedAttrEdge<TVal>::Add(const TVal & aValue,std::string anAttrib,tREAL8 aCost) 
{
   mValues.push_back(tOneAttr(aValue,anAttrib,aCost));
}
 
template <class TVal> 
    const cGrpValuated_OneAttrEdge<TVal> & cGrpValuatedAttrEdge<TVal>::AttrMinCost() const
{
    MMVII_INTERNAL_ASSERT_tiny(mIndMinCost>=0,"AttrMinCost non init");

    return    mValues.at(mIndMinCost);
}

/* ********************************************************* */
/*                                                           */
/*                cGrpValuatedSom                            */
/*                                                           */
/* ********************************************************* */

template <class TVal>  class cGrpValuatedSom
{
    public :
          typedef cGrpValuatedEdge<TVal>  tEdge;

          cGrpValuatedSom(const TVal & aValue,int aNum) ;
          cGrpValuatedSom(int aNumSom) ;  

          /// return the adress of Edge such aSom2 is the successor, 0 if none
          tEdge * EdgeOfSucc(size_t aSom2,bool OK0) ;

          /**  add a successor for s2, with attribute at adr aNumAttr, DirInit if the attribute is
               relative to way  "S1->S2" (else it's "S2->S1")
          */
          void AddEdge(int aS2,int aNumAttr,bool DirInit);

          int & HeapIndex() {return mHeapIndex;}
          const std::list<tEdge> &  LSucc() const {return mLSucc;}

          const TVal &  RefVal() const {return mRefValue;}
          void  SetRefVal(const TVal& aNewV) {mRefValue= aNewV;}

          const TVal &  CompVal() const {return mComputedValue;}
          void  SetCompVal(const TVal& aNewV) {mComputedValue= aNewV;}

          const tREAL8 &  HeapCost() const {return mHeapCost;}
          void  SetHeapCost(tREAL8 aNewV)  { mHeapCost=aNewV;}

          int   NumRegion() const {return mNumRegion;}
          void  SetNumRegion(int aNewV) {mNumRegion= aNewV;}
          
          int   NumSom() const {return mNumSom;}

          void SetHeapFather(int aNewV) {mNumHeapFather = aNewV;}
          int  NumHeapFather() const {return mNumHeapFather;}
    private :
          TVal              mRefValue;
          TVal              mComputedValue;
          int               mNumSom;
          int               mNumHeapFather;
          tREAL8            mHeapCost;
          std::list<tEdge>  mLSucc;
          int               mNumRegion;
          int               mHeapIndex;
};


template <class TVal>  cGrpValuatedSom<TVal>::cGrpValuatedSom(int aNumSom) : 
     mRefValue       (),
     mComputedValue  (),
     mNumSom         (aNumSom),
     mNumHeapFather  (-1),
     mHeapCost       (1e10),
     mNumRegion      (-1)
{
}


template <class TVal>  
     cGrpValuatedEdge<TVal>  * cGrpValuatedSom<TVal>::EdgeOfSucc(size_t aSucc,bool Ok0) 
{
   for (auto &  anEdge : mLSucc)
       if (anEdge.mSucc == aSucc)
          return & anEdge;

   MMVII_INTERNAL_ASSERT_tiny(Ok0,"Could not find EdgeOfSucc");
   
   return nullptr;
}

template <class TVal>  void cGrpValuatedSom<TVal>::AddEdge(int aS2,int aNumAttr,bool isDirInit)
{
   mLSucc.push_back(tEdge(aS2,aNumAttr,isDirInit));
}


template <class TVal>  class cHeapCmpGrpValuatedSom
{
    public :
       typedef cGrpValuatedSom<TVal> *         tSomPtr ;

       bool  operator () (const tSomPtr & aS1,const tSomPtr & aS2) {return aS1->HeapCost() < aS2->HeapCost();}
};

template <class TVal>  class cParamHeapCmpGrpValuatedSom
{
    public :
        typedef cGrpValuatedSom<TVal> *         tSomPtr ;

        static void SetIndex(const tSomPtr & aSom,tINT4 i) { aSom->HeapIndex() = i;} 
        static int  GetIndex(const tSomPtr & aSom)  
        {
             return aSom->HeapIndex();
        }

};


/* ********************************************************* */
/*                                                           */
/*                cGrpValuatedGraph                          */
/*                                                           */
/* ********************************************************* */

class cParamGrpRot
{
    public :
        static tREAL8 GrpDist(const tRotR & aR1,const tRotR& aR2)
        {
             return   aR1.Mat().L2Dist(aR2.Mat());
        }
};


template <class TVal,class TParam>  class cGrpValuatedGraph
{
    public :

          typedef cParamHeapCmpGrpValuatedSom<TVal> tParamHeap;
          typedef cHeapCmpGrpValuatedSom<TVal>      tCmpHeap;
          typedef cGrpValuatedEdge<TVal>            tEdge;
          typedef cGrpValuatedAttrEdge<TVal>        tAttr;
          typedef cGrpValuated_OneAttrEdge<TVal>    tOneAttr;
          typedef cGrpValuatedSom<TVal>             tSom ;

          cGrpValuatedGraph(int aNbSom,const TParam &,int aNbEdge=-1);
          void AddEdge(int aS1,int aS2,const TVal & aVal,std::string aAttrib,tREAL8 aCostAPriori);

          void MakeLoopCostApriori();
          std::vector<std::pair<int,int>>  MakeMinSpanTree();

          void  PropagateTreeSol();
          /// Return the relative position using  RefValue
          TVal  RelRef_2to1(int aS1,int aS2);

    protected :
          class  cLoop
          {
              public :
                  TVal   mVal;
                  size_t mSom;
                  int    mStackI;

                  cLoop (const TVal& aVal,size_t aSom,int aStackI ):  mVal(aVal), mSom(aSom), mStackI (aStackI) {}
          };

          void  Recurs_PropagateTreeSol(size_t aSom);

          void  MakeLoopCostApriori(size_t aS1,const tEdge&, tOneAttr &);


          const tAttr &  AttrOfEdge(const tEdge & anE) const {return mVAttr.at(anE.mNumAttr);}
          tAttr &  AttrOfEdge(const tEdge & anE)  {return mVAttr.at(anE.mNumAttr);}
          tAttr &  AttrOfEdge(size_t aS1,size_t aS2)  {return AttrOfEdge(*mVSoms.at(aS1).EdgeOfSucc(aS2,SVP::No));}
	
          const std::list<tEdge> &  SuccOfSom(size_t aKSom) const {return mVSoms.at(aKSom).LSucc();}
          const std::list<tEdge> &  SuccOfSom(const tSom & aSom) const {return SuccOfSom(aSom.NumSom());}
          
          tSom & S2OfEdge(const tEdge & anE) {return  mVSoms.at(anE.mSucc);}

          TParam              mParam;
          size_t              mNbSom;
          std::vector<tSom>   mVSoms;
          std::vector<tAttr>  mVAttr;

          std::map<std::string,cStdStatRes> mStats;
};

template <class TVal,class TParam> 
     cGrpValuatedGraph<TVal,TParam>::cGrpValuatedGraph(int aNbSom,const TParam & aParam,int aNbEdge) :
      mParam   (aParam),
      mNbSom   (aNbSom)
{
    for (int aK=0 ; aK<aNbSom ; aK++)
    {
         mVSoms.push_back(tSom(aK));
    }
}

template <class TVal,class TParam>  
    void cGrpValuatedGraph<TVal,TParam>::AddEdge(int aS1,int aS2,const TVal & aVal,std::string aAttrib, tREAL8 aCostAPriori)
{
    const tEdge * anEdge = mVSoms.at(aS1).EdgeOfSucc(aS2,SVP::Yes);
    if (anEdge)
    {
       tAttr&  anAttr =  AttrOfEdge(*anEdge); 
       TVal aValCor = anEdge->VRel2_to_1(aVal);
       anAttr.Add(aValCor,aAttrib,aCostAPriori);
       return;
    }
    mVSoms.at(aS1).AddEdge(aS2,mVAttr.size(),true);
    mVSoms.at(aS2).AddEdge(aS1,mVAttr.size(),false);
 
    mVAttr.push_back(cGrpValuatedAttrEdge(aVal,aAttrib,aCostAPriori));
}

template <class TVal,class TParam>  TVal  cGrpValuatedGraph<TVal,TParam>::RelRef_2to1(int aS1,int aS2)
{
    // Ori_2->1  = Ori_G->1  o Ori_2->G    ( PL2 -> PG -> PL1)
    return  mVSoms.at(aS1).RefVal().MapInverse() *  mVSoms.at(aS2).RefVal();
}



template <class TVal,class TParam>  
    std::vector<std::pair<int,int>>   cGrpValuatedGraph<TVal,TParam>::MakeMinSpanTree()
{
    std::vector<std::pair<int,int>> aTree;
    tCmpHeap   aHCmp;
    cIndexedHeap<tSom*,tCmpHeap,tParamHeap>  aHeap(aHCmp);

    //  -1 not reached , 0 in Heap, already oustside
    aHeap.Push(&mVSoms.at(0));
    mVSoms.at(0).SetNumRegion(0);
  
    tSom * aNewSom = nullptr;
    while (aHeap.Pop(aNewSom))
    {
         aNewSom->SetNumRegion(1);
         int aNFather = aNewSom->NumHeapFather();
         if (aNFather>=0)
         {
             aTree.push_back(std::pair<int,int>(aNewSom->NumSom(),aNFather));
         }
         for (const auto &  aSucc :  SuccOfSom(*aNewSom))
         {
             tSom & aSom2Update = S2OfEdge(aSucc);
             if (aSom2Update.NumRegion() <=0)
             {
                if (aSom2Update.NumRegion() == -1)
                {
                    aHeap.Push(&aSom2Update);
                    aSom2Update.SetNumRegion(0);
                }
                tREAL8 aNewCost = AttrOfEdge(aSucc).AttrMinCost().mCost;

                if (aNewCost<aSom2Update.HeapCost())
                {
                    aSom2Update.SetHeapCost(aNewCost);
                    aSom2Update.SetHeapFather(aNewSom->NumSom());
                    aHeap.UpDate(&aSom2Update);
                }
             }
         }
    }

    for (size_t aKS=0 ; aKS<mVSoms.size() ; aKS++)
    {
        mVSoms.at(aKS).SetNumRegion(-1);
        mVSoms.at(aKS).SetHeapFather(-1);
        mVSoms.at(aKS).SetHeapCost(1e10);
    }

    for (const auto & [aS1,aS2] : aTree)
    {
        AttrOfEdge(aS1,aS2).SetIsTree(true);
    }

    return aTree;
}

template <class TVal,class TParam>  
    void  cGrpValuatedGraph<TVal,TParam>::PropagateTreeSol()
{
     mVSoms.at(0).SetCompVal(mVSoms.at(0).RefVal());
   //mVSoms.at(0).SetRefVal(tRotR::Identity());
     mVSoms.at(0).SetNumRegion(0);
     Recurs_PropagateTreeSol(0);
}

template <class TVal,class TParam>  
    void  cGrpValuatedGraph<TVal,TParam>::Recurs_PropagateTreeSol(size_t aNumSom_A)
{
    for (const auto &  aSucc_AB :  SuccOfSom(aNumSom_A))
    {
        tAttr&  anAttr = AttrOfEdge(aSucc_AB);
        if (anAttr.IsTree())
        {
             size_t aNumSom_B = aSucc_AB.mSucc;
	     tSom & aSomB = mVSoms.at(aNumSom_B);
             if (aSomB.NumRegion() == -1)
             {
                 aSomB.SetNumRegion(0);
		 // Ori_2->G = Ori_1->G o Ori_2->1    2->1->G
		 TVal aVal_B2A = aSucc_AB.VRel2_to_1(anAttr.AttrMinCost().mVal);
		 TVal aCompA = mVSoms.at(aNumSom_A).CompVal();
		 TVal aCompB =  aCompA * aVal_B2A;
                 aSomB.SetCompVal(aCompB);
  StdOut()  << "DIST= " << mParam.GrpDist(aCompB,aSomB.RefVal()) << "\n";
		 // aSom_B.
                 Recurs_PropagateTreeSol(aNumSom_B);
             }
        }
    }
}
/*
          const TVal &  CompVal() const {return mComputedValue;}
          void  SetCompVal(const TVal& aNewV) {mComputedValue= aNewV;}
	  */


template <class TVal,class TParam>  
    void cGrpValuatedGraph<TVal,TParam>::MakeLoopCostApriori()
{
    for (size_t aSom_A=0 ; aSom_A<mNbSom ; aSom_A++)
    {
         for (const auto &  aSucc_AB :  SuccOfSom(aSom_A))
         {
              // no need to do it 2 way
              if (aSucc_AB.mDirInit)
              {
                 tAttr & aVecAttr_AB  = AttrOfEdge(aSucc_AB);
                 for (auto & aAttr_AB : aVecAttr_AB.Values())
                 {
                    MakeLoopCostApriori(aSom_A,aSucc_AB,aAttr_AB);
                 }
              }
          }
    }

    // now the cost are made, memorize the min one  
    for (size_t aSom_A=0 ; aSom_A<mNbSom ; aSom_A++)
    {
         for (const auto &  aSucc_AB :  SuccOfSom(aSom_A))
         {
              if (aSucc_AB.mDirInit)
              {
                 tAttr & aVecAttr_AB  = AttrOfEdge(aSucc_AB);
                 cWhichMin<size_t,tREAL8>  aMinCost;
                 for (size_t aKV=0 ; aKV<aVecAttr_AB.NbValues() ; aKV++)
                 {
                     aMinCost.Add(aKV,aVecAttr_AB.Values().at(aKV).mCost);
                 }
                 aVecAttr_AB.SetIndexMinCost(aMinCost.IndexExtre());
              }
          }
    }

}

template <class TVal,class TParam>  
          void  cGrpValuatedGraph<TVal,TParam>::MakeLoopCostApriori(size_t aSom_A,const tEdge& anE_AB, tOneAttr &aAttr_AB)
{
    tREAL8  aEstimOutLayer = 0.1;
    tREAL8  aNbSignif = 1.0;

    int     aNbTot = aNbSignif;
    tREAL8  aSumCost = aEstimOutLayer * aNbSignif;


    TVal TheIdent = TVal::Identity();
    std::vector<cLoop>  aVLoop;
    std::vector<tREAL8> aVDist;

    size_t aSom_B  = anE_AB.mSucc;
    TVal aV_B2A  = anE_AB.VRel2_to_1(aAttr_AB.mVal);
    aVLoop.push_back(cLoop(aV_B2A,aSom_B,-1));
    //StdOut() << "******************************************************************************\n";

    size_t aInd0 =0;
    int aDist2A=1;
    while (aDist2A<3)
    {
          // StdOut() << "  ------------  DGRR= "<< aDist   << " ------------------------------ \n";
          size_t aInd1 = aVLoop.size();
          for (size_t aInd= aInd0 ; aInd<aInd1 ; aInd++)
          {
              size_t aSom_X = aVLoop.at(aInd).mSom;
              const TVal  aV_X2A = aVLoop.at(aInd).mVal;
              for (const auto &  anE_XY :  SuccOfSom(aSom_X))
              {
                  size_t aSom_Y = anE_XY.mSucc;
                  const tAttr &  aAllAttrs_XY = AttrOfEdge(anE_XY);

                  for (const auto & a1Attr_XY : aAllAttrs_XY.Values())
                  {
                      TVal aV_Y2X  = anE_XY.VRel2_to_1(a1Attr_XY.mVal);
                      TVal aV_Y2A  =    aV_X2A * aV_Y2X; // V_X2A (V_Y2X (PY)) = V_X2A(PX) = PA

                      if (aSom_Y == aSom_A)
                      {
                           tREAL8 aDist2Id = mParam.GrpDist(TheIdent, aV_Y2A);
                           tREAL8 aCost =  (aDist2Id *aEstimOutLayer)  / (aDist2Id + aEstimOutLayer);
                           aSumCost += aCost;
                           aNbTot ++;
                      }
                      else
                      {
                           aVLoop.push_back(cLoop(aV_Y2A,aSom_Y,aInd));
                      }
                  }
              }
          }
          aDist2A++;
          aInd0 = aInd1;
    }

    aSumCost /= aNbTot;

    aAttr_AB.mCost = aSumCost;
    mStats[aAttr_AB.mAux].Add(aSumCost);
    //StdOut()  << " SSS "  << aSumCost << " " << aAttr_AB.mAttrib << "\n";
}




class cBenchGrid_GVG :  public  cRect2,
                        public  cGrpValuatedGraph<tRotR,cParamGrpRot>
{
    public :

          typedef cGrpValuatedGraph<tRotR,cParamGrpRot>  tGVG;
          using tGVG::tEdge;
 
          cBenchGrid_GVG
          (
              cPt2di  aSz,
              tREAL8  aAmplGlob,
              tREAL8  aNoiseInlayers,
              tREAL8  aPropOutLayer,
              tREAL8  aNoiseOutlayers,
              cPt2di  aNbEdge
          );

          tREAL8  mAmplGlob;
          tREAL8  mNoiseInlayers;
          tREAL8  mPropOutLayer;
          tREAL8  mNoiseOutlayers;
          cPt2di  mNbEdge;

          void Pt_AddEdge(const cPt2di & aP0,const cPt2di & aP1);

/*
          typedef cGrpValuatedAttrEdge<TVal>      tAttr;
          typedef cGrpValuated_OneAttrEdge<TVal>  tOneAttr;
          typedef cGrpValuatedSom<TVal>           tSom ;
*/
};

void cBenchGrid_GVG::Pt_AddEdge(const cPt2di & aPA,const cPt2di & aPB)
{
    if (! (Inside(aPA) &&  Inside(aPB)))
    {
        return;
    }

    int aNbEdge = RandInInterval(mNbEdge.x(),mNbEdge.y());

    int aSom_A = IndexeLinear(aPA);
    int aSom_B = IndexeLinear(aPB);
    tRotR aR_B2A_Ref  = RelRef_2to1(aSom_A,aSom_B);

    while (aNbEdge)
    {
         aNbEdge--;
         bool IsOutLayer =  (RandUnif_0_1()<mPropOutLayer);
         tREAL8 aAmplNoise = IsOutLayer ? mNoiseOutlayers : mNoiseInlayers;
         aAmplNoise /= std::sqrt(2.0);

         tRotR aR_B2A  = tRotR::RandomRot(aAmplNoise) * aR_B2A_Ref * tRotR::RandomRot(aAmplNoise);

         AddEdge(aSom_A,aSom_B,aR_B2A,(IsOutLayer?"1":"0"),1.0);
    }
}

cBenchGrid_GVG::cBenchGrid_GVG
(
    cPt2di aSz,
    tREAL8  aAmplGlob,
    tREAL8  aNoiseInlayers,
    tREAL8  aPropOutLayer,
    tREAL8  aNoiseOutlayers,
    cPt2di  aNbEdge
) :
    cRect2           (aSz),
    tGVG             (size_t(cRect2::NbElem()),cParamGrpRot()),
    mAmplGlob        (aAmplGlob),
    mNoiseInlayers   (aNoiseInlayers),
    mPropOutLayer    (aPropOutLayer),
    mNoiseOutlayers  (aNoiseOutlayers),
    mNbEdge          (aNbEdge)
{
   for (auto & aSom : mVSoms)
      aSom.SetRefVal(tRotR::RandomRot(aAmplGlob));
   //mVSoms.at(0).SetRefVal(tRotR::Identity());
StdOut() << "SOMM-ADDED \n";

   for (const auto & aPix : *this)
   {
       Pt_AddEdge(aPix,aPix+cPt2di(0,1));
       Pt_AddEdge(aPix,aPix+cPt2di(1,0));
       Pt_AddEdge(aPix,aPix+cPt2di(1,1));
       Pt_AddEdge(aPix,aPix+cPt2di(1,-1));
   }
StdOut() << "EDGE-ADDED \n";
   this->MakeLoopCostApriori();
   for (const auto & [aName,aStat] : mStats)
   {
        StdOut() << " AttR=" << aName;
        for (const auto aProp : {0.01,0.1,0.5,0.9,0.99})
            StdOut() <<  " " << aProp << "->" << aStat.ErrAtProp(aProp) ;
        StdOut() << " \n";
    }
StdOut() << "LOOP-COST DONE\n";
   this->MakeMinSpanTree();
StdOut() << "MIN SPAN TREE DONE\n";
   PropagateTreeSol();
StdOut() << "PROPAG TREE SOL\n";
}




/* ********************************************************* */
/*                                                           */
/* ********************************************************* */



template class cGrpValuatedAttrEdge<tRotR>;
template class cGrpValuatedSom<tRotR>;
template class cGrpValuatedGraph<tRotR,cParamGrpRot>;


void xxx_BenchGrpValuatedGraph(cParamExeBench & aParam)
{
    if (! aParam.NewBench("GroupGraph")) return;

    // cBenchGrid_GVG  aBGVG(cPt2di(10,20),100.0, 0.0, 0.0,0.0,cPt2di(3,3));
    // cBenchGrid_GVG  aBGVG(cPt2di(10,20),100.0, 0.01, 0.2,0.5,cPt2di(3,11));
    cBenchGrid_GVG  aBGVG(cPt2di(20,20),100.0, 0.00, 0.2,0.3,cPt2di(2,3));

    aParam.EndBench();
}
#endif

};

