#include "MMVII_util_tpl.h"
#include "MMVII_Bench.h"
#include "MMVII_TplHeap.h"
#include "cMMVII_Appli.h"
#include "MMVII_SetITpl.h"
#include "MMVII_Images.h"

//#include "MMVII_nums.h"
//#include "MMVII_Geom2D.h"
//#include "MMVII_Geom3D.h"

namespace MMVII
{

//  3 declaration of templates classes for Valuated-Graph (VGà , parametrised by 
//
//    - TA_Vertex : attribute  of vertex
//    - TA_Oriented : attribute  of oriented edge ( A(S1->S2) !=  A(S2->S1) )
//    - TA_Sym : attribute  of non oriented edge  :  A(S1->S2) and A(S2->S1)  are shared
//

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  class cVG_Vertex;
template <class TA_Vertex,class TA_Oriented,class TA_Sym>  class cVG_Edge;
template <class TA_Vertex,class TA_Oriented,class TA_Sym>  class cVG_Graph;
template <class TGraph>  class cAlgo_ParamVG;
template <class TGraph> class cAlgoPCC;

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  class cVG_Edge : public cMemCheck
{
     public :
          typedef cVG_Vertex<TA_Vertex,TA_Oriented,TA_Sym>  tVertex;
          typedef cVG_Graph<TA_Vertex,TA_Oriented,TA_Sym>   tGraph;
          typedef cVG_Edge<TA_Vertex,TA_Oriented,TA_Sym>    tEdge;
          friend tVertex;
          friend tGraph;


	  inline       tVertex & Succ() ;
	  inline const tVertex & Succ() const ;

	  inline TA_Oriented & AttrOriented()  {return mAttrO;}
	  inline const TA_Oriented & AttrOriented() const {return mAttrO;}

	  inline       TA_Sym & AttrSym() ;
	  inline const TA_Sym & AttrSym() const ;

     private :
          cVG_Edge(const tEdge&) = delete;
          inline ~cVG_Edge();
          inline cVG_Edge(tGraph*,const TA_Oriented &,int aNumSucc,int aNumAttr,bool DirInit);
          

	  tGraph *        mGr;
	  TA_Oriented     mAttrO;
	  int             mNumSucc;
          size_t          mNumAttr;
          bool            mDirInit;
};

template <class TA_Vertex,class TA_Oriented,class TA_Sym>  class cVG_Vertex : public cMemCheck
{
     public :
	  typedef cVG_Vertex<TA_Vertex,TA_Oriented,TA_Sym>    tVertex;
	  typedef cVG_Edge<TA_Vertex,TA_Oriented,TA_Sym>      tEdge;
          typedef cVG_Graph<TA_Vertex,TA_Oriented,TA_Sym>     tGraph;

          typedef cAlgoPCC<tGraph>                            tAlgoPCC;
          typedef cAlgo_ParamVG<tGraph>                       tParamAlgo;

          friend  tEdge;
          friend  tGraph;
          friend  tParamAlgo;
          friend  tAlgoPCC;

          inline void              SetAttr(const TA_Vertex & anAttr)  {mAttr = anAttr;}
          inline TA_Vertex &       Attr()       {return mAttr;}
          inline const TA_Vertex & Attr() const {return mAttr;}

          const tEdge * EdgeOfSucc(const tVertex  & aS2,bool SVP=false) const {return const_cast<tVertex*>(this)->EdgeOfSucc(aS2,SVP);}
          inline tEdge * EdgeOfSucc(const tVertex  & aS2,bool SVP=false) ;

          inline const  std::vector<tEdge*> & EdgesSucc()       {return mVEdges;}
          inline const  std::vector<tEdge*> & EdgesSucc() const {return mVEdges;}

          inline tREAL8  AlgoCost() const {return  mAlgoCost;}
          inline int &   AlgoIndexHeap() {return  mAlgoIndexHeap;}

          inline void SetBit1(size_t aBit) {mAlgoTmpMark.AddElem(aBit);}
          inline void SetBit0(size_t aBit) {mAlgoTmpMark.SuprElem(aBit);}
          inline bool BitTo1(size_t aBit) const {return mAlgoTmpMark.IsInside(aBit);}

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
	     
	  friend tVertex;
	  friend tEdge;

          size_t NbVertex()   const {return mV_Vertices.size();}
	  tVertex &        VertexOfNum(size_t aNum)       {return *mV_Vertices.at(aNum);}
	  const tVertex &  VertexOfNum(size_t aNum) const {return *mV_Vertices.at(aNum);}

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
/*                                               */
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
  cVG_Vertex<TA_Vertex,TA_Oriented,TA_Sym>::cVG_Vertex(tGraph * aGr,const TA_Vertex & anAttr,int aNum) :
      mGr          (aGr),
      mAttr        (anAttr),
      mNum         (aNum),
      mAlgoTmpMark (0)
{
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

template class cVG_Graph<int,int,int>;
template class cVG_Vertex<int,int,int>;
template class cVG_Edge<int,int,int>;

template <class TGraph>  class cAlgo_SubGr
{
        public :
	     typedef typename TGraph::tVertex  tVertex;
	     typedef typename TGraph::tEdge    tEdge;

             virtual bool   InsideVertex(const  tVertex &) const {return true;}
};

template <class TGraph>  class cAlgo_ParamVG : public cAlgo_SubGr<TGraph>
{
        public :
	     typedef typename TGraph::tVertex  tVertex;
	     typedef typename TGraph::tEdge    tEdge;

             virtual bool   InsideEdge(const    tEdge &) const {return true;}
             virtual tREAL8 WeightEdge(const    tEdge &) const {return 1.0;}

             inline bool   InsideEdgeAndSucc(const    tEdge & anEdge) const 
             {
                    return InsideEdge(anEdge) && this->InsideVertex(anEdge.Succ());
             }
};

template <class TGraph>   class cAlgoPCC
{
     public :
	  typedef typename TGraph::tVertex  tVertex;
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

          tVertex * MakePCCGen
               (
                    TGraph &                       aGraph,
                    bool                           aModeMinPCC,
                    const std::vector<tVertex*> &  aVSeeds,
                    const cAlgo_ParamVG<TGraph> &  aParam,
                    const cAlgo_SubGr<TGraph>   &  aGoal,
                    std::vector<tVertex*> &        aVReached
               );
     private :
};



template <class TGraph>   
    typename TGraph::tVertex * cAlgoPCC<TGraph>::MakePCCGen
         (
              TGraph &                      aGraph,
              bool                          aModeMinPCC,
              const std::vector<tVertex*> & aVSeeds,
              const cAlgo_ParamVG<TGraph> & aParam,
              const cAlgo_SubGr<TGraph>   & aGoal,
              std::vector<tVertex*>       & aVReached
         )
{
    tVertex * aResult = nullptr;
    aVReached.clear();
    size_t aBitReached = aGraph.AllocBitTemp();
    size_t aBitOutHeap  = aGraph.AllocBitTemp();

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
                      if (aParam.InsideEdgeAndSucc(*anEdge))
                      {
                         tVertex & aNewVIn = anEdge->Succ();
                         if (!aNewVIn.BitTo1(aBitOutHeap))
                         {
                            tREAL8 aNewCost =  aParam.WeightEdge(*anEdge);
                            if (aModeMinPCC)  
                               aNewCost += aNewVOut->mAlgoCost;

                            if (!aNewVIn.BitTo1(aBitReached))
                            {
                               aVReached.push_back(&aNewVIn);
                               aNewVIn.BitTo1(aBitReached);
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

    for (auto aPtrV : aVReached)
    {
         aPtrV->SetBit0(aBitReached);
         aPtrV->SetBit0(aBitOutHeap);
    }

    aGraph.FreeBitTemp(aBitReached);
    aGraph.FreeBitTemp(aBitOutHeap);

    return aResult;
}



template class cAlgoPCC<cVG_Graph<int,int,int>> ;

//template <class TGraph>   class cAlgoPCC



// template <class TGraph>  class cAlgo_SubGr
// template <class TGraph>  class cAlgo_ParamVG

/* *********************************************************** */

class cBGG_Vertex
{
     public :
           cBGG_Vertex(cPt2di  aPt,tREAL8 aZ) :
               mPt (aPt),
               mZ  (aZ)
           {
           }

           cPt2di mPt;
           tREAL8 mZ;
     private :
  
};

class cBGG_EdgeOriented
{
     public :
           cBGG_EdgeOriented(tREAL8 aDz) : mDz (aDz) {}
           tREAL8 mDz;
     private :
};

class cBGG_EdgeSym
{
     public :
           cBGG_EdgeSym(tREAL8 aDist) : mDist (aDist) {}
           tREAL8 mDist;
     private :
};

class  cBGG_Graph : public cVG_Graph<cBGG_Vertex,cBGG_EdgeOriented,cBGG_EdgeSym>
{
     public :
          typedef cVG_Graph<cBGG_Vertex,cBGG_EdgeOriented,cBGG_EdgeSym> tGraph;
          typedef typename tGraph::tVertex                              tVertex;
          typedef  tVertex*                                             tPtrV;

          cBGG_Graph (cPt2di aSzGrid);
          tPtrV & VertOfPt(const cPt2di & aPt) {return mGridVertices.at(aPt.y()).at(aPt.x());}

          void Bench();
     private :

          cPt2di                               mSzGrid;
          cRect2                               mBox;
          std::vector<std::vector<tPtrV>>   mGridVertices;
          void AddEdge(const cPt2di&aP0,const cPt2di & aP1);
};


cBGG_Graph::cBGG_Graph (cPt2di aSzGrid) :
      cVG_Graph<cBGG_Vertex,cBGG_EdgeOriented,cBGG_EdgeSym>(),
      mSzGrid        (aSzGrid),
      mBox           (mSzGrid),
      mGridVertices  (mSzGrid.y(),std::vector<tVertex*>(mSzGrid.x(),nullptr))
{
      for (const auto & aPix : mBox)
      {
          cPt2di aPix4Num 
                 (
                   (aPix.y()%2) ? (mSzGrid.x()-aPix.x()-1) : aPix.x(),
                   aPix.y()
                 );
          tREAL8 aZ = mBox.IndexeLinear(aPix4Num);
          VertOfPt(aPix) = NewSom(cBGG_Vertex(aPix,aZ));
          // if (aPix.x() == 0) StdOut() << "===================\n";
          // StdOut() << aPix << aZ << "\n";
      }

      for (const auto & aPix : mBox)
      {
           AddEdge(aPix,aPix+cPt2di(0,1));
           AddEdge(aPix,aPix+cPt2di(1,0));
           AddEdge(aPix,aPix+cPt2di(1,1));
           AddEdge(aPix,aPix+cPt2di(-1,1));
      }
}

void cBGG_Graph::Bench()
{
      for (size_t aKV1=0 ; aKV1<NbVertex() ; aKV1++)
      {
          tVertex &   aV1 = VertexOfNum(aKV1);
          cPt2di aP1 = aV1.Attr().mPt;
          size_t aNbSucc=0;
          for (const auto aDPt : cRect2::BoxWindow(2))
          {
              cPt2di aP2 = aP1+aDPt;
              bool OkInside =  mBox.Inside(aP2) ;
              bool OkSucc =  OkInside &&  (NormInf(aDPt)==1);
              aNbSucc += OkSucc;
              if (OkSucc)
              {
                 tVertex &   aV2 = *VertOfPt(aP2);
                 const tEdge * anE12=  aV1.EdgeOfSucc(aV2);
                 MMVII_INTERNAL_ASSERT_bench(std::abs(anE12->AttrSym().mDist-Norm2(aP1-aP2))<1e-10,"Dist in cBGG_Graph");
                 MMVII_INTERNAL_ASSERT_bench(aV2.Attr().mZ-aV1.Attr().mZ==anE12->AttrOriented().mDz,"Dz in cBGG_Graph");
                 MMVII_INTERNAL_ASSERT_bench( (&anE12->Succ() ==  &aV2) ," Adrr in cBGG_Graph");
              }
              else if (OkInside)
              {
                 tVertex &   aV2 = *VertOfPt(aP2);
                 MMVII_INTERNAL_ASSERT_bench(aV1.EdgeOfSucc(aV2,SVP::Yes)==nullptr,"EdgeOfSucc in cBGG_Graph");
              }
          }
          MMVII_INTERNAL_ASSERT_bench(aNbSucc==aV1.EdgesSucc().size(),"NbSucc in cBGG_Graph");
      }
}

void cBGG_Graph::AddEdge(const cPt2di&aP1,const cPt2di & aP2)
{
    if ( (!mBox.Inside(aP1)) || (!mBox.Inside(aP2)) )
       return;

    tVertex * aV1 = VertOfPt(aP1);
    tVertex * aV2 = VertOfPt(aP2);

    tREAL8 aDist = Norm2(aV1->Attr().mPt-aV2->Attr().mPt);
    tREAL8 aDZ12 = aV2->Attr().mZ-aV1->Attr().mZ;

    cBGG_EdgeOriented aA12(aDZ12);
    cBGG_EdgeOriented aA21(-aDZ12);

    cBGG_EdgeSym  aASym(aDist);

    tGraph::AddEdge(*aV1,*aV2,aA12,aA21,aASym);
}


void BenchGrpValuatedGraph(cParamExeBench & aParam)
{
    if (! aParam.NewBench("GroupGraph")) return;

    cBGG_Graph   aGr(cPt2di(3,7));
    aGr.Bench();

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
          tEdge * EdgeOfSucc(size_t aSom2,bool Ok0) ;

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


void BenchGrpValuatedGraph(cParamExeBench & aParam)
{
    if (! aParam.NewBench("GroupGraph")) return;

    // cBenchGrid_GVG  aBGVG(cPt2di(10,20),100.0, 0.0, 0.0,0.0,cPt2di(3,3));
mNumSucc    // cBenchGrid_GVG  aBGVG(cPt2di(10,20),100.0, 0.01, 0.2,0.5,cPt2di(3,11));
    cBenchGrid_GVG  aBGVG(cPt2di(20,20),100.0, 0.00, 0.2,0.3,cPt2di(2,3));

    aParam.EndBench();
}
#endif

};

