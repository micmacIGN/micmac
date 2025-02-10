#include "MMVII_nums.h"
#include "MMVII_util_tpl.h"
#include "MMVII_Geom2D.h"
#include "MMVII_Geom3D.h"

#include "MMVII_TplHeap.h"

namespace MMVII
{

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
          std::string  mAttrib;
          tREAL8       mCost; 
};

template <class TVal>  cGrpValuated_OneAttrEdge<TVal>::cGrpValuated_OneAttrEdge(const TVal & aVal,std::string aAttrib,tREAL8 aCost) :
     mVal    (aVal),
     mAttrib (aAttrib),
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

    private :
          tSetAttr          mValues;
          int               mIndMinCost;
};

template <class TVal>  
    cGrpValuatedAttrEdge<TVal>::cGrpValuatedAttrEdge() :
        mIndMinCost (-1)
{
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
          const tEdge * EdgeOfSucc(size_t aSom2) const;

          /**  add a successor for s2, with attribute at adr aNumAttr, DirInit if the attribute is
               relative to way  "S1->S2" (else it's "S2->S1")
          */
          void AddEdge(int aS2,int aNumAttr,bool DirInit);

          int & HeapIndex() {return mHeapIndex;}
          const std::list<tEdge> &  LSucc() const {return mLSucc;}

          const TVal &  CurVal() const {return mCurValue;}
          void  SetCurVal(const TVal& aNewV) {mCurValue= aNewV;}

          const tREAL8 &  HeapCost() const {return mHeapCost;}
          void  SetHeapCost(tREAL8 aNewV)  { mHeapCost=aNewV;}

          int   NumRegion() const {return mNumRegion;}
          void  SetNumRegion(int aNewV) {mNumRegion= aNewV;}
          
          int   NumSom() const {return mNumSom;}

          void SetHeapFather(int aNewV) {mNumHeapFather = aNewV;}
    private :
          TVal              mCurValue;
          int               mNumSom;
          int               mNumHeapFather;
          tREAL8            mHeapCost;
          std::list<tEdge>  mLSucc;
          int               mNumRegion;
          int               mHeapIndex;
};


template <class TVal>  cGrpValuatedSom<TVal>::cGrpValuatedSom(const TVal & aValue,int aNumSom) : 
     mCurValue       (aValue) ,
     mNumSom         (aNumSom),
     mNumHeapFather  (-1),
     mHeapCost       (1e10),
     mNumRegion      (-1)
{
}

template <class TVal>  cGrpValuatedSom<TVal>::cGrpValuatedSom(int aNumSom) :
     cGrpValuatedSom(TVal::Identity(),aNumSom)
{
}

template <class TVal>  
     const cGrpValuatedEdge<TVal>  * cGrpValuatedSom<TVal>::EdgeOfSucc(size_t aSucc) const
{
   for (const auto &  anEdge : mLSucc)
       if (anEdge.mSucc == aSucc)
          return & anEdge;
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
        static tREAL8 Dist(const tRotR & aR1,const tRotR& aR2)
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
          void MakeMinSpanTree();

          /// Return the relative position using  mCurValue
          TVal  Rel_2to1(int aS1,int aS2);

    protected :
          class  cLoop
          {
              public :
                  TVal   mVal;
                  size_t mSom;
                  int    mStackI;

                  cLoop (const TVal& aVal,size_t aSom,int aStackI ):  mVal(aVal), mSom(aSom), mStackI (aStackI) {}
          };

          void  MakeLoopCostApriori(size_t aS1,const tEdge&, tOneAttr &);


          const tAttr &  AttrOfEdge(const tEdge & anE) const {return mVAttr.at(anE.mNumAttr);}
          tAttr &  AttrOfEdge(const tEdge & anE)  {return mVAttr.at(anE.mNumAttr);}

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
    const tEdge * anEdge = mVSoms.at(aS1).EdgeOfSucc(aS2);
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

template <class TVal,class TParam>  TVal  cGrpValuatedGraph<TVal,TParam>::Rel_2to1(int aS1,int aS2)
{
    // Ori_2->1  = Ori_G->1  o Ori_2->G    ( PL2 -> PG -> PL1)
    return  mVSoms.at(aS1).CurVal().MapInverse() *  mVSoms.at(aS2).CurVal();
}



template <class TVal,class TParam>  
    void  cGrpValuatedGraph<TVal,TParam>::MakeMinSpanTree()
{
    tCmpHeap   aHCmp;
    cIndexedHeap<tSom*,tCmpHeap,tParamHeap>  aHeap(aHCmp);

    aHeap.Push(&mVSoms.at(0));
    mVSoms.at(0).SetNumRegion(0);
  
    tSom * aNewSom = nullptr;
    while (aHeap.Pop(aNewSom))
    {
         StdOut() << "NEWWWWW " << aNewSom->NumSom() << "\n";
         for (const auto &  aSucc :  SuccOfSom(*aNewSom))
         {
             tSom & aSom2Update = S2OfEdge(aSucc);
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


    for (const auto & [aName,aStat] : mStats)
    {
        StdOut() << " A=" << aName;
        for (const auto aProp : {0.01,0.1,0.5,0.9,0.99})
            StdOut() <<  " " << aProp << "->" << aStat.ErrAtProp(aProp) ;
        StdOut() << " \n";
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
                           tREAL8 aDist2Id = mParam.Dist(TheIdent, aV_Y2A);
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
    mStats[aAttr_AB.mAttrib].Add(aSumCost);
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
    tRotR aR_B2A_Ref  = Rel_2to1(aSom_A,aSom_B);

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
      aSom.SetCurVal(tRotR::RandomRot(aAmplGlob));
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
StdOut() << "LOOP-COST DONE\n";
   this->MakeMinSpanTree();
StdOut() << "MIN SPAN TREE DONE\n";
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
    // cBenchGrid_GVG  aBGVG(cPt2di(10,20),100.0, 0.01, 0.2,0.5,cPt2di(3,11));
    cBenchGrid_GVG  aBGVG(cPt2di(7,7),100.0, 0.02, 0.2,0.3,cPt2di(2,3));

    aParam.EndBench();
}

};

