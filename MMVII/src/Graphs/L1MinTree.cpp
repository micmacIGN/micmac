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
          cGrpValuated_OneAttrEdge(const TVal &,tREAL8 mCost=1.0);
          TVal     mVal; 
          tREAL8   mCost; 
};

template <class TVal>  cGrpValuated_OneAttrEdge<TVal>::cGrpValuated_OneAttrEdge(const TVal & aVal,tREAL8 aCost) :
     mVal  (aVal),
     mCost (aCost)
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
          cGrpValuatedAttrEdge(const TVal & aValues,tREAL8 aPrioriCost)  ;

          void   Add(const TVal & aValue,tREAL8 aPrioriCost) ;
          
          size_t  NbValues () const {return mValues.size();}
          const tOneAttr & KthVal (size_t aK) const {return mValues.at(aK);}
          tOneAttr & KthVal (size_t aK) {return mValues.at(aK);}

          const tSetAttr & Values () const {return mValues;}
          //  tSetAttr & Values () {return mValues;}

    private :
          tSetAttr          mValues;
};

template <class TVal>  
    cGrpValuatedAttrEdge<TVal>::cGrpValuatedAttrEdge()
{
}


template <class TVal>  
    cGrpValuatedAttrEdge<TVal>::cGrpValuatedAttrEdge(const TVal & aValue,tREAL8 aPrioriCost)  :
          cGrpValuatedAttrEdge<TVal>()
{
     Add(aValue,aPrioriCost);
}

template <class TVal> void  cGrpValuatedAttrEdge<TVal>::Add(const TVal & aValue,tREAL8 aCost) 
{
   mValues.push_back(tOneAttr(aValue,aCost));
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

          cGrpValuatedSom(const TVal & aValue) ;
          cGrpValuatedSom() ;  

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
          
    private :
          TVal              mCurValue;
          std::list<tEdge>  mLSucc;
          int               mNumReach;
          int               mHeapIndex;
};


template <class TVal>  cGrpValuatedSom<TVal>::cGrpValuatedSom(const TVal & aValue) : 
     mCurValue (aValue) ,
     mNumReach (-1)
{
}

template <class TVal>  cGrpValuatedSom<TVal>::cGrpValuatedSom() :
     cGrpValuatedSom(TVal::Identity())
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
          typedef cGrpValuatedEdge<TVal>          tEdge;
          typedef cGrpValuatedAttrEdge<TVal>      tAttr;
          typedef cGrpValuated_OneAttrEdge<TVal>  tOneAttr;
          typedef cGrpValuatedSom<TVal>           tSom ;

          cGrpValuatedGraph(int aNbSom,const TParam &,int aNbEdge=-1);
          void AddEdge(int aS1,int aS2,const TVal & aVal,tREAL8 aCostAPriori = 1.0);

          void MakeTriCostApriori();
          void MakeMinSpanTree();

          /// Return the relative position using  mCurValue
          TVal  Rel_2to1(int aS1,int aS2);

    protected :
          void  MakeTriCostApriori(size_t aS1,const tEdge & anE_AB);
          void  MakeTriCostApriori(std::vector<tREAL8> & aVDist,const TVal & aV1to2 ,const tEdge & anE23,const tEdge & anE31);

          const tAttr &  AttrOfEdge(const tEdge & anE) const {return mVAttr.at(anE.mNumAttr);}
          tAttr &  AttrOfEdge(const tEdge & anE)  {return mVAttr.at(anE.mNumAttr);}

          const std::list<tEdge> &  SuccOfSom(size_t aKSom) const {return mVSoms.at(aKSom).LSucc();}
          

          TParam              mParam;
          size_t              mNbSom;
          std::vector<tSom>   mVSoms;
          std::vector<tAttr>  mVAttr;
};

template <class TVal,class TParam> 
     cGrpValuatedGraph<TVal,TParam>::cGrpValuatedGraph(int aNbSom,const TParam & aParam,int aNbEdge) :
      mParam   (aParam),
      mNbSom   (aNbSom)
{
    for (int aK=0 ; aK<aNbSom ; aK++)
    {
         mVSoms.push_back(tSom());
    }
}

template <class TVal,class TParam>  
    void cGrpValuatedGraph<TVal,TParam>::AddEdge(int aS1,int aS2,const TVal & aVal, tREAL8 aCostAPriori)
{
    const tEdge * anEdge = mVSoms.at(aS1).EdgeOfSucc(aS2);
    if (anEdge)
    {
       tAttr&  anAttr =  AttrOfEdge(*anEdge); 
       TVal aValCor = anEdge->VRel2_to_1(aVal);
       anAttr.Add(aValCor,aCostAPriori);
       return;
    }
    mVSoms.at(aS1).AddEdge(aS2,mVAttr.size(),true);
    mVSoms.at(aS2).AddEdge(aS1,mVAttr.size(),false);
 
    mVAttr.push_back(cGrpValuatedAttrEdge(aVal,aCostAPriori));
}

template <class TVal,class TParam>  TVal  cGrpValuatedGraph<TVal,TParam>::Rel_2to1(int aS1,int aS2)
{
    // Ori_2->1  = Ori_G->1  o Ori_2->G    ( PL2 -> PG -> PL1)
    return  mVSoms.at(aS1).CurVal().MapInverse() *  mVSoms.at(aS2).CurVal();
}


template <class TVal,class TParam>  
    void cGrpValuatedGraph<TVal,TParam>::MakeTriCostApriori()
{
    for (size_t aSom_A=0 ; aSom_A<mNbSom ; aSom_A++)
    {
         for (const auto &  aSucc_AB :  SuccOfSom(aSom_A))
         {
              // no need to do it 2 way
              if (aSucc_AB.mDirInit)
              {
                 MakeTriCostApriori(aSom_A,aSucc_AB);
              }
         }
    }
}

template <class TVal,class TParam>  
          void  cGrpValuatedGraph<TVal,TParam>::MakeTriCostApriori(size_t aSom_A,const tEdge & anE_AB)
{
   size_t aSom_B = anE_AB.mSucc;
   tAttr & aVecAttr_AB  = AttrOfEdge(anE_AB);

   for (size_t aKVal=0 ; aKVal<aVecAttr_AB.NbValues() ; aKVal++)
   {
        tOneAttr &  aAttr_AB = aVecAttr_AB.KthVal(aKVal);
        // note we compute  "Map/Pose"  of A relatively 2 B , because in 3-loop it will be in other way
        TVal aV_A2B   = anE_AB.VRel1_to_2(aAttr_AB.mVal);

        //  VDist will store the distances between aV1to2 and its different evaluation by loops
        std::vector<tREAL8> aVDist;

        // -1- First basic contribution, in case there is multiple evals (i.e its ~ a loop size 2)
        // parse all atribud
        for (size_t aKV2=0 ; aKV2<aVecAttr_AB.NbValues() ; aKV2++)
        {
            if (aKV2 !=aKVal) 
            {
                tOneAttr &  aBis_Attr_AB = aVecAttr_AB.KthVal(aKV2);
                TVal aBis_V_A2B  = anE_AB.VRel1_to_2(aBis_Attr_AB.mVal);

                aVDist.push_back(mParam.Dist(aV_A2B,aBis_V_A2B));
            }
        }
        // -2-  now more sophisticated contribution we look for loop size 3
        for (const auto &  aSucc_BC :  SuccOfSom(aSom_B))
        {
            size_t aSom_C = aSucc_BC.mSucc;
            for (const auto &  aSucc_CA :  SuccOfSom(aSom_C) )
            {
                if (aSucc_CA.mSucc==aSom_A) // Ok,  we get a triangle loop ABC
                {
                    MakeTriCostApriori(aVDist,aV_A2B,aSucc_BC,aSucc_CA);
                }
            }
        }

        std::sort(aVDist.begin(),aVDist.end());
   }
}

template <class TVal,class TParam>  
   void  cGrpValuatedGraph<TVal,TParam>::MakeTriCostApriori
         (
             std::vector<tREAL8> & aVDist,
             const TVal & aV_A2B ,
             const tEdge & anE_BC,
             const tEdge & anE_CA
         )
{
    const tAttr & aVecAttr_BC  = AttrOfEdge(anE_BC);
    const tAttr & aVecAttr_CA  = AttrOfEdge(anE_CA);

    for (const auto & aAttr_BC : aVecAttr_BC.Values())
    {
        TVal aV_C2B  = anE_BC.VRel2_to_1(aAttr_BC.mVal);
        for (const auto &  aAttr_CA : aVecAttr_CA.Values())
        {
            TVal aV_A2C  = anE_CA.VRel2_to_1(aAttr_CA.mVal);
            //  aV_C2B * aV_A2C  (PA) = aV_C2B (PC) = PB = aV_A2B (PA)
            //  So the equation is  aV_C2B * aV_A2C  == aV_A2B
            tREAL8 aDist = mParam.Dist(aV_A2B, aV_C2B * aV_A2C);
StdOut() << "DDDDD " << aDist << "\n";
            aVDist.push_back(aDist);
        }
    }
StdOut() << "===================\n";
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
         tREAL8 aAmplNoise = (RandUnif_0_1()<mPropOutLayer) ? mNoiseOutlayers : mNoiseInlayers;
         aAmplNoise /= std::sqrt(2.0);

         tRotR aR_B2A  = tRotR::RandomRot(aAmplNoise) * aR_B2A_Ref * tRotR::RandomRot(aAmplNoise);

         AddEdge(aSom_A,aSom_B,aR_B2A);
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
   this->MakeTriCostApriori();
StdOut() << "TRI-COST DONE\n";
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

    // cBenchGrid_GVG  aBGVG(cPt2di(10,20),100.0, 0.0, 0.0,0.0,cPt2di(3,6));
    cBenchGrid_GVG  aBGVG(cPt2di(10,20),100.0, 0.0, 0.2,0.3,cPt2di(3,3));

    aParam.EndBench();
}

};

