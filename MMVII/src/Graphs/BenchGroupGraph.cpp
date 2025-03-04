#include "MMVII_nums.h"
#include "MMVII_util_tpl.h"
#include "MMVII_Geom3D.h"

#include "MMVII_Tpl_GraphAlgo_SPCC.h"
#include "MMVII_Tpl_GraphStruct.h"
#include "MMVII_Tpl_GraphAlgo_EnumCycles.h"
#include "MMVII_Interpolators.h"
#include "MMVII_Sensor.h"
#include "MMVII_PoseTriplet.h"
#include "MMVII_Tpl_GraphAlgo_Group.h"



namespace MMVII
{

/**  Classes on Bench "Group-Graph",  generating data on a grid (Grid-Group-Graph => G3)
 *
 *   For now it doesn test many things, it's rather an illustration on the functionnalities.
 *   The only thing tested is that using a graph with few outlayer an 0 noise on inlayers, the minimum
 *   spaning tree , computed with cycle-cost, use edges with no noise.
 */

/* ********************************************************* */
/*                                                           */
/*                     cBench_G3                             */
/*                                                           */
/* ********************************************************* */

template <class TGroup>  class cBench_G3  // Grid-Group-Graph
{
    public :
          typedef cGroupGraph<TGroup,cEmptyClass,cEmptyClass,cEmptyClass,cEmptyClass>     tGG;
          typedef typename tGG::tVertex   tVertex;
          typedef tVertex*                tVertPtr;

          class cBG3V
          {
               public :
                  TGroup     mValRef;
                  tVertex*   mVertex;
          };

          cBench_G3
          (
              const cPt2di & aSz,int aNbMin,int aNbMax,tREAL8 aPropOutLayer,
              tREAL8 aNoiseInLayer,tREAL8 aNoiseMinOutlayer,tREAL8 aNoiseMaxOutLayer
          );

          tGG &    GG() {return mGG;}

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
          tREAL8 mNoiseMinOutLayer;
          tREAL8 mNoiseMaxOutLayer;
};

template <class TGroup> 
    cBench_G3<TGroup>::cBench_G3
    (
        const cPt2di & aSz,
        int aNbMinE,int aNbMaxE,
        tREAL8 aPropOutLayer,tREAL8 aNoiseInLayer,
        tREAL8 aNoiseMinOutLayer, tREAL8 aNoiseMaxOutLayer
    ) :
       mSz                (aSz),
       mBox               (cPt2di(0,0),aSz),
       mGG                (true),
       mGridVals          (mSz.y(),std::vector<cBG3V>(mSz.x(),cBG3V())),
       mNbMinE            (aNbMinE),
       mNbMaxE            (aNbMaxE),
       mPropOutLayer      (aPropOutLayer),
       mNoiseInLayer      (aNoiseInLayer),
       mNoiseMinOutLayer  (aNoiseMinOutLayer),
       mNoiseMaxOutLayer  (aNoiseMaxOutLayer)
{
    for (const auto & aPix : mBox)
    {
        std::string aName = ToStr(aPix.x()) + "_" + ToStr(aPix.y());
        ValOfPt(aPix).mVertex = &mGG.AddVertex(cEmptyClass());
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
        TGroup aPerturb = TGroup::RandomSmallElem(mNoiseInLayer);

        if (! isInLayer)
           aPerturb= RandomGroupInInterval<TGroup>(mNoiseMinOutLayer,mNoiseMaxOutLayer);
        tREAL8 aNoise = aPerturb.Dist(TGroup::Identity());

        TGroup aRel_2To1 =  aRefRel_2To1 * aPerturb;


        cGG_1HypInit<TGroup,cEmptyClass> & aH =mGG.AddHyp(*(aVal0.mVertex),*(aVal1.mVertex),aRel_2To1,1.0,cEmptyClass());
        aH.mNoiseSim = std::abs(aNoise);
   }
}

template class cBench_G3<tRotR>;

void BenchGroupGraph(cParamExeBench & aParam)
{
    if (! aParam.NewBench("GroupGraph")) return;

    if (UserIsMPD())
    {
        cBench_G3<tRotR> aBG3
                     (
                         cPt2di(52,62),
                         3,10,
                         0.2,   //  Prop Out layer
                         0.05,0.0,0.5
                     );
         aBG3.GG().DoClustering(3,0.15);

         aBG3.GG().OneIterCycles(3,1.0,true);
         aBG3.GG().OneIterCycles(3,1.0,true);
         aBG3.GG().OneIterCycles(3,1.0,true);
         aBG3.GG().OneIterCycles(3,1.0,true);
         aBG3.GG().OneIterCycles(3,1.0,true);
         aBG3.GG().OneIterCycles(3,1.0,true);
         aBG3.GG().OneIterCycles(3,1.0,true);

         aBG3.GG().MakeMinSpanTree();
     }

     if (1)
     {

         cBench_G3<tRotR> aBG3
                     (
                         cPt2di(12,22),
                         3,10,
                         0.2,   //  Prop Out layer
                         0.00,0.2,0.5
                     );
         aBG3.GG().DoClustering(3,0.15);
         aBG3.GG().OneIterCycles(3,1.0,true);
         aBG3.GG().OneIterCycles(3,1.0,true);
         aBG3.GG().OneIterCycles(3,1.0,true);

         tREAL8 aNoiseMax = aBG3.GG().MakeMinSpanTree();

         StdOut() << "NM=" << aParam.Level() << " "<< aNoiseMax << "\n";
         // It's "highly probable" that with this random generation, the min span tree will
         // use only unoised edges, but not 100% (at least I cannot prouve it) so do it only for
         // 100 first, for which, empirically it was tested to be true
         if  (aParam.Level() <100)
         {
             MMVII_INTERNAL_ASSERT_bench(aNoiseMax==0,"Group graph MakeMinSpanTree ");
         }
     }

    aParam.EndBench();
}

}; // namespace MMVII
