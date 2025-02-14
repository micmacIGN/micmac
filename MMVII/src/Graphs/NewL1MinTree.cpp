#include "MMVII_util_tpl.h"
#include "MMVII_Bench.h"
#include "cMMVII_Appli.h"
#include "MMVII_Images.h"

#include "MMVII_Tpl_GraphStruct.h"
#include "MMVII_Tpl_GraphAlgo_SPCC.h"


namespace MMVII
{




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
           cBGG_EdgeOriented(tREAL8 aDz) : 
		   mDz (aDz) 
	   {}
           tREAL8 mDz;
     private :
};

class cBGG_EdgeSym
{
     public :
           cBGG_EdgeSym(tREAL8 aDist) : 
                mDist     (aDist) ,
                mRanCost  (RandUnif_C()),
                mIsOk     (true)
           {}
           tREAL8 mDist;
           tREAL8 mRanCost;
           bool   mIsOk;
     private :
};

class  cBGG_Graph : public cVG_Graph<cBGG_Vertex,cBGG_EdgeOriented,cBGG_EdgeSym>
{
     public :
          typedef cVG_Graph<cBGG_Vertex,cBGG_EdgeOriented,cBGG_EdgeSym> tGraph;
          typedef typename tGraph::tVertex                              tVertex;
          typedef  tVertex*                                             tPtrV;
          typedef  cAlgoSP<cBGG_Graph>                                  tAlgoSP;
          typedef  typename tAlgoSP::tSetPairVE                         tSetPairVE;
          typedef  cAlgoCC<cBGG_Graph>                                  tAlgoCC;
          typedef  tAlgoSP::tForest                                     tForest;
          typedef  cAlgo_ParamVG<cBGG_Graph>                            tParamA;


          cBGG_Graph (cPt2di aSzGrid);
          tPtrV & VertOfPt(const cPt2di & aPt) {return mGridVertices.at(aPt.y()).at(aPt.x());}

          void Bench_FoncElem();
          void Bench_ShortestPath();
          void Bench_ConnectedComponent(tVertex * aV0,tVertex * aV1);
          void Bench_MinSpanTree(tVertex * aSeed);
     private :
          void Bench_ShortestPath(tVertex * aV0,tVertex * aV1,int aMode);
          void Check_MinSpanTree(const tSetPairVE& aSet,const tParamA& );

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

void cBGG_Graph::Bench_FoncElem()
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

void cBGG_Graph::Bench_ShortestPath(tVertex * aV0,tVertex * aV1,int aMode)
{
    class cN4Cnx : public  cAlgo_ParamVG<cBGG_Graph>
    {
       public :
         bool   InsideEdge(const tVertex & aV1,const    tEdge & anE) const override 
		{ return Norm1(aV1.Attr().mPt-anE.Succ().Attr().mPt)==1; }
    };
    class cWDZ2 : public  cAlgo_ParamVG<cBGG_Graph>
    {
       public :
             tREAL8 WeightEdge(const tVertex &,const    tEdge & anE) const override
		{ return Square(anE.AttrOriented().mDz);}
    };



    cAlgo_ParamVG<cBGG_Graph> aParam0;
    cAlgo_ParamVG<cBGG_Graph> * aParam = & aParam0;

    cN4Cnx  aParam1;
    if (aMode==1) aParam = & aParam1;

    cWDZ2 aParam2;
    if (aMode==2) aParam = & aParam2;

    tVertex *  aTarget =  tAlgoSP::ShortestPath_A2B(*this,*aV0,*aV1,*aParam);

    int aDTh = NormInf(aV0->Attr().mPt -aV1->Attr().mPt);
    if (aMode==1) aDTh = Norm1(aV0->Attr().mPt -aV1->Attr().mPt);
    if (aMode==2) aDTh = std::abs(aV0->Attr().mZ -aV1->Attr().mZ);


    // StdOut() << aDTh << " " << aV1->AlgoCost() << "\n";
     MMVII_INTERNAL_ASSERT_bench(aDTh==aV1->AlgoCost(),"Cost in shortest path"); 
     MMVII_INTERNAL_ASSERT_bench(aTarget==aV1,"Target in shortest path");

     std::vector<tVertex*> aPath = aTarget->BackTrackFathersPath();
     MMVII_INTERNAL_ASSERT_bench(aV0==aPath.back(),"seed in BackTrackFathersPath");
     MMVII_INTERNAL_ASSERT_bench(aTarget->AlgoCost()+1==aPath.size(),"size in BackTrackFathersPath");
     for (size_t aKV=1 ; aKV<aPath.size() ; aKV++)
     {
          int aD = NormInf(aPath.at(aKV-1)->Attr().mPt-aPath.at(aKV)->Attr().mPt);
          if (aMode==1)  aD = Norm1(aPath.at(aKV-1)->Attr().mPt-aPath.at(aKV)->Attr().mPt);
          MMVII_INTERNAL_ASSERT_bench(aD==1,"Succ in BackTrackFathersPath");
     }
}

void cBGG_Graph::Bench_ConnectedComponent(tVertex * aSeed,tVertex * aV1)
{
    class cGrVert : public  cAlgo_ParamVG<cBGG_Graph>
    {
       public :
         bool   InsideEdge(const tVertex & aV1,const    tEdge & anE) const override 
		{ return aV1.Attr().mPt.x()==anE.Succ().Attr().mPt.x(); }
    };

    cGrVert aParamVert;

    std::vector<tVertex*>  aCC = tAlgoCC::ConnectedComponent(*this,*aSeed,aParamVert);

    MMVII_INTERNAL_ASSERT_bench(aCC.size()==(size_t)mSzGrid.y(),"Size in Bench_ConnectedComponent");
    for (auto aVertex : aCC)
        MMVII_INTERNAL_ASSERT_bench(aSeed->Attr().mPt.x()==aVertex->Attr().mPt.x(),"x-diff in Bench_ConnectedComponent");

    std::vector<tVertex *> aVecSeed;
    cPt2di aP0 = aSeed->Attr().mPt;
    cPt2di aP1 = aV1->Attr().mPt;
    MakeBox(aP0,aP1);
    for (int aX=aP0.x() ; aX<aP1.x() ; aX++)
        for (int aY=aP0.y() ; aY<aP1.y() ; aY++)
            aVecSeed.push_back(VertOfPt(cPt2di(aX,aY)));
 
    std::list< std::vector<tVertex*>> aVecCC =  tAlgoCC::Multiple_ConnectedComponent(*this,aVecSeed,aParamVert);

    int aThNbCC = (aP1.y()==aP0.y()) ? 0 :(aP1.x()-aP0.x());
    MMVII_INTERNAL_ASSERT_bench((int)aVecCC.size() == aThNbCC,"NbCC in Multiple_ConnectedComponent");

    aVecCC = tAlgoCC::All_ConnectedComponent(*this,aParamVert);

    MMVII_INTERNAL_ASSERT_bench((int)aVecCC.size() == mSzGrid.x(),"NbCC in All_ConnectedComponent");
    for (const auto & aCC : aVecCC)
        MMVII_INTERNAL_ASSERT_bench((int) aCC.size()==mSzGrid.y(),"Size in All_ConnectedComponent");
}

void cBGG_Graph::Check_MinSpanTree(const tSetPairVE& aSetPair,const tParamA& aParam )
{
     class cASymOk : public  cAlgo_ParamVG<cBGG_Graph>
     {
       public :
         bool   InsideEdge(const tVertex & ,const    tEdge & anE) const override {return anE.AttrSym().mIsOk;}
     };

     for (const auto & aPtrAttrSym : this->AllAttrSym())
        aPtrAttrSym-> mIsOk = false;
     for (const auto & [aV1,anE] : aSetPair )
         anE->AttrSym().mIsOk = true;

     for (const auto & [aV1,anETree] : aSetPair )
     {
         anETree->AttrSym().mIsOk = false;

         std::vector<tVertex*>  aCC1 = tAlgoCC::ConnectedComponent(*this,*aV1,cASymOk());
         std::vector<tVertex*>  aCC2 = tAlgoCC::ConnectedComponent(*this,anETree->Succ(),cASymOk());

         MMVII_INTERNAL_ASSERT_bench(aCC1.size() + aCC2.size()==aSetPair.size()+1,"Sizes in Check_MinSpanTree");

         tREAL8 aCostCut =anETree->AttrSym().mRanCost;
         for (const auto & aV1 : aCC1)
         {
              for (const auto & aV2 : aCC2)
              {
                  const tEdge * anE12 = aV1->EdgeOfSucc(*aV2,SVP::Yes);
                  if (anE12 &&  aParam.InsideEdge(*aV1,*anE12))
                  {
                      MMVII_INTERNAL_ASSERT_bench(anE12->AttrSym().mRanCost>= aCostCut,"Sizes in Check_MinSpanTree");
                  }
              }
         }
         

         anETree->AttrSym().mIsOk = true;
     }

     for (const auto & aPtrAttrSym : this->AllAttrSym())
        aPtrAttrSym-> mIsOk = true;
}


void cBGG_Graph::Bench_MinSpanTree(tVertex * aSeed)
{
    class cWRanCost : public  cAlgo_ParamVG<cBGG_Graph>
    {
       public :
             tREAL8 WeightEdge(const tVertex &,const    tEdge & anE) const override
		{ return anE.AttrSym().mRanCost;}
    };

    class cRC_Thr : public  cWRanCost
    {
       public :
             bool InsideEdge(const tVertex &,const    tEdge & anE) const override
             { 
                  return anE.AttrSym().mRanCost > mThreshold;
             }
             cRC_Thr(const tREAL8 aThrs) : mThreshold (aThrs) {}
             tREAL8  mThreshold;
    };



 
    for (const auto & aPtrAttrSym : this->AllAttrSym())
        aPtrAttrSym-> mRanCost = RandUnif_C();

    tSetPairVE aSetPair = tAlgoSP::MinimumSpanninTree(*this,*aSeed,cWRanCost());
    MMVII_INTERNAL_ASSERT_bench((int)aSetPair.size()==(mBox.NbElem()-1),"Size in All_ConnectedComponent");
    Check_MinSpanTree(aSetPair,cWRanCost());

    cRC_Thr  aRC(1.0 - 2*std::pow(RandUnif_0_1(),2.0));
    tForest aForest;
    tAlgoSP::MinimumSpanningForest(aForest,*this,this->AllVertices(), aRC);

    size_t aNbEdge = 0;
    for (const auto & [aVert,aVPair] : aForest)
    {
        Check_MinSpanTree(aVPair,aRC);
        aNbEdge += aVPair.size();
    }

    MMVII_INTERNAL_ASSERT_bench(aForest.size()+aNbEdge== (size_t)mBox.NbElem(),"Edge and CC in MinimumSpanningForest");
}

void cBGG_Graph::Bench_ShortestPath()
{
     for (int aKTime=0 ; aKTime<2*mBox.NbElem()  ; aKTime++)
     {
	 tVertex * aV0 =  VertOfPt(mBox.GeneratePointInside());
	 tVertex * aV1 =  VertOfPt(mBox.GeneratePointInside());

	 Bench_ConnectedComponent(aV0,aV1);

         Bench_ShortestPath(aV0,aV1,0);
         Bench_ShortestPath(aV0,aV1,1);
         Bench_ShortestPath(aV0,aV1,2);

         Bench_MinSpanTree(aV0);
      }
}


void BenchGrpValuatedGraph(cParamExeBench & aParam)
{
    if (! aParam.NewBench("GroupGraph")) return;

    cBGG_Graph   aGr(cPt2di(8,13));
    aGr.Bench_FoncElem();
    aGr.Bench_ShortestPath();

    aParam.EndBench();
}



};

