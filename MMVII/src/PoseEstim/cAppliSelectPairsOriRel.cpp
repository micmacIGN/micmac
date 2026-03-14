#include "MMVII_PoseRel.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_HeuristikOpt.h"
#include "MMVII_DeclareAllCmd.h"

#include "MMVII_Tpl_GraphAlgo_Group.h"
#include "MMVII_Tpl_GraphAlgo_SPCC.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Tpl_GraphAlgo_EnumCycles.h"

namespace MMVII
{






/// Class for vertex attribute
class cASPO_AV
{
   public :
        cASPO_AV(const std::string& aNameIm) ;

        std::string mNameIm;
};

cASPO_AV::cASPO_AV(const std::string& aNameIm) :
    mNameIm (aNameIm)
{
}


/// Class for oriented attribute : nothing
class cASPO_Or
{
   public :

};

/// Class for  symetric attribute : nothing
class cASPO_Sym
{
   public :
      cSetHomogCpleIm mCpleH;
      tREAL8          mWeight;
      int             mNbSel;
};

typedef cVG_Graph<cASPO_AV,cASPO_Or,cASPO_Sym> tGrASPO ;

typedef typename tGrASPO::tVertex tVertASPO;
typedef typename tGrASPO::tEdge   tEdgeASPO;

class cSubGrASPO : public cAlgo_ParamVG<tGrASPO>
{
   public :

     bool   InsideEdge(const    tEdgeASPO &anE)  const override
     {
         return ! anE.SymBitTo1(mFlag);
     }


     tREAL8 WeightEdge(const    tEdge & anE) const override
     {
         return anE.AttrSym().mWeight;
     }

     size_t mFlag;
};

class cAppli_SelectPairOriRel : public cMMVII_Appli,
                                public cActionOnCycle<tGrASPO>
{
     public :
        cAppli_SelectPairOriRel(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
        // std::vector<std::string>  Samples() const override;

        void  OnCycle(const cAlgoEnumCycle<tGrASPO>&) override;

     private :
      //  std::map<std::string,cExtSet<cTripletName>>

        void AddTriplet(const std::string&,const std::string&,const std::string&);
        void AddTriplet(const std::vector<tEdgeASPO*> & aPath);

        void  Add1MinSpanTrees();
        void  AddAllMinSpanTrees();
        void  AddKBestSom(tVertASPO*);


        cPhotogrammetricProject   mPhProj;
        cSetHomogCpleIm           mCpleH;

        std::string               mPatIm;
        tGrASPO                   mGr;
        cSubGrASPO                mSubGrMST;
        cAlgo_ParamVG<tGrASPO>    mSubGrAll;
      //  bool                      mShow;
        int                       mNbMinHom;
        int                       mNbMinSpanTree;
        int                       mNbLines;
        bool                      mLinesIsCirc;
        bool                      mAllPair;
        int                       mNbKBestSom;

        size_t                    mBitTmp0 ;

        std::vector<tEdgeASPO*>   mVEdges;
        std::vector<tVertASPO*>   mVertices;


        void AddEdge(tEdgeASPO&);
        int                     mNbEdgeSel;
        int                     mNbTestAddE;
        int                     mNbCycleTot;
        int                     mNbCycleFull;
        int                     mNbCycleOK2;
        int                     mNbTriExport;
        std::map<std::string,tSet3N>   mMapSet3;
        std::vector<tEdgeASPO*>        mVedgeCompC;
};

cAppli_SelectPairOriRel::cAppli_SelectPairOriRel(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli   (aVArgs,aSpec),
    mPhProj        (*this),
    mNbMinHom      (10),
    mNbMinSpanTree (4),
    mNbLines       (3),
    mLinesIsCirc   (true),
    mAllPair       (false),
    mNbKBestSom    (5),
    mNbEdgeSel     (0),
    mNbTestAddE    (0),
    mNbCycleTot    (0),
    mNbCycleFull   (0),
    mNbCycleOK2    (0),
    mNbTriExport   (0)
{
}





cCollecSpecArg2007 & cAppli_SelectPairOriRel::ArgObl(cCollecSpecArg2007 & anArgObl)
{

    return     anArgObl
              << Arg2007(mPatIm,"Pattern of images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
              << mPhProj.DPOriRel().ArgDirOutMand()
         ;

}

cCollecSpecArg2007 & cAppli_SelectPairOriRel::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return       anArgOpt
        <<  mPhProj.DPTieP().ArgDirInOpt()
        <<  mPhProj.DPGndPt2D().ArgDirInOpt()
        << AOpt2007(mNbMinSpanTree,"NbMSTree","Number of minimal spanning trees",{eTA2007::HDV})
        << AOpt2007(mNbKBestSom,"NbKBest","Number of k best point",{eTA2007::HDV})
        << AOpt2007(mNbLines,"NbLines","Number of image succ in lines",{eTA2007::HDV})
        << AOpt2007(mAllPair,"AllPair","Add all pairs",{eTA2007::HDV})

   ;
}


void cAppli_SelectPairOriRel::AddEdge(tEdgeASPO& anE)
{
    if (anE.AttrSym().mNbSel==0)
        mNbEdgeSel++;
    anE.AttrSym().mNbSel++;
    mNbTestAddE++;
}


void  cAppli_SelectPairOriRel::Add1MinSpanTrees()
{
   cAlgoSP<tGrASPO>     anAlgo;
   cVG_Forest<tGrASPO> aForest(mGr);
   anAlgo.MinimumSpanningForest(aForest,mGr,mVertices,mSubGrMST);

   for (const auto & aTree : aForest.VTrees())
   {
       for (const auto & aPtrE : aTree.Edges())
       {
          aPtrE->SymSetBit1(mBitTmp0);
          AddEdge(*aPtrE);
       }
   }
}

void  cAppli_SelectPairOriRel::AddAllMinSpanTrees()
{
  for (int aNbMSTree = 0 ; aNbMSTree<mNbMinSpanTree; aNbMSTree++)
  {
      Add1MinSpanTrees();
  }

  for (auto & anEdge : mVEdges)
      anEdge->SymSetBit0(mBitTmp0);

}

void cAppli_SelectPairOriRel::AddTriplet(const std::string& aN1,const std::string& aN2,const std::string& aN3)
{
    cTripletName a3(aN1,aN2,aN3);

    auto & aSet3 =  mMapSet3[a3.mNames[0]];
    if (!aSet3.In(a3))
        mNbTriExport++;

    aSet3.Add(a3);

}
void cAppli_SelectPairOriRel::AddTriplet(const std::vector<tEdgeASPO*> & aPath )
{
    AddTriplet
    (
         aPath.at(0)->Succ().Attr().mNameIm,
         aPath.at(1)->Succ().Attr().mNameIm,
         aPath.at(2)->Succ().Attr().mNameIm
    );
}


void  cAppli_SelectPairOriRel::OnCycle(const cAlgoEnumCycle<tGrASPO>& anAEnumC)
{
   mNbCycleTot++;

   const std::vector<tEdgeASPO*> & aPath = anAEnumC.CurPath();
   if (aPath.size() !=3)
       return;


      int aNbIn =0;
      cBoundVals<tREAL8> aBounds;
      tREAL8 aMinW = 1e8;
      tREAL8 aMaxW = 0.0;

      tEdgeASPO * anENew = nullptr;
      for (const auto & anEdgPtr : aPath)
      {
          if (anEdgPtr->AttrSym().mNbSel !=0 )
          {
             aNbIn++;
             aBounds.Add(anEdgPtr->AttrSym().mWeight);
             UpdateMin(aMinW,anEdgPtr->AttrSym().mWeight);
             UpdateMax(aMaxW,anEdgPtr->AttrSym().mWeight);
          }
          else
          {
              anENew = anEdgPtr;
          }
      }
      if (aNbIn==3)
      {
         mNbCycleFull ++;
         AddTriplet(aPath);
      }

      if (aNbIn !=2)
         return;

      // If the edge is not the "worst" of the cycle
      if (anENew->AttrSym().mWeight > aBounds.VMax())
          return;


   mNbCycleOK2++;
   AddTriplet(aPath);
   mVedgeCompC.push_back(anENew);
}

void  cAppli_SelectPairOriRel::AddKBestSom(tVertASPO* aVertex)
{
    std::vector<tEdgeASPO*> aVEdge =  aVertex->EdgesSucc() ;

    std::sort
    (
        aVEdge.begin(),
        aVEdge.end(),
        [](const auto & anE1,const auto & anE2){return anE1->AttrSym().mWeight<anE2->AttrSym().mWeight;}
    );

    for (int aKE=0 ; aKE<std::min((int)aVEdge.size(),mNbKBestSom); aKE++)
        AddEdge(*aVEdge.at(aKE));
}


int cAppli_SelectPairOriRel::Exe()
{
   // mTimeSegm = mShow ? new cTimerSegm(this) : nullptr ;
    mPhProj.FinishInit();

    mBitTmp0 =   mGr.AllocBitTemp();
    mSubGrMST.mFlag = mBitTmp0;

    std::vector aVstr = VectMainSet(0);

    for (const auto & aName : aVstr)
    {
        mGr.NewVertex(cASPO_AV(aName));
    }
    mVertices = mGr.AllVertices();

    for ( size_t aKV1 = 0 ; aKV1<mGr.NbVertex() ; aKV1++)
    {
        tVertASPO &  aV1 = mGr.VertexOfNum(aKV1);
        for ( size_t aKV2 = aKV1+1 ; aKV2<mGr.NbVertex() ; aKV2++)
        {
            tVertASPO &  aV2 = mGr.VertexOfNum(aKV2);

            cASPO_Sym anAS;
            anAS.mNbSel = 0;
            int aNbInit;
            mPhProj.ReadHomolMultiSrce(aNbInit,anAS.mCpleH,aV1.Attr().mNameIm,aV2.Attr().mNameIm);

            if (aNbInit==0)
            {
                MMVII_UnclasseUsEr("No homologous source indicated");
            }

            if ((int) anAS.mCpleH.NbH()  >= mNbMinHom)
            {
                cASPO_Or anAOr;
                mGr.AddEdge(aV1,aV2,anAOr,anAOr,anAS);
            }
        }
    }



    mVEdges = mGr.AllEdges_DirInit();

    // Make a weighting decreasing with number of homologous
    for (auto & anE : mVEdges)
    {
         anE->AttrSym().mWeight = 1.0/ (1.0 + anE->AttrSym().mCpleH.NbH());
    }

    // --------------  Add Lines ---------------------------------------------
    if (mAllPair)
    {
        mNbLines = mGr.NbVertex();
        mLinesIsCirc = true;
    }
    for ( int aKV1 = 0 ; aKV1<(int)mGr.NbVertex() ; aKV1++)
    {
        for (int aKV2Glob=aKV1+1; aKV2Glob<=aKV1+mNbLines; aKV2Glob++)
        {
            int aKV2 = aKV2Glob % mGr.NbVertex();
            if (mLinesIsCirc || (aKV2>aKV1))
            {
                tEdgeASPO * anEdge = mVertices.at(aKV1)->EdgeOfSucc(*mVertices.at(aKV2),true);
                if (anEdge)
                    AddEdge(*anEdge);
            }

        }
    }

    StdOut() << " LINE NBE= " << mNbEdgeSel  << " NBTest=" << mNbTestAddE << "\n";

    // ------------------------  Add Min Spaning trees ---------------------------
    AddAllMinSpanTrees();

    StdOut() << " TREE NBE= " << mNbEdgeSel  << " NBTest=" << mNbTestAddE << "\n";

    // ------------------------  Add K best succ of each vertex  ---------------------------
    for (auto aV: mVertices)
        AddKBestSom(aV);
    StdOut() << " KBEST NBE= " << mNbEdgeSel  << " NBTest=" << mNbTestAddE << "\n";

    // -------------------- Cycle --------------------------
    cAlgoEnumCycle<tGrASPO> aAEnumC(mGr,*this,mSubGrAll,3);
    aAEnumC.ExplorateAllCycles();
    for (const auto & aPtrE : mVedgeCompC)
    {
        aPtrE->AttrSym().mNbSel++;
    }

    tNameRel  aSetOfPair;
    for (const auto & anEdge : mVEdges)
    {
        if (anEdge->AttrSym().mNbSel)
        {
             std::string aN1 = anEdge->VertexInit().Attr().mNameIm;
             std::string aN2 = anEdge->Succ().Attr().mNameIm;
             aSetOfPair.Add(tNamePair(aN1,aN2));
        }
    }

    SaveInFile(aSetOfPair,mPhProj.OriRel_NamePairsOfAllImages(false));

    for (const auto & aNameIm : aVstr)
    {
       tSet3N aSet3 = mMapSet3[aNameIm];
       SaveInFile(aSet3,mPhProj.OriRel_NameOriAllTripletsOf1Image(aNameIm,false));
    }
   // delete mTimeSegm;


    StdOut() << " Eges: ["
              <<   " Tot=" << mGr.AllEdges().size()
              <<   " Sel="   <<    aSetOfPair.size()
              << "]"
              << " ||| "
             << " Cycle ["
             << "   Th=" << liBinomialCoeff(3,mGr.NbVertex())
             << "  Tot=" << mNbCycleTot
             << "  C3=" << mNbCycleFull
             << "  CM2=" << mNbCycleOK2
             << "  Sel=" << mNbTriExport
             << "] \n";
    return EXIT_SUCCESS;
}


/* ====================================================== */
/*               OriPoseEstimRel2Im                       */
/* ====================================================== */

tMMVII_UnikPApli Alloc_SelectPairOriRel(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_SelectPairOriRel(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_SelectPairOriRel
(
     "OriPoseSelecAllPAir",
      Alloc_SelectPairOriRel,
      "Select Pairs of images using different heuristics",
      {eApF::Ori},
      {eApDT::TieP},
      {eApDT::Orient},
      __FILE__
);


}; // MMVII




