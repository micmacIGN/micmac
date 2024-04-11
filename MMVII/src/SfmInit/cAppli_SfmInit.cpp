#include "MMVII_PoseRel.h"
#include "MMVII_PoseTriplet.h"
#include "MMVII_Tpl_Images.h"


namespace MMVII
{

/* todo:
     initialise graph with triplets
     partition the graph
*/

   /* ********************************************************** */
   /*                                                            */
   /*                         cVertex                            */
   /*                                                            */
   /* ********************************************************** */

class cVertex : public cMemCheck
{
    public:
        cVertex(int aId) : mId(aId) {}

        void Show();

    private:
        int mId;
};

void cVertex::Show()
{
    StdOut() << "Id=" << mId << std::endl;
};
   /* ********************************************************** */
   /*                                                            */
   /*                         cEdge                              */
   /*                                                            */
   /* ********************************************************** */

class cEdge : public cMemCheck
{
public:
    cEdge(cVertex* aStart,cVertex* aEnd) : mStart(aStart), mEnd(aEnd) {}

    const cVertex * StartId() const {return mStart;}
    cVertex * StartId()  {return mStart;}
    const cVertex * EndId() const {return mEnd;}
    cVertex * EndId()  {return mEnd;}

    bool operator==(const cEdge& compare) const {
            return (mStart == compare.StartId() && mEnd == compare.EndId()); }

    struct Hash {
            size_t operator()(const cEdge& aE) const {
                size_t hashStart = std::hash<const cVertex*>{}(aE.StartId());
                size_t hashEnd = std::hash<const cVertex*>{}(aE.EndId());
                return hashStart ^ (hashEnd << 1);
            }
        };

private:
    cVertex * mStart;
    cVertex * mEnd;
};

   /* ********************************************************** */
   /*                                                            */
   /*                        cHyperEdge                          */
   /*                                                            */
   /* ********************************************************** */

class cHyperEdge : public cMemCheck
{
    public:
        cHyperEdge(std::vector<cVertex*> aVV, int aQual);

        std::vector<cEdge*> Edges() {return mVEdges;}
        const std::vector<cEdge*> Edges() const {return mVEdges;}

        double Quality() {return mQual;}
        const double Quality() const {return mQual;}

        void Show();

    private:
        std::vector<cVertex*> mVVertices;
        std::vector<cEdge*>   mVEdges;

        double                mQual;
};


cHyperEdge::cHyperEdge(std::vector<cVertex*> aVV, int aQual) : mVVertices(aVV), mQual(aQual)
{
    mVEdges.push_back(new cEdge(mVVertices[0],mVVertices[1]));
    mVEdges.push_back(new cEdge(mVVertices[0],mVVertices[2]));
    mVEdges.push_back(new cEdge(mVVertices[1],mVVertices[2]));
}


void cHyperEdge::Show()
{
    StdOut() << "Quality=" << mQual << std::endl;
}

   /* ********************************************************** */
   /*                                                            */
   /*                        cHyperGraph                         */
   /*                                                            */
   /* ********************************************************** */

class cHyperGraph : public cMemCheck
{
    public:
         void                     AddHyperedge(cHyperEdge*);
         std::vector<cHyperEdge*> GetAdjacentHyperedges(cEdge*);


    private:
        std::vector<cHyperEdge*> mVHEdges;
        std::unordered_map<cEdge, std::vector<cHyperEdge*>, typename cEdge::Hash> mAdjMap;
};

void cHyperGraph::AddHyperedge(cHyperEdge *aHE)
{
    // push the hyperedge in the map
    mVHEdges.push_back(aHE);
    for (cEdge* aE : aHE->Edges())
    {
        mAdjMap[*aE].push_back(aHE);
    }

    // sort the hyperedges of each added edge by their quality metric
    for (cEdge* aE : aHE->Edges())
    {
        std::sort(mAdjMap[*aE].begin(),mAdjMap[*aE].end(),
                  [](const cHyperEdge* a, const cHyperEdge* b)
                  { return a->Quality() > b->Quality(); }
        );
    }
}


std::vector<cHyperEdge*> cHyperGraph::GetAdjacentHyperedges(cEdge* aEdge)
{
    if (mAdjMap.find(*aEdge) != mAdjMap.end())
    {
        return mAdjMap[*aEdge];
    }
    else
    {
        return std::vector<cHyperEdge*>();
    }
}

   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_SfmInitFromGraph                    */
   /*                                                            */
   /* ********************************************************** */

class cAppli_SfmInitFromGraph: public cMMVII_Appli
{
     public :
	typedef cIsometry3D<tREAL8>  tPose;

        cAppli_SfmInitFromGraph(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :
        cPhotogrammetricProject   mPhProj;

};

cAppli_SfmInitFromGraph::cAppli_SfmInitFromGraph(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this)
{
}

cCollecSpecArg2007 & cAppli_SfmInitFromGraph::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
              <<  mPhProj.DPOriTriplets().ArgDirInMand()
           ;
}

cCollecSpecArg2007 & cAppli_SfmInitFromGraph::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return    anArgOpt    ;
}



int cAppli_SfmInitFromGraph::Exe()
{
     mPhProj.FinishInit();


     cTripletSet * aTriSet = mPhProj.ReadTriplets();

     for (auto aT : aTriSet->Set())
     {
        for (auto aV : aT.PVec())
            StdOut() << aV.Name() << " ";
        StdOut() << std::endl;
     }


     delete aTriSet;

     return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_SfmInitFromGraph(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_SfmInitFromGraph(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_SfmInitFromGraph
(
     "SfmInitFromGraph",
      Alloc_SfmInitFromGraph,
      "Compute initial orientations from a graph of relative orientations",
      {eApF::Ori},
      {eApDT::TieP},
      {eApDT::Orient},
      __FILE__
);



}; // MMVII




