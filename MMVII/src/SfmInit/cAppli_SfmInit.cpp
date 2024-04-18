#include "MMVII_PoseRel.h"
#include "MMVII_PoseTriplet.h"
//#include "MMVII_Tpl_Images.h"
#include "MMVII_TplHeap.h"
#include <random>


namespace MMVII
{
class cRand19937;
class cVertex;
class cEdge;
class cHyperEdge;
class cHyperGraph;


   /* ********************************************************** */
   /*                                                            */
   /*                         cVertex                            */
   /*                                                            */
   /* ********************************************************** */

class cVertex : public cMemCheck
{
    public:
        cVertex(int,std::string);// : mId(aId), mPose(cView()), _ISORIENTED(false) {}

        const int Id() const {return mId;}

        const cView& Pose() const {return mPose;}
        cView& Pose() {return mPose;}

        void SetPose(cView& aP) {mPose=aP;}
        bool& FlagOriented() {return FLAG_ISORIENTED;}

        void Show();

    private:
        int   mId;
        cView mPose;


        bool FLAG_ISORIENTED;
};

cVertex::cVertex(int aId,std::string aN) :
    mId(aId),
    mPose(cView()),
    FLAG_ISORIENTED(false)
{
    mPose.Name() = aN;
}


void cVertex::Show()
{
    StdOut() << "Id=" << mId << " "
             << FLAG_ISORIENTED << std::endl;
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

    const cVertex * StartVertex() const {return mStart;}
    cVertex * StartVertex()  {return mStart;}
    const cVertex * EndVertex() const {return mEnd;}
    cVertex * EndVertex()  {return mEnd;}

    bool operator==(const cEdge& compare) const {
            return (
                        (mStart == compare.StartVertex() && mEnd == compare.EndVertex()) ||
                        (mStart == compare.EndVertex() && mEnd == compare.StartVertex())
                   );
    }

    struct Hash {
            size_t operator()(const cEdge& aE) const {
                size_t hashStart = std::hash<const cVertex*>{}(aE.StartVertex());
                size_t hashEnd = std::hash<const cVertex*>{}(aE.EndVertex());
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
        cHyperEdge(std::vector<cVertex*>, cTriplet&, tU_INT4);
        ~cHyperEdge();

        std::vector<cVertex*>& Vertices() {return mVVertices;}
        const std::vector<cVertex*>& Vertices() const {return mVVertices;}

        std::vector<cEdge*>& Edges() {return mVEdges;}
        const std::vector<cEdge*>& Edges() const {return mVEdges;}

        double Quality() {return mQual;}
        const double Quality() const {return mQual;}

        double BsurH() {return mBsurH;}
        const double BsurH() const {return mBsurH;}

        std::vector<double>& QualityVec() {return mQualVec;}
        const std::vector<double>& QualityVec() const {return mQualVec;}

        tU_INT4& Index() {return mIndex;}
        const tU_INT4& Index() const {return mIndex;}

        void  SetRelPoses(cTriplet& aT) {mRelPoses=aT;}
        const cView& RelPose(int aK) const {return mRelPoses.PVec()[aK];}

        const bool& FlagOriented() const {return FLAG_IS_ORIENTED;}
        bool& FlagOriented() {return FLAG_IS_ORIENTED;}

        void Show();

    private:
        std::vector<cVertex*> mVVertices;
        std::vector<cEdge*>   mVEdges;

        double                mBsurH;
        double                mQual;
        std::vector<double>   mQualVec; //< stores the quality estimate in each random spanning tree
        tU_INT4               mIndex;

        cTriplet              mRelPoses; // see if make it pointer

        bool                  FLAG_IS_ORIENTED;
};


cHyperEdge::~cHyperEdge()
{
    /// free memory
    for (cEdge* e : mVEdges)
        delete e;
}

cHyperEdge::cHyperEdge(std::vector<cVertex*> aVV,cTriplet& aT,tU_INT4 aId) :
    mVVertices(aVV),
    mBsurH(aT.BH()),
    mQual(aT.Residual()),
    mIndex(aId),
    mRelPoses(aT),
    FLAG_IS_ORIENTED(false)
{
    mVEdges.push_back(new cEdge(mVVertices[0],mVVertices[1]));
    mVEdges.push_back(new cEdge(mVVertices[0],mVVertices[2]));
    mVEdges.push_back(new cEdge(mVVertices[1],mVVertices[2]));
}


void cHyperEdge::Show()
{
    StdOut() << "Quality=" << mQual << std::endl;
}

class cCmpEdge
{
    public:
    bool operator()(const cHyperEdge * aE1,const cHyperEdge * aE2) const {return aE1->Quality() > aE2->Quality();}
};

class cIndexEdgeOnId
{
public:
    static void SetIndex(cHyperEdge * aE,tU_INT4 i) {aE->Index() = i;}
    static int  GetIndex(const cHyperEdge * aE) {return aE->Index();}
};


   /* ********************************************************** */
   /*                                                            */
   /*                        cHyperGraph                         */
   /*                                                            */
   /* ********************************************************** */

class cHyperGraph : public cMemCheck
{
    public:
         cHyperGraph() {};
         ~cHyperGraph();

         void                      AddHyperedge(cHyperEdge*);
         std::vector<cHyperEdge*>& GetAdjacentHyperedges(cEdge*);
         bool                      HasAdjacentHyperedges(cEdge*);

         cHyperEdge*               GetHyperEdge(int aK) {return mVHEdges[aK];}
         const cHyperEdge*         GetHyperEdge(int aK) const {return mVHEdges[aK];}

         const std::unordered_map<std::string,cVertex*>& GetMapVertices() const {return mMapVertices;}

         cVertex*                  GetVertex(std::string& aN) {return mMapVertices[aN];}

         tU_INT4                   NbHEdges() {return mVHEdges.size();}
         tU_INT4                   NbVertices() {return mMapVertices.size();}

         void SetVertices(std::unordered_map<std::string,cVertex*>&);

         void CoherencyOfHyperEdges();

         void ClearFlags();

         void SaveDotFile(std::string&);

         void Show();

    private:
        std::unordered_map<std::string,cVertex*>    mMapVertices;
        std::vector<cHyperEdge*>                    mVHEdges;
        std::unordered_map<cEdge, std::vector<cHyperEdge*>, typename cEdge::Hash> mAdjMap;


};

cHyperGraph::~cHyperGraph()
{
    for (auto aIt : mVHEdges)
        delete aIt;
}

void cHyperGraph::ClearFlags()
{
    // vertices set to un-oriented
    for (auto aV : mMapVertices)
        aV.second->FlagOriented() = false;

    // edges set to un-oriented
    for (auto aE : mVHEdges)
        aE->FlagOriented() = false;
}

void cHyperGraph::SaveDotFile(std::string& aName)
{

    cMMVII_Ofs* aDotFile = new cMMVII_Ofs(aName,eFileModeOut::CreateText);

    aDotFile->Ofs() <<  "digraph{" << "\n";

    for (auto aE : mAdjMap)
    {
       aDotFile->Ofs() << aE.first.StartVertex()->Id() << "->" << aE.first.EndVertex()->Id() << std::endl;
    }

   aDotFile->Ofs() << "}" << std::endl;

   delete aDotFile;
}

void cHyperGraph::Show()
{
    //< temporary inverse map to connect vertex with string
    std::unordered_map<const cVertex*,std::string>    aMapNames;
    for (auto aMapIt : mMapVertices)
        aMapNames[aMapIt.second] = aMapIt.first;


    for (auto aElem : mAdjMap)
    {
        StdOut() << "Edge: " << aMapNames[aElem.first.StartVertex()] << " "
                             << aMapNames[aElem.first.EndVertex()] << std::endl;
        for (auto aHypEl : aElem.second)
        {
            StdOut() << "\t" << aMapNames[aHypEl->Vertices()[0]] << "\n"
                     << "\t" << aMapNames[aHypEl->Vertices()[1]] << "\n"
                     << "\t" << aMapNames[aHypEl->Vertices()[2]]
                     << "\t quality:" << aHypEl->Quality() << std::endl;
        }
    }
}

void cHyperGraph::SetVertices(std::unordered_map<std::string,cVertex*>& aMapV)
{
    mMapVertices = aMapV;
}

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

bool cHyperGraph::HasAdjacentHyperedges(cEdge* anE)
{
    if (mAdjMap.find(*anE) != mAdjMap.end())
        return true;
    else
        return false;
}

//< this function must be preceded with HasAdjacentHyperedges
// otherwise throws a bug
std::vector<cHyperEdge*>& cHyperGraph::GetAdjacentHyperedges(cEdge* aEdge)
{
    bool IsInside = (mAdjMap.find(*aEdge) != mAdjMap.end());

    MMVII_INTERNAL_ASSERT_always(IsInside,
                               "cHyperGraph::GetAdjacentHyperedges(cEdge*) ==> "
                               "use HasAdjacentHyperedges(cEdge*) "
                               "before calling the function");

    return mAdjMap[*aEdge];

}

void cHyperGraph::CoherencyOfHyperEdges()
{
    ///
    /// Iterate over all triplets/hyperedges
    ///   not participating in the current solution
    /// and calculate their coherency score with the current global solution
    ///
    for (auto aTri : mVHEdges)
    {
        if (! aTri->FlagOriented())
        {
            std::vector<tPose> aVPoseGlob;
            std::vector<tPose> aVPoseLoc;

            aVPoseLoc.push_back(aTri->RelPose(0).Pose());
            aVPoseLoc.push_back(aTri->RelPose(1).Pose());
            aVPoseLoc.push_back(aTri->RelPose(2).Pose());

            aVPoseGlob.push_back(mMapVertices[aTri->RelPose(0).Name()]->Pose().Pose());
            aVPoseGlob.push_back(mMapVertices[aTri->RelPose(1).Name()]->Pose().Pose());
            aVPoseGlob.push_back(mMapVertices[aTri->RelPose(2).Name()]->Pose().Pose());

            cSimilitud3D<double> aSimGlob2Loc = ComputeSim3D(aVPoseLoc,aVPoseGlob);

            ///
            ///  transform global to local frame
            /// and compute the distance~coherence
            ///
            double aCohScore = 0.0;
            for (auto aI : {0,1,2})
            {
                cDenseMatrix<double> aRotGInL =  aSimGlob2Loc.Rot().Mat() * aVPoseGlob[aI].Rot().Mat();
                cPt3dr aTrGInL = aSimGlob2Loc.Tr() + aSimGlob2Loc.Scale() * aSimGlob2Loc.Rot().Mat() * aVPoseGlob[aI].Tr();

                double aRotDist = aRotGInL.L2Dist(aVPoseLoc[aI].Rot().Mat());
                double aTrDist = DistBase(aTrGInL,aVPoseLoc[aI].Tr()) * aTri->BsurH();

                aCohScore += (aRotDist+aTrDist);


            }
            aCohScore /= 3.0;

            aTri->QualityVec().push_back(aCohScore);

        }
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

        void DoOneSolution(cHyperGraph&);
        void AddToHeapWithEdge(cIndexedHeap<cHyperEdge*,cCmpEdge,cIndexEdgeOnId>&,
                             cHyperGraph&,cHyperEdge*,int); //< add hyperedges of an edge to heap
        void AddToHeapWithVertex(cIndexedHeap<cHyperEdge*,cCmpEdge,cIndexEdgeOnId>&,
                                 cHyperGraph&,
                                 cVertex*); //< if a vertex is connected to another oriented vertex in the graph,
                                            //<  add hyperedges passing through them
        void AddToHeap(cIndexedHeap<cHyperEdge*,cCmpEdge,cIndexEdgeOnId>&,
                       cHyperEdge*);

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

void cAppli_SfmInitFromGraph::AddToHeap(
               cIndexedHeap<cHyperEdge*,cCmpEdge,cIndexEdgeOnId>& aHeap,
               cHyperEdge* aHEdge)
{
    if ( aHeap.IsInHeap(aHEdge))
    {
        aHeap.Push(aHEdge);
        StdOut() << "Added triplet: " <<  aHEdge->Vertices()[0]->Pose().Name() << " "
                 << aHEdge->Vertices()[1]->Pose().Name() << " "
                 << aHEdge->Vertices()[2]->Pose().Name() << std::endl;
    }
}

void cAppli_SfmInitFromGraph::AddToHeapWithVertex(
                         cIndexedHeap<cHyperEdge*,cCmpEdge,cIndexEdgeOnId>& aHeap,
                         cHyperGraph& aGraph,
                         cVertex* aVertex)
{
    /// 2- check if the new vertex composes and edge with any other existing & oriented vertex
    ///    if true, add the hyperedges of that edge to the heap
    ///
    ///  create an edge with new pose and another oriented pose
    ///  compare that edge with all edges in mAdjMap
    ///  if found, add the hyperedges to the heap
    ///

    // iterate over all vertices
    for (auto aI : aGraph.GetMapVertices())
    {
        // if the vertex is oriented
        if (aI.second->FlagOriented())
        {
            cVertex * anOriVertex = aI.second;
            cEdge anEdgeHypoth(aVertex,anOriVertex);

            // if there are hyperedges passing through the oriented vertex and input vertex
            if (aGraph.HasAdjacentHyperedges(&anEdgeHypoth))
            {
                // add each such hyperedge if does not exist already
                for (auto anHE : aGraph.GetAdjacentHyperedges(&anEdgeHypoth))
                {
                    AddToHeap(aHeap,anHE);

                }

            }

        }
    }

    StdOut() << "== Heap size= " << aHeap.Sz() << std::endl;
}

void cAppli_SfmInitFromGraph::AddToHeapWithEdge(cIndexedHeap<cHyperEdge*,cCmpEdge,cIndexEdgeOnId>& aHeap,
                                              cHyperGraph& aGraph,cHyperEdge* anHyE,int aK)
{


    /// 1- add hyperedges connected to an edge of the current hyperedge
    if (aGraph.HasAdjacentHyperedges(anHyE->Edges()[aK]))
    {
        for (auto anE : aGraph.GetAdjacentHyperedges(anHyE->Edges()[aK])) //< for each hyperedge of an edge
        {
            if (anHyE != anE) // do not add to heap if current triplet
            {
                AddToHeap(aHeap,anE);
            }
        }
    }

    StdOut() << "== Heap size= " << aHeap.Sz() << std::endl;

}

void cAppli_SfmInitFromGraph::DoOneSolution(cHyperGraph& aGraph)
{
    StdOut() << "DoOneSolution" << std::endl;


    //< create the heap of hyperedges (triplets) that :
    //<  are un-oriented and
    //<  can be attached to the currently oriented block;
    //<  info: the heap is dynamic
    cCmpEdge aCmp;
    cIndexedHeap<cHyperEdge*,cCmpEdge,cIndexEdgeOnId>  aHeapHyperE(aCmp);


    tU_INT4 aNbTriplets = aGraph.NbHEdges();
    tU_INT4 aNbVertices = aGraph.NbVertices();
    StdOut() << "Number triplets= " << aNbTriplets << ", vertices=" << aNbVertices << std::endl;

    //< choose a random hyperedge
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    tU_INT4 aSeedTriplet = round_ni(aNbTriplets * dis(gen));
    StdOut() << "Seed triplet= " << aSeedTriplet << std::endl;

    /// this triplet sets the origin of the absolute frame
    cHyperEdge* anHyE = aGraph.GetHyperEdge(aSeedTriplet);
    anHyE->FlagOriented() = true;

    cView aPose0Seed = anHyE->RelPose(0); //< local and global ori because it is seed
    cView aPose1Seed = anHyE->RelPose(1); //< local and global ori because it is seed
    cView aPose2Seed = anHyE->RelPose(2); //< local and global ori because it is seed
    StdOut() << "\t " << aPose0Seed.Name() << " "
                      << aPose1Seed.Name() << " "
                      << aPose2Seed.Name() << std::endl;

    /// initialise the global poses of the seed triplet
    aGraph.GetVertex(aPose0Seed.Name())->Pose().Pose() = aPose0Seed.Pose(); //< set the global pose to be the local pose
    aGraph.GetVertex(aPose0Seed.Name())->FlagOriented() = true;
    aGraph.GetVertex(aPose1Seed.Name())->Pose().Pose() = aPose1Seed.Pose();
    aGraph.GetVertex(aPose1Seed.Name())->FlagOriented() = true;
    aGraph.GetVertex(aPose2Seed.Name())->Pose().Pose() = aPose2Seed.Pose();
    aGraph.GetVertex(aPose2Seed.Name())->FlagOriented() = true;

    /// add all neighbouring hyperedges to the heap
    for (auto aK : {0,1,2})
        AddToHeapWithEdge(aHeapHyperE,aGraph,anHyE,aK);



    // Propagate to best quality triplet=hyperedge
    tU_INT4 aNbOrientedV=3; //seed vertices

    /// iterate over the "available" triplets/hyperedges so long
    ///   there are hyperedges and not all vertices were oriented
    ///
    while ( (aNbOrientedV<=aNbVertices) && aHeapHyperE.Sz())
    {
        cHyperEdge* aCurBestHE = nullptr;
        bool IsOK = aHeapHyperE.Pop(aCurBestHE);
        StdOut() << "===== Pop()=" << aCurBestHE->RelPose(0).Name() <<
                                      aCurBestHE->RelPose(1).Name() <<
                                      aCurBestHE->RelPose(2).Name() << std::endl;

        if (IsOK)
        {
            /// 1- get global sommets with known poses
            ///   aCurP1Glob is the first vertex with currently known pose
            ///   aCurP2Glob is the second vertex currentlyknown pose
            ///
            std::string aCurV1Name = aCurBestHE->RelPose(0).Name();
            std::string aCurV2Name = aCurBestHE->RelPose(1).Name();
            std::string aCurV3Name = aCurBestHE->RelPose(2).Name();

            int   aIdCurP1Glob=1e3; //initialise the value with nonsense
            tPose aCurP1Glob;
            int   aIdCurP2Glob=1e3;
            tPose aCurP2Glob;
            int   aIdCurP3Glob=1e3;


            // get the known poses within the current triplet
            if (aGraph.GetVertex(aCurV1Name)->FlagOriented())
            {
                aCurP1Glob = aGraph.GetVertex(aCurV1Name)->Pose().Pose();
                aIdCurP1Glob = 0;

            }
            else if (aGraph.GetVertex(aCurV2Name)->FlagOriented())
            {
                aCurP1Glob = aGraph.GetVertex(aCurV2Name)->Pose().Pose();
                aIdCurP1Glob = 1;
            }

            if (aGraph.GetVertex(aCurV3Name)->FlagOriented())
            {
                aCurP2Glob = aGraph.GetVertex(aCurV3Name)->Pose().Pose();
                aIdCurP2Glob = 2;
            }
            else if (aGraph.GetVertex(aCurV2Name)->FlagOriented())
            {
                 aCurP2Glob = aGraph.GetVertex(aCurV2Name)->Pose().Pose();
                 aIdCurP2Glob = 1;
            }

            /// 2- get the global unknown sommet
            ///    the third vertex with unknown pose
            ///
            // get the unknown pose within the current triplet
            bool IsUnknownPose=true;
            if (! aGraph.GetVertex(aCurV1Name)->FlagOriented())
            {
                aIdCurP3Glob = 0;
            }
            else if (! aGraph.GetVertex(aCurV2Name)->FlagOriented())
            {
                aIdCurP3Glob = 1;
            }
            else if (! aGraph.GetVertex(aCurV3Name)->FlagOriented())
            {
                aIdCurP3Glob = 2;
            }
            else IsUnknownPose=false;

            /// 3- propagate the global poses to the unknown pose
            ///
            if (IsUnknownPose)
            {

                ///
                /// 4- Calculate the transformation that takes from local to global frame
                ///  C = Tr + lambda * M * c   ==>  Tr = C - lambda * M * c
                ///  M = R * r^t
                ///  lambda = distnace_global / distance_local
                ///
                cDenseMatrix<double> aM_ =  aCurP1Glob.Rot().Mat() * aCurBestHE->RelPose(aIdCurP1Glob).Pose().Rot().Mat().Transpose();
                cDenseMatrix<double> aM__ =  aCurP2Glob.Rot().Mat() * aCurBestHE->RelPose(aIdCurP2Glob).Pose().Rot().Mat().Transpose();
                cDenseMatrix<double> aM = (aM_+aM__).ClosestOrthog();//

                double aLambda = Norm2( aCurP2Glob.Tr() - aCurP1Glob.Tr() ) /
                                 Norm2( aCurBestHE->RelPose(aIdCurP2Glob).Pose().Tr() - aCurBestHE->RelPose(aIdCurP1Glob).Pose().Tr());


                cPt3dr aTr_  = aCurP1Glob.Tr() - aLambda * aM * aCurBestHE->RelPose(aIdCurP1Glob).Pose().Tr();
                cPt3dr aTr__ = aCurP2Glob.Tr() - aLambda * aM * aCurBestHE->RelPose(aIdCurP2Glob).Pose().Tr();
                cPt3dr aTr = 0.5*(aTr_+aTr__);

                ///
                /// 5- Set the new global pose
                ///
                cRotation3D<double> aNewRot( aM * aCurBestHE->RelPose(aIdCurP3Glob).Pose().Rot().Mat(), false );
                cPt3dr aNewCenter = aTr + aLambda * aM * aCurBestHE->RelPose(aIdCurP3Glob).Pose().Tr();

                std::string aNewPoseName = aCurBestHE->RelPose(aIdCurP3Glob).Name();
                aGraph.GetVertex(aNewPoseName)->Pose().Pose() = tPose(aNewCenter,aNewRot);
                aGraph.GetVertex(aNewPoseName)->FlagOriented() = true;
                StdOut() << "\t\t\t\t new vertex " << aNbOrientedV << " " << aNewPoseName
                         << aNewCenter << std::endl;

                ///
                /// 6- Add hyperedges of the new edges to aHeapHyperE
                ///    (between aCurP1Glob <--> new and aCurP2Glob <--> new)
                ///
                for (auto aId : {0,1,2})
                {
                    int anIdOfNewPose = aGraph.GetVertex(aNewPoseName)->Id();
                    int anIdOfCurrentV1 = aCurBestHE->Edges()[aId]->StartVertex()->Id();
                    int anIdOfCurrentV2 = aCurBestHE->Edges()[aId]->EndVertex()->Id();

                    if ((anIdOfCurrentV1==anIdOfNewPose) ||
                        (anIdOfCurrentV2==anIdOfNewPose) ) // if the new pose is involved in the edge
                    {
                        AddToHeapWithEdge(aHeapHyperE,aGraph,anHyE,aId);
                    }

                }

                ///
                /// 7- Add hyperedges passing through new vertex and any already oriented vertex
                ///
                AddToHeapWithVertex(aHeapHyperE,aGraph,aGraph.GetVertex(aNewPoseName));

                aCurBestHE->FlagOriented() = true; //flag as oriented to exclude from coherency score

                aNbOrientedV++;

                /// current hyperedge removed from heap with .Pop()

            }
            //else StdOut() << "UnknownPose ? " << IsUnknownPose << std::endl;
        }

    }

    const std::unordered_map<std::string,cVertex*> aMapV = aGraph.GetMapVertices();
    for (auto aV : aMapV)
    {
        if (aV.second->FlagOriented())
            StdOut() << aV.first << " " << aV.second->Pose().Pose().Tr().x() << " "
                     << aV.second->Pose().Pose().Tr().y() << " "
                     << aV.second->Pose().Pose().Tr().z() << std::endl;
    }


    StdOut() << "==END==" << std::endl;


}

int cAppli_SfmInitFromGraph::Exe()
{
     mPhProj.FinishInit();

     /// set of input triplets
     cTripletSet * aTriSet = mPhProj.ReadTriplets();

     /// a map of image strings with id's
     std::unordered_map<std::string,cVertex*> aMapIm2V;

     int aId=0;
     for (auto aT : aTriSet->Set())
     {
         for (auto aV : aT.PVec())
         {
            if (aMapIm2V.find(aV.Name()) == aMapIm2V.end() )
            {
                aMapIm2V[aV.Name()] =  new cVertex(aId++,aV.Name());
            }
         }
     }

     /// Fill the hypergraph
     cHyperGraph aHG;

     //< nodes
     aHG.SetVertices(aMapIm2V);

     //< (hyper)edges
     tU_INT4 anId=0;
     for (auto aT : aTriSet->Set())
     {
         aHG.AddHyperedge(
            new cHyperEdge( {aMapIm2V[aT.PVec()[0].Name()],
                             aMapIm2V[aT.PVec()[1].Name()],
                             aMapIm2V[aT.PVec()[2].Name()]},
                             aT,
                             anId++) );
     }

     //aHG.Show();


     for (auto aIter : {0,1,2,3,4,5,6,7,8,9})
     {
        StdOut() << aIter << std::endl;
        DoOneSolution(aHG);

        aHG.CoherencyOfHyperEdges();
        aHG.ClearFlags();

     }
     //< save the graph in dot format
     //std::string aDotName = mPhProj.DirPhp() + "test.dot";
     //aHG.SaveDotFile(aDotName);

     /// free memory
     for (auto aIt : aMapIm2V)
         delete aIt.second;

     delete aTriSet;

     return EXIT_SUCCESS;
}

/*
 * todo:
     OK initialise graph with triplets
     scenario1 : absolute solution with a random spanning tree
     scenario2 : partition the graph, spanning tree on each partition

*/

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




