#include "MMVII_SfmInit.h"
#include <random>
//#include "MMVII_TplHeap.h"
#include "graph.h"
#include "MMVII_HierarchicalProc.h"

namespace MMVII
{
class cRand19937;
struct cObjQual;
class cCmpObjQual;
class cIndexObjQual;


double PENALISE_NONVISITED_TRIPLETS=5;

/* ********************************************************** */
/*                                                            */
/*                         cVertex                            */
/*                                                            */
/* ********************************************************** */

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
/*                        cHyperEdge                          */
/*                                                            */
/* ********************************************************** */

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
    mIndexHeap(aId),
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
        bool operator()(const cHyperEdge * aE1,const cHyperEdge * aE2) const {return aE1->Quality() < aE2->Quality();}
};

class cIndexEdgeOnId
{
    public:
        static void SetIndex(cHyperEdge * aE,tU_INT4 i) {aE->IndexHeap() = i;}
        static int  GetIndex(const cHyperEdge * aE) {return aE->IndexHeap();}
};

/* ********************************************************** */
/*                                                            */
/*                        cHyperGraph                         */
/*                                                            */
/* ********************************************************** */


cHyperGraph::~cHyperGraph()
{
    /// free memory
    ///
    for (auto aIt : mVHEdges)
        delete aIt;

    for (auto aIt : mMapVertices)
        delete aIt.second;
}

void cHyperGraph::InitFromTriSet(const cTripletSet* aTSet)
{

    /// a map of image strings with id's
    std::unordered_map<std::string,cVertex*> aMapIm2V;

    int aId=0;
    for (auto aT : aTSet->Set())
    {
        for (auto aV : aT.PVec())
        {
           if (aMapIm2V.find(aV.Name()) == aMapIm2V.end() )
           {
               aMapIm2V[aV.Name()] =  new cVertex(aId++,aV.Name());
               //StdOut() << aV.Name() << std::endl;
           }
        }
    }

    /// 1- Fill in the hypergraph
    ///
    // nodes
    SetVertices(aMapIm2V);

    // (hyper)edges
    //tU_INT4 anId=0;
    for (auto aT : aTSet->Set())
    {
        AddHyperedge(
           new cHyperEdge( {aMapIm2V[aT.PVec()[0].Name()],
                            aMapIm2V[aT.PVec()[1].Name()],
                            aMapIm2V[aT.PVec()[2].Name()]},
                            aT,
                            aT.Id()) );
    }

    IS_INIT = true;
    //Show();
}

tU_INT4 cHyperGraph::NbPins()
{
    tU_INT4 aNbPins=0;

    if (IS_INIT)
    {
        for (auto aH : mVHEdges)
        {
            aNbPins += aH->Vertices().size();
        }
    }

    return aNbPins;
}

void cHyperGraph::ShowFlags(bool VraiFaux)
{
    int aNbE=0;
    StdOut() << "Edges" ;
    for (auto aE : mVHEdges)
    {
        if (aE->FlagOriented() == VraiFaux)
            aNbE++;
    }
    StdOut() << aNbE << " " << VraiFaux << std::endl;

    int aNbV=0;
    StdOut() << "Vertices" << std::endl;
    for (auto aV : this->mMapVertices)
    {
        if (aV.second->FlagOriented())
            aNbV++;
    }
    StdOut() << aNbV << " " << VraiFaux << std::endl;
}

void cHyperGraph::ShowHyperEdgeVQual()
{
    int aNb=0;
    for (auto aE : mVHEdges)
    {
        StdOut() << aNb++ << " ";
        for (auto aQ : aE->QualityVec())
            StdOut() << aQ << " ";

        StdOut() << "\t ===global metric=" << aE->Quality() << std::endl;
    }
}

void cHyperGraph::UpdateIndHeap()
{
    for (auto aE : mVHEdges)
    {
        aE->IndexHeap() = MMVII_HEAP_NO_INDEX; // E->Index();
    }
}

void cHyperGraph::UpdateQualFromVec(double aProp)
{


    for (auto aE : mVHEdges)
    {
        int aNb=0;
        cStdStatRes aStats;
        for (auto aQ : aE->QualityVec())
        {
            aStats.Add(aQ);
            aNb++;
        }

        //StdOut() << aE->Quality() << " => ";
        // if the quality vector is empty, penalise the initial metric
        if (aNb>2)
            aE->Quality() = aStats.ErrAtProp(aProp);
        else if (aNb==2)
            aE->Quality() = aStats.Avg();
        else
            aE->Quality() = aE->Quality() * PENALISE_NONVISITED_TRIPLETS;

        //StdOut() << aE->Quality() << " " << aNb << std::endl;

    }
}

void cHyperGraph::RandomizeQualOfHyperE()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 5.0);

    for (auto aE : mVHEdges)
    {
        aE->Quality() = dis(gen);
    }
}

void cHyperGraph::ClearFlags()
{
    // vertices set to un-oriented
    for (auto aV : mMapVertices)
        aV.second->FlagOriented() = false;

    // edges set to un-oriented
    for (auto aE : mVHEdges)
    {
        aE->FlagOriented() = false;
    }
}

cHyperEdge* cHyperGraph::CurrentBestHyperedge()
{
    cHyperEdge* aRes = mVHEdges[0]; //initialise with first element
    StdOut() << aRes->Quality() << std::endl;
    for (auto aE : this->mVHEdges)
    {
        if (aE->Quality() < aRes->Quality())
            aRes = aE;
    }

    return aRes;
}

void cHyperGraph::SaveDotFile(std::string& aName)
{

    cMMVII_Ofs* aFile = new cMMVII_Ofs(aName,eFileModeOut::CreateText);

    aFile->Ofs() <<  "digraph{" << "\n";

    for (auto aE : mAdjMap)
    {
       aFile->Ofs() << aE.first.StartVertex()->Id() << "->" << aE.first.EndVertex()->Id() << std::endl;
    }

   aFile->Ofs() << "}" << std::endl;

   delete aFile;
}

void cHyperGraph::SaveTriGraph(std::string& aName)
{
    /// save the graph in my triplet format
    cMMVII_Ofs* aFile = new cMMVII_Ofs(aName,eFileModeOut::CreateText);

    aFile->Ofs() << this->NbHEdges() << " " << this->NbVertices() << " 1" << "\n";

    //map to remove doubles
    std::map<std::string,std::pair<int,cPt2di>> aMapOfTripletPairs;
    int aNbDupl=0;

    double aUpScale=1.0;
    for (auto anEdg : this->mAdjMap)
    {
        /// if there is more than 1 triplet
        size_t aTriNum = anEdg.second.size();
        if (aTriNum>1)
        {
            /// explore all possible combinations of triplet pairs
            for (size_t aTi=0; aTi<aTriNum; aTi++)
            {
                for (size_t aTj=0; aTj<aTriNum; aTj++)
                {
                    /// iterate over the diagonal sup
                    if (aTi<aTj)
                    {

                       // StdOut() << aTi << " " << aTj << std::endl;
                        double aQual = //eventually this score should come from reprojection error of 5pts
                                (anEdg.second[aTi]->Quality()>0) ? 1.0/(1 + anEdg.second[aTi]->Quality()) : 0.0;
                      //  StdOut() << aQual << " " << anEdg.second[aTi]->Quality() << "\n";

                        std::string aCurTriPairDir = ToStr(int(anEdg.second[aTi]->Index()))+"to"+
                                                     ToStr(int(anEdg.second[aTj]->Index()));
                        std::string aCurTriPairInv = ToStr(int(anEdg.second[aTj]->Index()))+"to"+
                                                     ToStr(int(anEdg.second[aTi]->Index()));

                        if ( (aMapOfTripletPairs.find(aCurTriPairDir) == aMapOfTripletPairs.end()) &&
                             (aMapOfTripletPairs.find(aCurTriPairInv) == aMapOfTripletPairs.end()) )
                        {
                            aMapOfTripletPairs[aCurTriPairDir] = std::make_pair(round_up(aUpScale* aQual),
                                            cPt2di(anEdg.second[aTi]->Index(),anEdg.second[aTj]->Index()));

                            /// save a triplet pair (two pairs sharing an edge)
                            aFile->Ofs() << (aUpScale* aQual ) << " "
                                         << anEdg.second[aTi]->Index() << " "
                                      //   << anEdg.second[aTi]->Vertices()[0]->Pose().Name()
                                      //   << " " << anEdg.second[aTi]->Vertices()[1]->Pose().Name()
                                      //   << " " << anEdg.second[aTi]->Vertices()[2]->Pose().Name() << " "
                                         << anEdg.second[aTj]->Index()
                                   //      << " "
                                   //      << anEdg.second[aTj]->Vertices()[0]->Pose().Name()
                                   //      << " " << anEdg.second[aTj]->Vertices()[1]->Pose().Name()
                                   //      << " " << anEdg.second[aTj]->Vertices()[2]->Pose().Name()
                                         << std::endl;
                        }
                        else aNbDupl++;

                    }
                }
            }
        }
    }
    StdOut() << "Number of detected duplicates: " << aNbDupl << ", Nb of edges=" << mAdjMap.size() << std::endl;
    delete aFile;
}

void cHyperGraph::SaveHMetisFile(std::string& aName)
{
    /// save the graph in hmetis format
    cMMVII_Ofs* aFile = new cMMVII_Ofs(aName,eFileModeOut::CreateText);

    ///save a list of file names (to match img id)
    cMMVII_Ofs* aFileList = new cMMVII_Ofs(aName.substr(0,aName.size()-4)+"_imlist.txt",eFileModeOut::CreateText);

    aFile->Ofs() << this->NbHEdges() << " " << this->NbVertices() << " 1" << "\n";

    int aOffSet = 1;
    double aUpScale=10.0;
    for (auto aE : mVHEdges)
    {
        double aQual = (aE->Quality()>0) ? 1.0/aE->Quality() : 0.0;
        aFile->Ofs() << round_up(aUpScale* aQual ) << " "
                                             << aE->Vertices()[0]->Id() +aOffSet << " "
                                             << aE->Vertices()[1]->Id() +aOffSet << " "
                                             << aE->Vertices()[2]->Id() +aOffSet << "\n";

        aFileList->Ofs() << aE->Vertices()[0]->Id() +aOffSet << " " << aE->Vertices()[0]->Pose().Name() << " "
                         << aE->Vertices()[1]->Id() +aOffSet << " " << aE->Vertices()[1]->Pose().Name() << " "
                         << aE->Vertices()[2]->Id() +aOffSet << " " << aE->Vertices()[2]->Pose().Name() << "\n";

    }
    delete aFile;
    delete aFileList;


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
// otherwise can throw an error
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
    /// Iterate over all triplets/hyperedges not participating in the current solution
    /// and calculate their coherency with the current global solution
    ///
    for (auto aTri : mVHEdges)
    {
        if (! aTri->FlagOriented())
        {
            cVertex* aPGlob0 = mMapVertices[aTri->RelPose(0).Name()];
            cVertex* aPGlob1 = mMapVertices[aTri->RelPose(1).Name()];
            cVertex* aPGlob2 = mMapVertices[aTri->RelPose(2).Name()];

            if (aPGlob0->FlagOriented() &&
                aPGlob1->FlagOriented() &&
                aPGlob2->FlagOriented() )
            {
                std::vector<tPose> aVPoseGlob;
                std::vector<tPose> aVPoseLoc;

                aVPoseLoc.push_back(aTri->RelPose(0).Pose());
                aVPoseLoc.push_back(aTri->RelPose(1).Pose());
                aVPoseLoc.push_back(aTri->RelPose(2).Pose());

                aVPoseGlob.push_back(aPGlob0->Pose().Pose());
                aVPoseGlob.push_back(aPGlob1->Pose().Pose());
                aVPoseGlob.push_back(aPGlob2->Pose().Pose());

                cSimilitud3D<double> aSimGlob2Loc = ComputeSim3D(aVPoseLoc,aVPoseGlob);

                ///
                ///  transform global to local frame
                /// and compute the distance~coherence
                ///
                double aCohScore = 0.0;
                for (auto aI : {0,1,2})
                {
                    cDenseMatrix<double> aRotGInL =  aVPoseGlob[aI].Rot().Mat() * aSimGlob2Loc.Rot().Mat().Transpose();
                    cPt3dr aTrGInL = aSimGlob2Loc.Tr() + aSimGlob2Loc.Scale() * aSimGlob2Loc.Rot().Mat()  * aVPoseGlob[aI].Tr();

                    double aRotDist = aRotGInL.L2Dist(aVPoseLoc[aI].Rot().Mat());
                    //StdOut() << aTrGInL << " " << aVPoseLoc[aI].Tr() << std::endl;

                    double aTrDist = 0;
                    if ( (aTrGInL!=cPt3dr(0,0,0)) && aVPoseLoc[aI].Tr()!=cPt3dr(0,0,0)) //possible case with perfect triplets
                        aTrDist = DistBase(aTrGInL,aVPoseLoc[aI].Tr()) * aTri->BsurH();

                    aCohScore += (aRotDist+aTrDist);


                }
                aCohScore /= 3.0;

                aTri->QualityVec().push_back(aCohScore);
            }

        }
    }
}

void cHyperGraph::DFS(int anId, std::vector<bool>& aNodesVisited,
                      std::map<int,std::vector<int>>& aAdjMap)
{
    aNodesVisited[anId] = true;

    for (auto neighb : aAdjMap[anId])
    {
        if (!aNodesVisited[neighb])
            DFS(neighb,aNodesVisited,aAdjMap);
    }

}

bool cHyperGraph::CheckConnectivity(
        std::map<int,std::vector<int>>& aAdjMap)
{
    int aNumN=aAdjMap.size();
    std::vector<bool> aNVisited(aNumN,false);

    /// go to node with at least 1 edge
    ///
    int aK;
    for (aK=0; aK<aNumN; aK++)
    {
        if (!aAdjMap[aK].empty())
            break;
    }
    StdOut() << "aK= " << aK << " " << aAdjMap[aK].size() << std::endl;
    for (auto toto : aAdjMap[aK])
        StdOut() << toto << std::endl;

    /// depth-first algorithm
    ///
    DFS(aK,aNVisited,aAdjMap);

    /// check if all nodes visited
    ///
    for (int aK=0; aK<aNumN; aK++)
        if (!aNVisited[aK] && !aAdjMap[aK].empty())
        {
            StdOut() << aK << std::endl;
            return false;
        }
    return true;
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
     int                       mNbIter;
     double                    mBestProp;

     template <typename tObj,typename tCmp,typename tInd>
     void DoOneSolutionRandom(cHyperGraph&,cIndexedHeap<tObj,tCmp,tInd>&);
     template <typename tObj,typename tCmp,typename tInd>
     void DoOneSolution(cHyperGraph&,cIndexedHeap<tObj,tCmp,tInd>&,cHyperEdge*);

     template <typename tObj,typename tCmp,typename tInd>
     void AddToHeapWithEdge(cIndexedHeap<tObj,tCmp,tInd>&,
                            cHyperGraph&,cHyperEdge*,int); //< add hyperedges of an edge to heap

     template <typename tObj,typename tCmp,typename tInd>
     void AddToHeapWithVertex(cIndexedHeap<tObj,tCmp,tInd>&,
                              cHyperGraph&,
                              cVertex*); //< if a vertex is connected to another oriented vertex in the graph,
                                         //<  add hyperedges passing through them
     template <typename tObj,typename tCmp,typename tInd>
     void AddToHeap(cIndexedHeap<tObj,tCmp,tInd>&,
                    cHyperEdge*);



};


cAppli_SfmInitFromGraph::cAppli_SfmInitFromGraph(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mNbIter      (5),
    mBestProp    (0.5)
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
   return    anArgOpt
           << AOpt2007(mNbIter,"Iter", "Number of random spanning trees; def=5")
           << AOpt2007(mBestProp,"BestProp","Error proportion, def median i.e. 0.5");
}

template <typename tObj,typename tCmp,typename tInd>
void cAppli_SfmInitFromGraph::AddToHeap(
               cIndexedHeap<tObj,tCmp,tInd>& aHeap,
               cHyperEdge* aHEdge)
{
    if ( ! aHeap.IsInHeap(aHEdge)) //
    {
         {
            aHeap.Push(aHEdge);

            /*StdOut() << "Added triplet: " << aHEdge->IndexHeap() << " " << aHEdge->Index() << " "
                                          << aHEdge->Vertices()[0]->Pose().Name() << " "
                                          << aHEdge->Vertices()[1]->Pose().Name() << " "
                                          << aHEdge->Vertices()[2]->Pose().Name() << std::endl;*/
        }
    }

}

template <typename tObj,typename tCmp,typename tInd>
void cAppli_SfmInitFromGraph::AddToHeapWithVertex(
                         cIndexedHeap<tObj,tCmp,tInd>& aHeap,
                         cHyperGraph& aGraph,
                         cVertex* aVertex)
{
    /// 2- check if the new vertex composes and edge with any other existing & oriented vertex
    ///    if true, add the hyperedges of that edge to the heap
    ///
    ///  in practice: create an edge with new pose and another oriented pose
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

    //StdOut() << "== Heap size= " << aHeap.Sz() << std::endl;
}

template <typename tObj,typename tCmp,typename tInd>
void cAppli_SfmInitFromGraph::AddToHeapWithEdge(cIndexedHeap<tObj,tCmp,tInd>& aHeap,
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

    //StdOut() << "== Heap size= " << aHeap.Sz() << std::endl;

}

template <typename tObj,typename tCmp,typename tInd>
void cAppli_SfmInitFromGraph::DoOneSolutionRandom(cHyperGraph& aGraph,cIndexedHeap<tObj,tCmp,tInd>& aHeap)
{
    tU_INT4 aNbTriplets = aGraph.NbHEdges();

    //< choose a random hyperedge
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    tU_INT4 aSeedTriplet = round_ni( (aNbTriplets-1) * dis(gen));
    StdOut() << "Seed triplet= " << aSeedTriplet << std::endl;

    /// this triplet sets the origin of the absolute frame
    cHyperEdge* aSeedTrip = aGraph.GetHyperEdge(aSeedTriplet);

    DoOneSolution(aGraph,aHeap,aSeedTrip);
}

template <typename tObj,typename tCmp,typename tInd>
void cAppli_SfmInitFromGraph::DoOneSolution(cHyperGraph& aGraph,
                                            cIndexedHeap<tObj,tCmp,tInd>& aHeap,
                                            cHyperEdge* aSeedHE)
{

    tU_INT4 aNbTriplets = aGraph.NbHEdges();
    tU_INT4 aNbVertices = aGraph.NbVertices();
    StdOut() << "Number triplets= " << aNbTriplets << ", vertices=" << aNbVertices << std::endl;


    aSeedHE->FlagOriented() = true;

    cView aPose0Seed = aSeedHE->RelPose(0); //< local and global ori because it is seed
    cView aPose1Seed = aSeedHE->RelPose(1); //< local and global ori because it is seed
    cView aPose2Seed = aSeedHE->RelPose(2); //< local and global ori because it is seed
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
        AddToHeapWithEdge(aHeap,aGraph,aSeedHE,aK);
    StdOut() << "Added seed triplet hyperedges" << std::endl;



    // Propagate to best quality triplet=hyperedge
    tU_INT4 aNbOrientedV=3; //seed vertices

    /// iterate over the "available" triplets/hyperedges so long
    ///   there are hyperedges and not all vertices were oriented
    ///
    while ( (aNbOrientedV<=aNbVertices) && aHeap.Sz())
    {
        cHyperEdge* aCurBestHE = nullptr;

        bool IsOK = aHeap.Pop(aCurBestHE);

        /*StdOut() << "===== Pop()=" << aCurBestHE->RelPose(0).Name() <<
                                      aCurBestHE->RelPose(1).Name() <<
                                      aCurBestHE->RelPose(2).Name() << std::endl;*/

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
                // StdOut() << aCurBestHE->Quality() << std::endl;
                ///
                /// 4- Calculate the transformation that takes from local to global frame
                ///  C = Tr + lambda * M * c   ==>  Tr = C - lambda * M * c
                ///  M = R^t * r
                ///  lambda = distnace_global / distance_local
                ///
                cDenseMatrix<double> aM_  =  aCurP1Glob.Rot().Mat().Transpose() * aCurBestHE->RelPose(aIdCurP1Glob).Pose().Rot().Mat();
                cDenseMatrix<double> aM__ =  aCurP2Glob.Rot().Mat().Transpose() * aCurBestHE->RelPose(aIdCurP2Glob).Pose().Rot().Mat();
                cDenseMatrix<double> aM = (0.5*(aM_+aM__)).ClosestOrthog();//

                double aLambda = Norm2( aCurP2Glob.Tr() - aCurP1Glob.Tr() ) /
                                 Norm2( aCurBestHE->RelPose(aIdCurP2Glob).Pose().Tr() - aCurBestHE->RelPose(aIdCurP1Glob).Pose().Tr());


                cPt3dr aTr_  = aCurP1Glob.Tr() - aLambda * aM * aCurBestHE->RelPose(aIdCurP1Glob).Pose().Tr();
                cPt3dr aTr__ = aCurP2Glob.Tr() - aLambda * aM * aCurBestHE->RelPose(aIdCurP2Glob).Pose().Tr();
                cPt3dr aTr = 0.5*(aTr_+aTr__);


                ///
                /// 5- Set the new global pose
                ///
                cRotation3D<double> aNewRot(   aCurBestHE->RelPose(aIdCurP3Glob).Pose().Rot().Mat() * aM.Transpose(), true );
                cPt3dr aNewCenter = aTr + aLambda * aM * aCurBestHE->RelPose(aIdCurP3Glob).Pose().Tr();

                std::string aNewPoseName = aCurBestHE->RelPose(aIdCurP3Glob).Name();
                aGraph.GetVertex(aNewPoseName)->Pose().Pose() = tPose(aNewCenter,aNewRot);
                aGraph.GetVertex(aNewPoseName)->FlagOriented() = true;

                /*
                StdOut() << "Estimated from: " << aCurBestHE->Vertices()[0]->Pose().Name() << " "
                                               << aCurBestHE->Vertices()[1]->Pose().Name() << " "
                                               << aCurBestHE->Vertices()[2]->Pose().Name() << std::endl;*/

                        if (false)
                        {
                            StdOut() << "paerents: " <<  aCurBestHE->RelPose(aIdCurP1Glob).Name() << " "
                                                     <<  aCurBestHE->RelPose(aIdCurP2Glob).Name() << std::endl;
                            StdOut() << "daugther pose name=" << aNewPoseName << std::endl;
                            StdOut() << "====local" << std::endl;
                            StdOut() << "1"  << "\n";
                            for (int aK1=0; aK1<3; aK1++)
                            {
                                StdOut() << "[";
                                for (int aK2=0; aK2<3; aK2++)
                                {
                                    StdOut() << aCurBestHE->RelPose(aIdCurP1Glob).Pose().Rot().Mat()(aK1,aK2) << ((aK2<2) ? ", " : " ");
                                }
                                StdOut() << "]," << std::endl;
                            }
                            StdOut() << "tr= [" << aCurBestHE->RelPose(aIdCurP1Glob).Pose().Tr().x() << "],["
                                                << aCurBestHE->RelPose(aIdCurP1Glob).Pose().Tr().y() <<  "],["
                                                << aCurBestHE->RelPose(aIdCurP1Glob).Pose().Tr().z() << "]" << std::endl;

                            StdOut() << "2"  << "\n";
                            for (int aK1=0; aK1<3; aK1++)
                            {
                                StdOut() << "[";
                                for (int aK2=0; aK2<3; aK2++)
                                {
                                    StdOut() << aCurBestHE->RelPose(aIdCurP2Glob).Pose().Rot().Mat()(aK1,aK2) << ((aK2<2) ? ", " : " ");
                                }
                                StdOut() << "]," << std::endl;
                            }
                            StdOut() << "tr= [" << aCurBestHE->RelPose(aIdCurP2Glob).Pose().Tr().x() << "],["
                                                << aCurBestHE->RelPose(aIdCurP2Glob).Pose().Tr().y() <<  "],["
                                                << aCurBestHE->RelPose(aIdCurP2Glob).Pose().Tr().z() << "]" << std::endl;

                            StdOut() << "3"  << "\n";
                            for (int aK1=0; aK1<3; aK1++)
                            {
                                StdOut() << "[";
                                for (int aK2=0; aK2<3; aK2++)
                                {
                                    StdOut() << aCurBestHE->RelPose(aIdCurP3Glob).Pose().Rot().Mat()(aK1,aK2) << ((aK2<2) ? ", " : " ");
                                }
                                StdOut() << "]," << std::endl;
                            }
                            StdOut() << "tr= [" << aCurBestHE->RelPose(aIdCurP3Glob).Pose().Tr().x() << "],["
                                                << aCurBestHE->RelPose(aIdCurP3Glob).Pose().Tr().y() <<  "],["
                                                << aCurBestHE->RelPose(aIdCurP3Glob).Pose().Tr().z() << "]" << std::endl;

                            StdOut() << "====global" << std::endl;
                            StdOut() << "[";
                            for (int aK1=0; aK1<3; aK1++)
                            {
                                StdOut() << "[";
                                for (int aK2=0; aK2<3; aK2++)
                                {
                                    StdOut() << aCurP1Glob.Rot().Mat()(aK1,aK2) << ((aK2<2) ? ", " : " ");
                                }
                                StdOut() << "]," << std::endl;
                            }
                            StdOut() << "]" << std::endl;
                            StdOut() << "tr= [" << aCurP1Glob.Tr().x() << "],["
                                                << aCurP1Glob.Tr().y() <<  "],["
                                                << aCurP1Glob.Tr().z() << "]" << std::endl;
                            StdOut() << "==2" << std::endl;
                            StdOut() << "[";
                            for (int aK1=0; aK1<3; aK1++)
                            {
                                StdOut() << "[";
                                for (int aK2=0; aK2<3; aK2++)
                                {
                                    StdOut() << aCurP2Glob.Rot().Mat()(aK1,aK2) << ((aK2<2) ? ", " : " ");
                                }
                                StdOut() << "]," << std::endl;
                            }
                            StdOut() << "]" << std::endl;
                            StdOut() << "tr= [" << aCurP2Glob.Tr().x() << "],["
                                                << aCurP2Glob.Tr().y() <<  "],["
                                                << aCurP2Glob.Tr().z() << "]" << std::endl;

                            StdOut() << "=====similarity" << std::endl;
                            aM.Show();
                            StdOut() << "Tr=" << aTr << ", L=" << aLambda << std::endl;
                            StdOut() << "=====new pose" << std::endl;
                            aNewRot.Mat().Show();
                            StdOut() << "tr=" << aNewCenter << std::endl;

                        }
                   // }
               // }
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
                        AddToHeapWithEdge(aHeap,aGraph,aCurBestHE,aId);
                    }

                }

                ///
                /// 7- Add hyperedges passing through new vertex and any already oriented vertex
                ///
                AddToHeapWithVertex(aHeap,aGraph,aGraph.GetVertex(aNewPoseName));

                aCurBestHE->FlagOriented() = true; // flag as oriented to exclude from coherency score

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

     /*
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

     /// 1- Fill in the hypergraph
     ///
     cHyperGraph aHG;

     // nodes
     aHG.SetVertices(aMapIm2V);

     // (hyper)edges
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
     */

     /// 1- initialise the hypergraph from a set of triplets
     ///
     cHyperGraph aHG;
     aHG.InitFromTriSet(aTriSet);


    /// 2- run several spanning trees to evaluate triplets' quality
    ///

     for (int aIter=0; aIter<mNbIter; aIter++)
     {
        StdOut() << aIter << std::endl;

        //aHG.RandomizeQualOfHyperE();
        aHG.UpdateIndHeap();

        /// create the heap of hyperedges (triplets) that :
        ///  are un-oriented and
        ///  can be attached to the currently oriented block;
        ///  info: the heap is dynamic
        cCmpEdge aCmp;
        cIndexedHeap<cHyperEdge*,cCmpEdge,cIndexEdgeOnId> aHeap(aCmp);

        /// span a solution across the graph
        DoOneSolutionRandom(aHG,aHeap);

        /// compute the coherency of that solution with triplets
        aHG.CoherencyOfHyperEdges();  //=======

        aHG.ClearFlags();

        //getchar();

     }

     //aHG.ShowHyperEdgeVQual();

     /// 3- get the best solution
     ///
     //update the quality
     aHG.UpdateQualFromVec(mBestProp);
     aHG.UpdateIndHeap();
     //aHG.ShowHyperEdgeVQual();

     StdOut() << "==Best solution==" << std::endl;

     cHyperEdge* aSeed = aHG.CurrentBestHyperedge();
     StdOut() << "Quality=" << aSeed->Quality() << std::endl;
     cCmpEdge aCm;
     cIndexedHeap<cHyperEdge*,cCmpEdge,cIndexEdgeOnId> aH(aCm);
     DoOneSolution(aHG,aH,aSeed);


     //< save the graph in dot format
     //std::string aDotName = mPhProj.DirPhp() + "test.dot";
     //aHG.SaveDotFile(aDotName);

     /*/// free memory
     for (auto aIt : aMapIm2V)
         delete aIt.second;*/

     delete aSeed;
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

/* ********************************************************** */
/*                                                            */
/*                 cAppli_SfmInitWithPartition                */
/*                                                            */
/* ********************************************************** */



cAppli_SfmInitWithPartition::cAppli_SfmInitWithPartition(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mSourceInit  (1.0),
    mSinkInit  (1.0),
    mNbParts     (4),
    mImbalance   (0.03),
    mPartOutFile ("kah_parts_out.hgr")
{
}


cCollecSpecArg2007 & cAppli_SfmInitWithPartition::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
              <<  mPhProj.DPOriTriplets().ArgDirInMand()
              <<  Arg2007(mHMetisFile,"Hypergraph file in hmetis format")
              <<  Arg2007(mTGraphFile,"Triplet graph file in hmetis format")
           ;
}

cCollecSpecArg2007 & cAppli_SfmInitWithPartition::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return    anArgOpt
           << AOpt2007(mNbParts,"NbParts", "Number of partitions; def=4")
           << AOpt2007(mSourceInit,"SourceInit","Initialise source weight, Def=1.0")
           << AOpt2007(mSinkInit,"SinkInit","Initialise sink weight, Def=1.0")
           << AOpt2007(mImbalance,"Imb","Partition size imbalance, Def=0.03")
           << AOpt2007(mPartOutFile,"PartOutput","Write partition to a file (filename)")
              ;
}

void cAppli_SfmInitWithPartition::DFS(int anId, std::vector<bool>& aNodesVisited,
                      std::map<int,std::vector<int>>& aAdjMap)
{
    aNodesVisited[anId] = true;

    for (auto neighb : aAdjMap[anId])
    {
        if (!aNodesVisited[neighb])
            DFS(neighb,aNodesVisited,aAdjMap);
    }

}

bool cAppli_SfmInitWithPartition::CheckConnectivity(
        std::map<int,std::vector<int>>& aAdjMap)
{
    int aNumN=aAdjMap.size();
    std::vector<bool> aNVisited(aNumN,false);

    /// go to node with at least 1 edge
    ///
    int aK;
    for (aK=0; aK<aNumN; aK++)
    {
        if (!aAdjMap[aK].empty())
            break;
    }
    StdOut() << "aK= " << aK << " " << aAdjMap[aK].size() << std::endl;
    for (auto toto : aAdjMap[aK])
        StdOut() << toto << std::endl;

    /// depth-first algorithm
    ///
    DFS(aK,aNVisited,aAdjMap);

    /// check if all nodes visited
    ///
    for (int aK=0; aK<aNumN; aK++)
        if (!aNVisited[aK] && !aAdjMap[aK].empty())
        {
            StdOut() << aK << std::endl;
            return false;
        }
    return true;
}




int cAppli_SfmInitWithPartition::Exe()
{
    if (ExeHParal())
        return EXIT_SUCCESS;

    //if (ExeHierarch())
      //  return EXIT_SUCCESS;

    return EXIT_FAILURE;
}

int cAppli_SfmInitWithPartition::ExeHParal()
{

    /// ================ Read data
    ///
    mPhProj.FinishInit();
    cTripletSet * aTriSet = mPhProj.ReadTriplets();

    ThreadPool aThreadP;

    /// ================ Initialise tree structure
    ///
    auto aRoot = std::make_shared<cNodeHTreeMT>(nullptr,0,0);
    aRoot->Init(*aTriSet);
    aRoot->Descend(aThreadP,aRoot);
    aThreadP.addNode(aRoot); //adds parent to tp

    aThreadP.Exec(NbThreadMax);

    StdOut() << "tree propagated " <<   std::endl;



    StdOut() << "partition ended " <<   std::endl;


    delete aTriSet;

    return EXIT_SUCCESS;
}


int cAppli_SfmInitWithPartition::ExeHierarch()
{
    /// ================ Read data
    ///
    mPhProj.FinishInit();
    cTripletSet * aTriSet = mPhProj.ReadTriplets();



    /// ================ Initialise tree structure
    /*
                               o root (r)
                             /   \
                 r->mPart0  o     o r->mPart1
                           / \   /  \
       r->mPart0->mPart0  o   o  o   o r->mPart1->mPart1
              r->mPart0->mPart1  r->mPart1->mPart0
    */
    StdOut() << " ================ Tree structure" << std::endl;
    auto aRoot = std::make_shared<cNodeHTree>(1);
    aRoot->mPart0 = std::make_shared<cNodeHTree>(2);
    aRoot->mPart1 = std::make_shared<cNodeHTree>(3);
    aRoot->mPart0->mPart0 = std::make_shared<cNodeHTree>(4);
    aRoot->mPart0->mPart1 = std::make_shared<cNodeHTree>(5);
    aRoot->mPart1->mPart0 = std::make_shared<cNodeHTree>(6);
    aRoot->mPart1->mPart1 = std::make_shared<cNodeHTree>(7);

    /// ================ Partition
    ///
    StdOut() << " ================ Partition 0" << std::endl;
    aRoot->Init(*aTriSet);
    aRoot->Partition();
    aRoot->PushPartition(*aTriSet);

    ///
    StdOut() << " ================ Partition 1" << std::endl;
    StdOut() << " partition left" << std::endl;
    aRoot->mPart0->Partition();
    aRoot->mPart0->PushPartition(*aTriSet);


    StdOut() << " partition right" << std::endl;
    aRoot->mPart1->Partition();
    aRoot->mPart1->PushPartition(*aTriSet);


    /// ================ Solve the tree
    StdOut() << " ================ Solve: " << std::endl;
    ///
    StdOut() << " ================    Spanning tree leaves" << std::endl;
    aRoot->mPart0->mPart0->SpanTree();
    aRoot->mPart0->mPart1->SpanTree();
    aRoot->mPart1->mPart0->SpanTree();
    aRoot->mPart1->mPart1->SpanTree();

    StdOut() << " ================    Move up & Align" << std::endl;
    aRoot->mPart0->Align();
    aRoot->mPart1->Align();

    ///
    StdOut() << " ================    Move Top & Align" << std::endl;
    aRoot->Align();



    delete aTriSet;

    return EXIT_SUCCESS;
}

int cAppli_SfmInitWithPartition::ExeKahyPar()
{
    /*mPhProj.FinishInit();

    /// set of input triplets
    cTripletSet * aTriSet = mPhProj.ReadTriplets();

    /// 1- initialise the hypergraph from a set of triplets
    ///
    cHyperGraph aHG;
    aHG.InitFromTriSet(aTriSet); */

    /// - run kahypar partitioner
    ///
    // Initialize thread pool
    mt_kahypar_initialize_thread_pool(
        std::thread::hardware_concurrency() /* use all available cores */,
        true /* activate interleaved NUMA allocation policy */ );



    mt_kahypar_context_t* aKP_context = mt_kahypar_context_new();
    //mt_kahypar_configure_context_from_file(aKP_context, mKPInitFile.c_str());
    mt_kahypar_load_preset(aKP_context, DEFAULT /* corresponds to MT-KaHyPar-D */);

    mt_kahypar_set_partitioning_parameters(aKP_context,
                        mNbParts /* number of blocks */, mImbalance /* imbalance parameter */,
                        KM1 /* objective function */);
    mt_kahypar_set_seed(42);

    /// Enable logging
    mt_kahypar_set_context_parameter(aKP_context, VERBOSE, "1");
    //aKP_context->partition.verbose_output = true;

    /// Load Hypergraph for DEFAULT preset
    mt_kahypar_hypergraph_t aHypergraph =
        mt_kahypar_read_hypergraph_from_file(mHMetisFile.c_str(),
          DEFAULT, HMETIS /* file format */);

    /// Partition Hypergraph
    mt_kahypar_partitioned_hypergraph_t aPartAll = mt_kahypar_partition(aHypergraph, aKP_context);

    /// Extract Partition
    std::unique_ptr<mt_kahypar_partition_id_t[]> aPartOne =
        std::make_unique<mt_kahypar_partition_id_t[]>(mt_kahypar_num_hypernodes(aHypergraph));
    mt_kahypar_get_partition(aPartAll, aPartOne.get());

    /// Extract Block Weights
    std::unique_ptr<mt_kahypar_hypernode_weight_t[]> aPartAll_weights =
        std::make_unique<mt_kahypar_hypernode_weight_t[]>(mNbParts);
    mt_kahypar_get_block_weights(aPartAll, aPartAll_weights.get());

    /// Compute Metrics
    const double aImbalance = mt_kahypar_imbalance(aPartAll, aKP_context);
    const double aCostKm1 = mt_kahypar_km1(aPartAll);

    /// print cut matrix
    ///
    //printCutMatrix(aPartAll);

     //printCutMatrix(aPartAll);
    //mt_kahypar::PartitionerFacade::serializeCSV(
    //            aPartAll, aKP_context, 10);
    std::cout << "Test connectivity:" << std::endl;
    //mt_kahypar_connectivity(aPartAll);
    //mt_kahypar::ds::ConnectivitySets
    //mt_kahypar::io::printCutMatrix<mt_kahypar_partitioned_hypergraph_t>(aPartAll);


    // Output Results
    std::cout << "Partitioning Results:" << std::endl;
    std::cout << "Imbalance         = " << aImbalance << std::endl;
    std::cout << "Km1               = " << aCostKm1 << std::endl;
    for (int aB=0; aB<mNbParts; aB++)
    {
        std::cout << "Weight of Block " << aB << " = " << aPartAll_weights[aB] << std::endl;
    }

    // Write partition to file
    mt_kahypar_write_partition_to_file(aPartAll,mPartOutFile.c_str());

    mt_kahypar_free_context(aKP_context);
    mt_kahypar_free_hypergraph(aHypergraph);
    mt_kahypar_free_partitioned_hypergraph(aPartAll);

    return EXIT_SUCCESS;
}

tMMVII_UnikPApli Alloc_SfmInitWithPartition(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_SfmInitWithPartition(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_SfmInitWithPartition
(
     "SfmInitWithPartition",
      Alloc_SfmInitWithPartition,
      "Compute initial orientations with graph partitioning",
      {eApF::Ori},
      {eApDT::TieP},
      {eApDT::Orient},
      __FILE__
);

}; // MMVII


/*   CHRISTOPHE
#include <iostream>
#include <vector>
#include <deque>
#include <chrono>
#include <thread>
#include <mutex>

using namespace std;

static constexpr int NbRuns = 100;
static constexpr int NbThreadMax  = 10;

// Classe "Tache de calcul"
class Calculus
{
public:
    Calculus() : n(nbInstance++) {}
    void run() {
        val=0;
        start = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        // Calcul stupide qui prend du temps ...
        for (int i=0; i<1000000; i++)
            val = val + (double)std::rand() / std::rand();
        end = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    }

    int n;          // Numero du calcul
    double val;
    std::time_t start,end;
private:
    static int nbInstance;  // Pour numeroter les taches
};

int Calculus::nbInstance = 0;


// Pool de threads ...
class ThreadPool
{
public:
    ThreadPool() {}
    void Exec(std::vector<Calculus>& calculus,int nbThread)
    {
        for (auto& c : calculus)
            CalculusQueue.push_back(&c);    // On creee une queue de pointeurs sur les taches a executer
        std::vector<std::thread> threadList;
        for (int i = 0; i < nbThread; ++i) // On lance NbThreads, chaque thread execute ExecLoop
            threadList.emplace_back(std::thread(&ThreadPool::ExecLoop, this));
        for (auto& t : threadList)
            t.join();                       // On attend la fin de tous les threads (donc de touts les taches)
    }

private:
    void ExecLoop()
    {
        while (true) {  // boucle infinie: on prend l'élément suivant du tableau et on l'execute.
            Calculus *c;
            {
                // On protege la liste des taches a executer contre l'execution en parallle des threads avec un lock
                std::lock_guard<std::mutex> lock(mMutex_CalculusQueue);
                if (CalculusQueue.empty())
                    return;             // Si plus de tache, on sort. On va rejoindre le "t.join()"
                c = CalculusQueue.front();
                CalculusQueue.pop_front();
            }
            c->run();
        }
    }

    std::deque<Calculus*> CalculusQueue;    // Liste des pointeurs sur les taches a executer
    std::mutex mMutex_CalculusQueue;        // Mutex pour proteger l'acces a CalculusQueue entre les threads
};



std::vector<Calculus> calculus(NbRuns);     // Tableau des taches a executer


int main()
{
    ThreadPool tp;
    tp.Exec(calculus, NbThreadMax);     // On execute toutes les taches, NbThreadMax en paralleles
    for (auto& c : calculus) {
        std::cout << c.n << " " << c.val << " " << c.start << " " << c.end << std::endl;
    }

    return 0;
}




*/



