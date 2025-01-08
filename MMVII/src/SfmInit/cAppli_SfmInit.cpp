﻿#include "MMVII_SfmInit.h"
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

void Bench_SfmInit();

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
    //StdOut() << "Quality=" << mQual << std::endl;
    //StdOut() << mIndex << " ";
    for (const auto& aV : Vertices())
    {
        StdOut() << aV->Pose().Name() << " ";
    }
    StdOut() << std::endl;
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

    for (const auto& aE : mAdjMap)
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
    for (const auto& anEdg : this->mAdjMap)
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


    for (const auto& aElem : mAdjMap)
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
                {                          //  r = R * rotation.T
                                           // tr = translation + L * rotation C
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
            //StdOut() << aK << std::endl;
            return false;
        }
    return true;
}

std::map<int,std::vector<int>> cHyperGraph::CreateAdjGr(bool pair_only)
{

    std::map<int,std::vector<int>> aAdj;


    std::vector<int> EdgeIt;
    if (pair_only)
        EdgeIt = {1}; //first edge
    else
        EdgeIt = {1,2}; // two edges



    for (auto aH : mVHEdges)
    {

        for (auto Edge : EdgeIt)
        {

            int aIm0Id = aH->Vertices()[0]->Id();
            int aIm1Id = aH->Vertices()[Edge]->Id();

            if (aAdj.find(aIm0Id) == aAdj.end())
                aAdj[aIm0Id] = std::vector<int>();

            // add the edge to adjacency
            aAdj[aIm0Id].push_back(aIm1Id);
            // both ways
            aAdj[aIm1Id].push_back(aIm0Id);

        }
    }


    // print adj map
    for (auto aV1 : aAdj)
    {
        StdOut() << aV1.first << " ";
        for (auto aV2 : aV1.second )
            StdOut() << aV2 << " ";
        StdOut() << std::endl;
    }


    return aAdj;
}

// randomly reduced hypergraph (graph must be a connected component)
cTripletSet  cHyperGraph::ReduceGraph(std::string& aName)
{
    cTripletSet aNewSet;
    aNewSet.SetName(aName);

    std::map<std::string,int> aCurEdges;
    std::map<std::string,int>::iterator aCurEdgesIt;

    for (auto edge : mAdjMap)
    {
        // number of triplets touching the edge
        int nbTri = edge.second.size();

        // pick a random triplet for that edge
        int aRandTri = round_down(RandUnif_0_1()*nbTri);

        // name of edge vertices
        std::string aImage0 = edge.first.StartVertex()->Pose().Name();
        std::string aImage1 = edge.first.EndVertex()->Pose().Name();

        // if the start vertex is the start vertex of the rand triplet
        if (aImage0==edge.second[aRandTri]->RelPose(0).Name())
        {

            int SecondImId = (aImage1==edge.second[aRandTri]->RelPose(1).Name()) ? 1 : 2;
            int ThirdImId = (aImage1==edge.second[aRandTri]->RelPose(1).Name()) ? 2 : 1;

            // do not add duplicates
            if (aCurEdges.find(aImage0+aImage1) == aCurEdges.end())
            {
                StdOut() << aImage0 << " " << edge.second[aRandTri]->RelPose(SecondImId).Name() << " "
                         << edge.second[aRandTri]->RelPose(ThirdImId).Name() << std::endl;

                aCurEdges[aImage0+aImage1] = 1;

                cTriplet aTi;
                std::vector<cView> aPoses {edge.second[aRandTri]->RelPose(0),
                                           edge.second[aRandTri]->RelPose(SecondImId),
                                           edge.second[aRandTri]->RelPose(ThirdImId)};
                aTi.PVec() = aPoses;
                aTi.Id() = edge.second[aRandTri]->Index();
                aTi.BH() = edge.second[aRandTri]->BsurH();
                aTi.Residual() = edge.second[aRandTri]->Quality();

                aNewSet.PushTriplet(aTi);
            }

        }

    }

    // make sure it remains a connected component
    //   use only first edge to check connectivity
    cHyperGraph aHG;
    aHG.InitFromTriSet(&aNewSet);
    std::map<int,std::vector<int>> aAdjMap = aHG.CreateAdjGr(true);
    bool IsConnected = aHG.CheckConnectivity(aAdjMap);
    StdOut() << "IsConnected? " << IsConnected << std::endl;

    /*
            // redefine the origin to edge beginning
            cDenseMatrix<double> rot2NewOrg(3,3,eModeInitImage::eMIA_Null);
            cPt3dr               tr2NewOrg(0,0,0);
            std::map<std::string,tPose> aNewTri;

            // find the transformation to the new origin
            for (auto aI : {0,1,2})
            {
                if (edge.second[aRandTri]->RelPose(aI).Name() == aImage0 )
                {
                    rot2NewOrg = edge.second[aRandTri]->RelPose(aI).Pose().Rot().Mat().Inverse();
                    tr2NewOrg = -rot2NewOrg * edge.second[aRandTri]->RelPose(aI).Pose().Tr();
                }
            }

            // apply the transformation to all three images of the triplet
            for (auto aI : {0,1,2})
            {
                cPt3dr CurTr = edge.second[aRandTri]->RelPose(aI).Pose().Tr();
                cPt3dr NewTr = tr2NewOrg + rot2NewOrg * CurTr;

                cDenseMatrix<double> CurRot = edge.second[aRandTri]->RelPose(aI).Pose().Rot().Mat();
                cDenseMatrix<double> NewRot = rot2NewOrg * CurRot;

                aNewTri[edge.second[aRandTri]->RelPose(aI).Name()] =
                        tPose(NewTr, cRotation3D<double>(NewRot,false));
            }

            for (auto coord : {0,1,2})
            {
                // fill the vector of coefficients
                SetVecT(aVectCoef,
                        edge.first.StartVertex()->Id()*TR_UNK_SZ,
                        edge.first.EndVertex()->Id()*TR_UNK_SZ,
                        aNbVertices*TR_UNK_SZ+EdgeCnt,
                        aNewTri[aImage1].Rot().Mat().Inverse(),
                        aNewTri[aImage1].Tr(),
                        coord);


                //add equation
                aSysTr->PublicAddObservation(1.0,aVectCoef,0.0);
                //reset coeff
                aVectCoef.DIm().InitNull();
            }*/


    return aNewSet;
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

    cDenseMatrix<double> aZRot(3,3,eModeInitImage::eMIA_Null);
    aZRot.SetElem(0,0,1);
    aZRot.SetElem(1,1,-1);
    aZRot.SetElem(2,2,-1);

    /// initialise the global poses of the seed triplet
    aGraph.GetVertex(aPose0Seed.Name())->Pose().Pose() =
            tPose(aPose0Seed.Pose().Tr() ,
            cRotation3D<double>(aPose0Seed.Pose().Rot().Mat() ,false)); //< set the global pose to be the local pose
    aGraph.GetVertex(aPose0Seed.Name())->FlagOriented() = true;
    aGraph.GetVertex(aPose1Seed.Name())->Pose().Pose() =
            tPose(aPose1Seed.Pose().Tr() ,
            cRotation3D<double>(aPose1Seed.Pose().Rot().Mat(),false));
    aGraph.GetVertex(aPose1Seed.Name())->FlagOriented() = true;
    aGraph.GetVertex(aPose2Seed.Name())->Pose().Pose() =
            tPose(aPose2Seed.Pose().Tr() ,
            cRotation3D<double>(aPose2Seed.Pose().Rot().Mat(),false));;
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
                std::vector<tPose> VPLoc = {aCurBestHE->RelPose(aIdCurP1Glob).Pose(),
                                            aCurBestHE->RelPose(aIdCurP2Glob).Pose()};
                std::vector<tPose> VPGlob = {aCurP1Glob,aCurP2Glob};
                cSimilitud3D<double> aSim = cAppli_SfmInitGlob::Loc2GlobTrafo(VPLoc,VPGlob);


                ///
                /// 5- Set the new global pose
                ///
                // R = r M^t
                cRotation3D<double> aNewRot( aCurBestHE->RelPose(aIdCurP3Glob).Pose().Rot().Mat() * aSim.Rot().Mat().Transpose() , true );
                cPt3dr aNewCenter = aSim.Tr() + aSim.Scale() * aSim.Rot().Mat() * aCurBestHE->RelPose(aIdCurP3Glob).Pose().Tr();


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
                            StdOut() << "Q=" << aCurP1Glob.Rot().ToWPK() << std::endl;
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
                            StdOut() << "Q=" << aCurP2Glob.Rot().ToWPK() << std::endl;
                            StdOut() << "tr= [" << aCurP2Glob.Tr().x() << "],["
                                                << aCurP2Glob.Tr().y() <<  "],["
                                                << aCurP2Glob.Tr().z() << "]" << std::endl;

                            StdOut() << "=====similarity" << std::endl;
                            aSim.Rot().Mat().Show();
                            StdOut() << "Tr=" << aSim.Tr() << ", L=" << aSim.Scale() << std::endl;
                            StdOut() << "=====new pose" << std::endl;
                            //aNewRot.Mat().Show();
                            StdOut() << "[";
                            for (int aK1=0; aK1<3; aK1++)
                            {
                                StdOut() << "[";
                                for (int aK2=0; aK2<3; aK2++)
                                {
                                    StdOut() << aNewRot.Mat()(aK1,aK2) << ((aK2<2) ? ", " : " ");
                                }
                                StdOut() << "]," << std::endl;
                            }
                            StdOut() << "Q=" << aNewRot.ToWPK() << std::endl;
                            StdOut() << "tr=" << aNewCenter << std::endl;

                            if( (aCurBestHE->RelPose(aIdCurP1Glob).Name() == "image_002_00103.tif") &&
                                (aCurBestHE->RelPose(aIdCurP2Glob).Name() == "image_002_00104.tif") &&
                                    (aNewPoseName=="image_002_00105.tif"))
                                getchar();

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

    int CntUnOri=0;
    StdOut() << "#F=N X Y Z W P K" << std::endl;
    const std::unordered_map<std::string,cVertex*> aMapV = aGraph.GetMapVertices();
    for (auto aV : aMapV)
    {
        if (aV.second->FlagOriented())
        {
            cPt3dr aFlippedC = aV.second->Pose().Pose().Tr() * aZRot; //micmac convention
            cPt3dr aWPK = aV.second->Pose().Pose().Rot().ToWPK();
            StdOut() << aV.first << " " << aFlippedC.x() << " "
                     << aFlippedC.y() << " "
                     << aFlippedC.z() << " "
                     << aWPK[0] << " " << aWPK[1] << " " << aWPK[2] << std::endl;
        }
        else
            CntUnOri++;
    }
    StdOut() << "# Unoriented " << CntUnOri << std::endl;
    StdOut() << "==END==" << std::endl;


}

int cAppli_SfmInitFromGraph::Exe()
{
     mPhProj.FinishInit();

     /// set of input triplets
     cTripletSet * aTriSet = mPhProj.ReadTriplets();


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

     StdOut() << "END OF PRGM" << std::endl;

     //< save the graph in dot format
     //std::string aDotName = mPhProj.DirPhp() + "test.dot";
     //aHG.SaveDotFile(aDotName);

     /*/// free memory
     for (auto aIt : aMapIm2V)
         delete aIt.second;*/

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
    mSourceInit  (100.0),
    mSinkInit    (1.0),
    mNbDepthMax  (4),
    mNbThreadMax (8),
    mNbParts     (4),
    mImbalance   (0.03),
    mPartOutFile ("kah_parts_out.hgr")
{
}


cCollecSpecArg2007 & cAppli_SfmInitWithPartition::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
              <<  mPhProj.DPOriTriplets().ArgDirInMand()
          //    <<  Arg2007(mHMetisFile,"Hypergraph file in hmetis format",{eTA2007::FileAny})
              <<  Arg2007(mTGraphFile,"Triplet graph file in hmetis format",{eTA2007::FileAny})
           ;
}

cCollecSpecArg2007 & cAppli_SfmInitWithPartition::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return    anArgOpt
           << AOpt2007(mNbParts,"NbParts", "Number of partitions; def=4")
           << AOpt2007(mSourceInit,"SourceInit","Initialise source weight, Def=1.0")
           << AOpt2007(mSinkInit,"SinkInit","Initialise sink weight, Def=1.0")
           << AOpt2007(mNbDepthMax,"NbDepth","Maximum tree depth, Def=4")
           << AOpt2007(mNbThreadMax,"NbThread","Maximum number of threads, Def=8")
     //      << AOpt2007(mImbalance,"Imb","Partition size imbalance, Def=0.03")
           << AOpt2007(mPartOutFile,"PartOutput","Write partition to a file (filename)")
           << AOpt2007(mHMetisFile,"HMetis","Hypergraph file in hmetis format")
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
    
    ThreadPool aThreadP(mNbThreadMax);

    /// ================ Initialise tree structure
    ///
    auto aRoot = std::make_shared<cNodeHTreeMT>(nullptr,0,0,mSourceInit,mSinkInit);
    aRoot->Init(*aTriSet);
    aRoot->BuildChildren(aThreadP,aRoot,mNbDepthMax);
    aThreadP.addNode(aRoot);
    
    StdOut() << "PARTITION " <<   std::endl;
    aThreadP.ExecDown(*aTriSet);
    aRoot->Show();

    StdOut() << "ALIGN " <<   std::endl;
    aThreadP.ExecUp();

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


/* ********************************************************** */
/*                                                            */
/*                 cAppli_SfmInitGlob                         */
/*                                                            */
/* ********************************************************** */



cAppli_SfmInitGlob::cAppli_SfmInitGlob(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mSeed        (1),
    mNbTriEdge   (2),
    mEqTr        (nullptr),
    mSys         (nullptr),
    mRunL2       (false)
{
}


cCollecSpecArg2007 & cAppli_SfmInitGlob::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
              <<  mPhProj.DPOriTriplets().ArgDirInMand()
          //    <<  Arg2007(mHMetisFile,"Hypergraph file in hmetis format",{eTA2007::FileAny})
           ;
}

cCollecSpecArg2007 & cAppli_SfmInitGlob::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return    anArgOpt
           << AOpt2007(mSeed,"Seed", "Triplet that sets the origin (gauge); def=random")
           << AOpt2007(mNbTriEdge,"NbEdge", "Number of triplet edges used (1 or 2); def=2")
           << AOpt2007(mRunL2,"L2", "L2 optimisation, by default L1")
           << mPhProj.DPOriTriplets().ArgDirOutOpt("OriOut","Output triplets",true)
              ;
}


void cAppli_SfmInitGlob::SetVecT(cDenseVect<tREAL8>& aVC,
                                 tU_INT4 aId1, tU_INT4 aId2, tU_INT4 aIdLambda,
                                 const cMatrix<double>& rot, const cPt3dr& tr, int coord )
{
    aVC(aId1+coord) = 1.0;

    aVC(aIdLambda) = -tr[coord];

    // in MM convention, the 1st coordinate is the col
    aVC(aId2) = -rot(0,coord);
    aVC(aId2+1) = -rot(1,coord);
    aVC(aId2+2) = -rot(2,coord);
  /*  aVC(aId2) = -rot(coord,0);
    aVC(aId2+1) = -rot(coord,1);
    aVC(aId2+2) = -rot(coord,2); */

}

void cAppli_SfmInitGlob::SetVecQ(cDenseVect<tREAL8>& aVC,
                                 tU_INT4 aId1, tU_INT4 aId2,
                                 const cPt4dr& q, int coord)
{
    aVC(aId1+coord) = 1;
    aVC(aId2) = -q[coord];

    if (coord==0)
    {
        aVC(aId2+1) = q[1];
        aVC(aId2+2) = q[2];
        aVC(aId2+3) = q[3];
    }
    else if (coord==1)
    {
        aVC(aId2+1) = -q[0];
        aVC(aId2+2) = q[3];
        aVC(aId2+3) = -q[2];
    }
    else if (coord==2)
    {
        aVC(aId2+1) = -q[3];
        aVC(aId2+2) = -q[0];
        aVC(aId2+3) = q[1];
    }
    else if (coord==3)
    {
        aVC(aId2+1) = q[2];
        aVC(aId2+2) = -q[1];
        aVC(aId2+3) = -q[0];
    }

}

int cAppli_SfmInitGlob::Exe()
{

    Bench_SfmInit();
    getchar();

    mPhProj.FinishInit();

    /// set of input triplets
    cTripletSet * aTriSet = mPhProj.ReadTriplets();

    /// 1- initialise the hypergraph from a set of triplets
    ///
    cHyperGraph aHG;
    aHG.InitFromTriSet(aTriSet);

    //RobustTriplets(aHG);

    //std::string aNSetName = mPhProj.DirPhp() + mPhProj.DPOriTriplets().DirOut() + "/RedGraph";
    //cTripletSet aRedSet = aHG.ReduceGraph(aNSetName);
    //aRedSet.ToFile(aNSetName+".xml");

    getchar();

    //CheckRelOri(aHG);
    Solve(aHG);

    /// Print output poses
    for (auto aV : aHG.GetMapVertices())
    {
        cPt3dr aWPK = aV.second->Pose().Pose().Rot().ToWPK();
        StdOut() << aV.first << " " << aV.second->Pose().Pose().Tr().x() << " "
                 << aV.second->Pose().Pose().Tr().y() << " "
                 << aV.second->Pose().Pose().Tr().z() << " "
                 << aWPK[0] << " " << aWPK[1] << " " << aWPK[2] << std::endl;
       // aV.second->Pose().Pose().Rot().Mat().Show();
    }

    delete aTriSet;

    return EXIT_SUCCESS;
}

cSimilitud3D<double> cAppli_SfmInitGlob::Loc2GlobTrafo(const std::vector<tPose>& VPLoc,const std::vector<tPose>& VPGlob)
{


    ///
    /// 4- Calculate the transformation that takes from local to global frame
    ///  C = Tr + lambda * M * c   ==>  Tr = C - lambda * M * c
    ///  M = R^t * r => R = r * M^t
    ///  lambda = distnace_global / distance_local
    ///
    cDenseMatrix<double> aM_  =  VPGlob[0].Rot().Mat().Transpose() * VPLoc[0].Rot().Mat();
    cDenseMatrix<double> aM__ =  VPGlob[1].Rot().Mat().Transpose() * VPLoc[1].Rot().Mat();
    cDenseMatrix<double> aM = (0.5*(aM_+aM__)).ClosestOrthog();//

    double aLambda = Norm2( VPGlob[1].Tr() - VPGlob[0].Tr() ) /
                     Norm2( VPLoc[1].Tr() - VPLoc[0].Tr());

    cPt3dr aTr_  =   VPGlob[0].Tr() - aLambda * aM * VPLoc[0].Tr();
    cPt3dr aTr__ =   VPGlob[1].Tr() - aLambda * aM * VPLoc[1].Tr();
    cPt3dr aTr = 0.5*(aTr_+aTr__);

    cSimilitud3D<double> aSimil(aLambda,aTr,cRotation3D<double>(aM,false));


    return aSimil;
}

/*
    for every edge:
        find reference triplet/edge
            for every adjacent triplet
                comptute the distance between all other triplets in that edge
        take the triplet with best score as reference
        you can save this one triplet per edge as a minim graph
        otherwise, compute again scores of all triplets to the refernece triplpet
        save first two or three triplets conditioned on error


*/

void cAppli_SfmInitGlob::RobustTriplets(cHyperGraph& aHG)
{
    for (auto edge : aHG.AdjMap())
    {
        int aNbTri =  edge.second.size();
        double TotalRotDist=0;
        double TotalTrDist=0;

        int edgeId0 = edge.first.StartVertex()->Id();
        int edgeId1 = edge.first.EndVertex()->Id();

        StdOut() << edgeId0 << " " << edge.first.StartVertex()->Pose().Name() << " "
                 << edgeId1 << " " << edge.first.EndVertex()->Pose().Name()
                 << ", # triplets= " << aNbTri << " ";

        // pick a random reference edge
        int aRandTri = round_down(RandUnif_0_1()*aNbTri);
        std::vector<tPose> VPGlob(2);
        for (int aV : {0,1,2})
        {
            if (edge.second[aRandTri]->Vertices()[aV]->Id()==edgeId0)
                VPGlob[0] = edge.second[aRandTri]->RelPose(aV).Pose();
            else if (edge.second[aRandTri]->Vertices()[aV]->Id()==edgeId1)
                VPGlob[1] = edge.second[aRandTri]->RelPose(aV).Pose();
        }


        /// iterate over all triplets adjacent to the current edge
        ///
        int triCnt=0;
        for (auto aT : edge.second)
        {

            //identify sommets in aT that correspond to the current edge
            int VId0=1e5;
            int VId1=1e5;
            for (int aV : {0,1,2})
            {
                if (aT->Vertices()[aV]->Id()==edgeId0)
                    VId0 = aV;
                else if (aT->Vertices()[aV]->Id()==edgeId1)
                    VId1 = aV;
            }

            //compute relative orientation between VId0 and VId1
            //
            std::vector<tPose> VPLoc = {aT->RelPose(VId0).Pose(),
                                        aT->RelPose(VId1).Pose()};


            cSimilitud3D<double> aSim = cAppli_SfmInitGlob::Loc2GlobTrafo(VPLoc,VPGlob);



            double RotDist = 0.0;
            double TrDist = 0.0;

            //accumulate errors for each vertex of the edge
            for (auto i : {0,1})
            {

                cDenseMatrix<double> RLocInGlob = VPLoc[i].Rot().Mat() * aSim.Rot().Mat().Transpose();
                cPt3dr TrLocInGlob = aSim.Tr() + aSim.Scale() * aSim.Rot().Mat() * VPLoc[i].Tr();

                RotDist += RLocInGlob.L2Dist(VPGlob[i].Rot().Mat());

                if ( (TrLocInGlob!=cPt3dr(0,0,0)) && VPGlob[i].Tr()!=cPt3dr(0,0,0))
                    TrDist += DistBase(TrLocInGlob,VPGlob[i].Tr());  //DistBase(aTrGInL,aVPoseLoc[aI].Tr()) * aTri->BsurH();
            }

           // StdOut() << "\t" << triCnt << " RotDist=" << RotDist << ", TrDist=" << TrDist << std::endl;

            TotalRotDist+=RotDist;
            TotalTrDist+=TrDist;

            triCnt++;
        }
        StdOut() << "\t" << " RotDist=" << TotalRotDist << ", TrDist=" << TotalTrDist << std::endl;

    }

}

void cAppli_SfmInitGlob::CheckRelOri(cHyperGraph& aHG)
{
    int EdgeCnt=0;
    for (auto edge : aHG.AdjMap())
    {
        int aNbTri =  edge.second.size();

        // pick a random triplet for that edge
        int aRandTri = round_down(RandUnif_0_1()*aNbTri);

        StdOut() << edge.first.StartVertex()->Id() << " " << edge.first.StartVertex()->Pose().Name() << " "
                 << edge.first.EndVertex()->Id() << " " << edge.first.EndVertex()->Pose().Name()
                 << ", # triplets= " << aNbTri << " "
                 << ", rand=" << aRandTri << std::endl;



        StdOut() << edge.second[aRandTri]->RelPose(0).Name() << " "
                 << edge.second[aRandTri]->RelPose(1).Name() << " "
                 << edge.second[aRandTri]->RelPose(2).Name() << std::endl;

        StdOut() << edge.second[aRandTri]->Vertices()[0]->Id() << " "
                 << edge.second[aRandTri]->Vertices()[1]->Id() << " "
                 << edge.second[aRandTri]->Vertices()[2]->Id() << std::endl;


        cDenseMatrix<double> r0 = edge.second[aRandTri]->RelPose(0).Pose().Rot().Mat();
        cDenseMatrix<double> r01 = edge.second[aRandTri]->RelPose(1).Pose().Rot().Mat();
        cDenseMatrix<double> r02 = edge.second[aRandTri]->RelPose(2).Pose().Rot().Mat();

        cPt3dr c0 = edge.second[aRandTri]->RelPose(0).Pose().Tr();
        cPt3dr c01 = edge.second[aRandTri]->RelPose(1).Pose().Tr();
        cPt3dr c02 = edge.second[aRandTri]->RelPose(2).Pose().Tr();

        // express relative rotation with RelPose(1) at origin
        cDenseMatrix<double> r10 = r01.Inverse();
        cDenseMatrix<double> r12 = r01.Inverse() * r02;


        cDenseMatrix<double> cycleTest = (r01 * r12) * r02.Inverse();

        StdOut() << "Cycle ROT" << std::endl;
        cycleTest.Show();

        // make r02,c02 the origin
        cDenseMatrix<double> rotation = r02.Inverse();
        cPt3dr translation = -rotation * c02;

        cDenseMatrix<double> r0New = rotation * r0;
        cDenseMatrix<double> r01New = rotation * r01;
        cDenseMatrix<double> r02New = rotation * r02;

        cPt3dr c0New = translation + rotation * c0;
        cPt3dr c01New = translation + rotation * c01;
        cPt3dr c02New = translation + rotation * c02;

        StdOut() << c0 << " " << c01 << " " << c02 << std::endl;
        StdOut() << c0New << " " << c01New << " " << c02New << std::endl;

        EdgeCnt++;

        getchar();

    }
}


void cAppli_SfmInitGlob::Solve(cHyperGraph& aHG)
{
    tU_INT4 aNbTriplets = aHG.NbHEdges();
    tU_INT4 aNbVertices = aHG.NbVertices();

    int TR_UNK_SZ = 3;
    int ROT_UNK_SZ = 4;

    tU_INT4 aNbUnkTr = TR_UNK_SZ*aNbVertices + aNbTriplets;
    tU_INT4 aNbUnkRot = ROT_UNK_SZ*aNbVertices;

    tU_INT4 aNbObsTr = TR_UNK_SZ*aNbTriplets*mNbTriEdge;
    tU_INT4 aNbObsRot = ROT_UNK_SZ*aNbTriplets*mNbTriEdge;

    StdOut() << "# triplets " << aNbTriplets << ", vertices " <<  aNbVertices << std::endl;
    StdOut() << "# unknows related to Tr=" << aNbUnkTr << ", rot=" << aNbUnkRot << std::endl;
    StdOut() << "# obs related to Tr=" << aNbObsTr << ", rot=" << aNbObsRot << std::endl;

    cLinearOverCstrSys<tREAL8> *  aSysTr = nullptr;

    if (mRunL2)
        aSysTr = cLinearOverCstrSys<tREAL8>::AllocSSR(eModeSSR::eSSR_LsqNormSparse,aNbUnkTr);
    else
        aSysTr = AllocL1_Barrodale<tREAL8>(aNbUnkTr);
    aSysTr->PublicReset();

    cLinearOverCstrSys<tREAL8> *  aSysRot = nullptr;
    if (mRunL2)
        aSysRot = cLinearOverCstrSys<tREAL8>::AllocSSR(eModeSSR::eSSR_LsqNormSparse,aNbUnkRot);
    else
        aSysRot = AllocL1_Barrodale<tREAL8>(aNbUnkRot);
    aSysRot->PublicReset();

    //mSys

    // M poses, N triplets
    // vector of unknowns for translation:
    //     image1         image M     tri1        triN
    // [ C1x C1y C1z ... CMx CMy CMz lambda1 .. lambdaN ]
    //
    // vector of unknowns for rotations:
    //        image1             image M
    // [ Q1w Q1x Q1y Q1z ... QMw QMx QMy QMz ]

    // P0 = p01 * P1
    //     = [lambda c, r] [C1, R1]
    //     = [lambda c + r * C1, r * R1 ]

    cDenseVect<tREAL8> aSolTr(aNbUnkTr,eModeInitImage::eMIA_Null);
    cDenseVect<tREAL8> aSolRot(aNbUnkRot,eModeInitImage::eMIA_Null);

    std::string TRANSLATION="TR";
    std::string ROTATION="ROT";

    std::vector<int> EdgeNums = (mNbTriEdge==1) ? std::vector<int>{1} : std::vector<int>{1,2};

    std::map<std::string,int> AVOIDDUPL;

    cDenseVect<tREAL8> aVectCoefTr(aNbUnkTr,eModeInitImage::eMIA_Null);
    cDenseVect<tREAL8> aVectCoefR(aNbUnkRot,eModeInitImage::eMIA_Null);

    for (tU_INT4 aT=0; aT<aNbTriplets; aT++)
    {

        //get indices to know where to input the values
        std::string aN1 = aHG.GetHyperEdge(aT)->RelPose(0).Name();
        tU_INT4 aIdGlob1 = aHG.GetVertex(aN1)->Id();
        std::string aN2 = aHG.GetHyperEdge(aT)->RelPose(1).Name();
        tU_INT4 aIdGlob2 = aHG.GetVertex(aN2)->Id();
        std::string aN3 = aHG.GetHyperEdge(aT)->RelPose(2).Name();
        tU_INT4 aIdGlob3 = aHG.GetVertex(aN3)->Id();
        std::vector<tU_INT4> aIdGlobVec = {aIdGlob1,aIdGlob2,aIdGlob3};

        StdOut() << aN1 << " " << aN2 << " " << aN3 << std::endl;

        for (auto aEdge : EdgeNums) //  add an edge of the triplet
        {

            cPt3dr tr = aHG.GetHyperEdge(aT)->RelPose(aEdge).Pose().Tr();
            cDenseMatrix<double> rot = aHG.GetHyperEdge(aT)->RelPose(aEdge).Pose().Rot().Mat().Transpose();
            cPt4dr q = MatrRot2Quat(rot);


            for (auto aC : {0,1,2}) //add each translation coordinate
            {

                // fill the vector of coefficients
                SetVecT(aVectCoefTr,
                        aIdGlobVec[0]*TR_UNK_SZ,
                        aIdGlobVec[aEdge]*TR_UNK_SZ,
                        aNbVertices*TR_UNK_SZ+aT,
                        rot,
                        tr,
                        aC);

                // for (auto i : aVectCoef.ToStdVect())
                //     StdOut() << i << std::endl;
                // getchar();

                //add equation
                aSysTr->PublicAddObservation(1.0,aVectCoefTr,0.0);
                //reset coeff
                aVectCoefTr.DIm().InitNull();


            }
            // avoid duplicates for angles Barrodale does not like it
            //std::string CurEdg = aHG.GetHyperEdge(aT)->RelPose(0).Name()+
            //                     aHG.GetHyperEdge(aT)->RelPose(aEdge).Name();
            //if (AVOIDDUPL.find(CurEdg)==AVOIDDUPL.end())
            {
            //    AVOIDDUPL[CurEdg] = 1;
                for (auto aC : {0,1,2,3}) //add each quaternion coordinate
                {

                    SetVecQ(aVectCoefR,
                            aIdGlobVec[0]*ROT_UNK_SZ,
                            aIdGlobVec[aEdge]*ROT_UNK_SZ,
                            q, aC);

                    // for (auto i : aVectCoef.ToStdVect())
                    //     StdOut() << i << std::endl;
                    // getchar();

                    //add equation
                    aSysRot->PublicAddObservation(1,aVectCoefR,0.0);
                    //reset coeff
                    aVectCoefR.DIm().InitNull();
                }
            }
           // else
           //     StdOut() << "rot dupl" << std::endl;
        }
    }
    // fix origin origin
    aSysTr->AddObsFixVar(1,0,0);
    aSysTr->AddObsFixVar(1,1,0);
    aSysTr->AddObsFixVar(1,2,0);
    // fix  lambda
    aSysTr->AddObsFixVar(1,aNbVertices*TR_UNK_SZ,1);
    // first  rotation
    aSysRot->AddObsFixVar(1,0,1);
    aSysRot->AddObsFixVar(1,1,0);
    aSysRot->AddObsFixVar(1,2,0);
    aSysRot->AddObsFixVar(1,3,0);

    aSolTr = aSysTr->PublicSolve();
    StdOut()  << "solved tr" << std::endl;

    for (int aUT=0; aUT<int(aNbUnkTr); aUT++)
    {
        if (aUT < int(aNbUnkTr-aNbTriplets))
        {
            if ((aUT % 3) == 2)
                StdOut() << aSolTr(aUT) << " 0 0 0" << std::endl;
            else
                StdOut() << aSolTr(aUT) << " ";
        }
        else
            StdOut() << aSolTr(aUT) << " "; //lambda
    }

    aSolRot = aSysRot->PublicSolve();
    StdOut()  << "solved rot" << std::endl;

    StdOut() << ROTATION << std::endl;
    for (int aUR=0; aUR<int(aNbUnkRot); aUR++)
    {
        if ((aUR % 4) == 3)
            StdOut() << aSolRot(aUR) << std::endl;
        else
            StdOut() << aSolRot(aUR) << " ";
    }

    // invert the poses & save
    for (auto aV : aHG.GetMapVertices())
    {

        // retrieve the solution
        cPt3dr aTr = {aSolTr(aV.second->Id()*TR_UNK_SZ),
                    aSolTr(aV.second->Id()*TR_UNK_SZ+1),
                    aSolTr(aV.second->Id()*TR_UNK_SZ+2)};
        cPt4dr aQ = {aSolRot(aV.second->Id()*ROT_UNK_SZ),
                     aSolRot(aV.second->Id()*ROT_UNK_SZ+1),
                     aSolRot(aV.second->Id()*ROT_UNK_SZ+2),
                     aSolRot(aV.second->Id()*ROT_UNK_SZ+3)};

        // compute the inverse of rot and tr
        // tr = -R^-1 C => C = -R * tr = -rot^-1 * tr
        //rot = R^-1    => R = rot^-1
        cDenseMatrix<double> aZRot(3,3,eModeInitImage::eMIA_Null);
        aZRot.SetElem(0,0,1);
        aZRot.SetElem(1,1,-1);
        aZRot.SetElem(2,2,-1);


/*       micmac worked with that before changing inverse for mats and order in SetT
 *
 *        cRotation3D<double> aRotInv(Quat2MatrRot(aQ).Inverse(),true);
        cRotation3D<double> aRot(aRotInv.Mat().Transpose(),true);

        cPt3dr aCInv = -aRotInv.Mat() * aC;
        aCInv = aCInv *  aZRot;  //for viabon

        // save the output pose
        cView aOutPose( tPose(aCInv,aRot), aV.second->Pose().Name() );
        aV.second->SetPose(aOutPose);*/
        cRotation3D<double> aRotInv(Quat2MatrRot(aQ).Inverse(),true);
        cRotation3D<double> aRot(aRotInv.Mat().Transpose(),true);

        cPt3dr aCInv = -aRotInv.Mat() * aTr;
        //flip the pose
        aCInv = aCInv *  aZRot;

        // save the output pose
        cView aOutPose( tPose(aCInv,aRot), aV.second->Pose().Name() );
        aV.second->SetPose(aOutPose);

    }

    delete aSysTr;
    delete aSysRot;
}


/*void cAppli_SfmInitGlob::SolveL1Parallel(cHyperGraph& aHG)
{
    tU_INT4 aNbTriplets = aHG.NbHEdges();
    tU_INT4 aNbVertices = aHG.NbVertices();

    int TR_UNK_SZ = 3;
    int ROT_UNK_SZ = 4;

    tU_INT4 aNbUnkTr = TR_UNK_SZ*aNbVertices + aNbTriplets;
    tU_INT4 aNbUnkRot = ROT_UNK_SZ*aNbVertices;

    tU_INT4 aNbObsTr = TR_UNK_SZ*aNbTriplets*mNbTriEdge;
    tU_INT4 aNbObsRot = ROT_UNK_SZ*aNbTriplets*mNbTriEdge;

    StdOut() << "# triplets " << aNbTriplets << ", vertices " <<  aNbVertices << std::endl;
    StdOut() << "# unknows related to Tr=" << aNbUnkTr << ", rot=" << aNbUnkRot << std::endl;
    StdOut() << "# obs related to Tr=" << aNbObsTr << ", rot=" << aNbObsRot << std::endl;
    cLinearOverCstrSys<tREAL8> *  aSysTr = AllocL1_Barrodale<tREAL8>(aNbUnkTr);
    aSysTr->PublicReset();
    cLinearOverCstrSys<tREAL8> *  aSysRot = AllocL1_Barrodale<tREAL8>(aNbUnkRot);
    aSysRot->PublicReset();

    // M poses, N triplets
    // vector of unknowns for translation:
    //     image1         image M     tri1        triN
    // [ C1x C1y C1z ... CMx CMy CMz lambda1 .. lambdaN ]
    //
    // vector of unknowns for rotations:
    //        image1             image M
    // [ Q1w Q1x Q1y Q1z ... QMw QMx QMy QMz ]

    cDenseVect<tREAL8> aSolTr(aNbUnkTr,eModeInitImage::eMIA_Null);
    cDenseVect<tREAL8> aSolRot(aNbUnkRot,eModeInitImage::eMIA_Null);

    std::string TRANSLATION="TR";
    std::string ROTATION="ROT";

    std::vector<int> EdgeNums = (mNbTriEdge==1) ? std::vector<int>{1} : std::vector<int>{1,2};


    // solve  L1
    for (auto aType : {TRANSLATION,ROTATION})
    {

        if (aType == "TR")
        {

            StdOut() << TRANSLATION << std::endl;

            cDenseVect<tREAL8> aVectCoef(aNbUnkTr,eModeInitImage::eMIA_Null);

            for (tU_INT4 aT=0; aT<aNbTriplets; aT++)
            {

                //get indices to know where to input the values
                std::string aN1 = aHG.GetHyperEdge(aT)->RelPose(0).Name();
                tU_INT4 aIdGlob1 = aHG.GetVertex(aN1)->Id();
                std::string aN2 = aHG.GetHyperEdge(aT)->RelPose(1).Name();
                tU_INT4 aIdGlob2 = aHG.GetVertex(aN2)->Id();
                std::string aN3 = aHG.GetHyperEdge(aT)->RelPose(2).Name();
                tU_INT4 aIdGlob3 = aHG.GetVertex(aN3)->Id();
                std::vector<tU_INT4> aIdGlobVec = {aIdGlob1,aIdGlob2,aIdGlob3};

                //StdOut() << "names/ids" << std::endl;
                StdOut() << aN1 << " " << aN2 << " " << aN3 << std::endl;
                //StdOut() << aIdGlob1 << " " << aIdGlob2 << " " << aIdGlob3 << std::endl;

                for (auto aEdge : EdgeNums) //  add an edge of the triplet
                {

                    for (auto aC : {0,1,2}) //add each translation coordinate
                    {
                        // fill the vector of coefficients
                        SetVecT(aVectCoef,
                                aIdGlobVec[0]*TR_UNK_SZ,
                                aIdGlobVec[aEdge]*TR_UNK_SZ,
                                aNbVertices*TR_UNK_SZ+aT,
                                aHG.GetHyperEdge(aT)->RelPose(aEdge).Pose().Rot().Mat().Inverse(),
                                aHG.GetHyperEdge(aT)->RelPose(aEdge).Pose().Tr(),
                                aC);

                       // for (auto i : aVectCoef.ToStdVect())
                       //     StdOut() << i << std::endl;
                       // getchar();

                        //add equation
                        aSysTr->PublicAddObservation(1.0,aVectCoef,0.0);
                        //reset coeff
                        aVectCoef.DIm().InitNull();


                    }
                }
            }
            // fix origin origin
            aSysTr->AddObsFixVar(1,0,0);
            aSysTr->AddObsFixVar(1,1,0);
            aSysTr->AddObsFixVar(1,2,0);
            // fix  lambda
            aSysTr->AddObsFixVar(1,aNbVertices*TR_UNK_SZ,1);

            aSolTr = aSysTr->PublicSolve();

            StdOut()  << "solved" << std::endl;

            for (int aUT=0; aUT<int(aNbUnkTr); aUT++)
            {
                if (aUT < int(aNbUnkTr-aNbTriplets))
                {
                    if ((aUT % 3) == 2)
                        StdOut() << aSolTr(aUT) << " 0 0 0" << std::endl;
                    else
                        StdOut() << aSolTr(aUT) << " ";
                }
                else
                    StdOut() << aSolTr(aUT) << " "; //lambda
            }

        }
        else if (aType == "ROT")
        {
            cDenseVect<tREAL8> aVectCoef(aNbUnkRot,eModeInitImage::eMIA_Null);

            StdOut() << ROTATION << std::endl;

            for (tU_INT4 aT=0; aT<aNbTriplets; aT++)
            {

                //get indices to know where to input the values
                std::string aN1 = aHG.GetHyperEdge(aT)->RelPose(0).Name();
                tU_INT4 aIdGlob1 = aHG.GetVertex(aN1)->Id();
                std::string aN2 = aHG.GetHyperEdge(aT)->RelPose(1).Name();
                tU_INT4 aIdGlob2 = aHG.GetVertex(aN2)->Id();
                std::string aN3 = aHG.GetHyperEdge(aT)->RelPose(2).Name();
                tU_INT4 aIdGlob3 = aHG.GetVertex(aN3)->Id();
                std::vector<tU_INT4> aIdGlobVec = {aIdGlob1,aIdGlob2,aIdGlob3};


                for (auto aEdge : EdgeNums) //add an edge of the triplet
                {
                    cPt4dr q_rel = MatrRot2Quat(aHG.GetHyperEdge(aT)->RelPose(aEdge).Pose().Rot().Mat().Inverse());//.Inverse()

                    for (auto aC : {0,1,2,3}) //add each quaternion coordinate
                    {

                        SetVecQ(aVectCoef,
                                aIdGlobVec[0]*ROT_UNK_SZ,
                                aIdGlobVec[aEdge]*ROT_UNK_SZ,
                                q_rel, aC);

                       // for (auto i : aVectCoef.ToStdVect())
                       //     StdOut() << i << std::endl;
                       // getchar();

                        //add equation
                        aSysRot->PublicAddObservation(1,aVectCoef,0.0);
                        //reset coeff
                        aVectCoef.DIm().InitNull();
                    }
                }
            }
            // first  rotation
            aSysRot->AddObsFixVar(1,0,1);
            aSysRot->AddObsFixVar(1,1,0);
            aSysRot->AddObsFixVar(1,2,0);
            aSysRot->AddObsFixVar(1,3,0);

            aSolRot = aSysRot->PublicSolve();

            StdOut()  << "solved" << std::endl;

            StdOut() << ROTATION << std::endl;
            for (int aUR=0; aUR<int(aNbUnkRot); aUR++)
            {
                if ((aUR % 4) == 3)
                    StdOut() << aSolRot(aUR) << std::endl;
                else
                    StdOut() << aSolRot(aUR) << " ";
            }
        }
    }

    // invert the poses & save
    for (auto aV : aHG.GetMapVertices())
    {

        // retrieve the solution
        cPt3dr aC = {aSolTr(aV.second->Id()*TR_UNK_SZ),
                    aSolTr(aV.second->Id()*TR_UNK_SZ+1),
                    aSolTr(aV.second->Id()*TR_UNK_SZ+2)};
        cPt4dr aQ = {aSolRot(aV.second->Id()*ROT_UNK_SZ),
                     aSolRot(aV.second->Id()*ROT_UNK_SZ+1),
                     aSolRot(aV.second->Id()*ROT_UNK_SZ+2),
                     aSolRot(aV.second->Id()*ROT_UNK_SZ+3)};

        // compute the inverse of rot and tr
        // tr = -R^-1 C
        //rot = R^-1
        cDenseMatrix<double> aZRot(3,3,eModeInitImage::eMIA_Null);
        aZRot.SetElem(0,0,1);
        aZRot.SetElem(1,1,-1);
        aZRot.SetElem(2,2,-1);


        cRotation3D<double> aRotInv(Quat2MatrRot(aQ).Inverse(),true);
        cRotation3D<double> aRot(aRotInv.Mat().Transpose(),true);

        cPt3dr aCInv = -aRotInv.Mat() * aC;
        aCInv = aCInv * aZRot;

        // save the output pose
        cView aOutPose( tPose(aCInv,aRot), aV.second->Pose().Name() );
        aV.second->SetPose(aOutPose);

    }

    delete aSysTr;
    delete aSysRot;
}*/

tMMVII_UnikPApli Alloc_SfmInitGlob(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_SfmInitGlob(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_SfmInitGlob
(
     "SfmInitGlob",
      Alloc_SfmInitGlob,
      "Compute global initial orientations with a linear program",
      {eApF::Ori},
      {eApDT::TieP},
      {eApDT::Orient},
      __FILE__
);

void cAppli_SfmInitGlob::Bench_SfmInit()
{
    // rotation around x
    cDenseMatrix<double> aZRot(3,3,eModeInitImage::eMIA_Null);
    aZRot.SetElem(0,0,1);
    aZRot.SetElem(1,1,-1);
    aZRot.SetElem(2,2,-1);

    // identity rotation pointing away from the scene
    cDenseMatrix<double> anId(3,3,eModeInitImage::eMIA_MatrixId);
    anId.SetElem(1,1,-1);
    anId.SetElem(2,2,-1);


    StdOut() << "Poses" << std::endl;
    // generate N arbitrary poses in global frame (the reference)
    int N=5;
    std::vector<tPose> aGlobPosesRef; //tPose(aCInv,aRotInv)
    for (int p=0; p<N; p++)
    {
        cRotation3D<tREAL8> aR = cRotation3D<tREAL8>::Identity() ;//(anId,false);
        cPt3dr aC = {0,0,0};
        if (p!=0) //p==0 is set to identify to be coherent with SolveL1 setting
        {
            aR = cRotation3D<tREAL8>::RandomRot();
            aC = cPt3dr::PRandUnit() ;
        }

        aGlobPosesRef.push_back(tPose(aC,aR));
    }

    StdOut() << "Triplets" << std::endl;
    cTripletSet * TriSet = new cTripletSet;

    // non-randomised triplets
    std::map<std::string,std::vector<int>> aMapTri;


    aMapTri["one"] = std::vector<int>();
    aMapTri["one"].push_back(0);
    aMapTri["one"].push_back(2);
    aMapTri["one"].push_back(3);

    aMapTri["two"] = std::vector<int>();
    aMapTri["two"].push_back(2);
    aMapTri["two"].push_back(3);
    aMapTri["two"].push_back(4);

    aMapTri["three"] = std::vector<int>();
    aMapTri["three"].push_back(1);
    aMapTri["three"].push_back(2);
    aMapTri["three"].push_back(4);

    aMapTri["four"] = std::vector<int>();
    aMapTri["four"].push_back(0);
    aMapTri["four"].push_back(1);
    aMapTri["four"].push_back(3);


    int CountTri=0;
    for (auto aT : aMapTri)
    {
        StdOut() << aT.first << " " << aT.second[0] << " " << aT.second[1] << " " << aT.second[2] << ", ";

        cTriplet Tri;
        std::vector<cView> EdgeVec;

        // first image - origin
        EdgeVec.push_back(cView(tPose(cPt3dr(0,0,0),
                                cRotation3D<double>::Identity()),
                                "image_"+ToStr(aT.second[0])));


        double aLambda = (CountTri==0) ? 1 : RandUnif_C_NotNull(0.2); //set scale of first triplet to 1 (coherent with SolveL1 settings)
        StdOut() << "aLambda=" << aLambda << std::endl;

        // Pa Pb = [Ca,Ra] [Cb,Rb] = [Ca + Ra * Cb, Ra * Rb]

        // P0 * P1^-1   OKKKKKKK
        //  [ C0, R0 ] [-R1^-1 * C1, R1^-1]       = [lambda * c01,r01]
        //  [ C0 + R0 * (-R1^-1 * C1), R0 * R1^-1]= [lambda * c01,r01]

        // P1 * P0^-1 = p01
        //  [C1 , R1] [-R0^-1 * C0   , R0^-1]       = [lambda * c01,r01]
        //  [ C1 + R1 * (-R0^-1 * C0), R1 * R0^-1 ] = [lambda * c01,r01]
        // P0^-1 * P1
        //  [-R0^-1 * C0   , R0^-1] [C1 , R1]
        //  [-R0^-1 * C0 + R0^-1 * C1, R0^-1 * R1 ]
        // P1^-1 * P0
        //  [-R1^-1 * C1, R1^-1]  [ C0, R0 ]
        //  [ -R1^-1 * C1 + -R1^-1 * C0, R1^-1 * R0 ]

        for (auto i : {aT.second[1],aT.second[2]})
        {
            /*
            cDenseMatrix<double> R0Inv = aGlobPosesRef[aT.second[0]].Rot().Mat().Inverse();
            cPt3dr C0 = aGlobPosesRef[aT.second[0]].Tr();
            cDenseMatrix<double> R1 = aGlobPosesRef[i].Rot().Mat();
            cPt3dr C1 =  aGlobPosesRef[i].Tr();

            cDenseMatrix<double> r01 = R0Inv * R1;//
            cPt3dr c01  = 1/aLambda* (R0Inv * (C1-C0));

            EdgeVec.push_back(cView(tPose(c01,cRotation3D<double>(r01.Transpose(),false)), // works with .inverse()
                                                "image_"+ToStr(i)));*/

            cDenseMatrix<double> R0 = aGlobPosesRef[aT.second[0]].Rot().Mat();
            cDenseMatrix<double> R0Inv = R0.Inverse();
            cPt3dr C0 = aGlobPosesRef[aT.second[0]].Tr();

            cDenseMatrix<double> R1 = aGlobPosesRef[i].Rot().Mat();
            cDenseMatrix<double> R1Inv = R1.Inverse();
            cPt3dr C1 =  aGlobPosesRef[i].Tr();

            // P0 * P1^-1
            //  [ C0, R0 ] [-R1^-1 * C1, R1^-1]       = [lambda * c01,r01]
            //  [ C0 + R0 * (-R1^-1 * C1), R0 * R1^-1]= [lambda * c01,r01]

            // P0^-1 * P1
            //  [-R0^-1 * C0   , R0^-1] [C1 , R1]
            //[-R0^-1 * C0 + R0^-1 * C1, R0^-1 * R1 ]^Transpose
            cDenseMatrix<double> r01 =   R0 * R1Inv;//  ; //worked
            //cPt3dr c01  = 1/aLambda * (-R0 * C0 + R0 * C1);     //worked
            cPt3dr c01  = 1/aLambda * (-R0 * C0 + R0 * C1);


            EdgeVec.push_back(cView(tPose(c01,cRotation3D<double>(r01.Transpose(),false)), //
                                              "image_"+ToStr(i)));
        }


        Tri.PVec() = EdgeVec;
        TriSet->PushTriplet(Tri);
        CountTri++;
    }


    cHyperGraph aHG;
    aHG.InitFromTriSet(TriSet);

    // solve  L1
    Solve(aHG);

    // compare with the reference poses
    for (int i=0; i<N; i++)
    {
        std::string aN1 = "image_"+ToStr(i);
        cPt3dr aCRef = aGlobPosesRef[i].Tr();
        cDenseMatrix<double> aRRef = aGlobPosesRef[i].Rot().Mat();

        cPt3dr aC = aHG.GetVertex(aN1)->Pose().Pose().Tr() * aZRot  ; //
        cDenseMatrix<double> aR = aHG.GetVertex(aN1)->Pose().Pose().Rot().Mat();

        double aResC = Norm2(aCRef - aC);
        StdOut() << aCRef << std::endl;
        StdOut() << aC << std::endl;

        double aResR = aRRef.SqL2Dist(aR,true);

        StdOut() << aN1 << " " << ", ResC=" << aResC << ", ResR=" << aResR << std::endl;
    }

    delete TriSet;
}

}; // MMVII






