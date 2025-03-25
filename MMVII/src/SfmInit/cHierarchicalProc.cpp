#include "MMVII_HierarchicalProc.h"
#include <random>
#include "MMVII_TplHeap.h"
//#include "graph.h"


namespace MMVII
{

int getRand(int min, int max)
{
    return std::rand() % (max-min+1) + min;
}

/* ********************************************************** */
/*                                                            */
/*                        ThreadPool                          */
/*                                                            */
/* ********************************************************** */

void ThreadPool::addNode(tNodeHT_mt_ptr node)
{
    mAllNodes.push_back(node);
}

void ThreadPool::ExecUp()
{
    mRunQueue.clear();
    for (auto &node : mAllNodes) {
        node->ResetChildrenToWait();
        if (node->ChildrenCount() == 0)
            mRunQueue.push_back(node);  // push if leaf
    }
    std::vector<std::thread> threadList;
    for (int i = 0; i < mNbThread; ++i) // On lance NbThreads, chaque thread execute ExecLoop
        threadList.emplace_back(std::thread(&ThreadPool::ExecLoopUp, this));
    for (auto& t : threadList)
        t.join();                       // On attend la fin de tous les threads (donc de touts les taches)
    threadList.clear();
}


void ThreadPool::ExecLoopUp()
{
    StdOut() << "numbre of tasks at start= " << mRunQueue.size() << std::endl;
    while (true) {  // boucle infinie: on prend l'élément suivant du tableau et on l'execute.
        tNodeHT_mt_ptr node;
        {
            StdOut() << "   as we go= " << mRunQueue.size() << std::endl;
            // On protege la liste des taches a executer contre l'execution en parallle des threads avec un lock
            std::lock_guard<std::mutex> lock(mMutex_CalculusQueue);
            if (mRunQueue.empty())
                return;             // Si plus de tache, on sort. On va rejoindre le "t.join()"
            node = mRunQueue.front();
            mRunQueue.pop_front();
        }
        node->RunUp();
        if (node->isLastChild()) {
            std::lock_guard<std::mutex> lock(mMutex_CalculusQueue);
            mRunQueue.push_back(node->parent());/// if children processed, add the parent
        }
    }
}

void ThreadPool::ExecDown(const cTripletSet& aTSet)
{
    mRunQueue.clear();
    for (auto &node : mAllNodes) {
        if (node->IsRoot()) {
            mRunQueue.push_back(node);  // push if root
            break;
        }
    }
    mNbWorkingThread = 0;
    std::vector<std::thread> threadList;
    for (int i = 0; i < mNbThread; ++i) // On lance NbThreads, chaque thread execute ExecLoop
        threadList.emplace_back(std::thread(&ThreadPool::ExecLoopDown, this, aTSet));
    for (auto& t : threadList)
        t.join();                       // On attend la fin de tous les threads (donc de touts les taches)
    threadList.clear();
}


void ThreadPool::ExecLoopDown(const cTripletSet& aTSet)
{
    StdOut() << "numbre of tasks at start= " << mRunQueue.size() << std::endl;
    while (true) {  // boucle infinie: on prend l'élément suivant du tableau et on l'execute.
        tNodeHT_mt_ptr node;
        {
            StdOut() << "   as we go= " << mRunQueue.size() << std::endl;
            // On protege la liste des taches a executer contre l'execution en parallle des threads avec un lock
            std::unique_lock<std::mutex> lock(mMutex_CalculusQueue);
            cv.wait(lock, [this] { return (! mRunQueue.empty()) || (mNbWorkingThread == 0) ;});
            if (mRunQueue.empty() && mNbWorkingThread == 0)
                return;             // Si plus de tache, on sort. On va rejoindre le "t.join()"
            if (mRunQueue.empty())
                continue;
            node = mRunQueue.front();
            mRunQueue.pop_front();
            mNbWorkingThread++;
            lock.unlock();
        }
        node->RunDown(aTSet);
        {
            std::lock_guard<std::mutex> lock(mMutex_CalculusQueue);
            mNbWorkingThread--;
            for (const auto& child : node->Children())
                mRunQueue.push_back(child);
            cv.notify_all();
        }
    }
}



/* ********************************************************** */
/*                                                            */
/*                        cNodeHTreeMT                        */
/*                                                            */
/* ********************************************************** */


/* for binary tree i= {0,1}*/
cNodeHTreeMT::cNodeHTreeMT(tNodeHT_mt_ptr parent, int i, int depth, int source, int sink) :
    mSinkVal(sink),
    mSourceVal(source),
    mParent(parent),
    mDepth(depth+1)
{
    auto aParentName = parent ? parent->Name() + "" : "" ;
    mName = aParentName + std::to_string(i);
}

void cNodeHTreeMT::Init(cTripletSet &aSet)
{
    mSubGr.InitFromTriSet(&aSet);

    InitCutData();

}

void cNodeHTreeMT::InitCutData()
{
    /*
       - initialise data for maxflow input:
            - mMapOfTriplets
            - mMapId2GraphId
            - sink and source
    */
    int aGraphId=0;
    double aUpScale = 1.0;
    for (auto anEdg : mSubGr.AdjMap())
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

                        std::string aCurTriPairDir = ToStr(int(anEdg.second[aTi]->Index()))+"to"+
                                                     ToStr(int(anEdg.second[aTj]->Index()));
                        std::string aCurTriPairInv = ToStr(int(anEdg.second[aTj]->Index()))+"to"+
                                                     ToStr(int(anEdg.second[aTi]->Index()));

                        // check map to avoid duplicates
                        if ( (mMapOfTriplets.find(aCurTriPairDir) == mMapOfTriplets.end()) &&
                             (mMapOfTriplets.find(aCurTriPairInv) == mMapOfTriplets.end()) )
                        {
                            /// quality defined as 1/(1+x)^2 where x=max of the quality~residual
                            double aQual = aUpScale * (1.0/(1.0+std::pow(
                                            std::max(anEdg.second[aTi]->Quality(),
                                            anEdg.second[aTj]->Quality()),2)));
                           // StdOut() << "mmmmmmm" << aQual << std::endl;

                            /// add the edge to the graph of triplets
                            mMapOfTriplets[aCurTriPairDir] = std::make_pair(aQual,
                                            cPt2di(anEdg.second[aTi]->Index(),anEdg.second[aTj]->Index()));

                            /// add to the map if does not exist already
                            if (mMapId2GraphId.find(anEdg.second[aTi]->Index()) == mMapId2GraphId.end() )
                                mMapId2GraphId[anEdg.second[aTi]->Index()] = aGraphId++;

                            if (mMapId2GraphId.find(anEdg.second[aTj]->Index()) == mMapId2GraphId.end() )
                                mMapId2GraphId[anEdg.second[aTj]->Index()] = aGraphId++;

                            /// add to adjacency vector (symmetric)
                            ///
                            mMapAdj[anEdg.second[aTi]->Index()].push_back(anEdg.second[aTj]->Index());
                            mMapAdj[anEdg.second[aTj]->Index()].push_back(anEdg.second[aTi]->Index());
                        }
                    }
                }
            }
        }
    }

    mNumNodes = mMapId2GraphId.size();
    mNumEdges = mMapOfTriplets.size();

    cCmpObjQual                        aCmp;
    cIndexedHeap<cObjQual,cCmpObjQual> aHeap(aCmp);

    /// update heap
    ///   heap is used to get the best connected triplets for sink and source terminal
    ///
    for (auto anObj : mMapAdj)
    {
        cObjQual aTri;
        aTri.mId = anObj.first;
        aTri.mQual = anObj.second.size();

        aHeap.Push(aTri);
    }

    mSubGr.FindTerminals(aHeap,mMapAdj,mSource,mSink);

}

void cNodeHTreeMT::Show()
{
    // file structure: child parent value
    if (mChildrenV.size())
    {
        for (const auto& aKid : mChildrenV)
        {
            // if children are leafs, print their full names
            if (!aKid->mChildrenV.size())
            {
                for (const auto& aTri : aKid->mSubGr.VHEdges())
                {
                    for (const auto& aV : aTri->Vertices())
                    {
                        StdOut() << aV->Pose().Name() << ".";
                    }
                    StdOut() << " " << Name() << " 1" << std::endl;
                }
            }
            else
            {
                {
                    StdOut()
                             << aKid->Name() << " "
                             << Name() << " "
                             << aKid->mNumNodes << std::endl;
                }

                aKid->Show();
            }
        }
    }

}

void cNodeHTreeMT::BuildChildren(ThreadPool &threadPool, tNodeHT_mt_ptr input, int aNbDepth)
{

    bool isLeaf = (aNbDepth >= mDepth);
    if (isLeaf)
    {

        for (int i=0; i<NbKidsMax; i++)
        {
           auto aKid = std::make_shared<cNodeHTreeMT>(input,i,mDepth,mSourceVal,mSinkVal);
           threadPool.addNode(aKid);
           StdOut() << aKid->Name() << std::endl;

           mChildrenV.push_back(aKid);
           
           aKid->BuildChildren(threadPool,aKid,aNbDepth);
        }
    }

}

void cNodeHTreeMT::PartitionPlus(const cTripletSet& aSetTriFull)
{

    StdOut() << "======partition" << std::endl;

    /// ================ Find the MINCUT/MAXFLOW
    ///
    bool GRAPH_AUGUMENT=true;
    std::map<int,double> aNodeDegV;
    double               aTotalDeg=0;

    int aSourceId = mMapId2GraphId[mSource.mId];
    int aSinkId   = mMapId2GraphId[mSink.mId];

    int aNumNodeCur = mNumNodes;
    int aNumEdgeCur = mNumEdges;

    typedef Graph<double,double,double> GraphType;
    GraphType *g = new GraphType( aNumNodeCur,//
                                  aNumEdgeCur);

    /// initialise the nodes of the graph
    ///
    g->add_node(aNumNodeCur);

    /// add edges between nodes
    ///
    double aRevCap=0;
    double aUpscale=100.0;
    for (auto anE : mMapOfTriplets)
    {
        int aIdNode1 = mMapId2GraphId[anE.second.second.x()];
        int aIdNode2 = mMapId2GraphId[anE.second.second.y()];

        double aCap = anE.second.first;
        //

        /// edge weight
        double aCapSca=aUpscale*aCap;
        g->add_edge(aIdNode1,aIdNode2,aCapSca,aRevCap);
        StdOut() << " e " << aIdNode1 << " "
                          << aIdNode2 << " "
                          << aCapSca << "\n";

        if (GRAPH_AUGUMENT)
        {
            /// update degree of node
            aTotalDeg += aCapSca;
            if (aNodeDegV.find(aIdNode1) != aNodeDegV.end())
                aNodeDegV[aIdNode1] += aCapSca;
            else
                aNodeDegV[aIdNode1] = aCapSca;
        }

    }



    /// add terminal super-source and super-sink
    ///   - last nodes, after "real" nodes
    if (GRAPH_AUGUMENT)
    {


        /// add penalty for balancing
        ///
        for (auto aNDeg : aNodeDegV)
        {
            int aNodeCur = aNDeg.first;
            double aLambda = 1;
            double aPenalty = aLambda*aNDeg.second;

            if (aSourceId!=aNodeCur && aSinkId!=aNodeCur)
            {
                g->add_edge(aSourceId,aNodeCur,aPenalty,0);
                g->add_edge(aNodeCur,aSinkId,aPenalty,0);

                StdOut() << "Added edge from " << aSourceId << " to " << aNodeCur << " with capacity " << aPenalty << std::endl;
                StdOut() << "Added edge from " << aNodeCur << " to " << aSinkId << " with capacity " << aPenalty << std::endl;
            }
        }
    }

    /// add terminal SINK and SOURCE
    ///
    g->add_tweights(aSourceId,mSinkVal,mSourceVal);
    g->add_tweights(aSinkId,mSourceVal,mSinkVal);
    StdOut() << mSinkVal << " " << mSourceVal << std::endl;

    /// compute the flow
    ///
    int aFlow = g->maxflow();
    StdOut() << "Flow=" << aFlow << " "
                        << ((GRAPH_AUGUMENT) ? aFlow-aTotalDeg : aFlow) << std::endl;

    /// update children given the partition
    ///
    cTripletSet aSet0, aSet1;
    std::vector<cTriplet> aTri0V, aTri1V;
    for (const auto& aH : mSubGr.VHEdges())
    {
        for (const auto& aTri : mMapId2GraphId)
        {
            if (int(aH->Index()) == aTri.first)
            {
                if (g->what_segment(aTri.second) ==  GraphType::SOURCE)
                {
                    for (auto aT : aSetTriFull.Set())
                    {
                        if (aT.Id() == int(aH->Index()))
                        {
                            aTri0V.push_back(aT);
                            StdOut() << aTri.first << " 0" << std::endl;
                        }
                    }
                }
                else
                {
                    for (auto aT : aSetTriFull.Set())
                    {
                        if (aT.Id() == int(aH->Index()))
                        {
                            aTri1V.push_back(aT);
                            StdOut() << aTri.first << " 1" << std::endl;
                        }
                    }
                }
            }
        }
    }


    /// first kid
    ///
    std::string aN0 = aSetTriFull.Name() + "-Left";
    aSet0.SetName(aN0);
    aSet0.Set() = aTri0V; //new set containing the left partition
    mChildrenV[0]->Init(aSet0);

    /// second kid
    ///
    std::string aN1 = aSetTriFull.Name() + "-Right";
    aSet1.SetName(aN1);
    aSet1.Set() = aTri1V; //new set containing the right partition
    mChildrenV[1]->Init(aSet1);

    delete g;

}

void cNodeHTreeMT::Partition(const cTripletSet& aSetTriFull)
{

    StdOut() << "======partition" << std::endl;

    /// ================ Find the MINCUT/MAXFLOW
    ///
    typedef Graph<double,double,double> GraphType;
    GraphType *g = new GraphType( mNumNodes,
                                  mNumEdges);

    /// initialise the nodes of the graph
    ///
    g->add_node(mNumNodes);

    /// add edges between nodes
    ///
    double aRevCap=0;
    double aUpscale=1.0;
    for (auto anE : mMapOfTriplets)
    {
        int aIdNode1 = mMapId2GraphId[anE.second.second.x()];
        int aIdNode2 = mMapId2GraphId[anE.second.second.y()];

        double aCap = anE.second.first;
        //StdOut() << aUpscale*aCap << "\n";

        /// edge weight
        g->add_edge(aIdNode1,aIdNode2,aUpscale*aCap,aRevCap);

    }

    /// add terminal SINK and SOURCE
    ///
    g->add_tweights(mMapId2GraphId[mSource.mId],mSinkVal,mSourceVal);
    g->add_tweights(mMapId2GraphId[mSink.mId],mSourceVal,mSinkVal);
    //g->add_tweights(aMapMyId2GraphId[0],mSourceInit,mSinkInit);
    //g->add_tweights(aMapMyId2GraphId[aNumNodes-1],mSinkInit,mSourceInit);

    /// compute the flow
    ///
    int aFlow = g->maxflow();
    StdOut() << "Flow=" << aFlow << std::endl;

    /// update children given the partition
    ///
    cTripletSet aSet0, aSet1;
    std::vector<cTriplet> aTri0V, aTri1V;
    for (const auto& aH : mSubGr.VHEdges())
    {
        for (const auto& aTri : mMapId2GraphId)
        {
            if (int(aH->Index()) == aTri.first)
            {
                if (g->what_segment(aTri.second) ==  GraphType::SOURCE)
                {
                    for (auto aT : aSetTriFull.Set())
                    {
                        if (aT.Id() == int(aH->Index()))
                        {
                            aTri0V.push_back(aT);
                            StdOut() << aTri.first << " 0" << std::endl;
                        }
                    }
                }
                else
                {
                    for (auto aT : aSetTriFull.Set())
                    {
                        if (aT.Id() == int(aH->Index()))
                        {
                            aTri1V.push_back(aT);
                            StdOut() << aTri.first << " 1" << std::endl;
                        }
                    }
                }
            }
        }
    }


    /// first kid
    ///
    std::string aN0 = aSetTriFull.Name() + "-Left";
    aSet0.SetName(aN0);
    aSet0.Set() = aTri0V; //new set containing the left partition
    mChildrenV[0]->Init(aSet0);

    /// second kid
    ///
    std::string aN1 = aSetTriFull.Name() + "-Right";
    aSet1.SetName(aN1);
    aSet1.Set() = aTri1V; //new set containing the right partition
    mChildrenV[1]->Init(aSet1);

    delete g;

}

void cNodeHTreeMT::RunDown(const cTripletSet& aSet)
{

    if (mChildrenV.size())
        PartitionPlus(aSet);

}


void cNodeHTreeMT::RunUp()
{
    int val=0;

    Time start = std::chrono::system_clock::now();
    // Calcul stupide qui prend du temps ...
    auto nbLoop = getRand(50000,200000);
    for (int i=0; i<nbLoop; i++)
        val = val + (double)std::rand() / std::rand();
    Time end = std::chrono::system_clock::now();

    StdOut() << "ALIGN, partition id=" << this->Name() << " "
             << end.time_since_epoch().count() - start.time_since_epoch().count() << std::endl;

}

/* ********************************************************** */
/*                                                            */
/*                        cNodeHTree                          */
/*                                                            */
/* ********************************************************** */

void cNodeHTree::Init(cTripletSet& aSet)
{
    mSubGr.InitFromTriSet(&aSet);

    InitCut();
}

void cNodeHTree::InitCut()
{
    /*
       - initialise data for maxflow input:
            - mMapOfTriplets
            - mMapId2GraphId
            - sink and source
    */
    int aGraphId=0;
    double aUpScale = 10.0;
    for (auto anEdg : mSubGr.AdjMap())
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

                        std::string aCurTriPairDir = ToStr(int(anEdg.second[aTi]->Index()))+"to"+
                                                     ToStr(int(anEdg.second[aTj]->Index()));
                        std::string aCurTriPairInv = ToStr(int(anEdg.second[aTj]->Index()))+"to"+
                                                     ToStr(int(anEdg.second[aTi]->Index()));

                        // check map to avoid duplicates
                        if ( (mMapOfTriplets.find(aCurTriPairDir) == mMapOfTriplets.end()) &&
                             (mMapOfTriplets.find(aCurTriPairInv) == mMapOfTriplets.end()) )
                        {
                            /// quality defined as 1/1+x^2 where x=max of the quality
                            double aQual = aUpScale * (1.0/(1+std::pow(
                                            anEdg.second[aTi]->Quality()-
                                            anEdg.second[aTj]->Quality(),2)));
                            //StdOut() << aQual << std::endl;

                            /// add the edge to the graph of triplets
                            mMapOfTriplets[aCurTriPairDir] = std::make_pair(aQual,
                                            cPt2di(anEdg.second[aTi]->Index(),anEdg.second[aTj]->Index()));

                            /// add to the map if does not exist already
                            if (mMapId2GraphId.find(anEdg.second[aTi]->Index()) == mMapId2GraphId.end() )
                                mMapId2GraphId[anEdg.second[aTi]->Index()] = aGraphId++;

                            if (mMapId2GraphId.find(anEdg.second[aTj]->Index()) == mMapId2GraphId.end() )
                                mMapId2GraphId[anEdg.second[aTj]->Index()] = aGraphId++;

                            /// add to adjacency vector (symmetric)
                            ///
                            mMapAdj[anEdg.second[aTi]->Index()].push_back(anEdg.second[aTj]->Index());
                            mMapAdj[anEdg.second[aTj]->Index()].push_back(anEdg.second[aTi]->Index());
                        }
                    }
                }
            }
        }
    }

    mNumNodes = mMapId2GraphId.size();
    mNumEdges = mMapOfTriplets.size();

    cCmpObjQual                        aCmp;
    cIndexedHeap<cObjQual,cCmpObjQual> aHeap(aCmp);

    /// update heap
    ///   heap is used to get the best connected triplets for sink and source terminal
    ///
    for (auto anObj : mMapAdj)
    {
        cObjQual aTri;
        aTri.mId = anObj.first;
        aTri.mQual = anObj.second.size();

        aHeap.Push(aTri);
    }

    mSubGr.FindTerminals(aHeap,mMapAdj,mSource,mSink);
}

void cNodeHTree::Partition()
{
    /// ================ Find the MINCUT/MAXFLOW
    ///
    typedef Graph<double,double,double> GraphType;
    GraphType *g = new GraphType( mNumNodes,
                                  mNumEdges);

    /// initialise the nodes of the graph
    ///
    g->add_node(mNumNodes);


    /// add edges between nodes
    ///
    double aRevCap=0;
    double aUpscale=1.0;
    for (auto anE : mMapOfTriplets)
    {
        int aIdNode1 = mMapId2GraphId[anE.second.second.x()];
        int aIdNode2 = mMapId2GraphId[anE.second.second.y()];

        double aCap = anE.second.first;
        //StdOut() << aUpscale*aCap << "\n";

        /// edge weight
        g->add_edge(aIdNode1,aIdNode2,aUpscale*aCap,aRevCap);

    }

    /// add terminal SINK and SOURCE
    ///
    g->add_tweights(mMapId2GraphId[mSource.mId],mSinkVal,mSourceVal);
    g->add_tweights(mMapId2GraphId[mSink.mId],mSourceVal,mSinkVal);
    //g->add_tweights(aMapMyId2GraphId[0],mSourceInit,mSinkInit);
    //g->add_tweights(aMapMyId2GraphId[aNumNodes-1],mSinkInit,mSourceInit);

    /// compute the flow
    int aFlow = g->maxflow();
    StdOut() << "Flow=" << aFlow << std::endl;

    /// update partition vectors
    ///
    for (auto aH : mSubGr.VHEdges())
    {
        for (auto aTri : mMapId2GraphId)
        {
            if (int(aH->Index()) == aTri.first)
            {
                if (g->what_segment(aTri.second) ==  GraphType::SOURCE)
                {
                    mPart0IdV.push_back(aH->Index());
                    StdOut() << aTri.first << " 0" << std::endl;
                }
                else
                {
                    mPart1IdV.push_back(aH->Index());
                    StdOut() << aTri.first << " 1" << std::endl;
                }
            }
        }
    }

    IS_PARTITIONED = true;

    delete g;
}

void cNodeHTree::PushPartition(cTripletSet& aSetFull)
{
    StdOut() << "Push" << std::endl;

    MMVII_INTERNAL_ASSERT_User(IS_PARTITIONED, eTyUEr::eUnClassedError,
                               "cNodeHTree::PushPartition, Partition before propagating the partition down the tree.");


    /// push GraphType::SOURCE to mPart0 and
    ///      GraphType::SINK   to mPart1
    ///
    /// source
    StdOut() << "source " << mPart0IdV.size() << " "
                          << mPart1IdV.size() << std::endl;
    cTripletSet aSet0;
    std::vector<cTriplet> aTri0V;
    for (auto aP0 : mPart0IdV)
    {

        for (auto aT : aSetFull.Set())
        {
            if (aT.Id() == aP0)
            {
                aTri0V.push_back(aT);
                //StdOut() << aP0 << " " << aT.Id() << std::endl;
            }
        }
    }
    std::string aN0 = aSetFull.Name();
    aSet0.SetName(aN0);
    aSet0.Set() = aTri0V;

    mPart0->Init(aSet0);

    StdOut() << "Sink" << std::endl;
    /// sink
    ///
    cTripletSet aSet1;
    std::vector<cTriplet> aTri1V;
    for (auto aP1 : mPart1IdV)
    {

        for (auto aT : aSetFull.Set())
        {
            if (aT.Id() == aP1)
            {
                aTri1V.push_back(aT);
                //StdOut() << aP1 << std::endl;
            }
        }
    }
    aSet1.SetName(aN0);
    aSet1.Set() = aTri1V;

    mPart1->Init(aSet1);

    IS_PART_PUSHED = true;
}

void cNodeHTree::Align()
{
    int val=0;

    Time start = std::chrono::system_clock::now();
    // Calcul stupide qui prend du temps ...
    auto nbLoop = getRand(50000,200000);
    for (int i=0; i<nbLoop; i++)
        val = val + (double)std::rand() / std::rand();
    Time end = std::chrono::system_clock::now();

    StdOut() << "ALIGN, partition id=" << this->mId << " "
             << end.time_since_epoch().count() - start.time_since_epoch().count() << std::endl;
}

void cNodeHTree::SpanTree()
{
    int val=0;

    Time start = std::chrono::system_clock::now();
    // Calcul stupide qui prend du temps ...
    auto nbLoop = getRand(50000,200000);
    for (int i=0; i<nbLoop; i++)
        val = val + (double)std::rand() / std::rand();
    Time end = std::chrono::system_clock::now();


    StdOut() << "SpanTree, partition id=" << this->mId << " "
             << end.time_since_epoch().count() - start.time_since_epoch().count() << std::endl;

}



}; // MMVII





