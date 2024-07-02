#include "MMVII_HierarchicalProc.h"
#include <random>
#include "MMVII_TplHeap.h"
#include "graph.h"

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
    if (node->ChildrenCount() == 0)
    {
        mRunQueue.push_back(node);///push only the last leaf
        std::cerr << "eeeeeeee " << node->Name() << " " << node->ChildrenCount() << std::endl;
    }
}


void ThreadPool::Exec(int nbThread)
{
    std::vector<std::thread> threadList;
    for (int i = 0; i < nbThread; ++i) // On lance NbThreads, chaque thread execute ExecLoop
        threadList.emplace_back(std::thread(&ThreadPool::ExecLoop, this));
    for (auto& t : threadList)
        t.join();                       // On attend la fin de tous les threads (donc de touts les taches)
}

void ThreadPool::ExecLoop()
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
        node->Run();
        if (node->isLastChild()) {
            std::lock_guard<std::mutex> lock(mMutex_CalculusQueue);
            mRunQueue.push_back(node->parent());/// if children processed, add the parent
        }
    }
}



/* ********************************************************** */
/*                                                            */
/*                        cNodeHTreeMT                        */
/*                                                            */
/* ********************************************************** */


/* for binary tree i= {0,1}*/
cNodeHTreeMT::cNodeHTreeMT(tNodeHT_mt_ptr parent, int i, int depth) :
    mSinkVal(0),
    mSourceVal(1000),
    IS_PARTITIONED(false),
    mParent(parent),
    mChildrenCount(0),
    mDepth(depth+1)
{
    auto aParentName = mParent ? mParent->Name() + "-" : "" ;
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


void cNodeHTreeMT::Descend(ThreadPool &threadPool, tNodeHT_mt_ptr input)
{

    bool isLeaf = (NbDepthMax >= mDepth);
    if (isLeaf)
    {

        for (int i=0; i<NbKidsMax; i++)
        {
           auto aKid = std::make_shared<cNodeHTreeMT>(input,i,mDepth);
           StdOut() << aKid->Name() << std::endl;

           mChildrenCount++;
           mChildrenV.push_back(aKid);

           aKid->Descend(threadPool,aKid);
           threadPool.addNode(aKid);
        }
    }
    mChildrenToWait = ChildrenCount();

    StdOut() << "end " << mChildrenToWait <<  std::endl;

}

void cNodeHTreeMT::Partition()
{
    IS_PARTITIONED=true;
    StdOut() << "======partition" << std::endl;
}

void cNodeHTreeMT::Run()
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





