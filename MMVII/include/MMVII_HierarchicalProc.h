#ifndef  _MMVII_HIERARCHICALPROC_H_
#define  _MMVII_HIERARCHICALPROC_H_


#include "MMVII_TplHeap.h"
#include "MMVII_SfmInit.h"

#include <iostream>
#include <vector>
#include <deque>
#include <chrono>
#include <thread>
#include <mutex>
#include <memory>
#include <atomic>
#include <algorithm>

static constexpr int NbThreadMax  = 8;
static constexpr int NbDepthMax  = 3;
static constexpr int NbKidsMax  = 2;


namespace MMVII
{

class cNodeHTree;
class cNodeHTreeMT; //multi-threaded (MT) tree node
class ThreadPool;

typedef std::shared_ptr<cNodeHTreeMT> tNodeHT_mt_ptr;
typedef std::chrono::time_point<std::chrono::system_clock> Time;

/* ********************************************************** */
/*                                                            */
/*                        ThreadPool                          */
/*                                                            */
/* ********************************************************** */

class ThreadPool
{
    public:
        ThreadPool() {}

        void addNode(tNodeHT_mt_ptr node);
        void Exec(int nbThread);

        const std::vector<tNodeHT_mt_ptr> allNodes() const { return mAllNodes;}
        const std::deque<tNodeHT_mt_ptr> runQueue() const { return mRunQueue;}
    private:
        void ExecLoop();

        std::deque<tNodeHT_mt_ptr> mRunQueue;   // Liste des pointeurs sur les taches a executer
        std::deque<tNodeHT_mt_ptr> mRunQueueInv;
        std::mutex mMutex_CalculusQueue;        // Mutex pour proteger l'acces a CalculusQueue entre les threads

        std::vector<tNodeHT_mt_ptr> mAllNodes;
};

/* ********************************************************** */
/*                                                            */
/*                        cNodeHTree                         */
/*                                                            */
/* ********************************************************** */

class cNodeHTree
{
    public:
        cNodeHTree(int aId) :
            IS_PART_PUSHED(false),
            mId(aId),
            mSinkVal(0),
            mSourceVal(1000),
            IS_PARTITIONED(false),
            mMat(cDenseMatrix<double>(3,eModeInitImage::eMIA_Null))
        {}

        void Init(cTripletSet&);
        void InitCut();

        // Partition
        void Partition();
        void PushPartition(cTripletSet&);

        // Alignment
        //    at bottom leaf -> spanning graph, cHyperGraph, heap, seed hyperege
        //    the edges crossing the partition to do 3d sim
        //    at any other leafs -> 3d spatial sim with triplets, cHyperGraph,
        void Align();

        // Spanning tree
        void SpanTree();

        std::vector<int>            mPart0IdV;
        std::vector<int>            mPart1IdV;
        std::shared_ptr<cNodeHTree> mPart0;
        std::shared_ptr<cNodeHTree> mPart1;
        bool                        IS_PART_PUSHED;

    private:
        /* hypergraph -> graph of triplets */
        int                         mId;
        cHyperGraph                 mSubGr;
        int                         mNumNodes;
        int                         mNumEdges;

        /* Data for MAXFLOW/MINCUT */
        std::map<std::string,std::pair<int,cPt2di>> mMapOfTriplets;///< map of pairs of triplets (triplet graph)
        std::map<int,int>                           mMapId2GraphId; ///< map between hypergraph's ids and maxflow ids
        std::map<int,std::vector<int>>              mMapAdj; ///< adjacency map
        cObjQual                                    mSink;
        cObjQual                                    mSource;
        int                                         mSinkVal;
        int                                         mSourceVal;
        bool                                        IS_PARTITIONED;

        /* Alignment parameters  */
        double               mLambda;
        cPt3dr               mTr;
        cDenseMatrix<double> mMat;


};

/* ********************************************************** */
/*                                                            */
/*                        cNodeHTreeMT                        */
/*                                                            */
/* ********************************************************** */


class cNodeHTreeMT
{
    public:
        cNodeHTreeMT(tNodeHT_mt_ptr parent, int i, int depth);

        void Init(cTripletSet&);
        void InitCutData();

        void Descend(ThreadPool& threadPool, tNodeHT_mt_ptr me);


        void Run();
        void Partition();

        std::string Name() const {return mName;}
        const tNodeHT_mt_ptr parent() const { return mParent;}

        int  ChildrenCount() const { return mChildrenCount;}
        bool isLastChild() const {
            if (! parent())
                return false;
            return parent()->mChildrenToWait.fetch_sub(1) == 1;     // Atomic decrement; if mChildrenToWait was 1, we are the last child
        }
        bool isPartitioned() const {return IS_PARTITIONED;}

    private:
        /* Graph    */
        cHyperGraph                 mSubGr;
        int                         mNumNodes;
        int                         mNumEdges;

        /* Data for MAXFLOW/MINCUT */
        std::map<std::string,std::pair<int,cPt2di>> mMapOfTriplets;///< map of pairs of triplets (triplet graph)
        std::map<int,int>                           mMapId2GraphId; ///< map between hypergraph's ids and maxflow ids
        std::map<int,std::vector<int>>              mMapAdj; ///< adjacency map
        cObjQual                                    mSink;
        cObjQual                                    mSource;
        int                                         mSinkVal;
        int                                         mSourceVal;
        bool                                        IS_PARTITIONED;



        /* Paralelization */
        tNodeHT_mt_ptr              mParent;
        int                         mChildrenCount;
        int                         mNbChildren;
        std::atomic<int>            mChildrenToWait;

        int                         mDepth;
        int                         mDepthMax;
        std::string                 mName;
};

};
#endif // _MMVII_HIERARCHICALPROC_H_

