#ifndef TREETHREAD_H
#define TREETHREAD_H

#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
#include <iostream>

namespace MMVII
{


/*
 * TreeThreads: a class to execute tasks in parallel (multi threading), based on a tree dependancy order.
 *
 * Usage:
 *
 * User has to implement a tree-like structure based upon his own "Node" type which must implement this two methods:
 *  - void Node::finalize()  :
 *      code to be executed when all of its depandancies have been executed (i.e. children nodes finalize(), none if this node is a leaf)
 *  -  container<NodePtr> Node::depends()  (or container<NodePtr>& Node::depends() or const container<NodePtr>& Node::depends() :
 *      must return any forward-iterable container containing pointer-like elements to nodes upon which this node depends.
 *
 * Minimal example 1:
 * >  class MyNode
 * >  {
 * >   public:
 * >      typedef MyNode* MyNodePtr;
 * >      MyNode( ... ) { ... }
 * >      void finalize() { ... }
 * >      const std::vector<MyNodePtr>& depends() const {return mChildren;}
 * >  private:
 * >      std::vector<MyNodePtr> mChildren;		 // Must be filled somwhere ...
 * >  }
 *
 * Minimal example 2:
 * >  class MyNode
 * >  {
 * >  public:
 * >      typedef std::shared_ptr<MyNode> MyNodePtr;
 * >      MyNode(..) {...}
 * >      void finalize() { ... }
 * >      const std::list<MyNodePtr>& depends() const {return mChildren;}
 * >  private:
 * >      std::list<MyNodePtr> mChildren;        // Must be filled somwhere ...
 * >  }
 *
 *
 *  The user has to construct his tree structure, construct a TreeThreads object instantiated with
 *   pointer-like type on MyNode (the very same type that depends() returns), and
 *   call pass the root node of the tree to the exec method of TreeThreads object:
 * 
 * 
 * >  int nbThread = 8;
 * >  MyNodePtr myRoot = ... ;
 * >  TreeThreads<MyNode::MyNodePtr> treeThreads;
 * >  threadPool->exec(myRoot,nbThread);
 *
 *  WARNING: all pointers to user node (i.e. myRoot and all those returned by MyNode::depends() must
 * be valid at least until the node has been executed (its finalize() method called).
 * Best it to wait that TreeThreads::exec() terminates.
 * 
 */


// T must be a pointer-like type to user Node (i.e. "MyNode *", "std::shared_ptr<MyNode>", ...)
template<class T>
class TreeThreads
{
public:

    TreeThreads() {}

    // Will execute finalize() method of all nodes which root depends on (directly or not) in dependancy order and then the root finalize() method
    // Exec() runs in the main thread and will return only when all nodes heve executed their finalize() method
    void Exec(T root, int nbThread);

private:
    class Node {
    public:
        typedef std::shared_ptr<Node> PNode;						//  share_ptr to Node class: memory managment for our nodes will be automatic
        Node(T userNodePtr, PNode parent) : mUserNodePtr(userNodePtr),mParent(parent),mChildrenToWait(0) {}


        // Atomically decrement parent not-terminated-child count and return true if this was the last one
        bool isLastChild() const
        {
            if (! mParent)
                return false;
            return mParent->mChildrenToWait.fetch_sub(1) == 1;     // Atomic decrement; if mChildrenToWait was 1, we are the last child
        }

        // Recurvely build the dependancy tree using the userNode->depends() method
        // If the node has no dependancy (leaf) it will be added to the ready to execute queue
        void descend(TreeThreads *tt, PNode me)
        {
            std::cout << "ii " << &mUserNodePtr->depends() <<std::endl;
            std::cout << "descend " << mUserNodePtr->depends().size() << std::endl;
            for (const auto& userChild: mUserNodePtr->depends()) {
                if (userChild!=nullptr){
                auto child = std::make_shared<Node>(userChild,me);
                std::cout << "b " << std::endl;
                child->descend(tt, child);
                std::cout << "a " << std::endl;
                mChildrenToWait ++;}
            }
            std::cout << "alsmost " << std::endl;
            if (mChildrenToWait == 0)
                tt->mReadyQueue.push_back(me);
            std::cout << "descendeddd " << std::endl;
        }

        // Job to be done when all dependancies have been executed
        void finalize()
        {
            mUserNodePtr->finalize();
        }

        PNode parent() { return mParent;}

    private:
        T mUserNodePtr;                                     // Pointer to the associated user node
        PNode mParent;                                      // Pointer to parent: this implement our tree structure that will be used bottom up only (starting from leaves in readyQueue)
        std::atomic<int> mChildrenToWait;                   // Atomic because it can be decremented at the same time by N threads
    };

    void ExecLoop();                                        // This will be executed by each of the NbThread threads

    std::deque<typename Node::PNode> mReadyQueue;           // List of pointer to nodes ready to be executed. Running nodes are removed from this list before being executed
    std::mutex mMutex_ReadyQueue;                           // Mutex to protect push in and pop from the readyQueue from multiple threads. 
};




template <class T>
void TreeThreads<T>::Exec(T root, int nbThread)
{
    mReadyQueue.clear();
    auto rootNode = std::make_shared<Node>(root,std::shared_ptr<Node>(nullptr));		// Create our root node
    rootNode->descend(this,rootNode);													//  and resurvelu build our depandancy tree
    std::vector<std::thread> threadList;
    for (int i = 0; i < nbThread; ++i) 													// We start nbThread and each will execute ExecLoop
        threadList.emplace_back(std::thread(&TreeThreads::ExecLoop, this));
    for (auto& t : threadList)
		t.join();                       												// We wait that all threads are finished
}

template <class T>
void TreeThreads<T>::ExecLoop()
{std::cout << "start ExecLoop" << std::endl;
    // Loop while there are nodes to execute in the readyQueue.
    // We pop the first element in the readyQueue, execute it,
    //  and if it was the last child of its parent, push the parent node in the readyQueue
    while (true) {
        typename Node::PNode node;
        {
            // This lock protect mReadyQueue  access/modifying from different threads. The lock is removed at the end of this code bloc (destructor called)
            std::lock_guard<std::mutex> lock(mMutex_ReadyQueue);			
            if (mReadyQueue.empty())
                return;             // No more node to execute: end of this thread, it will be really ended in the 't.join()' above from the main thread
            node = mReadyQueue.front();
            mReadyQueue.pop_front();
        }
        std::cout << "before node->finalize()" << std::endl;
        node->finalize();           // do the job
        // Atomically decrement parent not-terminated-child count and return true if this was the last terminated child
        if (node->isLastChild()) {
			// Protect the mReadyQueue and add this node's parent: all its childs have terminated
            std::lock_guard<std::mutex> lock(mMutex_ReadyQueue);
            mReadyQueue.push_back(node->parent());
        }
        // Here the node have been removed for mReadyQueue and variable node is destroyed : the shared_ptr is de-allocated
    }
}

/* NB:
 * The readyQueue may be empty between mReadyQueue.pop_front() and mReadyQueue.push_back(node->parent) and
 *   this we'll stop all others threads.
 * But we will add one and oly one node to readyQueue and it will be executed by this thread in the following
 *   iteration of the loop.
 * Hence, the number of remaining nodes can't be greater than the number of threads, so we always garantee
 *   a maximal usage of the threads.
 */


/*
 Full test/example program

#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <algorithm>
#include "treethread.h"

using namespace std;

static constexpr int INVERSE_LEAF_PROBALITY=20;
static constexpr int MIN_CHILDREN=2;
static constexpr int MAX_CHILDREN=2;
static constexpr int NbThreadMax  = 16;

namespace {
int getRand(int min, int max)
{
    return std::rand() % (max-min+1) + min;
}
}

typedef std::chrono::time_point<std::chrono::system_clock> Time;

class CalculusNode
{
public:
    typedef CalculusNode* PNode;

    // Needed methods for TreeThreads
    void finalize();
    const std::vector<PNode>& depends() const {return mChildren;}
    //  End


    CalculusNode(PNode parent, int i, int depth);
    ~CalculusNode();

    void descend(PNode me);

    // Debug
    int instance() const { return mInstance; }
    std::string name() const { return mName ; }
    int depth() const { return mDepth;}
    int testCount() const { return mTestCount;}


    double val;
    Time start,end;
    std::vector<PNode> mChildren; // debug
private:
    int mInstance;
    int mChildNum;
    int mDepth;
    std::string mName;
    int mTestCount;
    static int nbInstance;
};

std::vector<CalculusNode::PNode> allNodes;


int CalculusNode::nbInstance = 0;

CalculusNode::CalculusNode(PNode parent, int i, int depth) : mInstance(nbInstance++),mChildNum(i),mDepth(depth+1),mTestCount(-1)
{
    auto parentName = parent ? parent->mName + "." : "" ;
    mName = parentName + std::to_string(mChildNum);
    std::cerr << "Create " << mName << std::endl;
}

CalculusNode::~CalculusNode()
{
    std::cerr << "Delete " << mName << std::endl;
}

void CalculusNode::descend(PNode me)
{
    auto r = getRand(0,INVERSE_LEAF_PROBALITY / depth());
    bool isLeaf = (r == 0);
    if ( ! isLeaf)  {
        auto nbChildren=getRand(MIN_CHILDREN, MAX_CHILDREN);
        for (int i=0; i<nbChildren; i++) {
            auto child = new CalculusNode(me, i, depth());
            child->descend(child);
            mChildren.push_back(child);
            allNodes.push_back(child);
        }
    }
}

void CalculusNode::finalize() {
    val=0;
    // Test: mTestCount will be the total number of nodes under it plus one (itself)
    // The count will be correct only if nodes are executed in the correct order (i.e. children before parent)
    mTestCount = 0;
    for (const PNode& child: mChildren)
        mTestCount += child->mTestCount;
    mTestCount++;

    start = std::chrono::system_clock::now();
    // do something stupid to consume CPU cycles ...
    auto nbLoop = getRand(500000,20000000);
    for (int i=0; i<nbLoop; i++)
        val = val + (double)std::rand() / std::rand();
    end = std::chrono::system_clock::now();
}

int main()
{
    TreeThreads<CalculusNode*> tp;
    auto root = new CalculusNode(nullptr, 0, 0);
    root->descend(root);
    allNodes.push_back(root);

    std::cerr << "============ Created nodes" << std::endl;
    std::cerr << "Total Nodes: "  << allNodes.size() << std::endl;
    for (const auto& node : allNodes) {
        std::cerr << node->instance() << " " << node->name() << " " << node->testCount() << std::endl;
    }
    std::cerr << "============" << std::endl;
    std::cerr << "Total Nodes: "  << allNodes.size() << std::endl;

    tp.Exec(root,NbThreadMax);

    std::cerr << "============ Execution order" << std::endl;
    auto v = allNodes;
    std::sort (v.begin(), v.end(), [](const CalculusNode::PNode& n1, const CalculusNode::PNode& n2) { return n1->start < n2->start; });
    auto start = v[0]->start.time_since_epoch();
    for (const auto& node : v) {
        std::cerr << (node->start.time_since_epoch() - start) / 1ns << " " <<  node->name() << " " << node->mChildren.size() << " " << node->testCount() << std::endl;
    }
    std::cerr << "Total Nodes: "  << allNodes.size() << std::endl;
    return 0;
}

*  End of full working example
*/


} // namespace MMVII

#endif // TREETHREAD_H
