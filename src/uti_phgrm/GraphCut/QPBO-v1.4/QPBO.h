/* QBPO.h */
/*
    Version 1.4

    Copyright 2006-2008 Vladimir Kolmogorov (vnk@ist.ac.at).

    This file is part of QPBO.

    QPBO is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    QPBO is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with QPBO.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
    /////////////////////////////////////////////////////////////////////////////

    Software for minimizing energy functions of the form
    E(x_1, ..., x_n) = \sum_i Ei(x_i) + \sum_{ij} Eij(x_i,x_j)
    where x_i are binary labels (0 or 1).

    Terms Eij can be submodular or supermodular, so in general the task is NP-hard.
    The software produces a *partial* labeling: each node is labeled as either 0,1 or
    ``unknown'' (represented by negative numbers).
    This labeling is guaranteed to be a part of at least one optimal solution.

    The following techniques are implemented:

    1. Basic roof duality algorithm ("QPBO"):

        P. L. Hammer, P. Hansen, and B. Simeone.
        Roof duality, complementation and persistency in quadratic 0-1 optimization.
        Mathematical Programming, 28:121–155, 1984.

        E. Boros, P. L. Hammer, and X. Sun.
        Network flows and minimization of quadratic pseudo-Boolean functions.
        Technical Report RRR 17-1991, RUTCOR Research Report, May 1991.

    2. "Probe" technique:

        E. Boros, P. L. Hammer, and G. Tavares
        Preprocessing of Unconstrained Quadratic Binary Optimization
        Technical Report RRR 10-2006, RUTCOR Research Report, April 2006.

    with implementational details described in

        C. Rother, V. Kolmogorov, V. Lempitsky, and M. Szummer
        Optimizing binary MRFs via extended roof duality
        CVPR 2007.

    3. QPBOI ("Improve") technique:

        C. Rother, V. Kolmogorov, V. Lempitsky, and M. Szummer
        Optimizing binary MRFs via extended roof duality
        CVPR 2007.

    The maxflow algorithm used is from

        Y. Boykov, V. Kolmogorov
        An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision
        PAMI, 26(9):1124-1137, September 2004.

    Functions Improve() and Probe() reuse search trees as described in

        "Efficiently Solving Dynamic Markov Random Fields Using Graph Cuts."
        Pushmeet Kohli and Philip H.S. Torr
        International Conference on Computer Vision (ICCV), 2005

    *************************************************************************************

    Example usage: minimize energy E(x,y) = 2*x + 3*(y+1) + (x+1)*(y+2), where x,y \in {0,1}.

    #include <stdio.h>
    #include "QPBO.h"

    int main()
    {
        typedef int REAL;
        QPBO<T>* q;

        q = new QPBO<T>(2, 1); // max number of nodes & edges
        q->AddNode(2); // add two nodes

        q->AddUnaryTerm(0, 0, 2); // add term 2*x
        q->AddUnaryTerm(1, 3, 6); // add term 3*(y+1)
        q->AddPairwiseTerm(0, 1, 2, 3, 4, 6); // add term (x+1)*(y+2)

        q->Solve();
        q->ComputeWeakPersistencies();

        int x = q->GetLabel(0);
        int y = q->GetLabel(1);
        printf("Solution: x=%d, y=%d\n", x, y);

        return 0;
    }

    *************************************************************************************
*/

#ifndef __QPBO_H__
#define __QPBO_H__

#include <string.h>
#include "block.h"

// NOTE: in UNIX, use -DNDEBUG flag to suppress assertions!!!
#include <assert.h>
#define user_assert assert // used for checking user input
#define code_assert assert // used for checking algorithm's correctness

// #define user_assert(ignore)((void) 0)
// #define code_assert(ignore)((void) 0)



// REAL: can be int, float, double.
// Current instantiations are in instances.inc
// NOTE: WITH FLOATING POINT NUMBERS ERRORS CAN ACCUMULATE.
// IT IS STRONGLY ADVISABLE TO USE INTEGERS!!! (IT IS ALSO *MUCH* FASTER).
template <typename T> class QPBO
{
public:
    typedef int NodeId;
    typedef int EdgeId;

    // Constructor.
    // The first argument gives an estimate of the maximum number of nodes that can be added
    // to the graph, and the second argument is an estimate of the maximum number of edges.
    // The last (optional) argument is the pointer to the function which will be called
    // if an error occurs; an error message is passed to this function.
    // If this argument is omitted, exit(1) will be called.
    //
    // IMPORTANT:
    // 1. It is possible to add more nodes to the graph than node_num_max
    // (and node_num_max can be zero). However, if the count is exceeded, then
    // the internal memory is reallocated (increased by 50%) which is expensive.
    // Also, temporarily the amount of allocated memory would be more than twice than needed.
    // Similarly for edges.
    //
    // 2. If Probe() is used with option=1 or option=2, then it is advisable to specify
    // a larger value of edge_num_max (e.g. twice the number of edges in the original energy).
    QPBO(int node_num_max, int edge_num_max, void (*err_function)(const char *) = NULL);
    // Copy constructor
    QPBO(QPBO<T>& q);

    // Destructor
    ~QPBO();

    // Save current reparameterisation of the energy to a text file. (Note: possibly twice the energy is saved).
    // Returns true if success, false otherwise.
    bool Save(char* filename);
    // Load energy from a text file. Current terms of the energy (if any) are destroyed.
    // Type identifier in the file (int/float/double) should match the type QPBO::REAL.
    // Returns true if success, false otherwise.
    bool Load(char* filename);

    // Removes all nodes and edges.
    // After that functions AddNode(), AddUnaryTerm(), AddPairwiseTerm() must be called again.
    //
    // Advantage compared to deleting QPBO and allocating it again:
    // no calls to delete/new (which could be quite slow).
    void Reset();

    int GetMaxEdgeNum(); // returns the number of edges for which the memory is allocated.
    void SetMaxEdgeNum(int num); // If num > edge_num_max then memory for edges is reallocated. Important for Probe() with option=1,2.

    ///////////////////////////////////////////////////////////////

    // Adds node(s) to the graph. By default, one node is added (num=1); then first call returns 0, second call returns 1, and so on.
    // If num>1, then several nodes are added, and NodeId of the first one is returned.
    // IMPORTANT: see note about the constructor
    NodeId AddNode(int num = 1);

    // Adds unary term Ei(x_i) to the energy function with cost values Ei(0)=E0, Ei(1)=E1.
    // Can be called multiple times for each node.
    void AddUnaryTerm(NodeId i, T E0, T E1);

    // Adds pairwise term Eij(x_i, x_j) with cost values E00, E01, E10, E11.
    // IMPORTANT: see note about the constructor
    EdgeId AddPairwiseTerm(NodeId i, NodeId j, T E00, T E01, T E10, T E11);

    // This function modifies an already existing pairwise term.
    void AddPairwiseTerm(EdgeId e, NodeId i, NodeId j, T E00, T E01, T E10, T E11);

    // If AddPairwiseTerm(i,j,...) has been called twice for some pairs of nodes,
    // then MergeParallelEdges() must be called before calling Solve()/Probe()/Improve().
    void MergeParallelEdges();

    ///////////////////////////////////////////////////////////////

    // Returns 0 or 1, if the node is labeled, and a negative number otherwise.
    // Can be called after Solve()/ComputeWeakPersistencies()/Probe()/Improve().
    int GetLabel(NodeId i);

    // Sets label for node i.
    // Can be called before Stitch()/Probe()/Improve().
    void SetLabel(NodeId i, int label);

    ///////////////////////////////////////////////////////////////
    // Read node & edge information.
    // Note: NodeId's are consecutive integers 0,1,...,GetNodeNum()-1.
    // However, EdgeId's are not necessarily consecutive.
    // The list of EdgeId's can be obtained as follows:
    //   QPBO<int>* q;
    //   QPBO<int>::EdgeId e;
    //   ...
    //   for (e=q->GetNextEdgeId(-1); e>=0; e=q->GetNextEdgeId(e))
    //   {
    //       ...
    //   }
    int GetNodeNum();
    EdgeId GetNextEdgeId(EdgeId e);

    // Read current reparameterization. Cost values are multiplied by 2 in the returned result.
    void GetTwiceUnaryTerm(NodeId i, T& E0, T& E1);
    void GetTwicePairwiseTerm(EdgeId e, /*output*/ NodeId& i, NodeId& j, T& E00, T& E01, T& E10, T& E11);

    ///////////////////////////////////////////////////////////////

    // Return energy/lower bound.
    // NOTE: in the current implementation Probe() may add constants to the energy
    // during transormations, so after Probe() the energy/lower bound would be shifted by some offset.

    // option == 0: returns 2 times the energy of internally stored solution which would be
    //              returned by GetLabel(). Negative values (unknown) are treated as 0.
    // option == 1: returns 2 times the energy of solution set by the user (via SetLabel()).
    T ComputeTwiceEnergy(int option = 0);
    // labeling must be an array of size nodeNum. Values other than 1 are treated as 0.
    T ComputeTwiceEnergy(int* labeling);
    // returns the lower bound defined by current reparameterizaion.
    T ComputeTwiceLowerBound();





    ///////////////////////////////////////////////////////////////
    //                   Basic QPBO algorithm                    //
    ///////////////////////////////////////////////////////////////

    // Runs QPBO. After calling Solve(), use GetLabel(i) to get label of node i.
    // Solve() produces a STRONGLY PERSISTENT LABELING. It means, in particular,
    // that if GetLabel(i)>=0 (i.e. node i is labeled) then x_i == GetLabel(i) for ALL global minima x.
    void Solve();

    // Can only be called immediately after Solve()/Probe() (and before any modifications are made to the energy).
    // Computes WEAKLY PERSISTENT LABELING. Use GetLabel() to read the result.
    // NOTE: if the energy is submodular, then ComputeWeakPersistences() will label all nodes (in general, this is not necessarily true for Solve()).
    void ComputeWeakPersistencies();

    // GetRegion()/Stitch():
    // ComputeWeakPersistencies() also splits pixels into regions (``strongly connected components'') U^0, U^1, ..., U^k as described in
    //
    //         A. Billionnet and B. Jaumard.
    //         A decomposition method for minimizing quadratic pseudoboolean functions.
    //         Operation Research Letters, 8:161–163, 1989.
    //
    //     For a review see also
    //
    //         V. Kolmogorov, C. Rother
    //         Minimizing non-submodular functions with graph cuts - a review
    //         Technical report MSR-TR-2006-100, July 2006. To appear in PAMI.
    //
    //     Nodes in U^0 are labeled, nodes in U^1, ..., U^k are unlabeled.
    //     (To find out to what region node i belongs, call GetRegion(i)).
    //     The user can use these regions as follows:
    //      -- For each r=1..k, compute somehow minimum x^r of the energy corresponding to region U^r.
    //         This energy can be obtained by calling GetPairwiseTerm() for edges inside the region.
    //         (There are no unary terms). Note that computing the global minimum is NP-hard;
    //	       it is up to the user to decide how to solve this problem.
    //      -- Set the labeling by calling SetLabel().
    //      -- Call Stitch(). It will compute a complete global minimum (in linear time).
    //      -- Call GetLabel() for nodes in U^1, ..., U^k to read new solution.
    //      Note that if the user can provides approximate rather than global minima x^r, then the stitching
    //      can still be done but the result is not guaranteed to be a *global* minimum.
    //
    // GetRegion()/Stitch() can be called only immediately after ComputeWeakPersistencies().
    // NOTE: Stitch() changes the stored energy!
    void Stitch();
    int GetRegion(NodeId i); // Returns a nonegative number which identifies the region. 0 corresponds to U^0.
                             // The numbers are not necessarily consecutive (i.e. some number may be missed).
                             // The maximum possible number is 2*nodeNum-5.

    //////////////////////////////////////////////////////////
    //                   QPBO extensions                    //
    //////////////////////////////////////////////////////////

    // Tries to improve the labeling provided by the user (via SetLabel()).
    // The new labeling is guaranteed to have the same or smaller energy than the input labeling.
    //
    // The procedure is as follows:
    //   1. Run QBPO
    //   2. Go through nodes in the order order_array[0], ..., order_array[N-1].
    //      If a node is unlabeled, fix it to the label provided by the user and run QBPO again.
    //   3. For remaining unlabeled nodes run set their labels to values provided by the user.
    //      (If order_array[] contains all nodes, then there should be no unlabeled nodes in step 3).
    //
    // New labeling can be obtained via GetLabel(). (The procedure also calls SetLabel() with
    // new labels, so Improve() can be called again). Returns true if success
    // (i.e. the labeling has changed and, thus, the energy has decreased), and false otherwise.
    //
    // If array fixed_pixels of size nodeNum is provided, then it is set as follows:
    // fixed_nodes[i] = 1 if node i was fixed during Improve(), and false otherwise.
    // order_array and fixed_pixels can point to the same array.
    bool Improve(int N, int* order_array, int* fixed_nodes = NULL);

    // Calls the function above with random permutation of nodes.
    // The user should initialize the seed before the first call (using srand()).
    // NOTE: IF THE CURRENT ITERATION IS UNSUCCESSFUL, THE NEXT
    // ITERATION MAY STILL BE SUCCESFULL SINCE A DIFFERENT PERMUTATION WILL BE USED.
    // A typical number of iterations could be e.g. 10-100.
    bool Improve();

    struct ProbeOptions
    {
        ProbeOptions()
            : directed_constraints(2),
              weak_persistencies(0),
              C(100000),
              order_array(NULL),
              order_seed(0),
              dilation(3),
              callback_fn(NULL)
        {
        }

        int directed_constraints; // 0: directed constraints are added only for existing edges
                                  // 1: all possible directed constraints are added, if there is sufficient space for edges (as specified by edge_num_max; see SetEdgeNumMax() function)
                                  // 2: all possible directed constraints are added. If necessary, new memory for edges is allocated.
        int weak_persistencies; // 0: use only strong persistency
                                // 1: use weak persistency in the main loop (but not for probing operations)

        T C; // Large constant used inside Probe() for enforcing directed constraints.
                // Note: small value may increase the number of iterations, large value may cause overflow.

        int* order_array; // if array of size nodeNum() is provided, then nodes are tested in the order order_array[0], order_array[1], ...
        unsigned int order_seed; // used only if order_array == NULL:
                                 // 0: default order (0,1,...,nodeNum()-1) is used.
                                 // otherwise: random permutation with random seed 'order_seed' is used.
        int dilation; // determines order of processing nodes (see Rother et al. CVPR'07):
                      // d<0:  one iteration tests all unlabeled nodes (i.e. fixes them to 0 and 1).
                      // d>=0: nodes within distance d from successful nodes are tested in the next iteration.

        bool (*callback_fn)(int unlabeled_num); // if callback_fn!=NULL, then after every testing a node Probe calls callback_fn();
                                                // unlabeled_num is the current number of remaining nodes in the energy.
                                                // If callback_fn returns true then Probe() terminates.
    };

    // Fixes some nodes to 0 or 1, contracts other nodes. These transformations
    // do not change global minima. The internally stored energy is modified accordingly.
    // (In particular, the new energy may have a different number of nodes and edges).
    //
    // Nodes of the old energy are associated with nodes of the new energy
    // (possibly with inversion: 0<-->1). This association is returned
    // in array mapping as follows:
    //   If old node i corresponds to new node j with inversion x (x=0,1) then
    //     mapping[i] = 2*j + x.
    //
    // If y is a global minimum of the new energy, then solution x defined by
    //     x[i] = (y[mapping[i]/2] + mapping[i]) % 2
    // is a global minimum of the original energy.
    //
    // Node 0 of the new energy is guaranteed to have optimal label 0 (y[0]=0),
    // therefore if mapping[i] < 2 then this is the optimal label for node i.
    //
    // Before calling Probe() you can call SetLabel() to set an input labeling x0.
    // During the procedure this labeling is transformed. The new labeling y0 can
    // be read via GetLabel() after Probe() (P+I method - see Rother et al, CVPR'07).
    void Probe(int* mapping, ProbeOptions& option);

    // If Probe() is called two times, then mappings mapping0 and mapping1 produced by the first and
    // second run can be combined using MergeMappings. Array mapping0 is updated accordingly.
    static void MergeMappings(int nodeNum0, int* mapping0, int* mapping1);


    //////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////










/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

private:
    // internal variables and functions

    struct Arc;

    struct Node
    {
        Arc		*first;		// first outcoming Arc

        Node	*next;		// pointer to the next active Node
                            // (or to itself if it is the last Node in the list)

        unsigned int is_sink : 1;	// flag showing whether the node is in the source or in the sink tree (if parent!=NULL)
        unsigned int is_marked : 1;	// set by mark_node()
        unsigned int is_in_changed_list : 1; // set by maxflow if the node is added to changed_list
        unsigned int is_removed : 1; // 1 means that the node is removed (for node[0][...])

        int	         label : 2;
        int	         label_after_fix0 : 2;
        int	         label_after_fix1 : 2;

        unsigned int list_flag : 2; // used in Probe() and Improve()

        unsigned int user_label : 1; // set by calling SetLabel()

        union
        {
            struct
            {
                // used inside maxflow algorithm
                int		TS;			// timestamp showing when DIST was computed
                int		DIST;		// distance to the terminal
                Arc		*parent;	// Node's parent
            };
            struct
            {
                int		region;
                Node	*dfs_parent;
                Arc		*dfs_current;
            };
        };

        T		tr_cap;		// if tr_cap > 0 then tr_cap is residual capacity of the Arc SOURCE->Node
                                // otherwise         -tr_cap is residual capacity of the Arc Node->SINK
    };

    struct Arc
    {
        Node		*head;		// Node the Arc points to
        Arc			*next;		// next Arc with the same originating Node
        Arc			*sister;	// reverse Arc

        T		r_cap;		// residual capacity
    };

    struct nodeptr
    {
        Node    	*ptr;
        nodeptr		*next;
    };
    static const int NODEPTR_BLOCK_SIZE = 128;

    Node	*nodes[2], *node_last[2], *node_max[2]; // node_last[k] = nodes[k]+node_num
                                                    // node_max[k] = nodes[k]+node_num_max
                                                    // nodes[1] = node_max[0]
    Arc		*arcs[2], *arc_max[2]; // arc_max[k] = arcs[k]+2*edge_num_max
                                   // arcs[1] = arc_max[0]

    Arc*	first_free; // list of empty spaces for edges.
    void InitFreeList();

    int		node_num;
    int		node_shift; // = node_num_max*sizeof(Node)
    int		arc_shift; // = 2*edge_num_max*sizeof(Arc)

    DBlock<nodeptr>		*nodeptr_block;

    void	(*error_function)(const char *);	// this function is called if a error occurs,
                                        // with a corresponding error message
                                        // (or exit(1) is called if it's NULL)

    T	zero_energy; // energy of solution (0,...,0)

    // reusing trees & list of changed pixels
    int					maxflow_iteration; // counter
    bool				keep_changed_list;
    Block<Node*>		*changed_list;

    /////////////////////////////////////////////////////////////////////////

    void get_type_information(const char*& type_name, const char*& type_format);

    void reallocate_nodes(int node_num_max_new);
    void reallocate_arcs(int arc_num_max_new);

    int	stage; // 0: maxflow is solved only for nodes in [nodes[0],node_last[0]).
               //    Arcs corresponding to supermodular edges are present in arcs[0] and arcs[1],
               //    but nodes do not point to them.
               // 1: maxflow is solved for the entire graph.
    bool all_edges_submodular;
    void TransformToSecondStage(bool copy_trees);

    static void ComputeWeights(T A, T B, T C, T D, T& ci, T& cj, T& cij, T& cji);
    bool IsNode0(Node* i) { return (i<nodes[1]); }
    Node* GetMate0(Node* i) { code_assert(i< nodes[1]); return (Node*)((char*)i + node_shift); }
    Node* GetMate1(Node* i) { code_assert(i>=nodes[1]); return (Node*)((char*)i - node_shift); }
    Node* GetMate(Node* i) { return IsNode0(i) ? GetMate0(i) : GetMate1(i); }
    bool IsArc0(Arc* a) { return (a<arcs[1]); }
    Arc* GetMate0(Arc* a) { code_assert(a< arcs[1]); return (Arc*)((char*)a + arc_shift); }
    Arc* GetMate1(Arc* a) { code_assert(a>=arcs[1]); return (Arc*)((char*)a - arc_shift); }
    Arc* GetMate(Arc* a) { return IsArc0(a) ? GetMate0(a) : GetMate1(a); }

    ProbeOptions probe_options;
    bool user_terminated;
    bool Probe(int* mapping); // Probe(int*,ProbeOptions&) iteratively calls Probe(int*)

    void TestRelaxedSymmetry(); // debug function

    T DetermineSaturation(Node* i);
    void AddUnaryTerm(Node* i, T E0, T E1);
    void FixNode(Node* i, int x); // fix i to label x. there must hold IsNode0(i).
    void ContractNodes(Node* i, Node* j, int swap); // there must hold IsNode0(i) && IsNode0(j) && (swap==0 || swap==1)
                                                    // enforces constraint i->label = (j->label + swap) mod 2
                                                    // i is kept, all arcs from j are deleted.
    int MergeParallelEdges(Arc* a1, Arc* a2); // there must hold (a1->sister->head == a2->sister->head) && IsNode0(a1->sister->head) &&
                                              //                 (a1->head == a2->head || a1->head = GetMate(a2->head))
                                              // returns 0 if a1 is removed, 1 otherwise
    bool AddDirectedConstraint0(Arc* a, int xi, int xj); // functions return true if the energy was changed.
    bool AddDirectedConstraint1(Arc* a, int xi, int xj); // ...0 checks whether submodurality needs to be swapped, ...1 preserves submodularity.
    void AddDirectedConstraint(Node* i, Node* j, int xi, int xj); // adds new edge. first_free must not be NULL.
    void AllocateNewEnergy(int* mapping);


    static void ComputeRandomPermutation(int N, int* permutation);

    struct FixNodeInfo { Node* i; T INFTY; };
    Block<FixNodeInfo>* fix_node_info_list;

    /////////////////////////////////////////////////////////////////////////

    Node				*queue_first[2], *queue_last[2];	// list of active nodes
    nodeptr				*orphan_first, *orphan_last;		// list of pointers to orphans
    int					TIME;								// monotonically increasing global counter

    /////////////////////////////////////////////////////////////////////////

    // functions for processing active list
    void set_active(Node *i);
    Node *next_active();

    // functions for processing orphans list
    void set_orphan_front(Node* i); // add to the beginning of the list
    void set_orphan_rear(Node* i);  // add to the end of the list

    void mark_node(Node* i);
    void add_to_changed_list(Node* i);

    void maxflow(bool reuse_trees = false, bool keep_changed_list = false);
    void maxflow_init();             // called if reuse_trees == false
    void maxflow_reuse_trees_init(); // called if reuse_trees == true
    void augment(Arc *middle_arc);
    void process_source_orphan(Node *i);
    void process_sink_orphan(Node *i);

    int what_segment(Node* i, int default_segm = 0);

    void test_consistency(Node* current_node=NULL); // debug function
};











///////////////////////////////////////
// Implementation - inline functions //
///////////////////////////////////////



template <typename T>
    inline typename QPBO<T>::NodeId QPBO<T>::AddNode(int num)
{
    user_assert(num >= 0);

    if (node_last[0] + num > node_max[0])
    {
        int node_num_max = node_shift / sizeof(Node);
        node_num_max += node_num_max / 2;
        if (node_num_max < (int)(node_last[0] + num - nodes[0]) + 1) node_num_max = (int)(node_last[0] + num - nodes[0]) + 1;
        reallocate_nodes(node_num_max);
    }

    memset(node_last[0], 0, num*sizeof(Node));
    NodeId i = node_num;
    node_num += num;
    node_last[0] += num;

    if (stage)
    {
        memset(node_last[1], 0, num*sizeof(Node));
        node_last[1] += num;
    }

    return i;
}

template <typename T>
    inline void QPBO<T>::AddUnaryTerm(NodeId i, T E0, T E1)
{
    user_assert(i >= 0 && i < node_num);

    nodes[0][i].tr_cap += E1 - E0;
    if (stage) nodes[1][i].tr_cap -= E1 - E0;

    zero_energy += E0;
}

template <typename T>
    inline void QPBO<T>::AddUnaryTerm(Node* i, T E0, T E1)
{
    code_assert(i >= nodes[0] && i<node_last[0]);

    i->tr_cap += E1 - E0;
    if (stage) GetMate0(i)->tr_cap -= E1 - E0;

    zero_energy += E0;
}

template <typename T>
    inline int QPBO<T>::what_segment(Node* i, int default_segm)
{
    if (i->parent)
    {
        return (i->is_sink) ? 1 : 0;
    }
    else
    {
        return default_segm;
    }
}

template <typename T>
    inline void QPBO<T>::mark_node(Node* i)
{
    if (!i->next)
    {
        /* it's not in the list yet */
        if (queue_last[1]) queue_last[1] -> next = i;
        else               queue_first[1]        = i;
        queue_last[1] = i;
        i -> next = i;
    }
    i->is_marked = 1;
}

template <typename T>
    inline int QPBO<T>::GetLabel(NodeId i)
{
    user_assert(i >= 0 && i < node_num);

    return nodes[0][i].label;
}

template <typename T>
    inline int QPBO<T>::GetRegion(NodeId i)
{
    user_assert(i >= 0 && i < node_num);
    user_assert(stage == 1);

    return nodes[0][i].region;
}

template <typename T>
    inline void QPBO<T>::SetLabel(NodeId i, int label)
{
    user_assert(i >= 0 && i < node_num);

    nodes[0][i].user_label = label;
}

template <typename T>
    inline void QPBO<T>::GetTwiceUnaryTerm(NodeId i, T &E0, T &E1)
{
    user_assert(i >= 0 && i < node_num);

    E0 = 0;
    if (stage == 0) E1 = 2*nodes[0][i].tr_cap;
    else            E1 = nodes[0][i].tr_cap - nodes[1][i].tr_cap;
}

template <typename T>
    inline void QPBO<T>::GetTwicePairwiseTerm(EdgeId e, NodeId& _i, NodeId& _j, T &E00, T &E01, T &E10, T &E11)
{
    user_assert(e >= 0 && arcs[0][2*e].sister);

    Arc* a;
    Arc* a_mate;
    if (IsNode0(arcs[0][2*e+1].head))
    {
        a = &arcs[0][2*e];
        a_mate = &arcs[1][2*e];
    }
    else
    {
        a = &arcs[1][2*e+1];
        a_mate = &arcs[0][2*e+1];
    }
    Node* i = a->sister->head;
    Node* j = a->head;
    _i = (int)(i - nodes[0]);

    if (IsNode0(j))
    {
        E00 = E11 = 0;
        if (stage == 0) { E01 = 2*a->r_cap; E10 = 2*a->sister->r_cap; }
        else            { E01 = a->r_cap + a_mate->r_cap; E10 = a->sister->r_cap + a_mate->sister->r_cap; }
        _j = (int)(j - nodes[0]);
    }
    else
    {
        E01 = E10 = 0;
        if (stage == 0) { E00 = 2*a->r_cap; E11 = 2*a->sister->r_cap; }
        else            { E00 = a->r_cap + a_mate->r_cap; E11 = a->sister->r_cap + a_mate->sister->r_cap; }
        _j = (int)(j - nodes[1]);
    }
}

template <typename T>
    inline int QPBO<T>::GetNodeNum()
{
    return (int)(node_last[0] - nodes[0]);
}

template <typename T>
    inline typename QPBO<T>::EdgeId QPBO<T>::GetNextEdgeId(EdgeId e)
{
    Arc* a;
    for (a=&arcs[0][2*(++e)]; a<arc_max[0]; a+=2, e++)
    {
        if (a->sister) return e;
    }
    return -1;
}

template <typename T>
    inline int QPBO<T>::GetMaxEdgeNum()
{
    return (int)(arc_max[0]-arcs[0])/2;
}


template <typename T>
    inline void QPBO<T>::ComputeWeights(T A, T B, T C, T D, // input - E00=A, E01=B, E10=C, E11=D
    T &ci, T &cj, T &cij, T &cji // output - edge weights
    )
{
    /*
    E = A A  +  0   B-A
        D D     C-D 0
    Add edges for the first term
    */
    ci = D - A;
    B -= A; C -= D;

    /* now need to represent
    0 B
    C 0
    */

    if (B < 0)
    {
        /* Write it as
        B B  +  -B 0  +  0   0
        0 0     -B 0     B+C 0
        */
        ci += -B; /* first term */
        cj = B; /* second term */
        cji = B+C; /* third term */
        cij = 0;
    }
    else if (C < 0)
    {
        /* Write it as
        -C -C  +  C 0  +  0 B+C
            0  0     C 0     0 0
        */
        ci += C; /* first term */
        cj = -C; /* second term */
        cij = B+C; /* third term */
        cji = 0;
    }
    else /* B >= 0, C >= 0 */
    {
        cj = 0;
        cij = B;
        cji = C;
    }
}

/*
    special constants for node->parent
*/
#define QPBO_MAXFLOW_TERMINAL ( (Arc *) 1 )		/* to terminal */
#define QPBO_MAXFLOW_ORPHAN   ( (Arc *) 2 )		/* orphan */

#endif
