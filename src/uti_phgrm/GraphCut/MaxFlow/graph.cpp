/* graph.cpp */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "graph.h"
#include "maxflow.h"  // MODIF MPD : AJOUT

template <typename captype, typename tcaptype, typename flowtype> 
	Graph<captype, tcaptype, flowtype>::Graph(int node_num_max, int edge_num_max, void (*err_function)(const char *))
	: node_num(0),
	  nodeptr_block(NULL),
	  error_function(err_function)
{
	if (node_num_max < 16) node_num_max = 16;
	if (edge_num_max < 16) edge_num_max = 16;

	nodes = (node*) malloc(node_num_max*sizeof(node));
	arcs = (arc*) malloc(2*edge_num_max*sizeof(arc));
	if (!nodes || !arcs) { if (error_function) (*error_function)("Not enough memory!"); exit(1); }

	node_last = nodes;
	node_max = nodes + node_num_max;
	arc_last = arcs;
	arc_max = arcs + 2*edge_num_max;

	maxflow_iteration = 0;
	flow = 0;
}

template <typename captype, typename tcaptype, typename flowtype> 
	Graph<captype,tcaptype,flowtype>::~Graph()
{
	if (nodeptr_block) 
	{ 
		delete nodeptr_block; 
		nodeptr_block = NULL; 
	}
	free(nodes);
	free(arcs);
}

template <typename captype, typename tcaptype, typename flowtype> 
	void Graph<captype,tcaptype,flowtype>::reset()
{
	node_last = nodes;
	arc_last = arcs;
	node_num = 0;

	if (nodeptr_block) 
	{ 
		delete nodeptr_block; 
		nodeptr_block = NULL; 
	}

	maxflow_iteration = 0;
	flow = 0;
}

template <typename captype, typename tcaptype, typename flowtype>
void Graph<captype, tcaptype, flowtype>::reallocate_nodes(int num) {
        int node_num_max = (int)(node_max - nodes);
        size_t nodes_old = (size_t)nodes;

        node_num_max += node_num_max / 2;
        if (node_num_max < node_num + num) node_num_max = node_num + num;

        nodes = (node*)realloc(nodes, node_num_max * sizeof(node));

        if (!nodes) {
                if (error_function) (*error_function)("Not enough memory!");
                exit(1);
        }

        node_last = nodes + node_num;
        node_max = nodes + node_num_max;

        if ((size_t)nodes != nodes_old) {
                node* i;
                arc* a;
                for (i = nodes; i < node_last; i++) {
                    if (i->next)
                        i->next =
                            (node*)((size_t)i->next +
                                    (((size_t)nodes) - ((size_t)nodes_old)));
                }
                for (a = arcs; a < arc_last; a++) {
                    a->head = (node*)((size_t)a->head +
                                      (((size_t)nodes) - ((size_t)nodes_old)));
                }
        }
}

template <typename captype, typename tcaptype, typename flowtype>
void Graph<captype, tcaptype, flowtype>::reallocate_arcs() {
        int arc_num_max = (int)(arc_max - arcs);
        int arc_num = (int)(arc_last - arcs);
        size_t arcs_old = (size_t)arcs;

        arc_num_max += arc_num_max / 2;
        if (arc_num_max & 1) arc_num_max++;
        arcs = (arc*)realloc(arcs, arc_num_max * sizeof(arc));
        if (!arcs) {
                if (error_function) (*error_function)("Not enough memory!");
                exit(1);
        }

        arc_last = arcs + arc_num;
        arc_max = arcs + arc_num_max;

        if ((size_t)arcs != arcs_old) {
                node* i;
                arc* a;
                for (i = nodes; i < node_last; i++) {
                    if (i->first)
                        i->first = (arc*)((size_t)i->first +
                                          (((size_t)arcs) - ((size_t)arcs_old)));
                    if (i->parent && i->parent != ORPHAN &&
                        i->parent != TERMINAL)
                        i->parent = (arc*)((size_t)i->parent +
                                           (((size_t)arcs) - ((size_t)arcs_old)));
                }
                for (a = arcs; a < arc_last; a++) {
                    if (a->next)
                        a->next = (arc*)((size_t)a->next +
                                         (((size_t)arcs) - ((size_t)arcs_old)));
                    a->sister = (arc*)((size_t)a->sister +
                                       (((size_t)arcs) - ((size_t)arcs_old)));
                }
        }
}

#include "instances.inc"
