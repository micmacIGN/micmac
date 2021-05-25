/* QPBO_maxflow.cpp */
/*
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


#include <stdio.h>
#include "QPBO.h"

#ifdef REAL
#undef REAL
#endif

#define INFINITE_D ((int)(((unsigned)-1)/2))		/* infinite distance to the terminal */

/***********************************************************************/

/*
    Functions for processing active list.
    i->next points to the next node in the list
    (or to i, if i is the last node in the list).
    If i->next is NULL iff i is not in the list.

    There are two queues. Active nodes are added
    to the end of the second queue and read from
    the front of the first queue. If the first queue
    is empty, it is replaced by the second queue
    (and the second queue becomes empty).
*/


template <typename REAL>
    inline void QPBO<REAL>::set_active(Node *i)
{
    if (!i->next)
    {
        /* it's not in the list yet */
        if (queue_last[1]) queue_last[1] -> next = i;
        else               queue_first[1]        = i;
        queue_last[1] = i;
        i -> next = i;
    }
}

/*
    Returns the next active node.
    If it is connected to the sink, it stays in the list,
    otherwise it is removed from the list
*/
template <typename REAL>
    inline typename QPBO<REAL>::Node* QPBO<REAL>::next_active()
{
    Node *i;

    while ( 1 )
    {
        if (!(i=queue_first[0]))
        {
            queue_first[0] = i = queue_first[1];
            queue_last[0]  = queue_last[1];
            queue_first[1] = NULL;
            queue_last[1]  = NULL;
            if (!i) return NULL;
        }

        /* remove it from the active list */
        if (i->next == i) queue_first[0] = queue_last[0] = NULL;
        else              queue_first[0] = i -> next;
        i -> next = NULL;

        /* a node in the list is active iff it has a parent */
        if (i->parent) return i;
    }
}

/***********************************************************************/

template <typename REAL>
    inline void QPBO<REAL>::set_orphan_front(Node *i)
{
    nodeptr *np;
    i -> parent = QPBO_MAXFLOW_ORPHAN;
    np = nodeptr_block -> New();
    np -> ptr = i;
    np -> next = orphan_first;
    orphan_first = np;
}

template <typename REAL>
    inline void QPBO<REAL>::set_orphan_rear(Node *i)
{
    nodeptr *np;
    i -> parent = QPBO_MAXFLOW_ORPHAN;
    np = nodeptr_block -> New();
    np -> ptr = i;
    if (orphan_last) orphan_last -> next = np;
    else             orphan_first        = np;
    orphan_last = np;
    np -> next = NULL;
}

/***********************************************************************/

template <typename REAL>
    inline void QPBO<REAL>::add_to_changed_list(Node *i)
{
    if (keep_changed_list)
    {
        if (!IsNode0(i)) i = GetMate1(i);
        if (!i->is_in_changed_list)
        {
            Node** ptr = changed_list->New();
            *ptr = i;;
            i->is_in_changed_list = true;
        }
    }
}

/***********************************************************************/

template <typename REAL>
    void QPBO<REAL>::maxflow_init()
{
    Node *i;

    queue_first[0] = queue_last[0] = NULL;
    queue_first[1] = queue_last[1] = NULL;
    orphan_first = NULL;

    TIME = 0;

    for (i=nodes[0]; i<node_last[stage]; i++)
    {
        if (i==node_last[0]) i = nodes[1];

        i -> next = NULL;
        i -> is_marked = 0;
        i -> is_in_changed_list = 0;
        i -> TS = TIME;
        if (i->tr_cap > 0)
        {
            /* i is connected to the source */
            i -> is_sink = 0;
            i -> parent = QPBO_MAXFLOW_TERMINAL;
            set_active(i);
            i -> DIST = 1;
        }
        else if (i->tr_cap < 0)
        {
            /* i is connected to the sink */
            i -> is_sink = 1;
            i -> parent = QPBO_MAXFLOW_TERMINAL;
            set_active(i);
            i -> DIST = 1;
        }
        else
        {
            i -> parent = NULL;
        }
    }
}

template <typename REAL>
    void QPBO<REAL>::maxflow_reuse_trees_init()
{
    Node* i;
    Node* j;
    Node* queue = queue_first[1];
    Arc* a;
    nodeptr* np;

    queue_first[0] = queue_last[0] = NULL;
    queue_first[1] = queue_last[1] = NULL;
    orphan_first = orphan_last = NULL;

    TIME ++;

    while ((i=queue))
    {
        queue = i->next;
        if (queue == i) queue = NULL;
        if (IsNode0(i))
        {
            if (i->is_removed) continue;
        }
        else
        {
            if (GetMate1(i)->is_removed) continue;
        }
        i->next = NULL;
        i->is_marked = 0;
        set_active(i);

        if (i->tr_cap == 0)
        {
            if (i->parent) set_orphan_rear(i);
            continue;
        }

        if (i->tr_cap > 0)
        {
            if (!i->parent || i->is_sink)
            {
                i->is_sink = 0;
                for (a=i->first; a; a=a->next)
                {
                    j = a->head;
                    if (!j->is_marked)
                    {
                        if (j->parent == a->sister) set_orphan_rear(j);
                        if (j->parent && j->is_sink && a->r_cap > 0) set_active(j);
                    }
                }
                add_to_changed_list(i);
            }
        }
        else
        {
            if (!i->parent || !i->is_sink)
            {
                i->is_sink = 1;
                for (a=i->first; a; a=a->next)
                {
                    j = a->head;
                    if (!j->is_marked)
                    {
                        if (j->parent == a->sister) set_orphan_rear(j);
                        if (j->parent && !j->is_sink && a->sister->r_cap > 0) set_active(j);
                    }
                }
                add_to_changed_list(i);
            }
        }
        i->parent = QPBO_MAXFLOW_TERMINAL;
        i -> TS = TIME;
        i -> DIST = 1;
    }

    code_assert(stage == 1);
    //test_consistency();

    /* adoption */
    while ((np=orphan_first))
    {
        orphan_first = np -> next;
        i = np -> ptr;
        nodeptr_block -> Delete(np);
        if (!orphan_first) orphan_last = NULL;
        if (i->is_sink) process_sink_orphan(i);
        else            process_source_orphan(i);
    }
    /* adoption end */

    //test_consistency();
}

template <typename REAL>
    void QPBO<REAL>::augment(Arc *middle_arc)
{
    Node *i;
    Arc *a;
    REAL bottleneck;


    /* 1. Finding bottleneck capacity */
    /* 1a - the source tree */
    bottleneck = middle_arc -> r_cap;
    for (i=middle_arc->sister->head; ; i=a->head)
    {
        a = i -> parent;
        if (a == QPBO_MAXFLOW_TERMINAL) break;
        if (bottleneck > a->sister->r_cap) bottleneck = a -> sister -> r_cap;
    }
    if (bottleneck > i->tr_cap) bottleneck = i -> tr_cap;
    /* 1b - the sink tree */
    for (i=middle_arc->head; ; i=a->head)
    {
        a = i -> parent;
        if (a == QPBO_MAXFLOW_TERMINAL) break;
        if (bottleneck > a->r_cap) bottleneck = a -> r_cap;
    }
    if (bottleneck > - i->tr_cap) bottleneck = - i -> tr_cap;


    /* 2. Augmenting */
    /* 2a - the source tree */
    middle_arc -> sister -> r_cap += bottleneck;
    middle_arc -> r_cap -= bottleneck;
    for (i=middle_arc->sister->head; ; i=a->head)
    {
        a = i -> parent;
        if (a == QPBO_MAXFLOW_TERMINAL) break;
        a -> r_cap += bottleneck;
        a -> sister -> r_cap -= bottleneck;
        if (!a->sister->r_cap)
        {
            set_orphan_front(i); // add i to the beginning of the adoption list
        }
    }
    i -> tr_cap -= bottleneck;
    if (!i->tr_cap)
    {
        set_orphan_front(i); // add i to the beginning of the adoption list
    }
    /* 2b - the sink tree */
    for (i=middle_arc->head; ; i=a->head)
    {
        a = i -> parent;
        if (a == QPBO_MAXFLOW_TERMINAL) break;
        a -> sister -> r_cap += bottleneck;
        a -> r_cap -= bottleneck;
        if (!a->r_cap)
        {
            set_orphan_front(i); // add i to the beginning of the adoption list
        }
    }
    i -> tr_cap += bottleneck;
    if (!i->tr_cap)
    {
        set_orphan_front(i); // add i to the beginning of the adoption list
    }
}

/***********************************************************************/

template <typename REAL>
    void QPBO<REAL>::process_source_orphan(Node *i)
{
    Node *j;
    Arc *a0, *a0_min = NULL, *a;
    int d, d_min = INFINITE_D;

    /* trying to find a new parent */
    for (a0=i->first; a0; a0=a0->next)
    if (a0->sister->r_cap)
    {
        j = a0 -> head;
        if (!j->is_sink && (a=j->parent))
        {
            /* checking the origin of j */
            d = 0;
            while ( 1 )
            {
                if (j->TS == TIME)
                {
                    d += j -> DIST;
                    break;
                }
                a = j -> parent;
                d ++;
                if (a==QPBO_MAXFLOW_TERMINAL)
                {
                    j -> TS = TIME;
                    j -> DIST = 1;
                    break;
                }
                if (a==QPBO_MAXFLOW_ORPHAN) { d = INFINITE_D; break; }
                j = a -> head;
            }
            if (d<INFINITE_D) /* j originates from the source - done */
            {
                if (d<d_min)
                {
                    a0_min = a0;
                    d_min = d;
                }
                /* set marks along the path */
                for (j=a0->head; j->TS!=TIME; j=j->parent->head)
                {
                    j -> TS = TIME;
                    j -> DIST = d --;
                }
            }
        }
    }

    if ((i->parent = a0_min))
    {
        i -> TS = TIME;
        i -> DIST = d_min + 1;
    }
    else
    {
        /* no parent is found */
        add_to_changed_list(i);

        /* process neighbors */
        for (a0=i->first; a0; a0=a0->next)
        {
            j = a0 -> head;
            if (!j->is_sink && (a=j->parent))
            {
                if (a0->sister->r_cap) set_active(j);
                if (a!=QPBO_MAXFLOW_TERMINAL && a!=QPBO_MAXFLOW_ORPHAN && a->head==i)
                {
                    set_orphan_rear(j); // add j to the end of the adoption list
                }
            }
        }
    }
}

template <typename REAL>
    void QPBO<REAL>::process_sink_orphan(Node *i)
{
    Node *j;
    Arc *a0, *a0_min = NULL, *a;
    int d, d_min = INFINITE_D;

    /* trying to find a new parent */
    for (a0=i->first; a0; a0=a0->next)
    if (a0->r_cap)
    {
        j = a0 -> head;
        if ((a=j->parent) && j->is_sink)
        {
            /* checking the origin of j */
            d = 0;
            while ( 1 )
            {
                if (j->TS == TIME)
                {
                    d += j -> DIST;
                    break;
                }
                a = j -> parent;
                d ++;
                if (a==QPBO_MAXFLOW_TERMINAL)
                {
                    j -> TS = TIME;
                    j -> DIST = 1;
                    break;
                }
                if (a==QPBO_MAXFLOW_ORPHAN) { d = INFINITE_D; break; }
                j = a -> head;
            }
            if (d<INFINITE_D) /* j originates from the sink - done */
            {
                if (d<d_min)
                {
                    a0_min = a0;
                    d_min = d;
                }
                /* set marks along the path */
                for (j=a0->head; j->TS!=TIME; j=j->parent->head)
                {
                    j -> TS = TIME;
                    j -> DIST = d --;
                }
            }
        }
    }

    if ((i->parent = a0_min))
    {
        i -> TS = TIME;
        i -> DIST = d_min + 1;
    }
    else
    {
        /* no parent is found */
        add_to_changed_list(i);

        /* process neighbors */
        for (a0=i->first; a0; a0=a0->next)
        {
            j = a0 -> head;
            if ((a=j->parent) && j->is_sink)
            {
                if (a0->r_cap) set_active(j);
                if (a!=QPBO_MAXFLOW_TERMINAL && a!=QPBO_MAXFLOW_ORPHAN && a->head==i)
                {
                    set_orphan_rear(j); // add j to the end of the adoption list
                }
            }
        }
    }
}

/***********************************************************************/

template <typename REAL>
    void QPBO<REAL>::maxflow(bool reuse_trees, bool _keep_changed_list)
{
    Node *i, *j, *current_node = NULL;
    Arc *a;
    nodeptr *np, *np_next;

    if (!nodeptr_block)
    {
        nodeptr_block = new DBlock<nodeptr>(NODEPTR_BLOCK_SIZE, error_function);
    }

    if (maxflow_iteration == 0)
    {
        reuse_trees = false;
        _keep_changed_list = false;
    }

    keep_changed_list = _keep_changed_list;
    if (keep_changed_list)
    {
        if (!changed_list) changed_list = new Block<Node*>(NODEPTR_BLOCK_SIZE, error_function);
    }

    if (reuse_trees) maxflow_reuse_trees_init();
    else             maxflow_init();

    // main loop
    while ( 1 )
    {
        // test_consistency(current_node);

        if ((i=current_node))
        {
            i -> next = NULL; /* remove active flag */
            if (!i->parent) i = NULL;
        }
        if (!i)
        {
            if (!(i = next_active())) break;
        }

        /* growth */
        if (!i->is_sink)
        {
            /* grow source tree */
            for (a=i->first; a; a=a->next)
            if (a->r_cap)
            {
                j = a -> head;
                if (!j->parent)
                {
                    j -> is_sink = 0;
                    j -> parent = a -> sister;
                    j -> TS = i -> TS;
                    j -> DIST = i -> DIST + 1;
                    set_active(j);
                    add_to_changed_list(j);
                }
                else if (j->is_sink) break;
                else if (j->TS <= i->TS &&
                         j->DIST > i->DIST)
                {
                    /* heuristic - trying to make the distance from j to the source shorter */
                    j -> parent = a -> sister;
                    j -> TS = i -> TS;
                    j -> DIST = i -> DIST + 1;
                }
            }
        }
        else
        {
            /* grow sink tree */
            for (a=i->first; a; a=a->next)
            if (a->sister->r_cap)
            {
                j = a -> head;
                if (!j->parent)
                {
                    j -> is_sink = 1;
                    j -> parent = a -> sister;
                    j -> TS = i -> TS;
                    j -> DIST = i -> DIST + 1;
                    set_active(j);
                    add_to_changed_list(j);
                }
                else if (!j->is_sink) { a = a -> sister; break; }
                else if (j->TS <= i->TS &&
                         j->DIST > i->DIST)
                {
                    /* heuristic - trying to make the distance from j to the sink shorter */
                    j -> parent = a -> sister;
                    j -> TS = i -> TS;
                    j -> DIST = i -> DIST + 1;
                }
            }
        }

        TIME ++;

        if (a)
        {
            i -> next = i; /* set active flag */
            current_node = i;

            /* augmentation */
            augment(a);
            /* augmentation end */

            /* adoption */
            while ((np=orphan_first))
            {
                np_next = np -> next;
                np -> next = NULL;

                while ((np=orphan_first))
                {
                    orphan_first = np -> next;
                    i = np -> ptr;
                    nodeptr_block -> Delete(np);
                    if (!orphan_first) orphan_last = NULL;
                    if (i->is_sink) process_sink_orphan(i);
                    else            process_source_orphan(i);
                }

                orphan_first = np_next;
            }
            /* adoption end */
        }
        else current_node = NULL;
    }
    // test_consistency();

    if (!reuse_trees || (maxflow_iteration % 64) == 0)
    {
        delete nodeptr_block;
        nodeptr_block = NULL;
    }

    maxflow_iteration ++;
}

/***********************************************************************/


template <typename REAL>
    void QPBO<REAL>::test_consistency(Node* current_node)
{
    Node *i;
    Arc *a;
    int r;
    int num1 = 0, num2 = 0;

    // test whether all nodes i with i->next!=NULL are indeed in the queue
    for (i=nodes[0]; i<node_last[stage]; i++)
    {
        if (i==node_last[0]) i = nodes[1];
        if ((IsNode0(i) && i->is_removed) || (!IsNode0(i) && GetMate1(i)->is_removed))
        {
            code_assert(i->first == NULL);
            continue;
        }

        if (i->next || i==current_node) num1 ++;
    }
    for (r=0; r<3; r++)
    {
        i = (r == 2) ? current_node : queue_first[r];
        if (i)
        for ( ; ; i=i->next)
        {
            code_assert((IsNode0(i) && !i->is_removed) || (!IsNode0(i) && !GetMate1(i)->is_removed));
            num2 ++;
            if (i->next == i)
            {
                if (r<2) code_assert(i == queue_last[r]);
                else     code_assert(i == current_node);
                break;
            }
        }
    }
    code_assert(num1 == num2);

    for (i=nodes[0]; i<node_last[stage]; i++)
    {
        if (i==node_last[0]) i = nodes[1];
        if ((IsNode0(i) && i->is_removed) || (!IsNode0(i) && GetMate1(i)->is_removed)) continue;

        // test whether all edges in seach trees are non-saturated
        if (i->parent == NULL) {}
        else if (i->parent == QPBO_MAXFLOW_ORPHAN) {}
        else if (i->parent == QPBO_MAXFLOW_TERMINAL)
        {
            if (!i->is_sink) code_assert(i->tr_cap > 0);
            else             code_assert(i->tr_cap < 0);
        }
        else
        {
            if (!i->is_sink) code_assert (i->parent->sister->r_cap > 0);
            else             code_assert (i->parent->r_cap > 0);
        }
        // test whether passive nodes in search trees have neighbors in
        // a different tree through non-saturated edges
        if (i->parent && !i->next)
        {
            if (!i->is_sink)
            {
                code_assert(i->tr_cap >= 0);
                for (a=i->first; a; a=a->next)
                {
                    if (a->r_cap > 0) code_assert(a->head->parent && !a->head->is_sink);
                }
            }
            else
            {
                code_assert(i->tr_cap <= 0);
                for (a=i->first; a; a=a->next)
                {
                    if (a->sister->r_cap > 0) code_assert(a->head->parent && a->head->is_sink);
                }
            }
        }
        // test marking invariants
        if (i->parent && i->parent!=QPBO_MAXFLOW_ORPHAN && i->parent!=QPBO_MAXFLOW_TERMINAL)
        {
            code_assert(i->TS <= i->parent->head->TS);
            if (i->TS == i->parent->head->TS) code_assert(i->DIST > i->parent->head->DIST);
        }
    }
}

#include "instances.inc"
