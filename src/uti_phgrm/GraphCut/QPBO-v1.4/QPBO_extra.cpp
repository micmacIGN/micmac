/* QPBO_extra.cpp */
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
#include <stdlib.h>
#include <string.h>
#include "QPBO.h"

#ifdef REAL
#undef REAL
#endif

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

template <typename REAL>
	void QPBO<REAL>::ComputeRandomPermutation(int N, int* permutation)
{
	int i, j, k;
	for (i=0; i<N; i++)
	{
		permutation[i] = i;
	}
	for (i=0; i<N-1; i++)
	{
		j = i + (int)((rand()/(1.0+(double)RAND_MAX))*(N-i));
		if (j>N-1) j = N-1;
		k = permutation[j]; permutation[j] = permutation[i]; permutation[i] = k;
	}
}

template <typename REAL>
	void QPBO<REAL>::MergeMappings(int nodeNum0, int* mapping0, int* mapping1)
{
	int i;
	for (i=0; i<nodeNum0; i++)
	{
		int j = mapping0[i] / 2;
		int k = mapping1[j] / 2;
		mapping0[i] = 2*k + ((mapping0[i] + mapping1[j]) % 2);
	}
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

#define SET_SISTERS(a, a_rev)    (a)->sister = (a_rev); (a_rev)->sister = (a);
#define SET_FROM(a, i)           (a)->next = (i)->first; (i)->first = (a);
#define REMOVE_FROM(a, i)        if ((i)->first==(a)) (i)->first=(a)->next;\
								 else { Arc* a_TMP_REMOVE_FROM; for (a_TMP_REMOVE_FROM=i->first; ; a_TMP_REMOVE_FROM=a_TMP_REMOVE_FROM->next)\
												 if (a_TMP_REMOVE_FROM->next==(a)) { a_TMP_REMOVE_FROM->next=(a)->next; break; } }
#define SET_TO(a, j)             (a)->head = (j);


template <typename REAL>
	inline void QPBO<REAL>::FixNode(Node* i, int x)
{
	Node* _i[2] = { i, GetMate0(i) };
	Arc* a;
	Arc* a_next;

	for (a=_i[x]->first; a; a=a->next)
	{
		mark_node(a->head);
		a->head->tr_cap += a->r_cap;
		REMOVE_FROM(a->sister, a->head);
		a->sister->sister = NULL;
		a->sister = NULL;
	}
	for (a=_i[1-x]->first; a; a=a_next)
	{
		mark_node(a->head);
		a->head->tr_cap -= a->sister->r_cap;
		REMOVE_FROM(a->sister, a->head);
		a->sister->sister = NULL;
		a->sister = NULL;

		a_next = a->next;
		a->next = first_free;
		first_free = a;
	}
	_i[0]->first = _i[1]->first = NULL;
}

template <typename REAL>
	inline void QPBO<REAL>::ContractNodes(Node* i, Node* j, int swap)
{
	code_assert(IsNode0(i) && IsNode0(j) && swap>=0 && swap<=1);

	Node* _i[2] = { i, GetMate0(i) };
	Node* _j[2];
	Arc* a;
	Arc* a_selfloop = NULL;
	int x;

	if (swap == 0) { _j[0] = j; _j[1] = GetMate0(j); }
	else           { _j[1] = j; _j[0] = GetMate0(j); }

	_i[0]->tr_cap += _j[0]->tr_cap;
	_i[1]->tr_cap += _j[1]->tr_cap;
	for (x=0; x<2; x++)
	{
		Arc* a_next;
		for (a=_j[x]->first; a; a=a_next)
		{
			mark_node(a->head);
			a_next = a->next;
			if (a->head == _i[x])
			{
				REMOVE_FROM(a->sister, _i[x]);
				a->sister->sister = NULL;
				a->sister = NULL;
				a_selfloop = a;
			}
			else if (a->head == _i[1-x])
			{
				REMOVE_FROM(a->sister, _i[1-x]);
				_i[x]->tr_cap   -= a->r_cap;
				_i[1-x]->tr_cap += a->r_cap;
				a->sister->sister = NULL;
				a->sister = NULL;
			}
			else
			{
				SET_FROM(a, _i[x]);
				SET_TO(a->sister, _i[x]);
			}
		}
	}
	_j[0]->first = _j[1]->first = NULL;

	if (a_selfloop)
	{
		a_selfloop->next = first_free;
		first_free = a_selfloop;
	}
}

template <typename REAL>
	int QPBO<REAL>::MergeParallelEdges(Arc* a1, Arc* a2)
{
	code_assert(a1->sister->head == a2->sister->head && IsNode0(a1->sister->head));
	code_assert(a1->head == a2->head || a1->head == GetMate(a2->head));

	REAL delta;
	int x;
	Node* _i[2];
	Node* _j[2];
	Arc* _a1[2] = { a1, GetMate(a1) };
	Arc* _a2[2] = { a2, GetMate(a2) };
	_i[0] = a1->sister->head; _i[1] = GetMate0(_i[0]);

	if (a1->head == a2->head)
	{
		a1->r_cap += a2->r_cap;
		a1->sister->r_cap += a2->sister->r_cap;
		_a1[1]->r_cap += _a2[1]->r_cap;
		_a1[1]->sister->r_cap += _a2[1]->sister->r_cap;
		x = 1;

		_j[0] = a1->head;
		_j[1] = GetMate(_j[0]);
	}
	else
	{
		code_assert(a1->head == GetMate(a2->head));
		// make sure that _a1[0]->r_cap == _a1[1]->r_cap and _a1[0]->sister->r_cap == _a1[1]->sister->r_cap by pushing flow
		delta = _a1[1]->r_cap - _a1[0]->r_cap;
		//_a1[1]->r_cap -= delta;   // don't do the subtraction - later we'll set explicitly _a1[1]->r_cap = _a1[0]->r_cap
		//_a1[1]->sister->r_cap += delta;
		_a1[1]->sister->head->tr_cap -= delta;
		_a1[1]->head->tr_cap         += delta;
		// same for a2
		delta = _a2[1]->r_cap - _a2[0]->r_cap;
		//_a2[1]->r_cap -= delta;   // don't do the subtraction - later we'll set explicitly _a2[1]->r_cap = _a2[0]->r_cap
		//_a2[1]->sister->r_cap += delta;
		_a2[1]->sister->head->tr_cap -= delta;
		_a2[1]->head->tr_cap         += delta;

		if (a1->r_cap + a1->sister->r_cap >= a2->r_cap + a2->sister->r_cap) x = 1;
		else // swap a1 <-> a2
		{
			Arc* tmp;
			tmp = a1; a1 = a2; a2 = tmp;
			_a1[0] = a1;
			_a2[0] = a2;
			tmp = _a1[1]; _a1[1] = _a2[1]; _a2[1] = tmp;
			x = 0;
		}

		_j[0] = a1->head;
		_j[1] = a2->head;

		REAL ci, cj, cij, cji;
		ci = a2->sister->r_cap - a2->r_cap;
		cj = 0;
		cij = -a2->r_cap;
		cji = -a2->sister->r_cap;

		_i[0]->tr_cap += ci; _i[1]->tr_cap -= ci;
		_j[0]->tr_cap += cj; _j[1]->tr_cap -= cj;
		_a1[0]->r_cap += cij;
		_a1[0]->sister->r_cap += cji;

		if (_a1[0]->r_cap < 0)
		{
			delta = _a1[0]->r_cap;
			_a1[0]->r_cap = 0;
			_a1[0]->sister->r_cap += delta;
			_i[0]->tr_cap -= delta; _i[1]->tr_cap += delta;
			_j[0]->tr_cap += delta; _j[1]->tr_cap -= delta;
		}
		if (_a1[0]->sister->r_cap < 0)
		{
			delta = _a1[0]->sister->r_cap;
			_a1[0]->sister->r_cap = 0;
			_a1[0]->r_cap += delta;
			_j[0]->tr_cap -= delta; _j[1]->tr_cap += delta;
			_i[0]->tr_cap += delta; _i[1]->tr_cap -= delta;
		}

		code_assert (_a1[0]->r_cap >= 0);

		_a1[1]->r_cap = _a1[0]->r_cap;
		_a1[1]->sister->r_cap = _a1[0]->sister->r_cap;
	}

	REMOVE_FROM(a2, _i[0]);
	REMOVE_FROM(a2->sister, a2->head);
	REMOVE_FROM(_a2[1], _a2[1]->sister->head);
	REMOVE_FROM(_a2[1]->sister, _i[1]);
	a2->sister->sister = NULL;
	a2->sister = NULL;
	_a2[1]->sister->sister = NULL;
	_a2[1]->sister = NULL;

	_a2[1]->next = first_free;
	first_free = _a2[1];

	return x;
}

template <typename REAL>
	inline REAL QPBO<REAL>::DetermineSaturation(Node* i)
{
	Arc* a;
	REAL c1 = -i->tr_cap;
	REAL c2 = i->tr_cap;
	for (a=i->first; a; a=a->next)
	{
		c1 += a->r_cap;
		c2 += a->sister->r_cap;
	}

	return (c1 > c2) ? c1 : c2;
}

template <typename REAL>
	inline void QPBO<REAL>::AddDirectedConstraint(Node* i, Node* j, int xi, int xj)
{
	code_assert(first_free && IsNode0(i) && IsNode0(j) && i!=j);

	int e = ((int)(first_free - arcs[IsArc0(first_free) ? 0 : 1])) & (~1);
	first_free = first_free->next;

	Arc* _a[2] = { &arcs[0][e], &arcs[1][e] };
	Node* _i[2] = { i, GetMate0(i) };
	Node* _j[2];
	if (xi==xj) { _j[0] = j; _j[1] = GetMate0(j); }
	else        { _j[1] = j; _j[0] = GetMate0(j); }

	SET_SISTERS(_a[0], _a[0]+1);
	SET_SISTERS(_a[1], _a[1]+1);

	SET_FROM(_a[0], _i[0]);
	SET_TO(_a[0], _j[0]);
	SET_FROM(_a[0]->sister, _j[0]);
	SET_TO(_a[0]->sister, _i[0]);
	SET_FROM(_a[1], _j[1]);
	SET_TO(_a[1], _i[1]);
	SET_FROM(_a[1]->sister, _i[1]);
	SET_TO(_a[1]->sister, _j[1]);

	if (xi==0) { _a[0]->r_cap = probe_options.C; _a[0]->sister->r_cap = 0; }
	else       { _a[0]->r_cap = 0; _a[0]->sister->r_cap = probe_options.C; }

	_a[1]->r_cap = _a[0]->r_cap;
	_a[1]->sister->r_cap = _a[0]->sister->r_cap;
}
/*
template <typename REAL>
	inline bool QPBO<REAL>::AddDirectedConstraint(Arc* a, int xi, int xj)
{
	Node* i = a->sister->head;
	Node* j = a->head;
	Node* _i[2] = { i, GetMate0(i) };
	Node* _j[2];
	Arc* _a[2] = { a, GetMate(a) };
	int x; // 0 if current edge is submodular, 1 otherwise
	REAL delta;

	_j[0] = j;
	if (IsNode0(j)) { _j[1] = GetMate0(j); x = 0; }
	else            { _j[1] = GetMate1(j); x = 1; }

	if ((xi + xj + x)%2 == 0)
	{
		// easy case - graph structure doesn't need to be changed
		if (xi == 0)
		{
			if (a->r_cap + _a[1]->r_cap >= 2*probe_options.C) return false;
			mark_node(_j[0]);
			mark_node(_j[1]);
			a->r_cap += 2*probe_options.C;
			_a[1]->r_cap += 2*probe_options.C;
			return true;
		}
		else
		{
			if (a->sister->r_cap + _a[1]->sister->r_cap >= 2*probe_options.C) return false;
			mark_node(_j[0]);
			mark_node(_j[1]);
			a->sister->r_cap += 2*probe_options.C;
			_a[1]->sister->r_cap += 2*probe_options.C;
			return true;
		}
	}

	mark_node(_j[0]);
	mark_node(_j[1]);

	// make sure that _a[0]->r_cap == _a[1]->r_cap and _a[0]->sister->r_cap == _a[1]->sister->r_cap by pushing flow
	delta = _a[1]->r_cap - _a[0]->r_cap;
	//_a[1]->r_cap -= delta;   // don't do the subtraction - later we'll set explicitly _a[1]->r_cap = _a[0]->r_cap
	//_a[1]->sister->r_cap += delta;
	_a[1]->sister->head->tr_cap -= delta;
	_a[1]->head->tr_cap         += delta;

	SET_TO(_a[0], _j[1]);
	SET_TO(_a[1]->sister, _j[0]);
	REMOVE_FROM(_a[0]->sister, _j[0]);
	SET_FROM(_a[0]->sister, _j[1]);
	REMOVE_FROM(_a[1], _j[1]);
	SET_FROM(_a[1], _j[0]);

	i->tr_cap += a->sister->r_cap - a->r_cap; _i[1]->tr_cap -= a->sister->r_cap - a->r_cap;
	a->r_cap = -a->r_cap;

	if (xi == 0) a->r_cap += 2*probe_options.C;
	else         a->sister->r_cap += 2*probe_options.C;

	if (a->r_cap < 0)
	{
		delta = a->r_cap;
		a->r_cap = 0;
		a->sister->r_cap += delta;
		i->tr_cap -= delta; _i[1]->tr_cap += delta;
		_j[1]->tr_cap += delta; j->tr_cap -= delta;
	}
	if (a->sister->r_cap < 0)
	{
		delta = a->sister->r_cap;
		a->sister->r_cap = 0;
		a->r_cap += delta;
		_j[1]->tr_cap -= delta; j->tr_cap += delta;
		i->tr_cap += delta; _i[1]->tr_cap -= delta;
	}

	_a[1]->r_cap = a->r_cap;
	_a[1]->sister->r_cap = a->sister->r_cap;

	return true;
}
*/
template <typename REAL>
	inline bool QPBO<REAL>::AddDirectedConstraint0(Arc* a, int xi, int xj)
{
	Node* i = a->sister->head;
	Node* j = a->head;
	Node* _i[2] = { i, GetMate0(i) };
	Node* _j[2];
	Arc* _a[2] = { a, GetMate(a) };
	int x; // 0 if current edge is submodular, 1 otherwise
	REAL delta;

	_j[0] = j;
	if (IsNode0(j)) { _j[1] = GetMate0(j); x = 0; }
	else            { _j[1] = GetMate1(j); x = 1; }

	if ((xi + xj + x)%2 == 0)
	{
		// easy case - graph structure doesn't need to be changed
		if (a->r_cap + a->sister->r_cap + _a[1]->r_cap + _a[1]->sister->r_cap >= 2*probe_options.C) return false;
		mark_node(_j[0]);
		mark_node(_j[1]);
		if (xi == 0)
		{
			a->r_cap += probe_options.C;
			_a[1]->r_cap += probe_options.C;
		}
		else
		{
			a->sister->r_cap += probe_options.C;
			_a[1]->sister->r_cap += probe_options.C;
		}
		return true;
	}

	mark_node(_j[0]);
	mark_node(_j[1]);

	// make sure that _a[0]->r_cap == _a[1]->r_cap and _a[0]->sister->r_cap == _a[1]->sister->r_cap by pushing flow
	delta = _a[1]->r_cap - _a[0]->r_cap;
	//_a[1]->r_cap -= delta;   // don't do the subtraction - later we'll set explicitly _a[1]->r_cap = _a[0]->r_cap
	//_a[1]->sister->r_cap += delta;
	_a[1]->sister->head->tr_cap -= delta;
	_a[1]->head->tr_cap         += delta;

	SET_TO(_a[0], _j[1]);
	SET_TO(_a[1]->sister, _j[0]);
	REMOVE_FROM(_a[0]->sister, _j[0]);
	SET_FROM(_a[0]->sister, _j[1]);
	REMOVE_FROM(_a[1], _j[1]);
	SET_FROM(_a[1], _j[0]);

	i->tr_cap += a->sister->r_cap - a->r_cap; _i[1]->tr_cap -= a->sister->r_cap - a->r_cap;
	a->r_cap = -a->r_cap;

	if (xi == 0) a->r_cap         += probe_options.C + a->sister->r_cap - a->r_cap;
	else         a->sister->r_cap += probe_options.C + a->sister->r_cap - a->r_cap;

	if (a->r_cap < 0)
	{
		delta = a->r_cap;
		a->r_cap = 0;
		a->sister->r_cap += delta;
		i->tr_cap -= delta; _i[1]->tr_cap += delta;
		_j[1]->tr_cap += delta; j->tr_cap -= delta;
	}
	if (a->sister->r_cap < 0)
	{
		delta = a->sister->r_cap;
		a->sister->r_cap = 0;
		a->r_cap += delta;
		_j[1]->tr_cap -= delta; j->tr_cap += delta;
		i->tr_cap += delta; _i[1]->tr_cap -= delta;
	}

	_a[1]->r_cap = a->r_cap;
	_a[1]->sister->r_cap = a->sister->r_cap;

	return true;
}

template <typename REAL>
	inline bool QPBO<REAL>::AddDirectedConstraint1(Arc* a, int xi, int xj)
{
	Node* j = a->head;
	Node* _j[2];
	Arc* _a[2] = { a, GetMate(a) };
	int x; GccUse(x);// 0 if current edge is submodular, 1 otherwise

	_j[0] = j;
	if (IsNode0(j)) { _j[1] = GetMate0(j); x = 0; }
	else            { _j[1] = GetMate1(j); x = 1; }

	code_assert((xi + xj + x)%2 == 0);

	if (xi == 0)
	{
		if (a->r_cap > 0 && _a[1]->r_cap > 0) return false;
		mark_node(_j[0]);
		mark_node(_j[1]);
		a->r_cap += probe_options.C;
		_a[1]->r_cap += probe_options.C;
		return true;
	}
	else
	{
		if (a->sister->r_cap > 0 && _a[1]->sister->r_cap > 0) return false;
		mark_node(_j[0]);
		mark_node(_j[1]);
		a->sister->r_cap += probe_options.C;
		_a[1]->sister->r_cap += probe_options.C;
		return true;
	}
}

template <typename REAL>
	void QPBO<REAL>::AllocateNewEnergy(int* mapping)
{
	int i_index, j_index;
	int nodeNumOld = GetNodeNum();
	int nodeNumNew = 1;
	int edgeNumOld = GetMaxEdgeNum();
	int e;
	Node* i;

	////////////////////////////////////////////////////////////////
	for (i_index=0, i=nodes[0]; i_index<nodeNumOld; i_index++, i++)
	{
		if (mapping[i_index] < 0)
		{
			mapping[i_index] = 2*nodeNumNew + ((i->user_label) % 2);
			nodeNumNew ++;
		}
		else if (mapping[i_index]>=2) mapping[i_index] = -mapping[i_index];
	}

	////////////////////////////////////////////////////////////////
	code_assert(nodes[0] + nodeNumNew <= node_max[0]);
	// Reset:
	node_last[0] = nodes[0];
	node_last[1] = nodes[1];
	node_num = 0;

	if (nodeptr_block)
	{
		delete nodeptr_block;
		nodeptr_block = NULL;
	}
	if (changed_list)
	{
		delete changed_list;
		changed_list = NULL;
	}
	if (fix_node_info_list)
	{
		delete fix_node_info_list;
		fix_node_info_list = NULL;
	}

	maxflow_iteration = 0;
	zero_energy = 0;

	stage = 0;
	all_edges_submodular = true;
	////////////////////////////////////////////////////////////////


	AddNode(nodeNumNew);
	AddUnaryTerm(0, (REAL)0, (REAL)1);

	i = nodes[0];
	i->user_label = i->label = 0;
	for (i_index=0; i_index<nodeNumOld; i_index++)
	{
		if (mapping[i_index] >= 2)
		{
			i = nodes[0] + (mapping[i_index]/2);
			i->user_label = i->label = mapping[i_index] & 1;
			mapping[i_index] &= ~1;
		}
	}
	////////////////////////////////////////////////////////////////
	for (i_index=0; i_index<nodeNumOld; i_index++)
	if (mapping[i_index] < 0)
	{
		int y[2];
		int x = 0;
		j_index = i_index;
		do
		{
			x = (x - mapping[j_index]) % 2;
			j_index = (-mapping[j_index])/2 - 1;
		} while (mapping[j_index] < 0);
		y[x] = mapping[j_index];
		y[1-x] = mapping[j_index] ^ 1;

		x = 0;
		j_index = i_index;
		do
		{
			int m_old = mapping[j_index];
			mapping[j_index] = y[x];
			x = (x - m_old) % 2;
			j_index = (-m_old)/2 - 1;
		} while (mapping[j_index] < 0);
	}

	////////////////////////////////////////////////////////////////
	int edgeNumNew = 0;
	for (e=0; e<edgeNumOld; e++)
	{
		if ( arcs[0][2*e].sister )
		{
			Arc* a;
			Arc* a_mate;
			if (IsNode0(arcs[0][2*e].sister->head))
			{
				a = &arcs[0][2*e];
				a_mate = &arcs[1][2*e];
			}
			else
			{
				a = &arcs[1][2*e+1];
				a_mate = &arcs[0][2*e+1];
			}
			i_index = mapping[(int)(a->sister->head - nodes[0])] / 2;
			code_assert(i_index > 0 && i_index < nodeNumNew);

			first_free = &arcs[0][2*edgeNumNew++];
			if (IsNode0(a->head))
			{
				j_index = mapping[(int)(a->head - nodes[0])] / 2;
				code_assert(j_index > 0 && j_index < nodeNumNew);
				AddPairwiseTerm(i_index, j_index,
					0, a->r_cap+a_mate->r_cap, a->sister->r_cap+a_mate->sister->r_cap, 0);
			}
			else
			{
				j_index = mapping[(int)(a->head - nodes[1])] / 2;
				code_assert(j_index > 0 && j_index < nodeNumNew);
				AddPairwiseTerm(i_index, j_index,
					a->r_cap+a_mate->r_cap, 0, 0, a->sister->r_cap+a_mate->sister->r_cap);
			}
		}
	}

	first_free = &arcs[0][2*edgeNumNew];
	memset(first_free, 0, (int)((char*)arc_max[0] - (char*)first_free));
	InitFreeList();
}

const int LIST_NUM = 4;
struct List // contains LIST_NUM lists containing integers 0,1,...,num-1. In the beginning, each integer is in list 1.
{
	List(int _num, int* order)
		: num(_num)
	{
		int i;
		prev = new int[num+LIST_NUM]; prev += LIST_NUM;
		next = new int[num+LIST_NUM]; next += LIST_NUM;
		if (order)
		{
			for (i=0; i<num; i++)
			{
				prev[order[i]] = (i==0)     ? -1 : order[i-1];
				next[order[i]] = (i==num-1) ? -1 : order[i+1];
			}
			prev[-1] = order[num-1];
			next[-1] = order[0];
		}
		else
		{
			for (i=0; i<num; i++)
			{
				prev[i] = i-1;
				next[i] = i+1;
			}
			prev[-1] = num-1;
			next[-1] = 0;
			next[num-1] = -1;
		}
		for (i=2; i<=LIST_NUM; i++)
		{
			prev[-i] = -i;
			next[-i] = -i;
		}
	}

	~List()
	{
		delete [] (prev - LIST_NUM);
		delete [] (next - LIST_NUM);
	}
	// i must be in the list
	void Remove(int i)
	{
		next[prev[i]] = next[i];
		prev[next[i]] = prev[i];
	}
	void Move(int i, int r_to) // moves node i to list r_to
	{
		code_assert (r_to>=1 && r_to<=LIST_NUM);
		Remove(i);
		next[i] = -r_to;
		prev[i] = prev[-r_to];
		next[prev[-r_to]] = i;
		prev[-r_to] = i;
	}
	void MoveList(int r_from, int r_to) // append list r_from to list r_to. List r_from becomes empty.
	{
		code_assert (r_from>=1 && r_from<=LIST_NUM && r_to>=1 && r_to<=LIST_NUM && r_from!=r_to);
		if (next[-r_from] < 0) return; // list r_from is empty
		prev[next[-r_from]] = prev[-r_to];
		next[prev[-r_to]] = next[-r_from];
		prev[-r_to] = prev[-r_from];
		next[prev[-r_from]] = -r_to;
		prev[-r_from] = next[-r_from] = -r_from;
	}
	// i must be in the list
	// (or -r, in which case the first element of list r is returned).
	// Returns -1 if no more elements.
	int GetNext(int i) { return next[i]; }

private:
	int num;
	int* next;
	int* prev;
};

template <typename REAL>
	void QPBO<REAL>::SetMaxEdgeNum(int num)
{
	if (num > GetMaxEdgeNum()) reallocate_arcs(2*num);
}

template <typename REAL>
	bool QPBO<REAL>::Probe(int* mapping)
{
	int i_index, i_index_next, j_index;
	Node* i;
	Node* j;
	Node* _i[2];
	Arc* a;
	Arc* a_next;
	int x;
	Node** ptr;
	bool is_enough_memory = true;
	int unlabeled_num;

	if (probe_options.order_array) probe_options.order_seed = 0;
	if (probe_options.order_seed != 0)
	{
		srand(probe_options.order_seed);
		probe_options.order_array = new int[GetNodeNum()];
		ComputeRandomPermutation(GetNodeNum(), probe_options.order_array);
	}
	List list(GetNodeNum(), probe_options.order_array);
	if (probe_options.order_seed != 0)
	{
		delete [] probe_options.order_array;
	}

	if (node_last[0] == node_max[0]) reallocate_nodes((int)(node_last[0] - nodes[0]) + 1);

	all_edges_submodular = false;
	Solve();

	int MASK_CURRENT = 1;
	int MASK_NEXT = 2;

	unlabeled_num = 0;
	for (i=nodes[0], i_index=0; i<node_last[0]; i++, i_index++)
	{
		if (i->label >= 0)
		{
			FixNode(i, i->label);
			mapping[i_index] = i->label;
			i->is_removed = 1;
			list.Remove(i_index);
		}
		else
		{
			i->is_removed = 0;
			i->list_flag = MASK_CURRENT;
			mapping[i_index] = -1;
			unlabeled_num ++;
		}
		i->label_after_fix0 = -1;
		i->label_after_fix1 = -1;
	}

	maxflow();

	// INVARIANTS:
	//    node i_index is removed <=> mapping[i_index] >= 0 <=> nodes[0][i_index].is_removed == 1
	//    edge e is removed <=> Arc::sister does not point to the correct arc for at least one out of the 4 arcs


	// Four lists are used: 1,2,3,4. In the beginning of each iteration
	// lists 3 and 4 are empty. current_list is set to the lowest non-empty list (1 or 2).
	// After the iteration current_list becomes empty, and its former nodes are moved
	// either to list 3 (if the node or its neighbor has changed) or to list 4.
	//
	// Invariants during the iteration:
	//   - i->list_flag == MASK_CURRENT              => i is in current_list
	//   - i->list_flag == MASK_CURRENT & MASK_NEXT  => i is in current_list, after processing should be moved to list 3
	//   - i->list_flag ==                MASK_NEXT  => i is in list 3
	//   - i->list_flag == 0                         => i is not in current_list and not in list 3
	//
	// After the iteration, list 3 is eroded probe_dilation-1 times (it acquired nodes from lists 2 and 4).
	// It is then moved to list 1 (which is now empty) and list 4 is added to list 2.

	while ( 1 )
	{
		bool success = true;

		// Try fixing nodes to 0 and 1
		int current_list = (list.GetNext(-1) >= 0) ? 1 : 2;
		for (i_index=list.GetNext(-current_list) ; i_index>=0; i_index=i_index_next)
		{
			i_index_next = list.GetNext(i_index);
			i = nodes[0]+i_index;
			code_assert (!i->is_removed && i->label<0);

			_i[0] = i;
			_i[1] = GetMate0(i);
			bool is_changed = false;

			REAL INFTY0 = DetermineSaturation(i);
			REAL INFTY1 = DetermineSaturation(_i[1]);
			REAL INFTY = ((INFTY0 > INFTY1) ? INFTY0 : INFTY1) + 1;


			// fix to 0, run maxflow
			mark_node(i);
			mark_node(_i[1]);
			AddUnaryTerm(i, 0, INFTY);
			maxflow(true, true);
			if (what_segment(i)!=0 || what_segment(_i[1])!=1)
			{
				printf("Error in Probe()! Perhaps, overflow due to large capacities?\n");
			}
			for (ptr=changed_list->ScanFirst(); ptr; ptr=changed_list->ScanNext())
			{
				j = *ptr;
				code_assert(!j->is_removed && j->label<0);
				j->label_after_fix0 = what_segment(j);
				if (j->label_after_fix0 == what_segment(GetMate0(j))) j->label_after_fix0 = -1;
				else if (i->user_label == 0) j->user_label = j->label_after_fix0;
			}

			// fix to 1, run maxflow
			mark_node(i);
			mark_node(_i[1]);
			AddUnaryTerm(i, 0, -2*INFTY);
			maxflow(true, true);
			if (what_segment(i)!=1 || what_segment(_i[1])!=0)
			{
				printf("Error in Probe()! Perhaps, overflow due to large capacities?\n");
			}
			mark_node(i);
			mark_node(_i[1]);
			AddUnaryTerm(i, 0, INFTY);

			// go through changed pixels
			bool need_to_merge = false;

			for (ptr=changed_list->ScanFirst(); ptr; ptr=changed_list->ScanNext())
			{
				j = *ptr;
				code_assert(!j->is_removed && j->label<0);
				j->label_after_fix1 = what_segment(j);
				if (j->label_after_fix1 == what_segment(GetMate0(j))) j->label_after_fix1 = -1;
				else if (i->user_label == 1) j->user_label = j->label_after_fix1;

				if (i == j || j->label_after_fix0 < 0 || j->label_after_fix1 < 0) continue;

				j_index = (int)(j - nodes[0]);

				is_changed = true;
				if (j->label_after_fix0 == j->label_after_fix1)
				{
					// fix j
					FixNode(j, j->label_after_fix0);
					mapping[j_index] = j->label_after_fix0;
				}
				else
				{
					// contract i and j
					ContractNodes(i, j, j->label_after_fix0);
					mapping[j_index] = 2*i_index + 2 + j->label_after_fix0;
					need_to_merge = true;
				}
				j->is_removed = 1;
				if (i_index_next == j_index) i_index_next = list.GetNext(j_index);
				list.Remove(j_index);
				unlabeled_num --;
			}

			if (need_to_merge)
			{
				// merge parallel edges incident to i
				for (a=i->first; a; a=a->next) // mark neighbor nodes
				{
					j_index = (int)(a->head - nodes[IsNode0(a->head) ? 0 : 1]);
					mapping[j_index] = (int)(a - arcs[0]);
				}
				for (a=i->first; a; a=a_next)
				{
					a_next = a->next;
					j_index = (int)(a->head - nodes[IsNode0(a->head) ? 0 : 1]);
					Arc* a2 = &arcs[0][mapping[j_index]];
					if (a2 == a) continue;
					mark_node(a->head);
					mark_node(GetMate(a->head));
					if (MergeParallelEdges(a2, a)==0)
					{
						mapping[j_index] = (int)(a - arcs[0]);
						a_next = a->next;
					}
				}
				for (a=i->first; a; a=a->next)
				{
					j_index = (int)(a->head - nodes[IsNode0(a->head) ? 0 : 1]);
					mapping[j_index] = -1;
				}
			}

			// add directed links for neighbors of i
			for (a=i->first; a; a=a->next)
			{
				j = a->head;
				int label[2];
				if (IsNode0(j))
				{
					label[0] = j->label_after_fix0;
					label[1] = j->label_after_fix1;
				}
				else
				{
					label[0] = GetMate1(j)->label_after_fix0;
					label[1] = GetMate1(j)->label_after_fix1;
				}
				for (x=0; x<2; x++)
				{
					if (label[x]>=0 && label[1-x]<0)
					{
						if (AddDirectedConstraint0(a, x, label[x])) is_changed = true;
					}
				}
			}
			maxflow(true, true);
			mark_node(i);
			mark_node(_i[1]);
			for (a=i->first; a; a=a->next)
			{
				j = a->head;
				int label[2];
				if (IsNode0(j))
				{
					label[0] = j->label_after_fix0; j->label_after_fix0 = -1;
					label[1] = j->label_after_fix1; j->label_after_fix1 = -1;
				}
				else
				{
					label[0] = GetMate1(j)->label_after_fix0; GetMate1(j)->label_after_fix0 = -1;
					label[1] = GetMate1(j)->label_after_fix1; GetMate1(j)->label_after_fix1 = -1;
				}
				for (x=0; x<2; x++)
				{
					if (label[x]>=0 && label[1-x]<0)
					{
						if (AddDirectedConstraint1(a, x, label[x])) is_changed = true;
					}
				}
			}

			// add directed constraints for nodes which are not neighbors
			if (probe_options.directed_constraints!=0)
			{
				for (ptr=changed_list->ScanFirst(); ptr; ptr=changed_list->ScanNext())
				{
					j = *ptr;
					int x, y;

					if (j->is_removed) continue;
					if (i == j) continue;
					if      (j->label_after_fix0 >= 0 && j->label_after_fix1 < 0) { x = 0; y = j->label_after_fix0; }
					else if (j->label_after_fix1 >= 0 && j->label_after_fix0 < 0) { x = 1; y = j->label_after_fix1; }
					else continue;

					if (first_free)
					{
						AddDirectedConstraint(i, j, x, y);
						is_changed = true;
					}
					else
					{
						is_enough_memory = false;
						break;
					}
				}
			}

			if (probe_options.dilation >= 0)
			{
				// if is_changed, add i and its neighbors to list 3, otherwise add i to list 2 (unless it's already there)
				if (is_changed)
				{
					i->list_flag |= MASK_NEXT;
					if (probe_options.dilation >= 1)
					{
						for (a=i->first; a; a=a->next)
						{
							j = a->head;
							if (!IsNode0(j)) j = GetMate1(j);
							if (!(j->list_flag & MASK_NEXT))
							{
								j->list_flag |= MASK_NEXT;
								if (!(j->list_flag & MASK_CURRENT))
								{
									j_index = (int)(j-nodes[0]);
									code_assert(j_index != i_index_next);
									list.Move(j_index, 3);
								}
							}
						}
					}
				}
				code_assert (i->list_flag & MASK_CURRENT);
				i->list_flag &= ~MASK_CURRENT;
				list.Move(i_index, (i->list_flag & MASK_NEXT) ? 3 : 4);
			}

			if (is_changed) success = false;

			// after fixes and contractions run maxflow, check whether more nodes have become labeled
			maxflow(true, true);

			for (ptr=changed_list->ScanFirst(); ptr; ptr=changed_list->ScanNext())
			{
				j = *ptr;

				j->is_in_changed_list = 0;
				j->label_after_fix0 = -1;
				j->label_after_fix1 = -1;

				if (j->is_removed) continue;

				j->label = what_segment(j);
				if (j->label == what_segment(GetMate0(j))) j->label = -1;

				if (j->label >= 0)
				{
					j_index = (int)(j - nodes[0]);
					FixNode(j, j->label);
					mapping[j_index] = j->label;
					j->is_removed = 1;
					if (i_index_next == j_index) i_index_next = list.GetNext(j_index);
					list.Remove(j_index);
					unlabeled_num --;
				}
			}
			changed_list->Reset();

			if (probe_options.callback_fn)
			{
				if (probe_options.callback_fn(unlabeled_num))
				{
					user_terminated = true;
					AllocateNewEnergy(mapping);
					return is_enough_memory;
				}
			}
		}

		if (probe_options.dilation < 0)
		{
			if (success) break;
			else         continue;
		}

		if (current_list == 2 && success) break;

		code_assert(list.GetNext(-1) == -1);
		list.MoveList(4, 2);
		if (list.GetNext(-3) < 0)
		{
			// list 3 is empty
			for (i_index=list.GetNext(-2); i_index>=0; i_index=list.GetNext(i_index))
			{
				i = nodes[0]+i_index;
				i->list_flag = MASK_CURRENT;
			}
		}
		else
		{
			int MASK_TMP = MASK_CURRENT; MASK_CURRENT = MASK_NEXT; MASK_NEXT = MASK_TMP;
			int r;
			for (r=1; r<probe_options.dilation; r++)
			{
				for (i_index=list.GetNext(-3); i_index>=0; i_index=list.GetNext(i_index))
				{
					i = nodes[0]+i_index;
					for (a=i->first; a; a=a->next)
					{
						j = a->head;
						if (!IsNode0(j)) j = GetMate1(j);
						if (!(j->list_flag & MASK_CURRENT))
						{
							j->list_flag = MASK_CURRENT;
							j_index = (int)(j-nodes[0]);
							code_assert(j_index != i_index_next);
							list.Move(j_index, 4);
						}
					}
				}
				list.MoveList(3, 1);
				list.MoveList(4, 3);
			}
			list.MoveList(3, 1);
		}
	}

	// almost done
	AllocateNewEnergy(mapping);
	Solve();

	return is_enough_memory;
}


template <typename REAL>
	void QPBO<REAL>::Probe(int* mapping, ProbeOptions& options)
{
	int nodeNum0 = GetNodeNum();
	bool is_enough_memory;
	user_terminated = false;

	memcpy(&probe_options, &options, sizeof(ProbeOptions));

	is_enough_memory = Probe(mapping);

	while ( 1 )
	{
		if (user_terminated) break;

		bool success = true;
		if ( probe_options.weak_persistencies )
		{
			int i;
			ComputeWeakPersistencies();
			for (i=1; i<GetNodeNum(); i++)
			{
				int ki = GetLabel(i);
				if (ki >= 0)
				{
					AddUnaryTerm(i, 0, (REAL)(1-2*ki));
					success = false;
				}
			}
		}

		if (probe_options.directed_constraints == 2 && !is_enough_memory)
		{
			SetMaxEdgeNum(GetMaxEdgeNum() + GetMaxEdgeNum()/2 + 1);
		}
		else
		{
			if (success) break;
		}

		int* mapping1 = new int[GetNodeNum()];
		is_enough_memory = Probe(mapping1);
		MergeMappings(nodeNum0, mapping, mapping1);
		delete [] mapping1;
	}
}

template <typename REAL>
	bool QPBO<REAL>::Improve(int N, int* order_array, int* fixed_nodes)
{
	int p, i_index;
	Node* i;
	Node* _i[2];
	FixNodeInfo* ptr;

	if (!fix_node_info_list)
	{
		fix_node_info_list = new Block<FixNodeInfo>(128);
	}

	maxflow();
	if (stage == 0)
	{
		TransformToSecondStage(true);
		maxflow(true);
	}

	for (p=0; p<N; p++)
	{
		i_index = order_array[p];
		code_assert(i_index>=0 && i_index<GetNodeNum());

		i = _i[0] = &nodes[0][i_index];
		_i[1] = &nodes[1][i_index];

		i->label = what_segment(i);
		if (i->label != what_segment(GetMate0(i)))
		{
			continue;
		}

		REAL INFTY0 = DetermineSaturation(i);
		REAL INFTY1 = DetermineSaturation(_i[1]);
		REAL INFTY = ((INFTY0 > INFTY1) ? INFTY0 : INFTY1) + 1;

		if (i->user_label == 1) INFTY = -INFTY;

		ptr = fix_node_info_list->New();
		ptr->i = i;
		ptr->INFTY = INFTY;

		mark_node(i);
		mark_node(GetMate0(i));
		AddUnaryTerm(i, 0, INFTY);
		maxflow(true);
	}

	if (fixed_nodes) memset(fixed_nodes, 0, GetNodeNum()*sizeof(int));
	for (ptr=fix_node_info_list->ScanFirst(); ptr; ptr=fix_node_info_list->ScanNext())
	{
		AddUnaryTerm(ptr->i, 0, -ptr->INFTY);
		if (fixed_nodes) fixed_nodes[(int)(ptr->i - nodes[0])] = 1;
	}
	fix_node_info_list->Reset();

	bool success = false;
	for (i=nodes[0]; i<node_last[0]; i++)
	{
		i->label = what_segment(i);
		if (i->label == what_segment(GetMate0(i))) i->label = i->user_label;
		else if (i->label != (int)i->user_label)
		{
			success = true;
			i->user_label = (unsigned int)i->label;
		}
	}

	return success;
}

template <typename REAL>
	bool QPBO<REAL>::Improve()
{
	int* permutation = new int[node_num];
	ComputeRandomPermutation(node_num, permutation);
	bool success = Improve(node_num, permutation);
	delete [] permutation;
	return success;
}

#include "instances.inc"
