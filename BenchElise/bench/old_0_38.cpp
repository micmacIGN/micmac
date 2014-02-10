/*eLiSe06/05/99
  
     Copyright (C) 1999 Marc PIERROT DESEILLIGNY

   eLiSe : Elements of a Linux Image Software Environment

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  Author: Marc PIERROT DESEILLIGNY    IGN/MATIS  
Internet: Marc.Pierrot-Deseilligny@ign.fr
   Phone: (33) 01 43 98 81 28
eLiSe06/05/99*/



#include "general/all.h"
#include "ext_stl/fifo.h"
#include "bench.h"


/*
static Pt3di P3r()
{
    return Pt3di
           (
               (INT)(NRrandom3()  * 1000),
               (INT)(NRrandom3()  * 1000),
               (INT)(NRrandom3()  * 1000)
           );
}

void bench_Heap_Pt3di()
{
     ElCmpZ  cmpz;

     ElHeap<Pt3di , ElCmpZ > h(Pt3di(0,0,-1000000000),cmpz);

     INT NB = 20;


     for (INT x = 0 ; x < NB ; x++)
     {
        Pt3di p = P3r();
        h.push(p);
     }

     Pt3di plast = h.top();

     while(! h.empty())
     {
         Pt3di p = h.top();
         BENCH_ASSERT(plast.z<=p.z);
         plast = p;
         h.pop();
         NB--;
     }

     BENCH_ASSERT(NB==0);
}
*/

void bench_InetgerHeap_Pt3di()
{
    INT NB = 6000;
    ElIntegerHeap<Pt3di> heap(10);

     for (INT x = 0 ; x < NB ; x++)
     {
        Pt3di p = P3r()-Pt3dr(500,500,500);
        heap.push(p,p.z);
     }

     INT ind;
     Pt3di plast = heap.top(ind);

     while(! heap.empty())
     {

         Pt3di p = heap.top(ind);
         BENCH_ASSERT(plast.z<=p.z);
         BENCH_ASSERT(ind==p.z);
         plast = p;
         heap.pop();
         NB--;
     }

     BENCH_ASSERT(NB==0);
}

void bench_ElBornedIntegerHeap()
{
    ElBornedIntegerHeap<Pt3di> heap(3000);

    heap.push(Pt3di(0,0,0),0);
}

void bench_stl1()
{
     bench_InetgerHeap_Pt3di();
     // bench_Heap_Pt3di();

     cout << "OK BENCH HEAP GEO \n";
}





