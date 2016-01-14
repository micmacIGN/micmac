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



#include "StdAfx.h"
#include "bench.h"



void test_card_rect_kd(int * p0,int *p1,int dim)
{
   INT s1,s2;

   ELISE_COPY(rectangle(p0,p1,dim),1,sigma(s1));

   s2 = 1;
   for(INT i=0 ; i < dim ; i++)
      s2 *= ElAbs(p0[i]-p1[i]);

   BENCH_ASSERT(s1 == s2);
}

void test_card_rect_kd_coord(int * p0,int *p1,int dim,int kth)
{
   INT s1,s2;

   ELISE_COPY(rectangle(p0,p1,dim),kth_coord(kth),sigma(s1));

   s2 = 1;
   for(INT d=0 ; d < dim ; d++)
      s2 *=  (d == kth)         ?
             som_x(p0[d],p1[d]) :
             ElAbs(p0[d]-p1[d])   ;

   BENCH_ASSERT(s1 == s2);
}

void test_card_rect_kd_coord(int * p0,int *p1,int dim)
{
    for (int kth = 0; kth<dim ; kth++)
        test_card_rect_kd_coord(p0,p1,dim,kth);
}



void test_card_rect_kd()
{
    {
         int p0[3] ={0,0,0};
         int p1[3] ={2,3,3};
         test_card_rect_kd(p0,p1,3);
         test_card_rect_kd(p1,p0,3);
         test_card_rect_kd_coord(p1,p0,3);

    }
    {
         int p0[5] ={2,3,4,3,2};
         int p1[5] ={6,1,8,7,5};
         test_card_rect_kd(p0,p1,5);
         test_card_rect_kd(p1,p0,5);

         test_card_rect_kd_coord(p1,p0,5);
    }
    {
         int p0[1] ={22};
         int p1[1] ={145};
         test_card_rect_kd(p0,p1,1);
         test_card_rect_kd(p1,p0,1);
         test_card_rect_kd_coord(p1,p0,1);
    }

}


void test_rect_kd()
{
     test_card_rect_kd();

     cout << "OK RECT KD \n";
}





