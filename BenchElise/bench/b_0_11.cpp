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


//*******************************************



void bench_1_pt2di(INT x, INT y,INT v)
{
    INT tx = 30;
    INT ty = 40;

    Im2D_U_INT1 i1 (tx,ty,0);
    Im2D_U_INT1 i2 (tx,ty,0);

    ELISE_COPY(Pt2di(x,y),v,i1.out());
    i2.data()[y][x] = v;

    INT nb_dif;
    ELISE_COPY(i1.all_pts(),i1.in()!=i2.in(),sigma(nb_dif));

    BENCH_ASSERT(nb_dif==0);
    
}
void bench_1_pt2di()
{
    bench_1_pt2di(1,2,3);

     for(INT x = 3; x < 12; x++)
        for(INT y = 13; y < 22; y++)
           bench_1_pt2di(x,y,x+y);
}

void bench_flux_pts_user()
{
      bench_1_pt2di();
}
