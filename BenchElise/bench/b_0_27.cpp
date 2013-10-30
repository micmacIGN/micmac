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



Fonc_Num integr_rand(Fonc_Num pot)
{
    Im1D_INT4 pond(257,0);
    Symb_FNum g (bobs_grad(pot));
    ELISE_COPY
    (
       pond.all_pts(),
       2+20*frandr(),
       pond.out()
    );
    return integr_grad(Virgule(pot,g.v0(),g.v1()),-128,pond);
}

void bench_integr_grad
     (
         Pt2di p1,
         Pt2di p2,
         Fonc_Num pot
     )
{
    INT dif;
    ELISE_COPY
    (
       rectangle(p1,p2),
       Abs(pot-integr_rand(pot)),
       VMax(dif)
    );

    cout << "DIF = " << dif << "\n";

    BENCH_ASSERT(dif==0);
}


void bench_integr_grad()
{
     bench_integr_grad
     (
         Pt2di(0,0),
         Pt2di(100,100),
         Iconv(10*sin(FX)+13*sin(2*FY) + 4*sin(FX*FY))
     );
     
}

void bench_algo_spec()
{
    bench_integr_grad();
    printf("END bench_algo_spec \n");
}
