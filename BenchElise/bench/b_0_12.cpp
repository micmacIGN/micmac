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



Fonc_Num fns_mul_pow_2(Fonc_Num f,INT n)
{
    while (n--> 0)
    {
        Symb_FNum sf (f);
        f = sf+sf;
    }
    return f;
}

Fonc_Num bov_mul_pow_2(Fonc_Num f,INT n)
{
    while (n--> 0)
        f = f+f;
    return f;
}

Fonc_Num quick_mul_pow_2(Fonc_Num f,INT n)
{
    return f * (1<<n);
}

//****************************************************

void bench_exh_fnum_symb(Fonc_Num f,INT n,bool show)
{
    INT dif1,dif2;

    if (show)
        cout << "deb fns \n";
    ELISE_COPY
    (
         rectangle(Pt2di(-100,-100),Pt2di(120,120)),
         Abs(quick_mul_pow_2(f,n)-fns_mul_pow_2(f,n)),
         sigma(dif1)
    );

    if (show)
        cout << "deb bov \n";
    ELISE_COPY
    (
         rectangle(Pt2di(-100,-100),Pt2di(120,120)),
         Abs(quick_mul_pow_2(f,n)-bov_mul_pow_2(f,n)),
         sigma(dif2)
    );

    if (show)
        cout << "fin bov \n";

    BENCH_ASSERT((dif1==0) && (dif2 == 0));
}



//****************************************************

void bench_fnum_symb()
{
     // bench_exh_fnum_symb(FX+FY,7,true);
     bench_exh_fnum_symb(FX+FY,4,false);
     cout << "OK bench_fnum_symb \n";
}





