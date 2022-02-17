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



void bench_sort(INT n)
{
     Im1D_REAL8 b(n);

     ELISE_COPY(b.all_pts(),frandr(),b.out());
     elise_sort(b.data(),n);

     INT wrong;
     ELISE_COPY(rectangle(0,n-1),b.in()>b.in()[FX+1],sigma(wrong));

    if (wrong)
    {
        cout << "n = " << n << "\n";
        for (INT i =0; i < n ; i++)
            cout << b.data()[i] << " ";
        cout << "\n";
     }
     BENCH_ASSERT(wrong == 0);


     ELISE_COPY(b.all_pts(),frandr(),b.out());
     Im1D_INT4 I(n);
     elise_indexe_sort(b.data(),I.data(),n);

     ELISE_COPY
     (
           rectangle(0,n-1),
               b.in()[I.in()]
          >    b.in()[I.in()[FX+1]],
          sigma(wrong)
     );
     BENCH_ASSERT(wrong == 0);
}


void bench_ecart_circ(REAL v1,REAL v2,INT nb)
{
    Im1D_REAL8  pds(nb+1);
    ELISE_COPY(pds.all_pts(),frandr(),pds.out());
    Im1D_INT4   ind(nb+1);
    elise_indexe_sort(pds.data(),ind.data(),nb+1);

    INT *  i = ind.data();
    REAL step = (v2-v1)/nb;

    Fonc_Num f = FX+ i[0]*step; 
    for (INT k =0; k<= nb ; k++)
        f = Virgule(FX+ i[k]*step,f);

    REAL dif;
    ELISE_COPY
    (
         rectangle(0,100),
         Abs(ecart_circ(f)-(v2-v1)),
         VMax(dif)
    );
//    cout << "DIFF = " << dif << "; nb = " << nb  << "\n";
    BENCH_ASSERT(dif<epsilon);
}


void bench_sort()
{
     bench_ecart_circ(0.5,1.5,3);
     bench_ecart_circ(-1,1.0,10);
     bench_ecart_circ(-3,3,100);

     bench_ecart_circ(0.5,1.5,3);
     bench_ecart_circ(-1,1.0,10);
     bench_ecart_circ(-3,3,100);

     for (INT i = 0; i < 200 ; i++)
         for (INT j=1 ; j< 7 ; j++)
             bench_sort(j);

	{
		 for (INT i = 0; i < 5 ; i++)
			 for (INT j=1 ; j< 100 ; j+=3)
				 bench_sort(j);
	}

     bench_ecart_circ(0.5,1.5,3);
     bench_ecart_circ(-1,1.0,10);
     bench_ecart_circ(-3,3,100);

      printf("OK sort \n");
}








