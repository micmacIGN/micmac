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

Fonc_Num ind_rect_2d(Pt2di c1,Pt2di c2)
{
   return 
         (
                       (FX >= c1.x)
                   &&  (FX <  c2.x)
                   &&  (FY >= c1.y)
                   &&  (FY <  c2.y)
         );
}

void bench_border_rect_2D
    (   
        Im2D_U_INT1  I,
        Pt2di        p1,
        Pt2di        p2,
        Pt2di        q1,
        Pt2di        q2
    )
{

    ELISE_COPY(I.all_pts(),0,I.out());
    ELISE_COPY(border_rect(p1,p2,q1,q2),1,I.out());
    Fonc_Num f =  ind_rect_2d(p1,p2) && (! ind_rect_2d(p1+q1,p2-q2));
                 
    INT sdif;
    ELISE_COPY(I.all_pts(),Abs(f-I.in()),sigma(sdif));

    BENCH_ASSERT(sdif== 0);
}



void bench_border_rect_2D()
{
     Im2D_U_INT1 I(200,200);

     bench_border_rect_2D(I,Pt2di(15,28),Pt2di(158,187),Pt2di(2,30),Pt2di(8,6));
     bench_border_rect_2D(I,Pt2di(15,28),Pt2di(158,187),Pt2di(1,2),Pt2di(3,4));
}

//*************************************************

Fonc_Num ind_rect_1d(INT c1,INT c2)
{
   return ((FX >= c1) && (FX < c2));
}

void bench_border_rect_1D
    (   
        Im1D_U_INT1  I,
        INT        x1,
        INT        x2,
        INT        b1,
        INT        b2
    )
{

    ELISE_COPY(I.all_pts(),0,I.out());
    ELISE_COPY(border_rect(x1,x2,b1,b2),1,I.out());
    Fonc_Num f =  ind_rect_1d(x1,x2) && (! ind_rect_1d(x1+b1,x2-b2));
                 
    INT sdif;
    ELISE_COPY(I.all_pts(),Abs(f-I.in()),sigma(sdif));


    cout << "border_rect_1D, sdif : " << sdif << "\n";
    BENCH_ASSERT(sdif== 0);
}

void bench_border_rect_1D()
{
    Im1D_U_INT1 I (10000);

   bench_border_rect_1D(I,20,9876,34,78);
   bench_border_rect_1D(I,20,9876,1,1);
}


//*******************************************

Fonc_Num ind_rect_Kd(INT * p1,INT * p2,INT dim)
{
   Fonc_Num f =  ((kth_coord(0) >= p1[0]) && (kth_coord(0) < p2[0]));

   for (int i = 1; i < dim ; i++)
         f =  f && ((kth_coord(i) >= p1[i]) && (kth_coord(i) < p2[i]));
   return f;
}

void bench_border_rect_KD
    (   
        Fonc_Num  check_sum,

        INT *   R1,
        INT *   R2,

        INT *   p1,
        INT *   p2,
        INT *   b1,
        INT *   b2,
        INT     dim
    )
{
   INT q1 [Elise_Std_Max_Dim];
   INT q2 [Elise_Std_Max_Dim];

   for (int i=0; i<dim ; i++)
   {
        q1[i] = p1[i] + b1[i];
        q2[i] = p2[i] - b2[i];
   }

   INT s1,s2;
   ELISE_COPY(border_rect(p1,p2,b1,b2,dim),check_sum,sigma(s1));

   ELISE_COPY
   (  select
      (  rectangle(R1,R2,dim),
         ind_rect_Kd(p1,p2,dim) && (! ind_rect_Kd(q1,q2,dim))
      ),
      check_sum,
      sigma(s2)
   );

    BENCH_ASSERT(s1== s2);
}

void bench_border_rect_KD()
{
    {
         INT R1[4] = {-2,-2,-2,-2};
         INT R2[4] = {12,12,12,12};
         INT p1[4] = {-1,0,0,-1};
         INT p2[4] = {11,10,11,10};

         INT b1[4] = {1,2,3,4};
         INT b2[4] = {4,3,2,1};

         Fonc_Num FT = kth_coord(3);

         bench_border_rect_KD
         (
            FX,
            R1,R2,p1,p2,b1,b2,1
         );

         bench_border_rect_KD
         (
            FX+FY,
            R1,R2,p1,p2,b1,b2,2
         );

         bench_border_rect_KD
         (
            FX+FY+FZ,
            R1,R2,p1,p2,b1,b2,3
         );

         bench_border_rect_KD
         (
            (FX+FY+FZ+FT),
            R1,R2,p1,p2,b1,b2,4
         );

         Fonc_Num f = 0;
         for (int i = 0; i<4 ; i++)
             f = f+Square(kth_coord(i));

         bench_border_rect_KD
         (
            f,
            R1,R2,p1,p2,b1,b2,4
         );
    }
}

//*******************************************

void bench_border_rect()
{
     bench_border_rect_2D();
     bench_border_rect_1D();
     bench_border_rect_KD();
}






