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



void q_bench_line_map_rect(INT dx,INT x0,INT x1)
{
   INT nb,sx,x_max,x_min;

   if (dx == 0)
      dx = -2;

    ELISE_COPY
    (
          line_map_rect(dx,x0,x1),
          Virgule(1,FX),
          Virgule
          (    sigma(nb),
               (sigma(sx)|VMax(x_max)|VMin(x_min))
          )
    );
    INT X_min = ElMin(x0,x1);
    INT X_max = ElMax(x0,x1);



    BENCH_ASSERT
    (
                 (nb == (X_max-X_min))
             &&  ((x_max == X_max-1) || (! nb))
             &&  ((x_min == X_min )  ||  (! nb))
             &&  (sx == som_x(X_min,X_max))
    );

}


void q_bench_line_map_rect(Pt2di u,Pt2di p0,Pt2di p1)
{
    INT nb,sx,sy,x_max,x_min,y_max,y_min;

    q_bench_line_map_rect(u.x+u.y,20*p0.x+15*p0.y,22*p1.x+33*p1.y);

    ELISE_COPY
    (
          line_map_rect(u,p0,p1),
          Virgule(1,FX,FY),
          Virgule
          (    sigma(nb),
               (sigma(sx)|VMax(x_max)|VMin(x_min)),
               (sigma(sy)|VMax(y_max)|VMin(y_min))
          )
    );

    Pt2di pmax = Sup(p0,p1);
    Pt2di pmin = Inf(p0,p1);


    BENCH_ASSERT
    (
                 (nb == (pmax.x-pmin.x)*(pmax.y-pmin.y))
             &&  ((x_max == pmax.x-1) || (! nb))
             &&  ((x_min == pmin.x)   ||  (! nb))
             &&  (sx == som_x(pmin.x,pmax.x) * (pmax.y-pmin.y))

             &&  ((y_max == pmax.y-1) || (! nb))
             &&  ((y_min == pmin.y)   ||  (! nb))
             &&  (sy == som_x(pmin.y,pmax.y) * (pmax.x-pmin.x))
    );
}

void q_bench_line_map_rect()
{
     q_bench_line_map_rect(Pt2di(10,10),Pt2di(0,0),Pt2di(5,5));

     q_bench_line_map_rect(Pt2di(3,4),Pt2di(-10,-10),Pt2di(23,67));
     q_bench_line_map_rect(Pt2di(-3,-4),Pt2di(-10,-10),Pt2di(23,67));
     q_bench_line_map_rect(Pt2di(3,4),Pt2di(23,67),Pt2di(-10,-10));

     q_bench_line_map_rect(Pt2di(3,4),Pt2di(23,23),Pt2di(24,24));
     q_bench_line_map_rect(Pt2di(3,4),Pt2di(23,23),Pt2di(23,23));
     q_bench_line_map_rect(Pt2di(3,4),Pt2di(23,23),Pt2di(23,24));
     q_bench_line_map_rect(Pt2di(3,4),Pt2di(23,23),Pt2di(24,23));

     INT pas_rect = 10;
     INT pas_dir = 5;

     for (int ux = -10; ux < 10; ux += pas_dir)
         for (int uy = -10; uy < 10; uy += pas_dir)
         {
             if (ux || uy)
                 for (int x0 = -12; x0 < 20 ; x0 +=pas_rect)
                     for (int y0 = -12; y0 < 20 ; y0 +=pas_rect)
                         for (int x1 = -12; x1 < 20 ; x1 +=pas_rect)
                             for (int y1 = -12; y1 < 20 ; y1 +=pas_rect)
                                  q_bench_line_map_rect
                                  (
                                      Pt2di(ux,uy),
                                      Pt2di(x0,y0),
                                      Pt2di(x1,y1)
                                  );
         }
}



//*********************************************

void bench_flux_line_map_rect()
{
     q_bench_line_map_rect();
     cout << "bench_flux_line_map_rect bench_line_map_rect \n";
}

//*********************************************

template <class Type,class TypeBase> void bench_filtr_line_map_rect
(
            Type     *,
            TypeBase *,
            const OperAssocMixte & op,
            Pt2di sz,
            Pt2di dir,
            INT   k0,
            INT   k1,
            REAL  def_out
)
{
    Im2D<Type,TypeBase> I(sz.x,sz.y);

    ELISE_COPY
    (    I.all_pts(),
         (FX+FX*FY + FX/(FX%3+FY%4 + 2)+ FY%8) %  128,
         I.out()
   );

    Fonc_Num f = I.in(def_out)[Virgule(FX+k0*dir.x,FY+k0*dir.y)];
    for(int k = k0+1; k <= k1; k++)
       f = op.opf
          (
                 f,
                 I.in(def_out)[Virgule(FX+k*dir.x,FY+k*dir.y)]
           );


   INT nb_dif;

   TypeBase truc;
   ELISE_COPY
   (
       line_map_rect(dir,Pt2di(0,0),sz),
       linear_red(op,I.in(),k0,k1),
       sigma(truc)
   );


   ELISE_COPY
   (
       line_map_rect(dir,Pt2di(0,0),sz),
       f,
       sigma(truc)
   );


   ELISE_COPY
   (
       line_map_rect(dir,Pt2di(0,0),sz),
       linear_red(op,I.in(),k0,k1) != f,
       sigma(nb_dif)
   );

   BENCH_ASSERT(nb_dif==0);
}

template <class Type,class TypeBase> void bench_filtr_line_map_rect_1D
(
            Type     *,
            TypeBase *,
            const OperAssocMixte & op,
            INT   sz,
            INT   dir,
            INT   k0,
            INT   k1,
            REAL  def_out
)
{
    Im1D<Type,TypeBase> I(sz);

    INT sdir = (dir > 0) ? 1 : -1; 

    ELISE_COPY
    (    I.all_pts(),
         (FX+ FX/(FX%3+ 2)+ Square(FX)) %  128,
         I.out()
   );

    Fonc_Num f = trans(I.in(def_out),sdir*k0);
    for(int k = k0+1; k <= k1; k++)
          f = op.opf(f,trans(I.in(def_out),sdir*k));


   INT nb_dif;

   ELISE_COPY
   (
       line_map_rect(dir,0,sz),
       linear_red(op,I.in(),k0,k1) != f,
       sigma(nb_dif)
   );

   BENCH_ASSERT(nb_dif==0);
}



void bench_filtr_Kdim_line_map_rect
(
            const OperAssocMixte & op,
            Pt2di sz,
            Pt2di dir,
            INT   k0,
            INT   k1
)
{

   INT d1,d2,d3;
   ELISE_COPY
   (
       line_map_rect(dir,Pt2di(0,0),sz),
       Abs
       (
              linear_red(op,Virgule(FX,FY,FX+FY),k0,k1)
           -  Virgule
              (
                  linear_red(op,FX,k0,k1),
                  linear_red(op,FY,k0,k1),
                  linear_red(op,FX+FY,k0,k1)
              )
       ),
       Virgule(VMax(d1),VMax(d2),VMax(d3))
   );


   BENCH_ASSERT((d1==0) && (d2==0) && (d3==0));
}




void bench_filtr_line_map_rect()
{

     bench_filtr_line_map_rect_1D
     (
         (REAL4 *) 0,
         (REAL  *) 0,
         OpMax,
         1005,
         1,
         -7,
         5,
         OpMax.rneutre()
     );

     bench_filtr_line_map_rect_1D
     (
         (INT4 *) 0,
         (INT4  *) 0,
         OpSum,
         1005,
         -11,
         -7,
         5,
         OpSum.ineutre()
     );

     bench_filtr_line_map_rect_1D
     (
         (INT4 *) 0,
         (INT4  *) 0,
         OpSum,
         1005,
         -11,
         7,
         17,
         OpSum.ineutre()
     );

     bench_filtr_line_map_rect_1D
     (
         (INT4 *) 0,
         (INT4  *) 0,
         OpSum,
         1005,
         -11,
         -17,
         -7,
         OpSum.ineutre()
     );





     bench_filtr_line_map_rect
     (
         (REAL4 *) 0,
         (REAL  *) 0,
         OpMax,
         Pt2di(103,134),
         Pt2di(1,1),
         -7,
         5,
         OpMax.rneutre()
     );


     for (int k=0 ; k<8 ; k++)
        bench_filtr_line_map_rect
        (
            (REAL4 *) 0,
            (REAL  *) 0,
            OpMax,
            Pt2di(13,14) +TAB_8_NEIGH[k]*2,
            TAB_8_NEIGH[k],
            -7+k/2,
            5+k/3,
            OpMax.rneutre()
        );

     bench_filtr_line_map_rect
     (
         (REAL4 *) 0,
         (REAL  *) 0,
         OpSum,
         Pt2di(103,134),
         Pt2di(1,1),
         -7,
         5,
         OpSum.rneutre()
     );


     bench_filtr_line_map_rect
     (
         (U_INT1 *) 0,
         (INT  *) 0,
         OpSum,
         Pt2di(103,134),
         Pt2di(1,1),
         -7,
         15,
         OpSum.ineutre()
     );


     bench_filtr_line_map_rect
     (
         (U_INT1 *) 0,
         (INT  *) 0,
         OpMin,
         Pt2di(103,134),
         Pt2di(1,1),
         5,
         15,
         OpMin.ineutre()
     );

     bench_filtr_line_map_rect
     (
         (U_INT1 *) 0,
         (INT  *) 0,
         OpMin,
         Pt2di(103,134),
         Pt2di(1,1),
         -15,
         -5,
         OpMin.ineutre()
     );

     bench_filtr_Kdim_line_map_rect
     (
           OpSum,
           Pt2di(100,120),
           Pt2di(3,5),
           -5,
           10
     );

     cout << "OK bench_filtr_line_map_rect \n";
}













