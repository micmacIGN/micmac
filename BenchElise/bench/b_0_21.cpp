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





void bench_fr_front_to_surf
     (
          Flux_Pts       front,
          Im2D_U_INT1    i0,
          Neighbourhood  v
     )
{
     Pt2di sz (i0.tx(),i0.ty());

     Im2D_U_INT1 i1(sz.x,sz.y,0);
     ELISE_COPY(i0.all_pts(),i0.in(),i1.out());

     ELISE_COPY
     (
          dilate
          (
               select
               (
                    i1.all_pts(),
                    i1.in()==1
               ),
               sel_func(v,i1.in()==0)
          ),
          2,
          i1.out()
     );

     ELISE_COPY(front,2,i0.out());

     INT dif;
     ELISE_COPY
     (
          i1.all_pts(),
          Abs(i1.in()-i0.in()),
          sigma(dif)
     );

      BENCH_ASSERT(dif == 0);
}


void bench_front_to_surf
     (
          Flux_Pts   front_8,
          Flux_Pts   front_4,
          Flux_Pts   surf_with_fr,
          Flux_Pts   surf_without_fr,
          Fonc_Num   fcarac,
          bool       with_fc,
          bool       test_dil,
          Pt2di      sz
     )
{
     Im2D_U_INT1 i1(sz.x,sz.y,0);
     Im2D_U_INT1 i2(sz.x,sz.y,0);

     INT dif;
     if (with_fc)
     {
         ELISE_COPY(surf_without_fr,1,i1.out());
         ELISE_COPY(i2.all_pts(),fcarac,i2.out());

         ELISE_COPY(i1.all_pts(),Abs(i1.in()-i2.in()),sigma(dif));

          BENCH_ASSERT(dif == 0);
     }
     else
         ELISE_COPY(surf_without_fr,1,i1.out()|i2.out());

     Neighbourhood v4 (TAB_4_NEIGH,4);
     Neighbourhood v8 (TAB_8_NEIGH,8);
     
     if (test_dil)
     {
         bench_fr_front_to_surf(front_8,i1,v4);

         bench_fr_front_to_surf(front_4,i2,v8);
     }


     ELISE_COPY(i1.all_pts(),1,i1.out());
     ELISE_COPY(i1.border(1),0,i1.out());
     ELISE_COPY(front_8,0,i1.out());
     ELISE_COPY(conc(Pt2di(1,1),sel_func(v4,i1.in()==1)),0,i1.out());

     ELISE_COPY(i2.all_pts(),0,i2.out());
     ELISE_COPY(surf_without_fr,1,i2.out());
     ELISE_COPY(i1.all_pts(),Abs(i1.in()-i2.in()),sigma(dif));
      
     BENCH_ASSERT(dif == 0);


     ELISE_COPY(i1.all_pts(),0,i1.out() | i2.out());

     ELISE_COPY(surf_with_fr   ,1,i1.out());
     ELISE_COPY(surf_without_fr,2,i1.histo());

     ELISE_COPY(surf_with_fr   ,3,i2.out());
     ELISE_COPY(front_8        ,1,i2.out());

     
     ELISE_COPY(i1.all_pts(),Abs(i1.in()-i2.in()),sigma(dif));

     BENCH_ASSERT(dif == 0);

}


void bench_disque_cercle
     (
          Pt2di  c,
          REAL   r
     )
{
    Pt2dr c_r( (REAL)c.x, (REAL)c.y );
    bench_front_to_surf
    (
         //circle(c,r,true),
         //circle(c,r,false),
         //disc(c,r,true),
         //disc(c,r,false),
         circle(c_r,r,true),  // __NEW
         circle(c_r,r,false), // __NEW
         disc(c_r,r,true),    // __NEW
         disc(c_r,r,false),   // __NEW
         Square(FX-c.x)+Square(FY-c.y) < ElSquare(r),
         true,
         true,
         c+round_ni(Pt2dr(r,r)+Pt2dr(6,6))
    );
}


void bench_polyg_line(ElList<Pt2di> l)
{
    bench_front_to_surf
    (
         line(l,true),
         line_4c(l,true),
         polygone(l,true),
         polygone(l,false),
         FX,
         false,
         false,
         Pt2di(150,150)
    );
}

void bench_flux_geom()
{
     for (INT i = 2; i < 10; i++)
         //bench_disque_cercle(Pt2dr(2+8,2+8),2);
         bench_disque_cercle(Pt2di(2+8,2+8),2); // __NEW
     for (INT i = 2; i < 10; i++)
         //bench_disque_cercle(Pt2dr(i+8,i+8),i);
         bench_disque_cercle(Pt2di(i+8,i+8),i); // __NEW

     //bench_disque_cercle(Pt2dr(23,23),15.3);
     bench_disque_cercle(Pt2di(23,23),15.3); // __NEW

     bench_polyg_line 
     ( 
             NewLPt2di(Pt2di(5,5))    + Pt2di(  5,140) 
           +      Pt2di(140,140) + Pt2di(140,  5)
     );

     bench_polyg_line 
     ( 
           NewLPt2di(Pt2di(20,20)) + Pt2di(80,70) + Pt2di(30,90)
     );

     bench_polyg_line 
     ( 
           NewLPt2di(Pt2di(20,20)) + Pt2di(30,120)
         +      Pt2di(20,120) + Pt2di(30,20)

     );

     cout << "OK FLUX GEOM \n";
}
