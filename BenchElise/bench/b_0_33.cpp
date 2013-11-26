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


void bench_i1_eq_i2(Im2D_U_INT1 i1,Im2D_U_INT1 i2)
{
   INT nb_dif;

   ELISE_COPY(i1.all_pts(),Abs(i1.in()-i2.in()),sigma(nb_dif));

   BENCH_ASSERT(nb_dif ==0);
}

void zonec_at_hand
     (   Im2D_U_INT1 i0,
         Neighbourhood neigh,
         Pt2di p,
         INT coul
     )
{
    Liste_Pts_INT2 l(2);

    ELISE_COPY
    (
        p,
        coul,
        l|i0.out()
    );

    while (! l.empty())
    {
         Liste_Pts_INT2 nl(2);
         ELISE_COPY
         (
             dilate
             (   l.all_pts(),
                 i0.neigh_test_and_set(neigh,1,coul,20)
             ),
             coul,
             nl|i0.out()
          );
          l = nl;
    }
}

void zonec_at_hand(Im2D_U_INT1 i0,Pt2di * pts,INT nb_pts)
{
     Neighbourhood neigh = Neighbourhood(pts,nb_pts);
     INT tx = i0.tx();
     INT ty = i0.ty();
     Im2D_U_INT1 i1 (tx,ty);
     U_INT1 ** d1 = i1.data();

     ELISE_COPY (i0.all_pts(),i0.in(),i1.out());

     INT coul = 0;
     for (INT y=0; y<ty ; y++)
         for (INT x=0; x<tx ; x++)
             if (d1[y][x] == 1)
              {
                    zonec_at_hand(i1,neigh,Pt2di(x,y),2+coul%6);
                    coul++;
              }
}

Im2D_U_INT1 zonec_conc(INT mode,Im2D_U_INT1 i0,Pt2di * pts,INT nb_pts)
{
     Neighbourhood neigh = Neighbourhood(pts,nb_pts);
     INT tx = i0.tx();
     INT ty = i0.ty();
     Im2D_U_INT1 i1 (tx,ty);
     U_INT1 ** d1 = i1.data();

     ELISE_COPY (i0.all_pts(),i0.in(),i1.out());

     INT nb = 2;
     for (INT y=0; y<ty ; y++)
         for (INT x=0; x<tx ; x++)
             if (d1[y][x] == 1)
              {
                    INT coul =  2 + nb % 6;
                    switch(mode)
                    {
                       case 0:
                         ELISE_COPY
                         (
                             conc(Pt2di(x,y),sel_func(neigh,i1.in()==1)),
                             coul,
                             i1.out()
                         );
                       break;

                       case 1 :
                         ELISE_COPY
                         (
                               conc(Pt2di(x,y),i1.neigh_test_and_set(neigh,1,coul,10)),
                               coul,
                               Output::onul()
                         );
                       break;
                       
                       default :
                           zonec_at_hand(i1,neigh,Pt2di(x,y),coul);
                       break;
                    }
                    nb++;
              }
    return i1;

}


void bench_zonec_at_hand(Im2D_U_INT1 i0)
{

     ELISE_COPY (i0.border(1),P8COL::red,i0.out());

     Im2D_U_INT1 I1 = zonec_conc(1,i0,TAB_8_NEIGH,8);
     {
         Im2D_U_INT1 I0 = zonec_conc(0,i0,TAB_8_NEIGH,8);
         bench_i1_eq_i2(I0,I1);
     }

     {
         Im2D_U_INT1 I2 = zonec_conc(2,i0,TAB_8_NEIGH,8);
         bench_i1_eq_i2(I1,I2);
     }
}

void bench_zonec_simple(Im2D_U_INT1 i0)
{
     ELISE_COPY
     (
         i0.all_pts(),
         i0.in() / 255.0 < 
              (1+sin((FX+Square(FX)/150.0)/5.0))
            * (1+sin((FY+Square(FY)/150.0)/5.0))
            / 4
          ,
          i0.out()
     );
     bench_zonec_at_hand(i0);

     ELISE_COPY(i0.all_pts(),1,i0.out());
     bench_zonec_at_hand(i0);
}




void one_step_dilate(Im2D_U_INT1 i2,INT col_out,Pt2di * pts,INT nb_pts)
{
     INT tx = i2.tx();
     INT ty = i2.ty();
     U_INT1 ** d2 = i2.data();

     for (INT y=0 ; y<ty ; y++)
         for (INT x=0 ; x<tx ; x++)
             if (d2[y][x] == 1)
                 for (INT i=0; i<nb_pts; i++)
                 {
                     INT xn = x+ pts[i].x;
                     INT yn = y+ pts[i].y;
                     if (d2[yn][xn] == 0)
                        d2[yn][xn] = col_out;
                 }
}



void bench_dilate_simple(INT mode,Im2D_U_INT1 i0,Pt2di * pts,INT nb_pts)
{
     Neighbourhood neigh = Neighbourhood(pts,nb_pts);
     INT tx = i0.tx();
     INT ty = i0.ty();
     Im2D_U_INT1 i1 (tx,ty);
     Im2D_U_INT1 i2 (tx,ty);

     ELISE_COPY
     (
         i0.all_pts(),
         (i0.in() >> 6) & 1,
          i1.out()
     );

     ELISE_COPY (i1.border(1),P8COL::red,i1.out());
     ELISE_COPY(i1.all_pts(),i1.in(),i2.out());

     INT nb_dil1;

     switch(mode)
     {
           case 0 :
               ELISE_COPY
               (
                  select
                  (
                      dilate(select(i1.all_pts(),i1.in()==1),neigh),
                      i1.in() == 0
                  ),
                  P8COL::blue,
                  i1.out() | (sigma(nb_dil1) << 1)
               );
          break;

          case 1 :
               ELISE_COPY
               (
                  dilate
                  (
                     select(i1.all_pts(),i1.in()==1),
                     sel_func(neigh,i1.in() == 0)
                  ),
                  P8COL::blue,
                  i1.out()  | (sigma(nb_dil1) << 1)
               );
          break;

          case 2 :
          {
              Im1D<INT4,INT> isel(100,0);
              Im1D<INT4,INT> iupd(100,0);

              isel.data()[P8COL::white] = 1;
              iupd.data()[P8COL::white] = P8COL::blue;

              ELISE_COPY
              (
                 dilate
                 (     select(i1.all_pts(),i1.in()==1),
                       i1.neigh_test_and_set(neigh,isel,iupd)
                 ),
                  P8COL::blue,
                  (sigma(nb_dil1) << 1)
              );
               break;
          }

          default :
          {
              ELISE_COPY
              (
                 dilate
                 (     select(i1.all_pts(),i1.in()==1),
                       i1.neigh_test_and_set(neigh,P8COL::white,P8COL::blue,100)
                 ),
                  P8COL::blue,
                  (sigma(nb_dil1) << 1)
              );
              break;
          }
     }


     one_step_dilate(i2,P8COL::blue,pts,nb_pts);
     bench_i1_eq_i2(i1,i2);

     INT nb_dil2;
     ELISE_COPY(i1.all_pts(),i2.in() == P8COL::blue,sigma(nb_dil2));

     BENCH_ASSERT(nb_dil1 == nb_dil2);
}


void bench_dilate_simple(Im2D_U_INT1 i0)
{

     bench_dilate_simple(2,i0,TAB_8_NEIGH,8);
     bench_dilate_simple(2,i0,TAB_4_NEIGH,4);

     bench_dilate_simple(3,i0,TAB_8_NEIGH,8);
     bench_dilate_simple(3,i0,TAB_4_NEIGH,4);

     bench_dilate_simple(0,i0,TAB_8_NEIGH,8);
     bench_dilate_simple(0,i0,TAB_4_NEIGH,4);

     bench_dilate_simple(1,i0,TAB_8_NEIGH,8);
     bench_dilate_simple(1,i0,TAB_4_NEIGH,4);

}







//*************************************************************************



void bench_zonec_dilate()
{
     INT SZX = 300;
     INT SZY = 200;

     Im2D_U_INT1 I(SZX,SZY);


     ELISE_COPY
     (
           I.all_pts(),
           (     canny_exp_filt(frandr(),0.5,0.6,20)
              /  canny_exp_filt(1.0,0.5,0.6,20)
           )  * 255,
           I.out()
     );

     ELISE_COPY
     (
           select(I.all_pts(),frandr() < 0.1),
           frandr()*255,
           I.out()
     );




      bench_dilate_simple(I);
      bench_zonec_simple(I);

      printf("OK dilate/zonec \n");
}



