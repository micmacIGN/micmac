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



void bench_dilate_simple(INT mode,Video_Win w,Im2D_U_INT1 i0,Pt2di * pts,INT nb_pts)
{
     Neighbourhood neigh = Neighbourhood(pts,nb_pts);
     INT tx = i0.tx();
     INT ty = i0.ty();
     Im2D_U_INT1 i1 (tx,ty);
     Im2D_U_INT1 i2 (tx,ty);

     ELISE_COPY
     (
         w.all_pts(),
         (i0.in() >> 6) & 1,
          i1.out() | w.odisc()
     );

     ELISE_COPY (i1.border(1),P8COL::red,w.odisc()|i1.out());
     ELISE_COPY(i1.all_pts(),i1.in(),i2.out()|w.odisc());

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
                  i1.out() | w.odisc() | (sigma(nb_dil1) << 1)
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
                  i1.out() | w.odisc() | (sigma(nb_dil1) << 1)
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
                  w.odisc() | (sigma(nb_dil1) << 1)
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
                  w.odisc() | (sigma(nb_dil1) << 1)
              );
              break;
          }
     }

     {
         Video_Win w2 = w.chc(Pt2di(200,200),Pt2di(3,3));
         ELISE_COPY(w2.all_pts(),i1.in(),w2.odisc());
     }

     one_step_dilate(i2,P8COL::blue,pts,nb_pts);
     bench_i1_eq_i2(i1,i2);

     INT nb_dil2;
     ELISE_COPY(i1.all_pts(),i2.in() == P8COL::blue,sigma(nb_dil2));

     BENCH_ASSERT(nb_dil1 == nb_dil2);
}


void bench_dilate_simple(Video_Win w,Im2D_U_INT1 i0)
{

     bench_dilate_simple(2,w,i0,TAB_8_NEIGH,8);
     bench_dilate_simple(2,w,i0,TAB_4_NEIGH,4);

     bench_dilate_simple(3,w,i0,TAB_8_NEIGH,8);
     bench_dilate_simple(3,w,i0,TAB_4_NEIGH,4);

     bench_dilate_simple(0,w,i0,TAB_8_NEIGH,8);
     bench_dilate_simple(0,w,i0,TAB_4_NEIGH,4);

     bench_dilate_simple(1,w,i0,TAB_8_NEIGH,8);
     bench_dilate_simple(1,w,i0,TAB_4_NEIGH,4);

}





