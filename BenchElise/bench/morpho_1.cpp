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




void zonec_at_hand
     (   Video_Win w,
         Im2D_U_INT1 i0,
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
        l|i0.out()|w.odisc()
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
             nl|i0.out()|w.odisc()
          );
          l = nl;
    }
}

Im2D_U_INT1 zonec_at_hand(Video_Win w,Im2D_U_INT1 i0,Pt2di * pts,INT nb_pts)
{
     Neighbourhood neigh = Neighbourhood(pts,nb_pts);
     INT tx = i0.tx();
     INT ty = i0.ty();
     Im2D_U_INT1 i1 (tx,ty);
     U_INT1 ** d1 = i1.data();

     ELISE_COPY (w.all_pts(),i0.in(),i1.out());

     INT coul = 0;
     for (INT y=0; y<ty ; y++)
         for (INT x=0; x<tx ; x++)
             if (d1[y][x] == 1)
              {
                    zonec_at_hand(w,i1,neigh,Pt2di(x,y),2+coul%6);
                    coul++;
              }
}

Im2D_U_INT1 zonec_conc(INT mode,Video_Win w,Im2D_U_INT1 i0,Pt2di * pts,INT nb_pts)
{
     Neighbourhood neigh = Neighbourhood(pts,nb_pts);
     INT tx = i0.tx();
     INT ty = i0.ty();
     Im2D_U_INT1 i1 (tx,ty);
     U_INT1 ** d1 = i1.data();

     ELISE_COPY (w.all_pts(),i0.in(),i1.out()|w.odisc());

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
                              w.odisc() |  i1.out()
                         );
                       break;

                       case 1 :
                         ELISE_COPY
                         (
                               conc(Pt2di(x,y),i1.neigh_test_and_set(neigh,1,coul,10)),
                               coul,
                               w.odisc()
                         );
                       break;
                       
                       default :
                           zonec_at_hand(w,i1,neigh,Pt2di(x,y),coul);
                       break;
                    }
                    nb++;
              }
    return i1;

    getchar();
}


void bench_zonec_at_hand(Video_Win w,Im2D_U_INT1 i0)
{

     ELISE_COPY (i0.border(1),P8COL::red,w.odisc()|i0.out());

     Im2D_U_INT1 I1 = zonec_conc(1,w,i0,TAB_8_NEIGH,8);
     {
         Im2D_U_INT1 I0 = zonec_conc(0,w,i0,TAB_8_NEIGH,8);
         bench_i1_eq_i2(I0,I1);
     }

     {
         Im2D_U_INT1 I2 = zonec_conc(2,w,i0,TAB_8_NEIGH,8);
         ELISE_COPY(I1.all_pts(),I1.in()!=I2.in(),w.odisc());
         bench_i1_eq_i2(I1,I2);
     }
}

void bench_zonec_simple(Video_Win w,Im2D_U_INT1 i0)
{
     ELISE_COPY
     (
         w.all_pts(),
         i0.in() / 255.0 < 
              (1+sin((FX+square(FX)/150.0)/5.0))
            * (1+sin((FY+square(FY)/150.0)/5.0))
            / 4
          ,
          i0.out() | w.odisc()
     );
     bench_zonec_at_hand(w,i0);

     ELISE_COPY(w.all_pts(),1,i0.out()|w.odisc());
     bench_zonec_at_hand(w,i0);
}




