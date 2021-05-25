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



void dist_chamfer_cabl(Im2D<U_INT1,INT> I,INT v_max)
{
     Im2D<U_INT1,INT> I0(I.tx(),I.ty(),0);

     ELISE_COPY(I0.all_pts(),I.in(),I0.out());
     Chamfer::d32.im_dist(I);

     INT nb_dif;
     ELISE_COPY
     (
          I.all_pts(),
          I0.in()!=(I.in()!=0),
          sigma(nb_dif)
     );
     BENCH_ASSERT(nb_dif == 0);

     INT tx = I.tx();
     INT ty = I.ty();
     U_INT1 ** d = I.data();
     INT     vmax = I.vmax()-1;
     
     for (int x=1; x<tx-1 ; x++)
         for (int y=1; y<ty-1 ; y++)
         {
              
              INT v;
              if (d[y][x])
                 v  = ElMin3 
                      (
                          ElMin3(d[y+1][x-1]+3,d[y+1][x]+2,d[y+1][x+1]+3),
                          ElMin3(d[y][x-1]+2,vmax,d[y][x+1]+2),
                          ElMin3(d[y-1][x-1]+3,d[y-1][x]+2,d[y-1][x+1]+3)
                      );
              else
                 v = 0;

              BENCH_ASSERT(v == d[y][x]);
         }

      INT dif;
      ELISE_COPY 
      (
           I.all_pts(),
           Abs
           (
               Min(I.in(),v_max)
             - extinc_32(I0.in(0),v_max)
           ),
           VMax(dif)
      );

      BENCH_ASSERT(dif == 0);
}


void bench_env_klip(Pt2di sz)
{
     Im2D_U_INT1 I1(sz.x,sz.y);
     Im2D_U_INT1 I2(sz.x,sz.y);
     INT vmax = 30;




     ELISE_COPY
     (
         I1.all_pts(),
         Min
         (
             vmax,
                1
             +  frandr()*5
             +  unif_noise_4(3) * 30
         ),
         I1.out() 
     );
     ELISE_COPY
     (
         I1.border(1),
         0,
         I1.out()
     );
     ELISE_COPY
     (
         I1.all_pts(),
         EnvKLipshcitz_32(I1.in(0),vmax),
         I2.out() 
     );


     U_INT1 ** i1 = I1.data();
     U_INT1 ** i2 = I2.data();

     for (INT x=0; x<sz.x ; x++)
         for (INT y=0; y<sz.y ; y++)
            if (i1[y][x])
            {
                 INT v  = ElMin3 
                      (
                          ElMin3(i2[y+1][x-1]+3,i2[y+1][x]+2,i2[y+1][x+1]+3),
                          ElMin3(i2[y][x-1]+2,(INT)i1[y][x],i2[y][x+1]+2),
                          ElMin3(i2[y-1][x-1]+3,i2[y-1][x]+2,i2[y-1][x+1]+3)
                      );
                 BENCH_ASSERT(v==i2[y][x]);
           }
}

void  bench_dist_chamfrain()
{

       bench_env_klip(Pt2di(200,300));

      {
          Im2D<U_INT1,INT> I(300,300,0);
          //ELISE_COPY(disc(Pt2di(150,150),120),1,I.out());
          ELISE_COPY(disc(Pt2dr(150,150),120),1,I.out()); // __NEW
          dist_chamfer_cabl(I,35);
      }


      {
          Im2D<U_INT1,INT> I(600,600,0);
          //ELISE_COPY(disc(Pt2di(300,300),290),1,I.out());
          //ELISE_COPY(disc(Pt2di(300,300),5),0,I.out());
          ELISE_COPY(disc(Pt2dr(300,300),290),1,I.out()); // __NEW
          ELISE_COPY(disc(Pt2dr(300,300),5),0,I.out());   // __NEW
          dist_chamfer_cabl(I,144);
      }




     cout << "OK chamfer TEST \n";
}
