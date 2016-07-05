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




void bench_flag_vois_KD(Pt2di p1,Pt2di p2,Pt2di sz,Pt2di P0,Pt2di P1)
{
     
     Im2D_INT4  r1(sz.x,sz.y);
     Im2D_INT4  r2(sz.x,sz.y);
     Im2D_INT4  t1(sz.x,sz.y);
     Im2D_INT4  t2(sz.x,sz.y);


     Im2D_U_INT1  I1(sz.x,sz.y);
     Im2D_U_INT1  I2(sz.x,sz.y);

     ELISE_COPY
     (
         I1.all_pts(),
         frandr()>0.5,
         I1.out() | r1.out() | t1.out()
     );

     ELISE_COPY
     (
         I2.all_pts(),
         frandr()>0.5,
         I2.out() | r2.out() | t2.out()
     );

     //  Calcul K-d

     ELISE_COPY
     (
        rectangle(P0,P1),
        flag_vois(Virgule(I1.in(1),I2.in(1)),Box2di(p1,p2)),
        Virgule(r1.out() , r2.out())
     );


     //  K Calculs 1-d

     ELISE_COPY
     (
        rectangle(P0,P1),
        flag_vois(I1.in(1),Box2di(p1,p2)),
        t1.out() 
     );
     ELISE_COPY
     (
        rectangle(P0,P1),
        flag_vois(I2.in(1),Box2di(p1,p2)),
        t2.out() 
     );


     INT dif = 0;
     ELISE_COPY
     (
         r1.all_pts(),
            (r1.in() != t1.in())
         +  (r2.in() != t2.in()),
         sigma(dif)
     );

     BENCH_ASSERT(dif==0);
}



     // BENCH 1-D en out 

void bench_flag_vois(Pt2di p1,Pt2di p2,Pt2di sz,Pt2di P0,Pt2di P1)
{


     Im2D_INT4  r1(sz.x,sz.y);
     Im2D_INT4  r2(sz.x,sz.y);


     Im2D_U_INT1  I(sz.x,sz.y);

     // Initialisation de I a une image random

     ELISE_COPY
     (
         I.all_pts(),
         frandr()>0.5,
         I.out() | r1.out() | r2.out()
     );


     // Calcul, dans r1, du flagage, par l'operateur flag_vois

     ELISE_COPY
     (
        rectangle(P0,P1),
        flag_vois(I.in(1),Box2di(p1,p2)),
        r1.out()
     );

     // Calcul, dans r2, du flagage, par formule brutale

     {
        Fonc_Num f0 = 0;
        INT fl = 0;
        for (INT x = p1.x; x <= p2.x; x++)
            for (INT y = p1.y ; y <= p2.y ; y++)
            {
                f0 = f0 | (trans(I.in(1),Pt2di(x,y)) << fl);
                fl++;
            }
        ELISE_COPY
        (
           rectangle(P0,P1),
           f0,
           r2.out()
        );
     }

     // Verification 

     INT dif = 0;
     ELISE_COPY
     (
         r1.all_pts(),
         r1.in() != r2.in(),
         sigma(dif)
     );

     BENCH_ASSERT(dif==0);

     bench_flag_vois_KD(p1,p2,sz,P0,P1);
}




void bench_flag_vois()
{
     bench_flag_vois(Pt2di(-1,-1),Pt2di(1,1),Pt2di(30,50),Pt2di(0,0),Pt2di(30,50));

     bench_flag_vois(Pt2di(-1,-1),Pt2di(2,2),Pt2di(60,50),Pt2di(10,5),Pt2di(53,47));
}

Fonc_Num K_erod_8(Fonc_Num f,INT nb)
{
      for (INT i =0; i<nb ; i++)
          f = erod_8_hom(f);
      return f;
}

void verif_ss_Image_homot
     (
         Im2D_INT1 Isup,
         Im2D_INT1 Iinf,
         Neighbourhood v,
         INT coul
     )
{

   {   // [1] verif que Isup >= Iinf
       INT not_sup_inf;
       ELISE_COPY
       (
           Isup.all_pts(),
           (Iinf.in()==coul)&&(Isup.in() != coul),
           sigma(not_sup_inf)
       );
       BENCH_ASSERT(not_sup_inf==0);
    }


    ELISE_COPY
    (
        Iinf.border(1), 
        2,
        Iinf.out() |Isup.out()
    );

    INT1 ** d = Iinf.data();
    INT tx = Iinf.tx();
    INT ty = Iinf.ty();

    for (INT x =0; x < tx ; x++)
        for (INT y =0; y < ty ; y++)
            if (coul==d[y][x])
            {
                ELISE_COPY
                (
                    conc(Pt2di(x,y),sel_func(v,Iinf.in()==coul)),
                    3,
                    Iinf.out() 
                );

            // [2] pour chaque cc de Iinf verif que la cd dans Isup
            // dans laquelle elle est incluse, ne contient pas
            // d'autres point de Iinf

                INT nb_inf;
                ELISE_COPY
                (
                    conc(Pt2di(x,y),sel_func(v,Isup.in()==coul)),
                    3,
                    Isup.out() | (sigma(nb_inf) << (Iinf.in() == coul))
                );
                BENCH_ASSERT(nb_inf==0);
            }

     // [3] verif que tout les points de Isup ont ete colorie
     //  (== pas de cc de Isup vide de point de Iinf)

     INT nb_sup_not_col;
     ELISE_COPY
     (
         Isup.all_pts(),
         Isup.in() == coul,
         sigma(nb_sup_not_col)
     );
     BENCH_ASSERT(nb_sup_not_col==0);

     ELISE_COPY(select(Isup.all_pts(),Isup.in()==3),coul,Isup.out());
     ELISE_COPY(select(Iinf.all_pts(),Iinf.in()==3),coul,Iinf.out());
}

void bench_eros_hom
     (
         Pt2di    sz
     )
{
    Im2D_INT1 I1(sz.x,sz.y);
    Im2D_INT1 I2(sz.x,sz.y);

     ELISE_COPY
     (
         I1.all_pts(),
         rect_median
         (
             Iconv(frandr()*255),
             Box2di(Pt2di(-1,-1),Pt2di(2,2)),
             256
         ) < 128,
         I1.out() 
     );

     ELISE_COPY
     (
         I1.border(2),
         0,
         I1.out() 
     );

     ELISE_COPY(I1.all_pts(),K_erod_8(I1.in(0),3),I2.out());


     verif_ss_Image_homot(I2,I1,Neighbourhood(TAB_4_NEIGH,4),0);
     verif_ss_Image_homot(I1,I2,Neighbourhood(TAB_8_NEIGH,8),1);
}


void bench_eros_hom()
{
     bench_eros_hom(Pt2di(150,170));
     bench_eros_hom(Pt2di(70,50));
     cout << "OK bench_eros_hom \n";
}




void  bench_op_buf_2()
{

     {
         All_Memo_counter MC_INIT;
         stow_memory_counter(MC_INIT);
         bench_eros_hom();
         verif_memory_state(MC_INIT);
         cout << "OK eros hom \n";
     }

     {
         All_Memo_counter MC_INIT;
         stow_memory_counter(MC_INIT);
         bench_flag_vois();
         verif_memory_state(MC_INIT);
         cout << "OK flag vois \n";
     }

    
     cout << "OK bench_op_buf_2 \n";
}
