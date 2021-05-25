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



#include "general/all.h"
#include "private/all.h"


All_Memo_counter MC_INIT;


/*********** UTIL ********************************/




main(int,char *)
{
     ELISE_DEBUG_USER = true;
     stow_memory_counter(MC_INIT);

{

    Disc_Pal       Pdisc  = Disc_Pal::P8COL();
    Elise_Set_Of_Palette SOP(newl(Pdisc));
    Video_Display Ecr((char *) NULL);
    Ecr.load(SOP);

    Video_Win   W  (Ecr,SOP,Pt2di(50,50),Pt2di(500,500));



    ELISE_COPY(W.all_pts(),P8COL::white,W.odisc());

    ELISE_COPY(W.all_pts(),P8COL::white,W.odisc());

    ELISE_COPY
    (
        sector_ang(Pt2di(256,256),220,1.0,3.0),
        P8COL::yellow,
        W.out(Pdisc)
    );
    ELISE_COPY
    (
        fr_sector_ang(Pt2di(256,256),220,1.0,3.0),
        P8COL::black,
        W.out(Pdisc)
    );
getchar();
    ELISE_COPY
    (
        sector_ang(Pt2di(256,256),220,1.0,2.0),
        P8COL::black,
        W.out(Pdisc)
    );
getchar();


    W = W.chc(Pt2dr(0,0),Pt2dr(3,3));

    ELISE_COPY(W.all_pts(),P8COL::white,W.odisc());

    ELISE_COPY
    (
            sector_ang(Pt2dr(80,80),30,1.0123,1.0124),
            P8COL::red,
            W.odisc()
    );
    getchar();

    ELISE_COPY
    (
            sector_ang(Pt2dr(80,80),30,1.0124,1.0123),
            P8COL::blue,
            W.odisc()
    );
    getchar();


    ELISE_COPY
    (
             ell_fill(Pt2dr(80,80),70,40,0.3),
             P8COL::yellow,
             W.odisc()
    );
    ELISE_COPY
    (
             sector_ell(Pt2dr(80,80),70,40,0.3,0.0,PI/2),
             P8COL::red,
             W.odisc()
    );

    ELISE_COPY
    (
             chord_ell(Pt2dr(80,80),70,40,0.3,PI/2+0.2,(3*PI)/2-0.2),
             P8COL::magenta,
             W.odisc()
    );
    ELISE_COPY
    (
             fr_chord_ell(Pt2dr(80,80),70,40,0.3,PI/2+0.2,(3*PI)/2-0.2),
             P8COL::blue,
             W.odisc()
    );



    ELISE_COPY
    (
             fr_sector_ell(Pt2dr(80,80),70,40,0.3,0.5,2.0),
             P8COL::black,
             W.odisc()
    );


    getchar();

    ELISE_COPY(W.all_pts(),P8COL::white,W.odisc());
    ELISE_COPY(fr_sector_ang(Pt2dr(80,80),30,-0.5,PI/2),P8COL::blue,W.odisc());
    ELISE_COPY(sector_ang(Pt2dr(80,80),30,PI,(3*PI)/2+0.5,true),
          P8COL::red,W.odisc());
    ELISE_COPY(sector_ang(Pt2dr(80,80),30,PI,(3*PI)/2+0.5,false),
          P8COL::green,W.odisc());

    getchar();
    ELISE_COPY(chord_ang(Pt2dr(80,80),30,PI-0.5,(3*PI)/2+0.5,true),
          P8COL::cyan,W.odisc());
    ELISE_COPY(fr_chord_ang(Pt2dr(80,80),30,PI-0.5,(3*PI)/2+0.5,true),
          P8COL::black,W.odisc());
    getchar();

    ELISE_COPY(W.all_pts(),P8COL::white,W.odisc());
    ELISE_COPY
    (
             rectangle(Pt2di(10,10),Pt2di(50,50))
         ||  disc(Pt2di(60,60),30)
         ||  rectangle(Pt2di(80,80),Pt2di(110,110)),
         P8COL::green,
         W.odisc()
    );

    ELISE_COPY
    (
             rectangle(Pt2di(10,10),Pt2di(50,50))
         ||  select(disc(Pt2di(60,60),30),FX%2)
         ||  rectangle(Pt2di(80,80),Pt2di(110,110)),
         P8COL::blue,
         W.odisc()
    );


    getchar();
    ELISE_COPY(W.all_pts(),P8COL::white,W.odisc());

    ELISE_COPY
    (
          polygone
          (
               newl(Pt2di(0,0))
             + Pt2di(  0,150)
             + Pt2di(150,150)
             + Pt2di(150,  0)
          ),
          P8COL::green,
          W.odisc()
    );

    ELISE_COPY
    (
          polygone
          (
               newl(Pt2di(20,20))
             + Pt2di(80,70)
             + Pt2di(30,90)
          ),
          P8COL::red,
          W.odisc()
    );

    ELISE_COPY
    (
          polygone
          (
               newl(Pt2di(20,20))
             +      Pt2di(30,120)
             +      Pt2di(20,120)
             +      Pt2di(30,20)
          ),
          P8COL::blue,
          W.odisc()
    );

    ELISE_COPY
    (
          polygone
          (
               newl(Pt2di(20,20))
             +      Pt2di(30,120)
             +      Pt2di(20,120)
             +      Pt2di(30,20)   ,  
             false
          ),
          P8COL::black,
          W.odisc()
    );

    getchar();

    ELISE_COPY
    (
          polygone
          (
               newl(Pt2di(10,10))
             +      Pt2di(120, 100 )
             +      Pt2di( 50, 100 )
             +      Pt2di(140,  10)
             +      Pt2di(140,140)
             +      Pt2di(  10,140)
          ),
          P8COL::blue,
          W.odisc()
    );

    ELISE_COPY
    (
          polygone
          (
               newl(Pt2di(10,10))
             +      Pt2di(120, 100 )
             +      Pt2di( 50, 100 )
             +      Pt2di(140,  10)
             +      Pt2di(140,140)
             +      Pt2di(  10,140),
             false
          ),
          P8COL::white,
          W.odisc()
    );



    getchar();


    ELISE_COPY(W.all_pts(),P8COL::white,W.odisc());
    ELISE_COPY(disc(Pt2dr(80,80),15,true ),P8COL::black,W.odisc());
    ELISE_COPY(disc(Pt2dr(80,80),15,false),P8COL::green,W.odisc());

    getchar();

    ELISE_COPY(W.all_pts(),P8COL::white,W.odisc());
    ELISE_COPY(circle(Pt2dr(80,80),15,true),P8COL::black,W.odisc());
    getchar();

    ELISE_COPY(disc(Pt2dr(80,80),15),P8COL::green,W.odisc());
    getchar();
    ELISE_COPY(circle(Pt2dr(80,80),15,false),P8COL::black,W.odisc());

    ELISE_COPY(circle(Pt2dr(80,80),10),P8COL::red,W.odisc());
    ELISE_COPY(circle(Pt2dr(80,80),30),P8COL::red,W.odisc());

    ELISE_COPY(arc_cir(Pt2dr(80,80),30,0,PI/2),P8COL::blue,W.odisc());

    getchar();

    ELISE_COPY(W.all_pts(),P8COL::white,W.odisc());
    ELISE_COPY
    (
         ellipse(Pt2dr(80,80),40,20,0.0),
         P8COL::red,
         W.odisc()
    );

    ELISE_COPY
    (
         ellipse(Pt2dr(80,80),20,40,0.0),
         P8COL::blue,
         W.odisc()
    );

    getchar();

    for (INT i = 0 ; i < 78 ; i++)
        ELISE_COPY
        (
             ellipse(Pt2dr(80,80),20,40,i*0.3),
             1 + i%4,
             W.odisc()
        );

    getchar();

    for (INT i = 0 ; i < 18 ; i++)
        ELISE_COPY
        (
             ell_fill(Pt2dr(80,80),20,40,i*0.3),
             1 + i%4,
             W.odisc()
        );

    getchar();

    ELISE_COPY(W.all_pts(),P8COL::white,W.odisc());
    ELISE_COPY
    (
         ellipse(Pt2dr(80,80),40,20,0.0),
         P8COL::red,
         W.odisc()
    );

    ELISE_COPY
    (
         arc_ellipse(Pt2dr(80,80),40,20,0.0,0.0,PI/2.0),
         P8COL::blue,
         W.odisc()
    );

    getchar();

}
     verif_memory_state(MC_INIT);
     cout << "OK BENCH 0 \n";
}


