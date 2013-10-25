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




template <class Type> void bench_bob(Type *,Pt2di sz,Pt2di p0,Pt2di p1)
{
     Im2D<Type,Type> i0(sz.x,sz.y,(Type)0);
     Im2D<Type,Type> igx(sz.x,sz.y,(Type)0);
     Im2D<Type,Type> igy(sz.x,sz.y,(Type)0);

     Im2D<Type,Type> vgx(sz.x,sz.y,(Type)0);
     Im2D<Type,Type> vgy(sz.x,sz.y,(Type)0);


     ELISE_COPY
     (
            i0.all_pts(),
            100*cos(FX+Square(FY)) + (FY%10) * sin(FY),
            i0.out()
     );
     ELISE_COPY
     (
        rectangle(p0,p1),
        bobs_grad(i0.in()),
        Virgule(igx.out(),igy.out())
     );

     ELISE_COPY
     (
        rectangle(p0,p1),
        Virgule
        ( 
           trans(i0.in(),Pt2di(1,0)) - i0.in(),
           trans(i0.in(),Pt2di(0,1)) - i0.in()
        ),
        Virgule(vgx.out(),vgy.out())
     );

     Type difx,dify;

     ELISE_COPY 
     ( 
          i0.all_pts(),
          Abs(Virgule(igx.in(),igy.in()) - Virgule(vgx.in(),vgy.in())),
          Virgule(VMax(difx),VMax(dify))
     );

     BENCH_ASSERT
     (
          (difx < epsilon) && (dify < epsilon)
          
     );

     
     Fonc_Num d2xx = trans(i0.in(),Pt2di(1,0))+trans(i0.in(),Pt2di(-1,0))-2*i0.in();
     Fonc_Num d2yy = trans(i0.in(),Pt2di(0,1))+trans(i0.in(),Pt2di(0,-1))-2*i0.in();
     Fonc_Num d2xy =  (
                       trans(i0.in(),Pt2di(1,1))+trans(i0.in(),Pt2di(-1,-1))
                      -trans(i0.in(),Pt2di(-1,1))-trans(i0.in(),Pt2di(1,-1))
                      ) / 4;

     Type  Dif2 [3];
     ELISE_COPY
     (
        rectangle(p0,p1),
        Abs(Virgule(d2xx,d2xy,d2yy) -sec_deriv(i0.in())),
        VMax(Dif2,3)
     );


     BENCH_ASSERT
     (
             (Dif2[0]<epsilon) 
          && (Dif2[1]<epsilon) 
          && (Dif2[2]<epsilon)
          
     );
}


void bench_bob()
{
     bench_bob((INT *) 0,Pt2di(120,320),Pt2di(2,4),Pt2di(115,290));
     bench_bob((REAL *) 0,Pt2di(120,50),Pt2di(2,4),Pt2di(115,45));
}

template <class Type> void bench_red_rect_op_ass
                      (
                          const OperAssocMixte & op,
                          Type *,
                          Pt2di sz,
                          Pt2di p0,
                          Pt2di p1,
                          Pt2di sid0,
                          Pt2di sid1
                      )
{
     Im2D<Type,Type> i0(sz.x,sz.y,(Type)0);
     Im2D<Type,Type> ibuf(sz.x,sz.y,(Type)0);
     Im2D<Type,Type> iver(sz.x,sz.y,(Type)0);

     ELISE_COPY
     (
         i0.all_pts(),
         Square(FX)%11 + FY/3.0 + 1.2 +8*cos(FX*FY) + 3* sin(FY/12.0),
         i0.out()
     );

     Type ntr;
     op.set_neutre(ntr);
     ELISE_COPY
     (
         rectangle(p0,p1),
         rect_red(op,i0.in(ntr),Box2di(sid0,sid1)),
         ibuf.out()
     );

    Fonc_Num f = ntr;
    for(int x = sid0.x; x <= sid1.x; x++)
       for(int y = sid0.y; y <= sid1.y; y++)
          f = op.opf
              (
                 f,
                 trans(i0.in(ntr),Pt2di(x,y))
              );

     ELISE_COPY
     (
         rectangle(p0,p1),
         f,
         iver.out()
     );

     Type dif;
     ELISE_COPY(i0.all_pts(),Abs(ibuf.in()-iver.in()),VMax(dif));

     BENCH_ASSERT(dif < epsilon);
}

void bench_red_rect_op_ass()
{

      bench_red_rect_op_ass
      (
           OpMin,
           (REAL *) 0,
           Pt2di(50,30),
           Pt2di(0,0),
           Pt2di(50,30),
           Pt2di(6,4),
           Pt2di(9,10)
      );


	  bench_red_rect_op_ass
      (
           OpMin,
           (REAL *) 0,
           Pt2di(50,30),
           Pt2di(0,0),
           Pt2di(50,30),
           Pt2di(-9,-10),
           Pt2di(-6,-4)
      );



      bench_red_rect_op_ass
      (
           OpMax,
           (INT *) 0,
           Pt2di(100,200),
           Pt2di(10,10),
           Pt2di(90,190),
           Pt2di(-1,-1),
           Pt2di(1,1)
      );


	
      bench_red_rect_op_ass
      (
           OpSum,
           (REAL *) 0,
           Pt2di(50,30),
           Pt2di(0,0),
           Pt2di(50,30),
           Pt2di(-1,-2),
           Pt2di(3,4)
      );


      bench_red_rect_op_ass
      (
           OpMin,
           (REAL *) 0,
           Pt2di(50,30),
           Pt2di(0,0),
           Pt2di(50,30),
           Pt2di(-2,-2),
           Pt2di(-1,4)
      );

	
	  bench_red_rect_op_ass
      (
           OpMin,
           (REAL *) 0,
           Pt2di(50,30),
           Pt2di(0,0),
           Pt2di(50,30),
           Pt2di(1,-2),
           Pt2di(2,4)
      );


	  bench_red_rect_op_ass
      (
           OpSum,
           (REAL *) 0,
           Pt2di(50,30),
           Pt2di(0,0),
           Pt2di(50,30),
           Pt2di(-1,-2),
           Pt2di(-1,4)
      );



	  bench_red_rect_op_ass
      (
           OpSum,
           (REAL *) 0,
           Pt2di(50,30),
           Pt2di(0,0),
           Pt2di(50,30),
           Pt2di(1,-2),
           Pt2di(1,4)
      );

      // test with various small values for optimization

      for (int x = 0; x < 5 ; x++)
           for (int y = 0; y < 5 ; y++)
           {
               bench_red_rect_op_ass
               (
                    OpSum,
                    (REAL *) 0,
                    Pt2di(50,30),
                    Pt2di(0,0),
                    Pt2di(50,30),
                    Pt2di(x-2,y-2),
                    Pt2di(2*x-2,2*y-2)
               );
           }
}



void bench_Kdim_rect_op_ass
                      (
                          const OperAssocMixte & op,
                          Pt2di p0,
                          Pt2di p1,
                          Pt2di sid0,
                          Pt2di sid1
                      )
{
     Box2di side = Box2di(sid0,sid1);

     INT d1,d2,d3;
     ELISE_COPY
     (
         rectangle(p0,p1),
         Abs
         (
              rect_red(op,Virgule(FX,FY,FX+FY),side)
              - Virgule
                (
                      rect_red(op,FX,side),
                      rect_red(op,FY,side),
                      rect_red(op,FX+FY,side)
                )
         ),
         Virgule(VMax(d1),VMax(d2),VMax(d3))
     );

     BENCH_ASSERT( (d1==0)&&(d2==0)&&(d3==0));
}

void bench_Kdim_rect_op_ass()
{
   bench_Kdim_rect_op_ass
   (
         OpSum,
         Pt2di(20,30),
         Pt2di(200,150),
         Pt2di(-2,-3),
         Pt2di(4,5)
   );

   bench_Kdim_rect_op_ass
   (
         OpMax,
         Pt2di(20,30),
         Pt2di(200,150),
         Pt2di(8,8),
         Pt2di(20,12)
   );

   bench_Kdim_rect_op_ass
   (
         OpMin,
         Pt2di(20,30),
         Pt2di(200,150),
         Pt2di(-20,-12),
         Pt2di(-8,-8)
   );
}

void bench_opb_iter_clip_def()
{
     Pt2di p0 (0,0);
     Pt2di p1 (100,100);
     Im2D<REAL8,REAL8> i0(p1.x,p1.y,0.0);
     Im2D<REAL8,REAL8> i1(p1.x,p1.y,0.0);

     ELISE_COPY
     (
         i0.all_pts(),
         Square(FX)%11 + (FY-50)/3.0 + 1.2 +8*cos(FX*FY) + 3* sin(FY/12.0),
         i0.out() | i1.out()
     );


     Box2di b1 (Pt2di(-1,-2),Pt2di(2,3));
     REAL def1 = -1;
     ELISE_COPY
     (
         i1.all_pts(),
         rect_max(i1.in(def1),b1),
         i1.out()
     );

     Box2di b2 (Pt2di(-3,-3),Pt2di(3,3));
     REAL def2 = 1.2;
     ELISE_COPY
     (
         i1.all_pts(),
         rect_som(i1.in(def2),b2),
         i1.out()
     );

     ELISE_COPY
     ( 
         i0.all_pts(),
         rect_som
         (
              clip_def
              (
                    rect_max(clip_def(i0.in(),def1,p0,p1), b1),
                    def2,
                    p0,
                    p1
              ),
              b2
         ),
         i0.out()
      );
      REAL dif;

      ELISE_COPY(i0.all_pts(),Abs(i0.in()-i1.in()),VMax(dif));

      cout << "Dif Op iter " << dif << "\n";
}

void  bench_op_buf_0()
{
     {
         cout << "BEGIN BOB \n";
         All_Memo_counter MC_INIT;
         stow_memory_counter(MC_INIT);
         bench_bob();
         verif_memory_state(MC_INIT);
         cout << "OK BOB \n";
     }
     {
         All_Memo_counter MC_INIT;
         stow_memory_counter(MC_INIT);
         bench_red_rect_op_ass();
         verif_memory_state(MC_INIT);
         cout << "OK rect red ass \n";
     }
     {
         All_Memo_counter MC_INIT;
         stow_memory_counter(MC_INIT);
         bench_Kdim_rect_op_ass();
         verif_memory_state(MC_INIT);
         cout << "OK Kdim_rect_op_ass \n";
     }
     {
         All_Memo_counter MC_INIT;
         stow_memory_counter(MC_INIT);
         bench_opb_iter_clip_def();
         verif_memory_state(MC_INIT);
         cout << "OK iter_clip_def \n";
     }
}
