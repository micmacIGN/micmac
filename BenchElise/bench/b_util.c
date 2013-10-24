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



/*********** UTIL ********************************/

const int NB_I_TO_R = 20;

static REAL TAB_IR[NB_I_TO_R] = 
{
                 0.0  ,      0.1  ,      0.5  ,      0.6  ,      1.2  ,
                 1.5  ,      1.9  ,    123.4  ,    123.5  ,    123.9  ,
                -0.01 ,     -0.1  ,     -0.5  ,     -0.6  ,     -1.2  ,
                -1.5  ,     -1.9  ,   -123.4  ,   -123.5  ,   -123.9  ,
};


    // For a firts verification "by hand".

void cout_bench_int_to_real()
{
    for (int i =0 ; i<NB_I_TO_R ; i++)
        cout <<  TAB_IR[i] << " => " << round_ni(TAB_IR[i]) << "\n";
}


void bench_int_to_real(REAL r)
{
      int i = round_ni(r);
      BENCH_ASSERT(fabs(r-i) <= 0.5);
      if (fabs(r-i)  == 0.5)
         BENCH_ASSERT(r == i-0.5);



      i = round_up(r);
      BENCH_ASSERT( (i >=r) && (i < r+1));

      i = round_Uup(r);
      BENCH_ASSERT( (i >r) && (i <= r+1));




      i = round_down(r);
      BENCH_ASSERT( (i <=r) && (i > r-1));

      i = round_Ddown(r);
      BENCH_ASSERT( (i <r) && (i >= r-1));
}






void bench_int_to_real()
{
    for (int i =0 ; i<NB_I_TO_R ; i++)
         bench_int_to_real(TAB_IR[i]);
    cout << "OK bench_int_to_real\n";
}

//==============================

void bench_div()
{
     for (int a = -100; a < 100 ; a++)
         for (int b = 1 ; b < 100 ; b++)
         {
              INT r = Elise_div(a,b);
              BENCH_ASSERT ((b*r<=a) && (a<b*(r+1)));
         }
     cout << "OK bench div\n";
}



//==============================

INT SZFX = 1*2*5*6*7;
INT SZFY = 1*2*5*6*7;


void bench_X11_rep_ent(Fen_X11 f0,Pt2di tr,Pt2di sc)
{
     Fen_X11 f = f0.ch_coord(tr,sc);

     Pt2di p1,p2;
     f.box_user_geom(p1,p2);

/*
     cout << "Translat : " << tr.x << " " << tr.y 
          << " Scale : "   << sc.x << " " << sc.y << "\n";

     cout << "p1 : " << p1.x << " " << p1.y 
          << " P2 : "   << p2.x << " " << p2.y << "\n";
*/

     BENCH_ASSERT
     (
               (p1.x == -round_ni(tr.x))
          &&   (p1.y == -round_ni(tr.y))  
          &&   (p2.x == -round_ni(tr.x) + SZFX /sc.x ) 
          &&   (p2.y == -round_ni(tr.y) + SZFY /sc.y ) 
     );
}




void bench_X11_rep_inv_ent(Fen_X11 f0,Pt2dr tr,Pt2di sc)
{
     Fen_X11 f = f0.ch_coord(tr,Pt2dr(1.0/sc.x,1.0/sc.y));

     Pt2di p1,p2;
     f.box_user_geom(p1,p2);

     cout << "Translat : " << tr.x << " " << tr.y 
          << " Scale : "   << sc.x << " " << sc.y << "\n";

     cout << "p1 : " << p1.x << " " << p1.y 
          << " P2 : "   << p2.x << " " << p2.y << "\n";

     cout << "Theor : " << SZFX *sc.x  << " " << SZFY *sc.y << "\n";


     BENCH_ASSERT
     (
               (p1.x == -round_ni(tr.x))
          &&   (p1.y == -round_ni(tr.y))  
          &&   (p2.x == -round_ni(tr.x) + SZFX *sc.x ) 
          &&   (p2.y == -round_ni(tr.y) + SZFY *sc.y ) 
     );
}

void bench_X11_rep(void)
{
     Fen_X11 f(Pt2di(40,40),Pt2di(SZFX,SZFY));

     bench_X11_rep_ent(f,Pt2di(0,0),Pt2di(6,7));
     bench_X11_rep_ent(f,Pt2di(10,30),Pt2di(5*6,7*2));


     bench_X11_rep_inv_ent(f,Pt2dr(0,0),Pt2di(6,7));
     bench_X11_rep_inv_ent(f,Pt2dr(1e-3,1e-3),Pt2di(6,7));
     bench_X11_rep_inv_ent(f,Pt2dr(-1e-3,-1e-3),Pt2di(6,7));
}

main(int,char *)
{
     bench_int_to_real();
     bench_X11_rep();
     bench_div();
}






