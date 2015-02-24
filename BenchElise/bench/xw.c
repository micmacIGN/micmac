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



/*
       Definition de Mandelbrot :
         F0(z) = 0
         Fk+1(z) = z +square(Fk(z))
         Ensemble de point tels que ca reste borne.
*/

#include "general/all.h"
#include "private/all.h"



const INT SZX = 512;
const INT SZY = 512-64;

INT SZ_FICH[2] = {512,512};


void verif(Fen_X11 w,Im2D_U_INT1 b1,Im2D_U_INT1 bw)
{
    copy(bw.all_pts(),w.in(),bw.out());
    INT sdiff;
    copy
    (    bw.all_pts(),
         abs(b1.in()-bw.in()) ,
         sigma(sdiff)
    );

    BENCH_ASSERT(sdiff== 0);
}

int  main(int,char **)
{

   ELISE_DEBUG_USER = true;
   All_Memo_counter MC_INIT;
   stow_memory_counter(MC_INIT);

   {
       Fen_X11 W (Pt2di(50,50),Pt2di(SZX,SZY));
       Elise_File_Im FLena("../IM_ELISE/lena",2,SZ_FICH,GenIm::u_int1,1,0);

       Im2D_U_INT1 ILena(SZX,SZY);

       Im2D_U_INT1 B1(SZX,SZY);
       Im2D_U_INT1 Bw(SZX,SZY);

       Im1D_U_INT1 Lndg = lut_ndg();
       Im1D_U_INT1 Ldisc = lut_disc();


       copy(B1.all_pts(),FLena.in(),ILena.out());

       copy(B1.all_pts(),ILena.in(),W.ndg());
       copy(B1.all_pts(),Lndg.in()[ILena.in()],B1.out());
       verif(W,B1,Bw);
       
       copy(rectangle(Pt2di(0,0),Pt2di(195,106)),ILena.in()/32,W.disc());
       copy(rectangle(Pt2di(0,0),Pt2di(195,106)),Ldisc.in()[ILena.in()/32],B1.out());
       verif(W,B1,Bw);


       {
          INT r = Ldisc.data()[ROUGE];
          for (int i = 0 ; i < 10 ; i++)
          {
              copy(line(Pt2di(0,0),Pt2di(i*40,300)),ROUGE,W.disc());
              copy(line(Pt2di(0,0),Pt2di(i*40,300)),r,B1.out());
          }
       }
       verif(W,B1,Bw);

       {
          INT v = Ldisc.data()[VERT];
          copy(select(B1.all_pts(),ILena.in()>200),VERT,W.disc());
          copy(select(B1.all_pts(),ILena.in()>200),v,B1.out());
       }
       verif(W,B1,Bw);


       copy(select(B1.all_pts(),(FX+FY)%2),255-ILena.in(),W.ndg());
       copy(select(B1.all_pts(),(FX+FY)%2),Lndg.in()[255-ILena.in()],B1.out());
       verif(W,B1,Bw);


       cout << "FIN BENCH \n";
       getchar();

   }

    char ** m = NEW_MATRICE(Pt2di(-2,6),Pt2di(120,34),char);
    DELETE_MATRICE(m,Pt2di(-2,6),Pt2di(120,34));


   verif_memory_state(MC_INIT);
   return 0;
}










