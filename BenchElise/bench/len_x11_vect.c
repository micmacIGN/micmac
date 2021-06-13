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
*/

#include "general/all.h"


const INT SZX = 512;
const INT SZY = 512;


void zoom_ndg_lena
     (
            Video_Win    w,
            Im2D_U_INT1  i,
            Pt2dr        tr,
            Pt2dr        sc
     )
{
   Video_Win wc = w.chc(tr,sc);
    copy (wc.all_pts(),i.in(255),wc.ogray());
}




void lena_for_ever()
{

    BiCol_Pal      Prb  ( 
                            Elise_colour::black,
                            Elise_colour::red,
                            Elise_colour::blue,
                            12,
                            12
                        );
    Gray_Pal       Pgray  (50);
    Disc_Pal       Pdisc  = Disc_Pal::P8COL();

    Elise_Set_Of_Palette SOP(newl(Prb)+Pgray+Pdisc);
    Video_Display Ecr((char *) NULL);
    Ecr.load(SOP);

    Video_Win   W  (Ecr,SOP,Pt2di(50,50),Pt2di(SZX,SZY));


    Elise_File_Im FLena("../IM_ELISE/lena",Pt2di(SZX,SZY),GenIm::u_int1);
    Im2D_U_INT1 I(SZX,SZY);

    copy(W.all_pts(),FLena.in(),W.ogray()|I.out());
    getchar();

    Col_Pal        red  = Pdisc(P8COL::red);
    Line_St s1 = red;
    for (INT y = 0; y < 300; y+= 50)
         for (INT x = 0; x < 300; x++)
              W.draw_seg(Pt2dr(100+x,100+y),Pt2dr(150+x,200+y),s1);
    getchar();

    for (INT x = 0; x < 256; x+= 20)
        W.draw_seg(Pt2dr(100+x,130),Pt2dr(150+x,230),Line_St(Pgray(x),1+x/20));
    getchar();

    Col_Pal        g129 = Pgray(128);


    Line_St s2 = g129;

}

int  main(int,char **)
{
   ELISE_DEBUG_USER = true;
   All_Memo_counter MC_INIT;
   stow_memory_counter(MC_INIT);


   lena_for_ever();

   verif_memory_state(MC_INIT);
   cout << "OK LENA \n";
   return 0;
}
