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
#include "private/all.h"


void bench_i1_eq_i2(Im2D_U_INT1 i1,Im2D_U_INT1 i2)
{
   INT nb_dif;

   ELISE_COPY(i1.all_pts(),Abs(i1.in()-i2.in()),sigma(nb_dif));

   BENCH_ASSERT(nb_dif ==0);
}


//*************************************************************************

#include "morpho_0.cpp"
#include "morpho_1.cpp"

const INT SZX = 512;
const INT SZY = 512;
INT SXY[2] = {SZX,SZY};

void lena_for_ever()
{
     Video_Display Ecr((char *) NULL);
     Gray_Pal       Pgray  (30);
     Disc_Pal       Pdisc  = Disc_Pal::P8COL();
     Elise_Set_Of_Palette    SOP(newl(Pgray)+Pdisc);


     Ecr.load(SOP);

     Video_Win W (Ecr,SOP,Pt2di(50,50),Pt2di(SZX,SZY));
     Elise_File_Im FLena("../IM_ELISE/lena",2,SXY,GenIm::u_int1,1,0);
     Im2D_U_INT1 I(SZX,SZY);

     ELISE_COPY
     (
           W.all_pts(),
           FLena.in(),
           W.ogray() | I.out()
     );

      bench_dilate_simple(W,I);
      bench_zonec_simple(W,I);
}




int  main(int,char **)
{
   ELISE_DEBUG_USER = true;
   All_Memo_counter MC_INIT;
   stow_memory_counter(MC_INIT);

   lena_for_ever();


   getchar();

   verif_memory_state(MC_INIT);
   return 0;
}
