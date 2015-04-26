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

const INT SZX = 600;
const INT SZY = 600;

const INT NB_PTS = 1000;

INT rani(INT sz)
{
    return (INT) (10 + NRrandom3()*(sz-20));
}

Pt2di * tab_pts(Col_Pal col,Video_Win W)
{
     Pt2di larg(2,2);

     Pt2di * res = NEW_TAB(NB_PTS,Pt2di);

     for (INT k=0 ; k<NB_PTS ; k++)
     {
         res[k] = Pt2di(rani(SZX),rani(SZY));
         W.fill_rect(res[k]-larg,res[k]+larg,col);
     }

     return res;
}

void lena_for_ever()
{

    Gray_Pal       Pgray  (80);
    Disc_Pal       Pdisc  = Disc_Pal::P8COL();
    Circ_Pal       Pcirc  = Circ_Pal::PCIRC6(70);


    Elise_Set_Of_Palette SOP(newl(Pcirc)+Pgray+Pdisc);
    Video_Display Ecr((char *) NULL);
    Ecr.load(SOP);

    Video_Win   W (Ecr,SOP,Pt2di(50,50),Pt2di(SZX,SZY));


    Pt2di * tr = tab_pts(Pdisc(P8COL::red),W);
    Pt2di * tb = tab_pts(Pdisc(P8COL::blue),W);

    Im2D_INT4 cost(NB_PTS,NB_PTS);
    INT ** c = cost.data();

    for (INT r =0; r <NB_PTS; r++)
         for (INT b =0; b <NB_PTS; b++)
             c[r][b] = (INT) (1000*euclid(tr[r]-tb[b]));

    Im1D_INT4 aff = hongrois(cost);
    INT * a = aff.data();

    for (INT k=0; k<NB_PTS; k++)
    {
        printf("%d %d\n",k,a[k]);
        Pt2di pr = tr[a[k]];
        Pt2di pb = tb[k];
/*
        Pt2di pr = tr[k];
        Pt2di pb = tb[a[k]];
*/
        W.draw_seg(pr,pb,Pdisc(P8COL::black));
    }


    DELETE_TAB(tr);
    DELETE_TAB(tb);

    getchar();
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
