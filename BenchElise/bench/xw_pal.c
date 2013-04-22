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

void f0()
{
    Elise_Palette  p1  = Elise_Palette::Gray(100);

    Elise_Set_Of_Palette  sp0(newl(p1));

    Elise_Display d((char *) NULL);

    Elise_Disp_W  W(d,sp0,Pt2di(30,50),Pt2di(512,512),5);

    d.load(sp0);

    copy (W.all_pts(),FX/2,W.out(NAME_GRAY_LEVEL));
getchar();
    copy (W.all_pts(),255-FX/2,W.out(NAME_GRAY_LEVEL));

    

/*
    {
        Elise_Disp_W  W2(d,sp1,Pt2di(30,50),Pt2di(300,300),5);
        Elise_Disp_W  W3(d,sp1,Pt2di(30,50),Pt2di(300,300),5);
    }
*/
    getchar();
}

main(int,char *)
{

     stow_memory_counter(MC_INIT);

     f0();

     verif_memory_state(MC_INIT);
     cout << "OK XW PALETTE 0 \n";
}


