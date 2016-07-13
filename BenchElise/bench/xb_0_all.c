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


main(int,char *)
{
     stow_memory_counter(MC_INIT);

     Fen_X11 F (Pt2di(50,50),Pt2di(128,128));

     getchar();
     copy(rectangle(Pt2di(-50,-50),Pt2di(500,500)),ROUGE,F.disc());


     copy(F.all_pts(),Iconv(FX+128.0),F.ndg());
     getchar();


     copy(F.all_pts(),FY+FX,F.ndg());
     getchar();

     copy(F.all_pts(),Max(FX,FY),F.ndg());
     getchar();

     copy(F.all_pts(),Min(FX,FY),F.ndg());
     getchar();






     copy(F.all_pts(),FY,F.ndg());
     copy(rectangle(Pt2di(0,0),Pt2di(100,100)),JAUNE,F.disc());

     getchar();
     verif_memory_state(MC_INIT);

     cout << "OK BENCH X11  :  0 \n";


     Im2D<U_INT1,INT> b (30,30);
     cout << "SZ, DIM = " << b.sz_el() << " " << b.dim() << "\n";
     GenIm * gb = &b;
     cout << "SZ, DIM = " << gb->sz_el() << " " << gb->dim() << "\n";

     GenIm2D *gb2 = &b;
     cout << "SZ, DIM = " << gb2->sz_el() << " " << gb2->dim() << "\n";

     copy(gb->all_pts(),0,F.ndg());


}
