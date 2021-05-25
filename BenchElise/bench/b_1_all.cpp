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

All_Memo_counter MC_INIT;


/*********** UTIL ********************************/



INT sigma_inf_x(INT x)
{
    return (x*(x-1)) / 2;
}

INT som_x(INT x1,INT x2)
{
    if (x1 > x2)
       swap(x1,x2);
    return
      abs
      (
         sigma_inf_x(x2) - sigma_inf_x(x1)
      );
}

/*************************************************************/


#include "b_1_0.C"

main(int,char *)
{

     stow_memory_counter(MC_INIT);


     test_Elise_File_Format();

 
     cout << "OK BENCH 1 \n";

     verif_memory_state(MC_INIT);
}






/*
   b_0_1 :
      * test de declartion d'un objet Elise_File_Im 
*/





