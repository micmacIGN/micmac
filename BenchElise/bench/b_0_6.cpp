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


/***************************************************/

void test_0_fonc_chc(Pt2di sz)
{
     Im2D_U_INT1 b1(sz.x,sz.y,0);
     Im2D_U_INT1 b2(sz.y,sz.x,0);

     ELISE_COPY
     (
         b1.all_pts(),
         FX%5 +FY%8 + (FX*FY) %17,
         b1.out()
     );

     ELISE_COPY
     (
         b2.all_pts(),
         (b1.in()[Virgule(FY,FX)]),
         b2.out()
     );

     for(int y=0 ; y<sz.y ; y++)
         for(int x=0 ; x<sz.x ; x++)
            if (b1.data()[y][x] != b2.data()[x][y])
            {
               cout << "PB dans test_0_fonc_chc " 
                    << " x ,y : " << x << " " << y
                    << " b1 " << b1.data()[y][x]
                    << " b2 " << b1.data()[x][y] << "\n";
               exit(0);
            }

     cout << "FIN test_0_fonc_chc\n";
}
void test_0_fonc_chc()
{
     test_0_fonc_chc(Pt2di(134,76));
     test_0_fonc_chc(Pt2di(176,224));
}

/***************************************************/


void test_flux_chc(Pt2di sz)
{
     Im2D_U_INT1 b1(sz.x,sz.y,0);
     Im2D_U_INT1 b2(sz.x,sz.y,0);
     Im2D_U_INT1 b3(sz.x,sz.y,0);
     

     ELISE_COPY
     (
         rectangle(Pt2di(0,0),(Pt2di(1,1)+Pt2di(sz.y,sz.x))/2).chc(Virgule(FY,FX)*2),
         (FX+FY)%256,
         b1.out()
     );

     ELISE_COPY
     (
         select(b1.all_pts(),(FX%2 == 0) && (FY%2 == 0)),
         (FX+FY)%256,
         b2.out()
     );


     ELISE_COPY
     (
         select
         (
            rectangle
            (
                    Pt2di(0,0),
                    (Pt2di(1,1)+Pt2di(sz.y,sz.x))/2
            ),
            1
         ).chc(Virgule(FY,FX)*2),

         (FX+FY)%256,

         b3.out()
     );


     INT dif_12,dif_23;

     ELISE_COPY
     (
        b1.all_pts(),
        Virgule((b1.in() != b2.in()), (b2.in() != b3.in())),
        Virgule(sigma(dif_12), sigma(dif_23))

     );

     BENCH_ASSERT((dif_12==0) && (dif_23 == 0));

}

void test_flux_chc()
{
     test_flux_chc(Pt2di(133,187));
     test_flux_chc(Pt2di(134,184));
}
/***************************************************/

void test_chc()
{
    All_Memo_counter MC_INIT;
    stow_memory_counter(MC_INIT);
    {
        test_0_fonc_chc();
        test_flux_chc();

        cout << "OK chang coord \n";
     }
     verif_memory_state(MC_INIT);
}





