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


Fonc_Num iter_som(Fonc_Num f,INT nb,INT nb_iter)
{
   INT nbv = 2*nb+1;
   REAL av = 0.5;
   REAL sd = sqrt(1/12.0) / nbv;

   for (INT i=0 ; i<nb_iter ; i++)
   {
        av *= square(nbv);
        sd *= nbv * (nbv-1);
        f = rect_som(f,nb);
   }

   return (f-av)/sd;
}


main(int,char *)
{

    Gray_Pal       Pgray(200);
    Disc_Pal       Pdisc  = Disc_Pal::P8COL();
    Elise_Set_Of_Palette SOP(newl(Pgray)+Pdisc);
    Video_Display Ecr((char *) NULL);
    Ecr.load(SOP);

    Video_Win   W  (Ecr,SOP,Pt2di(50,50),Pt2di(512,512));


    INT nb = 10;

    Plot_1d  MyPlot
    (
        W,
        Pdisc(P8COL::blue),
        Pdisc(P8COL::red),
        Interval(0,256),              // x-plotting interval
           newl(PlBox(0,300,512,490)) // box of plotter in w
        +  PlOriY(0.0)                // set Y origin to bottom of box
        +  PlAutoScalY(true) // adapt Y scaling to exact fitting of box
    );
 
     Im1D_INT4 his(256);

for (INT k= 11 ; k<13 ; k++)
{
    copy(his.all_pts(),0,his.out());
    Symb_FNum f = 255*erfcc(iter_som(frandr(),nb,k));
    copy
    (
         W.all_pts(),
         f,
           W.ogray()     
         | (his.histo() << 1).chc(Iconv(f))
    );

    MyPlot.plot(his.in(),newl(PlModePl(Plots::dirac)));
    getchar();
}

    // Gif_File Lena_GF ("../IM_ELISE/GIF/lena.gif");
    // Gif_Im  Lena_GI = Lena_GF.kth_im(0);
    Gif_Im  Lena_GI("../IM_ELISE/GIF/lena.gif");
    Im2D_U_INT1  Lena(512,512);

    copy(W.all_pts(),Lena_GI.in(),W.ogray()|Lena.out());
    getchar();

    copy
    (
          select
          (
             W.all_pts(),
             ((FX+15)%30 < 2) || ((FY+15)%30 < 2)
          ),
          255,
          W.ogray() | Lena.out()
    );
    getchar();

for(INT i = 0; i<2 ; i++)
{
    Fonc_Num r1 = 255*erfcc(iter_som(frandr(),20,5));
    Im2D_REAL4   CY(512,512);
    REAL steep = 120.0/(30+i);
    copy
    (
         W.all_pts(),
         proj_cav(r1,128,steep).v0(),
         CY.out()      | 
         (W.ogray() << Max(0,Min(255,CY.in()/2.0)))
    );

    copy
    (
         W.all_pts(),
         Lena.in(0)
         [
             proj_cav(r1,128,steep).v0(),
             CY.in()[FY,FX]
         ],
         W.ogray() 
    );
    getchar();
}


}

