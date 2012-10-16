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


#include <fstream.h>


void Data_Elise_PS_Disp::test_ps()
{
   
    Disc_Pal  Pdisc = Disc_Pal::P8COL();
    Gray_Pal  Pgr(50);
    Circ_Pal  Pcirc = Circ_Pal::PCIRC6(20);

    Pt2di SZ(512,512);

    Lin1Col_Pal   BluePal
                  (
                      Elise_colour::rgb(0,0,0.25),
                      Elise_colour::rgb(0.5,0.75,1),
                      30
                  );
     RGB_Pal     Prgb(3,3,3);
     BiCol_Pal   Prb  (
                          Elise_colour::black,
                          Elise_colour::red,
                          Elise_colour::blue,
                          2,
                          2
                      );
      


    Elise_Set_Of_Palette SOP(newl(BluePal)+Pgr+Pcirc+Pdisc+Prgb+Prb);
    PS_Display disp("TMP/test.ps","Mon beau fichier ps",SOP);

    PS_Window  Wps0
               (
                   disp,
                   SZ,
                   Pt2dr(2.0,5.0),
                   Pt2dr(18.0,21.0)
               );

/*
    PS_Window  W1
               (
                   disp,
                   Pt2di(512,512),
                   Pt2dr(10.0,13.0),
                   Pt2dr(18.0,21.0)
               );
*/

   Video_Display Ecr((char *) NULL);
   Ecr.load(SOP);
   Video_Win   Wv  (Ecr,SOP,Pt2di(50,50),Pt2di(SZ.x,SZ.y));


   Tiff_Im     Flena("../IM_ELISE/TIFF/lena.tif");
   Im2D_U_INT1 Lena (512,512);
   ELISE_COPY(Lena.all_pts(),8*(Flena.in()/8),Lena.out());

   Wv.fill_rect(Pt2di(0,0),Pt2di(512,512),Prgb(255,255,0));
   Wps0.fill_rect(Pt2di(0,0),Pt2di(512,512),Prgb(255,255,0));

   Pt2dr TR (100.0,50.0);
   Pt2dr SC (2.0,2.0);

   Video_Win  Wv_ch  = Wv.chc(TR,SC);
   PS_Window  Wps_ch0 = Wps0.chc(TR,SC);
   
   ELISE_COPY
   (
         Wv_ch.all_pts(),
         Lena.in(0),
         Wv_ch.out(Pgr) | Wps_ch0.out(Pgr)
   );

   ELISE_COPY
   (
        select(Wv_ch.all_pts(),Lena.in(0)> 128),
        P8COL::red,
        Wv_ch.out(Pdisc) | Wps_ch0.out(Pdisc)
   );


   Wv.draw_rect(Pt2di(200,200),Pt2di(300,300),Pdisc(P8COL::green));
   Wv_ch.draw_rect(Pt2di(200,200),Pt2di(300,300),Pdisc(P8COL::green));


   El_Window We = Wps0;
   El_Window We_chc = We.chc(TR,SC);

   Wps0.draw_rect(Pt2di(200,200),Pt2di(300,300),Pdisc(P8COL::green));
   Wps_ch0.draw_rect(Pt2di(200,200),Pt2di(300,300),Pdisc(P8COL::green));

   We.draw_rect(Pt2di(200,200),Pt2di(300,300),Pdisc(P8COL::blue));
   We_chc.draw_rect(Pt2di(200,200),Pt2di(300,300),Pdisc(P8COL::blue));
/*

   ELISE_COPY
   (
      Wps0.all_pts(),
      Lena.in(0),
      Wps0.ogray() | Wv.ogray() | (W1.out(Pcirc) << Lena.in()[(FY,FX)])
   );

   for (INT x = -60 ; x<50+SZ.x ; x+= 20)
       for (INT y = -80 ; y<80+SZ.y ; y+= 40)
       {
           Wps0.draw_rect(Pt2dr(x,y),Pt2dr(x+15,y+8),Prgb(255,0,0));
       }

   ELISE_COPY
   (
      disc(Pt2di(256,256),30),
      255-Lena.in(0),
      Wps0.ogray() | Wv.ogray()
   );

   Wps0.fill_rect(Pt2di(0,0),Pt2di(1,1),Prgb(256,256,0));
   Wps0.fill_rect(SZ-Pt2di(1,1),SZ,Prgb(256,256,0));
*/


}




int  main(int,char **)
{
    All_Memo_counter MC_INIT;
    stow_memory_counter(MC_INIT);
    Data_Elise_PS_Disp::test_ps();
    verif_memory_state(MC_INIT);
}

/*

 struct tm *localtime(const time_t *timep);
  time_t time(0);
*/







