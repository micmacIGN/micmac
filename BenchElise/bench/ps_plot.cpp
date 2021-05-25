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


void lena_for_ever()
{

    Gray_Pal       Pgray  (30);
    Disc_Pal       Pdisc  = Disc_Pal::P8COL();

    Elise_Set_Of_Palette SOP(newl(Pgray)+Pdisc);
    Video_Display Ecr((char *) NULL);
    Ecr.load(SOP);

    Video_Win   Wv  (Ecr,SOP,Pt2di(50,50),Pt2di(SZX,SZY));


    PS_Display disp("TMP/test.ps","Mon beau fichier ps",SOP);

    Pt2di NBW(4,20);
    Pt2di sz(200,100);
    Mat_PS_Window mps(disp,sz,Pt2dr(2.0,2.0),NBW,Pt2dr(0.5,0.1));

    for (int x = 0; x < NBW.x ; x++)
        for (int y = 0; y < NBW.y ; y++)
        {
            mps(x,y).fill_rect(Pt2di(0,0),sz,Pgray((y*255)/( NBW.y-1)));
            mps(x,y).draw_rect(Pt2di(0,0),sz,Pdisc(P8COL::red));
        }
/*
    PS_Window  Wps = disp.w_centered_max(Pt2di(SZX,SZY),Pt2dr(0.0,8.0));


    Elise_File_Im FLena("../IM_ELISE/lena",Pt2di(SZX,SZY),GenIm::u_int1);
    Im2D_U_INT1 I(SZX,SZY);



    Col_Pal        red    = Pdisc(P8COL::red);
    Col_Pal        blue   = Pdisc(P8COL::blue);
    Col_Pal        green  = Pdisc(P8COL::green);
    Col_Pal        black  = Pdisc(P8COL::black);
    Col_Pal        cyan   = Pdisc(P8COL::cyan);
    Col_Pal        white  = Pdisc(P8COL::white);
    Col_Pal        yellow  = Pdisc(P8COL::yellow);


    El_Window W = Wv|Wps;

    ELISE_COPY
    (
       Wps.all_pts(),
       32*(FLena.in()/32),
       W.ogray()|I.out()
    );
    W.fill_rect(Pt2dr(100,200),Pt2dr(400,300),yellow);


    Line_St s1     (red,3);
    Line_St s2     (green,2);

    Plot_1d  Plot1  (W,s1,s2,Interval(-10,10),newl(PlBox(50,50,300,400)));


    Plot1.show_axes();

    Plot1.show_axes
    (
              newl(PlBox(150,150,250,250)) + PlOriY(0.2)
            + PlAxeSty(s2)
    );


    Plot_1d  Plot2(W,s1,s2,Interval(-100,100));


    Plot1.show_box
    (
            newl(PlBox(150,150,250,250))  + PlBoxSty(cyan,2)
    );


    Plot2.set
    (
              newl(PlBox(20,200,400,400)) 
            + PlOriY(0.4)
            + PlAxeSty(Line_St(Pdisc(P8COL::yellow),1))
            + PlBoxSty(Line_St(black,2))
            + PlClearSty(white)
    );
    Plot2.show_axes();
    Plot2.show_box();


    Plot2.plot(10*sin(FX/4.0));

    Plot2.plot
    (
         10*sin(FX/4.0),
            newl(PlIntervBoxX(-50,50))
         +  PlotLinSty(red,2) 
    );

    Plot2.plot
    (
         10*sin(FX/4.0),
           newl( PlotLinSty(blue,2) )
         + PlIntervPlotX(-30,70)
         + PlAutoScalY(true)
         + PlShAxes(true)
         + PlAxeSty(cyan,3)
    );

    
    Plot2.plot
    (
         10*(1.2+sin(FX/4.0)),
           newl( PlotLinSty(red,2) )
         + PlAutoScalOriY(true)
         + PlShAxes(true)
         + PlAxeSty(cyan,2)
         + PlAutoClear(true)
    );



    Plot2.clear(newl(PlClearSty(Pgray(196))));

    Plot2.plot
    ( 
         50*cos(FX/9.0),
         newl(PlotLinSty(Pdisc(P8COL::red)))
       + PlClipY(false) + PlStepX(1.0)
    );


    Plot2.plot
    (
         70*sin(square(FX) / 500.0),
         newl(PlotLinSty(Pdisc(P8COL::blue)))
       + PlClipY(true) + PlStepX(0.15)
    );
    
    
    Plot2.set ( newl(PlBox(20,0,400,200)) );
    Plot2.set(newl(PlIntervBoxX(-20,20)));
    Plot2.clear();

    Plot2.plot
    (
         10 * cos(FX/4.0),
         newl(PlotFilSty(green))
        + PlClipY(false)
        + PlModePl(Plots::fill_box)
    );

     Plot2.plot
     (
         10 * cos(FX/4.0),
         newl(PlotFilSty(red))
        + PlClipY(true)
        + PlModePl(Plots::fill_box)
     );



     Plot2.plot
     (
         10 * cos(FX/4.0),
         newl(PlotLinSty(black,2))
        + PlClipY(true)
        + PlModePl(Plots::draw_box)
        + PlShAxes(true)
        + PlAxeSty(Line_St(Pdisc(P8COL::blue),2))
     );

cout << "aaaaaaaaaa\n";

getchar();
*/
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
