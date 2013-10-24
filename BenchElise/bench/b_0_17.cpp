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


#if (ELISE_X11 && USE_X11)


void test_clip_seg(Pt2di sz,Pt2dr p0,Pt2dr p1)
{
    Disc_Pal       Pdisc  = Disc_Pal::P8COL();

    Elise_Set_Of_Palette SOP(NewLElPal(Pdisc));
    Video_Display Ecr((char *) NULL);
    Ecr.load(SOP);

    Video_Win   W  (Ecr,SOP,Pt2di(50,50),sz);


    Col_Pal red   = Pdisc(P8COL::red);
    Col_Pal green = Pdisc(P8COL::green);
    Col_Pal black = Pdisc(P8COL::black);

    ELISE_COPY(W.all_pts(),P8COL::white,W.odisc());
    W.draw_rect(p0,p1,red);


    Box2dr b (p0,p1);
    for (int k = 0; k < 5000 ; k++)
    {
         Pt2dr q1 (sz.x*(1+sin(1.1*k)),sz.y*(1+sin(2.1*k)));
         Pt2dr q2 (sz.x*(1+sin(1.2*k)),sz.y*(1+sin(2.3*k)));
         W.draw_seg(q1,q2,green);

         Seg2d cl = Seg2d(q1,q2).clip(b);
         if (! cl.empty())
            W.draw_seg(cl.p0(),cl.p1(),black);
    }
}

void test_box_seg()
{
     static bool deja = false;
     if (deja)
        return;
   
     deja = true;
     test_clip_seg(Pt2di(450,450),Pt2dr(150,150),Pt2dr(300,300));
}

#else
void test_box_seg()
{
}
#endif
