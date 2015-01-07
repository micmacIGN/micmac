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

int main(int,char**)
{
     int SZX = 200;
     int SZY = 150;
     int ZOOM = 3;


     Gray_Pal     Pgr(100);
     Disc_Pal Pdisc = Disc_Pal::P8COL();
     Elise_Set_Of_Palette SOP(newl(Pdisc)+Pgr);
     Video_Display Ecr((char *) NULL);
     Ecr.load(SOP);
     Col_Pal red = Pdisc(P8COL::red);
     Col_Pal blue = Pdisc(P8COL::blue);
     Col_Pal green = Pdisc(P8COL::green);
     Col_Pal yellow = Pdisc(P8COL::yellow);
     Col_Pal cyan = Pdisc(P8COL::cyan);

     Video_Win   W  (Ecr,SOP,Pt2di(50,50),Pt2di(SZX,SZY)*ZOOM);
     W = W.chc(Pt2dr(-0.50,-0.50),Pt2di(ZOOM,ZOOM));

     Im2D_U_INT1 IDist(SZX,SZY,P8COL::blue);
     Im2D_U_INT1 IVein(SZX,SZY,P8COL::blue);
     Im2D_U_INT2 ISom(SZX,SZY,P8COL::blue);

     Im2D_U_INT1 I0(SZX,SZY);


    // Tiff_Im Ftif("../IM_ELISE/cci.tif");
    //  Pt2di tr (500,0);
    Tiff_Im Ftif("../IM_ELISE/TIFF/ccitt_1.tif");
    Pt2di tr (300,700);
    Fonc_Num f0 =  trans(Ftif.in(0),tr);

    for (int i = 0; i<1; i++)
    {

        ELISE_COPY
        (
           IDist.all_pts(),
           f0,
           IDist.out()|W.out(Pdisc) | I0.out()
        );

        ELISE_COPY
        (
           select(IDist.all_pts(),frandr()<0.01),
           1,
           IDist.out()|W.out(Pdisc) | I0.out()
        );
        ELISE_COPY
        (
           IDist.border(2),
           0,
           IDist.out()|W.out(Pdisc) | I0.out()
        );



        Liste_Pts_U_INT2 l =
        Skeleton
        (
             IVein,
             IDist,
               newl(AngSkel(6.14))
             + SurfSkel(9)
             + SkelOfDisk(false)
             + ProlgtSkel(false)
             + ResultSkel(true)
        );

/*
        Skeleton
        (
                IVein.data(),
                IDist.data(),
                IDist.tx(),
                IDist.ty(),
                5,
                3.14,
                true,
                true,
                true,
                0
        );
*/

        ELISE_COPY(l.all_pts(),P8COL::red,W.out(Pdisc));

        ELISE_COPY(IVein.all_pts(),IVein.in(),W.out_graph(blue,false));
        getchar();
    }
}

