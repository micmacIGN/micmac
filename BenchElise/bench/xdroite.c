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

const INT SZX = 100;
const INT SZY = 100;
const INT ZOOM = 5;

Fen_X11 F (Pt2di(50,50),Pt2di(SZX*ZOOM,SZY*ZOOM));
Im2D_U_INT1 Im(SZX,SZY);

void show()
{
     copy (F.all_pts(),Im.in()[FX/ZOOM,FY/ZOOM],F.disc());
     getchar();
}

int main(int,char *)
{
    copy(Im.all_pts(),0,Im.out());


    copy
    (
        line(Pt2di(0,0),Pt2di(99,99)),
        BLEU,
        Im.out()
    );
    show();

    copy
    (
        line_4c(Pt2di(10,0),Pt2di(89,99)),
        VERT,
        Im.out()
    );
    show();


    copy
    (
        line
        ( 
             newl(Pt2di(50,30)) + Pt2di(70,50) + Pt2di(50,70) + Pt2di(30,50)
        ),
        NOIR,
        Im.out()
    );
    show();

    copy
    (
        line
        ( 
             newl(Pt2di(50,30)) + Pt2di(70,50) + Pt2di(50,70) + Pt2di(30,50),
             true
        ),
        NOIR,
        Im.out()
    );
    show();


    copy
    (
        line_4c
        ( 
             newl(Pt2di(50,30)) + Pt2di(70,50) + Pt2di(50,70) + Pt2di(30,50)
        ),
        ROUGE,
        Im.out()
    );
    show();


    return 0;
}


