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



template <class Type,class TypeBase> void  test_interpole_im1d
                     (INT tx,Type *,TypeBase *)
{

    Im1D<Type,TypeBase> b1(tx);

    ELISE_COPY(b1.all_pts(),(33-3*FX),b1.out());

    REAL fx = 2.1;
    REAL dx = 3.1;


    REAL dif;

    // VERIF INTERPOLATION SUR LES BITMAPS 
    ELISE_COPY
    (
        b1.all_pts(),
        Abs
        (
               b1.in()[FX/fx+dx]       
          -   (33 - 3*(FX/fx+dx))
        ),
        VMax(dif)
    );

    BENCH_ASSERT(dif < epsilon);


    // VERIF CLIP  POUR LES BITMAPS  AVEC PTS REELS

    ELISE_COPY(b1.all_pts(),1,b1.out());

    REAL som;
    ELISE_COPY
    (
        rectangle(-10,tx+20),
        b1.in(2.0)[FX+0.5],
       sigma(som)
    );

    dif = fabs(som -(2*(tx+30)-(tx-1)));

    BENCH_ASSERT(dif < epsilon);
}


template <class Type,class TypeBase> void  test_interpole_im2d
                     (INT tx,INT ty,bool bilin,Type *,TypeBase *)
{

    Im2D<Type,TypeBase> b1(tx,ty);

    Fonc_Num fbil = (bilin) ? (FX*FY) : Fonc_Num(0);

    ELISE_COPY(b1.all_pts(),(2+FX+2*FY+fbil),b1.out());

    REAL fx = 2.1;
    REAL dx = 3.1;

    REAL fy = 1.4;
    REAL dy = 2.1;

    REAL dif;

    // VERIF INTERPOLATION SUR LES BITMAPS 

    Fonc_Num f2bil = (bilin) ?  ((FX/fx+dx)* (FY/fy+dy)) : Fonc_Num(0);
    ELISE_COPY
    (
        rectangle(Pt2di(0,0),Pt2di(tx,ty)),
        Abs
        (
               b1.in()[Virgule(FX/fx+dx,FY/fy+dy)]       
          -   (2 + (FX/fx+dx) + 2 * (FY/fy+dy) + f2bil)
        ),
        VMax(dif)
    );

    BENCH_ASSERT(dif < epsilon);

    // VERIF FONCTIONS COORDONNEES   POUR LES FLUX REELS
    ELISE_COPY
    (
        rectangle(Pt2di(0,0),Pt2di(tx,ty)),
        Abs
        (
               b1.in()[Virgule(FX/fx+dx,FY/fy+dy)]       
          -   (2 +FX + 2 *FY +fbil)[Virgule(FX/fx+dx,FY/fy+dy)]
        ),
        VMax(dif)
    );
    BENCH_ASSERT(dif < epsilon);


    // VERIF CLIP  POUR LES BITMAPS  AVEC PTS REELS

    ELISE_COPY(b1.all_pts(),1,b1.out());

    REAL som;
    ELISE_COPY
    (
        rectangle(Pt2di(-10,-20),Pt2di(tx+10,ty+20)),
        (b1.in(2.0)[Virgule(FX+0.5,FY+0.5)]),
       sigma(som)
    );

    dif = fabs(som -((tx+20) * (ty + 40) * 2 - (tx-1) * (ty-1)));

    BENCH_ASSERT(dif < epsilon);
}




void test_im_mode_reel()
{
     test_interpole_im2d(50,50,false,(U_INT1 *) 0,(INT *) 0);
     test_interpole_im2d(250,150,false,(INT2 *) 0,(INT *) 0);
     test_interpole_im2d(50,50,true,(U_INT2 *) 0,(INT *) 0);
     test_interpole_im2d(50,50,true,(REAL4 *) 0,(REAL *) 0);

     test_interpole_im1d(30,(INT1 *) 0,(INT *) 0);
     test_interpole_im1d(50,(INT2 *) 0,(INT *) 0);
     test_interpole_im1d(50,(INT *) 0,(INT *) 0);
     test_interpole_im1d(50,(REAL4 *) 0,(REAL *) 0);
     test_interpole_im1d(50,(REAL *) 0,(REAL *) 0);
}
