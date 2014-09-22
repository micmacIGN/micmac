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



template <class Type,class TypeB> void bench_r2d_lin_cumul
    (
          const OperAssocMixte & op,
          Type *,
          TypeB *,
          Pt2di sz
    )
{
    Im2D<Type,TypeB> I(sz.x,sz.y,(Type)0);
    Im2D_REAL8       R1(sz.x,sz.y,(REAL8)0);
    Im2D_REAL8       R2(sz.x,sz.y,(REAL8)0);

    ELISE_COPY(I.all_pts(),255*frandr(),I.out());

    ELISE_COPY(I.all_pts(),lin_cumul_ass(op,I.in()),R1.out());
    ELISE_COPY(I.lmr_all_pts(Pt2di(1,0)),lin_cumul_ass(op,I.in()),R2.out());

    REAL dif;
    ELISE_COPY (R1.all_pts(),Abs(R1.in()-R2.in()),VMax(dif));
    BENCH_ASSERT(dif<epsilon);
}



template <class Type,class TypeB> void bench_r2d_assoc
    (
          const OperAssocMixte & op,
          Type *,
          TypeB *,
          Pt2di sz,
          INT   x0,
          INT   x1
    )
{
    Im2D<Type,TypeB> I(sz.x,sz.y,(Type)0);
    Im2D_REAL8       R1(sz.x,sz.y,0.0);
    Im2D_REAL8       R2(sz.x,sz.y,0.0);
    Im2D_REAL8       R3(sz.x,sz.y,0.0);

    ELISE_COPY(I.all_pts(),255*frandr(),I.out());

    ELISE_COPY
    (
         I.all_pts(),
         linear_red(op,I.in(),x0,x1),
         R1.out()
    );

    TypeB vdef;
    op.set_neutre(vdef);
    ELISE_COPY
    (
         I.all_pts(),
         rect_red(op,I.in(vdef),Box2di(Pt2di(x0,0),Pt2di(x1,0))),
         R2.out()
    );

    ELISE_COPY
    (
         I.lmr_all_pts(Pt2di(1,0)),
         linear_red(op,I.in(),x0,x1),
         R3.out()
    );

    REAL d12,d23;
    ELISE_COPY 
    (
         R1.all_pts(),
         Virgule(
              Abs(R1.in()-R2.in()),
              Abs(R2.in()-R3.in())
         ),
         Virgule(VMax(d12),VMax(d23))
    );
    BENCH_ASSERT((d12<epsilon)&&(d23<epsilon));
}

void bench_r2d_lin_cumul()
{
      bench_r2d_lin_cumul(OpSum,(REAL4 *)0,(REAL8 *) 0,Pt2di(200,300));
      bench_r2d_lin_cumul(OpMax,(U_INT1 *)0,(INT *) 0,Pt2di(250,320));
      bench_r2d_lin_cumul(OpMin,(INT2 *)0,(INT *) 0,Pt2di(350,220));
}


void bench_r2d_assoc()
{
    bench_r2d_assoc(OpSum,(REAL4 *)0,(REAL *)0,Pt2di(200,300),-3,7);
    bench_r2d_assoc(OpSum,(INT4 *)0,  (INT *)0,Pt2di(150,450),3,17);
    bench_r2d_assoc(OpMin,(INT4 *)0,  (INT *)0,Pt2di(150,450),-17,3);
}


void bench_r2d_shading()
{
    Pt2di sz(120,50);

    Im2D_REAL8       MNT(sz.x,sz.y,0.0);
    Im2D_REAL8       SHAD1(sz.x,sz.y,0.0);
    Im2D_REAL8       SHAD2(sz.x,sz.y,0.0);


    ELISE_COPY(MNT.all_pts(),frandr(),MNT.out());

    ELISE_COPY
    (
        MNT.all_pts(),
        binary_shading(MNT.in(),1.0),
        SHAD1.out()
    );

    ELISE_COPY
    (
        MNT.lmr_all_pts(Pt2di(1,0)),
        binary_shading(MNT.in(),1.0),
        SHAD2.out()
    );

    REAL dif;

    ELISE_COPY (MNT.all_pts(),Abs(SHAD1.in()-SHAD2.in()),VMax(dif));
    BENCH_ASSERT(dif<epsilon);


    ELISE_COPY
    (
        MNT.all_pts(),
        gray_level_shading(MNT.in()),
        SHAD1.out()
    );

    ELISE_COPY
    (
        MNT.lmr_all_pts(Pt2di(1,0)),
        gray_level_shading(MNT.in()),
        SHAD2.out()
    );


    ELISE_COPY (MNT.all_pts(),Abs(SHAD1.in()-SHAD2.in()),VMax(dif));
    BENCH_ASSERT(dif<epsilon);
}



void bench_r2d_proj()
{
    Pt2di sz(120,50);

    Im2D_REAL8       MNT(sz.x,sz.y,0.0);
    Im2D_REAL8       X1(sz.x,sz.y,0.0);
    Im2D_REAL8       Y1(sz.x,sz.y,0.0);
    Im2D_REAL8       X2(sz.x,sz.y,0.0);
    Im2D_REAL8       Y2(sz.x,sz.y,0.0);


    ELISE_COPY(MNT.all_pts(),frandr(),MNT.out());

    ELISE_COPY
    (
        MNT.all_pts(),
        proj_cav(MNT.in(),0.5,2.0),
        Virgule(X1.out(),Y1.out())
    );

    ELISE_COPY
    (
        MNT.lmr_all_pts(Pt2di(1,0)),
        proj_cav(MNT.in(),0.5,2.0),
        Virgule(X2.out(),Y2.out())
    );

    REAL dif;

    ELISE_COPY 
    (
         MNT.all_pts(),
         Max(Abs(X1.in()-X2.in()),Abs(Y1.in()-Y2.in())),
         VMax(dif)
    );
    BENCH_ASSERT(dif<epsilon);


    ELISE_COPY
    (
        MNT.all_pts(),
        (2*PI)*phasis_auto_stereogramme(MNT.in(),10.5,2.0),
        X1.out()
    );

    ELISE_COPY
    (
        MNT.lmr_all_pts(Pt2di(1,0)),
        (2*PI)*phasis_auto_stereogramme(MNT.in(),10.5,2.0),
        X2.out()
    );

    ELISE_COPY 
    (
         MNT.all_pts(),
         Max
         (
              Abs(cos(X1.in())-cos(X2.in())),
              Abs(sin(X1.in())-sin(X2.in()))
         ),
         VMax(dif)
    );

    BENCH_ASSERT(dif<epsilon);
}



void bench_r2d_adapt_lin()
{
    bench_r2d_proj();
    bench_r2d_shading();
    bench_r2d_lin_cumul();
    bench_r2d_assoc();
    cout << "OK bench_r2d_adapt_lin \n";
}






