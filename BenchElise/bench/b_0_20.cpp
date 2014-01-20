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


template <class Type,class TyBase> void bench_histo_2D
    (
        const OperAssocMixte &     op, 
        Type                 *       ,
        TyBase               *       ,
        TyBase                   v0  ,
        Fonc_Num                   fonc,
        Pt2di                      sz,
        INT                        sc  
    )
{
    Im2D<Type,TyBase>  I       (sz.x,sz.y);
    Pt2di              sz_red  ((sz.x+sc-1)/sc,(sz.y+sc-1)/sc);

    ELISE_COPY(I.all_pts(),fonc,I.out());

    // Bench RLE

    {
         Fonc_Num           f2 = (100+FX+FY+Square(FX%(FY+2)))%256;
         Im2D<Type,TyBase>  I2      (sz.x,sz.y);
         Im2D<Type,TyBase>  I3      (sz.x,sz.y);

         ELISE_COPY(I.all_pts(),f2,I2.out());
         ELISE_COPY(I.all_pts(),fonc,I2.oper_ass_eg(op,true));
         ELISE_COPY(I3.all_pts(),op.opf(fonc,f2),I3.out());

         TyBase dif;

         ELISE_COPY(I2.all_pts(),Abs(I2.in()-I3.in()),VMax(dif));
         BENCH_ASSERT(dif < epsilon);
    }

    Type ** d = I.data();

          // - * - * - * - * - * - * - * - * - * - * - *

    {
        Im2D<TyBase,TyBase> Ired1(sz_red.x,sz_red.y);
        Im2D<TyBase,TyBase> Ired2(sz_red.x,sz_red.y);

        ELISE_COPY(Ired1.all_pts(),v0,Ired1.out()|Ired2.out());
        ELISE_COPY
        (
           I.all_pts(),
           I.in(),
           (Ired1.oper_ass_eg(op,true)).chc(Virgule(FX,FY)/sc)
        );

        TyBase ** d2 = Ired2.data();
        for (INT x=0 ; x<sz.x ; x++)
            for (INT y=0 ; y<sz.y ; y++)
                d2[y/sc][x/sc] = 
                    op.opel(d2[y/sc][x/sc],d[y][x]);

        TyBase dif;
        ELISE_COPY(Ired1.all_pts(),Abs(Ired1.in()-Ired2.in()),VMax(dif));
        BENCH_ASSERT(dif < epsilon);
    }

          // - * - * - * - * - * - * - * - * - * - * - *

    {
        Im1D<TyBase,TyBase> Iproj1(sz.x);
        Im1D<TyBase,TyBase> Iproj2(sz.x);

        ELISE_COPY(Iproj1.all_pts(),v0,Iproj1.out()|Iproj2.out());
        ELISE_COPY
        (
           I.all_pts(),
           I.in(),
           (Iproj1.oper_ass_eg(op,true)).chc(FX)
        );


        TyBase * p2 = Iproj2.data();
        for (INT x=0 ; x<sz.x ; x++)
            for (INT y=0 ; y<sz.y ; y++)
                p2[x] = op.opel(p2[x],d[y][x]);

        TyBase dif;
        ELISE_COPY(Iproj1.all_pts(),Abs(Iproj1.in()-Iproj2.in()),VMax(dif));
        BENCH_ASSERT(dif < epsilon);

                 // -------------------------------

        Im1D<TyBase,TyBase> Icum1(sz.x);
        ELISE_COPY
        (
            Icum1.lmr_all_pts(1),
            lin_cumul_ass(op,Iproj1.in()),
            Icum1.out()
        );

        Im1D<TyBase,TyBase> Icum2(sz.x);
        TyBase * c2 = Icum2.data();
        c2[0] = p2[0];
        for (INT i=1 ; i<sz.x ; i++)
            c2[i] = op.opel(c2[i-1],p2[i]);

        ELISE_COPY(Iproj1.all_pts(),Abs(Icum1.in()-Icum2.in()),VMax(dif));
        BENCH_ASSERT(dif < epsilon);
        
    }

          // - * - * - * - * - * - * - * - * - * - * - *

    {
        Im1D<TyBase,TyBase> Iproj1(sz.y);
        Im1D<TyBase,TyBase> Iproj2(sz.y);

        ELISE_COPY(Iproj1.all_pts(),v0,Iproj1.out()|Iproj2.out());
        ELISE_COPY
        (
           I.all_pts(),
           I.in(),
           (Iproj1.oper_ass_eg(op,true)).chc(FY)
        );


        TyBase * p2 = Iproj2.data();
        for (INT x=0 ; x<sz.x ; x++)
            for (INT y=0 ; y<sz.y ; y++)
                p2[y] = op.opel(p2[y],d[y][x]);

        TyBase dif;
        ELISE_COPY(Iproj1.all_pts(),Abs(Iproj1.in()-Iproj2.in()),VMax(dif));
        BENCH_ASSERT(dif < epsilon);
    }
}

void bench_histo_2D()
{
    bench_histo_2D
    (
        OpSum,(INT2 *)0,(INT *)0,
        0,
        (FX+FY+Square(FX)%256),
        Pt2di(100,200),
        5
    );

    bench_histo_2D
    (
        OpMax,(REAL *)0,(REAL *)0,
        -1e9,
        (FX*cos(FX*FY)+sin(Square(FY))),
        Pt2di(50,30),
        8
    );
}

void bench_histo()
{
    bench_histo_2D();

    cout << "OK histo \n";
}



