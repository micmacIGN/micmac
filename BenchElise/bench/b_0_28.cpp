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


// Une petite verif a la main

void bench_red_op_ass_0 (Pt2di sz)
{
    Im2D_U_INT1 ISEL(sz.x,sz.y,0);

    ELISE_COPY(ISEL.all_pts(),frandr()>0.5,ISEL.out());
    ELISE_COPY(ISEL.border(2),0,ISEL.out());

    Im2D_U_INT1 IVOIS(sz.x,sz.y,0);
    ELISE_COPY(IVOIS.all_pts(),frandr()>0.5,IVOIS.out());

    Im2D_U_INT1 IFONC(sz.x,sz.y,0);
    ELISE_COPY(IFONC.all_pts(),255*frandr(),IFONC.out());


    Im2D_INT4  IV0(sz.x,sz.y,0);
    ELISE_COPY(IV0.all_pts(),0,IV0.out());

    U_INT1 ** sel   = ISEL.data();
    U_INT1 ** vois  = IVOIS.data();
    U_INT1 ** fonc  = IFONC.data();
    INT4 ** v0      = IV0.data();

    for (INT x=0; x<sz.x ; x++)
        for (INT y=0; y<sz.y ; y++)
        {
            v0[y][x] = 0;
            if (sel[y][x])
            {
                for (INT xv = x-1; xv <= x+1; xv++)
                    for (INT yv = y-1; yv <= y+1; yv++)
                        if (vois[yv][xv] &&((xv!=x)||(yv!=y)))
                           v0[y][x] += fonc[yv][xv];
            }
       }


    Im2D_INT4  IV1(sz.x,sz.y,0);
    ELISE_COPY(IV1.all_pts(),0,IV1.out());

    Neigh_Rel v8 = Neighbourhood::v8();
    ELISE_COPY
    (
          select(ISEL.all_pts(),ISEL.in()),
          sel_func(v8,IVOIS.in()).red_sum(IFONC.in()),
          IV1.out()
    );

    Im2D_INT4  IV2(sz.x,sz.y,0);
    ELISE_COPY(IV2.all_pts(),0,IV2.out());

    Fonc_Num f = 0;
    for (INT i =0; i<8; i++)
    {
        Pt2di p = TAB_8_NEIGH[i];
        f = f+trans(IVOIS.in()*IFONC.in(),p);
    }
    ELISE_COPY(select(ISEL.all_pts(),ISEL.in()),f,IV2.out());

    INT dif_01,dif_12,dif_02;
    ELISE_COPY(IV1.all_pts(),Abs(IV1.in()-IV0.in()),sigma(dif_01));
    ELISE_COPY(IV1.all_pts(),Abs(IV2.in()-IV0.in()),sigma(dif_02));
    ELISE_COPY(IV1.all_pts(),Abs(IV2.in()-IV1.in()),sigma(dif_12));

    BENCH_ASSERT
    (
            (dif_01==0)
         && (dif_12==0)
         && (dif_02==0)
    );
}


void bench_red_op_ass ()
{

    INT dif;
    Neigh_Rel v4 = Neighbourhood::v4();



//------- avec fonction et voisinage constant
    ELISE_COPY
    (
          select(rectangle(Pt2di(0,0),Pt2di(20,30)),1),
          Abs(v4.red_sum(1)-4),
          sigma(dif)
    );
    BENCH_ASSERT(dif==0);



//------- avec fonction variable
    ELISE_COPY
    (
          select(rectangle(Pt2di(0,0),Pt2di(20,15)),1),
          Abs(v4.red_sum(FX)-4*FX),
          sigma(dif)
    );
    BENCH_ASSERT(dif==0);



//------- avec voisinage variable
    ELISE_COPY
    (
          select(rectangle(Pt2di(0,0),Pt2di(20,40)),1),
          Abs(sel_func(v4,FX%2).red_sum(1)-2),
          sigma(dif)
    );
    BENCH_ASSERT(dif==0);

//------- avec voisinage et fonction variable
    ELISE_COPY
    (
          select(rectangle(Pt2di(0,0),Pt2di(20,15)),1),
          Abs(sel_func(v4,FX%2).red_sum(FX+FY)-2*(FX+FY)),
          sigma(dif)
    );
    BENCH_ASSERT(dif==0);

//------- avec voisinage et fonction variable et flux reelement selectionne
    ELISE_COPY
    (
          select(rectangle(Pt2di(0,0),Pt2di(20,20)),(FX+FY)%2),
          Abs(sel_func(v4,FX%2).red_sum(FX+FY)-2*(FX+FY)),
          sigma(dif)
    );
    BENCH_ASSERT(dif==0);

//------- avec flux/RLE
    ELISE_COPY
    (
          //disc(Pt2di(10,10),30),
          disc(Pt2dr(10,10),30), // __NEW
          Abs(v4.red_sum(1)-4),
          sigma(dif)
    );
    BENCH_ASSERT(dif==0);

//------- avec flux/RLE; oper max et min

    ELISE_COPY
    (
          //disc(Pt2di(10,10),30),
          disc(Pt2dr(10,10),30), // __NEW
          Abs(v4.red_sum(1)-4),
          sigma(dif)
    );
    BENCH_ASSERT(dif==0);

    ELISE_COPY
    (
          //disc(Pt2di(10,10),30),
          disc(Pt2dr(10,10),30), // __NEW
          Abs(v4.red_sum(FX)-4*FX),
          sigma(dif)
    );
    BENCH_ASSERT(dif==0);

    ELISE_COPY
    (
          //disc(Pt2di(10,10),30),
          disc(Pt2dr(10,10),30), // __NEW
          Abs(v4.red_max(FX)-(FX+1)),
          sigma(dif)
    );
    BENCH_ASSERT(dif==0);

    ELISE_COPY
    (
          //disc(Pt2di(10,10),30),
          disc(Pt2dr(10,10),30), // __NEW
          Abs(v4.red_min(FX)-(FX-1)),
          sigma(dif)
    );
    BENCH_ASSERT(dif==0);



    bench_red_op_ass_0(Pt2di(40,30));
    bench_red_op_ass_0(Pt2di(30,40));

    printf("END Red assoc \n");
}






