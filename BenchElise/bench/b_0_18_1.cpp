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


template <class Type> void bench_op_buf_cat
                           (
                               Type *,
                               Fonc_Num aOpFonc,
                               Fonc_Num aFoncInit,
                               Fonc_Num aFoncCat,
                               Pt2di  p0,
                               Pt2di  p1
                           )
{

     INT dOp   =   aOpFonc.dimf_out();
     INT dInit = aFoncInit.dimf_out();
     INT dTot = aFoncCat.dimf_out();




     BENCH_ASSERT(dTot== (dInit+dOp));


     Type * dif = new Type [dTot];


     ELISE_COPY
     (
         rectangle(p0,p1),
         Abs(aFoncCat-Virgule(aOpFonc,aFoncInit)),
         sigma(dif,dTot)
     );

    for (INT d=0; d<dTot ; d++)
    {
         BENCH_ASSERT(dif[d]==0);
    }


    delete [] dif;
}

static Pt2di bench_op_buf_cat_PRand_centre(INT V)

{
    return Pt2di(Pt2dr(V*(NRrandom3()-0.5),V*(NRrandom3()-0.5)));
}


void bench_op_buf_cat
    (
           Pt2di szIm
    )
{

    Pt2di p0 = Pt2di(Pt2dr(szIm.x*(NRrandom3()/2.1 -0.1),szIm.y *(NRrandom3()/2.1-0.1)));
    Pt2di p1 = Pt2di(Pt2dr(szIm.x*(1.1 -NRrandom3()/2.1),szIm.y *(1.1-NRrandom3()/2.1)));
    INT fact = 50;

    Im2D_REAL8 Im1(szIm.x,szIm.y);
    Im2D_REAL8 Im2(szIm.x,szIm.y);
    ELISE_COPY
    (
         Im1.all_pts(),
         Virgule(unif_noise_4(3) * fact, unif_noise_4(3) * fact),
         Virgule(Im1.out(),Im2.out())
   );


   Box2di aBox(bench_op_buf_cat_PRand_centre(20),bench_op_buf_cat_PRand_centre(20));

    aBox._p0.SetInf(Pt2di(0,0));
    aBox._p1.SetSup(Pt2di(0,0));



    bench_op_buf_cat<INT> ((INT *)0, FX,FY,Virgule(FX,FY), p0,p1);

    Fonc_Num RFonc = Im1.in(0);
    Fonc_Num IFonc = Iconv(Im1.in(0));

    bench_op_buf_cat<INT> 
    ( 
         (INT *) 0,
         rect_som(IFonc,aBox),
         IFonc,
         rect_som(IFonc,aBox,true),
         p0,p1
    );

    bench_op_buf_cat<REAL> 
    ( 
         (REAL *) 0,
         rect_som(RFonc,aBox),
         RFonc,
         rect_som(RFonc,aBox,true),
         p0,p1
    );

    bench_op_buf_cat<REAL> 
    ( 
         (REAL *) 0,
         rect_min(RFonc,aBox),
         RFonc,
         rect_min(RFonc,aBox,true),
         p0,p1
    );

    {
       Fonc_Num f2 = Virgule(IFonc,1-IFonc,FX*1-IFonc);
       bench_op_buf_cat<INT> 
       ( 
            (INT *) 0,
            rect_max(f2,aBox),
            f2,
            rect_max(f2,aBox,true),
            p0,p1
       );
    }

    {
        Fonc_Num RhoTeta = Virgule(Im1.in(0),Im2.in(0));
        REAL Ouv  = NRrandom3();
        bool Oriented = (NRrandom3() > 0.5);
        REAL RhoCalc = 0.1 + 2.0*NRrandom3();
        bench_op_buf_cat<REAL>
        (
             (REAL *) 0,
             RMaxLocDir(RhoTeta,Ouv,Oriented,RhoCalc,false),
             RhoTeta,
             RMaxLocDir(RhoTeta,Ouv,Oriented,RhoCalc,true),
             p0,p1
        );                 
    }

    bench_op_buf_cat<INT> 
    ( 
         (INT *) 0,
         rect_median(IFonc,aBox,256),
         IFonc,
         rect_median(IFonc,aBox,256,true),
         p0,p1
    );

    {
        Fonc_Num  f2 = Virgule(IFonc,mod(IFonc*FX*FY,256));
        bench_op_buf_cat<INT> 
        ( 
             (INT *) 0,
             rect_median(f2,aBox,256),
             f2,
             rect_median(f2,aBox,256,true),
             p0,p1
        );
    }




}



void bench_op_buf_cat()
{
    for (INT k=0 ; k< 100; k++)
    {
       bench_op_buf_cat(Pt2di(100,100));
       bench_op_buf_cat(Pt2di(150,100));
       bench_op_buf_cat(Pt2di(150,150));
       cout << "K= " << k << "\n";
    }

}

