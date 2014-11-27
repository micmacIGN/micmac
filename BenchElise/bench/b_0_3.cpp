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



void verif_select_0(Fonc_Num f)
{
    INT tx =153;
    INT ty =120;
    Im2D_U_INT1 b1(tx,ty);
    Im2D_U_INT1 b2(tx,ty);

    ELISE_COPY (b1.all_pts(),f!=0,b1.out());

    ELISE_COPY (b2.all_pts(),0,b2.out());
    ELISE_COPY (select(b2.all_pts(),f),1,b2.out());

    for (int y=0 ; y<ty ; y++)
        for (int x=0 ; x<tx ; x++)
            if (b1.data()[y][x] !=  b2.data()[y][x])
            {
                 cout << " Pb avec verif bitm  : " 
                      << " x  = " << x
                      << ", y  = " << y
                      << ", im  = " << b1.data()[y][x] << "\n";
                 exit(0);
            }
}



template <class Type,class TypeBase>
         void verif_select_1(Fonc_Num f,Type *,TypeBase *)
{
    INT tx =153;
    INT ty =120;
    Im2D<Type,TypeBase> b1(tx,ty);
    Im2D<Type,TypeBase> b2(tx,ty);

    ELISE_COPY (rectangle(Pt2di(-20,-20),Pt2di(tx+30,ty+40)),(Iconv(f)!=0)*f,b1.out());

    ELISE_COPY (b2.all_pts(),0,b2.out());
    ELISE_COPY 
    (     select(rectangle(Pt2di(-120,-110),Pt2di(tx+130,ty+140)),Iconv(f)),
          f,
          b2.oclip()
    );

    for (int y=0 ; y<ty ; y++)
        for (int x=0 ; x<tx ; x++)
            if (b1.data()[y][x] !=  b2.data()[y][x])
            {
                 cout << " Pb avec verif_select_1  : " 
                      << " x  = " << x
                      << ", y  = " << y
                      << ", im1  = " << b1.data()[y][x] 
                      << ", im2  = " << b2.data()[y][x] << "\n";
                 exit(0);
            }

    TypeBase dif;
    Im2D<Type,TypeBase> b3(b2.data_lin(),b2.data(),tx,ty);
    ELISE_COPY
    (
        b1.all_pts(),
        Abs(b3.in()-b2.in()),
        VMax(dif)
    );
 
    BENCH_ASSERT(dif<epsilon);
    
}


void verif_select_0()
{
    verif_select_0(FX>FY);
    verif_select_0(  ((FX%3) + (FY%7) + Square(FX)) % 7);
    verif_select_0(  (cos(FX) + cos(FX*FY) +sin(3.2 * FY)) > 0);
    verif_select_1(  ((FX%3) + (FY%7) + Square(FX)) % 7,(INT2 *)0,(INT *)0);
    verif_select_1(  ((FX%3) + (FY%7) + Square(FX)) % 7,(REAL4 *)0,(REAL *)0);

    verif_select_1(  (10*cos(FX)+20*sin(FY)) ,(INT2 *)0,(INT *)0);
    verif_select_1(  (10*cos(FX)+20*sin(FY)) ,(REAL4 *)0,(REAL *)0);
}

template <class Type,class TypeBase>
            void tpl_verif_im_input_integer_no_def
     (
          Type *,
          TypeBase *,
          Fonc_Num f,
          Fonc_Num fsel,
          Pt2di sz
     )
{
     Im2D<Type,TypeBase> b1(sz.x,sz.y);

     ELISE_COPY(b1.all_pts(),f,b1.out());

     // control also the initialization of sigma to 0
     TypeBase s1 = (TypeBase) 23;
     TypeBase s2 = (TypeBase) -23;

     ELISE_COPY(b1.all_pts(),b1.in(),sigma(s1));
     ELISE_COPY(select(b1.all_pts(),1),b1.in(),sigma(s2));
     BENCH_ASSERT(s1==s2);

     ELISE_COPY(b1.all_pts(),b1.in()*(fsel!=0),sigma(s1));
     ELISE_COPY(select(b1.all_pts(),fsel),b1.in(),sigma(s2));
     BENCH_ASSERT(s1==s2);

}

void verif_im_input_integer_no_def()
{
     tpl_verif_im_input_integer_no_def
     (
          (INT *)0,
          (INT *)9,
          FX+FY,
          (FX+FY)%2,
          Pt2di(67,189)
     );
     tpl_verif_im_input_integer_no_def
     (
          (U_INT1 *)0,
          (INT *)9,
          FX+FY,
          (FX+FY)%2,
          Pt2di(67,189)
     );
     tpl_verif_im_input_integer_no_def
     (
          (REAL *)0,
          (REAL *)9,
          FX+FY,
          (FX+FY)%2,
          Pt2di(67,189)
     );
}


template <class Type,class TypeBase> void verif_conv_out_type_fonc
                           (Fonc_Num f,Type *,TypeBase *)
{
     Pt2di sz (97,109);

     Im2D<Type,TypeBase> b1(sz.x,sz.y);
     Im2D<Type,TypeBase> b2(sz.x,sz.y);

     ELISE_COPY
     (
          b1.all_pts(),
          f,
          b1.out()
     );

     ELISE_COPY
     (
          select(b1.all_pts(),(FX+FY)%2),
          f,
          b2.out()
     );
     
     ELISE_COPY
     (
          select(b1.all_pts(),(FX+FY+1)%2),
          f,
          b2.out()
     );


     INT nb_dif;
     TypeBase dif_tot;


     
     ELISE_COPY(b1.all_pts(),b1.in() != b2.in(),sigma(nb_dif));
     ELISE_COPY(b1.all_pts(),Abs(b1.in()-b2.in()),sigma(dif_tot));


     BENCH_ASSERT(nb_dif == 0);
}


//***********************************************************

void verif_select_imbr(Fonc_Num f1,Fonc_Num f2)
{
    INT tx =153;
    INT ty =120;
    Im2D_U_INT1 b1(tx,ty);
    Im2D_U_INT1 b2(tx,ty);

    ELISE_COPY (b1.all_pts(),(f1!=0)&&(f2!=0),b1.out());

    ELISE_COPY (b2.all_pts(),0,b2.out());
    ELISE_COPY (select(select(b2.all_pts(),f1),f2),1,b2.out());

    for (int y=0 ; y<ty ; y++)
        for (int x=0 ; x<tx ; x++)
            if (b1.data()[y][x] !=  b2.data()[y][x])
            {
                 cout << " Pb avec verif bitm  : " 
                      << " x  = " << x
                      << ", y  = " << y
                      << ", im  = " << b1.data()[y][x] << "\n";
                 exit(0);
            }
}

void verif_select_imbr()
{
     verif_select_imbr(FX%3,FY%2);
     verif_select_imbr(FY%3,FX%2);
     verif_select_imbr(FX%3,FX%2);
     verif_select_imbr(FY%3,FY%2);
}

// Pas un bench, juste un programme qui
// a buggue un jour

void bug_flux()
{
     Im2D_U_INT1 PsGr(256,256,1);

     ELISE_COPY
     (
         line(Pt2di(0,0),Pt2di(128,128)),
         Fonc_Num(0,0,0),
         (PsGr.oclip())
     );
}


//***********************************************************

void verif_flux()
{
    All_Memo_counter MC_INIT;
    stow_memory_counter(MC_INIT);
    {
         bug_flux();
         verif_select_0();
         cout << "OK verif_select \n";

         verif_im_input_integer_no_def();

         verif_conv_out_type_fonc((FX+FY)%13,(INT *)0,(INT *)0);
         verif_conv_out_type_fonc(20*cos(FX)+10*sin(FY),(REAL *)0,(REAL *)0);
         verif_conv_out_type_fonc((FX+FY)%13,(REAL *)0,(REAL *)0);
         verif_conv_out_type_fonc(20*cos(FX)+10*sin(FY),(INT *)0,(INT *)0);
     
         verif_select_imbr();
    }
    verif_memory_state(MC_INIT);
    cout << "OK verif_flux \n";
}




