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


void test_sigm_0_0_cat_coord(Pt2di p1,Pt2di p2)
{
     INT s[2];

     ELISE_COPY(rectangle(p1,p2),Virgule(FX,FY),sigma(s,2));

     BENCH_ASSERT
     (
             (s[0] == ElAbs(som_x(p1.x,p2.x)*(p1.y-p2.y)))
          && (s[1] == ElAbs(som_x(p1.y,p2.y)*(p1.x-p2.x)))
     );

}

void test_sigm_0_1_cat_coord(Pt2di p1,Pt2di p2)
{
     REAL s[2];

     ELISE_COPY(rectangle(p1,p2),Virgule(FX,Rconv(FY)),sigma(s,2));

     BENCH_ASSERT
     (
             (s[0] == ElAbs(som_x(p1.x,p2.x)*(p1.y-p2.y)))
          && (s[1] == ElAbs(som_x(p1.y,p2.y)*(p1.x-p2.x)))
     );

}



void test_sigm_0_2_cat_coord(Pt2di p1,Pt2di p2)
{
     REAL s[2];

     ELISE_COPY(rectangle(p1,p2),Rconv(Virgule(FX,FY)),sigma(s,2));

     BENCH_ASSERT
     (
             (s[0] == ElAbs(som_x(p1.x,p2.x)*(p1.y-p2.y)))
          && (s[1] == ElAbs(som_x(p1.y,p2.y)*(p1.x-p2.x)))
     );

}


void test_sigm_0_cat_coord()
{
    test_sigm_0_0_cat_coord(Pt2di(-20,15),Pt2di(113,89));
    test_sigm_0_1_cat_coord(Pt2di(-20,15),Pt2di(113,89));
    test_sigm_0_2_cat_coord(Pt2di(-20,15),Pt2di(113,89));
}


      //***************************************************************


void test_sigm_1_0_cat_coord()
{
    int s12[2],s1,s2;
    Flux_Pts r = rectangle(Pt2di(-10,10),Pt2di(120,120));

    ELISE_COPY
    (
        r,
        ( Virgule(FX,FY) + Square(Virgule(FY,FX))),
        sigma(s12,2)
    );

    ELISE_COPY(r,FX+Square(FY),sigma(s1));
    ELISE_COPY(r,FY+Square(FX),sigma(s2));

    BENCH_ASSERT((s12[0] == s1) && (s12[1] == s2));
}


void test_sigm_1_1_cat_coord()
{
    REAL s12[2],s1,s2;
    Flux_Pts r = rectangle(Pt2di(-10,10),Pt2di(120,120));

    ELISE_COPY
    (
        r,
        ( cos(Virgule(FX,FY)) + (  Virgule(FY,FX) & Square(Virgule(FX,FY)))),
        sigma(s12,2)
    );

    ELISE_COPY(r,cos(FX)+(FY&Square(FX)),sigma(s1));
    ELISE_COPY(r,cos(FY)+(FX&Square(FY)),sigma(s2));

    BENCH_ASSERT((s12[0] == s1) && (s12[1] == s2));
}


void test_sigm_1_cat_coord()
{
     test_sigm_1_0_cat_coord();
     test_sigm_1_1_cat_coord();
}

      //***************************************************************

void test_sigm_2_0_cat_coord(Pt2di p1,Pt2di p2)
{
    INT s1,sx,sy;

    ELISE_COPY
    (
         rectangle(p1,p2),
         Virgule(1,FX,FY),
         Virgule(sigma(s1),sigma(sx),sigma(sy))
    );

     BENCH_ASSERT
     (
             (s1 == ElAbs((p1.x-p2.x)*(p1.y-p2.y)))
          && (sx == ElAbs(som_x(p1.x,p2.x)*(p1.y-p2.y)))
          && (sy == ElAbs(som_x(p1.y,p2.y)*(p1.x-p2.x)))
     );

}


void test_sigm_2_1_cat_coord(Pt2di sz)
{
     Im2D_REAL8  b0(sz.x,sz.y);
     Im2D_REAL8  b1(sz.x,sz.y);
     Im2D_REAL8  b2(sz.x,sz.y);

     Fonc_Num f0 = FX+0.5;
     Fonc_Num f1 = FX-FY;
     Fonc_Num f2 = FY-0.5;


     ELISE_COPY
     (   b1.all_pts(),
         Virgule(f0,f1,f2),
         Virgule(b0.out(),b1.out(),b2.out())
     );

     REAL s0,s1,s2;

     ELISE_COPY(b0.all_pts(),Abs(b0.in()-f0),sigma(s0));
     ELISE_COPY(b1.all_pts(),Abs(b1.in()-f1),sigma(s1));
     ELISE_COPY(b2.all_pts(),Abs(b2.in()-f2),sigma(s2));
     
     BENCH_ASSERT
     (
            (s0 == 0.0)
         && (s1 == 0.0)
         && (s2 == 0.0)
     );

}

void test_sigm_2_cat_coord()
{
    test_sigm_2_0_cat_coord(Pt2di(-30,5),Pt2di(103,83));
    test_sigm_2_1_cat_coord(Pt2di(83,108));
}


      //***************************************************************

void test_0_pipe_out(Pt2di p1,Pt2di p2)
{
    INT s0,s1,s2;
    INT theor;

    theor =  ElAbs(som_x(p1.x,p2.x)*(p1.y-p2.y))
          +  ElAbs(som_x(p1.y,p2.y)*(p1.x-p2.x));

    ELISE_COPY
    (    rectangle(p1,p2),
         FX+FY,
         sigma(s0) | sigma(s1) | sigma(s2)
    );
    
    BENCH_ASSERT
    (
             (s0 == theor)
        &&   (s1 == theor)
        &&   (s2 == theor)
    );

}

void test_1_pipe_out(Pt2di sz,bool pipe_before)
{
    Im2D_INT4 b0(sz.x,sz.y);
    Im2D_INT4 b1(sz.x,sz.y);
    INT  sx,sy;

    if (pipe_before)
       ELISE_COPY
       (
            b1.all_pts(),
            Virgule(FX,FY),
            Virgule(   (b0.out() | sigma(sx)) , (sigma(sy) | b1.out()))
       );
    else
       ELISE_COPY
       (
            b1.all_pts(),
            Virgule(FX,FY),
             Virgule(b0.out(),b1.out()) | Virgule(sigma(sx),sigma(sy))
       );

    for( INT y = 0; y<sz.y ; y++)
        for( INT x = 0; x<sz.x ; x++)
           if(      (b0.data()[y][x] != x)
                ||  (b1.data()[y][x] != y)
           )
           {
               cout << "PB avec test_1_pipe_out \n";
               exit(0);
           }

     BENCH_ASSERT
     (
             (sx == ElAbs(som_x(0,sz.x)*sz.y))
        &&   (sy == ElAbs(som_x(0,sz.y)*sz.x))
     );

}



void test_pipe_out()
{
    test_0_pipe_out(Pt2di(-30,5),Pt2di(103,83));
    test_1_pipe_out(Pt2di(147,129),true);
    test_1_pipe_out(Pt2di(147,129),false);
}
      //***************************************************************

void test_0_redir_out(Pt2di sz)
{
    Im2D_INT4 b0(sz.x,sz.y);
    Im2D_INT4 b1(sz.x,sz.y);

    ELISE_COPY
    (  b0.all_pts(),
       FX,
       b0.out() | (b1.out() << FY+b0.in())
    );

    INT dif;

    ELISE_COPY(b1.all_pts(),Abs(b1.in()-FX-FY),sigma(dif));

    BENCH_ASSERT(dif == 0);


}


void test_redir_out()
{
      test_0_redir_out(Pt2di(172,149));
}


      //***************************************************************

void test_kth_proj()
{
    {
         Fonc_Num f1 = Virgule(FX,FY,FX-FY,1,2);
         Fonc_Num f2 = Virgule(FY,FX-FY,FX,2,1,FX);
         INT perm[6] = {1,2,0,4,3,0};
         std::vector<INT>  VPerm(perm,perm+6);
         INT Dif[6];

         Fonc_Num f3 = Virgule(2,FX,FY,FX-FY,1);

         ELISE_COPY
         (
             rectangle(Pt2di(-10,-10),Pt2di(120,130)),
               Abs(f1.permut(VPerm)-f2)
             + Abs(Virgule(0,f1.shift_coord(1)-f3)),
             sigma(Dif,6)
         );

         for (INT k=0 ; k<6 ; k++)
         {
              BENCH_ASSERT(Dif[k]==0);
         }
     }
     INT difi;

     ELISE_COPY
     (
         rectangle(Pt2di(0,0),Pt2di(100,100)),
         Abs(Virgule(FX,FY).v0()-FX),
         sigma(difi)
     );
     BENCH_ASSERT(difi == 0);


     ELISE_COPY
     (
         rectangle(Pt2di(0,0),Pt2di(100,100)),
         Abs(Virgule(FX,FY).v1()-FY),
         sigma(difi)
     );

     BENCH_ASSERT(difi == 0);

     INT p0[4] = {-1,-2,-3,-4};
     INT p1[4] = {2,3,4,5};
     REAL dr[4];

     Symb_FNum f1 = Virgule(FX+0.1,FY+0.2,FZ+0.3,kth_coord(3)+0.4);
     Symb_FNum f2 = Virgule(FX+0.1,FY+0.2,FZ+0.3,kth_coord(3)+0.4);
     ELISE_COPY
     (
         rectangle(p0,p1,4),
         Abs
         (
            f1-Virgule(f2.v0(),f2.v1(),f2.v2(),f2.kth_proj(3))
          ),
         sigma(dr,4)
     );

     BENCH_ASSERT
     (
              (dr[0] == 0)
          &&  (dr[1] == 0)
          &&  (dr[2] == 0)
          &&  (dr[3] == 0)
     );



}


      //***************************************************************

template <class Type> void test_clip_def
     (
          Flux_Pts flx,
          Fonc_Num  f,
          Type      def_val,
          Pt2di     p1,
          Pt2di     p2
     )
{

     Symb_FNum ins = 
                         (p1.x <= FX)
                      && (FX < p2.x)
                      && (p1.y <= FY)
                      && (FY < p2.y) ;
     Type s1,s2;

     ELISE_COPY
     (
         flx,
         ins *f + (1-ins) * def_val,
         sigma(s1)
     );
     ELISE_COPY
     (
         flx,
         clip_def(f,def_val,p1,p2),
         sigma(s2)
     );

     Type s3;
     Symb_FNum ins2 = inside(p1,p2);
     ELISE_COPY
     (
         flx,
         ins2 *f + (1-ins2) * def_val,
         sigma(s3)
     );

     BENCH_ASSERT(ElAbs(s1-s2)<epsilon);
     BENCH_ASSERT(ElAbs(s3-s2)<epsilon);
}

void test_clip_def()
{
     test_clip_def
     (
          disc(Pt2dr(50,50),30),
          (FX+FY),
          12,
          Pt2di(2,2),Pt2di(90,90)
      );

     test_clip_def
     (
          disc(Pt2dr(50,50),60),
          (FX+FY),
          12,
          Pt2di(2,2),Pt2di(90,90)
      );

     test_clip_def
     (
          select(disc(Pt2dr(50,50),60),1),
          (FX+FY),
          12,
          Pt2di(2,2),Pt2di(90,90)
     );

     test_clip_def
     (
          disc(Pt2dr(50,50),60).chc(Virgule(FX+0.123456,FY-0.7654321)),
          (FX+FY),
          12.0,
          Pt2di(2,2),Pt2di(90,90)
     );
}



      //***************************************************************

void test_sigm_cat_coord()
{
    test_sigm_0_cat_coord();
    test_sigm_1_cat_coord();
    test_sigm_2_cat_coord();

    test_pipe_out();
    test_redir_out();

    test_kth_proj();

    test_clip_def();
    cout << "OK test_sigm_cat_coord \n";
}










