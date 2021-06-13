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



void  bench_can_exp_op_buf(Pt2di sz,Pt2di p0,REAL fx,REAL fy)
{
      Im2D_REAL8  I0(sz.x,sz.y);
      ELISE_COPY(I0.all_pts(),0.0,I0.out());

      REAL ** d0 = I0.data();
      for (INT x = p0.x; x < sz.x; x++)
          for (INT y = p0.y; y <sz.y; y++)
          {
              d0[y][x] = Pow(fx,x-p0.x) * Pow(fy,y-p0.y) ;
          }
      
      Im2D_REAL8  I1(sz.x,sz.y);

      ELISE_COPY
      (
           I1.all_pts(),
           semi_cef
           (
                (FX==p0.x)&&(FY==p0.y),
                fx,fy
           ),
           I1.out()
      );


      REAL dif;
      ELISE_COPY(I1.all_pts(),Abs(I0.in()-I1.in()),VMax(dif));

     BENCH_ASSERT(dif < epsilon);
}

void  bench_can_exp_inv(Pt2di sz,Pt2di p0,REAL fx,REAL fy)
{

      Im2D_REAL8  I1(sz.x,sz.y);
      Fonc_Num DeltaP0 =  (FX==p0.x)&&(FY==p0.y);
      ELISE_COPY
      (
           I1.all_pts(),
           canny_exp_filt(DeltaP0,fx,fy),
           I1.out()
      );


      Im2D_REAL8  I2(sz.x,sz.y);
      ELISE_COPY
      (
        I1.all_pts(),
        inv_canny_exp_filt(I1.in(0),fx,fy),
        I2.out()
      );


       REAL dif;
       ELISE_COPY
       (
          I2.interior(1),
          Abs(I2.in()-DeltaP0),
          VMax(dif)
       );
       BENCH_ASSERT(dif < epsilon);
      
}


void  bench_can_exp_op_buf(Pt2di sz,Pt2di p0,REAL fx,REAL fy,INT nb)
{
      Im2D_REAL8  I0(sz.x,sz.y);
      REAL ** d0 = I0.data();

      for (INT x = 0; x <sz.x; x++)
          for (INT y = 0; y <sz.y; y++)
          {
              d0[y][x] = 
                          (y >= p0.y -nb)                              ?
                          (Pow(fx,ElAbs(x-p0.x)) * Pow(fy,ElAbs(y-p0.y)))  :
                          0.0                                          ;
          }
      
      Im2D_REAL8  I1(sz.x,sz.y);

      ELISE_COPY
      (
           I1.all_pts(),
           canny_exp_filt
           (
                (FX==p0.x)&&(FY==p0.y),
                fx,fy,nb
           ),
           I1.out()
      );


      REAL dif;
      ELISE_COPY(I1.all_pts(),Abs(I0.in()-I1.in()),VMax(dif));

     BENCH_ASSERT(dif < epsilon);

     bench_can_exp_op_buf(sz,p0,fx,fy);
}




void  bench_can_exp_op_buf()
{
     bench_can_exp_inv(Pt2di(100,100),Pt2di(50,50),0.5,0.5);
     bench_can_exp_inv(Pt2di(100,100),Pt2di(50,50),0.3,0.7);
     bench_can_exp_inv(Pt2di(100,100),Pt2di(50,50),0.7,0.3);

     bench_can_exp_op_buf(Pt2di(100,100),Pt2di(50,50),0.9,0.7,3);
     bench_can_exp_op_buf(Pt2di(100,100),Pt2di(50,50),0.7,0.9,8);
}


/******************************************************/

INT comp_int(const void * v1, const void *v2)
{
    INT i1 = * ((INT *)const_cast<void *>(v1));
    INT i2 = * ((INT *)const_cast<void *>(v2));

    if (i1>i2) return 1;

    return (i1<i2) ? -1 : 0;
}

void sort_int(INT * t,INT nb)
{
     qsort(t,nb,sizeof(INT),comp_int);
}

void bench_rect_kth(Pt2di sz,Box2di s,INT kth)
{
     Im2D_U_INT1 I0(sz.x,sz.y);

     ELISE_COPY
     (
         I0.all_pts(),
         256 * frandr(),
         I0.out()
     );

     Im2D_U_INT1 Ik1(sz.x,sz.y);

     ELISE_COPY
     (
        I0.all_pts(),
        rect_kth(I0.in(0),kth,s,256),
        Ik1.out()
     );

     Im2D_U_INT1 Ik2(sz.x,sz.y);
     U_INT1 ** d2 = Ik2.data();
     U_INT1 ** im = I0.data();
     INT * sorted = 
             NEW_VECTEUR
             (
                0,
                ElAbs(s._p1.x-s._p0.x+1) * ElAbs(s._p1.y-s._p0.y+1),
                INT
             );

     for(INT y=0 ; y<sz.y ; y++)
         for(INT x=0 ; x<sz.x ; x++)
         {
              INT nb=0;
               for(INT dx =s._p0.x; dx <= s._p1.x; dx++)
               {
                   for(INT dy =s._p0.y; dy <= s._p1.y; dy++)
                   {
                      INT xl = x+dx;
                      INT yl = y+dy;
                      if ((xl>=0)&&(xl<sz.x)&&(yl>=0)&&(yl<sz.y))
                          sorted[nb++] = im[yl][xl];
                      else
                          sorted[nb++] = 0;

                   }
                }
                sort_int(sorted,nb);
                d2[y][x] = sorted[kth];
         }
      DELETE_VECTOR(sorted,0);

      INT dif;
      ELISE_COPY(Ik1.all_pts(),Abs(Ik1.in()-Ik2.in()),sigma(dif));

      BENCH_ASSERT(dif == 0);
}


void bench_rect_kth()
{
      bench_rect_kth(Pt2di(30,30),Box2di(Pt2di(-1,-1),Pt2di(1,1)),4);

      bench_rect_kth(Pt2di(30,30),Box2di(Pt2di(-1,-1),Pt2di(2,2)),2);
      bench_rect_kth(Pt2di(10,40),Box2di(Pt2di(-2,-1),Pt2di(3,2)),18);
      bench_rect_kth(Pt2di(40,10),Box2di(Pt2di(-2,-2),Pt2di(2,2)),12);

      bench_rect_kth(Pt2di(30,30),Box2di(Pt2di(-3,-3),Pt2di(4,4)),63);
      bench_rect_kth(Pt2di(30,30),Box2di(Pt2di(-3,-3),Pt2di(4,4)),0);
      bench_rect_kth(Pt2di(30,30),Box2di(Pt2di(-1,-1),Pt2di(1,1)),8);
      bench_rect_kth(Pt2di(30,30),Box2di(Pt2di(-1,-1),Pt2di(1,1)),0);
}

void bench_rect_var_som(Pt2di sz,Box2di side)
{
    Im2D_U_INT1 I (sz.x,sz.y);
    Im2D_INT1 X0 (sz.x,sz.y);
    Im2D_INT1 Y0 (sz.x,sz.y);

    Im2D_INT1 X1 (sz.x,sz.y);
    Im2D_INT1 Y1 (sz.x,sz.y);

    ELISE_COPY
    (
        I.all_pts(),
        1,
           (X0.out() << ((frandr()-0.5) * 8))
        |  (Y0.out() << ((frandr()-0.5) * 8))
        |  (X1.out() << ((frandr()-0.5) * 8))
        |  (Y1.out() << ((frandr()-0.5) * 8))
        |  (I.out() <<  (frandr() * 256))
    );


    {
         Symb_FNum x0(Min(X0.in(),X1.in()));
         Symb_FNum x1(Max(X0.in(),X1.in()));
         Symb_FNum y0(Min(Y0.in(),Y1.in()));
         Symb_FNum y1(Max(Y0.in(),Y1.in()));

         ELISE_COPY
         (
              I.all_pts(),
              1,
                 (X0.out()<<x0) |  (X1.out()<<x1)
              |  (Y0.out()<<y0) |  (Y1.out()<<y1)
         );
    }




    Im2D_INT4 S1 (sz.x,sz.y);

    ELISE_COPY
    (
        I.all_pts(),
        rect_var_som(I.in(0),Virgule(X0.in(),Y0.in(),X1.in(),Y1.in()),side),
        S1.out()
    );

    Im2D_INT4 S2 (sz.x,sz.y);
    ELISE_COPY(I.all_pts(),0,S2.out());

    INT4 ** s2 = S2.data();
    INT1 ** x0 = X0.data();
    INT1 ** y0 = Y0.data();
    INT1 ** x1 = X1.data();
    INT1 ** y1 = Y1.data();
    U_INT1 ** i  = I.data();

    for (INT y =0; y<sz.y;y++)
    {
        for (INT x =0; x<sz.x;x++)
        {
             INT u0 = ElMax(0,x+ElMax(side._p0.x,(int)x0[y][x]));
             INT v0 = ElMax(0,y+ElMax(side._p0.y,(int)y0[y][x]));
             INT u1 = ElMin(sz.x-1,x+ElMin(side._p1.x,(int)x1[y][x]));
             INT v1 = ElMin(sz.y-1,y+ElMin(side._p1.y,(int)y1[y][x]));
    
             INT som = 0;
             for (INT v = v0; v<=v1 ; v++)
                  for (INT u = u0; u<=u1 ; u++)
                      som += i[v][u];
             s2[y][x] = som;
        }
    }

    INT dif,nb_dif;
    ELISE_COPY(S1.all_pts(),Abs(S1.in()-S2.in()),VMax(dif));
    ELISE_COPY(S1.all_pts(),S1.in()!=S2.in(),sigma(nb_dif));

    BENCH_ASSERT(dif == 0);
  

    ELISE_COPY
    (
        I.all_pts(),
        Abs(Virgule(X0.in(),Y0.in())),
        Virgule(X0.out(),Y0.out())
    );

    ELISE_COPY
    (
        I.all_pts(),
        Abs
        (
            rect_var_som(I.in(0),Virgule(-X0.in(),-Y0.in(),X0.in(),Y0.in()),side)
           -rect_var_som(I.in(0),Virgule(X0.in(),Y0.in()),side)
        ),
        VMax(dif)
    );

    BENCH_ASSERT(dif == 0);


    ELISE_COPY
    (
        I.all_pts(),
        Abs
        (
            rect_var_som(I.in(0),Virgule(X0.in(),X0.in()),side)
           -rect_var_som(I.in(0),X0.in(),side)
        ),
        VMax(dif)
    );

    BENCH_ASSERT(dif == 0);
}


void bench_rect_var_som()
{
    bench_rect_var_som(Pt2di(30,40),Box2di(Pt2di(-2,-3),Pt2di(4,2)));
}


void bench_rank(Pt2di sz,Box2di box)
{
     Im2D_U_INT1 I(sz.x,sz.y);
     ELISE_COPY(I.all_pts(),255*frandr(),I.out());

     Im2D_U_INT2 RK(sz.x,sz.y);

     ELISE_COPY
     (
         RK.all_pts(),
         rect_rank(I.in(3),box,256),
         RK.out()
     );

     U_INT2 ** rk = RK.data();
     U_INT1 **  i =  I.data();

     for (INT x= -box._p0.x; x < sz.x-box._p1.x-1; x++)
         for (INT y= -box._p0.y; y < sz.y-box._p1.y-1; y++)
         {

              INT res = 0;
              for (INT dx = box._p0.x ; dx<=box._p1.x ; dx++)
                  for (INT dy = box._p0.y ; dy<=box._p1.y ; dy++)
                      res += (i[y][x] >= i[y+dy][x+dx]);


              BENCH_ASSERT(rk[y][x] == res);
         }
}
void bench_rank()
{

    bench_rank(Pt2di(30,40),Box2di(Pt2di(-1,-1),Pt2di(1,1)));
    bench_rank(Pt2di(30,40),Box2di(Pt2di(0,0),Pt2di(1,1)));
    bench_rank(Pt2di(30,40),Box2di(Pt2di(-5,-6),Pt2di(7,6)));

    bench_rank(Pt2di(100,120),Box2di(Pt2di(-3,-4),Pt2di(5,2)));
}

void  bench_op_buf_1()
{
       bench_can_exp_op_buf();
       bench_rect_kth();
       bench_rank();
       bench_rect_var_som();
       cout << "OK OP BUF RECT 1 \n";
}



