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


//*******************************************

void bench_rect_liste_pts
    (   Im2D_U_INT1 b1,
        Im2D_U_INT1 b2,
        Flux_Pts flx
    )
{
     Liste_Pts_INT2 l(2);

     ELISE_COPY(b1.all_pts(),0,b1.out()|b2.out());


     INT nb_pts;
     ELISE_COPY 
     (
          flx,
          1,
          l |  b2.out() | sigma(nb_pts)
     );

     ELISE_COPY
     (
        l.all_pts(),
        1,
        b1.out()
     );
     
     INT tx = b1.tx();
     INT ty = b2.ty();
     U_INT1 ** i1 = b1.data();
     U_INT1 ** i2 = b2.data();

     Im2D_U_INT1 b3(tx,ty);
     ELISE_COPY(b3.all_pts(),0,b3.out());

     Im2D_INT2 pts = l.image();
     U_INT1 ** i3 = b3.data();
     INT nb3 = pts.tx();
     INT2 * px = pts.data()[0];
     INT2 * py = pts.data()[1];
     for (INT k = 0; k<nb3 ; k++)
         i3[py[k]][px[k]] = 1;

     for (int y = 0; y<ty ; y++)
          for (int x = 0; x<tx ; x++)
          {
              BENCH_ASSERT(i1[y][x] == i2[y][x]);
              BENCH_ASSERT(i1[y][x] == i3[y][x]);
          }

    BENCH_ASSERT(nb_pts == l.card());
    BENCH_ASSERT(nb_pts == nb3);
    BENCH_ASSERT((nb_pts == 0) == l.empty());

}

void bench_rect_liste_pts
    (   Im2D_U_INT1 b1,
        Im2D_U_INT1 b2,
        Pt2di p1,
        Pt2di p2
    )
{
    bench_rect_liste_pts
    (
       b1,
       b2,
       select(rectangle(p1,p2),(FX+FY)%2)
    );
}


void bench_rect_liste_pts()
{
      INT tx = 105, ty = 120;
      Im2D_U_INT1 b1(tx,ty);
      Im2D_U_INT1 b2(tx,ty);

      int i;
      for (i = 0; i < 10 ; i++)
           bench_rect_liste_pts(b1,b2,Pt2di(i,i),Pt2di(2*i,2*i+1));

      for (i = 0; i < 10 ; i++)
          bench_rect_liste_pts(b1,b2,Pt2di(i,i),Pt2di(i*10,i*10));

      bench_rect_liste_pts(b1,b2,Pt2di(0,0),Pt2di(1,1));
      bench_rect_liste_pts(b1,b2,Pt2di(1,1),Pt2di(2,2));

      bench_rect_liste_pts
      (
         b1,
         b2,
         disc(Pt2dr(55,55),23)
      );
}


//*******************************************

void bench_multiple_write_lpts
    (   Im2D_U_INT1 b1,
        Im2D_U_INT1 b2,
        Pt2di p1,
        Pt2di p2,
        INT   per
    )
{
     Liste_Pts_INT2 l(2);

     ELISE_COPY(b1.all_pts(),0,b1.out()|b2.out());

     ELISE_COPY(rectangle(p1,p2),1,b2.out());

     for(int i=0 ; i<per ; i++)
        ELISE_COPY(select(rectangle(p1,p2),(FX+FY)%per == i),1,l);

    ELISE_COPY(l.all_pts(),1,b1.out());

    INT tx = b1.tx();
    INT ty = b2.ty();
    U_INT1 ** i1 = b1.data();
    U_INT1 ** i2 = b2.data();

    for (int y=0; y<ty ; y++)
         for (int x=0; x<tx ; x++)
             BENCH_ASSERT(i1[y][x] == i2[y][x]);
}

void bench_multiple_write_lpts()
{
      INT tx = 98, ty = 106;
      Im2D_U_INT1 b1(tx,ty);
      Im2D_U_INT1 b2(tx,ty);


      for (int i = 0; i < 10 ; i++)
          bench_multiple_write_lpts(b1,b2,Pt2di(i,i),Pt2di(i*9,i*8),i+2);
}

//*******************************************

void bench_lpts_1d
    (   Im1D_U_INT1 b1,
        Im1D_U_INT1 b2,
        INT x1,
        INT x2,
        INT per
    )
{
    Liste_Pts_INT4 l(1);
    INT nb_pts;
     
    ELISE_COPY(b1.all_pts(),0,b1.out()|b2.out());

    ELISE_COPY(    select(rectangle(x1,x2),(FX%per)%3)
           , 1
           , b2.out() | sigma(nb_pts)
    );


    for(int i=0 ; i<per ; i++)
       if (i%3)
          ELISE_COPY(select(rectangle(x1,x2),(FX%per) == i),1,l);

    ELISE_COPY(l.all_pts(),1,b1.out());


    INT tx = b1.tx();
    U_INT1 * i1 = b1.data();
    U_INT1 * i2 = b2.data();

    for (int x=0; x<tx ; x++)
        BENCH_ASSERT(i1[x] == i2[x]);

    BENCH_ASSERT(nb_pts == l.card());
    BENCH_ASSERT((nb_pts == 0) == l.empty());
}


void bench_lpts_1d()
{
    INT tx = 10000;
    Im1D_U_INT1 b1(tx);
    Im1D_U_INT1 b2(tx);

     
     bench_lpts_1d(b1,b2,0,0,1);
     bench_lpts_1d(b1,b2,0,1,1);

     for (int i = 0; i < 10 ; i++)
         bench_lpts_1d(b1,b2,i+1,i*999,i+2);
    
}

template <class Type,class TypeBase> 
void     bench_liste_to_im
         (
              Type *,
              TypeBase *,
              Pt2di sz,
              INT   dim
         )
{
    {
         Im2D_INT4           I1(sz.x,sz.y,0);
         Im2D_INT4           I2(sz.x,sz.y,0);
         Liste_Pts<Type,TypeBase> l (2);

         ELISE_COPY
         (
             rectangle(0,sz.x*sz.y).chc(Iconv(Virgule(sz.x*frandr(),sz.y*frandr()))),
             1,
             l
         );

         ELISE_COPY(l.all_pts(),1,I1.histo());

         Im2D<Type,TypeBase> b = l.image();
         Type * x = b.data()[0];
         Type * y = b.data()[1];
         INT nb = b.tx();
         INT ** i2 = I2.data();

         for (INT k=0; k<nb ; k++)
             i2[y[k]][x[k]]++;


         INT dif;
         ELISE_COPY(l.all_pts(),Abs(I1.in()-I2.in()),sigma(dif));
     
         BENCH_ASSERT(dif == 0);
    }


    {
        INT NB = 1000;

        Liste_Pts<Type,TypeBase>  l1(dim);
        Liste_Pts<Type,TypeBase>  l2(dim);

        Fonc_Num f = Iconv(sz.x*frandr());
        for (INT d =1 ; d< dim ; d++)
             f = Virgule(f,Iconv(sz.x*frandr()));

        ELISE_COPY
        (
            rectangle(0,NB).chc(f),
            0,
            l1
        );

        Im2D<Type,TypeBase> i1 = l1.image();
        Im1D<Type,TypeBase> ibuf(dim);

        Type ** pts = i1.data();
        Type *  buf = ibuf.data();

        for (int n=0 ; n < NB ; n++)
        {
            for (INT d=0 ; d<dim ; d++)
                buf[d] = pts[d][n];
            l2.add_pt(buf);
        }
        Im2D<Type,TypeBase> i2 = l2.image();

        INT dif;
        ELISE_COPY(i1.all_pts(),Abs(i1.in()-i2.in()),sigma(dif));
        BENCH_ASSERT(dif == 0);
        BENCH_ASSERT(i1.tx() == i2.tx());
        BENCH_ASSERT(i1.ty() == i2.ty());
    }
}

void bench_liste_to_im()
{
    bench_liste_to_im((INT1 *)  0,(INT *)0,Pt2di(30,40),1);
    bench_liste_to_im((INT2 *)  0,(INT *)0,Pt2di(320,40),2);
    bench_liste_to_im((INT2 *)  0,(INT *)0,Pt2di(20,520),3);
    bench_liste_to_im((INT  *)  0,(INT *)0,Pt2di(30,40),4);
    bench_liste_to_im((U_INT1 *)0,(INT *)0,Pt2di(30,40),5);
    bench_liste_to_im((U_INT2 *)0,(INT *)0,Pt2di(320,40),6);


    Liste_Pts<INT1,INT> l (3);
    INT p0[3] ={-1,-1,-1};
    INT p1[3] ={5,5,5};

    ELISE_COPY
    (
        select(rectangle(p0,p1,3),(FX==FY)&&(FY==FZ)),
        1,
        l
    );
    Im2D_INT1 i1 = l.image();
    BENCH_ASSERT((i1.tx()==6)&&(i1.ty()==3));
    for (INT i=0;i<6;i++)
    {
        BENCH_ASSERT
        (
               (i1.data()[0][i]==i-1)
            && (i1.data()[1][i]==i-1)
            && (i1.data()[2][i]==i-1)
        );
    }
}


template <class Type,class TypeBase> void bench_create_liste_data(Type*,TypeBase*)
{
    int tx = 200;

    Im2D <Type,TypeBase> b1 (tx,2);
    ELISE_COPY
    (
         b1.all_pts(),
         Fonc_Num(100,100),//(100 *frandr(),100*frandr()),
         b1.out()
    );

    // Liste_Pts<Type,TypeBase> l (b1.data()[0],b1.data()[1],tx);
    Liste_Pts<Type,TypeBase> l (2,b1.data(),tx);

    Im2D <Type,TypeBase> b2 = l.image();

    TypeBase dif;

    ELISE_COPY
    (
        b1.all_pts(),
        Abs(b1.in()-b2.in()),
        VMax(dif)
    );

    BENCH_ASSERT
    (
          (dif < epsilon)
       && (b1.tx() == b2.tx())
       && (b1.ty() == b2.ty())
    );
}

//*******************************************

template <class Type> void   bench_im_to_flux(Type *,INT nb)
{
    Im2D<Type,Type> Im(nb,3);

    ELISE_COPY(Im.all_pts(),(frandr()-0.5) *1000,Im.out());

    Type a0[3],b0[3],c0[3];
    Type a1[3],b1[3],c1[3];
    ELISE_COPY
    (
        to_flux(Im),
        Virgule(FX,FY,FZ),
           sigma(a0,3)
        |   VMax(b0,3)
        |   VMin(c0,3)
    );

    ELISE_COPY   
    (
       rectangle(0,nb),
       Virgule
       (
           Im.in()[Virgule(FX,0)],
           Im.in()[Virgule(FX,1)],
           Im.in()[Virgule(FX,2)]
       ),
           sigma(a1,3)
        |   VMax(b1,3)
        |   VMin(c1,3)
    );

    for (INT k=0; k<3 ; k++)
    {
        BENCH_ASSERT(ElAbs(a0[k]-a1[k])<epsilon);
        BENCH_ASSERT(ElAbs(b0[k]-b1[k])<epsilon);
        BENCH_ASSERT(ElAbs(c0[k]-c1[k])<epsilon);
    }
}

void     bench_im_to_flux(INT nb)
{
    bench_im_to_flux((INT  *)0,nb);
    bench_im_to_flux((REAL *)0,nb);
}

void     bench_im_to_flux()
{
      bench_im_to_flux(1);
      bench_im_to_flux(2);
      bench_im_to_flux(10);
      bench_im_to_flux(100);
      bench_im_to_flux(1000);
}

//*******************************************

void bench_liste_pts()
{

    bench_im_to_flux();

     bench_create_liste_data((U_INT1 *)0,(INT * ) 0);
     bench_create_liste_data((INT2 *)0,(INT * ) 0);
     bench_create_liste_data((REAL4 *)0,(REAL8 * ) 0);

{
    Liste_Pts<INT,INT> l (2);
    ELISE_COPY(rectangle(Pt2di(0,0),Pt2di(100,100)).chc(Virgule(FX,FY)),1,l);
    
    Im2D_U_INT1 b2(100,100);
    ELISE_COPY(l.all_pts(),FX+FY,b2.out());
    l.image();
}

     bench_rect_liste_pts();
     bench_multiple_write_lpts();
     bench_lpts_1d();

     bench_liste_to_im();


     cout << "OK liste_pts \n";
}






