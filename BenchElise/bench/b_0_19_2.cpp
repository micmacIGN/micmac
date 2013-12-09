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



template <class Type> void bench_som_masq
                           (
                               Type *,
                               bool integer,
                               Pt2di p0,
                               Pt2di sz_masq,
                               Pt2di sz_im
                           )
{
    Im2D<REAL,REAL> masq(sz_masq.x,sz_masq.y);
    if (integer)
        ELISE_COPY(masq.all_pts(),round_ni(255*frandr()),masq.out());
    else
        ELISE_COPY(masq.all_pts(),255*frandr(),masq.out());

    ELISE_COPY(select(masq.all_pts(),frandr()<0.3),0,masq.out());

    Im2D<Type,Type> image(sz_im.x,sz_im.y);
    ELISE_COPY(image.all_pts(),255*frandr(),image.out());


    Fonc_Num f0 = (Type) 0;

    for (int x =0; x<sz_masq.x; x++)
        for (int y =0; y<sz_masq.y; y++)
            f0 =    f0 
                 +    ((Type)masq.data()[y][x])
                    * trans(image.in(0),Pt2di(x,y)+p0) ;

    Type dif;

    ELISE_COPY
    (
        image.all_pts(),
        Abs(f0-som_masq(image.in(0),masq,p0)),
        VMax(dif)
    );

    BENCH_ASSERT(dif<epsilon);

    if ((sz_masq.x%2)&&(sz_masq.y%2))
    {
        ELISE_COPY
        (
            image.all_pts(),
            Abs(som_masq(image.in(0),masq)-som_masq(image.in(0),masq,-(sz_masq/2))),
            VMax(dif)
       );

       BENCH_ASSERT(dif<epsilon);
    }

    Im2D<Type,Type> image_2(sz_im.x,sz_im.y);
    ELISE_COPY(image_2.all_pts(),255*frandr(),image_2.out());

    Type dif2;
    ELISE_COPY
    (
        image.all_pts(),
        Abs
        (   Virgule(som_masq(image.in(0),masq,p0),som_masq(image_2.in(0),masq,p0))
           -som_masq(Virgule(image.in(0),image_2.in(0)),masq,p0)
        ),
        Virgule(VMax(dif), VMax(dif2))
    );

    BENCH_ASSERT((dif<epsilon)&&(dif2<epsilon));
}

void bench_som_masq()
{
     
     bench_som_masq
     (
          (INT *)0,
          true,
          Pt2di(-3,-4),
          Pt2di(8,9),
          Pt2di(20,30)
     );

     bench_som_masq
     (
          (INT *)0,
          true,
          Pt2di(-3,-4),
          Pt2di(7,9),
          Pt2di(20,30)
     );
     
     bench_som_masq
     (
          (REAL *)0,
          false,
          Pt2di(-3,-4),
          Pt2di(8,9),
          Pt2di(20,30)
     );

     bench_som_masq
     (
          (REAL *)0,
          false,
          Pt2di(-3,-4),
          Pt2di(7,9),
          Pt2di(20,30)
     );
}


template <class Type> void bench_rle_som_masq
                           (
                               Type *,
                               Pt2di p0,
                               Pt2di sz_masq,
                               Pt2di sz_im
                           )
{

    Im2D_U_INT1   m1(sz_masq.x,sz_masq.y);
    ELISE_COPY(m1.all_pts(),frandr()>0.5,m1.out());

    Im2D<REAL,REAL>   m2(sz_masq.x,sz_masq.y);
    ELISE_COPY(m1.all_pts(),m1.in(),m2.out());

    Im2D<Type,Type> I(sz_im.x,sz_im.y);

    ELISE_COPY(I.all_pts(),10*frandr(),I.out());

    // ELISE_COPY(I.all_pts(),1,I.out());

    Type dif;
    ELISE_COPY
    (
        I.all_pts(),
        Abs
        (
             som_masq(I.in(1),m2,p0)
           - rle_som_masq_binaire(Virgule(I.in(1),1),m1,0,p0)
        ),
        VMax(dif) 
    );
    BENCH_ASSERT(dif<epsilon);

    ELISE_COPY
    (
        I.all_pts(),
        Abs
        (
             som_masq(I.in(1),m2)
           - rle_som_masq_binaire(Virgule(I.in(1),1),m1,0)
        ),
        VMax(dif) 
    );
    BENCH_ASSERT(dif<epsilon);

    Im2D_U_INT1  Isel(sz_im.x,sz_im.y);
    ELISE_COPY(Isel.all_pts(),frandr()>0.5,Isel.out());

    Im2D<Type,Type> Ir1(sz_im.x,sz_im.y);
    ELISE_COPY
    (
        I.all_pts(),
        rle_som_masq_binaire(Virgule(I.in(0),Isel.in(0)),m1,5,p0),
        Ir1.out()
    );

    Im2D<Type,Type> Ir2(sz_im.x,sz_im.y);
    ELISE_COPY
    (
        I.all_pts(),
        rle_som_masq_binaire(Virgule(I.in(0),1),m1,0,p0),
        Ir2.out()
    );
    ELISE_COPY
    (
        select(I.all_pts(),!Isel.in()),
        5,
        Ir2.out()
    );

    ELISE_COPY
    (
        I.all_pts(),
        Abs(Ir1.in()-Ir2.in()),
        VMax(dif) 
    );
    BENCH_ASSERT(dif<epsilon);
}




void bench_rle_som_masq()
{
    bench_rle_som_masq
    (
          (INT *) 0,
          Pt2di(-3,-4),
          Pt2di( 9, 11),
          Pt2di(22,33)
    );
    bench_rle_som_masq
    (
          (REAL *) 0,
          Pt2di(-3,-4),
          Pt2di( 9, 11),
          Pt2di(22,33)
    );
}

//*******************************************

void bench_maj_etiq
     (
           Pt2di  sz,
           Pt2di  brd,
           Box2di  b,
           INT     nb_etiq,
           bool    Complexe
     )
{



     Im2D_U_INT1  I(sz.x,sz.y);

     Flux_Pts rect = rectangle(brd,sz-brd);

     ELISE_COPY
     (
         I.all_pts(),
         Min(nb_etiq-1,Iconv(nb_etiq*frandr())),
         I.out() 
      );

     Im2D_U_INT1 I_et(sz.x,sz.y,0);
     ELISE_COPY
     (
        rect,
        label_maj(I.in_proj(),nb_etiq,b,1,Complexe),
        I_et.out() 
     );



     Im2D_U_INT2 Max_Nb_Et(sz.x,sz.y);
     {
         Fonc_Num f0 = 0;
         for(int e=0 ; e<nb_etiq ; e++)
            f0 = Max(f0,rect_som(I.in_proj()==e,b));

         ELISE_COPY
         (
              Max_Nb_Et.all_pts(),
              f0,
              Max_Nb_Et.out()
         );
     }


     for(int e=0 ; e<nb_etiq ; e++)
     {
        INT dif;
        ELISE_COPY
        (
              rect,
                   I_et.in() == e
              &&   (rect_som(I.in_proj()==e,b) != Max_Nb_Et.in()),
              sigma(dif) 
        );
        BENCH_ASSERT(dif == 0);
     }
}




void bench_maj_etiq(bool Complexe)
{
     bench_maj_etiq
     (
          Pt2di(250,210),
          Pt2di(5,5),
          Box2di(Pt2di(-12,-12),Pt2di(12,12)),
          20,
          Complexe
    );

     bench_maj_etiq
     (
          Pt2di(100,120),
          Pt2di(0,0),
          Box2di(Pt2di(0,0), Pt2di(0,0)),
          4,
          Complexe
    );


     bench_maj_etiq
     (
          Pt2di(100,120),
          Pt2di(2,2),
          Box2di(Pt2di(-1,-1), Pt2di(1,1)),
          20,
          Complexe
    );



     bench_maj_etiq
     (
          Pt2di(100,120),
          Pt2di(0,0),
          Box2di(Pt2di(10,10),Pt2di(10,10)),
          20,
          Complexe
    );

}
void bench_maj_etiq()
{
     bench_maj_etiq(true);
     bench_maj_etiq(false);
}

void bench_etiq()
{
    bench_maj_etiq();
}


//*******************************************

void bench_dilate_etiq
     (
           INT dist,
           Pt2di  sz,
           Pt2di ,
           Box2di  b,
           INT     nb_etiq,
           const Chamfer & chamf
     )
{
     Im2D_INT4  I(sz.x,sz.y);

     Flux_Pts rect = I.all_pts(); // rectangle(brd,sz-brd);

     ELISE_COPY
     (
         I.all_pts(),
         Min(nb_etiq-1,Iconv(nb_etiq*frandr())),
         I.out() 
      );


     Fonc_Num f = Min(nb_etiq-1,Iconv(nb_etiq*frandr()));
     ELISE_COPY
     (
        rect,
        label_maj
        (
           label_maj
           (
               label_maj(f,nb_etiq,Box2di(Pt2di(-5,-5),Pt2di(5,5))),
               nb_etiq,b
           ),
           nb_etiq,b
        ),
        I.out() 
     );

     ELISE_COPY
     (
          I.all_pts(),
         (I.in()%2)* (I.in()/2),
          I.out()
     );

     ELISE_COPY
     (
        I.border(chamf.radius()+1),
        1,
        I.out() 
     );



     Im2D_U_INT1  I0(sz.x,sz.y);
     INT vm;
     ELISE_COPY(I.all_pts(),I.in(),I0.out() | VMax(vm));
     vm++;


     Im2D_U_INT1  Id(sz.x,sz.y);
     chamf.dilate_label(Id,I,dist);


     ELISE_COPY
     (
          Id.all_pts(),
          I0.in() == 0,
          Id.out() 
     );
     chamf.im_dist(Id);

    ELISE_COPY(I0.border(chamf.radius()),vm,I0.out());


    U_INT1 ** i0 = I0.data();
    U_INT1 ** id = Id.data();
    INT4   ** lab = I.data();

    const INT * pds     = chamf.pds();
    const Pt2di * vois  = chamf.neigh();
    const INT nbv       = chamf.nbv();

    for (int y=0 ; y<sz.y ; y++)
        for (int x=0 ; x<sz.x ; x++)
        {
            if (i0[y][x] == vm)
            {
            }
            else if (id[y][x] == 0)
            {
                BENCH_ASSERT(lab[y][x] == i0[y][x]);
            }
            else if (id[y][x] >= dist)
            {
                BENCH_ASSERT(lab[y][x] == 0);
            }
            else
            {
                bool got = false;
                INT l0 = lab[y][x];
                INT d0 = id[y][x];
                for (int iv= 0; iv<nbv ; iv++)
                {
                    INT xv = x + vois[iv].x;
                    INT yv = y + vois[iv].y;

                    if 
                    (
                          (lab[yv][xv] == l0)
                       && (id[yv][xv] + pds[iv] == d0)
                    )
                         got = true;
                }
                BENCH_ASSERT(got);
            }
        }



}



void bench_dilate_etiq()
{
     bench_dilate_etiq
     (
          25,
          Pt2di(650,610),
          Pt2di(5,5),
          Box2di(Pt2di(-7,-7),Pt2di(7,7)),
          20,
          Chamfer::d32
    );
}

template <class Type>  void bench_fonc_a_trou(Type *)
{
    Pt2di SZ(200,300);

    Pt2di P0(3,5);
    Pt2di P1(196,288);


    Liste_Pts<Type,INT> lpt(5);

    {
       Liste_Pts<Type,INT> l0(5);
       ELISE_COPY
       (
          rectangle(0,5000).chc
          (
             Iconv
             (Virgule(
                  frandr() * (SZ.x -1),
                  frandr() * (SZ.y-1),
                  frandr() * 1000,
                  frandr() * 1000,
                  frandr() * 1000
             ))
          ),
          0,
          l0
       );
       Im2D_U_INT2 H(SZ.x,SZ.y,0);
       ELISE_COPY(l0.all_pts(),1,H.histo().chc(Virgule(FX,FY)));
       ELISE_COPY(select(l0.all_pts(),H.in()[Virgule(FX,FY)]==1),1,lpt);
    }


    Im2D_INT4  I1(SZ.x,SZ.y);
    Im2D_INT4  I2(SZ.x,SZ.y);

    ELISE_COPY
    (
       I1.all_pts(),
       Virgule(frandr(),frandr()) *200,
       Virgule(I1.out(),I2.out())
    );

    Im2D_INT4  IV1(SZ.x,SZ.y);
    Im2D_INT4  IV2(SZ.x,SZ.y);
    Im2D_INT4  IV3(SZ.x,SZ.y);

    ELISE_COPY
    (
       I1.all_pts(),
       Virgule(I1.in(),I2.in(),I2.in()),
       Virgule(IV1.out(),IV2.out(),IV3.out())
    );

    ELISE_COPY
    (
         select
         (
            lpt.all_pts(),
                  (P0.x <= FX) &&  (FX   < P1.x)
              &&  (P0.y <= FY) &&  (FY   < P1.y)
         ),
         Virgule(
            kth_coord(2),
            kth_coord(3),
            kth_coord(4)
         ),
         Virgule(IV1.out(),IV2.out(),IV3.out()).chc(Virgule(FX,FY))
    );

    
    INT dif[3];

    ELISE_COPY
    (
         rectangle(P0,P1),
         Abs
         (
              fonc_a_trou(Virgule(I1.in(),I2.in()),lpt)
            - Virgule(IV1.in(),IV2.in(),IV3.in())
         ),
         VMax(dif,3)
    );

   BENCH_ASSERT
   (
           (dif[0] == 0)
        && (dif[1] == 0)
        && (dif[2] == 0)
   );
}

void bench_fonc_a_trou()
{
     bench_fonc_a_trou((INT *)0);
     bench_fonc_a_trou((U_INT2 *)0);
}

//*******************************************

void bench_op_buf_3()
{

    All_Memo_counter MC_INIT;
    stow_memory_counter(MC_INIT);

    bench_rle_som_masq();
    bench_fonc_a_trou();
    bench_dilate_etiq();
   
    bench_etiq();
    bench_som_masq();

    verif_memory_state(MC_INIT);
    cout << "OK bench_op_buf_3 \n";
}
