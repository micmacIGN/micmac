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


bool egale_a_une_cste_pres(Im2D_U_INT1 i1,Im2D_U_INT1 i2)
{
    INT dif_00;
    ELISE_COPY(rectangle(Pt2di(0,0),Pt2di(1,1)),i1.in()-i2.in(),VMax(dif_00));

    INT diff;
    ELISE_COPY
    (
            i2.all_pts(),
            Abs(i1.in()-i2.in()-dif_00),
            VMax(diff)  
    );

    if (diff)
    {
        {
            int eq,neq;
            ELISE_COPY
            (
                i2.all_pts(),
                Virgule
                ( 
                      (i1.in()!= i2.in()+dif_00),
                      (i1.in()== i2.in()+dif_00)
                ),
                Virgule(sigma(neq),sigma(eq))
            );
            cout << "EQUAL " << eq << "; NOT EQUAL " << neq << "\n";
        }
        U_INT1 ** d1 = i1.data();
        U_INT1 ** d2 = i2.data();
        INT    tx = i1.tx();
        INT    ty = i2.ty();

        for (int y = 0; y<ty ; y++)
            for (int x = 0; x<tx ; x++)
                if (d1[y][x] != d2[y][x] + dif_00)
                {
                   cout     << x << " " << y << 
                        "=>" 
                            << (INT) d1[y][x]  << " "
                            << (INT) d1[y][x] -dif_00 << " "
                            << " " << (INT) d2[y][x] << "\n";
                   BENCH_ASSERT(0);
                }
    }

    return diff == 0;
}

void bench_rle_bitm_win_chc(Pt2di sz,Pt2dr tr,Pt2dr sc)
{
    INT nb_0 = 30;
    Gray_Pal       G30  (nb_0);
    Elise_Set_Of_Palette  SOP(ElList<Elise_Palette>() +Elise_Palette(G30));

    Bitm_Win   bwA ("toto",SOP,sz);
    Bitm_Win   bwB ("toto",SOP,sz);



    Im2D_U_INT1 IA = bwA.im();
    Im2D_U_INT1 IB = bwB.im();
    El_Window bw = bwA | bwB;

    Im2D_U_INT1 I2  (sz.x,sz.y);


    Fonc_Num fgr = (FX+FY) % 256; // (FX+FY)%256;
    bw = bw.chc(tr,sc);
    ELISE_COPY(bw.all_pts(),fgr,bw.ogray());

    ELISE_COPY
    (
         I2.all_pts(),
         (fgr[Virgule(round_ni(FX/sc.x+tr.x),round_ni(FY/sc.y+tr.y))]*nb_0)/256,
         I2.out()
    );

   INT v0 ;
   ELISE_COPY(I2.all_pts(),I2.in()-IA.in(),VMax(v0));

   INT nb,nb_dif;

    ELISE_COPY
    (
            I2.all_pts(),
            Abs(I2.in()-IA.in()-v0),
            VMax(nb_dif)  | (sigma(nb) << 1)
    );



    BENCH_ASSERT((nb_dif==0)&&(nb == sz.x * sz.y));
    BENCH_ASSERT(egale_a_une_cste_pres(IA,IB));


    // test with integer mode

    Fonc_Num f2 = (2*FX +4 +FX*FY)% 256;
    ELISE_COPY(select(bw.all_pts(),(FX+FY)%2),f2,bw.ogray());
    ELISE_COPY
    (
         select
         (
             I2.all_pts(),
             (round_ni(FX/sc.x+tr.x)+round_ni(FY/sc.y+tr.y))%2
         ),
         (f2[Virgule(round_ni(FX/sc.x+tr.x),round_ni(FY/sc.y+tr.y))]*nb_0)/256,
         I2.out()
    );

    BENCH_ASSERT(egale_a_une_cste_pres(IA,I2));
    BENCH_ASSERT(egale_a_une_cste_pres(IA,IB));
}

void bench_rle_bitm_win(Pt2di sz)
{
    INT nb_0 = 30;

    Gray_Pal       G30  (nb_0);
    BiCol_Pal      Prb  ( 
                               Elise_colour::black,
                               Elise_colour::red,
                               Elise_colour::blue,
                               10,
                               10
                        );
    Disc_Pal    G5  = Disc_Pal::P8COL();

    Elise_Set_Of_Palette  SOP(NewLElPal(G5)+Elise_Palette(Prb)+Elise_Palette(G30));

    Bitm_Win   bwA ("toto",SOP,sz);
    Bitm_Win   bwB ("toto",SOP,sz);

     El_Window bw = bwA | bwB;

    Im2D_U_INT1 IA = bwA.im();
    Im2D_U_INT1 IB = bwB.im();
    Im2D_U_INT1 I2  (sz.x,sz.y);

    {
        ELISE_COPY(bw.all_pts(),FX,bw.ogray());
        ELISE_COPY(bw.all_pts(),(FX*nb_0)/256,I2.out());

        BENCH_ASSERT(egale_a_une_cste_pres(IA,I2));
        BENCH_ASSERT(egale_a_une_cste_pres(IA,IB));
    }

    {
        ELISE_COPY(bw.all_pts(),0,bw.ogray());
        ELISE_COPY(bw.all_pts(),0/nb_0,I2.out());

        ELISE_COPY(select(bw.all_pts(),FX>FY),FX,bw.ogray());
        ELISE_COPY(select(bw.all_pts(),FX>FY),(FX*nb_0)/256,I2.out());

        BENCH_ASSERT(egale_a_une_cste_pres(IA,I2));
        BENCH_ASSERT(egale_a_une_cste_pres(IA,IB));
    }

    INT v0;
    {
       INT nb,nb_dif;

       Fonc_Num fgr = (FX+FY)%256;
       ELISE_COPY(bw.all_pts(),fgr,bw.ogray());
       v0 = IA.data()[0][0];

       ELISE_COPY(bw.all_pts(),(fgr*nb_0)/256+v0,I2.out());

       ELISE_COPY
       (
            bw.all_pts(),
            Abs(I2.in()-IA.in()),
            VMax(nb_dif)  | (sigma(nb) << 1)
       );

       BENCH_ASSERT((nb_dif==0)&&(nb == sz.x * sz.y));

        // Add lines


        for (int k=0 ; k < 100; k++)
        {
            Pt2di p0 (-k,-k);
            Pt2di p1 (3*k, 4*k);
            INT gr = k*2;

            ELISE_COPY(line(p0,p1),(gr*nb_0)/256+v0,I2.oclip());
            //bw.draw_seg(p0,p1,G30(gr));
            Pt2dr p0_r ( (REAL)p0.x, (REAL)p0.y ); // __NEW
            Pt2dr p1_r ( (REAL)p1.x, (REAL)p1.y ); // __NEW
            bw.draw_seg(p0_r,p1_r,G30(gr));        // __NEW
        }
        BENCH_ASSERT(egale_a_une_cste_pres(IA,I2));
        BENCH_ASSERT(egale_a_une_cste_pres(IA,IB));

    }


    INT v1;
    {
       INT nb,nb_dif;
       Pt2di p0 (10,11);
       Pt2di p1 (sz.x-13,sz.y-12);


       ELISE_COPY(rectangle(p0,p1) ,Virgule(FX,FY),bw.obicol());
       v1 = IA.data()[p0.y][p0.x];

       ELISE_COPY(rectangle(p0,p1) ,(FX*10)/256+ ((FY*10)/256)*10+v1,I2.out());

       ELISE_COPY
       (
            bw.all_pts(),
            Abs(I2.in()-IA.in()),
            VMax(nb_dif)  | (sigma(nb) << 1)
       );


       BENCH_ASSERT((nb_dif==0)&&(nb == sz.x * sz.y));
    }
    BENCH_ASSERT(v0 != v1);
    BENCH_ASSERT(egale_a_une_cste_pres(IA,IB));

}

// surtout fait pour traquer un bug x11
void bench_chc_bitm_win()
{
    INT Tail = 500;
    Pt2di sz(Tail,Tail);
    //Pt2dr c = sz / 2;
    Pt2dr c( Tail/2, Tail/2 ); // __NEW
    REAL  Ray = (Tail/2.0) * 0.95;
    REAL C = cos(1.0); 
    REAL S = sin(1.0); 

    Disc_Pal    G5  = Disc_Pal::P8COL();
    Elise_Set_Of_Palette  SOP(NewLElPal (G5));


    Bitm_Win   bwA ("toto",SOP,sz);
    Bitm_Win   bwB ("toto",SOP,sz);

    El_Window bw = bwA | bwB;

    Im2D_U_INT1 IA = bwA.im();
    Im2D_U_INT1 IB = bwB.im();

    Im2D_U_INT1 I2  (sz.x,sz.y);
    Im2D_U_INT1 ver  (sz.x,sz.y,0);


    ELISE_COPY
    (
         I2.all_pts(),
         P8COL::red,
         bw.out(G5) | I2.out() 
    );
    egale_a_une_cste_pres(I2,IA);
    BENCH_ASSERT(egale_a_une_cste_pres(IA,IB));


    ELISE_COPY
    (
         disc(c,Ray),
         P8COL::blue,
            ver.out()
         | (bw.out(G5) /*|W.out(G5)*/).chc
           (
             Iconv
             (Virgule(
                 (c.x + (FX-c.x) * C - (FY-c.y) * S),
                 (c.y + (FX-c.x) * S + (FY-c.y) * C)
             ))
           )
    );

    U_INT1 ** dv = ver.data();
    U_INT1 ** i2 = I2.data();

    for (INT x=0 ; x<sz.x ; x++)
         for (INT y=0 ; y<sz.y ; y++)
             if (dv[y][x])
             {
                 INT xi = (INT)(c.x+(x-c.x)*C-(y-c.y)*S);
                 INT yi = (INT)(c.y+(x-c.x)*S+(y-c.y)*C);
                 i2[yi][xi] = P8COL::blue;
             }


    egale_a_une_cste_pres(I2,IA);
    BENCH_ASSERT(egale_a_une_cste_pres(IA,IB));
}


/**************************************************************/

               // Elise_Palette::to_rgb

void bench_pal_to_rgb(Fonc_Num rgb1,Fonc_Num rgb2,INT nb)
{
     INT dr,dg,db;
     ELISE_COPY
     (
         rectangle(0,nb),
         Abs(rgb1-rgb2),
         Virgule(VMax(dr),VMax(dg),VMax(db))
     );

     BENCH_ASSERT((dr < 4) && (dg < 4) && (db < 4));
     cout << dr << " " << dg << " " << db << "\n";
}

void bench_Elise_Palette_to_rgb()
{

     INT nb = 1000;

     Im1D_U_INT1 r1(nb);
     Im1D_U_INT1 g1(nb);
     Im1D_U_INT1 b1(nb);

     Output  Orgb1 = Virgule(r1.out(),g1.out(),b1.out());
     Fonc_Num  Frgb1 = Virgule(r1.in(),g1.in(),b1.in());

     Im1D_U_INT1 r2(nb);
     Im1D_U_INT1 g2(nb);
     Im1D_U_INT1 b2(nb);

     Output  Orgb2 = Virgule(r2.out(),g2.out(),b2.out());
     Fonc_Num  Frgb2 = Virgule(r2.in(),g2.in(),b2.in());

    // bench gray 
    {
         Im1D_U_INT1 gray(nb);
         ELISE_COPY(gray.all_pts(),frandr()*255,gray.out());

         Gray_Pal Pgr(10);

         ELISE_COPY(gray.all_pts(),Virgule(gray.in(),gray.in(),gray.in()),Orgb1);
         ELISE_COPY(gray.all_pts(),Pgr.to_rgb(gray.in()),Orgb2);

         bench_pal_to_rgb(Frgb1,Frgb2,nb);
     }



    // Indexed pallette 

     {
         static const int nbc = 10;
         Elise_colour tabc[nbc];
		 INT c;

         for ( c =0 ; c< nbc ; c++)
             tabc[c] =  Elise_colour::rgb(NRrandom3(),NRrandom3(),NRrandom3());

         Im1D_U_INT1 indexe(nb);

         ELISE_COPY(indexe.all_pts(),mod(frandr()*10000,nbc),indexe.out());

         Disc_Pal Pdisc(tabc,nbc);

         for ( c=0;c<nbc ; c++)
         {
              Elise_colour ec = tabc[c];
              ELISE_COPY
              (
                   select(indexe.all_pts(),indexe.in() == c),
                   255*Fonc_Num(ec.r(),ec.g(),ec.b()),
                   Orgb1
              );
         }
         ELISE_COPY(indexe.all_pts(),Pdisc.to_rgb(indexe.in()),Orgb2);
         bench_pal_to_rgb(Frgb1,Frgb2,nb);
     }

}

void bench_bitm_colour(REAL i,REAL t,REAL s,Elise_colour exp)
{
     Elise_colour got = Elise_colour::its(i,t,s);
     REAL d = got.eucl_dist(exp);
     BENCH_ASSERT(d < 1e-7);
}

void TrueCol16Bit_RGB::Bench()
{
    Fonc_Num Rand1Canal =  Iconv(MasqBits*frandr())<< Complem8NbBit;
    Symb_FNum R (Rand1Canal);
    Symb_FNum G (Rand1Canal);
    Symb_FNum B (Rand1Canal);

    Symb_FNum rgb (Virgule(R,G,B));


    INT DifR,DifG,DifB;

    ELISE_COPY
    (
         rectangle(0,1000),
         Abs(I2RGB(RGB2I(rgb))-rgb),
         Virgule(VMax(DifR),VMax(DifG),VMax(DifB))
    );

    Symb_FNum Ind(frandr() * ((1<<15)-1));
    INT DifInd;

    ELISE_COPY
    (
         rectangle(0,1000),
         Abs(RGB2I(I2RGB(Ind))-Ind),
         VMax(DifInd)
    );
    BENCH_ASSERT(DifR==0);
    BENCH_ASSERT(DifG==0);
    BENCH_ASSERT(DifB==0);
    BENCH_ASSERT(DifInd==0);
}

void bench_bitm_colour()
{
    for (INT k=0 ; k<30 ; k++)
        TrueCol16Bit_RGB::Bench();

    // Verifie que its o (to_its) => identite
	REAL p1, p2, p3;

    for ( p1 =0.0 ; p1 < 1.0 ; p1 += 0.0333333)
        for ( p2 =0.0 ; p2 < 1.0 ; p2 += 0.0333)
            for ( p3 =0.0 ; p3 < 1.0 ; p3 += 0.03333)
            {
                   Elise_colour c = Elise_colour::rgb(p1,p2,p3);
                   REAL i,t,s;
                   c.to_its(i,t,s);
                   bench_bitm_colour(i,t,s,c);
            }

    //  Verifie que its donne le bon RVB pour les couleur pure
    
     bench_bitm_colour(1/3.0  ,  0.0    ,  1.0  ,  Elise_colour::rgb(1,0,0));
     bench_bitm_colour(1/3.0  ,  1/3.0  ,  1.0  ,  Elise_colour::rgb(0,1,0));
     bench_bitm_colour(1/3.0  ,  2/3.0  ,  1.0  ,  Elise_colour::rgb(0,0,1));
     bench_bitm_colour(1/3.0  ,  1.0    ,  1.0  ,  Elise_colour::rgb(1,0,0));
     bench_bitm_colour(1/3.0  ,  2.0    ,  1.0  ,  Elise_colour::rgb(1,0,0));

     bench_bitm_colour(2/3.0  ,  0.5    ,  1.0  ,  Elise_colour::rgb(0,1,1));
     bench_bitm_colour(2/3.0  ,  1/6.0  ,  1.0  ,  Elise_colour::rgb(1,1,0));
     bench_bitm_colour(2/3.0  ,  5/6.0  ,  1.0  ,  Elise_colour::rgb(1,0,1));


    //  Verifie que its donne le bon RVB pour les niveaux de gris

    for ( p1 =0.0 ; p1 < 1.0 ; p1 += 0.1)
        for ( p2 =0.0 ; p2 < 1.0 ; p2 += 0.1)
        {
            bench_bitm_colour(0.0,p1,p2,Elise_colour::rgb(0,0,0));
            bench_bitm_colour(1.0,p1,p2,Elise_colour::rgb(1,1,1));
            bench_bitm_colour(p1,p2,0,Elise_colour::rgb(p1,p1,p1));
        }

    //   Verif pour les versions Fonc_Num :

    static INT nb = 1000;

    Im1D_REAL8 r(nb);
    Im1D_REAL8 g(nb);
    Im1D_REAL8 b(nb);

    Im1D_REAL8 i(nb);
    Im1D_REAL8 t(nb);
    Im1D_REAL8 s(nb);

    ELISE_COPY
    (
         i.all_pts(),
         Virgule(frandr(),frandr(),frandr()),
         Virgule(i.out(),t.out(),s.out())
    );

    ELISE_COPY
    (
        i.all_pts(),
        its_to_rgb(255*Virgule(i.in(),t.in(),s.in()))/255,
        Virgule(r.out(),g.out(),b.out())
    );

    {
       for (INT k = 0; k < nb ; k++)
       {
            bench_bitm_colour
            (
               i.data()[k],t.data()[k],s.data()[k],
               Elise_colour::rgb(r.data()[k],g.data()[k],b.data()[k])
            );
       }
    }

    ELISE_COPY
    (
         i.all_pts(),
         Virgule(frandr(),frandr(),frandr())*255,
         Virgule(r.out(),g.out(),b.out())
    );
    REAL dr,dg,db;

     ELISE_COPY
     (
         i.all_pts(),
         Abs
         (
             its_to_rgb(rgb_to_its(Virgule(r.in(),g.in(),b.in())))-
             Virgule(r.in(),g.in(),b.in())
          ),
         Virgule(VMax(dr),VMax(dg),VMax(db))
     );

     BENCH_ASSERT
     (
              (dr < 1e-7)
           && (dg < 1e-7)
           && (db < 1e-7)
     );
}

void bench_bitm_win()
{
     bench_bitm_colour();


     bench_Elise_Palette_to_rgb();
     bench_chc_bitm_win();
     bench_rle_bitm_win(Pt2di(256,256));

     bench_rle_bitm_win(Pt2di(120,150));



      bench_rle_bitm_win_chc(Pt2di(120,150),Pt2dr(40,30),Pt2dr(2,2));
      bench_rle_bitm_win_chc(Pt2di(120,150),Pt2dr(40,30),Pt2dr(2,3));
      bench_rle_bitm_win_chc(Pt2di(120,150),Pt2dr(40.12,30.34),Pt2dr(2.1,3.4));
      bench_rle_bitm_win_chc(Pt2di(120,150),Pt2dr(40.12,30.34),Pt2dr(1,1));


    cout << "OK bench_bitm_win \n";
}
