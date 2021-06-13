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

/*
TPL : 2.74
ELISE DYN : 4.8
Tpl dup : 2.85
Tpl BOV  : 1.28
Tpl BOV (inv)  : 1.78
Tpl BOV memcpy  : 0.27
TPL PIPE : 3.49
*/


#include "StdAfx.h"
#include "general/geom_vecteur.h"
#include "bench.h"

static INT  DifImages
            (
              Im2D_U_INT1 i1,
              Im2D_U_INT1 i2
            )
{
     INT dif;
     ELISE_COPY
     (
          i1.all_pts(),
          Abs(i1.in()-i2.in()),
          VMax(dif)
     );
     return dif;
}

void bench_time_tpl_elise
     (
         Im2D_U_INT1       I1,
         TIm2D<U_INT1,INT> TI1
     )
{
         Pt2di  SZ(I1.tx(),I1.ty());
         ElTimer chrono;
         INT nb = 300;
		 INT k;
         for ( k = 0; k<nb; k++)
             TElCopy
             (
                    TI1.all_pts(),
                    TI1,
                    TI1
             );

         cout << "TPL : " << chrono.uval() << "\n";

         chrono.reinit();
         for ( k = 0; k<nb; k++)
             ELISE_COPY
             (
                    I1.all_pts(),
                    I1.in(),
                    I1.out()
             );
         cout << "ELISE DYN : " << chrono.uval() << "\n";


         chrono.reinit();
         for ( k = 0; k<nb; k++)
             Tdup(TI1,TI1);
         cout << "Tpl dup : " << chrono.uval() << "\n";

         chrono.reinit();
         for ( k = 0; k<nb; k++)
         {
              U_INT1 ** i0 = I1.data();
              U_INT1 ** i2 = I1.data();
              for (INT y= 0; y<SZ.y ; y++)
                  for (INT x= 0; x<SZ.x ; x++)
                      i0[y][x] = i2[y][x];
         }
         cout << "Tpl BOV  : " << chrono.uval() << "\n";

         chrono.reinit();
         for ( k = 0; k<nb; k++)
         {
              U_INT1 ** i0 = I1.data();
              U_INT1 ** i2 = I1.data();
              for (INT x= 0; x<SZ.x ; x++)
                  for (INT y= 0; y<SZ.y ; y++)
                      i0[y][x] = i2[y][x];
         }
         cout << "Tpl BOV (inv)  : " << chrono.uval() << "\n";

         chrono.reinit();
         for ( k = 0; k<nb; k++)
         {
              U_INT1 ** i0 = I1.data();
              U_INT1 ** i2 = I1.data();
              for (INT y= 0; y<SZ.y ; y++)
                  memcpy(i0[y],i2[y],SZ.x*sizeof(U_INT1));
         }
         cout << "Tpl BOV memcpy  : " << chrono.uval() << "\n";


         chrono.reinit();
         for ( k = 0; k<nb; k++)
             TElCopy
             (
                    TI1.all_pts(),
                    TI1,
                    TPipe(TI1,TI1)
             );
         cout << "TPL PIPE : " << chrono.uval() << "\n";
}


static Pt2di  PRandI()
{
     //return (Pt2dr(NRrandom3(),NRrandom3()) - Pt2dr(0.5,0.5)) * 1000;
     return Pt2di( (Pt2dr(NRrandom3(),NRrandom3()) - Pt2dr(0.5,0.5)) * 1000 ); // __NEW
}

static bool  PileOuFace() {return NRrandom3() > 0.5;}

void bench_TDigiline()
{
     for (INT NB =0; NB <1000; NB ++)
     {
          Pt2di p0     = PRandI();
          Pt2di p1     = PRandI();
          bool cx8     = PileOuFace();
          bool wlast   = PileOuFace();

          ElFilo<Pt2di> F;

          TElCopy
          (
               TFlux_Line2d(p0,p1,cx8,wlast),
               TCste(1),
               TPushPt(F)
          );

          BENCH_ASSERT (p0 == F[0]);
          if (wlast)
             BENCH_ASSERT (p1 == F.top());
          else
          {
               if (cx8)
                  BENCH_ASSERT(1 == dist8(p1-F.top()));
               else
                  BENCH_ASSERT(1 == dist4(p1-F.top()));
          }

          BENCH_ASSERT 
          ( F.nb() == (cx8 ? dist8(p1-p0) : dist4(p1-p0)) + (wlast?1:0));

		   INT k;
           for ( k = 1; k<F.nb() ; k++)
           {
               if (cx8)
                  BENCH_ASSERT(1 == dist8(F[k-1]-F[k]));
               else
                  BENCH_ASSERT(1 == dist4(F[k-1]-F[k]));
           }
	   
           //SegComp SEG(p0,p1);
	   Pt2dr p0_r(p0); // __NEW
           Pt2dr p1_r(p1); // __NEW
           //SegComp SEG( Pt2dr(p0), Pt2dr(p1) ); devrait marcher mais non oO
           SegComp SEG( p0_r, p1_r );                                   // __NEW
           for ( k = 0; k<F.nb() ; k++)                                 // __NEW
               BENCH_ASSERT(SEG.square_dist_droite( Pt2dr(F[k]) ) < 1); // __NEW
     }
}

static Pt2di  PRandI(Pt2di SZ)
{
     //return Pt2dr(NRrandom3(),NRrandom3()).mcbyc(Pt2dr(SZ.x-1,SZ.y-1));
     return Pt2di( Pt2dr(NRrandom3(),NRrandom3()).mcbyc(Pt2dr(SZ.x-1,SZ.y-1)) ); // __NEW
}

void bench_Telcopy_0()
{
     bench_TDigiline();




     Pt2di SZ(100,300);

     Im2D_U_INT1 I0(SZ.x,SZ.y);
     ELISE_COPY(I0.all_pts(),255*frandr(),I0.out());

     Im2D_U_INT1 I1(SZ.x,SZ.y);
     Im2D_U_INT1 I2(SZ.x,SZ.y);


     TIm2D<U_INT1,INT> TI0(I0);
     TIm2D<U_INT1,INT> TI2(I2);
     ELISE_COPY(I0.all_pts(),I0.in(),I1.out());




     TElCopy(TI2.all_pts(),TI0,TI2);

     INT dif;
     ELISE_COPY
     (
          I0.all_pts(),
          Abs(I0.in()-I2.in()),
          VMax(dif)
     );

     BENCH_ASSERT( DifImages(I1,I2) == 0);

     ELISE_COPY
     (
         rectangle(Pt2di(3,3),Pt2di(32,56)),
         25,
         I1.out()
     );
     TElCopy
     (
         TFlux_Rect2d(Pt2di(3,3),Pt2di(32,56)),
         TCste(25),
         TI2
     );

     BENCH_ASSERT( DifImages(I1,I2) == 0);


     ELISE_COPY(I0.all_pts(),255*frandr(),I0.out());
     for (INT k= 0; k < 100 ; k++)
     {
          Pt2di p0 = PRandI(SZ);
          Pt2di p1 = PRandI(SZ);
          ELISE_COPY(line(p0,p1),I0.in(),I1.out());
          TElCopy ( TFlux_Line2d(p0,p1),TI0,TI2);
     }
     cout << "LINE : " <<  DifImages(I1,I2) << "\n";
     BENCH_ASSERT( DifImages(I1,I2) == 0);




     {
           Im2D_Bits<2>    Isel(SZ.x,SZ.y);  
           TIm2DBits<2>    TIsel(Isel);

           ELISE_COPY(I0.all_pts(),255*frandr(),I0.out());
           ELISE_COPY(I0.all_pts(),frandr()>0.5,Isel.out());

           ELISE_COPY
           (
                select(Isel.all_pts(),Isel.in()),
                I0.in(),
                I1.out()
           );


           TElCopy
           (
                TSelect(TIsel.all_pts(),TIsel),
                TI0,
                TI2
           );

           BENCH_ASSERT( DifImages(I1,I2) == 0);

           
           TElCopy
           (
                TSelect(TIsel.all_pts(),TIsel),
                TMod(TPlus(TFX(),TFY()),TCste(253)),
                TI2
           );
           ELISE_COPY
           (
                select(Isel.all_pts(),Isel.in()),
                (FX+FY)%253,
                I1.out()
           );
           BENCH_ASSERT( DifImages(I1,I2) == 0);
     }

     {
         Im2D_U_INT1 K0(SZ.x,SZ.y);
         ELISE_COPY(K0.all_pts(),255*frandr(),K0.out());

         Im2D_U_INT1 K1(SZ.x,SZ.y);
         Im2D_U_INT1 K2(SZ.x,SZ.y);
         TIm2D<U_INT1,INT> TK0(K0);
         TIm2D<U_INT1,INT> TK2(K2);

         
         TElCopy
         (
                TI0.all_pts(),
                TCatF(TI0,TK0),
                TCatO(TK2,TI2)
         );

         ELISE_COPY
         (
                I0.all_pts(),
                Virgule(I0.in(),K0.in()),
                Virgule(K1.out(),I1.out())
         );


           BENCH_ASSERT( DifImages(I1,I2) == 0);
           BENCH_ASSERT( DifImages(K1,K2) == 0);


           ELISE_COPY(I0.all_pts(),255*frandr(),I0.out());
           TElCopy
           (
                TI0.all_pts(),
                TI0,
                TPipe(TK2,TI2)
           );

           ELISE_COPY
           (
                I0.all_pts(),
                I0.in(),
                (K1.out()|I1.out())
           );

           BENCH_ASSERT( DifImages(I1,I2) == 0);
           BENCH_ASSERT( DifImages(K1,K2) == 0);



     }

     ELISE_COPY(I0.all_pts(),255*frandr(),I0.out());

     {
        static Pt2di TP0 [6] = {Pt2di(33,35),Pt2di(3,3),Pt2di(3,3),Pt2di(3,3),Pt2di(3,3),Pt2di(3,3)};
        static Pt2di TP1 [6] = {Pt2di(56,87),Pt2di(3,3),Pt2di(3,4),Pt2di(5,5),Pt2di(4,3),Pt2di(4,4)};

        for (INT k=0; k< 6 ; k++)
        {
            INT v1,v2;
            INT s1,s2;
            ELISE_COPY
            (
               border_rect(TP0[k],TP1[k]),
               I0.in(),
               I1.out() | sigma(v1) | (sigma(s1) << 1)
            );
            TElCopy
            (
               TFlux_BordRect2d(TP0[k],TP1[k]),
               TI0,
               TPipe
               (
                    TPipe(TI2,TSigma(v2)),
                    TRedir(TSigma(s2),TCste(1))
               )
            );

            // pour k = 4,5, il y a un bug dans border_rect (=> 1 point, devrait etre 0)
            if (k<4)
               BENCH_ASSERT((s1==s2) && (v1==v2));
        }
     }
     BENCH_ASSERT( DifImages(I1,I2) == 0);



     bench_time_tpl_elise(I0,TI0);

	 printf("OK bench_Telcopy_0\n");
}










