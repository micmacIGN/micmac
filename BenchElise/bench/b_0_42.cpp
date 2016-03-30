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

Pt2dr ab_0_42_PRand()
{
   return Pt2dr(NRrandom3()-0.5,NRrandom3()-0.5)*100;
}

template <const INT b> class  BenchFixed : public  ElFixed<b>
{
    public :

        static void bench_get();
        static void bench_seg();
        static void bench_seg(INT NBPts);
};
REAL DMAX = 0;


void bench_som_seg(Pt2di sz)
{
    Im2D_INT1        im(sz.x,sz.y);
    TIm2D<INT1,INT>  Tim(im);

    ELISE_COPY
    (
       im.all_pts(),
       gauss_noise_3(5) > 0,
       im.out()
    );
    Tim.algo_dist_32_neg();

     for(INT K=0; K<1000; K++)
     {
         Pt2dr p1 = Pt2dr(3,3) + Pt2dr(NRrandom3()*(sz.x-6),NRrandom3()*(sz.y-6));
         Pt2dr p2 = Pt2dr(3,3) + Pt2dr(NRrandom3()*(sz.x-6),NRrandom3()*(sz.y-6));

          INT NbPts = 1 << ((INT) (1 + 10 * ElSquare(NRrandom3())));

         REAL v0 = FixedSomSegDr(Tim,p1,p2,NbPts,1e20);

         REAL v1;
         Symb_FNum pds2 (FX/(REAL)NbPts);
         Symb_FNum pds1 (1-pds2);
         ELISE_COPY
         (
            rectangle(0,1+NbPts),
            im.in()
            [Virgule( 
                 (pds1*p1.x + pds2*p2.x),
                 (pds1*p1.y + pds2*p2.y)
            )],
            sigma(v1)
         );

         REAL dif = (ElAbs(v0-v1) * 256) /NbPts;
         BENCH_ASSERT(dif<6.0);
     }
}






template <const INT b> void BenchFixed<b>::bench_get()
{
     Im2D_INT1 im(100,100);
     TIm2D<INT1,INT> Tim(im);
     ELISE_COPY
     (
          im.all_pts(),
          10 * (frandr() - 0.5),
          im.out()
     );

     Im1D_REAL8 tx(10000);
     Im1D_REAL8 ty(10000);

     ELISE_COPY
     (
          tx.all_pts(),
          3 + 94 * Virgule(frandr(),frandr()),
          Virgule(tx.out(),ty.out())
     );

     Im1D_REAL8 val(10000);
     ELISE_COPY
     (
         tx.all_pts(),
         im.in()[Virgule(tx.in(),ty.in())],
         val.out()
     );

     REAL * x  = tx.data();
     REAL * y  = ty.data();
     REAL * v  = val.data();

     REAL difmax = (22.0 / (1<<b));

     for (INT k=0; k<10000;k++)
     {
         REAL v0 = v[k];
         ElPFixed<b> p = Pt2dr(x[k],y[k]);
         REAL v1 = TImGet<INT1,INT,b>::getb2(Tim,p) / (REAL) (1 << (2*b));
         REAL dif = ElAbs(v0-v1);

         BENCH_ASSERT(dif < difmax);
     }

}





template <const INT b> void BenchFixed<b>::bench_seg(INT NBPts)
{
    Pt2dr pr1 = ab_0_42_PRand();
    Pt2dr pr2 = ab_0_42_PRand();

    ElPFixed<b> pf1 (pr1);
    ElPFixed<b> pf2 (pr2);

    ElSegIter<b> Seg(pf1,pf2,NBPts);

    ElPFixed<b> pcur;
    INT k=0;
    REAL dmax = 1/ ((1<<(b-1)) -1.0);
    while(Seg.next(pcur))
    {
        Pt2dr prc = barry(k/((REAL)NBPts),pr2,pr1);
        REAL dist = dist8(prc-pcur.Pt2drConv());
         k++;
        BENCH_ASSERT(dist < dmax);
    }
    BENCH_ASSERT(k == NBPts +1);
}

template <const INT b> void BenchFixed<b>::bench_seg()
{
     for (INT k=0 ; k < 10 ; k++)
        bench_seg(1<<k);
}


void   bench_seg()
{
    bench_som_seg(Pt2di(200,170));
    bench_som_seg(Pt2di(200,270));

    for (int k=0;k<10;k++)
    {
        BenchFixed<2>::bench_seg();
        BenchFixed<4>::bench_seg();
        BenchFixed<6>::bench_seg();
        BenchFixed<8>::bench_seg();
        BenchFixed<9>::bench_seg();
    }

    BenchFixed<8>::bench_get();
    BenchFixed<10>::bench_get();
    BenchFixed<12>::bench_get();
}

void bench_fixed()
{
     bench_seg();
}

#if ELISE_X11
Pt2di PRand(Pt2di sz)
{
   return Pt2di
          (
	     Pt2dr
	     (
                1+NRrandom3()*(sz.x-2),
                1+NRrandom3()*(sz.y-2)
	     )
          );
}
void bench_optim_seg_dr()
{
     Pt2di SZ(500,500);

     Disc_Pal       Pdisc  = Disc_Pal::P8COL();

     Elise_Set_Of_Palette SOP(NewLElPal(Pdisc));
     Video_Display Ecr((char *) NULL);
     Ecr.load(SOP);

     Video_Win   W  (Ecr,SOP,Pt2di(50,50),SZ);       
     
     Im2D_INT1 Im (SZ.x,SZ.y,0);

     for (INT k = 0; k < 10 ; k++)
     {
         ELISE_COPY 
         ( 
            line(PRand(SZ),PRand(SZ)),
            1,
            Im.out()
         );
     }
     
     ELISE_COPY ( W.all_pts(), Im.in(), W.odisc());

     {
        TIm2D<INT1,INT> TIm (Im);
        ElTimer chrono;
        TIm.algo_dist_32_neg();
        cout << "Time d32 neg " << chrono.uval() << " " << chrono.sval() << "\n";
     }


     for(int k =0;k<1;k++)
     {
         Pt2dr p0 =  W.clik_in()._pt;
         Pt2dr p1 =  W.clik_in()._pt;
         W.draw_seg(p0,p1,Pdisc(P8COL::blue));

         REAL score;
         Seg2d S;

         S = OptimizeSegTournantSomIm
                   ( 
                        score,
                        Im,
                        Seg2d(p0,p1),
                        32,
                        2.0,
                        0.2,  true,true
                   );


         W.draw_seg(S.p0(),S.p1(),Pdisc(P8COL::red));

        // Pt2dr  q0 =  prolongt_std_uni_dir(Im,S,1.0,-0.5);
        Seg2d S2 =  retr_prol_std_bi_dir(Im,S,1.0,-0.5);
        W.draw_seg(S2.p0(),S2.p1(),Pdisc(P8COL::green));
     }
}
#else
void bench_optim_seg_dr()
{
}
#endif
   /***********************************************************/

#if ELISE_unix
#include <sys/time.h>
#include <unistd.h> 
#else
#include <time.h>
#endif
       
void  bench_hongrois(INT nb)
{

    Im2D_INT4   cost(nb,nb,1000000);
    ELISE_COPY
    (
         rectangle(0,3*nb).chc(Iconv(Min(nb-1,Virgule(nb*frandr(),nb*frandr())))),
         1000 * frandr(),
         cost.out()
    );

    ElTimer chrono; chrono.reinit();
    // REAL t0 = ElTimeOfDay();
    
    // Im1D_INT4  aff = hongrois(cost);   
    Im1D_INT4  aff(nb);
    cout <<  " nb = " << nb << "\n";
    ALGOHONGR(cost,aff);
   
/*
    REAL t1 =  (ElTimeOfDay() - t0) ;
    REAL tam = t1 / pow(nb/300.0,3);
    cout << "NB = " << nb << " ; Time = "  << t1 << " ; T/N3 = " << tam << "\n";
*/
/*
    for (INT k = 0 ; k< nb ; k++)
        cout << k << " " << aff.data()[k] << "\n";
*/

}

void  bench_hongrois()
{
  All_Memo_counter MC_INIT;
  stow_memory_counter(MC_INIT);    
  for (INT k = 10 ; k < 400 ; k+= 30) 
      bench_hongrois(k);
   bench_hongrois(10);
   
  verif_memory_state(MC_INIT);
}


   /***********************************************************/


void bench_vecteur_raster_0()
{
    bench_optim_seg_dr();

    bench_hongrois();
    bench_fixed();
    printf("OK  bench_vecteur_raster_0\n");
}





