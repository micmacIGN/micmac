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


void  bench_algo_dist_32(Pt2di sz)
{
    Im2D_U_INT1  I0(sz.x,sz.y); 
    Im2D_U_INT1  Id_plus(sz.x,sz.y); 
    Im2D_U_INT1  Id_moins(sz.x,sz.y); 
    Im2D_INT1    Idneg1(sz.x,sz.y); 
    Im2D_INT1    Idneg2(sz.x,sz.y); 

    ELISE_COPY
    (
         I0.all_pts(),
         gauss_noise_1(10)>0.5,
         I0.out() 
    );
    ELISE_COPY(I0.border(2),0,I0.out());

    ELISE_COPY
    (
         I0.all_pts(),
         extinc_32(I0.in(0),255),
         Id_plus.out()
    );
    ELISE_COPY
    (
         I0.all_pts(),
         extinc_32(! I0.in(0),255),
         Id_moins.out()
    );

    ELISE_COPY
    (
        select(I0.all_pts(),I0.in()),
        Min(Id_plus.in()-1,127),
        Idneg1.out()
    );
    ELISE_COPY
    (
        select(I0.all_pts(),! I0.in()),
        Max(1-Id_moins.in(),-128),
        Idneg1.out()
    );

    ELISE_COPY(I0.all_pts(),I0.in(),Idneg2.out());
    TIm2D<INT1,INT> TIdneg2(Idneg2);
    TIdneg2.algo_dist_32_neg();


    INT dif;
    ELISE_COPY
    (
         I0.all_pts(),
         Abs(Idneg2.in()-Idneg1.in()),
         VMax(dif)
    );
    BENCH_ASSERT(dif==0);




    ELISE_COPY
    (
        Idneg1.all_pts(),
        Iconv((gauss_noise_1(10)-0.5)* 20),
        Idneg1.out() | Idneg2.out()
    );
    ELISE_COPY(Idneg1.border(2),-100, Idneg1.out() | Idneg2.out());
    ELISE_COPY
    (
        Idneg1.all_pts(),
        128-EnvKLipshcitz_32(128-Idneg1.in(-100),100),
        Idneg1.out()
    );
    TIdneg2.algo_dist_env_Klisp32_Sup();

    INT Max2 [2];
    INT Min2 [2];

    ELISE_COPY
    (
         I0.interior(1),
         Virgule(Idneg1.in(),Idneg2.in()),
         VMax(Max2,2) | VMin(Min2,2)
    );

    ELISE_COPY
    (
         I0.interior(1),
         Abs(Idneg2.in()-Idneg1.in()),
         VMax(dif)
    );

    BENCH_ASSERT(dif==0);
}

void  bench_algo_dist_32()
{
	 bench_algo_dist_32(Pt2di(10,10));
     bench_algo_dist_32(Pt2di(100,100));
     bench_algo_dist_32(Pt2di(150,100));
     bench_algo_dist_32(Pt2di(100,150));
}



void bench_im_reech
     (
           Fonc_Num   Fonc,
           Pt2di      SzIm,
           Fonc_Num   reechantX,
           Fonc_Num   reechantY,
           INT        sz_grid,
           REAL       aMaxDif
    )
{
    Im2D_U_INT1 AnIm(SzIm.x,SzIm.y);

    ELISE_COPY(AnIm.all_pts(),Fonc,AnIm.out());
    REAL dif;

    ELISE_COPY
    (
        AnIm.interior(3),
        Abs
        (
             AnIm.ImGridReech (reechantX,reechantY,sz_grid,-100)
           - Fonc[Virgule(reechantX,reechantY)]
        ),
        VMax(dif)
    );

    BENCH_ASSERT(dif<aMaxDif);
}

void bench_im_reech()
{
    bench_im_reech
    (
        FX+FY,
        Pt2di(50,50),
        FX , 
        FY , 
        4,
        epsilon
    );
    bench_im_reech
    (
        FX*3+FY*2+1,
        Pt2di(50,50),
        FX/2.0+FY/4.0 +3.0,
        FX/4.0+FY/2.0 +4.5,
        4,
        epsilon
     );

    bench_im_reech
    (
        FX*3+FY*2+1,
        Pt2di(50,50),
        FX/2.1+FY/4.8 +8.1,
        FX/4.0+FY/2.0 +4.5,
        9,
        1/10.0
     );
}

void BenchcDbleGrid
     (
         Pt2dr aP0In,Pt2dr aP1In,
         REAL               aStepDir,
         ElDistortion22_Gen & aDist
     )
{
     //cDbleGrid aDGr(aP0In,aP1In,aStepDir,aDist);
     Pt2dr stepDir2(aStepDir,aStepDir);                // __NEW
     cDbleGrid aDGr(false,aP0In,aP1In,stepDir2,aDist); // __NEW

     for (REAL aX = aP0In.x ; aX<aP1In.x ; aX += aStepDir)
         for (REAL aY = aP0In.y ; aY<aP1In.y ; aY += aStepDir)
         {
             REAL x = aX + NRrandom3() * aStepDir;
             SetInRange(aP0In.x,x,aP1In.x);
             REAL y = aY + NRrandom3() * aStepDir;
             SetInRange(aP0In.y,y,aP1In.y);

	     Pt2dr aP(x,y);
	     Pt2dr aQ0 = aDist.Direct(aP);
	     Pt2dr aQ1 = aDGr.Direct(aP);
	     Pt2dr aR0 = aDist.Inverse(aQ0);
	     Pt2dr aR1 = aDist.Inverse(aQ1);

	     REAL aDQ = euclid(aQ0,aQ1);
	     REAL aDR = euclid(aR0,aP) +  euclid(aR1,aP);
	     aDQ /= ElSquare(aStepDir);
	     aDR /= ElSquare(aStepDir);
	     BENCH_ASSERT(aDQ<0.1);
	     BENCH_ASSERT(aDR<0.1);
         }
}

void BenchcDbleGrid()
{
    ElDistRadiale_PolynImpair aPol(2.0,Pt2dr(0.1,0.1));

    aPol.PushCoeff(0.05);
    aPol.PushCoeff(-0.01);

    BenchcDbleGrid(Pt2dr(-1.01,-0.95),Pt2dr(0.98,1.02),0.01,aPol);
}


void bench_im_tpl_0()
{
	BenchcDbleGrid();
     bench_im_reech();

     bench_algo_dist_32();
     printf("OK  bench_im_tpl_0 \n");
}


void bench_Proj32()
{
    Pt2di aSz(200,300);
    Pt2di aC = aSz/2;
    double aR = 50;
    Fonc_Num aF = Square(FX-aC.x)+Square(FY-aC.y) < ElSquare(aR);
    // aF = 0;

    Im2D_INT2 aI1(aSz.x,aSz.y);
    ELISE_COPY
    (
        aI1.all_pts(),
          aF * Polar_Def_Opun::polar(Virgule(FX-aC.x,FY-aC.y),0).v1()*(255.0/(2.0*PI))
        + (1-aF) * (frandr() * 255),
        aI1.out()
    );

   Video_Win aW= Video_Win::WStd(aSz,1.0);
   ELISE_COPY(aI1.all_pts(),aI1.in(),aW.ocirc());

   getchar();
   cResProj32  aR32 = Projection32(aF,aSz);
   ELISE_COPY
   (
         aI1.all_pts(),
         aI1.in()[Virgule(aR32.PX().in(),aR32.PY().in())],
         aW.ocirc()
   );
   getchar();
}







