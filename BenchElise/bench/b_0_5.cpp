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




/***************************************************/


void test_0_mul_compl(Pt2di sz,Fonc_Num Ax,Fonc_Num Ay,Fonc_Num Bx,Fonc_Num By)
{
     Im2D_REAL8  X1(sz.x,sz.y);
     Im2D_REAL8  Y1(sz.x,sz.y);
     Im2D_REAL8  X2(sz.x,sz.y);
     Im2D_REAL8  Y2(sz.x,sz.y);

     ELISE_COPY
     (  X1.all_pts(),
        Ax*Bx -Ay*By,
        X1.out()
     );

     ELISE_COPY
     (  X1.all_pts(),
        Ax*By + Ay*Bx,
        Y1.out()
     );


     ELISE_COPY
     (  X1.all_pts(),
        mulc(Virgule(Ax,Ay),Virgule(Bx,By)),
        Virgule(X2.out(),Y2.out())
     );


     REAL dif;

     ELISE_COPY
     (  X1.all_pts(),
        Abs(X1.in()-X2.in()) + Abs(Y1.in()-Y2.in()),
        sigma(dif)
     );

     BENCH_ASSERT(fabs(dif) < epsilon);

}

void test_mul_compl()
{
     test_0_mul_compl
     (
          Pt2di(103,109),
          FX+2.0*FY,
          Rconv(FX * FY),
          Rconv(FX % 25),
          10.0 /(FY+23.0)
     );
}
/***************************************************/

void test_0_square_compl(Pt2di sz,Fonc_Num Ax,Fonc_Num Ay)
{
     Im2D_REAL8  X1(sz.x,sz.y);
     Im2D_REAL8  Y1(sz.x,sz.y);
     Im2D_REAL8  X2(sz.x,sz.y);
     Im2D_REAL8  Y2(sz.x,sz.y);

     ELISE_COPY
     (  X1.all_pts(),
        mulc(Virgule(Ax,Ay),Virgule(Ax,Ay)),
        Virgule(X1.out(),Y1.out())
     );

     ELISE_COPY
     (  X2.all_pts(),
        squarec(Virgule(Ax,Ay)),
        Virgule(X2.out(),Y2.out())
     );

     REAL dif;

     ELISE_COPY
     (  X1.all_pts(),
        Abs(X1.in()-X2.in()) + Abs(Y1.in()-Y2.in()),
        sigma(dif)
     );

     BENCH_ASSERT(fabs(dif) < epsilon);

}

void test_square_compl()
{
     test_0_square_compl
     (
         Pt2di(103,109),
          FX+2.0*FY + Rconv(FX * FY),
          Rconv(FX % 25)+ 10.0 /(FY+23.0)
     );

}


void test_polar_div(Pt2di sz,Fonc_Num fx,Fonc_Num fy)
{
     Im2D_REAL8  rho(sz.x,sz.y);
     Im2D_REAL8  teta(sz.x,sz.y);
     Im2D_REAL8  IX(sz.x,sz.y);
     Im2D_REAL8  IY(sz.x,sz.y);


     // ----------------- polar ---------------------

     ELISE_COPY
     (
         rho.all_pts(),
         //polar(Virgule(fx,fy)),
	 Polar_Def_Opun::polar(Virgule(fx,fy),0), // __NEW
         Virgule(rho.out() , teta.out())
     );

     ELISE_COPY
     (
        rho.all_pts(),
        rho.in()*Virgule(cos(teta.in()),sin(teta.in())),
        Virgule(IX.out(),IY.out())
     );

     REAL dif;

     ELISE_COPY
     (  IX.all_pts(),
        Abs(fx-IX.in()) +  Abs(fy-IY.in()),
        VMax(dif)
     );
     BENCH_ASSERT(fabs(dif) < epsilon);

     // ----------------- div unary ---------------------

     ELISE_COPY
     (
         rho.all_pts(),
         divc(Virgule(fx,fy)),
         Virgule(rho.out() , teta.out())
     );

     ELISE_COPY
     (
        rho.all_pts(),
        mulc
        (
           Virgule(rho.in(),teta.in()), 
           Virgule(fx,fy)
        ),
        Virgule(IX.out(),IY.out())
     );

     ELISE_COPY
     (  
        IX.all_pts(),
        Abs(IX.in() - 1.0) +  Abs(IY.in() -0.0),
        VMax(dif)
     );
     BENCH_ASSERT(fabs(dif) < epsilon);


     // ----------------- div binary ---------------------

     Fonc_Num f0 = Virgule(FY-12.78,FX-21.76);

     ELISE_COPY
     (
         rho.all_pts(),
         mulc(f0,divc(Virgule(fx,fy))),
         Virgule(rho.out() , teta.out())
     );

     ELISE_COPY
     (
        rho.all_pts(),
        divc(f0,Virgule(fx,fy)),
        Virgule(IX.out(),IY.out())
     );

     ELISE_COPY
     (  
        IX.all_pts(),
        Abs(IX.in() - rho.in()) +  Abs(IY.in() -teta.in()),
        VMax(dif)
     );


     BENCH_ASSERT(fabs(dif) < epsilon);


}

void test_polar_def()
{
   INT sz = 10;
   Im2D_REAL8  rho1(sz,sz);
   Im2D_REAL8  teta1(sz,sz);
   Im2D_REAL8  rho2(sz,sz);
   Im2D_REAL8  teta2(sz,sz);

   REAL vdef = 2.39;
   ELISE_COPY
   (
       rho1.all_pts(),
       //polar((FX-FY),vdef),
       Polar_Def_Opun::polar((FX-FY),vdef), // __NEW
       Virgule(rho1.out(),teta1.out())
   );

   REAL pi = teta1.data()[1][0];

   ELISE_COPY(rho2.all_pts(),Abs(FX-FY),rho2.out());
   ELISE_COPY(teta2.all_pts(),pi*(FX<FY),teta2.out());
   ELISE_COPY(line(Pt2di(0,0),Pt2di(sz-1,sz-1)),vdef,teta2.out());


   REAL dteta,dhro;
   ELISE_COPY
   (
        teta1.all_pts(),
        Abs(Virgule(rho1.in(),teta1.in())-Virgule(rho2.in(),teta2.in())),
        Virgule(VMax(dhro),VMax(dteta))
   );

   BENCH_ASSERT((dhro < epsilon) && (dteta < epsilon));
}

/***************************************************/

void test_op_complex()
{
     test_mul_compl();
     test_square_compl();

     test_polar_div(Pt2di(130,150),FX-20.52,FY-30.348);

     test_polar_def();
     cout << "OK complex \n";
}





