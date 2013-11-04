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


template <class T1,class T2,class T12>
void flux_cat_verif_sigma
     (
          T1 *,
          T2 *,
          T12 *,
          Flux_Pts  fl1,
          Flux_Pts  fl2,
          Fonc_Num  f
     )
{
     T1  v1;
     T2  v2;
     T12 v12;

     ELISE_COPY(fl1,f,sigma(v1));
     ELISE_COPY(fl2,f,sigma(v2));

     ELISE_COPY(fl1||fl2,f,sigma(v12));

     BENCH_ASSERT(ElAbs(v12-(v1+v2))<epsilon);
}


void bench_flux_concat()
{
     flux_cat_verif_sigma
     (
          (INT *) 0,(INT *) 0,(INT *) 0,
          //disc(Pt2di(10,10),22.3),
          //disc(Pt2di(10,10),22.3),
          disc(Pt2dr(10,10),22.3), // __NEW
          disc(Pt2dr(10,10),22.3), // __NEW
          FX+FY
     );
     flux_cat_verif_sigma
     (
          (INT *) 0,(INT *) 0,(INT *) 0,
          line_4c(Pt2di(-20,30),Pt2di(22,33)),
          line_4c(Pt2di(-20,30),Pt2di(22,33)),
          FX+FY
     );

     flux_cat_verif_sigma
     (
          (INT *) 0,(INT *) 0,(INT *) 0,
          //disc(Pt2di(10,10),22.3),
          disc(Pt2dr(10,10),22.3), // __NEW
          line_4c(Pt2di(-20,30),Pt2di(22,33)),
          FX+FY
     );

     flux_cat_verif_sigma
     (
          (INT *) 0,(REAL *) 0,(REAL *) 0,
          //disc(Pt2di(10,10),22.3),
          disc(Pt2dr(10,10),22.3), // __NEW
          line_4c(Pt2di(-20,30),Pt2di(22,33)).chc(Virgule(FX+FY*0.1,0.3*FX+1.2*FY)),
          FX+FY
     );

     flux_cat_verif_sigma
     (
          (REAL *) 0,(INT *) 0,(REAL *) 0,
          line_4c(Pt2di(-20,30),Pt2di(22,33)).chc(Virgule(FX+FY*0.1,0.3*FX+1.2*FY)),
          //disc(Pt2di(10,10),22.3),
          disc(Pt2dr(10,10),22.3), // __NEW
          FX+FY
     );

     flux_cat_verif_sigma
     (
          (REAL *) 0,(REAL *) 0,(REAL *) 0,
          line_4c(Pt2di(-20,30),Pt2di(22,33)).chc(Virgule(FX+FY*0.1,0.3*FX+1.2*FY)),
          //disc(Pt2di(10,10),22.3).chc(Virgule(FX+FY*0.2,0.4*FX-1.6*FY)),
          disc(Pt2dr(10,10),22.3).chc(Virgule(FX+FY*0.2,0.4*FX-1.6*FY)),  // __NEW
          FX+FY
     );

}


void bench_oper_flux()
{
     bench_flux_concat();
    cout << "OK OPER FLUX  \n";
}
