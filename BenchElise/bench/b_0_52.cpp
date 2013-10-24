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


/**********************************************/
/**********************************************/
/**********************************************/

void One_bench_geo3d_basic()
{
    Pt3dr aTr(NRrandC(),NRrandC(),NRrandC());
    ElRotation3D aRot
                 (
                     aTr * 20,
                     NRrandC() * 10,
                     NRrandC() * 10,
                     NRrandC() * 10
                 );

     REAL z = NRrandC() * 10;

     REAL  tetaA = NRrandC() * 10;
     REAL  RhoA = 0.1 + NRrandom3() * 10;

     Pt3dr P0A (0,0,z);
     Pt3dr P1A (RhoA*cos(tetaA),RhoA*sin(tetaA),z);

     REAL  tetaB = tetaA + 0.01 + NRrandom3();
     REAL  RhoB = 0.1 + NRrandom3() * 10;

     Pt3dr P0B (0,0,-z);
     Pt3dr P1B (RhoB*cos(tetaB),RhoB*sin(tetaB),-z);

     Pt3dr Pinter(0,0,0);

     ElSeg3D SegPA(P0A,P1A);
     ElSeg3D SegPB(P0B,P1B);
     Pt3dr PPseudo = SegPA.PseudoInter(SegPB);

     BENCH_ASSERT(euclid(Pinter-PPseudo) < epsilon);

     Pt3dr Q0A = aRot.ImAff(P0A);
     Pt3dr Q1A = aRot.ImAff(P1A);
     

     Pt3dr Q0B = aRot.ImAff(P0B);
     Pt3dr Q1B = aRot.ImAff(P1B);

     Pt3dr Qinter = aRot.ImAff(Pinter);

     ElSeg3D SegQA(Q0A,Q1A);
     ElSeg3D SegQB(Q0B,Q1B);
    
     Pt3dr QPseudo = SegQA.PseudoInter(SegQB);
     BENCH_ASSERT(euclid(Qinter-QPseudo) < epsilon);

     Pt3dr QProjOnA,QProjOnB;
     SegQA.Projections(QProjOnA,SegQB,QProjOnB);

     Pt3dr ImPProjOnA = aRot.ImAff(P0A);
     Pt3dr ImPProjOnB = aRot.ImAff(P0B);

     BENCH_ASSERT(euclid(ImPProjOnA-QProjOnA) < epsilon);
     BENCH_ASSERT(euclid(ImPProjOnB-QProjOnB) < epsilon);
}

void bench_geo3d_basic()
{
    for (INT k=0; k<10000; k++)
    {
        One_bench_geo3d_basic();
    }
}

/**********************************************/
/**********************************************/
/**********************************************/

void bench_geo3d()
{
    bench_geo3d_basic();
    cout << "END Bench Geo3D \n";
}

