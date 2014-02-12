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


void VerifEq(Im2D_INT4 aVerif   ,Im2D_INT4 aSom   ,Pt2di aP0 ,Pt2di aP1)
{

    REAL S[3];
    ELISE_COPY
    (
       rectangle(aP0,aP1),
       Rconv(Virgule(Abs(aVerif.in()-aSom.in()),aVerif.in(),aSom.in())),
       sigma(S,3)
    );

    BENCH_ASSERT(S[0]<epsilon);

}

void VerifCorrel_SommeSpec(Pt2di aP0,Pt2di aP1,INT aNb)
{
   pt_set_min_max(aP0,aP1);
   Pt2di aSz(aP1.x+aNb+1,aP1.y+aNb+1);


   Im2D_INT4 aSom(aSz.x,aSz.y)  ,  aVerif(aSz.x,aSz.y);
   Im2D_INT4 aSom1(aSz.x,aSz.y) , aVerif1(aSz.x,aSz.y);
   Im2D_INT4 aSom2(aSz.x,aSz.y) , aVerif2(aSz.x,aSz.y);
   Im2D_INT4 aSom11(aSz.x,aSz.y),aVerif11(aSz.x,aSz.y);
   Im2D_INT4 aSom12(aSz.x,aSz.y),aVerif12(aSz.x,aSz.y);
   Im2D_INT4 aSom22(aSz.x,aSz.y),aVerif22(aSz.x,aSz.y);
   Im2D_INT4 aBuf(aSz.x,aSz.y);


   Im2D_U_INT1 anIm1(aSz.x,aSz.y);
   Im2D_U_INT1 anIm2(aSz.x,aSz.y);
   Im2D_U_INT1 aPond(aSz.x,aSz.y);

   ELISE_COPY
   (
        anIm1.all_pts(),
        Virgule(255*frandr(),255*frandr(),2*frandr()),
        Virgule(anIm1.out(),anIm2.out(),aPond.out())
   );

   ELISE_COPY
   (
        rectangle(aP0,aP1),
        rect_som
        (
             (aPond.in()  !=0)
           * Virgule 
             (
                1,anIm1.in(),anIm2.in(),
                ElSquare(anIm1.in()),
                anIm1.in() * anIm2.in(),
                ElSquare(anIm2.in())
             ),
             aNb
        ),
        Virgule
        (
            aVerif.out()   ,  aVerif1.out()   ,  aVerif2.out(),
            aVerif11.out() ,  aVerif12.out()  ,  aVerif22.out()
        )
   );

   Somme__1_2_11_12_22
   (
       aSom,aSom1,aSom2,
       aSom11,aSom12,aSom22,
       aBuf,
       anIm1,anIm2,aPond,
       aP0,aP1,aNb
   );

   VerifEq(aVerif   , aSom   , aP0 , aP1);
   VerifEq(aVerif1  , aSom1  , aP0 , aP1);
   VerifEq(aVerif2  , aSom2  , aP0 , aP1);
   VerifEq(aVerif11 , aSom11 , aP0 , aP1);
   VerifEq(aVerif12 , aSom12 , aP0 , aP1);
   VerifEq(aVerif22 , aSom22 , aP0 , aP1);
}


void Bench_SommeSpec()
{
     VerifCorrel_SommeSpec(Pt2di(10,15),Pt2di(87,76),6);
     VerifCorrel_SommeSpec(Pt2di(19,15),Pt2di(77,96),8);
}


