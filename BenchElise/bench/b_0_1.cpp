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


void verif_som_x_rect(Pt2di p1,Pt2di p2)
{
     int res,res_sel;

     ELISE_COPY(rectangle(p1,p2),FX,sigma(res));
     ELISE_COPY(select(rectangle(p1,p2),1),FX,sigma(res_sel));

     BENCH_ASSERT(res == ElAbs(som_x(p1.x,p2.x)*(p1.y-p2.y)));
     BENCH_ASSERT(res == res_sel);


}

void verif_som_y_rect(Pt2di p1,Pt2di p2)
{
     int res,res_sel;

     ELISE_COPY(rectangle(p1,p2),FY,sigma(res));
     ELISE_COPY(select(rectangle(p1,p2),1),FY,sigma(res_sel));
     BENCH_ASSERT(res == ElAbs(som_x(p1.y,p2.y)*(p1.x-p2.x)));
     BENCH_ASSERT(res == res_sel);
}


void verif_som_coord_rect(Pt2di p1,Pt2di p2)
{
    verif_som_x_rect(p1,p2);
    verif_som_y_rect(p1,p2);
}


void verif_som_coord_rect()
{
    All_Memo_counter MC_INIT;
    stow_memory_counter(MC_INIT);

    {
        verif_som_coord_rect(Pt2di(-10,-13),Pt2di(103,105));
        verif_som_coord_rect(Pt2di(0,0),Pt2di(10,10));

        { 
            INT res;
            ELISE_COPY(rectangle(Pt2di(-10,-13),Pt2di(103,105)),FY+FX*3,sigma(res));
        }
    }
    verif_memory_state(MC_INIT);
}

 //=============================================

template <class Type> void verif_max_or_min_x
                      (   const OperAssocMixte & op,
                          Type expected,
                          Pt2di p1,
                          Pt2di p2,
                          bool    reel
                      )
{
    Type v;

    ELISE_COPY(rectangle(p1,p2),reel ? Rconv(FX) : FX,reduc(op,v));
    BENCH_ASSERT(v == expected);
}

template <class Type> void verif_max_and_min_x(Pt2di p1,Pt2di p2,Type *,bool reel)
{
     verif_max_or_min_x(OpMax,(Type) (p2.x-1),p1,p2,reel);
     verif_max_or_min_x(OpMin,(Type) (p1.x),p1,p2,reel);
}

void verif_max_min_x(Pt2di p1,Pt2di p2)
{
     verif_max_and_min_x(p1,p2,(INT *) 0,false);
     verif_max_and_min_x(p1,p2,(REAL *) 0,true);
}

void verif_witch_max_min()
{
     INT   imin,imax;
     REAL  rmin,rmax;
     ELISE_COPY
     (
         rectangle(0,10),
         FX,
           WhichMin(imin) | WhichMax (imax)
         | WhichMin(rmin) | WhichMax (rmax)
     );

     BENCH_ASSERT
     (
           (imin==0) && (imax==9)
        && (rmin==0) && (rmax==9)
     );

     Pt2di pimin,pimax;
     Pt2dr prmin,prmax;

     Pt2dr q;

     ELISE_COPY
     (
         rectangle(Pt2di(-20,-20),Pt2di(12,10)).chc(Virgule(FX+0.1,FY+0.1)),
         Virgule
         (
             sqrt(Square(FX)+Square(FY)),
             FX+FY
         ),
         Virgule
         (
           
                  pimin.WhichMin() | pimax.WhichMax ()
                | prmin.WhichMin() | prmax.WhichMax ()
           , q.WhichMax()
         )
     );

      BENCH_ASSERT
      (
               (euclid(pimin,Pt2di((INT)0.1,(INT)0.1)        ) < epsilon)
           &&  (euclid(pimax,Pt2di((INT)-19.9,(INT)-19.9)    ) < epsilon)
           &&  (euclid(prmin,Pt2dr(0.1,0.1)    ) < epsilon)
           &&  (euclid(prmax,Pt2dr(-19.9,-19.9)) < epsilon)
           &&  (euclid(    q,Pt2dr(11.1,9.1)   ) < epsilon)
      );

}


void verif_max_min()
{
      verif_witch_max_min();

      verif_max_min_x(Pt2di(2,2),Pt2di(10,10));
}


void bench_cmpFN()
{
    std::vector <Fonc_Num> vF;
    cVarSpec aVX(1,"X");
    cVarSpec aVY(2,"Y");

    vF.push_back(FX);
    vF.push_back(FY);
    vF.push_back(cos(1+FY));
    vF.push_back(cos(2+FY));
    for (INT k=0 ; k< 3; k++)
       vF.push_back(cos(1+FZ));
    for (INT k=0 ; k< 3; k++)
       vF.push_back(1.0);
    vF.push_back(2.0);
    for (INT k=0 ; k< 3; k++)
       vF.push_back(aVX*aVY);
    vF.push_back(2.0);

    INT aNB = vF.size();

    // Reflexivite
    for (INT aK=0 ; aK < aNB ; aK++)
    {
        Fonc_Num F = vF[aK];
        BENCH_ASSERT(F.CmpFormel(F)==0);
    }

    INT NbEq =0;
    // Anti symetrie
    for (INT aK1=0 ; aK1 < aNB ; aK1++)
    {
      for (INT aK2=aK1+1 ; aK2 < aNB ; aK2++)
      {
          Fonc_Num F1 = vF[aK1];
          Fonc_Num F2 = vF[aK2];
	  INT C12 = F1.CmpFormel(F2);
	  INT C21 = F2.CmpFormel(F1);
          BENCH_ASSERT(C12+C21==0);
          BENCH_ASSERT(ElAbs(C12)<2);
	  if (C12 == 0)
	  {
	      NbEq ++;
	      PtsKD aP(8);
	      for (INT aD =0 ; aD<8 ; aD++)
                  aP(aD) = NRrandom3();

	      REAL aV1 = F1.ValFonc(aP);
	      REAL aV2 = F2.ValFonc(aP);
              BENCH_ASSERT(ElAbs(aV1-aV2)<epsilon);
	  }
      }
    }
    BENCH_ASSERT(NbEq==10);

    // Reflexivite
    for (INT aK1=0 ; aK1 < aNB ; aK1++)
    {
      for (INT aK2=aK1+1 ; aK2 < aNB ; aK2++)
      {
          for (INT aK3=aK2+1 ; aK3 < aNB ; aK3++)
	  {
              Fonc_Num F1 = vF[aK1];
              Fonc_Num F2 = vF[aK2];
              Fonc_Num F3 = vF[aK3];
	      INT C12 = F1.CmpFormel(F2);
	      INT C23 = F2.CmpFormel(F3);
	      INT C31 = F3.CmpFormel(F1);

	      BENCH_ASSERT(ElAbs(C12+C23+C31)<2);

	      INT NbEq = (C12==0) + (C23==0) + (C31==0);
	      BENCH_ASSERT(NbEq != 2);
	  }
      }
    }

}



