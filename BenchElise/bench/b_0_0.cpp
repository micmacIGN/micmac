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

// compute sum of val on the rectangle p1xp2, using a Flux_Pts anf a Fonc_Num
// and controls it gives the expected values

void som1_cste_int_rect_2d_by_hand(INT val, Pt2di p1,Pt2di p2)
{
    Arg_Flux_Pts_Comp a;
    Flux_Pts  r = rectangle(p1,p2);
    Flux_Pts_Computed *rc = r.compute(a);


    Fonc_Num f(val);
    Fonc_Num_Computed *fc = f.compute(Arg_Fonc_Num_Comp(rc)); 

    int res = 0;
    Pack_Of_Pts * pck;
    while((pck = (Pack_Of_Pts *)rc->next()))
    {
         Int_Pack_Of_Pts   *vals = 
                     SAFE_DYNC(Int_Pack_Of_Pts *,fc->values(pck));

          res = OpSum.red_tab(vals->_pts[0],vals->nb(),res);
    }

    BENCH_ASSERT(res == ElAbs(val*(p1.x-p2.x)*(p1.y-p2.y)));
    delete rc;
    delete fc;

}

// compute sum of val on the rectangle p1xp2, using a Flux_Pts, a Fonc_Num and an Output
// and controls it gives the expected values

void som2_cste_int_rect_2d_by_hand(INT val, Pt2di p1,Pt2di p2)
{
    int res = 244;
    {
        Arg_Flux_Pts_Comp a;
        Flux_Pts  r = rectangle(p1,p2);
        Flux_Pts_Computed *rc = r.compute(a);


        Fonc_Num f(val);
        Fonc_Num_Computed *fc = f.compute(Arg_Fonc_Num_Comp(rc)); 

        Output o =  reduc(OpSum,res);
        Output_Computed *oc = o.compute(Arg_Output_Comp(rc,fc));

        Pack_Of_Pts * pck;
        while((pck = (Pack_Of_Pts *) rc->next()))
        {
              oc->update(pck,fc->values(pck));
        }
        delete rc;
        delete fc;
        delete oc;
    }

    BENCH_ASSERT(res == ElAbs(val*(p1.x-p2.x)*(p1.y-p2.y)));
}


void som3_cste_int_rect_2d_by_hand(INT val, Pt2di p1,Pt2di p2)
{
    INT res;

    ELISE_COPY(rectangle(p1,p2),val,reduc(OpSum,res));
    BENCH_ASSERT(res == ElAbs(val*(p1.x-p2.x)*(p1.y-p2.y)));

    ELISE_COPY(rectangle(p1,p2),val,sigma(res));
    BENCH_ASSERT(res == ElAbs(val*(p1.x-p2.x)*(p1.y-p2.y)));
}


//  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// To success this bench , val must have some special value
//  because od the test "==" on DOUBLE.
//  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

void som4_cste_int_rect_2d_by_hand(REAL val, Pt2di p1,Pt2di p2)
{
    REAL res;

    ELISE_COPY(rectangle(p1,p2),val,sigma(res));
    BENCH_ASSERT(res == ElAbs(val*(p1.x-p2.x)*(p1.y-p2.y)));
}


void somme_cste_int_rect_2d_by_hand(INT val, Pt2di p1,Pt2di p2)
{
     som1_cste_int_rect_2d_by_hand(val,p1,p2);
     som2_cste_int_rect_2d_by_hand(val,p1,p2);
     som3_cste_int_rect_2d_by_hand(val,p1,p2);
     som4_cste_int_rect_2d_by_hand(val,p1,p2);
}


void somme_cste_int_rect_2d_by_hand(void)
{
     All_Memo_counter MC_INIT;
     stow_memory_counter(MC_INIT);

     {
         somme_cste_int_rect_2d_by_hand (3,Pt2di(20,20),Pt2di(220,220));
         somme_cste_int_rect_2d_by_hand (3,Pt2di(220,220),Pt2di(20,20));
         somme_cste_int_rect_2d_by_hand ( 3,Pt2di(0,1),Pt2di(1,0));
         somme_cste_int_rect_2d_by_hand ( 3,Pt2di(1,1),Pt2di(1,1));
     }

     verif_memory_state(MC_INIT);
     cout << "OK somme_cste_int_rect_2d_by_hand \n";
}


void bench_Tab_CPT_REF()
{
	/*
   {
      Tab_CPT_REF<Pt2dr> tab(10);

      {
          for (int i =0; i< 10; i++)
          {
              tab.push(Pt2dr(i,0));
              tab[i].y = i;
          }
      }

      {
          for (int i =0; i< 10; i++)
          {
              Pt2dr p = tab[i];
              BENCH_ASSERT(p.x==i&&p.y==i);
          }
      }
   }
   {
      Pt2dr t0[10];
      for (int i =0; i< 10; i++)
          t0[i] = Pt2dr(i,i);

      Tab_CPT_REF<Pt2dr> tab(t0,10);

      {
          for (int i =0; i< 10; i++)
              t0[i] = Pt2dr(-i,-i);
	  }

      {
          for (int i =0; i< 10; i++)
          {
              Pt2dr p = tab[i];
              BENCH_ASSERT(p.x==i&&p.y==i);
          }
      }
   }
   */

     cout << "OK bench_Tab_CPT_REF \n";
}






