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

class WCEP : public Cont_Vect_Action 
{
    public :
        void action(const ElFifo<Pt2di> & pts,bool ext)
        {
          if (ext)
             _nb_ext ++;
          else
             _nb_int ++;
          ElList<Pt2di> l;
          for (INT k=0; k<pts.nb(); k++)
              l = l + pts[k];
           ELISE_COPY ( polygone(l,ext), (_check.in()+1) % 2, _check.out());
       }
       virtual ~WCEP(){}

       WCEP(Im2D_U_INT1 check) : 
            _nb_int(0), 
            _nb_ext(0), 
            _check(check)
       {
       }


       INT            _nb_int;
       INT            _nb_ext;
       Im2D_U_INT1    _check;
       
};

void bench_flag_front (Pt2di   sz,bool FN)
{
    Neighbourhood V4 = Neighbourhood::v4();
    Neighbourhood V8 = Neighbourhood::v8();

    Im2D_U_INT1 Im(sz.x,sz.y,0);
    Liste_Pts_INT4 ListTrou(2);

    ELISE_COPY ( Im.all_pts(), frandr() > 0.5, Im.out() );

    ELISE_COPY(Im.border(2),0,Im.out());
    ELISE_COPY
    (
        select
        (
              select(Im.all_pts(),Im.in()),
              Neigh_Rel(V8).red_max(Im.in()) == 0
        ),
        2,
        ListTrou | Im.out() 
    );

    INT nbZ1 = 0;
    INT nbZ0 = -1;  // pour tenir compte de la zone externe
    {
          Im2D_U_INT1 ImZ(sz.x,sz.y); 
          ELISE_COPY(ImZ.all_pts(),Im.in()==1,ImZ.out());
          ELISE_COPY(ImZ.border(1),2,ImZ.out());
          U_INT1 ** z = ImZ.data();
          for (INT x=0 ; x<sz.x ; x++)
              for (INT y=0 ; y<sz.y ; y++)
                  if (z[y][x] != 2)
                  {
                      if (z[y][x] == 1) 
                         nbZ1 ++; 
                      else 
                         nbZ0++;
                      Neighbourhood  neigh = ((z[y][x] == 1) ? V8 : V4);
                      ELISE_COPY
                      (
                          conc(Pt2di(x,y),ImZ.neigh_test_and_set(neigh,z[y][x],2,10)),
                          2,
                          ImZ.out()
                      );
                  }
    }



    Im2D_U_INT1 IFront(sz.x,sz.y);
    ELISE_COPY
    (
       Im.all_pts(),
       flag_front8(Im.in(0)),
       IFront.out()
    );

    Im2D_U_INT1 check(sz.x,sz.y,0);

    ElFifo<Pt2di>   pts;
    WCEP wcep(check);

    if (FN)
    {
	ELISE_COPY
        (
	    Im.all_pts(),
            cont_vect (Im.in(),&wcep,true),
            Output::onul() 
        );

    }
    else
        explore_cyle(&wcep,pts,IFront);

    ELISE_COPY(ListTrou.all_pts(),check.in()+2,check.out());

    INT dif;
    ELISE_COPY(Im.all_pts(),Abs(Im.in()-check.in()),sigma(dif));

    BENCH_ASSERT(dif==0);
    BENCH_ASSERT(nbZ0 == wcep._nb_int);
    BENCH_ASSERT(nbZ1 == wcep._nb_ext);
    
}

void bench_flag_front(bool FN)
{
     for (INT x = 20 ; x < 100 ; x+= 22)
         for (INT y = 40 ; y < 300 ; y+= 12)
		 {
                      bench_flag_front(Pt2di(x,y),FN);
		 }
}
void bench_flag_front()
{
     bench_flag_front(true);
     bench_flag_front(false);
     ncout() << "OK BENCH FLAG FRONT \n";
}





