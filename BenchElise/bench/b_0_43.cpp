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


class PtDelTest
{
    public :
       PtDelTest(INT x,INT y) : _pt (x,y) {}

       PtDelTest()  {}

       Pt2di pt  () const {return _pt;}

    private :
       Pt2di _pt;
};

//Pt2di Pt_of_DPT(const PtDelTest & PDT) { return PDT.pt();}
Pt2dr Pt_of_DPT(const PtDelTest & PDT) { return Pt2dr(PDT.pt());} // __NEW

class DelTestAct
{
    public :
       DelTestAct(Video_Win W) : _w (W) {}

      void operator () (PtDelTest & p1,PtDelTest & p2,bool ArcDegen)
      {
           if (! ArcDegen)
                //_w.draw_seg(p1.pt(),p2.pt(),_w.pdisc()(P8COL::blue));
                _w.draw_seg(Pt2dr(p1.pt()),Pt2dr(p2.pt()),_w.pdisc()(P8COL::blue)); // __NEW
      }

    private :
       Video_Win _w;
       
};

#include <list>
#include <vector>

void bench_delaunay()
{
    Pt2di sz (950,950);

    Disc_Pal       Pdisc  = Disc_Pal::P8COL();

    Elise_Set_Of_Palette SOP(NewLElPal(Pdisc));
    Video_Display Ecr((char *) NULL);
    Ecr.load(SOP);                         
    Video_Win   W  (Ecr,SOP,Pt2di(50,50),sz);  


    for (INT nb = 100; nb < 500 ; nb +=50)
    {

       ElSTDNS list<PtDelTest> L;
       ElFilo<PtDelTest> Pts;

       for (INT k= 0; k< nb; k++)
       {
           PtDelTest pt
                     (
                             (INT)(  10 + (sz.x -20) * NRrandom3()),
                             (INT)(  10 + (sz.y -20) * NRrandom3())
                     );
            // Pts.pushlast(pt);
            L.push_back(pt);
            //W.draw_circle_loc(pt.pt(),4,Pdisc(P8COL::red));
            W.draw_circle_loc(Pt2dr(pt.pt()),4,Pdisc(P8COL::red)); // __NEW
       }
       DelTestAct Act(W);
       Delaunay_Mediatrice 
       (
             L.begin(),
             L.end(),
             Pt_of_DPT,
             Act,
	     200.0,
            (PtDelTest *) 0
       );

       // W.clik_in();
       W.clear();
    }
}



















