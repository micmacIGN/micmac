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



#include "general/all.h"
#include "api/vecto.h"

template <class T> Pt2d<T> complex2Pt2d(complex<T> p) 
                           {return Pt2d<T>(p.real(),p.imag());}


class MyVectoAct : public EL_CALL_BACK_VECTO
{
      public :

           void  ElVectoAction
                 (
                     ComplexI p0,
                     ComplexI p1,
                     const vector<ComplexI> & pts_interm,
                     const vector<int> & d_interm
                 )
            {
                _W1.draw_seg
                (
                     complex2Pt2d(p0),
                     complex2Pt2d(p1),
                     _Pdisc(P8COL::red)
                );

                for (INT k=0; k< (int)pts_interm.size() ; k++)
                {
                    _W2.draw_circle_abs
                    (
                         complex2Pt2d(pts_interm[k]),
                         2.0,
                        _Pdisc(P8COL::green)
                    );
                    _W3.draw_circle_loc
                    (
                         complex2Pt2d(pts_interm[k]),
                         d_interm[k] * 0.5,
                        _Pdisc(P8COL::blue)
                    );
                }
            }

           MyVectoAct
           (
                Video_Win   W1,
                Video_Win   W2,
                Video_Win   W3,
                Disc_Pal    Pdisc

           )  :
              _W1     (W1),
              _W2     (W2),
              _W3     (W3),
              _Pdisc  (Pdisc)
           {
           }

      private :

           Video_Win   _W1;
           Video_Win   _W2;
           Video_Win   _W3;
           Disc_Pal    _Pdisc;
};



void t0
     (
        Pt2di       sz,
        Video_Win   W1,
        Video_Win   W2,
        Video_Win   W3,
        Disc_Pal    Pdisc
     )
{
     char * name = "/home/pierrot/Data/Cad/TPuzzle/SKEL.tif";

     EL_API_VECTO::ParamSkel     PSkel;
     EL_API_VECTO::ParamApprox   PApp;

     EL_API_VECTO  API(PSkel,PApp,20);

     EL_API_VECTO::ParamFile PFile(name,false);


     INT x,y;
     scanf("%d %d",&x,&y);

     Im2D_Bits<1>  Im(sz.x,sz.y);

     EL_API_VECTO::ParamImage aPIm
                              (
                                ComplexI(sz.x,sz.y),
                                false,  // <
                                1,  // SEUIL
                                Im.data()[0],  // Adr 1er element
                                1  // Nb Bit
                              );

     ELISE_COPY
     (
         W1.all_pts(),
         trans(Tiff_Im(name).in(0),Pt2di(x,y)),
         W1.odisc() | W2.odisc() | W3.odisc() | Im.out()
     );


     MyVectoAct MVA(W1,W2,W3,Pdisc); 

     API.vecto
     (
          // PFile,
          aPIm,
          MVA,
          ComplexI(x,y),
          ComplexI(sz.x,sz.y)
     );
}

int main(int,char**)
{
     int SZX = 400;
     int SZY = 400;
     int ZOOM = 1;


     Video_Win   W1  = Video_Win ::WStd(Pt2di(SZX,SZY),ZOOM);
     Video_Win   W2  = Video_Win ::WStd(Pt2di(SZX,SZY),ZOOM,W1);
     Video_Win   W3  = Video_Win ::WStd(Pt2di(SZX,SZY),ZOOM,W1);


     while (true)
           t0 (Pt2di(SZX,SZY),W1,W2,W3,W1.pdisc());
}




