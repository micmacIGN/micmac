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
#include "private/all.h"
#include "bench.h"






class Bench_Act_Vect : public Br_Vect_Action
{
    public :
       Bench_Act_Vect 
       (
          Video_Win   W ,
          REAL        prec,
          Line_St     col1,
          Line_St     col2
       ) : 
           _W        (W),
           _prec     (prec),
           _col1     (col1),
           _col2     (col2)
       {}

    private  :
       void action(const ElFifo<Pt2di> &);

       Video_Win   _W;
       REAL        _prec;
       Line_St     _col1;
       Line_St     _col2;
};

void Bench_Act_Vect::action(const ElFifo<Pt2di> & pts)
{

      ElFifo<INT>  app(20);
      ArgAPP arg
             (
                 _prec,
                 50,
                 ArgAPP::D2_droite,
                 ArgAPP::Extre,
                 10
              );

      approx_poly(app,pts,arg);


      for (INT i = 0; i<app.nb() ; i++)
          _W.draw_circle_loc
          (
              pts[app[i]],
              3,
              _col2
          );

      for (INT i = 0; i<app.nb()-1 ; i++)
          _W.draw_seg
          (
              pts[app[i]],
              pts[app[i+1]],
              _col1
          );
};

void bench_vecto 
     (
          Video_Win   W ,
          Fonc_Num    f,
          Pt2di       tr,
          REAL        prec,
          Line_St     col1,
          Line_St     col2
     )
{
   Pt2di sz  = W.p1();
   Im2D_U_INT1 I(sz.x,sz.y);

   ELISE_COPY
   (
        W.all_pts(),
        trans(f,tr),
        W.odisc() | I.out()
   );

    L_ArgSkeleton larg  =
               newl(AngSkel(3.14))
             + SurfSkel(8)
             + ProlgtSkel(true);


   ELISE_COPY
   (
      I.all_pts(),
      sk_vect
      (
         skeleton(I.in(0),20,larg),
         new Bench_Act_Vect (W,prec,col1,col2)
      ),
      Output::onul()
   );

}


Pt2di SZPL(11023,11023);




int main(int,char **)
{
     int ZOOM = 1;
     int SZX = 800;
     int SZY = 600;
     Pt2di SZ (SZX,SZY);


     Gray_Pal     Pgr(100);
     Disc_Pal Pdisc = Disc_Pal::P8COL();
     Elise_Set_Of_Palette SOP(newl(Pdisc)+Pgr);
     Video_Display Ecr((char *) NULL);
     Ecr.load(SOP);
     Video_Win   W  (Ecr,SOP,Pt2di(0,0),SZ*ZOOM);
     W  = W.chc(Pt2di(0,0),Pt2di(ZOOM,ZOOM));

     // Tiff_Im tif("/clavecin2/cartes/CAD_TIF/69240-ac.tif");
     // tif.show();

     Elise_File_Im  FInput
     (
          "/valandre3/cartes/gabon/altikang.bin",
          SZPL,
          GenIm::u_int1
     );


     while (1)
     {
         INT x,y;
         REAL prec;
         cout << "ENTER x y,   precision\n";
         scanf("%d %d %lf",&x,&y,&prec);
         cout << " GOT : " << x << " " << y << " " << " " << prec << "\n";
         bench_vecto 
         (
               W,
               FInput.in(0) != 0,
               Pt2di(x,y),
               prec,
               Line_St(Pdisc(P8COL::red  ),2),
               Line_St(Pdisc(P8COL::green),2)
         );
     }
     return 0;
}

