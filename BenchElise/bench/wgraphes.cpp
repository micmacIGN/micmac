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


All_Memo_counter MC_INIT;


/*********** UTIL ********************************/




void gr()
{
    ELISE_DEBUG_USER = true;
    stow_memory_counter(MC_INIT);

    Elise_File_Im fg =  Elise_File_Im::pnm("../x.pgm");
    Elise_File_Im fp =  Elise_File_Im::pnm("../x.ppm");
    Elise_File_Im fb =  Elise_File_Im::pnm("../x.pbm");

    Disc_Pal       Pdisc   = Disc_Pal::P8COL();
    Gray_Pal       Pgray   (30);
    RGB_Pal        Prgb    (5,5,5);

    Elise_Set_Of_Palette SOP(newl(Pdisc)+Pgray+Prgb);
    Video_Display Ecr((char *) NULL);
    Ecr.load(SOP);

    Video_Win   W  (Ecr,SOP,Pt2di(50,50),Pt2di(500,500));

    ELISE_COPY(W.all_pts(),fb.in(0),W.out(Pdisc));
getchar();
    ELISE_COPY(fb.all_pts(),1-fb.in(),fb.out());
    ELISE_COPY(W.all_pts(),fb.in(0),W.out(Pdisc));
getchar();
    ELISE_COPY(disc(Pt2di(100,100),100),1-fb.in(0),fb.out());
    ELISE_COPY(W.all_pts(),fb.in(0),W.out(Pdisc));
getchar();



    ELISE_COPY(W.all_pts(),fg.in(0),W.out(Pgray));
getchar();
    ELISE_COPY(fg.all_pts(),255-fg.in(),fg.out());
    ELISE_COPY(W.all_pts(),fg.in(0),W.out(Pgray));
getchar();
    ELISE_COPY(W.all_pts(),fp.in(0),W.out(Prgb));
getchar();
    ELISE_COPY(fp.all_pts(),255-fp.in(),fp.out());
    ELISE_COPY(W.all_pts(),fp.in(0),W.out(Prgb));
getchar();
    ELISE_COPY
    (
         rectangle(Pt2di(50,50),Pt2di(250,250)),
         64+fp.in(0)/2,
         W.out(Prgb)
    );
getchar();
    ELISE_COPY
    (
         rectangle(Pt2di(50,50),Pt2di(250,250)),
         rect_median(fp.in(0),8,256),
         W.out(Prgb)
    );
getchar();

    W = W.chc(Pt2dr(0,0),Pt2dr(30,30));


    for (INT i = 0; i<8; i++)
    {
        ELISE_COPY (W.all_pts(),(FX+FY)%2,W.out(Pdisc));
        ELISE_COPY (W.all_pts(),1<<i,W.out_graph(Pdisc(P8COL::red)));
        getchar();
    }

    getchar();

}

main(int,char *)
{
     ELISE_DEBUG_USER = true;
     stow_memory_counter(MC_INIT);

{
   INT p,q;
   rationnal_approx(PI,p,q);

   std::cout << "FRAC " << p << "/" << q <<" = " << (p/(REAL) q) <<  "\n";

   rationnal_approx(sqrt(2.0),p,q);
   std::cout << "FRAC " << p << "/" << q <<" = " << (p/(REAL) q) <<  "\n";


   rationnal_approx(-sqrt(2.0),p,q);
   std::cout << "FRAC " << p << "/" << q <<" = " << (p/(REAL) q) <<  "\n";

   rationnal_approx(0.1,p,q);
   std::cout << "FRAC " << p << "/" << q <<" = " << (p/(REAL) q) <<  "\n";

   rationnal_approx(0.125,p,q);
   std::cout << "FRAC " << p << "/" << q <<" = " << (p/(REAL) q) <<  "\n";

   rationnal_approx(1,p,q);
   std::cout << "FRAC " << p << "/" << q <<" = " << (p/(REAL) q) <<  "\n";

   rationnal_approx(0,p,q);
   std::cout << "FRAC " << p << "/" << q <<" = " << (p/(REAL) q) <<  "\n";

   rationnal_approx(300,p,q);
   std::cout << "FRAC " << p << "/" << q <<" = " << (p/(REAL) q) <<  "\n";

   rationnal_approx(1/3.0,p,q);
   std::cout << "FRAC " << p << "/" << q <<" = " << (p/(REAL) q) <<  "\n";
}
     for(int i = -3; i < 10 ; i++)
        std::cout << i << " POW OF 2 : " << is_pow_of_2(i) << "\n";

     {
         char * a = "aaaaaa";
         char * b = "bbbbbb";
         char * c = "cccccc";
         char * d = "dddddd";

         ELISE_fp fp("toto",ELISE_fp::WRITE);

         fp.write(a,1,2);
         fp.write(a,1,2);
         fp.write(a,1,2);
         fp.write(a,1,2);

         fp.seek_begin(2);
         fp.write(b,1,2);
         // fp.write(c,1,2);
         fp.seek_cur(2);
         fp.write(d,1,2);

         fp.close();
         fp.ropen("toto");

         INT i;
         while ((i = fp.fgetc()) != ELISE_fp::eof)
               std::cout << (char) i;
         std::cout << "\n";
         fp.close();
     }



/*
     {
          Im2D_U_INT1 b (10,10);
          Liste_Pts_REAL4 l (1);
          INT x;
          ELISE_COPY
          (
              b.all_pts(),
              Min(FX,FY)*30,
              b.out()
          );
      
      }

     gr();
*/
     verif_memory_state(MC_INIT);
     std::cout << "OK BENCH 0 \n";
}

