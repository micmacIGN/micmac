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

void test_bmp
     (
           char * n0
         , Video_Win      W
         , Video_Win      W2
         , Video_Display  Ecr
     )
{
    char name[200];
    sprintf(name,"../IM_ELISE/BMP/%s",n0);

    ELISE_fp fp;
    if (! fp.ropen(name,true))
    {
        cout << "cannnot open : " << name << "\n";
        return;
    }
    fp.close();


     cout << "[" << name << "] \n";

     Bmp_Im fbmp (name);

     if (fbmp.bpp() == 24)
     {
          RGB_Pal        Prgb  (5,6,6);
          Elise_Set_Of_Palette SOP(newl(Prgb));
          Ecr.load(SOP);
          W.set_sop(SOP);
          W2.set_sop(SOP);

         ELISE_COPY ( W.all_pts(), fbmp.in(0), W.orgb());

         ELISE_COPY
         (
             rectangle(Pt2di(20,20),Pt2di(200,200)),
             255-fbmp.in(0),
             W.orgb()
         );

         if (fbmp.compr() == Bmp_Im::no_compr)
         {
              char name_dup[200];
              sprintf(name_dup,"../IM_ELISE/BMP_DUP/%s",n0);
              ELISE_fp fp;
              if (! fp.ropen(name_dup,true))
              {
                  cout << "cannnot open : " << name_dup << "\n";
                  return;
              }
              fp.close();

              Bmp_Im fdup (name_dup);

              ELISE_COPY
              (
                  W.all_pts(),
                  255-fbmp.in(0),
                  W.orgb() | fdup.out()
              );

              ELISE_COPY
              (
                  disc(Pt2di(50,50),40),
                  fbmp.in(0),
                  W.orgb() | fdup.out()
              );

              ELISE_COPY
              (
                  W.all_pts(),
                  255-fdup.in(0),
                  W.orgb()
              );


         }
     }
     else
     {
         INT nb_col = 128;
         Disc_Pal p = fbmp.pal();
         Disc_Pal     Pdisc  = Disc_Pal::P8COL();

         Im1D_INT4 lut(p.nb_col());
         if (p.nb_col() < nb_col)
            ELISE_COPY(lut.all_pts(),FX,lut.out());
         else
            p = p.reduce_col(lut,nb_col);


         Elise_Set_Of_Palette SOP(newl(p)+Pdisc);
         Ecr.load(SOP);
         W.set_sop(SOP);
         W2.set_sop(SOP);


/*
         Im2D_U_INT1 v1(fbmp.sz().x,fbmp.sz().y);
         Im2D_U_INT1 v2(fbmp.sz().x,fbmp.sz().y);


*/
         ELISE_COPY (W.all_pts(), lut.in()[fbmp.in(0)] , W2.out(p));
/*
         ELISE_COPY (W.all_pts(), lut.in()[f0.in(0)] , v2.out());

         INT nbc = min(nb_col,p.nb_col());
         ELISE_COPY
         (
            rectangle(Pt2di(20,20),Pt2di(100,100)),
            (fbmp.in(0)+1)%nbc,
            W.out(p)
         );
         ELISE_COPY
         (
                disc(Pt2di(50,50),40),
                1-fbmp.in(0),
                W.out(p) | v1.out()
         );
         ELISE_COPY
         (
                disc(Pt2di(50,50),40),
                1-f0.in(0),
                W.out(p) | v2.out()
         );
         INT nb_dif;
         ELISE_COPY
         (
             select(v1.all_pts(),v1.in() != v2.in()),
             P8COL::red,
             W2.out(Pdisc) | sigma(nb_dif)
         );

         cout << "nb dif = " << nb_dif << "\n";
*/
/*
         ELISE_COPY 
         (
               W.all_pts(), 
               lut.in()[fbmp.in(0)] , 
               W.out(p).chc((FX,FY+200))
         );
*/
         if (fbmp.compr() == Bmp_Im::no_compr)
         {
              char name_dup[200];
              sprintf(name_dup,"../IM_ELISE/BMP_DUP/%s",n0);
              ELISE_fp fp;
              if (! fp.ropen(name_dup,true))
              {
                  cout << "cannnot open : " << name_dup << "\n";
                  return;
              }
              fp.close();

              Bmp_Im fdup (name_dup);
              Im2D_U_INT1 f0(fbmp.sz().x,fbmp.sz().y);
              ELISE_COPY
              (
                  f0.all_pts(),
                  frandr()<0.5,
                  f0.out() | fdup.out()
              );

/*
              ELISE_COPY
              (
                   disc(Pt2di(50,50),40),
                   frandr() < 0.5,
                   f0.out() | fdup.out()
              );
              ELISE_COPY
              (
                   disc(Pt2di(50,50),40),
                   P8COL::white,
                   W2.out(Pdisc) 
              );
*/

              ELISE_COPY
              (
                   rectangle(Pt2di(2,0),Pt2di(32,1)),
                   frandr() < 0.5,
                   f0.out() | fdup.out() |  W2.out(p)
              );
              ELISE_COPY
              (
                   disc(Pt2di(50,50),40),
                   frandr() < 0.5,
                   f0.out() | fdup.out()  | W2.out(p)
              );
              for (INT i = 0 ; i < 10 ; i++)
              ELISE_COPY
              (
                   disc(Pt2di(i*10,i*5+20),i*3+4),
                   Min(frandr(),0.99) * p.nb_col(),
                   f0.out() | fdup.out()  | W2.out(p)
              );

/*
              ELISE_COPY
              (
                   rectangle(Pt2di(100,10),Pt2di(200,90)),
                   P8COL::white,
                   W2.out(Pdisc) 
              );
*/


              INT nb_dif,x;
              ELISE_COPY
              (
                  select(f0.all_pts(),f0.in(0) != fdup.in(0)),
                  P8COL::red,
                  W.out(Pdisc)  | (sigma(nb_dif) << 1) | VMin(x)
              );

              INT v;
              ELISE_COPY
              (
                   rectangle(Pt2di(2,0),Pt2di(3,1)),
                   fdup.in(0),
                   sigma(v)
              );

              cout <<  "f0[2][0] = " << (INT) f0.data()[0][2] << "\n";
              cout <<  "fdup[2][0] = " << v << "\n";
              cout << "nb dif = " << nb_dif << "\n";
              cout << "xmin dif = " << x << "\n";

              INT vmax,vmin;
              {
                   ELISE_COPY(W.all_pts(),fbmp.in(0),VMax(vmax)|VMin(vmin));

                   Symb_FNum Fcomp  (vmax-fbmp.in(0));
                   ELISE_COPY
                   (
                       W.all_pts(),
                       Fcomp,
                          fdup.out()
                       |  (W2.out(p)  << lut.in()[Fcomp])
                   );
              }

              {
                   Symb_FNum Fcomp (fbmp.in(0));


                   ELISE_COPY
                   (
                       disc(Pt2di(50,50),40),
                       Fcomp,
                          fdup.out()
                       |  (W2.out(p)  << lut.in()[Fcomp])
                   );
              }

              {

                   ELISE_COPY
                   (
                       W.all_pts(),
                       lut.in()[vmax-fdup.in(0)],
                       W2.out(p)
                   );
              }


         }
     }
}



/*********** UTIL ********************************/




main(int,char *)
{
     ELISE_DEBUG_USER = true;
     stow_memory_counter(MC_INIT);

     {

         Gray_Pal       PGray (2);
         RGB_Pal        Prgb  (3,3,3);

         Elise_Set_Of_Palette SOP(newl(PGray)+Prgb);
         Video_Display Ecr((char *) NULL);
         Ecr.load(SOP);
         Video_Win   W  (Ecr,SOP,Pt2di(50,50),Pt2di(512,512));
         Video_Win   W2 (Ecr,SOP,Pt2di(50,50),Pt2di(512,512));


         test_bmp("greennews.bmp",W,W2,Ecr);
         test_bmp("greenview3.bmp",W,W2,Ecr);
         test_bmp("world.bmp",W,W2,Ecr);
         test_bmp("texw3.bmp",W,W2,Ecr);
         test_bmp("land3.bmp",W,W2,Ecr);

         test_bmp("SeasonOfTheWitch.bmp",W,W2,Ecr);
         test_bmp("flag_t24.bmp",W,W2,Ecr);
         test_bmp("new-1.bmp",W,W2,Ecr); // compressed
         test_bmp("4winds.bmp",W,W2,Ecr);  // compressed
         test_bmp("window2.bmp",W,W2,Ecr); // compressed

         // test_bmp("texw2.bmp",W,W2,Ecr); // bmpd.2
         test_bmp("gua.bmp",W,W2,Ecr);  // sz header = 12


/*
*/
         test_bmp("teton6.bmp",W,W2,Ecr);
         test_bmp("Scabbing.bmp",W,W2,Ecr);

         test_bmp("redstar4.bmp",W,W2,Ecr);
         test_bmp("CAIROF.bmp",W,W2,Ecr);

         test_bmp("flag_b24.bmp",W,W2,Ecr);

         test_bmp("blk.bmp",W,W2,Ecr);
         test_bmp("blu.bmp",W,W2,Ecr);
         test_bmp("grn.bmp",W,W2,Ecr);
         test_bmp("land.bmp",W,W2,Ecr);
         test_bmp("land2.bmp",W,W2,Ecr);
         test_bmp("ray.bmp",W,W2,Ecr);
         test_bmp("red.bmp",W,W2,Ecr);
         test_bmp("tru256.bmp",W,W2,Ecr);
         test_bmp("venus.bmp",W,W2,Ecr);
         test_bmp("wht.bmp",W,W2,Ecr);
         test_bmp("yel.bmp",W,W2,Ecr);
         test_bmp("xing_b24.bmp",W,W2,Ecr);

     }

     cout << "OK BENCH 0 \n";
     verif_memory_state(MC_INIT);
}


