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

Fonc_Num K_erod_8(Fonc_Num f,INT nb)
{
      for (INT i =0; i<nb ; i++)
          f = erod_8_hom(f);
      return f;
}




void test
     (
         const char * name,
         Video_Win   W ,
         Pt2di       tr
     )
{

    char buf[200];
    sprintf(buf,"../IM_ELISE/TIFF/%s",name);

    Tiff_Im  Image(buf);

    Image.show();

    Fonc_Num f = Image.in(0);

    ELISE_COPY
    (
         W.all_pts(),
         trans(f,tr),
         W.odisc()
    );

    ELISE_COPY
    (
         select
         (
              W.all_pts(),
              trans(K_erod_8(f,8),tr)
         ),
         P8COL::red,
         W.odisc()
    );



    ELISE_COPY
    (
         select
         (
              W.all_pts(),
              trans (K_erod_8(f,8),tr)
         ),
         P8COL::blue,
         W.odisc()
    );



    getchar();
}

void test
     (
         const char * name,
         Video_Win   W 
     )
{
    INT x=0,y=0;

     cout << "Fichier [" << name << "]\n";
     while((x>=0)&& (y>=0))
     {
         scanf("%d %d",&x,&y);
         test(name,W.chc(Pt2di(0,0),Pt2di(2,2)),Pt2di(x,y));
     }
}

main(int,char *)
{

     ELISE_DEBUG_USER = true;
     stow_memory_counter(MC_INIT);

     {

         Disc_Pal      Pdisc = Disc_Pal::P8COL();

         Elise_Set_Of_Palette SOP(newl(Pdisc));
         Video_Display Ecr((char *) NULL);
         Ecr.load(SOP);
         Video_Win   W  (Ecr,SOP,Pt2di(50,50),Pt2di(512,512));


         
         test("ccitt_1.tif",W.chc(Pt2di(0,0),Pt2di(2,2)));
         test("ccitt_2.tif",W.chc(Pt2di(0,0),Pt2di(2,2)));
         test("ccitt_3.tif",W.chc(Pt2di(0,0),Pt2di(2,2)));
         test("ccitt_4.tif",W.chc(Pt2di(0,0),Pt2di(2,2)));
     }

     cout << "OK BENCH 0 \n";
     verif_memory_state(MC_INIT);
}


