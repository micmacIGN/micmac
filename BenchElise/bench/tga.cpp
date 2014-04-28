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

void test_tga
     (
           char * name
         , Video_Win      W
         , Video_Display
         , INT dx = 0
         , INT dy = 0
         , bool invert = false
     )
{

     ELISE_fp fp;

     if (! fp.open(name,ELISE_fp::READ,true))
     {
          cout << "cannnot open : " << name << "\n";
          return;
     }
     fp.close();

     cout << "[" << name << "] \n";

     Tga_Im TGA (name);

     if  (TGA.toi() == Tga_Im::bw_image)
          ELISE_COPY
          (
              W.all_pts(),
              trans(TGA.in(0),Pt2di(dx,dy)),
              (invert ? (W.ogray().chc((FX,W.p1().y-FY-1))) : W.ogray())
              //  W.ogray()
          );
     else
     {
          ELISE_COPY (W.all_pts(),TGA.in(0),W.orgb());
          // ELISE_COPY (rectangle(Pt2di(20,20),Pt2di(200,200)),255-TGA.in(0),W.orgb());
     }




}





main(int,char *)
{
     ELISE_DEBUG_USER = true;
     stow_memory_counter(MC_INIT);

     {

         Gray_Pal       PGray (20);
         RGB_Pal        Prgb  (5,6,6);

         Elise_Set_Of_Palette SOP(newl(PGray)+Prgb);
         Video_Display Ecr((char *) NULL);
         Ecr.load(SOP);
         Video_Win   W  (Ecr,SOP,Pt2di(50,50),Pt2di(512,512));

         test_tga("../IM_ELISE/TGA/FRK.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/FRK2.tga",W,Ecr);


         test_tga("../IM_ELISE/TGA/frk1.tga",W,Ecr,500,500,false);
         test_tga("../IM_ELISE/TGA/frk2.tga",W,Ecr,500,500);
         test_tga("../IM_ELISE/TGA/flag_b16.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/flag_b24.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/flag_b32.tga",W,Ecr);

         test_tga("../IM_ELISE/TGA/flag_t16.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/flag_t24.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/flag_t32.tga",W,Ecr);

         test_tga("../IM_ELISE/TGA/jk14.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/xing_b16.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/xing_b24.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/xing_b32.tga",W,Ecr);

         test_tga("../IM_ELISE/TGA/xing_t16.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/xing_t24.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/xing_t32.tga",W,Ecr);

         test_tga("../IM_ELISE/TGA/lena1.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/Chant-LWIR.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/Chant-NIR.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/Clementine_Imagery.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/HiRes-Apollo_16.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/HiRes-Moon_Surf.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/LWIR-Moon_Surf.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/NIR-Moon_Surf.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/StarTracker-Big_Dpr.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/StarTracker-Earth_Lim.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/UVVIS-Moon_Surf.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/UV_Vis-Apollo_16.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/UV_Vis-Earth.tga",W,Ecr);

         test_tga("../IM_ELISE/TGA/barb.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/zelda.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/lena1.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/lena2.tga",W,Ecr);
         test_tga("../IM_ELISE/TGA/lena3.tga",W,Ecr);

     }

     cout << "OK BENCH 0 \n";
     verif_memory_state(MC_INIT);
}


