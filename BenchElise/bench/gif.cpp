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


void test_gif_write(char * name,Video_Win W,Video_Display D)
{
    ELISE_fp  fp;
    if (! fp.ropen(name,true))
    {
        cout << "WRITE : cannnot open : " << name << "\n";
        return;
    }
    fp.close();


    Gif_File gf (name);


   Gif_Im  i = gf.kth_im(0);
   Disc_Pal    p  = i.pal();

   INT nb_col = p.nb_col() ;
   Elise_colour * tabc = NEW_TAB(nb_col,Elise_colour);
   p.getcolors(tabc);
   
   Pt2di sz = i.sz();

   system(ELISE_SYS_RM ELISE_BFI_DATA_DIR  "tmp.gif");

   INT nbb =0;
   for (int p2 = 1; p2 < nb_col; p2 *= 2)
        nbb++;

   ELISE_COPY
   (
        rectangle(Pt2di(0,0),sz),
        i.in(),
        Gif_Im::create
        (
            ELISE_BFI_DATA_DIR "tmp.gif",
            sz,
            tabc,
            nbb
        )
   );

   DELETE_TAB(tabc);

   test_gif(ELISE_BFI_DATA_DIR "tmp.gif",W,D);
}


main(int,char *)
{
     ELISE_DEBUG_USER = true;
     stow_memory_counter(MC_INIT);

{

    Disc_Pal       Pdisc  = Disc_Pal::P8COL();
    Elise_Set_Of_Palette SOP(newl(Pdisc));
    Video_Display Ecr((char *) NULL);
    Ecr.load(SOP);

    Video_Win   W  (Ecr,SOP,Pt2di(50,50),Pt2di(500,500));

    test_gif_write("../IM_ELISE/GIF/lena.gif",W,Ecr);  // empty block : probably
    test_gif_write("../IM_ELISE/GIF/astro.gif",W,Ecr);

    test_gif("../TMP/image1.gif",W,Ecr); // : end code prematured
    test_gif("../TMP/image2.gif",W,Ecr); // : end code prematured
    test_gif("../TMP/image3.gif",W,Ecr); // : end code prematured
    test_gif("../TMP/image4.gif",W,Ecr); // : end code prematured

    test_gif("../IM_ELISE/GIF/rabbit.gif",W,Ecr); // : end code prematured

    test_gif("../IM_ELISE/GIF/lena.gif",W,Ecr);  // empty block : probably
                                                 // where end code was expected
    test_gif("../IM_ELISE/GIF/AUNT.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/New.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/Virtual_Library.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/annalena.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/astro.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/bwselena.gif",W,Ecr);

    test_gif("../IM_ELISE/GIF/clinton.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/deux.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/europe.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/eyes.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/glob.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/lapin.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/lena_ok.gif",W,Ecr);  
    test_gif("../IM_ELISE/GIF/lena2.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/lena3.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/letter_un.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/letter_zero.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/lune_0.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/maelena.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/mandelbrot.gif",W,Ecr);
    // test_gif("../IM_ELISE/GIF/rabbit.gif"); // : end code prematured
    test_gif("../IM_ELISE/GIF/st.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/stereolena.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/text.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/text_lena.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/un.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/wfpc01.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/wfpc02.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/wfpc03.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/wfpc04.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/wfpc05.gif",W,Ecr);
    test_gif("../IM_ELISE/GIF/wfpc06.gif",W,Ecr);


    test_gif("../IM_ELISE/ANI_GIF/A_8earth.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/animail.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/badvirus.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/birdfl_7E1.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/book.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/broute2.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/canonew.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/catmini.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/cigar.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/dogrun.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/elemail.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/email.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/emailed.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/faq.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/flamewar.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/gear_box_small.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/hummingbird.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/justr2.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/mailbox1.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/mailbox_7F_7F.GIF",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/mseb.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/nekorun.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/porte.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/rido.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/smack.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/splat.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/splatman.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/tasmania.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/vient-de.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/webup.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/wrhammer.gif",W,Ecr);
    test_gif("../IM_ELISE/ANI_GIF/smack.gif",W,Ecr);

/*
    test_gif("../IM_ELISE/GIF/lena3.gif");
    test_gif("../IM_ELISE/GIF/wfpc01.gif");
    test_gif("../IM_ELISE/GIF/wfpc02.gif");
    test_gif("../IM_ELISE/GIF/wfpc03.gif");
    test_gif("../IM_ELISE/GIF/wfpc04.gif");
    test_gif("../IM_ELISE/GIF/wfpc05.gif");
    test_gif("../IM_ELISE/GIF/wfpc06.gif");
    test_gif("../IM_ELISE/GIF/lena2.gif");
    test_gif("../IM_ELISE/GIF/un.gif");
    test_gif("../IM_ELISE/GIF/deux.gif");

    test_gif("../IM_ELISE/GIF/rabbit.gif");
*/


}
{
   Elise_colour  tec[256];


   for (INT i=0 ; i<64 ; i++)
   {
       REAL gray = i/64.0;
       tec[i] = Elise_colour::rgb(gray,gray,gray);
   }

   Gray_Pal       Pgray  (50);

   Elise_Set_Of_Palette SOP(newl(Pgray));
   Video_Display Ecr((char *) NULL);
   Ecr.load(SOP);

   Video_Win   W  (Ecr,SOP,Pt2di(50,50),Pt2di(300,400));

   Pt2di sz (300,400);

   ELISE_COPY
   (
       rectangle(Pt2di(0,0),sz),
       (FX+FY)%64,
       Gif_Im::create("toto.gif",sz,tec,6)  | W.ogray()
   );
   getchar();
   Gif_Im GI("toto.gif");
   ELISE_COPY
   (
        W.all_pts(),
        255-GI.in(0) * 4,
        W.ogray()
        
   );
   getchar();
}
     verif_memory_state(MC_INIT);
     cout << "OK BENCH 0 \n";
}


