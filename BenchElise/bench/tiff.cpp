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




/*********** UTIL ********************************/

bool can_open(const char * name)
{
    ELISE_fp fp;
    if (! fp.ropen(name,true))
       return false;

   fp.close();
   return true;
}

void test
     (
         const char * name,
         Video_Win   W ,
         Video_Display Ecr,
         Elise_Set_Of_Palette  SOP_ori
     )
{

    char buf[200];
    sprintf(buf,"../IM_ELISE/TIFF/%s",name);

    if (! can_open(buf)) return;
    Tiff_Im  Image(buf);


    if ( Image.phot_interp()  != Tiff_Im::RGBPalette)
        return;

    cout << name << ", PH interp = "<< Image.phot_interp() << "\n";

    cout << name << " " << Image.mode_compr() << "\n";
    Image.show();

    const char * why = Image.why_elise_cant_use();
    if (why)
    {
       cout << why << "\n";
       return;
    }


   INT max_val = 1 << Image.bitpp();
   INT mult = 255 / (max_val-1);

   if (Image.nb_chan() == 1) 
   {
      if (Image.phot_interp() == Tiff_Im::RGBPalette)
      {
cout << "BEGIN LOAD INDEXED \n";
          Disc_Pal  Pim = Image.pal();

          INT nb_col = 200;
          Im1D_INT4 lut(1<<Image.bitpp());
          Disc_Pal Pred = Pim.reduce_col(lut,nb_col);
   
          Elise_Set_Of_Palette SOP(newl(Pred));
          W.set_sop(SOP);
          Ecr.load(SOP);
          ELISE_COPY
          (
              W.all_pts(),
              lut.in()[Image.in(0)],
              W.out(Pred)
          );

cout << " END LOAD INDEXED \n";

          getchar();
          W.set_sop(SOP_ori);
          Ecr.load(SOP_ori);
      }
      else
      {


          ELISE_COPY
          (
              W.all_pts(),
              mult * Image.in(max_val/2),
              W.ogray()
          );
      }
   }
   else if  (Image.nb_chan() == 3)
   {
      ELISE_COPY
      (
          W.all_pts(),
          mult * Image.in(max_val/2),
          W.orgb()
      );
      ELISE_COPY
      (
          disc(Pt2dr(150,150),100),
          255-mult * Image.in(max_val/2),
          W.orgb()
      );
   }
}

void test_indexed_color
     (
         const char * name,
         Video_Win   W ,
         Video_Display Ecr,
         Elise_Set_Of_Palette  SOP_ori
     )
{

    Disc_Pal pf = Disc_Pal::P8COL();
    char buf[200];
    sprintf(buf,"../IM_ELISE/TIFF/%s",name);
    if (! can_open(buf)) return;

    Tiff_Im  Image(buf);
    if ( Image.phot_interp()  != Tiff_Im::RGBPalette)
        return;

    Disc_Pal p0 = Image.pal();

    Elise_colour * tabc = p0.create_tab_c();
    INT nbc = p0.nb_col();
    for (int c=0 ; c<nbc ; c++)
        tabc[c] = Elise_colour::rgb(1-tabc[c].r(),1-tabc[c].g(),1-tabc[c].b());
    pf = Disc_Pal(tabc,nbc);

    L_Arg_Opt_Tiff l;
    l = l + Tiff_Im::ANoStrip();
    system("rm ../IM_ELISE/TIFF/tmp_indexed.tif");
    Tiff_Im NewIm
            (
                "../IM_ELISE/TIFF/tmp_indexed.tif",
                Image.sz(),
                Image.type_el(),
                Image.mode_compr(),
                pf,
                l
            );
   DELETE_VECTOR(tabc,0);


    ELISE_COPY
    (
         rectangle(Pt2di(0,0),Image.sz()),
         Image.in(),
         NewIm.out() | W.ogray()
    );

    test("tmp_indexed.tif",W,Ecr,SOP_ori);
}

void test_write
     (   
          char *           nc,
          Video_Win         W,
          Pt2di             sz,
          Pt2di             szt,
          INT               nb_ch,
          GenIm::type_el    type
      )
{
       char buf[200];
       sprintf(buf,"../IM_ELISE/BTIFF/%s.tif",nc);
       INT bps = nbb_type_num(type);

      
       Tiff_Im::PH_INTER_TYPE  ph_interp= 
                 (nb_ch==3)                ?
                 Tiff_Im::RGB              :
                 Tiff_Im::BlackIsZero      ;

       L_Arg_Opt_Tiff l;
       if (szt.x != sz.x)
          l = l+ Tiff_Im::ATiles(szt);
       else if (szt.y != sz.y)
       {
          l = l+ Tiff_Im::AStrip(szt.y);
       }
       else 
          l = l + Tiff_Im::ANoStrip();


       Tiff_Im  Image =
                Tiff_Im 
                (
                     buf,
                     sz,
                     type,
                     Tiff_Im::No_Compr,
                     ph_interp,
                     l
                ) ;

      Image.show();
      ELISE_COPY   
      (
          rectangle(Pt2di(0,0),Pt2di(250,250)),
          (FX+FY,FX,FY)%(1<<bps),
          Image.out()
      );


      {
         Tiff_Im  Image(buf);
         ELISE_COPY   
         (
             rectangle(Pt2di(0,0),Pt2di(400,400)),
             (Image.in((1<<bps)-1)*255)/((1<<bps)-1),
             (nb_ch == 3) ? W.orgb() : W.ogray()
         );
      }
      getchar();
}




void test_write( Video_Win   W )
{
     // test_write("f1",W,Pt2di(303,304),Pt2di(32,32),1,GenIm::u_int1);
     test_write("f22",W,Pt2di(320,320),Pt2di(32,32),3,GenIm::u_int1);
     test_write("f2",W,Pt2di(304,313),Pt2di(32,32),3,GenIm::u_int1);
     test_write("f3",W,Pt2di(305,322),Pt2di(32,32),1,GenIm::bits1_msbf);
     test_write("f4",W,Pt2di(403,341),Pt2di(32,32),1,GenIm::bits2_msbf);

     // NOT xv supported; xv does not know why
     test_write("f5",W,Pt2di(403,341),Pt2di(32,32),3,GenIm::bits4_msbf);

     // stripped
     test_write("f6",W,Pt2di(303,304),Pt2di(303,32),1,GenIm::u_int1);

     // NOT xv supported; XV know why
     test_write("f7",W,Pt2di(303,304),Pt2di(64,32),1,GenIm::bits4_lsbf);
}

void test_write_pckb
     (   
          char *           nc,
          Video_Win         W,
          Pt2di             sz,
          Pt2di             szt,
          INT               nb_ch,
          GenIm::type_el    type
      )
{
       char buf[200];
       sprintf(buf,"../IM_ELISE/BTIFF/%s.tif",nc);
       INT bps = nbb_type_num(type);

       // bool tiled = (szt.x != sz.x);
       // cout << (tiled ?  "TILED : " : "STRIPED") << "\n";
      
       Tiff_Im::PH_INTER_TYPE  ph_interp= 
                 (nb_ch==3)                ?
                 Tiff_Im::RGB              :
                 Tiff_Im::BlackIsZero      ;

       L_Arg_Opt_Tiff l;
       if (szt.x != sz.x)
          l = l+ Tiff_Im::ATiles(szt);
       else if (szt.x != sz.y)
       {
          l = l+ Tiff_Im::AStrip(szt.y);
       }
       else 
          l = l + Tiff_Im::ANoStrip();

       l = l+ Tiff_Im::APred(Tiff_Im::Hor_Diff);

       Tiff_Im  Image =
                Tiff_Im 
                (
                     buf,
                     sz,
                     type,
                     Tiff_Im::LZW_Compr,
                     ph_interp,
                     l
                );



      Symb_FNum f ((FX/11 +FY/9)%(1<<bps));
      ELISE_COPY   
      (
          rectangle(Pt2di(0,0),sz),
          f,
          Image.out()  | (W.odisc() << (f%8))
      );
      // Image.show();

      // getchar();

      INT dif = 0;

      ELISE_COPY   
      (
             rectangle(Pt2di(0,0),Pt2di(400,400)),
             (Image.in((1<<bps)-1)*255)/((1<<bps)-1),
             (nb_ch == 3) ? W.orgb() : W.ogray()
      );
      ELISE_COPY   
      (
            rectangle(Pt2di(0,0),sz),
            Abs(f-Image.in()),
            sigma(dif)
      );
      cout << "DIF = " << dif << "\n";
      El_Internal.assert(dif==0,EEM0 << "CHECK SUM TIFF");
}

void test_write_pckb( Video_Win   W )
{
   test_write_pckb
   (
         "f0",
         W,
         Pt2di(128,128),
         Pt2di(128,128),
         1,
         GenIm::bits1_msbf
   );
   test_write_pckb
   (
         "f1",
         W,
         Pt2di(322,325),
         Pt2di(32,32),
         1,
         GenIm::bits2_msbf
   );
   test_write_pckb
   (
         "f2",
         W,
         Pt2di(321,329),
         Pt2di(32,32),
         1,
         GenIm::bits4_msbf
   );
   test_write_pckb
   (
         "f2",
         W,
         Pt2di(321,329),
         Pt2di(32,32),
         1,
         GenIm::u_int1
   );

}



void test_write_ccitt
     (   
          char *           nc,
          Video_Win         W,
          Pt2di             sz,
          Pt2di             szt,
          bool              msbf,
          Fonc_Num          f
      )
{
       char buf[200];
       sprintf(buf,"../IM_ELISE/BTIFF/%s.tif",nc);

      
       L_Arg_Opt_Tiff l;
       if         (szt.x != sz.x)     l = l+ Tiff_Im::ATiles(szt);
       else if    (szt.y != sz.y)     l = l+ Tiff_Im::AStrip(szt.y); 
       else                           l = l + Tiff_Im::ANoStrip();

       Tiff_Im  Image =
                Tiff_Im 
                (
                     buf,
                     sz,
                     msbf ? GenIm::bits1_msbf : GenIm::bits1_lsbf,
                     Tiff_Im::CCITT_G3_1D_Compr,
                     Tiff_Im::BlackIsZero,
                     l
                );


      ELISE_COPY   
      (
          rectangle(Pt2di(0,0),sz),
          f,
          Image.out() |W.odisc() 
      );

      Symb_FNum  SIm (Image.in(0));
      INT dif = 0;
      ELISE_COPY   
      (
             rectangle(Pt2di(0,0),sz),
             SIm,
                     W.odisc()
              |      (sigma(dif) << Abs(f-SIm))
      );
      cout << "DIF = " << dif << "\n";
      El_Internal.assert(dif==0,EEM0 << "CHECK SUM TIFF");
      getchar();
}
void test_write_ccitt (Video_Win W)
{
      test_write_ccitt
      (   
          "f0",
           W,
           Pt2di(300,300),
           Pt2di(300,300),
           true,
           FX<=FY
      );

      test_write_ccitt
      (   
          "f1",
           W,
           Pt2di(8000,546),
           Pt2di(8000,546),
           true,
           Tiff_Im("../IM_ELISE/TIFF/cci.tif").in(0)
      );

      test_write_ccitt
      (   
          "f2",
           W,
           Pt2di(2643,546),
           Pt2di(2643,546),
           true,
           Tiff_Im("../IM_ELISE/TIFF/cci.tif").in(0)
      );

      test_write_ccitt
      (   
          "f3",
           W,
           Pt2di(8000,546),
           Pt2di(8000,546),
           true,
           Tiff_Im("../IM_ELISE/TIFF/cci.tif").in(1)
      );

}


main(int,char *)
{
    // test_huff();
 //test_lzw("01234567");
 //test_lzw("777887766");
 //test_lzw("120003551200355");
 //test_lzw("00000000000000000000");

     ELISE_DEBUG_USER = true;
     All_Memo_counter MC_INIT;
     stow_memory_counter(MC_INIT);

     {

         Gray_Pal       PGray (100);
         Disc_Pal      Pdisc = Disc_Pal::P8COL();
         RGB_Pal        Prgb  (2,2,2);

         Elise_Set_Of_Palette SOP(newl(PGray)+Prgb+Pdisc);
         Video_Display Ecr((char *) NULL);
         Ecr.load(SOP);
         Video_Win   W  (Ecr,SOP,Pt2di(50,50),Pt2di(512,512));


#if (0)
        test_write_ccitt (W);
#endif
#if (0)
         test_write_pckb(W);
#endif
#if (0)
         test_write(W);
#endif

#if (1)


       test_indexed_color
       (
            "a.tif",
            W,Ecr,SOP
       );
       test_indexed_color
       (
            "p.tif",
            W,Ecr,SOP
       );

       test_indexed_color
       (
            "jello.tif",
            W,Ecr,SOP
       );
       test_indexed_color
       (
            "lena.tif",
            W,Ecr,SOP
       );
       test_indexed_color
       (
            "300dpi_20white.tif",
            W,Ecr,SOP
       );
       test_indexed_color
       (
            "300dpi_20yellow.tif",
            W,Ecr,SOP
       );
/*




         test("t.tif",W,Ecr,SOP);
         test("lzw0.tif",W,Ecr,SOP);
         test("lzwbin.tif",W,Ecr,SOP);

         test("LeMan1.tif",W,Ecr,SOP);
         test("300dpi_20white.tif",W,Ecr,SOP);
         test("300dpi_20yellow.tif",W,Ecr,SOP);
         test("COM0.tif",W,Ecr,SOP);
         test("TX_CC3.tif",W,Ecr,SOP);
         test("TX_LZW.tif",W,Ecr,SOP);
         test("TX_PCKB.tif",W,Ecr,SOP);
         test("camera256x256.tif",W,Ecr,SOP);

         test("jello.tif",W,Ecr,SOP);
         test("world.tif",W,Ecr,SOP);
         test("oxford.tif",W,Ecr,SOP);
         test("quad-lzw.tif",W,Ecr,SOP);
         test("sampler.tif",W,Ecr,SOP);
         test("skater.tif",W,Ecr,SOP);
         test("thumb-peppers512x512c.tif",W,Ecr,SOP);
         test("toys1.tif",W,Ecr,SOP);

         test("jk14_col.tif",W,Ecr,SOP);
         test("jk14_gray.tif",W,Ecr,SOP);

         test("inon.tif",W,Ecr,SOP);


         test("a.tif",W,Ecr,SOP);
         test("pckbin.tif",W,Ecr,SOP);
         test("bin.tif",W,Ecr,SOP);
         test("gr.tif",W,Ecr,SOP);
         test("sauv_lena.tiff",W,Ecr,SOP);
         test("cci.tif",W,Ecr,SOP);
         test("flag_t24.tif",W,Ecr,SOP);
         test("inon.tif",W,Ecr,SOP);
         test("ccitt3_xv.tif",W,Ecr,SOP);
         test("g31d.tif",W,Ecr,SOP);
         test("lena.tif",W,Ecr,SOP);
         test("xing_t24.tif",W,Ecr,SOP);
         test("g31ds.tif",W,Ecr,SOP);
         test("g32d.tif",W,Ecr,SOP);
         test("g32ds.tif",W,Ecr,SOP);
         test("p.tif",W,Ecr,SOP);
         test("pck.tif",W,Ecr,SOP);

         test("ccitt4_xv.tif",W,Ecr,SOP);
         test("g4.tif",W,Ecr,SOP);
         test("ccitt_1.tif",W,Ecr,SOP);
         test("ccitt_2.tif",W,Ecr,SOP);
         test("ccitt_4.tif",W,Ecr,SOP);
         test("ccitt_3.tif",W,Ecr,SOP);
         test("ccitt_5.tif",W,Ecr,SOP);
         test("ccitt_6.tif",W,Ecr,SOP);
         test("ccitt_7.tif",W,Ecr,SOP);
         test("ccitt_8.tif",W,Ecr,SOP);


         test("edf0.tif",W,Ecr,SOP);
         test("edf1.tif",W,Ecr,SOP);
         test("edf2.tif",W,Ecr,SOP);
         test("edf3.tif",W,Ecr,SOP);
         test("edf4.tif",W,Ecr,SOP);
         test("g4s.tif",W,Ecr,SOP);
*/
#endif

#if(0)
       Pt2di sz(256,256);

      L_Arg_Opt_Tiff l;
      l =      l
              + Tiff_Im::ATiles(Pt2di(256,256))
              + Tiff_Im::AMinMax(0,255)
          ;

       Tiff_Im  Image 
                (
                     "toto.tif",
                     sz,
                     GenIm::u_int1,
                     Tiff_Im::LZW_Compr,
                     Tiff_Im::BlackIsZero,
                     l
                );

       ELISE_COPY
       (
          rectangle(Pt2di(0,0),sz),
         64+(FX+FY)/4,
         Image.out()
       );

       Image.show();
#endif

/*
      Elise_File_Im Im2 ("toto.tif",sz,GenIm::u_int2,8);

      ELISE_COPY
      (
          rectangle(Pt2di(0,0),sz),
         (Im2.in()-400)/2,
         W.ogray()
      );

       getchar();
*/

     }

     cout << "OK BENCH 0 \n";
     verif_memory_state(MC_INIT);
}


