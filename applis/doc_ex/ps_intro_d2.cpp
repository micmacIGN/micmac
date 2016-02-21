#include "general/all.h"




PS_Window PS(char * name, bool auth_lzw = false)
{
      // sz of images we will use

         Pt2di SZ(256,256);

     //  palette allocation

         Disc_Pal  Pdisc = Disc_Pal::P8COL();
         Gray_Pal  Pgr (80);
         Circ_Pal  Pcirc = Circ_Pal::PCIRC6(30);
         BiCol_Pal Prb  (
                           Elise_colour::black,
                           Elise_colour::red,
                           Elise_colour::blue,
                           10,10
                       );
         RGB_Pal   Prgb(10,10,10);



         Elise_Set_Of_Palette SOP
                              (
                                      NewLElPal(Pdisc)
                                   +  Elise_Palette(Prb)
                                   +  Elise_Palette(Pgr)
                                   +  Elise_Palette(Prgb)
                                   +  Elise_Palette(Pcirc)
                              );

     // Creation of postscript windows

           char  buf[200];
           sprintf(buf,"DOC/PS/%s.eps",name);
    
           PS_Display disp(buf,"Mon beau fichier ps",SOP,auth_lzw);

           return  disp.w_centered_max(SZ,Pt2dr(4.0,4.0));
}



int  main(int,char **)
{
    Tiff_Im  FLenaCol("DOC/lena_col.tif");
    Tiff_Im  FLena("DOC/mini_lena.tif");
    Im2D_U_INT1 I(256,256);
    ELISE_COPY(I.all_pts(),FLena.in(),I.out());


    PS_Window W =  PS("d2_FX");
    ELISE_COPY(W.all_pts(),FX,W.ogray());


    W =  PS("d2_FY");
    ELISE_COPY(W.all_pts(),FY,W.ogray());

    W =  PS("d2_FXpY");
    ELISE_COPY(W.all_pts(),(FX+FY)/2,W.ogray());


    W =  PS("d2_exp_cos_XY");
    ELISE_COPY
    (
          W.all_pts(), 
          127.5 * (1.0 + cos(FY*0.2+0.06*FX*sin(0.06*FX))),
          W.ogray()
    );

   //  5
       W =  PS("lena_first");
       ELISE_COPY
       (
           I.all_pts(),
           FLena.in(),
           I.out() | W.ogray()
       );

    //  6
       W =  PS("lena_neg");
       ELISE_COPY
       (
           I.all_pts(),
           255-I.in(),
           W.ogray()
       );

    //  7
       W =  PS("lena_pcirc");
       ELISE_COPY
       (
           I.all_pts(),
           I.in(),
           W.ocirc()
       );

       W =  PS("lena_pal_disc");
       ELISE_COPY
       (
           I.all_pts(),
           I.in()/64,
           W.odisc()
       );

    //
       W =  PS("lena_bin_128");
       ELISE_COPY
       (
           I.all_pts(),
           I.in()<128,
           W.odisc()
       );
       W =  PS("lena_bin_FX");
       ELISE_COPY
       (
           I.all_pts(),
           I.in()<FX,
           W.odisc()
       );
       W =  PS("lena_bin_trame");
       ELISE_COPY
       (
           I.all_pts(),
           I.in()<127.5*(1+sin(FX)*sin(FY)),
           W.odisc()
       );

    //
       W =  PS("lena_sel_rouge");
       ELISE_COPY(I.all_pts(),I.in(),W.ogray());
       ELISE_COPY
       (
           select(I.all_pts(),I.in()>128),
           P8COL::red,
           W.odisc()
       );

      // operateur ,

       W =  PS("virg_X0Y",false);
       ELISE_COPY
       (
           I.all_pts(),
           Virgule(FX,0,FY),
           W.orgb()
       );
       W =  PS("virg_XY0",false);
       ELISE_COPY
       (
           I.all_pts(),
           Virgule(FX,FY,0),
           W.orgb()
       );

       W =  PS("Lena_rgb",false);
       ELISE_COPY
       (
           I.all_pts(),
           FLenaCol.in(),
           W.orgb()
       );


       W =  PS("Lena_gbr",false);
       ELISE_COPY
       (
           I.all_pts(),
           Virgule
           (
              FLenaCol.in().v1(),
              FLenaCol.in().v0(),
              FLenaCol.in().v2()
           ),
           W.orgb()
       );



       W =  PS("lena_YX");
       ELISE_COPY
       (
           I.all_pts(),
           I.in()[Virgule(FY,FX)],
           W.ogray()
       );


       W =  PS("lena_rotat");
       ELISE_COPY
       (
           I.all_pts(),
           I.in(0)
           [  Virgule
              (
                  128+(FX-128)+(FY-128)/2.0,
                  128-(FX-128)/2.0+(FY-128)
             )
           ],
           W.ogray()
       );

       W =  PS("lena_3x2y");
       ELISE_COPY
       (
           I.all_pts(),
           I.in()[Virgule(FX*3,FY*2)%256],
           W.ogray()
       );

       W =  PS("lena_morph_cos");
       ELISE_COPY
       (
           I.all_pts(),
           I.in(0)
           [
            Virgule
            (
               FX+20*sin(FX/50.0)+4*cos(FY/15.0),
               FY+8*sin(FX/20.0)+7*cos(FY/18.0)
            )
           ],
           W.ogray()
       );



       W =  PS("lena_double_gray");
       ELISE_COPY
       (
           I.all_pts(),
           (I.in()+I.in()[Virgule(FY,FX)])/2,
           W.ogray()
       );

       W =  PS("lena_double_rb",false);
       ELISE_COPY
       (
           I.all_pts(),
           Virgule(I.in(),0,I.in()[Virgule(FY,FX)]),
           W.orgb()
       );

     
       {

             //
/*
             W =  PS("lena_disc_bleue");
             ELISE_COPY(I.all_pts(),P8COL::blue,W.odisc()| Obin);
             ELISE_COPY(disc(Pt2di(128,128),100),I.in(),W.ogray()|Ogray);
*/

             Im2D_U_INT1 Isbin(256,256,1);
             Im2D_U_INT1 PsBin(256,256,1);
             Im2D_U_INT1 PsGr(256,256,1);

             Output Obin  = (Isbin.oclip()<<1) | (PsBin.oclip());
             Output Ogray = (Isbin.oclip()<<0) | (PsGr.oclip());


             Im2D_U_INT1 R(256,256);
             Im2D_U_INT1 G(256,256);
             Im2D_U_INT1 B(256,256);



             Output Orgb  = Virgule(R.oclip(),G.oclip(),B.oclip());
             Symb_FNum Is (I.in(0));
             Fonc_Num  Igray = Virgule(Is,Is,Is);


             ELISE_COPY(I.all_pts(),Fonc_Num(0,0,255),Orgb|(Obin<<P8COL::blue));
             ELISE_COPY(disc(Pt2di(128,128),100),Igray,Orgb|Ogray);

             W =  PS("lena_db1");
             ELISE_COPY(W.all_pts(),PsBin.in(),W.odisc());
             ELISE_COPY(select(W.all_pts(),!Isbin.in()),PsGr.in(),W.ogray());

             ELISE_COPY
             (
                   ell_fill(Pt2di(128,128),135,70,1.2),
                   255-Igray,
                  Orgb | Ogray
             );

            ELISE_COPY
            (
                sector_ang(Pt2di(128,128),100,1.0,3.0),
                (I.in() >= 128)*Fonc_Num(255,255,255),
                Orgb | (Obin<<(I.in() < 128))
            );

             W =  PS("lena_db2");
             ELISE_COPY(W.all_pts(),PsBin.in(),W.odisc());
             ELISE_COPY(select(W.all_pts(),!Isbin.in()),PsGr.in(),W.ogray());


            for (INT x = 0; x < 256; x+= 4)
                 ELISE_COPY
                 (
                     line(Pt2di(x,0),Pt2di(128,128)),
                     Igray,
                     Orgb | Ogray
                 );

          ELISE_COPY
          (
               polygone
               (
                       NewLPt2di(Pt2di(5,250))+Pt2di(128,5)
                   +   Pt2di(250,250)+Pt2di(128,128)
               ),
               (Igray/64)*64,
               Orgb | Ogray
          );

             W =  PS("lena_db3");
             ELISE_COPY(W.all_pts(),PsBin.in(),W.odisc());
             ELISE_COPY(select(W.all_pts(),!Isbin.in()),PsGr.in(),W.ogray());

        for (INT x = 1; x< 5; x++)
             ELISE_COPY
             (
                 border_rect(Pt2di(10,10),Pt2di(15+10*x,15+10*x),5-x),
                 Fonc_Num(255,255,255),
                 Orgb  | (Obin << P8COL::white)
             );


          ELISE_COPY
          (
              W.border(8),
              Fonc_Num(0,255,0),
              Orgb | (Obin << P8COL::green)
          );


          ELISE_COPY
          (
              ellipse(Pt2di(128,128),135,70,1.2),
              Fonc_Num(255,0,0),
              Orgb  | (Obin << P8COL::red)
          );

           W =  PS("lena_db4");
           ELISE_COPY(W.all_pts(),PsBin.in(),W.odisc());
           ELISE_COPY(select(W.all_pts(),!Isbin.in()),PsGr.in(),W.ogray());

           W =  PS("lena_multi_flx",false);
           ELISE_COPY(W.all_pts(),Virgule(R.in(),G.in(),B.in()),W.orgb());
       }
    

// Pour generer l'arc en ciel
       W =  PS("arc_en_ciel",false);
       Fonc_Num f = polar(Virgule(FX-128,FY-128),0);
       ELISE_COPY
       (
           I.all_pts(),
           P8COL::black,
           W.odisc()
       );
       ELISE_COPY
       (
           disc(Pt2di(128,128),100),
           (f.v1()/(2*PI)+1.0)*256,
           W.ocirc()
       );
       ELISE_COPY
       (
           disc(Pt2di(128,128),30),
           P8COL::black,
           W.odisc()
       );
       W =  PS("lena_gamma");
       ELISE_COPY
       (
           I.all_pts(),
           pow(I.in()/255.0,5.0/3.0)*255.0,
           W.ogray()
       );

       W =  PS("lena_gamma_tab");
       {
          Im1D_U_INT1 lut(256);
          ELISE_COPY
          (
             lut.all_pts(),
             pow(FX/255.0,3.0/5.0)*255.0,
             lut.out()
          );
          ELISE_COPY
          (
              I.all_pts(),
              lut.in()[I.in()],
              W.ogray()
          );
       }

       W =  PS("lena_sel_12513");
       {  
          Im1D_U_INT1 lut(256,0);
          U_INT1 * d = lut.data();
          d[1]  = d[2] = d[5] = 1;
          d[10] = d[13] = d[14] = 2;
          ELISE_COPY
          (
              I.all_pts(),
              lut.in()[I.in()/16],
              W.odisc()
          );
       }

        PS_Window Wr =  PS("lena_col_r");
        PS_Window Wg =  PS("lena_col_g");
        PS_Window Wb =  PS("lena_col_b");

        ELISE_COPY
        (
            Wr.all_pts(),
            0,
               (Wr.ogray() << FLenaCol.in().v0())
            |  (Wg.ogray() << FLenaCol.in().v1())
            |  (Wb.ogray() << FLenaCol.in().v2())
        );


      {
           Im2D_U_INT1 IO(256,256,0);
        
           W =  PS("disc_chc");
           ELISE_COPY(I.all_pts(),0,W.odisc());
           ELISE_COPY
           (
                disc(Pt2di(64,64),64).chc(2*Virgule(FX,FY)),
                1,
                W.odisc()
           );

           W =  PS("chc_lenlen");
           ELISE_COPY(I.all_pts(),I.in(),IO.out());
           ELISE_COPY
           (
                rectangle(Pt2di(0,0),Pt2di(256,128)).chc(Virgule(FX,FY*2)),
                I.in()[Virgule(FX,255-FY)],
                IO.out()
           );
           ELISE_COPY(I.all_pts(),IO.in(), W.ogray());


           W =  PS("chc_lenFY");

           ELISE_COPY(I.all_pts(),I.in(),IO.out());
           ELISE_COPY
           (
                rectangle(Pt2di(0,0),Pt2di(256,128)).chc(Virgule(FX,2*FY-FY%2)),
                FY,
                IO.out()
           );
           ELISE_COPY(I.all_pts(),IO.in(), W.ogray());
      }
}



