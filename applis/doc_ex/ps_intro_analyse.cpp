#include "general/all.h"


void show_brd(PS_Window W)
{

    W.draw_rect
   (
        Pt2di(0,0),
        Pt2di(256,256),
        Line_St(W.pdisc()(P8COL::black),6)
   );
}

PS_Window PS(char * name, bool auth_lzw = false,Pt2di SZ = Pt2di(256,256))
{
      // sz of images we will use


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

    // sz of images we will use
       // Pt2di SZ(256,256);
       Pt2di SZ_RED (80,80);
       Pt2di TR     (80,80);

/*
   //  palette allocation
       Disc_Pal  Pdisc = Disc_Pal::P8COL();
       Gray_Pal  Pgr (30);
       Circ_Pal  Pcirc = Circ_Pal::PCIRC6(30);
       RGB_Pal   Prgb  (5,5,5);
       Elise_Set_Of_Palette SOP(newl(Pdisc)+Pgr+Prgb+Pcirc);

       Video_Display Ecr((char *) NULL);
       Ecr.load(SOP);

       Video_Win   WV (Ecr,SOP,Pt2di(50,50),Pt2di(SZ.x,SZ.y));
       WV = WV.chc(Pt2di(0,0),Pt2di(3,3));
*/




     Tiff_Im  FLenaCol("DOC/lena_col.tif");
     Tiff_Im  FLena("DOC/mini_lena.tif");
     Im2D_U_INT1 I0(256,256);



     ELISE_COPY
     (
           I0.all_pts(),
           FLena.in(),
           I0.out() 
     );

     PS_Window  W = PS("Lena_Median");

          Im2D_U_INT1 Im(256,256);
          ELISE_COPY
          (
              W.all_pts(),
              rect_median(I0.in_proj(),9,256),
              Im.out() | W.ogray()
          );
     {

          W = PS("Lena_lpts_rouge");
          ELISE_COPY ( W.all_pts(),Im.in(), W.ogray());
          Liste_Pts_INT2 l2(2);
          ELISE_COPY
          (
              select(W.all_pts(),Im.in() < 80),
              P8COL::red,
              W.odisc() | l2
          );



          W = PS("Lena_lpts_blue");
          ELISE_COPY ( W.all_pts(),Im.in(), W.ogray());
          ELISE_COPY
          (
              l2.all_pts(),
              P8COL::blue,
              W.odisc()
          );


          W = PS("Lena_lpts_blue_green");
          ELISE_COPY (W.all_pts(),Im.in(), W.ogray());
          ELISE_COPY (l2.all_pts(),P8COL::blue, W.odisc());
          ELISE_COPY
          (
              select(W.all_pts(),Im.in() > 160),
              P8COL::green,
              W.odisc() | l2
          );

          W = PS("Lena_lpts_yellow");
          ELISE_COPY (W.all_pts(),Im.in(), W.ogray());
          ELISE_COPY
          (
              l2.all_pts(),
              P8COL::yellow,
              W.odisc()
          );

          W = PS("Lena_lpts_cyan");
          ELISE_COPY (W.all_pts(),Im.in(), W.ogray());
          ELISE_COPY
          (
              l2.all_pts(),
              P8COL::yellow,
              W.odisc()
          );
          ELISE_COPY
          (
              l2.all_pts().chc(Virgule(FY,FX)),
              P8COL::cyan,
              W.odisc()
          );



          W = PS("Lena_lpts_3D");
          Liste_Pts_INT2 l3(3);
          Fonc_Num pds = Im.in()>128;
          ELISE_COPY
          (
              W.all_pts(),
              pds*I0.in() + (1-pds)*Im.in(),
              W.ogray()
          );

          W = PS("Lena_lpts_image");
          ELISE_COPY
          (
              W.all_pts(),
              pds*(255-I0.in()) + (1-pds)*Im.in(),
              W.ogray()
          );

     }




       W = PS("Lena_morpho_bin");
       Im2D_U_INT1 Ibin(256,256);
       ELISE_COPY
       (
          W.all_pts(),
          I0.in() < 128,
          Ibin.out()
       );
       ELISE_COPY(W.border(1),2,  Ibin.out());
       ELISE_COPY(Ibin.all_pts(),Ibin.in(),W.odisc());




       Pt2di Tv4[4] = {Pt2di(1,0),Pt2di(0,1),Pt2di(-1,0),Pt2di(0,-1)};
       Neighbourhood V4 (Tv4,4);
       Neighbourhood V8 = Neighbourhood::v8();



       Im2D_U_INT1 Ips(256,256,0);
       ELISE_COPY(Ibin.all_pts(),Ibin.in(),Ips.out());
       ELISE_COPY
       (
          dilate
          (
             select(Ibin.all_pts(),Ibin.in() == 1),
             V4
          ),
          P8COL::cyan,
          Ips.out()
       );


       W = PS("Lena_dilate_blue",false,SZ_RED);
       ELISE_COPY(W.all_pts(),trans(Ips.in(),TR),W.odisc());


       ELISE_COPY
       (
          select(Ibin.all_pts(),Ibin.in() == 1),
          P8COL::black,
          Ips.out()
       );
       W = PS("Lena_bord_blue",false,SZ_RED);
       ELISE_COPY(W.all_pts(),trans(Ips.in(),TR),W.odisc());

     
       ELISE_COPY(W.all_pts(),Ibin.in(),Ips.out());
       ELISE_COPY
       (
          dilate
          (
             select(Ibin.all_pts(),Ibin.in() == 1),
             sel_func(V8,Ibin.in() == 0)
          ),
          P8COL::red,
          Ips.out()
       );
       W = PS("Lena_bord_red",false,SZ_RED);
       ELISE_COPY(W.all_pts(),trans(Ips.in(),TR),W.odisc());

       Liste_Pts_INT2 l2(2);
       ELISE_COPY
       (
          dilate
          (
             select(Ibin.all_pts(),Ibin.in() == 1),
             sel_func(V8,Ibin.in() == 0)
          ),
          P8COL::yellow,
          Ips.out() | Ibin.out() | l2
       );
       W = PS("Lena_bord_yel1",false,SZ_RED);
       ELISE_COPY(W.all_pts(),trans(Ips.in(),TR),W.odisc());

       for (int k = 0; k < 5 ; k++)
       {
          Liste_Pts_INT2 newl(2);
          ELISE_COPY
          (
             dilate
             (
                l2.all_pts(),
                Ibin.neigh_test_and_set
                (
                   V8,
                   P8COL::white,
                   P8COL::yellow,
                   20
                )
             ),
             10000,
             newl
          );
          l2 = newl ;
       }
       ELISE_COPY(Ibin.all_pts(),Ibin.in(),Ips.out());
       W = PS("Lena_bord_yel5",false,SZ_RED);
       ELISE_COPY(W.all_pts(),trans(Ips.in(),TR),W.odisc());


      {
          Neighbourhood V8 = Neighbourhood::v8();
          Im2D_U_INT1 Ibin(256,256);
          Im2D_U_INT1 Ips(256,256,0);

          ELISE_COPY
          (
              I0.all_pts(),
              rect_median(I0.in_proj(),2,256)/8 < 8,
              Ips.out() | Ibin.out()
          );
          ELISE_COPY(I0.border(1),0,Ibin.out()|Ips.out());

          Pt2di pt(97,139);
          W = PS("Lena_bin_circle",false);
          ELISE_COPY(W.all_pts(),Ips.in(),W.odisc());
          show_brd(W);
          W.draw_circle_loc(pt,3,Line_St(W.pdisc()(P8COL::red),20));

          ELISE_COPY
          (
              conc
              (
                 pt,
                 Ibin.neigh_test_and_set
                 (
                    V8,
                    P8COL::black,
                    P8COL::magenta,
                    20
                 )
              ),
              P8COL::magenta,
              Ips.out()
          );
          W = PS("Lena_conc_pt",false);
          ELISE_COPY(W.all_pts(),Ips.in(),W.odisc());
          show_brd(W);

          Pt2di p1(1,254),p2(254,1);
          ELISE_COPY
          (
              conc
              (
                 line(p1,p2),
                 Ibin.neigh_test_and_set
                 (
                    V8,
                    P8COL::black,
                    P8COL::green,
                    20
                 )
              ),
              P8COL::green,
              Ips.out()
          );
          W = PS("Lena_conc_line",false);
          ELISE_COPY(W.all_pts(),Ips.in(),  W.odisc());
          ELISE_COPY(line(p1,p2),P8COL::red,W.odisc());
          show_brd(W);

          Im2D_U_INT1 I2(256,256);
          ELISE_COPY
          (
              Ibin.all_pts(),
              Ibin.in()!=0,
              Ibin.out()|I2.out()
          );
          ELISE_COPY
          (
              select
              (
                  Ibin.all_pts(),
                  Ibin.in()[Virgule(FY,FX)]&& (FX<FY) && (!Ibin.in())
               ),
              P8COL::red,
              I2.out()
          );
          W = PS("Lena_conc_sel_1",false);
          ELISE_COPY(W.all_pts(),I2.in(),  W.odisc());
          show_brd(W);


          ELISE_COPY
          (
              conc
              (
                 select(I2.all_pts(),I2.in()==P8COL::red),
                 I2.neigh_test_and_set
                 (
                    V8,
                    P8COL::black,
                    P8COL::blue,
                    20
                 )
              ),
              P8COL::blue,
              Output::onul()
          );
          W = PS("Lena_conc_sel_2",false);
          ELISE_COPY(W.all_pts(),I2.in(),  W.odisc());
          show_brd(W);


           W = PS("Lena_ana_conc",false);
           for (INT f=0; f < 2; f++)
           {
                 ELISE_COPY
                 (
                      Ibin.all_pts(),
                      Ibin.in()!=0,
                      Ibin.out() | Ips.out()
                 );
                 U_INT1 ** d = Ibin.data();

                for (INT x=0; x < 256; x++)
                {
                    for (INT y=0; y < 256; y++)
                    {
                       if (d[y][x] == 1)
                       {
                           Liste_Pts_INT2 cc(2);
                           ELISE_COPY
                           (
                               conc
                               (
                                  Pt2di(x,y),
                                  Ibin.neigh_test_and_set
                                  (
                                     V8,
                                     P8COL::black,
                                     P8COL::green,
                                     20
                                  )
                               ),
                               P8COL::green,
                               Ips.out() | cc
                           );

                           if (cc.card() > 200)
                           {
                              Line_St lstbox(W.pdisc()(P8COL::blue),10);
                              Line_St lstcdg(W.prgb()(256,128,0),20);
                              Pt2di pmax,pmin,cdg;
                              ELISE_COPY
                              (
                                   cc.all_pts(),
                                   Virgule(FX,FY),
                                      (pmax.VMax())
                                   |  (pmin.VMin())
                                   |  (cdg.sigma())
                                   |  (Ips.out() << P8COL::cyan)
                               );
                               if (f == 1)
                               {
                                  W.draw_circle_loc(cdg/cc.card(),3,lstcdg);
                                  W.draw_rect(pmin,pmax+Pt2di(1,1),lstbox);
                               }
                           }
                       }
                    }
                 }
                 if (f==0)
                 {
                    ELISE_COPY(W.all_pts(),Ips.in(),W.odisc());
                    show_brd(W);
                 }
           }

       }

       {
          W = PS("ImBits_bin",false);
          Im2D_Bits<2> Ibin(256,256);
          ELISE_COPY
          (
              W.all_pts(),
              rect_median(I0.in_proj(),2,256)/8 < 8,
              W.odisc() | Ibin.out()
          );


          for (INT x = 0 ; x< 256; x++)
              for (INT y = 0 ; y< 256; y++)
              {
                  INT v = Ibin.get(x,y);
                  Ibin.set(x,y,v+2);
              }

          W = PS("ImBits_set_get",false);
          ELISE_COPY(W.all_pts(),Ibin.in(),W.odisc());

          for (INT y = 0 ; y< 256; y++)
          {
              U_INT1 * d = Ibin.data()[y];
              for (INT x = 0 ; x< 64; x++)
                  d[x] = ~d[x];
          }
          W = PS("ImBits_data",false);
          Im2D_Bits<4> Ips(256,256);
          ELISE_COPY(W.all_pts(),Ibin.in(),W.odisc()|Ips.out());


          ELISE_COPY(W.border(1),1,Ibin.out()|Ips.out());


          Neighbourhood V8 = Neighbourhood::v8();
          ELISE_COPY
          (
               select
               (
                   select(Ibin.all_pts(), Ibin.in()==0),
                   Neigh_Rel(V8).red_max(Ibin.in())
               ),
               P8COL::red,
               Ips.out()
          );
          W = PS("Lena_erod_rouge",false,SZ_RED);
          ELISE_COPY(W.all_pts(),trans(Ips.in(),TR),W.odisc());


          Neighbourhood V4 = Neighbourhood::v4();
          ELISE_COPY
          (
              select(Ibin.all_pts(), Ibin.in()==0),
              2+Neigh_Rel(V4).red_sum(Ibin.in()),
              Ips.out()
          );
          W = PS("Lena_reduc_NR_sum",false,SZ_RED);
          ELISE_COPY(W.all_pts(),trans(Ips.in(),TR),W.odisc());

      }
}







