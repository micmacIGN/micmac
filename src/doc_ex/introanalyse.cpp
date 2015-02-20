#include "StdAfx.h"

#if (ELISE_unix)

int  DocEx_Introanalyse_main(int,char **)
{

    // sz of images we will use
       Pt2di SZ(256,256);

   //  palette allocation
       Disc_Pal  Pdisc = Disc_Pal::P8COL();
       Gray_Pal  Pgr (30);
       Circ_Pal  Pcirc = Circ_Pal::PCIRC6(30);
       RGB_Pal   Prgb  (5,5,5);
       Elise_Set_Of_Palette SOP ( NewLElPal(Pdisc)
								+ Elise_Palette(Pgr)
								+ Elise_Palette(Prgb)
								+ Elise_Palette(Pcirc)	);
   // Creation of video windows
       Video_Display Ecr((char *) NULL);
       Ecr.load(SOP);

       Video_Win   W  (Ecr,SOP,Pt2di(50,50),Pt2di(SZ.x,SZ.y));


       Tiff_Im  FLena = Tiff_Im::StdConv(MMDir() + "data/lena_gray.tif");
	   Tiff_Im  FLenaCol = Tiff_Im::StdConv(MMDir() + "data/lena_col.tif");
       Im2D_U_INT1 I(256,256);

       ELISE_COPY
       (
           W.all_pts(),
           FLena.in(),
           I.out() | W.out(Pgr)
       );
       getchar();

//=========================================================
//
//    LISTE DE POINTS 
//
//=========================================================
      
       {
          Im2D_U_INT1 Im(256,256);
          ELISE_COPY
          (
              W.all_pts(),
              rect_median(I.in_proj(),5,256),
              Im.out() | W.out(Pgr)
          );
          getchar();

          Liste_Pts_INT2 l2(2);
          ELISE_COPY
          (
              select(W.all_pts(),Im.in() < 80),
              P8COL::red,
              W.out(Pdisc) | l2
          );
          getchar();
          ELISE_COPY
          (
              l2.all_pts(),
              P8COL::blue,
              W.out(Pdisc)
          );
          getchar();


          ELISE_COPY
          (
              select(W.all_pts(),Im.in() > 160),
              P8COL::green,
              W.out(Pdisc) | l2
          );
          getchar();
          ELISE_COPY
          (
              l2.all_pts(),
              P8COL::yellow,
              W.out(Pdisc) 
          );
          getchar();
          
          ELISE_COPY
          (
              l2.all_pts().chc(Virgule(FY,FX)),
              P8COL::cyan,
              W.out(Pdisc) 
          );
          getchar();

          ELISE_COPY(W.all_pts(),Im.in(),W.out(Pgr));
          Liste_Pts_INT2 l3(3);
          ELISE_COPY
          (
              select
              (
                  W.all_pts(),
                  Im.in()>128
              ).chc(Virgule(FX,FY,I.in())),
              1,
              l3
          );
          ELISE_COPY(l3.all_pts(),FZ,W.out(Pgr).chc(Virgule(FX,FY)));
          getchar();

          Im2D_INT2 Il3 = l3.image();
          INT2 **  d = Il3.data();
          INT  nb    = Il3.tx();
          INT2 * tx =    d[0];
          INT2 * ty =    d[1];
          INT2 * gray = d[2];
          U_INT1 ** im = Im.data();
          for (INT  k=0 ; k<nb ; k++)
              im[ty[k]][tx[k]] = 255-gray[k];
           ELISE_COPY(Im.all_pts(),Im.in(),W.out(Pgr)); 
          getchar();
       }


//=========================================================
//
//    RELATION DE VOSINAGES ET DILATATION 
//
//=========================================================

       {
           Im2D_U_INT1 Ibin(256,256);

           ELISE_COPY 
           (
              W.all_pts(),
              I.in() < 128,
              W.out(Pdisc) | Ibin.out()
           );
           ELISE_COPY(W.border(1),P8COL::red, W.out(Pdisc) | Ibin.out());
           getchar();
         
           Pt2di Tv4[4] = {Pt2di(1,0),Pt2di(0,1),Pt2di(-1,0),Pt2di(0,-1)};
           Neighbourhood V4 (Tv4,4);
           Neighbourhood V8 = Neighbourhood::v8();
          
           ELISE_COPY
           (
              dilate
              (
                 select(Ibin.all_pts(),Ibin.in() == 1),
                 V4
              ),
              P8COL::cyan,
              W.out(Pdisc)
           );
           getchar();
    
           ELISE_COPY
           (
              select(Ibin.all_pts(),Ibin.in() == 1),
              P8COL::black,
              W.out(Pdisc)
           );
           getchar();
    
           INT nb_pts;
           ELISE_COPY(W.all_pts(),Ibin.in(),W.out(Pdisc));
           ELISE_COPY
           (
              dilate
              (
                 select(Ibin.all_pts(),Ibin.in() == 1),
                 sel_func(V8,Ibin.in() == 0)
              ),
              P8COL::red,
              W.out(Pdisc) | (sigma(nb_pts)<< 1)
           );
           cout << "found " << nb_pts << "\n";
           getchar();

           ELISE_COPY(W.all_pts(),Ibin.in(),W.out(Pdisc));
           Liste_Pts_INT2 l2(2);
           ELISE_COPY
           (
              dilate
              (
                 select(Ibin.all_pts(),Ibin.in() == 1),
                 sel_func(V8,Ibin.in() == 0)
              ),
              P8COL::yellow,
              W.out(Pdisc) | Ibin.out() | l2
           );
           cout << "found " << l2.card() << "\n";
           getchar();
    

           for (int k = 0; k < 5 ; k++)
           {
              Liste_Pts_INT2 aNewL(2);
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
                 aNewL
              );
              l2 = aNewL ;
           }
           ELISE_COPY(Ibin.all_pts(),Ibin.in(),W.out(Pdisc));
           getchar();
       }

//=========================================================
//
//    COMPOSANTES CONNEXES 
//
//=========================================================

      {
          Im2D_U_INT1 Ibin(256,256);
          ELISE_COPY
          (
              W.all_pts(),
              rect_median(I.in_proj(),2,256)/8 < 8,
              W.out(Pdisc) | Ibin.out()
          );
          ELISE_COPY(W.border(1),0,Ibin.out()|W.out(Pdisc));

          Pt2di pt;
          Col_Pal red =Pdisc(P8COL::red);
          for (;;)
          {
             std::cout << "CLIKER SUR UN POINT NOIR \n";
             Pt2dr ptTmp = Ecr.clik()._pt;
             pt.x=round_ni(ptTmp.x);
             pt.y=round_ni(ptTmp.y);
             if (Ibin.data()[pt.y][pt.x] == 1)
             {
                 W.draw_circle_loc(ptTmp,3.0,W.pdisc()(P8COL::red));
                 break;
             }
          }
          getchar();
        
          
          Neighbourhood V8 = Neighbourhood::v8();
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
              W.out(Pdisc)
          );
          getchar();
          
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
              W.out(Pdisc)
          );
          ELISE_COPY(line(p1,p2),P8COL::red,W.out(Pdisc));
          getchar();



          Im2D_U_INT1 I2(256,256);
          ELISE_COPY
          (
              Ibin.all_pts(),
              Ibin.in()!=0,
              Ibin.out()|I2.out()|W.out(Pdisc)
          );
          ELISE_COPY
          (
              select
              (
                  Ibin.all_pts(),
                  Ibin.in()[Virgule(FY,FX)]&& (FX<FY) && (!Ibin.in())
               ),
              P8COL::red,
              I2.out()|W.out(Pdisc)
          );
          getchar();
          
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
              W.out(Pdisc)
          );
          getchar();



          ELISE_COPY(Ibin.all_pts(),Ibin.in()!=0,Ibin.out()|W.out(Pdisc));
          U_INT1 ** d = Ibin.data();

   // Parcourt toute les composant connexes de l'image
   // et les met en vert, si elles ont + de 200 point:
   //   * les mets en cyan
   //   * affiche leur boite englobant et centre de gravite

          for (INT x=0; x < 256; x++)
               for (INT y=0; y < 256; y++)
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
                          W.out(Pdisc) | cc
                      );

                      if (cc.card() > 200)
                      {
                         Line_St lstbox(Prgb(0,0,255),2);
                         Line_St lstcdg(Prgb(255,128,0),3);
                         Pt2dr pmax,pmin,cdg;
                         ELISE_COPY
                         (
                              cc.all_pts(),
                              Virgule(FX,FY),
                                 (pmax.VMax())
                              |  (pmin.VMin())
                              |  (cdg.sigma())
                              |  (W.out(Pdisc) << P8COL::cyan)
                          );
                          W.draw_circle_loc(cdg/cc.card(),5,lstcdg);
                          W.draw_rect(pmin,pmax,lstbox);
                      }
                  }
          getchar();
       }


      {
          Im2D_Bits<2> Ibin(256,256);
          ELISE_COPY
          (
              W.all_pts(),
              rect_median(I.in_proj(),2,256)/8 < 8,
              W.out(Pdisc) | Ibin.out()
          );
       
          Col_Pal red =Pdisc(P8COL::red);
          getchar();


          for (INT x = 0 ; x< 256; x++)
              for (INT y = 0 ; y< 256; y++)
              {
                  INT v = Ibin.get(x,y);
                  Ibin.set(x,y,v+2);
              }

          ELISE_COPY(W.all_pts(),Ibin.in(),W.out(Pdisc));
          getchar();

          for (INT y = 0 ; y< 256; y++)
          {
              U_INT1 * d = Ibin.data()[y];
              for (INT x = 0 ; x< 64; x++)
                  d[x] = ~d[x];
          }
          ELISE_COPY(W.all_pts(),Ibin.in(),W.out(Pdisc));
          getchar();



          ELISE_COPY(W.border(1),1,Ibin.out()|W.out(Pdisc));
          Neighbourhood V8 = Neighbourhood::v8();
          ELISE_COPY
          (
               select
               (
                   select(W.all_pts(), Ibin.in()==0),
                   Neigh_Rel(V8).red_max(Ibin.in())
               ),
               P8COL::red,
               W.out(Pdisc)
          );
          getchar();


          Neighbourhood V4 = Neighbourhood::v4();
          ELISE_COPY
          (
              select(W.all_pts(), Ibin.in()==0),
              2+Neigh_Rel(V4).red_sum(Ibin.in()),
              W.out(Pdisc)
          );
          getchar();    
      }
      
      return 0;
}

#endif





