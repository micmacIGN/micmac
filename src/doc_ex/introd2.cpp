#include "StdAfx.h"




int  DocEx_Introd2_main(int,char **)
{
    // sz of images we will use
        Pt2di SZ(256,256);

   //  palette allocation
        Disc_Pal  Pdisc = Disc_Pal::P8COL();
        Gray_Pal  Pgr (30);
        Circ_Pal  Pcirc = Circ_Pal::PCIRC6(30);
        RGB_Pal   Prgb  (5,5,5);
        Elise_Set_Of_Palette SOP(NewLElPal(Pdisc)+Elise_Palette(Pgr)+Elise_Palette(Prgb)+Elise_Palette(Pcirc));

   // Creation of video windows
        Video_Display Ecr((char *) NULL);
        Ecr.load(SOP);
        Video_Win   W  (Ecr,SOP,Pt2di(50,50),Pt2di(SZ.x,SZ.y));

        W.set_title("Une fenetre");

    // show_FX
        ELISE_COPY(W.all_pts(),FX,W.ogray());
        getchar();


    // 2-3-4
        ELISE_COPY(W.all_pts(),FY,W.ogray());
        getchar();

        ELISE_COPY(W.all_pts(),(FX+FY)/2,W.ogray());
        getchar();

        ELISE_COPY
        (
             W.all_pts(),
             127.5 * (1.0 + cos(FY*0.2+0.06*FX*sin(0.06*FX))),
             W.ogray()
        );
        getchar();

    //  5
       Tiff_Im  FLena = Tiff_Im::StdConv(MMDir() + "data/lena_gray.tif");
       Im2D_U_INT1 I(256,256);
       ELISE_COPY
       (
           I.all_pts(),
           FLena.in(),
           I.out() | W.ogray()
       );
       getchar();

    //  6
       ELISE_COPY
       (
           I.all_pts(),
           255-I.in(),
           W.ogray()
       );
       getchar();

    //  7


       ELISE_COPY
       (
           I.all_pts(),
           I.in(),
           W.out(Pcirc)
       );
       getchar();

       ELISE_COPY
       (
           I.all_pts(),
           I.in(),
           W.out(Pgr)
       );
       getchar();

       ELISE_COPY
       (
           I.all_pts(),
           I.in()/64,
           W.out(Pdisc)
       );
       getchar();


    //
       ELISE_COPY
       (
           I.all_pts(),
           I.in()<128,
           W.out(Pdisc)
       );
       getchar();
       ELISE_COPY
       (
           I.all_pts(),
           I.in()<FX,
           W.out(Pdisc)
       );
       getchar();
       ELISE_COPY
       (
           I.all_pts(),
           I.in()<127.5*(1+sin(FX)*sin(FY)),
           W.out(Pdisc)
       );
       getchar();



      // operateur ,
 


       ELISE_COPY
       (
           I.all_pts(),
           I.in()[Virgule(FY,FX)],
           W.out(Pgr)
       );
       getchar();

       ELISE_COPY
       (
           I.all_pts(),
           I.in()[Virgule(FX*3,FY*2)%256],
           W.out(Pgr)
       );
       getchar();

       ELISE_COPY
       (  
           I.all_pts(),
           I.in(0)
           [Virgule( 
              128+(FX-128)+(FY-128)/2.0,
              128-(FX-128)/2.0+(FY-128)
             )
           ],
           W.out(Pgr)
       );
       getchar();


       ELISE_COPY
       (
           I.all_pts(),
           I.in(0)
           [Virgule(
               FX+20*sin(FX/50.0)+4*cos(FY/15.0),
               FY+8*sin(FX/20.0)+7*cos(FY/18.0)
            )
           ],
           W.out(Pgr)
       );
       getchar();


       ELISE_COPY
       (
           I.all_pts(),
           (I.in()+I.in()[Virgule(FY,FX)])/2,
           W.out(Pgr)
       );
       getchar();
   // 
       ELISE_COPY
       (
           I.all_pts(),
           pow(I.in()/255.0,5.0/3.0)*255.0,
           W.out(Pgr)
       );
       getchar();

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
              W.out(Pgr)
          );
          getchar();
       }

       {
          Im1D_U_INT1 lut(256,0);
          U_INT1 * d = lut.data();
          d[1]  = d[2] = d[5] = 1;
          d[10] = d[13] = d[14] = 2;
          ELISE_COPY(I.all_pts(),I.in(),W.out(Pgr));
          ELISE_COPY
          (
              I.all_pts(),
              lut.in()[I.in()/16],
              W.out(Pdisc)
          );
          getchar();
       }



   // 
       ELISE_COPY
       (
           I.all_pts(),
           Virgule(FX,0,FY),
           W.out(Prgb)
       );
       getchar();

       ELISE_COPY
       (
           I.all_pts(),
           Virgule(FX,FY,0),
           W.out(Prgb)
       );
       getchar();

       ELISE_COPY
       (
           I.all_pts(),
           Virgule(I.in(),0,I.in()[Virgule(FY,FX)]),
           W.out(Prgb)
       );
       getchar();

       Tiff_Im  FLenaCol =  Tiff_Im::StdConv(MMDir() + "data/lena_col.tif");
       ELISE_COPY
       (
           I.all_pts(),
           FLenaCol.in(),
           W.out(Prgb)
       );
       getchar();


       ELISE_COPY
       (
           I.all_pts(),
           Virgule(
              FLenaCol.in().v1(),
              FLenaCol.in().v0(),
              FLenaCol.in().v2()
           ),
           W.out(Prgb)
       );
       getchar();



    //  8
       ELISE_COPY(I.all_pts(),I.in(),W.out(Pgr));
       ELISE_COPY
       (
           select(I.all_pts(),I.in()>128),
           P8COL::red,
           W.out(Pdisc)
       );
       getchar();

       ELISE_COPY
       (
           I.all_pts(),
           P8COL::blue,
           W.out(Pdisc)
       );
       ELISE_COPY
       (
           disc(Pt2dr(128,128),100),
           I.in(),
           W.out(Pgr)
       );
       getchar();

    ELISE_COPY
    (
        ell_fill(Pt2dr(128,128),135,70,1.2),
        255-I.in(0),
        W.out(Pgr)
    );
    getchar();

    ELISE_COPY
    (
        sector_ang(Pt2dr(128,128),100,1.0,3.0),
        (I.in() < 128),
        W.out(Pdisc)
    );
    getchar();


   for (INT x = 0; x < 256; x+= 4)
   {
        ELISE_COPY
        (
            line(Pt2di(x,0),Pt2di(128,128)),
            I.in(),
            W.out(Pgr)
        );
    }
    getchar();


    ELISE_COPY
    (
         polygone
         (
                 NewLPt2di(Pt2di(5,250))+Pt2di(128,5)
             +   Pt2di(250,250)+Pt2di(128,128)
         ),
         (I.in()/64)*64,
         W.out(Pgr)
    );
    getchar();


    for (INT x = 1; x< 5; x++)
        ELISE_COPY
        (
            border_rect(Pt2di(10,10),Pt2di(15+10*x,15+10*x),5-x),
            P8COL::white,
            W.out(Pdisc)
        );


       

    ELISE_COPY
    (
        W.border(8),
        P8COL::green,
        W.out(Pdisc)
    );


    ELISE_COPY
    (
        ellipse(Pt2dr(128,128),135,70,1.2),
         P8COL::red,
        W.out(Pdisc)
    );
    getchar();

       ELISE_COPY(I.all_pts(),255,W.out(Pgr));
       ELISE_COPY
       (
            disc(Pt2dr(64,64),64).chc(2*Virgule(FX,FY)),
            0,
            W.out(Pgr)
       );
       getchar();

       ELISE_COPY(I.all_pts(),I.in(),W.out(Pgr));
       ELISE_COPY
       (
            rectangle(Pt2di(0,0),Pt2di(256,128)).chc(Virgule(FX,FY*2)),
            I.in()[Virgule(FX,255-FY)],
            W.out(Pgr)
       );
       getchar();



       ELISE_COPY(I.all_pts(),I.in(),W.out(Pgr));
       ELISE_COPY
       (
            rectangle(Pt2di(0,0),Pt2di(256,128)).chc(Virgule(FX,2*FY-FY%2)),
            FY,
            W.out(Pgr)
       );
       getchar();



      //  On declare trois fenetre Wr, Wg et Wb
        Video_Win   Wr = W;  
        Video_Win   Wg (Ecr,SOP,Pt2di(50,50),SZ);
        Video_Win   Wb (Ecr,SOP,Pt2di(50,50),SZ);

        Wr.set_title("red chanel");
        Wg.set_title("green chanel");
        Wb.set_title("blue chanel");

        // affichage des 3 canaux en niveau de gris, version "bovine"
        ELISE_COPY
        (
            W.all_pts(),
            0,
               (Wr.out(Pgr) << FLenaCol.in().v0())
            |  (Wg.out(Pgr) << FLenaCol.in().v1())
            |  (Wb.out(Pgr) << FLenaCol.in().v2())
        );

        getchar();

        Wr.clear(); Wg.clear(); Wb.clear();

        // affichage des 3 canaux en niveau de gris,
        //  version operateur "," sur les Output
        ELISE_COPY
        (
            W.all_pts(),
            FLenaCol.in(),
            (Wr.out(Pgr),Wg.out(Pgr),Wb.out(Pgr))
        );
        getchar();


       Wr.clear(); Wg.clear(); Wb.clear();
    
       Im2D_U_INT1 R(256,256);
       Im2D_U_INT1 G(256,256);
       Im2D_U_INT1 B(256,256);

// affichage des trois canaux dans Wr, Wg et Wb
// et memo en meme temps dans R,G et B
       ELISE_COPY
       (
            W.all_pts(),
            FLenaCol.in(),
               (Wr.out(Pgr),Wg.out(Pgr),Wb.out(Pgr))
            |  (R.out(),G.out(),B.out())
       );
// verif que R,G,B contiennent les bonnes valeurs
       ELISE_COPY
       (
            W.all_pts(),
            Virgule(R.in(),G.in(),B.in()),
            W.out(Prgb)
       );
       getchar();

// une autre facon de faire 
// affichage des trois canaux dans Wr, Wg et Wb
// et memo en meme temps dans R,G et B
       ELISE_COPY
       (
            W.all_pts(),
            FLenaCol.in(),
            (
               Wr.out(Pgr)|R.out(),
               Wg.out(Pgr)|G.out(),
               Wb.out(Pgr)|B.out()
            )
       );
       ELISE_COPY
       (
            W.all_pts(),
            Virgule(R.in(),G.in(),B.in()),
            W.out(Prgb)
       );
       getchar();


       return 0;
}






