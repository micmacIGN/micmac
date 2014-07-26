#include "general/all.h"

template <class  Type>
         Type sobel_el (Type ** im,INT x,INT y)
{
   return
       ElAbs
       (
          im[y-1][x-1]+2*im[y][x-1]+im[y+1][x-1]
        - im[y-1][x+1]-2*im[y][x+1]-im[y+1][x+1]
       )
     + ElAbs
       (
          im[y-1][x-1]+2*im[y-1][x]+im[y-1][x+1]
        - im[y+1][x-1]-2*im[y+1][x]-im[y+1][x+1]
       );
}


template <class  Type>  void
         sobel_buf
         (
            Type **                  out,
            Type ***                 in,
            const Simple_OPBuf_Gen & arg
         )
{
     for (int d =0; d<arg.dim_out(); d++)
          for (int x = arg.x0(); x<arg.x1() ; x++)

               out[d][x] = sobel_el(in[d],x,0);
}

Fonc_Num sobel(Fonc_Num f)
{
     return create_op_buf_simple_tpl
            (
                sobel_buf, sobel_buf,
                f, f.dimf_out(),
                Box2di(Pt2di(-1,-1),Pt2di(1,1))
            );
}




PS_Window PS(const char * name, bool auth_lzw = false)
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
                                  + Elise_Palette(Prb)
                                  + Elise_Palette(Pgr)
                                  + Elise_Palette(Prgb)
                                  + Elise_Palette(Pcirc)
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
       Pt2di SZ(256,256);

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
*/




     Tiff_Im  FLenaCol("DOC/lena_col.tif");
     Tiff_Im  FLena("DOC/mini_lena.tif");
     Im2D_U_INT1 I(256,256);



     ELISE_COPY
     (
           I.all_pts(),
           FLena.in(),
           I.out() 
     );

       PS_Window  W = PS("LenaGradX");
       ELISE_COPY
       (
           W.all_pts(),
           Min(255,3*Abs(I.in(0)- I.in(0)[Virgule(FX+1,FY)])),
           W.ogray()
       );

       W = PS("LenaGradY");
       ELISE_COPY
       (
           W.all_pts(),
           Min(255,3*Abs(I.in(0)-trans(I.in(0),Pt2di(0,1)))),
           W.ogray()
       );


       W = PS("LenaClipFile");
       ELISE_COPY(W.all_pts(),P8COL::red,W.odisc());
       ELISE_COPY
       (
          rectangle(Pt2di(0,0),Pt2di(128,128)),
          trans(FLena.in(),Pt2di(64,64)),
          W.ogray() 
       );


       W = PS("LenaSom49Trans");
       Fonc_Num f = 0;
       for (INT x = -3; x <= 3; x++)
           for (INT y = -3; y <= 3; y++)
               f = f + trans(I.in(0),Pt2di(x,y));
       ELISE_COPY
       (
           W.all_pts(),
           f/49,
           W.ogray()
       );

       W = PS("Lena_rect_max");
       ELISE_COPY
       (
           W.all_pts(),
           rect_max(I.in(0),12),
           W.ogray()
       );

       W = PS("Lena_rect_max_min");
       ELISE_COPY
       (
           W.all_pts(),
           rect_min(rect_max(I.in(0),7),7),
           W.ogray()
       );

       {
           REAL fact = 0.9;
           W = PS("Lena_can_exp_filt");
           ELISE_COPY
           (
               W.all_pts(),
                      canny_exp_filt(I.in(0),fact,fact)
                  /   canny_exp_filt(I.inside(),fact,fact),
               W.ogray()
           );
       }

       {
           REAL fact = 0.9;
           W = PS("Lena_can_exp_filt_col");
           ELISE_COPY
           (
               W.all_pts(),
                      canny_exp_filt(FLenaCol.in(0),fact,fact)
                  /   canny_exp_filt(I.inside(),fact,fact),
               W.orgb()
           );
       }

       
       W = PS("LenDerNorm");
       PS_Window  Wteta = PS("LenDerTeta");

       ELISE_COPY
       (
           W.all_pts(),
           mod
           (
              Min
              (
                   Iconv
                   (     polar(deriche(I.in(0),1.0),0)
                       * Fonc_Num(1.0, 256/(2*PI))
                   ),
                   255
              ),
              256
           ),
           Virgule(W.ogray(), Wteta.ocirc())
       );

       W = PS("Len_ef_binar");
       Im2D_U_INT1 Ibin(256,256);
       {
          REAL fact = 0.8;
          ELISE_COPY
          (
              W.all_pts(),
                  canny_exp_filt(I.in(0),fact,fact)
              /   canny_exp_filt(I.inside(),fact,fact) > 128,
              W.odisc()
             | Ibin.out()
          );
       }

       Im2D_U_INT1 Imorph(256,256);
       ELISE_COPY
       (
           W.all_pts(),
           Ibin.in(0),
           Imorph.out()
       );
       ELISE_COPY
       (
           select(W.all_pts(),open_5711(Ibin.in(0),40)),
           P8COL::cyan,
           Imorph.out()
       );
       ELISE_COPY
       (
           select(W.all_pts(),erod_5711(Ibin.in(0),40)),
           P8COL::blue,
           Imorph.out()
       );
       W = PS("Len_open_erod");
       ELISE_COPY(W.all_pts(),Imorph.in(),W.odisc());

       

       ELISE_COPY
       (
           W.all_pts(),
             dilat_5711(Ibin.in(0),40)
           * P8COL::yellow,
           Imorph.out()
       );
       ELISE_COPY
       (
           select(W.all_pts(),close_5711(Ibin.in(0),40)),
           P8COL::green,
           Imorph.out()
       );
       ELISE_COPY
       (
           select(W.all_pts(),Ibin.in(0)),
           P8COL::black,
           Imorph.out()
       );
       W = PS("Len_dilat_close");
       ELISE_COPY(W.all_pts(),Imorph.in(),W.odisc());

      
       {
          Im2D_U_INT1 Idist(256,256);
          INT vmax;
          ELISE_COPY
          (
              W.all_pts(),
              extinc_32(Ibin.in(0)),
              VMax(vmax) | Idist.out()
          );
          ELISE_COPY
          (
                Idist.all_pts(),
                Idist.in()*(255.0/vmax),
                Idist.out()
          );
          ELISE_COPY
          (
                select(Idist.all_pts(),! Ibin.in()),
                255,
                Idist.out()
          );
          W = PS("Len_extinc");
          ELISE_COPY
          (
              W.all_pts(), 
              Idist.in(),
              W.ogray() 
          );
       }


      for (int i = 0 ; i < 2; i++)
      {
           W = PS(i?"buglena_histo1" :"lena_histo1");
           Disc_Pal  Pdisc = W.pdisc();

            Im1D_INT4 H(256,0);
            ELISE_COPY(W.all_pts(),I.in(),W.ogray());
            ELISE_COPY
            (
               W.all_pts().chc(I.in()),
               1,
               H.histo()
            );

           INT hmax;
           ELISE_COPY(H.all_pts(),H.in(),VMax(hmax));

           Plot_1d  Plot1
                     (
                             W,
                             Line_St(Pdisc(P8COL::green),3),
                             Line_St(Pdisc(P8COL::black),2),
                             Interval(0,256),
                                NewlArgPl1d(PlBox(Pt2di(3,3),SZ-Pt2di(3,3)))
                             // + PlAutoScalOriY(true)
                             + Arg_Opt_Plot1d(PlOriY(0.0))
                             + Arg_Opt_Plot1d(PlScaleY(255.0/hmax))
                             + Arg_Opt_Plot1d(PlBoxSty(Pdisc(P8COL::blue),3))
                             + Arg_Opt_Plot1d(PlModePl(Plots::line))
                             + Arg_Opt_Plot1d(PlotLinSty(Pdisc(P8COL::red),8))
                     );

            ELISE_COPY(Plot1.all_pts(),H.in(),Plot1.out());
            if (i)
            {
                 Im1D_INT4 H2(256,0);
                 ELISE_COPY
                 (
                    W.all_pts().chc(I.in()),
                    1+H2.in(),
                    H2.out()
                 );
                 Plot1.set
                 (
                     NewlArgPl1d(PlotLinSty(Pdisc(P8COL::green),8))
                 );
                 ELISE_COPY(Plot1.all_pts(),H2.in(),Plot1.out());
            }
       }

       {
            Im2D_INT4 Cooc(256,256,0);
            INT cmax;
            W = PS("Len_Cooc");
            Disc_Pal  Pdisc = W.pdisc();
            ELISE_COPY
            (
               W.interior(1).chc
               (
                   Virgule
                   (
                     I.in(),
                     trans(I.in(),Pt2di(1,0))
                   )
               ),
               1,
                  Cooc.histo()
               |  (VMax(cmax) << Cooc.in())
            );
            ELISE_COPY
            (
                  Cooc.all_pts(),
                     255 -log(Cooc.in()+1)
                  *  (255.0/log(cmax+1)),
                  W.ogray()
            );
            W.draw_rect(Pt2di(0,0),SZ,Line_St(Pdisc(P8COL::red),6));
       }

       W = PS("sobel_gray");
       ELISE_COPY
       (
             W.all_pts(),
             Min(255,sobel(I.in(0))),
             W.ogray()
       );
       W = PS("sobel_col");
       ELISE_COPY
       (
             W.all_pts(),
             Min(255,sobel(FLenaCol.in(0))),
             W.orgb()
       );
       W = PS("sobelc_cany");
       {
            REAL f = 0.9;
            Fonc_Num Fonc =
                    canny_exp_filt(I.in(0),f,f)
                 /  canny_exp_filt(I.inside(),f,f);
            ELISE_COPY
            (
                  W.all_pts(),
                  Min(255,sobel(Fonc)*8),
                  W.ogray()
            );
       }

}







