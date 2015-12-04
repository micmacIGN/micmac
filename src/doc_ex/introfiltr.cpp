#include "StdAfx.h"

template  <class T> void f (T * d, INT nb)
{
   for (INT i=0;i<nb;i++) cout<<d[i]<<"\n";
}


Fonc_Num moy(Fonc_Num f,INT nb)
{
   Fonc_Num res = 0;
   for (INT x = -nb; x <= nb; x++)
       for (INT y = -nb; y <= nb; y++)
           res = res + trans(f,Pt2di(x,y));
   return res/ElSquare(2*nb+1);
}


Fonc_Num sobel_0(Fonc_Num f)
{
    Im2D_REAL8 Fx
               (  3,3,
                  " -1 0 1 "
                  " -2 0 2 "
                  " -1 0 1 "
                ); 
    Im2D_REAL8 Fy
               (  3,3,
                  " -1 -2 -1 "
                  "  0  0  0 "
                  "  1  2  1 "
                ); 
   return 
       Abs(som_masq(f,Fx,Pt2di(-1,-1)))
     + Abs(som_masq(f,Fy));
}


template <class  Type,class TyBase> class Filters
{  
  public :

  static inline TyBase  sobel (Type ** im,INT x,INT y)
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
};

template <class Type,class TyBase> 
void std_sobel 
     (
          Im2D<Type,TyBase> Iout,
          Im2D<Type,TyBase> Iin,
          Pt2di             p0,
          Pt2di             p1
     )
{
    Type ** out = Iout.data();
    Type **  in = Iin.data();
    INT x1 = ElMin3(Iout.tx()-1,Iin.tx()-1,p1.x);
    INT y1 = ElMin3(Iout.ty()-1,Iin.ty()-1,p1.y);
    INT x0 = ElMax(1,p0.x);
    INT y0 = ElMax(1,p0.y);
// pour eviter les overflow
    TyBase vmax = Iout.vmax()-1;
    for (INT x=x0; x<x1 ; x++)
        for (INT y=y0; y<y1 ; y++)
        {
            out[y][x] = ElMin(vmax,Filters<Type,TyBase>::sobel(in,x,y));
        }
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
               out[d][x] = Filters<Type,Type>::sobel(in[d],x,0);
}

Fonc_Num sobel(Fonc_Num f)
{
     return create_op_buf_simple_tpl
            (
                sobel_buf,
                sobel_buf,
                f,
                f.dimf_out(),
                Box2di(Pt2di(-1,-1),Pt2di(1,1))
            );
}





int  DocEx_Introfiltr_main(int,char **)
{

    // sz of images we will use
       Pt2dr SZ(256,256);

   //  palette allocation
       Disc_Pal  Pdisc = Disc_Pal::P8COL();
       Gray_Pal  Pgr (30);
       Circ_Pal  Pcirc = Circ_Pal::PCIRC6(30);
       RGB_Pal   Prgb  (5,5,5);
       Elise_Set_Of_Palette SOP
	   (	NewLElPal(Pdisc)
			+ Elise_Palette(Pgr)
			+ Elise_Palette(Prgb)
			+ Elise_Palette(Pcirc)	);

   // Creation of video windows
       Video_Display Ecr((char *) NULL);
       Ecr.load(SOP);

       Video_Win   W2 (Ecr,SOP,Pt2di(50,50),Pt2di(SZ.x,SZ.y));
       Video_Win   W  (Ecr,SOP,Pt2di(50,50),Pt2di(SZ.x,SZ.y));

       Plot_1d  Plot1
                (
                        W,
                        Line_St(Pdisc(P8COL::green),3),
                        Line_St(Pdisc(P8COL::black),2),
                        Interval(0,256),
                           NewlArgPl1d(PlBox(Pt2dr(3,3),Pt2dr (SZ)-Pt2dr(3,3)))
                           
                        + Arg_Opt_Plot1d(PlAutoScalOriY(true))
                        + Arg_Opt_Plot1d(PlBoxSty(Pdisc(P8COL::blue),3))
                        + Arg_Opt_Plot1d(PlModePl(Plots::line))
                        + Arg_Opt_Plot1d(PlotLinSty(Pdisc(P8COL::red),2))
               );
                     

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

/*
       ELISE_COPY
       (
           W.all_pts(),
           Min(255,Abs(I.in(0)[(FX+1,FY)]-I.in(0))*3),
           W.out(Pgr)
       );
       getchar();
*/

       ELISE_COPY
       (
           W.all_pts(),
           Min(255,3*Abs(I.in(0)-trans(I.in(0),Pt2di(0,1)))),
           W.out(Pgr)
       );
       getchar();
       
       ELISE_COPY(W.all_pts(),P8COL::red,W.out(Pdisc));
       
       ELISE_COPY
       (
          rectangle(Pt2di(0,0),Pt2di(128,128)),
          trans(FLena.in(),Pt2di(64,64)),
          W.out(Pgr)
       );
       getchar();


      // voir la deinifition de moy en tete
      // de fichier
       ELISE_COPY
       (
           W.all_pts(),
           moy(I.in(0),3),
           W.out(Pgr)
       );

       ELISE_COPY
       (
           W.all_pts(),
           rect_max(I.in(0),12), 
           W.out(Pgr)
       );
       getchar();


       ELISE_COPY
       (
           W.all_pts(),
           rect_min
           (
               rect_max(I.in(0),Pt2di(7,7)),
               Box2di(Pt2di(-7,-7),Pt2di(7,7))
           ), 
           W.out(Pgr)
       );
       getchar();

       {
          REAL fact = 0.9;
          ELISE_COPY
          (
              W.all_pts(),
                  canny_exp_filt(I.in(0),fact,fact) 
              /   canny_exp_filt(I.inside(),fact,fact),
              W.out(Pgr)
          );
       }
       getchar();

       {
          REAL fact = 0.9;
          ELISE_COPY
          (
              W.all_pts(),
                  canny_exp_filt(FLenaCol.in(0),fact,fact) 
              /   canny_exp_filt(I.inside(),fact,fact),
              W.out(Prgb)
          );
       }
       getchar();
/*
       ELISE_COPY
       (
           W.all_pts(),
           Min
           (
                  polar(deriche(I.in(0),1.0),0)
              * Fonc_Num(1.0, 256/(2*PI)),
              255
           ),
           (W.out(Pgr), W2.out(Pcirc))
       );
       getchar();
*/
    
       Im2D_U_INT1 Ibin(256,256);
       {
          REAL fact = 0.8;
          ELISE_COPY
          (
              W.all_pts(),
                  canny_exp_filt(I.in(0),fact,fact) 
              /   canny_exp_filt(I.inside(),fact,fact) < 128,
              W.out(Pdisc)
             | Ibin.out()
          );
          getchar();
       }


       ELISE_COPY
       (
           W.all_pts(),
           Ibin.in(0),
           W.out(Pdisc)
       );
       getchar();
       
       ELISE_COPY
       (
           select(W.all_pts(),open_5711(Ibin.in(0),40)),
           P8COL::cyan,
           W.out(Pdisc)
       );
	   getchar();
       
       ELISE_COPY
       (
           select(W.all_pts(),erod_5711(Ibin.in(0),40)),
           P8COL::blue,
           W.out(Pdisc)
       );
       getchar();

       ELISE_COPY
       (
           W.all_pts(),
             dilat_5711(Ibin.in(0),40)
           * P8COL::yellow,
           W.out(Pdisc)
       );
       ELISE_COPY
       (
           select(W.all_pts(),close_5711(Ibin.in(0),40)),
           P8COL::green,
           W.out(Pdisc)
       );
       ELISE_COPY
       (
           select(W.all_pts(),Ibin.in(0)),
           P8COL::black,
           W.out(Pdisc)
       );
       getchar();

       {
          W.clear();
          Im2D_U_INT1 Is(256,256,0);
          std_sobel(Is,I,Pt2di(0,0),Pt2di(256,256));
          ELISE_COPY(Is.all_pts(),Is.in(),W.out(Pgr));
          getchar();
       }
       ELISE_COPY
       (
             W.all_pts(),
             Min(255,sobel_0(I.in(0))),
             W.out(Pgr)
       );
       getchar();

       W.clear();
       ELISE_COPY
       (
             W.all_pts(),
             Min(255,sobel(I.in(0))),
             W.out(Pgr)
       );
       getchar();

       ELISE_COPY
       (
             W.all_pts(),
             Min(255,sobel(FLenaCol.in(0))),
             W.out(Prgb)
       );
       getchar();

       {
            REAL f = 0.9;
            Fonc_Num Fonc = 
                    canny_exp_filt(I.in(0),f,f)
                 /  canny_exp_filt(I.inside(),f,f);
            ELISE_COPY
            (
                  W.all_pts(),
                  Min(255,sobel(Fonc)*8),
                  W.out(Pgr)
            );
       }
       getchar();
       
       
       Im2D_U_INT1 Idist(256,256);
       INT vmax;
       ELISE_COPY
       (
          W.all_pts(),
          extinc_32(Ibin.in(0)),
          VMax(vmax) | Idist.out() | W.out(Pgr)
       );
       getchar();
       
       ELISE_COPY(Idist.all_pts(),Idist.in()*(255.0/vmax),W.out(Pgr));
       getchar();       
       
       ELISE_COPY
       (
          select(W.all_pts(), ! Ibin.in()),
          P8COL::white,
          W.out(Pdisc) 
       );
       getchar();

       Im1D_INT4 H(256,0);
       ELISE_COPY(W.all_pts(),I.in(),W.out(Pgr));
       ELISE_COPY
       (
          W.all_pts().chc(I.in()),
          1,
          H.histo()
       );
       
       ELISE_COPY(Plot1.all_pts(),H.in(),Plot1.out());
       getchar();

		std::cout << "Last examples need refurbishment" << std::endl;
/*
       Im2D_INT4 Cooc(256,256,0);
       INT cmax;
       ELISE_COPY
       (
          W.interior(1).chc
          ((
                I.in(),
                trans(I.in(),Pt2di(1,0))
          )),
          1,
             Cooc.histo()
          |  (VMax(cmax) << Cooc.in())
       );
       
       getchar();
       
       ELISE_COPY
       (
             Cooc.all_pts(),
                255 -log(Cooc.in()+1)
             *  (255.0/log(cmax+1)),
             W.out(Pgr)
       );
       getchar();
*/

    return 0;      
}
