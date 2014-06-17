#include "general/all.h"


typedef void (* ACT)(Plot_1d,Line_St);

   //************************
   //  1-1  
   //************************

       //=======
       //  1-1-1 
       //=======


void TEST_plot_FX (Plot_1d plot,Line_St)
{
     ELISE_COPY
     (
         plot.all_pts(),
         FX,
         plot.out()
     );
}


void TEST_plot_2 (Plot_1d plot,Line_St)
{
     ELISE_COPY
     (
         plot.all_pts(),
         2,
         plot.out()
     );
}



Im1D_REAL4 init_image(INT tx)
{
     Im1D_REAL4 I(tx);
     REAL4  * d = I.data();

     for (INT x =0; x < tx ; x++)
         d[x] = (x*x)/ (double) tx;
     return I;
}
void TEST_plot_Im0 (Plot_1d plot,Line_St)
{
     Im1D_REAL4 I = init_image(40);
     ELISE_COPY
     (
         I.all_pts(),
         I.in(),
         plot.out()
     );
}
void TEST_plot_Im1 (Plot_1d plot,Line_St)
{
     Im1D_REAL4 I = init_image(40);
     ELISE_COPY
     (
         plot.all_pts(),
         I.in(4.5),
         plot.out()
     );
}
void TEST_plot_Im2 (Plot_1d plot,Line_St)
{
     Im1D_REAL4 I = init_image(40);
     ELISE_COPY
     (
         plot.all_pts(),
         I.in_proj(),
         plot.out()
     );
}


       //=======
       //  1 -2 
       //=======


void TEST_plot_expr_1 (Plot_1d plot,Line_St)
{
     ELISE_COPY
     (
         plot.all_pts(),
         FX/2.0,
         plot.out()
     );
}
void TEST_plot_expr_2 (Plot_1d plot,Line_St)
{
     ELISE_COPY
     (
         plot.all_pts(),
         4*cos(FX/2.0)+ 3+ (FX/5.0) * sin(FX/4.9),
         plot.out()
     );
}
void TEST_plot_expr_Im0 (Plot_1d plot,Line_St)
{
     Im1D_REAL4 I = init_image(40);
     ELISE_COPY
     (
         I.all_pts(),
         40-1.7*I.in(),
         plot.out()
     );
}


void TEST_plot_expr_Im1 (Plot_1d plot,Line_St)
{
     Im1D_REAL4 I = init_image(40);
     ELISE_COPY
     (
         plot.all_pts(),
         3.0*I.in()[Abs(FX)%20],
         plot.out()
     );
}


   //************************
   //  2  
   //************************

       //=======
       //  2 -1 
       //=======


void TEST_plot_rects (Plot_1d plot,Line_St)
{
     ELISE_COPY
     (
         rectangle(-50,-40),
         FX,
         plot.out()
     );
     ELISE_COPY
     (
         rectangle(-20,40),
         FX,
         plot.out()
     );
}


       //=======
       //  2 -2 
       //=======


void TEST_plot_op_flx0 (Plot_1d plot,Line_St)
{
     ELISE_COPY
     (
            rectangle(-50,-40)
         || rectangle(-20,-5)
         || rectangle(5,13)
         || rectangle(37,42),
         4*cos(FX/2.0)+ 3+ (FX/5.0) * sin(FX/4.9),
         plot.out()
     );
}

void TEST_plot_op_flx1 (Plot_1d plot,Line_St)
{
     ELISE_COPY
     (
         select(plot.all_pts(),(FX%2) || (FX >20)),
         4*cos(FX/2.0)+ 3+ (FX/5.0) * sin(FX/4.9),
         plot.out()
     );
}

   //**************************
   //  3  -VARIATION ON OUTPUT
   //**************************

       //=======
       //  3 -1 
       //=======


void TEST_plot_out_image (Plot_1d plot,Line_St)
{
     Im1D_REAL4 I (50);

     ELISE_COPY 
     ( 
         I.all_pts(), 
         (FX%10) * 5,
         I.out()
     );

     ELISE_COPY 
     ( 
         I.all_pts(), 
         I.in(),
         plot.out()
     );
}

       //=======
       //  3 -2 
       //=======


void TEST_plot_oper_out_0(Plot_1d plot,Line_St lst)
{
     Im1D_REAL4 I (50);

     ELISE_COPY 
     ( 
         I.all_pts(), 
         cos(FX/2.0)*30,
         I.out() |  plot.out()
     );

     plot.set(NewlArgPl1d(PlModePl(Plots::line)));
     plot.set(NewlArgPl1d(PlotLinSty(lst)));

     ELISE_COPY 
     ( 
         I.all_pts(), 
         -I.in(),
         plot.out()
     );
}

void TEST_plot_oper_out_1(Plot_1d plot,Line_St)
{
     Im1D_REAL4 I (50);

     ELISE_COPY 
     ( 
         I.all_pts(), 
         Square(FX)/50.0,
           I.out()
        |  plot.out()
        |  (plot.out().chc(FX-50))
        |  (plot.out() << (-I.in()))
     );
}




void PS(const char * name,ACT action)
{
      // sz of images we will use

         Pt2di SZ(512,512);


     //  palette allocation
         Disc_Pal  Pdisc = Disc_Pal::P8COL();
         RGB_Pal   Prgb(255,255,255);

         Elise_Set_Of_Palette SOP(NewLElPal(Pdisc)+Elise_Palette(Prgb));

     // Creation of postscript windows

           char  buf[200];
           sprintf(buf,"DOC/PS/%s.eps",name);
    
           PS_Display disp(buf,"Mon beau fichier ps",SOP,false);

           PS_Window  Wps = disp.w_centered_max(SZ,Pt2dr(4.0,4.0));


     // define a window to draw simultaneously in 

         Plot_1d  Plot1  
                  (
                        Wps,
                        Line_St(Pdisc(P8COL::green),3),
                        Line_St(Pdisc(P8COL::black),3),
                        Interval(-50,50),
                           NewlArgPl1d(PlBox(Pt2dr(3,3),Pt2dr(SZ)-Pt2dr(3,3)))
                        + Arg_Opt_Plot1d(PlScaleY(1.0))
                        + Arg_Opt_Plot1d(PlBoxSty(Pdisc(P8COL::blue),3))
                        + Arg_Opt_Plot1d(PlClipY(true))
                        + Arg_Opt_Plot1d(PlModePl(Plots::fill_box))
                        + Arg_Opt_Plot1d(PlClearSty(Pdisc(P8COL::white)))
                        + Arg_Opt_Plot1d(PlotFilSty(Prgb(255,0,0)))
                   );


        action(Plot1,Line_St(Pdisc(P8COL::black),6));
        Plot1.set(NewlArgPl1d(PlModePl(Plots::draw_fill_box)));
        Plot1.show_axes();
        Plot1.show_box();

}



int  main(int,char **)
{
    PS("Int_TEST_plot_FX",TEST_plot_FX);
    PS("Int_TEST_plot_2",TEST_plot_2);
    PS("Int_TEST_plot_Im0",TEST_plot_Im0);

    PS("Int_TEST_plot_Im1",TEST_plot_Im1);
    PS("Int_TEST_plot_Im2",TEST_plot_Im2);


    PS("Int_TEST_plot_expr_1",TEST_plot_expr_1);
    PS("Int_TEST_plot_expr_2",TEST_plot_expr_2);

    PS("Int_TEST_plot_expr_Im0",TEST_plot_expr_Im0);
    PS("Int_TEST_plot_expr_Im1",TEST_plot_expr_Im1);



    PS("Int_TEST_plot_rects",TEST_plot_rects);
    PS("Int_TEST_plot_op_flx0",TEST_plot_op_flx0);
    PS("Int_TEST_plot_op_flx1",TEST_plot_op_flx1);


    PS("Int_TEST_plot_out_image",TEST_plot_out_image);
    PS("Int_TEST_plot_oper_out_0",TEST_plot_oper_out_0);
    PS("Int_TEST_plot_oper_out_1",TEST_plot_oper_out_1);
}





