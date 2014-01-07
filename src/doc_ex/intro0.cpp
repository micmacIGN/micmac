#include "StdAfx.h"

   //**** Fonctions Numerique

       // Fonc_Num primitives

void TEST_plot_FX (Plot_1d plot)
{
     ELISE_COPY(plot.all_pts(),FX, plot.out());
}

void TEST_plot_2 (Plot_1d plot)
{
     ELISE_COPY(plot.all_pts(),2,plot.out());
}

Im1D_REAL4 init_image(INT tx)
{
     Im1D_REAL4 I(tx);
     REAL4  * d = I.data();

     for (INT x =0; x < tx ; x++)
         d[x] = (x*x)/ (double) tx;
     return I;
}
void TEST_plot_Im0 (Plot_1d plot)
{
     Im1D_REAL4 I = init_image(40);
     ELISE_COPY(I.all_pts(),I.in(),plot.out());
}

       //  Operator sur Fonc_Num

void TEST_plot_expr_1 (Plot_1d plot)
{
     ELISE_COPY (plot.all_pts(),FX/2.0,plot.out());
}
void TEST_plot_expr_2 (Plot_1d plot)
{
     ELISE_COPY
     (
         plot.all_pts(),
         4*cos(FX/2.0)+ 3+ (FX/5.0) * sin(FX/4.9),
         plot.out()
     );
}
void TEST_plot_expr_Im0 (Plot_1d plot)
{
     Im1D_REAL4 I = init_image(40);
     ELISE_COPY(I.all_pts(),40-1.7*I.in(),plot.out());
}

void TEST_plot_expr_Im1 (Plot_1d plot)
{
     Im1D_REAL4 I = init_image(40);
     ELISE_COPY
     (
         plot.all_pts(),
         3.0*I.in()[Abs(FX)%20],
         plot.out()
     );
}

   //*** Flux_Pts

       //=== Flux_Pts primitifs

void TEST_plot_rects (Plot_1d plot)
{
     ELISE_COPY(rectangle(-50,-40),FX,plot.out());
     ELISE_COPY(rectangle(-20,40),FX,plot.out());
}

       //=== operateur sur Flux_Pts

void TEST_plot_op_flx0 (Plot_1d plot)
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

void TEST_plot_op_flx1 (Plot_1d plot)
{
     ELISE_COPY
     (
         select(plot.all_pts(),(FX%2) || (FX >20)),
         4*cos(FX/2.0)+ 3+ (FX/5.0) * sin(FX/4.9),
         plot.out()
     );
}

   //***   Output

       //== Output primitifs

void TEST_plot_out_image (Plot_1d plot)
{
     Im1D_REAL4 I (50);
     ELISE_COPY(I.all_pts(),(FX%10)*5,I.out());
     ELISE_COPY(I.all_pts(),I.in(),plot.out());
}

       //=== Operateur sur Output

void TEST_plot_oper_out_0(Plot_1d plot,Line_St lst)
{
     Im1D_REAL4 I (50);
     ELISE_COPY(I.all_pts(),cos(FX/2.0)*30,I.out()|plot.out());

     plot.set(NewlArgPl1d(PlModePl(Plots::line)));
     plot.set(NewlArgPl1d(PlotLinSty(lst)));

     ELISE_COPY 
     ( 
         I.all_pts(), 
         -I.in(),
         plot.out()
     );
}

void TEST_plot_oper_out_1(Plot_1d plot)
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

   //***   Output

void TEST_plot_Im0_Bug (Plot_1d plot)
{
     Im1D_REAL4 I = init_image(40);
     ELISE_COPY(plot.all_pts(),I.in(),plot.out());
}
void TEST_plot_Im1 (Plot_1d plot)
{
     Im1D_REAL4 I = init_image(40);
     ELISE_COPY(plot.all_pts(),I.in(4.5),plot.out());
}
void TEST_plot_Im2 (Plot_1d plot)
{
     Im1D_REAL4 I = init_image(40);
     ELISE_COPY(plot.all_pts(),I.in_proj(),plot.out());
}


int  DocEx_Intro0_main(int,char **)
{
      // sz of images we will use
         Pt2di SZ(512,512);


     //  palette allocation
         Disc_Pal  Pdisc = Disc_Pal::P8COL();
         Elise_Set_Of_Palette SOP(NewLElPal(Pdisc));

     // Creation of video windows
         Video_Display Ecr((char *) NULL);
         Ecr.load(SOP);
         Video_Win   Wv  (Ecr,SOP,Pt2di(50,50),Pt2di(SZ.x,SZ.y));


     // define a window to draw simultaneously in 

        Plot_1d  Plot1
                  (
                        Wv,
                        Line_St(Pdisc(P8COL::green),3),
                        Line_St(Pdisc(P8COL::black),2),
                        Interval(-50,50),
                           NewlArgPl1d(PlBox(Pt2dr(3,3),Pt2dr(SZ)-Pt2dr(3,3)))
                        + Arg_Opt_Plot1d(PlScaleY(1.0))
                        + Arg_Opt_Plot1d(PlBoxSty(Pdisc(P8COL::blue),3))
                        + Arg_Opt_Plot1d(PlClipY(true))
                        + Arg_Opt_Plot1d(PlModePl(Plots::draw_fill_box))
                        + Arg_Opt_Plot1d(PlClearSty(Pdisc(P8COL::white)))
                        + Arg_Opt_Plot1d(PlotFilSty(Pdisc(P8COL::red)))
                   );



     //  NOW TRY SOME PLOTS 
   
          // 1- VARIATION ON FUNCTIONS

               // 1-1 "primitives" functions

        Plot1.clear();
        TEST_plot_FX(Plot1);
        Plot1.show_axes();
        Plot1.show_box();
        getchar();

        Plot1.clear();
        TEST_plot_2(Plot1);
        Plot1.show_axes();
        Plot1.show_box();
        getchar();

        Plot1.clear();
        TEST_plot_Im0(Plot1);
        Plot1.show_axes();
        Plot1.show_box();
        getchar();


               // 1-2 Arithmetic operator + composition

        Plot1.clear();
        TEST_plot_expr_1(Plot1);
        Plot1.show_axes();
        Plot1.show_box();
        getchar();

        Plot1.clear();
        TEST_plot_expr_2(Plot1);
        Plot1.show_axes();
        Plot1.show_box();
        getchar();

        Plot1.clear();
        TEST_plot_expr_Im0(Plot1);
        Plot1.show_axes();
        Plot1.show_box();
        getchar();


        Plot1.clear();
        TEST_plot_expr_Im1(Plot1);
        Plot1.show_axes();
        Plot1.show_box();
        getchar();



          // 2- VARIATION ON FLUX

               // 2-1 "primitives" flux

        Plot1.clear();
        TEST_plot_rects(Plot1);
        Plot1.show_axes();
        Plot1.show_box();
        getchar();

               // 2-2 "operator on flux

        Plot1.clear();
        TEST_plot_op_flx0(Plot1);
        Plot1.show_axes();
        Plot1.show_box();
        getchar();


        Plot1.clear();
        TEST_plot_op_flx1(Plot1);
        Plot1.show_axes();
        Plot1.show_box();
        getchar();

          // 3- VARIATION ON FLUX

               // 3-1 "primitives" output

        Plot1.clear();
        TEST_plot_out_image(Plot1);
        Plot1.show_axes();
        Plot1.show_box();
        getchar();

               // 3-2 "primitives" output

        Plot1.clear();
        TEST_plot_oper_out_0
        (
            Plot1, 
            Line_St(Pdisc(P8COL::magenta),3)
        );
        Plot1.set(NewlArgPl1d(PlotLinSty(Pdisc(P8COL::black),2)));
        Plot1.set(NewlArgPl1d(PlModePl(Plots::draw_fill_box)));
        Plot1.show_axes();
        Plot1.show_box();
        getchar();

        Plot1.clear();
        TEST_plot_oper_out_1(Plot1);
        Plot1.show_axes();
        Plot1.show_box();
        getchar();


        // 4 ERROR/ handling over
/*
        Plot1.clear();
        TEST_plot_Im0_Bug(Plot1);
        Plot1.show_axes();
        Plot1.show_box();
        getchar();
*/

        Plot1.clear();
        TEST_plot_Im1(Plot1);
        Plot1.show_axes();
        Plot1.show_box();
        getchar();

        Plot1.clear();
        TEST_plot_Im2(Plot1);
        Plot1.show_axes();
        Plot1.show_box();
        getchar();


       return 0;
}


