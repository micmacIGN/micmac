#include "general/all.h"
#include "ext_stl/fifo.h"

PS_Window PS(const char * name,  Pt2di SZ)
{
      // sz of images we will use


     //  palette allocation

         Disc_Pal  Pdisc = Disc_Pal::P8COL();



         Elise_Set_Of_Palette SOP(NewLElPal(Pdisc));

     // Creation of postscript windows

           char  buf[200];
           sprintf(buf,"DOC/PS/%s.eps",name);

           PS_Display disp(buf,"Mon beau fichier ps",SOP,false);

           return  disp.w_centered_max(SZ,Pt2dr(4.0,4.0));
}


void ps_skel
     (
         Im2D_U_INT1 Iskel,
         PS_Window   W,
         Line_St     lst
     )
{
     U_INT1 ** sk = Iskel.data();

     for (INT x= 0; x<Iskel.tx(); x++)
         for (INT y= 0; y<Iskel.ty(); y++)
             for (INT k=0 ; k<4 ; k++)
                 if (sk[y][x] & (1 << k))
                    W.draw_seg
                    (
                       Pt2dr(x,y) + Pt2dr(0.5,0.5),
                       Pt2dr(x,y) + TAB_8_NEIGH[k] + Pt2dr(0.5,0.5),
                       lst
                    );
}



void test_skel
     (
         Tiff_Im        I0,
         char *         name,
         L_ArgSkeleton  larg
     )
{
     PS_Window  W = PS(name,I0.sz());
     INT tx = I0.sz().x;
     INT ty = I0.sz().y;

     Im2D_U_INT1 Iskel(tx,ty);
     Im2D_U_INT1 ImIn (tx,ty);

     ELISE_COPY
     (
          I0.all_pts(),
          ! I0.in(),
             ImIn.out()
          |  (W.odisc() << (P8COL::yellow * (ImIn.in()==0)))
     );


     Liste_Pts_U_INT2 l = Skeleton(Iskel,ImIn,larg);

     Line_St   lst (W.pdisc()(P8COL::black),2);
     ELISE_COPY(l.all_pts(),P8COL::blue,W.odisc());
     ps_skel(Iskel,W,lst);
/*
     for (INT x= 0; x<tx; x++)
         for (INT y= 0; y<ty; y++)
             for (INT k=0 ; k<4 ; k++)
                 if (sk[y][x] & (1 << k))
                    W.draw_seg
                    (
                       Pt2dr(x,y) + Pt2dr(0.5,0.5),
                       Pt2dr(x,y) + TAB_8_NEIGH[k] + Pt2dr(0.5,0.5),
                       lst
                    );

     ELISE_COPY
     (
          Iskel.all_pts(),
          Iskel.in(), 
          W.out_graph(Line_St(W.pdisc()(P8COL::red),2))
    );
*/
}


class DDR_Test_Action :  public  Br_Vect_Action
{

       private :


           virtual void action
                   (
                          const ElFifo<Pt2di> & pts,
                          const ElFifo<INT>   *,
                          INT
                    )
           {
                Pt2dr d(0.5,0.5);
                INT nb = pts.nb();

                for (INT k = 0;  k< nb + pts.circ()-1; k++)
                   _W.draw_seg(d+pts[k],d+pts[k+1],Line_St(_c4,2));   
                if ( pts.circ())
                {
                    for (INT k = 0;  k< nb; k++)
                     _W.draw_circle_loc(d+pts[k],0.4,Line_St(_c3,1));
                }
                else
                {
                    for (INT k = 0;  k< nb; k++)
                     _W.draw_circle_loc(d+pts[k],0.4,Line_St(_c1,1));
                   _W.draw_circle_loc(d+pts[0]    ,0.4,Line_St(_c2,1));
                   _W.draw_circle_loc(d+pts[nb-1] ,0.4,Line_St(_c2,1));
                }
           }

            PS_Window  _W;
           Col_Pal    _c1;
           Col_Pal    _c2;
           Col_Pal    _c3;
           Col_Pal    _c4;

       public :
           DDR_Test_Action
           (
                  PS_Window W,
                  Col_Pal c1,
                  Col_Pal c2,
                  Col_Pal c3,
                  Col_Pal c4
           ) : _W (W),_c1 (c1),_c2 (c2),_c3 (c3),_c4 (c4)
           {}
};                       


class Test_Attr_Chainage :  public  Br_Vect_Action
{

       private :


           virtual void action
                   (
                          const ElFifo<Pt2di> & pts,
                          const ElFifo<INT>   * attr,
                          INT
                    )
           {
                INT nb = pts.nb();
                Pt2dr d(0.5,0.5);

                for (INT k = 0;  k< nb; k++)
                {
                    _W.draw_circle_loc
                    (
                        d+pts[k],
                        attr[1][k]/10.0,
                        _pal(attr[0][k])
                    );
                }
           }

           PS_Window   _W;
           Disc_Pal   _pal;

       public :
           Test_Attr_Chainage
           (
                  PS_Window  W,
                  Disc_Pal pal
           ) : _W (W)  , _pal (pal)
           {}
};                   


class Test_Approx :  public  Br_Vect_Action
{

       private :


           virtual void action
                   (
                          const ElFifo<Pt2di> & pts,
                          const ElFifo<INT>   *,
                          INT
                    )
           {
                Pt2dr d(0.5,0.5);
                ElFifo<INT> app;
                approx_poly(app,pts,_arg);
                INT nb = app.nb();

                for (INT k = 0;  k< nb-1; k++)
                    _W.draw_seg(d+pts[app[k]],d+pts[app[k+1]],_c1);
                for (INT k = 0;  k< nb; k++)
                    _W.draw_circle_loc(d+pts[app[k]],0.5,_c2);
           }

           PS_Window  _W;
           Col_Pal    _c1;
           Col_Pal    _c2;
           ArgAPP     _arg;

       public :

           Test_Approx
           (
                  PS_Window W,
                  Col_Pal c1,
                  Col_Pal c2,
                  REAL    prec
           ) : _W (W)  ,
               _c1 (c1) ,
               _c2(c2),
               _arg (
                        prec,
                        20,
                        ArgAPP::D2_droite,
                        ArgAPP::Extre
                    )
          {}
};
                      

int  main(int,char **)
{
       Tiff_Im  FeLiSe("DOC/eLiSe.tif");
       Pt2di SZ = FeLiSe.sz();

   //  palette allocation


        test_skel
        (
            FeLiSe,
            "skel_1",
               L_ArgSkeleton()
        );

        test_skel
        (
            FeLiSe,
            "skel_2",
               L_ArgSkeleton()
           +   ArgSkeleton(SurfSkel(10))
           +   ArgSkeleton(AngSkel(4.2))
        );

        test_skel
        (
            FeLiSe,
            "skel_3",
               L_ArgSkeleton()
           +   ArgSkeleton(SurfSkel(3))
           +   ArgSkeleton(AngSkel(2.2))
        );

        test_skel
        (
            FeLiSe,
            "skel_4",
               L_ArgSkeleton()
            +  ArgSkeleton(ProlgtSkel(true))
        );

        test_skel
        (
            FeLiSe,
            "skel_5",
               L_ArgSkeleton()
            +  ArgSkeleton(Cx8Skel(false))
            +  ArgSkeleton(ProlgtSkel(true))
        );

        test_skel
        (
            FeLiSe,
            "skel_6",
               L_ArgSkeleton()
            +  ArgSkeleton(Cx8Skel(false))
            +  ArgSkeleton(ProlgtSkel(true))
            +  ArgSkeleton(SkelOfDisk(true))
        );

         test_skel
        (
            FeLiSe,
            "skel_7",
               L_ArgSkeleton()
           +   ArgSkeleton(SurfSkel(10))
           +   ArgSkeleton(AngSkel(4.2))
           +   ArgSkeleton(ProlgtSkel(true))
           +   ArgSkeleton(ResultSkel(true))
        );


     for (int i= 0; i<2; i++)
     {
        PS_Window  W = PS((i==0)?"Skel_Fnum4":"Skel_Fnum8",FeLiSe.sz());
        Line_St lst(W.pdisc()(P8COL::red),2);
        ELISE_COPY
        (
            FeLiSe.all_pts(),
            FeLiSe.in(),
            W.odisc()
        );
        Im2D_U_INT1 Iskel(SZ.x,SZ.y);
        ELISE_COPY
        (
            FeLiSe.all_pts(),
            skeleton
            (
                  FeLiSe.in(0),
                  0,
                  L_ArgSkeleton()
              +   ArgSkeleton(SurfSkel(8))
              +   ArgSkeleton(ProlgtSkel(true))
              +   ArgSkeleton(Cx8Skel(i==1))
            ),
            Iskel.out()
         );
         ps_skel(Iskel,W,lst);
    }

    {
        PS_Window  W = PS("Skel_Vecto_Circ",SZ);
        ELISE_COPY
        (
            FeLiSe.all_pts(),
            ! FeLiSe.in(),
            W.odisc()
        );
        ELISE_COPY
        (
           FeLiSe.all_pts(),
           sk_vect
           (
               skeleton(! FeLiSe.in(1)),
               new DDR_Test_Action
               (
                   W,
                   W.pdisc()(P8COL::yellow),
                   W.pdisc()(P8COL::blue),
                   W.pdisc()(P8COL::green),
                   W.pdisc()(P8COL::red)
               )
           ),
           Output::onul()
        );                
    }


    {
        PS_Window  W = PS("Skel_Vecto_Attr_Circ",SZ);
        ELISE_COPY
        (
            FeLiSe.all_pts(),
            ! FeLiSe.in(),
            W.odisc()
        );
        ELISE_COPY
        (
           FeLiSe.all_pts(),
           sk_vect
           (
              Virgule
              (
                 skeleton(! FeLiSe.in(1)),
                 FX%2 +2,
                 (10+FY)/7
              ),
              new Test_Attr_Chainage (W,W.pdisc())
           ),
           Output::onul()
        );                
    }

    {
        REAL prec[3] = {1.0,3.0,10.0};
        for (INT k = 0; k < 3 ; k++)
        {
            char buf[200];
            sprintf(buf,"Skel_Approx_%d",k);
            PS_Window  W = PS(buf,SZ);
            ELISE_COPY
            (
                FeLiSe.all_pts(),
                ! FeLiSe.in(),
                W.odisc()
            );
            ELISE_COPY
            (
               FeLiSe.all_pts(),
               sk_vect
               (
                   skeleton(! FeLiSe.in(1)),
                   new Test_Approx
                   (
                       W,
                       W.pdisc()(P8COL::blue),
                       W.pdisc()(P8COL::red),
                       prec[k]
                   )
               ),
               Output::onul()
            );
        }           
    }           
}






