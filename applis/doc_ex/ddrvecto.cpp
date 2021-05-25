#include "general/all.h"
#include "ext_stl/fifo.h"


void test_skel
     (
         Tiff_Im        I0,
         Video_Win      W,
         L_ArgSkeleton  larg
     )
{
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

     ELISE_COPY(l.all_pts(),P8COL::blue,W.odisc());

     ELISE_COPY
     (
          Iskel.all_pts(),
          Iskel.in(), 
          W.out_graph(Line_St(W.pdisc()(P8COL::black),2))
    );

    getchar();
}


class Test_Chainage :  public  Br_Vect_Action
{

       private :


           virtual void action
                   (
                          const ElFifo<Pt2di> & pts,
                          const ElFifo<INT>   *,
                          INT               
                    )
           {
                INT nb = pts.nb();

                for (INT k = 0;  k< nb + pts.circ()-1; k++)
                   _W.draw_seg(pts[k],pts[k+1],Line_St(_c4,2));

                if ( pts.circ())
                {
                    for (INT k = 0;  k< nb; k++)
                     _W.draw_circle_loc(pts[k],0.5,_c3);
                }
                else
                {
                    for (INT k = 0;  k< nb; k++)
                     _W.draw_circle_loc(pts[k],0.5,_c1);
                   _W.draw_circle_loc(pts[0]    ,0.5,_c2);
                   _W.draw_circle_loc(pts[nb-1] ,0.5,_c2);
                }

           }

           Video_Win  _W;
           Col_Pal    _c1;
           Col_Pal    _c2;
           Col_Pal    _c3;
           Col_Pal    _c4;

       public :
           Test_Chainage
           (
                  Video_Win W,
                  Col_Pal c1,
                  Col_Pal c2,
                  Col_Pal c3,
                  Col_Pal c4
           ) : _W (W)  , _c1 (c1) , _c2(c2) , _c3(c3),_c4(c4)
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


                for (INT k = 0;  k< nb; k++)
                {
                    _W.draw_circle_loc
                    (
                        pts[k],
                        attr[1][k]/10.0,
                        _pal(attr[0][k])
                    );
                }
           }

           Video_Win  _W;
           Disc_Pal   _pal;

       public :
           Test_Attr_Chainage
           (
                  Video_Win W,
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
                ElFifo<INT> app;
                approx_poly(app,pts,_arg);
                INT nb = app.nb();

                for (INT k = 0;  k< nb-1; k++)
                    _W.draw_seg(pts[app[k]],pts[app[k+1]],_c1);
                for (INT k = 0;  k< nb; k++)
                    _W.draw_circle_loc(pts[app[k]],0.5,_c2);
           }

           Video_Win  _W;
           Col_Pal    _c1;
           Col_Pal    _c2;
           ArgAPP     _arg;

       public :

           Test_Approx
           (
                  Video_Win W,
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
       INT ZOOM = 8;
       Tiff_Im  FeLiSe("../DOC_ELISE/eLiSe.tif");

    // sz of images we will use
        Pt2di SZ = FeLiSe.sz();

   //  palette allocation
        Disc_Pal  Pdisc = Disc_Pal::P8COL();
        Elise_Set_Of_Palette SOP(NewLElPal(Pdisc));

   // Creation of video windows
        Video_Display Ecr((char *) NULL);
        Ecr.load(SOP);
        Video_Win   W (Ecr,SOP,Pt2di(50,50),SZ*ZOOM);
        W =    W.chc(Pt2dr(-0.5,-0.5),Pt2dr(ZOOM,ZOOM));
        W.set_title("eLiSe dans une fenetre ELISE");

#if (0)

        test_skel
        (
            FeLiSe,
            W,
               L_ArgSkeleton()
        );

        test_skel
        (
            FeLiSe,
            W,
               L_ArgSkeleton()
           +   SurfSkel(10)
           +   AngSkel(4.2)
        );

        test_skel
        (
            FeLiSe,
            W,
               L_ArgSkeleton()
           +   SurfSkel(3)
           +   AngSkel(2.2)
        );

        test_skel
        (
            FeLiSe,
            W,
               L_ArgSkeleton()
            +  ProlgtSkel(true)
        );

        test_skel
        (
            FeLiSe,
            W,
               L_ArgSkeleton()
            +  Cx8Skel(false)
            +  ProlgtSkel(true)
        );

        test_skel
        (
            FeLiSe,
            W,
               L_ArgSkeleton()
            +  Cx8Skel(false)
            +  ProlgtSkel(true)
            +  SkelOfDisk(true)
        );

        test_skel
        (
            FeLiSe,
            W,
               L_ArgSkeleton()
           +   SurfSkel(10)
           +   AngSkel(4.2)
           +   ProlgtSkel(true)
           +   ResultSkel(true)
        );


   for (INT i = 0; i< 2 ; i++)
   {
       ELISE_COPY
       (
            FeLiSe.all_pts(),
            FeLiSe.in(),
            W.odisc()
       );
       Line_St lst(Pdisc(P8COL::red),2);
       ELISE_COPY
       (
            FeLiSe.all_pts(),
            skeleton
            (
                  FeLiSe.in(0),
                  30,
                  newl(SurfSkel(8))
              +   ProlgtSkel(true)
              +   Cx8Skel(i==1)
            ),
            W.out_graph(lst)
       );
       getchar();
  }
#endif


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
         new Test_Chainage
         (
             W,
             Pdisc(P8COL::yellow),
             Pdisc(P8COL::blue),
             Pdisc(P8COL::green),
             Pdisc(P8COL::red)
         )
     ),
     Output::onul()
  );
  getchar();



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
         (
            skeleton(! FeLiSe.in(1)),
            FX%2 +2,
            (10+FY)/7
         ),
         new Test_Attr_Chainage (W,Pdisc)
     ),
     Output::onul()
  );
  getchar();

 
  REAL prec[3] = {1.0,3.0,10.0};
  for (INT k = 0; k < 3 ; k++)
  {

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
                 Pdisc(P8COL::blue),
                 Pdisc(P8COL::red),
                 prec[k]
             )
         ),
         Output::onul()
      );
      getchar();
  }

  
}







