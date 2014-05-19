/*eLiSe06/05/99
  
     Copyright (C) 1999 Marc PIERROT DESEILLIGNY

   eLiSe : Elements of a Linux Image Software Environment

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  Author: Marc PIERROT DESEILLIGNY    IGN/MATIS  
Internet: Marc.Pierrot-Deseilligny@ign.fr
   Phone: (33) 01 43 98 81 28
eLiSe06/05/99*/



#include "StdAfx.h"
#include "bench.h"




template <class Type>
void bench_som_rel_flag
     (
           Flux_Pts flx,
           Fonc_Num fonc,
           Pt2di sz,
           Type  *
     )
{
    Neighbourhood v8 = Neighbourhood::v8();
    Im2D_U_INT1 flag(sz.x,sz.y);
    ELISE_COPY(flag.all_pts(),Min(255,frandr()*256),flag.out());


    //=========================================================
    // Bench sur un sel_flag simple 
    //=========================================================

    Fonc_Num sflag = 0;
	INT k;
    for (k =0; k<8 ; k++)
        sflag =    sflag
                 +   ((flag.in()&(1<<k)) !=0)
                   * trans(fonc,TAB_8_NEIGH[k]);

    Type dif;
    ELISE_COPY
    (
         flx,
         Abs
         (
             sel_flag(v8,flag.in()).red_sum(fonc)
            - sflag
         ),
         VMax(dif)
    );
    BENCH_ASSERT(dif<epsilon);


    Im2D_U_INT1 sel(sz.x,sz.y);
    ELISE_COPY(sel.all_pts(),frandr()<0.5,sel.out());


    //=========================================================
    // Bench sur un sel_flag compose avec un sel fonc 
    //=========================================================

    Fonc_Num som_sel_flag = 0;

    for (k =0; k<8 ; k++)
        som_sel_flag =   som_sel_flag
                       +   ((flag.in()&(1<<k)) !=0)
                         * trans(fonc*(sel.in()!=0),TAB_8_NEIGH[k]);

    ELISE_COPY
    (
         flx,
         Abs
         (
              sel_func(sel_flag(v8,flag.in()),sel.in()).red_sum(fonc)
            - som_sel_flag
         ),
         VMax(dif)
    );

    BENCH_ASSERT(dif<epsilon);

    //=========================================================
    // Bench sur un sel fonc  compose avec un sel_flag
    //=========================================================

    ELISE_COPY
    (
         flx,
         Abs
         (
              sel_flag(sel_func(v8,sel.in()),flag.in()).red_sum(fonc)
            - som_sel_flag
         ),
         VMax(dif)
    );

    BENCH_ASSERT(dif<epsilon);
}



void bench_som_rel_flag()
{
     bench_som_rel_flag
     (
          //disc(Pt2di(30,30),25),
          disc(Pt2dr(30,30),25), // __NEW
          FX+FY,
          Pt2di(60,60),
         (INT *) 0
     );

     bench_som_rel_flag
     (
          //select(disc(Pt2di(30,30),25),(FX+FY)%2),
          select(disc(Pt2dr(30,30),25),(FX+FY)%2), // __NEW
          FX+FY,
          Pt2di(60,60),
         (INT *) 0
     );
}

      //============================================
      //   Bench de skeletisation                       
      //============================================


void bench_skel_opb
     (
         INT max_d,
         Pt2di     sz,
         REAL      fcan,
         REAL      salt_peper,
         REAL      ang_thresh,
         INT       surf_thresh,
         bool      sk_of_disc,
         bool      prolgt_sk
     )
{
     Im2D_U_INT1 I(sz.x,sz.y);
     Im2D_U_INT1 IVein(sz.x,sz.y);
     Im2D_U_INT2 ISurf(sz.x,sz.y);

     ELISE_COPY
     (
          I.all_pts(),
          (     canny_exp_filt(frandr(),fcan,fcan,20)
              /  canny_exp_filt(1.0,fcan,fcan,20)
          )  > 0.5,
          I.out()  
     );

     ELISE_COPY
     (
          select(I.all_pts(),frandr()<salt_peper),
          frandr() < 0.5,
          I.out()  
     );

     ELISE_COPY
     (
          select
          (
              I.all_pts(),
                (((FX+max_d/2)%max_d) == 0)
             || (((FY+max_d/2)%max_d) == 0)
          ),
          0,
          I.out()  
     );

     ELISE_COPY(I.border(2),0,I.out());

     L_ArgSkeleton larg  =
               NewLArgSkel(AngSkel(ang_thresh))
             + ArgSkeleton(SurfSkel(surf_thresh))
             + ArgSkeleton(SkelOfDisk(sk_of_disc))
             + ArgSkeleton(ProlgtSkel(prolgt_sk))
             + ArgSkeleton(ResultSkel(true));

     Skeleton (IVein,I,larg+ArgSkeleton(TmpSkel(ISurf)));

     INT dif;

     ELISE_COPY
     (
         I.all_pts(),
         skeleton(I.in(0),max_d,larg) != IVein.in(0),
         sigma(dif)
     );

     BENCH_ASSERT(dif == 0);

     ELISE_COPY
     (
         rectangle((sz*3)/10,(sz*7)/10),
         skeleton(I.in(0),max_d,larg) != IVein.in(0),
         sigma(dif)
     );

     BENCH_ASSERT(dif == 0);


     INT dif_SKD[2];

     ELISE_COPY
     (
         I.all_pts(),
         Abs
         (
            Virgule(skeleton(I.in(0),max_d,larg),extinc_32(I.in(0),max_d))
          - skeleton_and_dist(I.in(0),max_d,larg)
         ),
         VMax(dif_SKD,2)
     );

     BENCH_ASSERT
     (
            (dif_SKD[0] == 0)
         && (dif_SKD[1] == 0)
     );
}


void bench_skel
     (
         INT       max_d,
         Pt2di     sz,
         REAL      fcan,
         REAL      salt_peper,
         REAL      ang_thresh,
         INT       surf_thresh,
         bool      sk_of_disc,
         bool      prolgt_sk
     )
{
     bench_skel_opb
     (
          max_d,
          sz,
          fcan,
          salt_peper,
          ang_thresh,
          surf_thresh,
          sk_of_disc,
          prolgt_sk
     );

     Im2D_U_INT1 I(sz.x,sz.y);

     ELISE_COPY
     (
          I.all_pts(),
          (     canny_exp_filt(frandr(),fcan,fcan,20)
              /  canny_exp_filt(1.0,fcan,fcan,20)
          )  > 0.5,
          I.out() 
     );

     ELISE_COPY
     (
          select(I.all_pts(),frandr()<salt_peper),
          frandr() < 0.5,
          I.out() 
     );

     ELISE_COPY(I.border(2),0,I.out());

     Im2D_U_INT1 IVein(sz.x,sz.y);
     Liste_Pts_U_INT2 l =
        Skeleton
        (
             IVein,
             I,
               NewLArgSkel(AngSkel(ang_thresh))
             + ArgSkeleton(SurfSkel(surf_thresh))
             + ArgSkeleton(SkelOfDisk(sk_of_disc))
             + ArgSkeleton(ProlgtSkel(prolgt_sk))
             + ArgSkeleton(ResultSkel(true))
        );


     INT Im_nb_cc_8 =0;
     INT Im_nb_trou_4 =0;
     Neighbourhood v4 = Neighbourhood::v4();
     Neighbourhood v8 = Neighbourhood::v8();

     ELISE_COPY(I.all_pts(),I.in()!=0,I.out());
     ELISE_COPY(I.border(1),2,I.out());

     U_INT1 ** i = I.data();

     {
         for (int x=0; x<sz.x ; x++)
             for (int y=0; y<sz.y ; y++)
                 if (i[y][x] < 2)
                 {
                     Neighbourhood v = Neighbourhood::v8();
                     if (i[y][x]) 
                     {
                         Im_nb_cc_8 ++;
                         v = v8;
                     }
                     else
                     {
                          Im_nb_trou_4 ++;
                         v = v4;
                     }
    
                     ELISE_COPY
                     (
                          conc(Pt2di(x,y),sel_func(v,I.in()== i[y][x])),
                          i[y][x]+2,
                          I.out()  
                     );
                 }
     }

     Im1D_U_INT1 nbb = NbBits(8);

     INT Sk_nb_cc_8 = l.card();
     INT Sk_nb_trou_4 =1;

     U_INT1 ** v = IVein.data();

     {
        for (INT x=0; x<sz.x ; x++)
        {
             for (int y=0; y<sz.y ; y++)
             {
                 if ((i[y][x] != 0) && (v[y][x] != 0))
                 {

                     INT nbs,nba;

                     Neigh_Rel r = 
                           (x%2)                                     ?
                           sel_flag(sel_func(v8,I.in()),IVein.in())  :
                           sel_func(sel_flag(v8,IVein.in()),I.in())  ;

                     ELISE_COPY
                     (
                        conc(Pt2di(x,y),r),
                        0,
                               I.out()
                        | (sigma(nbs)<<1)
                        | (sigma(nba)<<nbb.in()[IVein.in()])
                     );

                     Sk_nb_cc_8++;
                     Sk_nb_trou_4 += (nba/2-nbs+1);
                 }
             }
        }
     }

     BENCH_ASSERT
     (
               (Im_nb_cc_8 == Sk_nb_cc_8)
            && (Im_nb_trou_4 == Sk_nb_trou_4)
     );
}


void bench_sym_flag (Pt2di sz)
{

     Im2D_U_INT1 Ifl(sz.x,sz.y);

     Im2D_U_INT1 Ifls1(sz.x,sz.y);
     Im2D_U_INT1 Ifls2(sz.x,sz.y,0);

     ELISE_COPY
     (
         Ifl.all_pts(),
         255*frandr(),
         Ifl.out()
     );
     ELISE_COPY(Ifl.border(2),0,Ifl.out());
     ELISE_COPY(Ifl.all_pts(),nflag_sym(Ifl.in(0)),Ifls1.out());

     for (INT k=0; k<8; k++)
         ELISE_COPY
         (
             select
             (
                  Ifl.interior(1),
                  trans(Ifl.in()&(1<<((k+4)%8)),TAB_8_NEIGH[k])
             ),
             (Ifls2.in() | (1<<k)),
             Ifls2.out()
         );

     INT nb_dif;
     ELISE_COPY
     (
         Ifls1.all_pts(),
         Ifls1.in()!=Ifls2.in(),
         sigma(nb_dif)
     );
     BENCH_ASSERT(nb_dif==0);

     ELISE_COPY
     (
         Ifls1.all_pts(),
         nflag_close_sym(Ifl.in(0))!=(Ifls1.in()|Ifl.in()),
         sigma(nb_dif)
     );
     BENCH_ASSERT(nb_dif==0);

     ELISE_COPY
     (
         Ifls1.all_pts(),
         nflag_open_sym(Ifl.in(0))!=(Ifls1.in()&Ifl.in()),
         sigma(nb_dif)
     );
     BENCH_ASSERT(nb_dif==0);
}




class Bench_Act_Vect : public Br_Vect_Action
{
    public :
       Bench_Act_Vect 
       (
           Im2D_U_INT1   init,
           Im2D_U_INT1   verif
       ) : 
           _init     (init),
           _verif    (verif)
       {}

       ~Bench_Act_Vect () 
       {
	       RC_Object * o = this;
	        cout << "~~ Bench_Act_Vec  " << (void * ) o << "\n";
       }

    private  :
       void action(const ElFifo<Pt2di> &, const ElFifo<INT>   *,INT);

       Im2D_U_INT1   _init;
       Im2D_U_INT1   _verif;
};

void Bench_Act_Vect::action
     (
          const ElFifo<Pt2di> & pts,
          const ElFifo<INT>   * attrs,
          INT                  nb_attrs
     )
{

  BENCH_ASSERT(nb_attrs==2);
  BENCH_ASSERT(pts.nb()==attrs[0].nb());
  BENCH_ASSERT(pts.nb()==attrs[1].nb());

  {
      U_INT1 ** i = _init.data();
      U_INT1 ** v = _verif.data();

      for (INT k = 0; k<pts.nb() ; k++)
      {
          bool last = (k == pts.nb()-1);
          Pt2di p = pts[k];
          BENCH_ASSERT
          (
               (attrs[0][k] == (p.x+p.y))
            && (attrs[1][k] == (p.x-p.y))
          );
          INT nbb = SkVein::NbBitFlag[i[p.y][p.x]];
          if (pts.circ())
          {
             BENCH_ASSERT(nbb==2);
          }
          else
          {
             if ((k==0) || last)
             {
                BENCH_ASSERT((nbb!=0) && (nbb!=2));
             }
             else
             {
                BENCH_ASSERT(nbb==2);
             }
          }

          if (pts.circ() || (! last))
          {
             Pt2di q = pts[k+1];
             INT kp = freeman_code(q-p);

             BENCH_ASSERT(kp>=0);
             INT kq = (kp+4)%8;

             BENCH_ASSERT
             (
                     (((i[p.y][p.x]) & (1<<kp)) != 0)
                  && (((i[q.y][q.x]) & (1<<kq)) != 0)
                  && (((v[p.y][p.x]) & (1<<kp)) == 0)
                  && (((v[q.y][q.x]) & (1<<kq)) == 0)
             );

             v[p.y][p.x] |= 1<<kp;
             v[q.y][q.x] |= 1<<kq;
          }
      }
  }
};

void bench_vecto 
     (
          Im2D_U_INT1 Ifl
     )
{


    Pt2di sz (Ifl.tx(),Ifl.ty());
    Im2D_U_INT1 Iverif(sz.x,sz.y,0);

   ELISE_COPY
   (
      Ifl.all_pts(),
      sk_vect
      (
         Fonc_Num(11,22,33), // FX+FY,FX-FY),
         new Bench_Act_Vect (Ifl,Iverif)
      ),
      Output::onul()
   );



   ELISE_COPY
   (
      Ifl.all_pts(),
      sk_vect
      (
         Virgule(Ifl.in(0),FX+FY,FX-FY),
         new Bench_Act_Vect (Ifl,Iverif)
      ),
      Output::onul()
   );


   INT nb_dif;
   ELISE_COPY
   (
       Ifl.all_pts(),
       (Ifl.in() != Iverif.in()),
       sigma(nb_dif)
   );
   BENCH_ASSERT(nb_dif==0);

}


void bench_vecto 
     (
          Pt2di       sz
     )
{

    {
cout << "bench_vecto 0\n";
         Im2D_U_INT1 I0(sz.x,sz.y,255);
         ELISE_COPY
         (
             I0.all_pts(),
             unif_noise_4(2) < 0.3,
             I0.out() 
         );
cout << "bench_vecto 1\n";

         Im2D_U_INT1 Ifl(sz.x,sz.y,255);

cout << "bench_vecto 2\n";
         Liste_Pts_U_INT2 l = 
            Skeleton
            (
                Ifl,I0,
                   NewLArgSkel(AngSkel(300.0))
                + ArgSkeleton(SurfSkel(5000))
                + ArgSkeleton(ResultSkel(true))
            );

         ELISE_COPY(l.all_pts(),0,I0.out());

cout << "bench_vecto 3\n";
         Skeleton
         (
                Ifl,I0,
                   NewLArgSkel(AngSkel(300.0))
                +  ArgSkeleton(SurfSkel(5000))
         );
         bench_vecto(Ifl);

cout << "bench_vecto 4\n";
       //*************

         ELISE_COPY
         (
             I0.all_pts(),
             unif_noise_4(2) < 0.3,
             I0.out() 
         );
         Skeleton ( Ifl,I0, NewLArgSkel(AngSkel(3.0)) + ArgSkeleton(SurfSkel(7)));
         ELISE_COPY(l.all_pts(),0,I0.out());
         bench_vecto(Ifl);
    }
cout << "bench_vecto 5\n";

    {
       Im2D_U_INT1 Ifl(sz.x,sz.y,255);


       for (INT nb =0; nb < 8 ; nb++)
       {
          if (nb < 4)
             ELISE_COPY (Ifl.all_pts(),(255*frandr()) & Ifl.in(),Ifl.out());
          else
             ELISE_COPY (Ifl.all_pts(),(255*frandr()) | Ifl.in(),Ifl.out());

          ELISE_COPY(Ifl.border(1),0,Ifl.out());
          ELISE_COPY(Ifl.all_pts(),nflag_open_sym(Ifl.in(0)),Ifl.out());


          bench_vecto(Ifl);
       }
    }
cout << "bench_vecto 6\n";
}


void bench_skel_clip(Pt2di sz1,Pt2di sz2)
{
     Im2D_U_INT1 Id1(sz1.x,sz1.y);
     Im2D_U_INT1 Iskel1(sz1.x,sz1.y);

     ELISE_COPY(Id1.all_pts(),gauss_noise_1(3) > 0.5,Id1.out());


     Im2D_U_INT1 Id2(sz2.x,sz2.y);
     Im2D_U_INT1 Iskel2(sz2.x,sz2.y);

     Im2D_U_INT1 IdVerif(sz2.x,sz2.y);
     Im2D_U_INT1 ISkelVerif(sz2.x,sz2.y);


     ELISE_COPY(Id2.all_pts(),frandr() > 0.5,Id2.out()    | IdVerif.out()   );
     ELISE_COPY(Id2.all_pts(),frandr() > 0.5,Iskel2.out() | ISkelVerif.out());

     ELISE_COPY(Id1.all_pts(),Id1.in(),Id2.out());

     Skeleton(Iskel1,Id1, L_ArgSkeleton());
     Skeleton(Iskel2,Id2, L_ArgSkeleton(),sz1);

     INT difD,difSk,NB;

     ELISE_COPY
     (
           Iskel1.all_pts(),
           Virgule(
               Abs(Iskel1.in()-Iskel2.in()),
               Abs(Id1.in()-   Id2.in())
           ),
           Virgule(VMax(difD),VMax(difSk))
     );


     BENCH_ASSERT((difD==0) && (difSk==0));

     ELISE_COPY
     (
           select(Iskel2.all_pts(),! inside(Pt2di(0,0),sz1)),
           Virgule(
               Abs(ISkelVerif.in()-Iskel2.in()),
               Abs(IdVerif.in()-   Id2.in()),
               1
           ),
           Virgule(VMax(difD),VMax(difSk),sigma(NB))
     );

     BENCH_ASSERT
     (
           (difD==0) 
        && (difSk==0) 
        && (NB == (sz2.x*sz2.y -sz1.x*sz1.y))
     );
}

void bench_skel_clip()
{
     bench_skel_clip(Pt2di(140,150),Pt2di(160,155));
     bench_skel_clip(Pt2di(150,140),Pt2di(180,175));

     bench_skel_clip(Pt2di(140,150),Pt2di(160,195));
     bench_skel_clip(Pt2di(150,140),Pt2di(180,185));
}

void bench_skel()
{
	cout << "bench skel 0 \n";
     bench_vecto(Pt2di(50,40));
	cout << "bench skel 1 \n";
     bench_vecto(Pt2di(240,250));
	cout << "bench skel 2 \n";

     int SZX =  250;
     int SZY =  240;
     

      
     bench_skel(30,Pt2di(SZX,SZY),0.6,0.02,3.0,5,true,true);
cout << "bench skel 3 \n";
     bench_skel(15,Pt2di(SZX,SZY),0.6,0.02,-1.0,-1,false,false);
cout << "bench skel 4 \n";
     bench_skel(20,Pt2di(SZX,SZY),0.6,0.02,3.0,5,false,false);
cout << "bench skel 5 \n";
     bench_skel(10,Pt2di(SZX,SZY),0.001,1.00,3.0,5,false,false);
cout << "bench skel 6 \n";
     bench_sym_flag (Pt2di(SZX,SZY));
cout << "bench skel 7 \n";
}



void bench_skel_dout()
{
    Pt2di sz(30,40);

    Im2D_U_INT1 I1(sz.x,sz.y);
    Im2D_U_INT1 I2(sz.x,sz.y);

    ELISE_COPY
    (
          I1.all_pts(),
          Virgule(frandr()<0.5,frandr()<0.5),
          Virgule(I1.out() , I2.out())
    );

    INT dif1,dif2;
    ELISE_COPY
    (
          I1.all_pts(),
          Abs
          (
              Virgule(skeleton(I1.in(0),20),skeleton(I2.in(0),20))
             -skeleton(Virgule(I1.in(0),I2.in(0)),20)
          ),
          Virgule(VMax(dif1),VMax(dif2))
    );
    BENCH_ASSERT
    (
            (dif1==0)
         && (dif2==0)
    );
}

void bench_new_skel()
{
    Pt2di sz (150,150);
    INT ZOOM = 5;

    Disc_Pal       Pdisc  = Disc_Pal::P8COL();

    Elise_Set_Of_Palette SOP(NewLElPal(Pdisc));
    Video_Display Ecr((char *) NULL);
    Ecr.load(SOP);                         

    Video_Win   W  (Ecr,SOP,Pt2di(50,50),sz*ZOOM);  
    W = W.chc(Pt2dr(0,0),Pt2dr(ZOOM,ZOOM));

    Im2D_U_INT1 I(sz.x,sz.y);
    ELISE_COPY
    (
       I.all_pts(),
       gauss_noise_4(1)> 0.0,
       I.out() | W.odisc()
    );

    ELISE_COPY
    (
       I.all_pts(),
       skeleton
       (
          I.in(0),
          100,
          NewLArgSkel(ProlgtSkel(true))
       ),
       W.out_graph(Pdisc(P8COL::red))
    );
}

class BugAV : public Br_Vect_Action
{
    public :
       BugAV 
       (
           Im2D_U_INT1   ,
           Im2D_U_INT1   
       )  
       {}

       ~BugAV () 
       {
	       RC_Object * o = this;
	        cout << "~~ Bench_Act_Vec  " << (void * ) o << "\n";
       }

    private  :
       void action(const ElFifo<Pt2di> &, const ElFifo<INT>   *,INT){}

};


void bench_rel_flag()
{
	cout << "Enter   bench_ske \n";
   bench_skel();
	cout << "Out   bench_ske \n";
   bench_skel_clip();
   bench_skel_dout();
   bench_som_rel_flag();
   printf("OK SKEL \n");
}



















