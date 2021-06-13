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






ElList<Pt2di> lpts_square_L1(INT nb)
{
    return   NewLPt2di(Pt2di( nb,  0))
           +      Pt2di(  0, nb)
           +      Pt2di(-nb,  0)
           +      Pt2di(  0,-nb);
}




ElList<Pt2di> lpts_square_Linf(INT nb)
{
    nb /= 2;
    return   NewLPt2di(Pt2di( nb, nb))
           +      Pt2di(-nb, nb)
           +      Pt2di(-nb,-nb)
           +      Pt2di( nb,-nb);
}


ElList<Pt2di> lpts_zigzag(INT nb)
{
    ElList<Pt2di> l ;
    for (INT k=0; k <= 12; k++)
        l = l + Pt2di(k*nb,(k%2)*nb);
    return   l;
}


typedef enum
{
    LPTS_L1,
    LPTS_Linf,
    LPTS_zigag
} MODE_LPTS_BENCH_APP;


void bench_approx_poly
     (
         INT                      nb,
         MODE_LPTS_BENCH_APP    mode,
         bool                   circ
     )
{
     ElList<Pt2di> lsom;
     switch(mode)
     {
         case LPTS_L1 :  
              lsom = lpts_square_L1(nb);
         break;

         case LPTS_Linf :  
              lsom = lpts_square_Linf(nb);
         break;

         case LPTS_zigag :  
              lsom = lpts_zigzag(nb);
         break;

     };
     lsom = lsom.reverse();
     INT nb_som_init = lsom.card();

     ElFifo<Pt2di> fif_pts(10,circ);

     Liste_Pts_INT2 lpt(2);
     ELISE_COPY(line(lsom,circ),1,lpt);
     INT nb_pts_init = lpt.card();
     INT offset = circ ? (INT)(NRrandom3() * nb_pts_init) : 0;
     Im2D_INT2 il = lpt.image();

     INT  nb_pts  = il.tx();
     INT2 * x     = il.data()[0];
     INT2 * y     = il.data()[1];

     for (int k =0; k<nb_pts ; k++)
         fif_pts.pushlast(Pt2di(x[(k+offset)%nb_pts],y[(k+offset)%nb_pts]));

     // Aprox tres grossiere , jump de 2, step infini => 2 extre
     ElFifo<int> res(10);
     ArgAPP arg(1e20,2,ArgAPP::D2_droite,ArgAPP::Extre,false,1000);
     approx_poly(res,fif_pts,arg);

     if (! circ )
     {
         BENCH_ASSERT
         (
                  (res.nb() == 2)
              &&  (res[0]    == 0)
              &&  (res[1]    == (nb_pts_init-1))
         );

         // Aprox tres grossiere , jump de racine nb pts, step 2, => 2 extre
         INT rac = round_up(sqrt(double(nb_pts_init)));

         arg  = ArgAPP(1e20,rac,ArgAPP::D2_droite,ArgAPP::Extre,false,2);
         approx_poly(res,fif_pts,arg);
         BENCH_ASSERT
         (
                  (res.nb() == 2)
              &&  (res[0]    == 0)
              &&  (res[1]    == (nb_pts_init-1))
         );


         // Aprox tres fine , jump de jmp diviseur de nb_pts_init-&, step 1,
         // resul    "0 nb_som_init 2*nb_som_init  ....    "

         if ((nb_pts_init-1)%(nb_som_init-1) == 0)
         {
             INT jmp = nb_som_init-1;
             arg  = ArgAPP(1e20,jmp,ArgAPP::D2_droite,ArgAPP::Extre,false,1);
             approx_poly(res,fif_pts,arg);
             BENCH_ASSERT
             (
                  res.nb() == (nb_pts_init-1)/jmp + 1
             );
             for (INT  k = 0 ; k<res.nb() ; k++)
                 BENCH_ASSERT
                 (
                     res[k] == jmp * k
                 );
         }
     }


     // Approx tres fine, avec jump >= au seuil et step asse grand;
     // on droit retomber sur les extres 
     for (INT jmp = nb;  jmp < 3*nb; jmp += (1+nb/3))
     {
         arg  = ArgAPP(1e-2,jmp,ArgAPP::D2_droite,ArgAPP::Extre,false,1);
         approx_poly(res,fif_pts,arg);

         if (! circ)
         {
            BENCH_ASSERT(res.nb() == nb_som_init);
            for (INT  k = 0 ; k<res.nb() ; k++)
                BENCH_ASSERT(res[k] == nb * k);
         }
         else
         {
            BENCH_ASSERT( (res[0] +offset) % nb == 0);
            BENCH_ASSERT(res.nb() == nb_som_init +1);
            INT r0 = res[0];
            for (INT  k = 0 ; k<res.nb() ; k++)
                BENCH_ASSERT(res[k] == nb * k +r0);
         }
     }
}

void bench_approx_poly()
{
    for (INT f =0; f< 3; f++)
        for (int i = 0; i < 20 ; i++)
        {
           bench_approx_poly(i+6,LPTS_L1,true);
           bench_approx_poly(2*i+6,LPTS_Linf,true);
        }

    bench_approx_poly(5,LPTS_L1,false);
    bench_approx_poly(20,LPTS_Linf,false);
    bench_approx_poly(30,LPTS_zigag,false);
}

static Pt2dr random_pt(bool not_nul = false)
{
     REAL teta   = NRrandom3() *100;
     REAL rho = NRrandom3() *100 + (not_nul ? 1e-2 : 0);
     return Pt2dr::FromPolar(rho,teta);
}

static void random_polyl(ElFifo<Pt2dr> & f,INT nb)
{
     f.clear();
     for (INT k=0; k<nb ; k++)
         f.pushlast(random_pt());
}

static SegComp random_seg(bool not_nul)
{
      Pt2dr p1 = random_pt();
      return SegComp(p1,p1+random_pt(not_nul));
}

static SegComp  SegNotPar (const SegComp& s0)
{
      SegComp s1 = s0;
      while(ElAbs(angle(s1.tangente(),s0.tangente())) < 1e-3)
           s1 = random_seg(true);
      return s1;
}

/*
static SegComp  SegPar (const SegComp& s0)
{
      Pt2dr tr = random_pt();
      return SegComp(s0.p0()+tr,s0.p1()+tr);
}
*/

static bool pt_loin_from_bande(const SegComp & s,Pt2dr p)
{
       REAL absc = s.abscisse(p);
       return 
                  (ElAbs(absc)            > BIG_epsilon)
              &&  (ElAbs(absc-s.length()) > BIG_epsilon);
}

bool seg_prim_inside
     (
           const SegComp & s,
           Pt2dr pt,
           SegComp::ModePrim mode
     )
{

    switch(mode)
    {
         case SegComp::droite :
              return true;

         case SegComp::demi_droite :
              return (scal(pt-s.p0(),s.tangente()) > 0) ;

         default :
              return  (scal(pt-s.p0(),s.tangente()) <s.length()) 
                   && (scal(pt-s.p0(),s.tangente()) > 0)         ;
    }
}

SegComp::ModePrim ran_prim_seg()
{
     return  SegComp::ModePrim  ((INT) (2.999 *NRrandom3()));
}

void bench_dist_point_seg_droite()
/*
    On tire un segment vertical V et un point p, le calcul de d0 = D2(V,p)
    est trivial ;
   
    On tire une rotation affine r,  soit d1 la distance r(V), r(p),
    elle doit etre invariante par rotation. Ce processus donne
    des pointe te sgement quelconuq

   on verifie d1=d0
*/
{    INT f;
    for (f =0; f< 1000; f++)
    {
         ElFifo<Pt2dr> poly;
         poly.set_circ(NRrandom3() > 0.5);

         random_polyl(poly,(INT)(2+20*NRrandom3()));
         SegComp s = random_seg(true);
         SegComp::ModePrim  mode = ran_prim_seg();

         ElFifo<Pt2dr> inters;
         ElFifo<INT  > index;

         s.inter_polyline(mode,poly,index,inters);

         for (INT k=0;  k<index.nb(); k++)
         {
             INT ind = index[k];
             Pt2dr inter = inters[k];
             Pt2dr p0 = poly[ind];
             Pt2dr p1 = poly[ind+1];
             BENCH_ASSERT
             (
                   (s.square_dist(mode,inter)<epsilon)
                && (SegComp(p0,p1).square_dist_seg(inter) < epsilon)
             );
         }
         if ((mode==SegComp::droite) && poly.circ())
            BENCH_ASSERT((inters.nb()%2)==0);
    }
    
    for ( f = 0; f<10000 ; f++)
    {
         bool ok;
         SegComp::ModePrim m0 =  ran_prim_seg();
         SegComp::ModePrim m1 =  ran_prim_seg();
         SegComp s0 = random_seg(true);
         SegComp s1 = SegNotPar(s0);
         Pt2dr i = s0.inter(m0,s1,m1,ok);

         BENCH_ASSERT
         (
              (s0.square_dist_droite(i) < BIG_epsilon)
           && (s1.square_dist_droite(i) < BIG_epsilon)
         );
         if (    pt_loin_from_bande(s0,i)
              && pt_loin_from_bande(s1,i) 
            )
         {
            
            BENCH_ASSERT
            (
              ok == (        seg_prim_inside(s0,i,m0)
                          && seg_prim_inside(s1,i,m1)
                    )
            );
         }
             
    }

    for ( f = 0; f<10000 ; f++)
    {
        Pt2dr p1 = Pt2dr(0,NRrandom3()*1e3);
        Pt2dr p2 = Pt2dr(0,p1.y +10+1e3*NRrandom3());
        Pt2dr q  = Pt2dr((NRrandom3()-0.5)*1e4,(NRrandom3()-0.5)*1e4);

        SegComp::ModePrim  mode = ran_prim_seg();

        Pt2dr proj_q = Pt2dr(0,q.y);

         Pt2dr projP_q = proj_q;

        double d0 = ElSquare(q.x);

        double  dp0 = d0;
        if (proj_q.y>p2.y)
        {
            if (mode == SegComp::seg)
            {
               dp0 += ElSquare(proj_q.y-p2.y);
               projP_q.y = p2.y;
            }
        }
        else if (proj_q.y<p1.y)
        {
            if (mode != SegComp::droite)
            {
               dp0 += ElSquare(proj_q.y-p1.y);
               projP_q.y = p1.y;
            }
        }

        Pt2dr tr = Pt2dr((NRrandom3()-0.5)*1e5,(NRrandom3()-0.5)*1e5);
        REAL teta = NRrandom3() *100;
        Pt2dr  rot(cos(teta),sin(teta));
 
        p1 = tr + p1 * rot;
        p2 = tr + p2 * rot;
        q  = tr + q  * rot;
        proj_q  = tr + proj_q * rot;

        projP_q  = tr + projP_q * rot;

        SegComp s(p1,p2);
        REAL d1 = s.square_dist_droite(q);
        REAL dp1 = s.square_dist(mode,q);

        BENCH_ASSERT(ElAbs(d0 -d1) < BIG_epsilon);
        BENCH_ASSERT(ElAbs(dp0 -dp1) < BIG_epsilon);

        Pt2dr proj_q_2 = s.proj_ortho_droite(q);
        BENCH_ASSERT( euclid(proj_q-proj_q_2) < BIG_epsilon);


        BENCH_ASSERT(euclid(projP_q,s.proj_ortho(mode,q))<BIG_epsilon);
    }

    for ( f = 0; f<10000 ; f++)
    {
        REAL rho = 1+NRrandom3()*1e3;
        REAL teta = (NRrandom3()-0.5)*1.9999*PI;
        Pt2dr p1 = Pt2dr::FromPolar(rho,teta);
        REAL teta2 = angle(p1);
        Pt2dr p2 = Pt2dr::FromPolar(1+NRrandom3()*1e3,NRrandom3()*1e3);
        REAL teta3 = angle(p2,p1*p2);


        BENCH_ASSERT(ElAbs(teta2-teta)<epsilon);
        BENCH_ASSERT(ElAbs(teta3-teta)<epsilon);
        
    }

    for ( f =0; f< 2000; f++)
    {
         SegComp::ModePrim m0 =  ran_prim_seg();
         SegComp::ModePrim m1 =  ran_prim_seg();
         SegComp s0 = random_seg(true);
         SegComp s1 = SegNotPar(s0);


         Seg2d  proj = s0.proj_ortho(m0,s1,m1);

         BENCH_ASSERT
         (
             ElAbs
             (
                 square_euclid(proj.p0()-proj.p1())
               -s0.square_dist(m0,s1,m1)
             ) < epsilon
         );

         BENCH_ASSERT
         (
               (s0.square_dist(m0,proj.p0())<epsilon)
            && (s1.square_dist(m1,proj.p1())<epsilon)
         );

         for (INT k=0; k< 8*(2+(INT)m0)*(2+(INT)m1) ; k++)
         {
             Pt2dr q0 = proj.p0() + s0.tangente()*((NRrandom3()-0.5) * (1<<(k%10))) ;
             Pt2dr q1 = proj.p1() + s1.tangente()*((NRrandom3()-0.5) * (1<<(k%10))) ;

             q0 = s0.proj_ortho(m0,q0);
             q1 = s1.proj_ortho(m1,q1);

             BENCH_ASSERT
             (
                 euclid(proj.p0(),proj.p1())
               < (euclid(q0,q1)+epsilon)
             );

         }
    }


    cout << "OK OK OK DIIIIIIST \n";
}


/*
     On tire un paquet de point pi; pou chaque pi :
        * on accunule dans une RMat_Inertie
        * on accumule le carre de la distance a la droite;

    On verifie a la fin que resultats identique;
*/

void bench_dist_InerMat_seg_droite()
{
    for (int i = 0; i<100 ; i++)
    {
        Pt2dr p1  = Pt2dr((NRrandom3()-0.5)*1e4,(NRrandom3()-0.5)*1e4);
        Pt2dr p2 = p1;

        while (euclid(p1-p2) < 1e2)
              p2  = Pt2dr((NRrandom3()-0.5)*1e4,(NRrandom3()-0.5)*1e4);

        SegComp s(p1,p2);
        int nb = (int)(50 * NRrandom3());
        REAL d0 = 0.0;
        RMat_Inertie m;
        for (int j =0; j<nb ; j++)
        {
             Pt2dr q  = Pt2dr((NRrandom3()-0.5)*1e4,(NRrandom3()-0.5)*1e4);
             REAL pds = NRrandom3();
             m = m.plus_cple(q.x,q.y,pds);
             d0 += pds * s.square_dist_droite(q);
        }
        REAL d1 = square_dist_droite(s,m);

        BENCH_ASSERT(ElAbs(d0 -d1) < BIG_epsilon);
    }
}

/*
    On tire une rotation, on parcour l'image d'une rectangle centre en 0,0 et
    allonge horizontalement par cette rotation, on verifie que
    le cdg se trouve en l'image de 0,0 et que la direction se trouve
    (a pi pres) = la l'image de horizontal par cette rotation.
*/

void bench_seg_mean_square()
{
    for (int i = 0; i<100 ; i++)
    {
        INT nx = (INT) (10 +NRrandom3()*20);
        INT ny = nx  -5;

        Pt2dr  tr = Pt2dr((NRrandom3()-0.5)*1e4,(NRrandom3()-0.5)*1e4);
        Pt2dr  rot = Pt2dr::FromPolar(1.0,NRrandom3() *100);

        RMat_Inertie m;
        for (int x= -nx; x <= nx ; x++)
            for (int y= -ny; y <= ny ; y++)
            {
                 Pt2dr Z = tr+rot*Pt2dr(x,y);
                 m.add_pt_en_place(Z.x,Z.y);
            }

        Seg2d s = seg_mean_square(m,100.0);
        Pt2dr cdg = s.p0();
        Pt2dr all = (s.p1()-s.p0())/ 100.0;

        BENCH_ASSERT
        (
                (euclid(cdg-tr) < BIG_epsilon)
             && (ElAbs(all^rot)   < BIG_epsilon)
        );

        // BENCH_ASSERT(Abs(d0 -d1) < BIG_epsilon);
   }
}


class III 
{
     public :

     III(INT I) :
         _i  (I),
         adr (STD_NEW_TAB(10,INT))
     {
        NB++;
     }

     III(const III & I) :
         _i  (I._i),
         adr (STD_NEW_TAB(10,INT))
     {
        NB++;
     }

     ~III()
      {
        NB--;
          if (adr)
             STD_DELETE_TAB(adr);
      }

      III() :
         _i  (-10000),
         adr (0)
      {
        NB++;
      }

      INT & i() {return _i;}
        static INT NB;

     private :

        INT _i;
        INT * adr;

};
INT III::NB = 0;

void bench_filo()
{
    All_Memo_counter MC_INIT;
    stow_memory_counter(MC_INIT);
    INT nb = 200;

    {
         ElFilo<III> Fi(4);
		 INT i;

         for ( i= 0; i<nb; i++)
             Fi.pushlast(III(i));

         for ( i= 0; i<nb; i++)
             Fi[i].i() *= 2;

         for ( i= 0; i<nb; i++)
             BENCH_ASSERT(Fi[i].i() == 2*i);

         for ( i= 2*(nb-1) ; i>= 2; i-= 2)
             BENCH_ASSERT(Fi.poplast().i() == i);

         BENCH_ASSERT(Fi.nb() == 1);
    }
    verif_memory_state(MC_INIT);
}


/***********************************************/
/***********************************************/
/**                                           **/
/**           Box2d                           **/
/**                                           **/
/***********************************************/
/***********************************************/

class bench_Box2di
{
    public :
        static void  Suppr(Pt2di sz);

        static Pt2di PtRand(Pt2di sz)
        {
             return Pt2di(Pt2dr(sz.x*NRrandom3(),sz.y*NRrandom3()));
        }

        static Box2di BoxRand(Pt2di sz)
        {
             return Box2di(PtRand(sz),PtRand(sz));
        }

        static void Set(Im2D_U_INT1 I,Box2di box,INT val)
        {
            ELISE_COPY ( rectangle(box), val, I.out());
        }

       static bool verif_equal(Im2D_U_INT1 Im1,Im2D_U_INT1 Im2)
       {
           INT Dif;

           ELISE_COPY(Im1.all_pts(),Abs(Im1.in()-Im2.in()),sigma(Dif));
           BENCH_ASSERT(Dif==0);
           return Dif == 0;
       }
};

void bench_Box2di::Suppr(Pt2di sz)
{
    Im2D_U_INT1 Im1(sz.x,sz.y,0);
    Im2D_U_INT1 Im2(sz.x,sz.y,0);

    Box2di Add = BoxRand(sz);
    Box2di Suppr = BoxRand(sz);
    
    Set(Im1,Add,1);
    Set(Im1,Suppr,0);
    
    ModelBoxSubstr Model;
    Model.MakeIt(Add,Suppr);

    for (INT k=0 ; k<Model.NbBox() ; k++)
        ELISE_COPY(rectangle(Model.Box(k)),1+Im2.in(),Im2.out());

    verif_equal(Im1,Im2);

}




void bench_Box2di()
{
    for (INT k=0; k< 500 ; k++)
    {
         bench_Box2di::Suppr(Pt2di(5,5));
         bench_Box2di::Suppr(Pt2di(10,10));
         bench_Box2di::Suppr(Pt2di(20,20));
         bench_Box2di::Suppr(Pt2di(50,50));
    }
}

void bench_inter_Hor()
{

   for (INT k=0; k<5000 ; k++)
   {
       Pt2dr p0 = random_pt();
       Pt2dr p1 = random_pt();
       if (ElAbs(p0.y-p1.y)<1e-2)
          p1.y += 1;
       SegComp aSp(p0,p1);

       REAL anY =  NRrandom3() *100;
       Pt2dr q0 (0,anY);
       Pt2dr q1 (1,anY);

       SegComp aSq(q0,q1);

       bool OkInter;
       Pt2dr aInter0 = aSp.inter(aSq,OkInter);
       Pt2dr aInter2(aSp.AbsiceInterDroiteHoriz(anY),anY);

       REAL DH = euclid(aInter0,aInter2) ;
       BENCH_ASSERT(DH<BIG_epsilon);
   }
}

static ElSimilitude  RandomSim()
{
     return ElSimilitude( random_pt(), random_pt(true));
}

static ElAffin2D  RandAff2D()
{
	Pt2dr v10 =  random_pt(true);
	Pt2dr v01 =  random_pt();
	while (ElAbs(v10 ^v01) < 1e-3)
              v01 =  random_pt();
	     
	return ElAffin2D ( random_pt(), v10, v01);
}

void bench_simil_elAF2D()
{
    for (INT k=0 ; k< 1000 ; k++)
    {
        ElSimilitude s1 = RandomSim();  
        ElSimilitude s2 = RandomSim();  

	Pt2dr p1 = random_pt();
	Pt2dr p2 = s1(s2(p1));
	Pt2dr p3 = (s1*s2)(p1);
	Pt2dr p4 = (s1*s1.inv())(p1);
	Pt2dr p5 = ElSimilitude()(p1);

	BENCH_ASSERT(euclid(p1,p4)<epsilon);
	BENCH_ASSERT(euclid(p2,p3)<epsilon);
	BENCH_ASSERT(euclid(p1,p5)<epsilon);


	ElAffin2D a1 = RandAff2D();
	ElAffin2D a2 = RandAff2D();

	Pt2dr q1 = random_pt();
	Pt2dr q2 = a1(a2(q1));
	Pt2dr q3 = (a1*a2)(q1);
	Pt2dr q4 = (a1*a1.inv())(q1);
	Pt2dr q5 = ElAffin2D()(q1);


	BENCH_ASSERT(euclid(q1,q4)<epsilon);
	BENCH_ASSERT(euclid(q2,q3)<epsilon);
	BENCH_ASSERT(euclid(q1,q5)<epsilon);
	BENCH_ASSERT(euclid(a1(Pt2dr(0,0)),a1.I00())<epsilon);
	BENCH_ASSERT(euclid(a1(Pt2dr(1,0)),a1.I00()+a1.I10())<epsilon);
	BENCH_ASSERT(euclid(a1(Pt2dr(0,1)),a1.I00()+a1.I01())<epsilon);
    }
}


/******************************************************************/
/******************************************************************/
/******************************************************************/
/******************************************************************/
/******************************************************************/


void bench_algo_geo_0()
{
     bench_simil_elAF2D();
     bench_inter_Hor();
     bench_Box2di();
     bench_dist_InerMat_seg_droite();
     bench_seg_mean_square();
     bench_dist_point_seg_droite();
     bench_filo();
     bench_approx_poly();
     printf("OK ALGO GEO \n");
}





