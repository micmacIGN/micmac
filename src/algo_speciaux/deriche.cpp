/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr

   
    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in 
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte 
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/


#include "StdAfx.h"


/* Merci a Canny, Rachid Deriche et Tuan Dang */

//  Test commit
//  Test commit New PSW 03/2010

class TD_Canny_Deriche
{
    public :
       ~TD_Canny_Deriche();

       TD_Canny_Deriche(REAL alpha,int tx);

       void reset_recur_ar();

       void filtr_H_gx(REAL4 *);
       REAL4 * filtr_V_av_gx(REAL4 *);
       REAL4 * filtr_V_ar_gx(REAL4 *);


       void filtr_H_gy(REAL4 *);
       REAL4 * filtr_V_av_gy(REAL4 *);
       REAL4 * filtr_V_ar_gy(REAL4 *);

   private :

       REAL a,a1,a2,a3,a4;
       REAL b1,b2;
       REAL mAmpl;
       INT _tx;

       REAL4 * buf_P;
       REAL4 * buf_M;

       REAL4 ** Buf_gx_av;
       REAL4 ** Buf_V_XM;
       REAL4 ** Buf_gx_ar;
       REAL4 ** Buf_V_XP;

       REAL4 ** Buf_gy_av;
       REAL4 ** Buf_V_YM;
       REAL4 ** Buf_gy_ar;
       REAL4 ** Buf_V_YP;

       REAL4 ** matrice(Pt2di p1,Pt2di p2);
       void reset_matr(REAL4 **,Pt2di,Pt2di);
};

      /***********************************************************/
      /*                                                         */
      /*         Deriche_OPB_Comp                                */
      /*                                                         */
      /***********************************************************/

typedef REAL4 TY_DERICHE;

class Deriche_OPB_Comp : public Fonc_Num_OPB_TPL<REAL>
{
         public :


               Deriche_OPB_Comp
               (
                    const Arg_Fonc_Num_Comp & arg,
                    Fonc_Num                f0,
                    REAL                    alpha,
                    INT                     d0,
                    INT                     delta_ar,
                    INT                     per_reaf
               );


            virtual ~Deriche_OPB_Comp() 
            {
                  DELETE_MATRICE(_buf_gx_Ar,_p0_ar,_p1_ar);
                  DELETE_MATRICE(_buf_gy_Ar,_p0_ar,_p1_ar);
            };

         private :

            virtual void post_new_line(bool);
            void filtr_H(INT y);

            REAL4 * filtr_V_av_gx(INT y);
            REAL4 * filtr_V_av_gy(INT y);

            REAL4 * filtr_V_ar_gx(INT y);
            REAL4 * filtr_V_ar_gy(INT y);


            INT  _per_reaff;

            INT  _y_cpt;
            INT  _tx;

            TD_Canny_Deriche _tdcd;

            Pt2di    _p0_ar;
            Pt2di    _p1_ar;

            REAL4 ** _buf_gx_Ar;
            REAL4 ** _buf_gy_Ar;


};

REAL4 * Deriche_OPB_Comp::filtr_V_av_gx(INT y)
{
     return _tdcd.filtr_V_av_gx(kth_buf((TY_DERICHE *)0,0)[0][y]+_x0_buf) - _x0_buf;
}

REAL4 * Deriche_OPB_Comp::filtr_V_av_gy(INT y)
{
     return _tdcd.filtr_V_av_gy(kth_buf((TY_DERICHE *)0,0)[1][y]+_x0_buf) - _x0_buf;
}

REAL4 * Deriche_OPB_Comp::filtr_V_ar_gx(INT y)
{
     return _tdcd.filtr_V_ar_gx(kth_buf((TY_DERICHE *)0,0)[0][y]+_x0_buf) - _x0_buf;
}

REAL4 * Deriche_OPB_Comp::filtr_V_ar_gy(INT y)
{
     return _tdcd.filtr_V_ar_gy(kth_buf((TY_DERICHE *)0,0)[1][y]+_x0_buf) - _x0_buf;
}





void Deriche_OPB_Comp::filtr_H(INT y)
{

    _tdcd.filtr_H_gx(kth_buf((TY_DERICHE *)0,0)[0][y]+_x0_buf);
    _tdcd.filtr_H_gy(kth_buf((TY_DERICHE *)0,0)[1][y]+_x0_buf);

}

void Deriche_OPB_Comp::post_new_line(bool first)
{
     if (first)
     {
         for (int y = _y0_side; y< _y1_side; y++)
             filtr_H(y);
     }
     filtr_H(_y1_side);

     if (first)
     {
         for (int y = _y0_side; y< 0; y++)
         {
             filtr_V_av_gx(y);
             filtr_V_av_gy(y);
         }
     }

     if (! _y_cpt)
     {
        _tdcd.reset_recur_ar();
        for (INT y = _y1_side; y>=0 ; y--)
        {
              REAL4 *  lx = filtr_V_ar_gx(y);
              REAL4 *  ly = filtr_V_ar_gy(y);
              if (y < _per_reaff)
              {
                 convert(_buf_gx_Ar[y]+_x0_buf,lx+_x0_buf,_tx);
                 convert(_buf_gy_Ar[y]+_x0_buf,ly+_x0_buf,_tx);
              }
        }
     }

     REAL  * gx = _buf_res[0];
     REAL  * gy = _buf_res[1];
     REAL4 * gxav = filtr_V_av_gx(0);
     REAL4 * gyav = filtr_V_av_gy(0);

     REAL4 * gxar =  _buf_gx_Ar[_y_cpt];
     REAL4 * gyar =  _buf_gy_Ar[_y_cpt];

     for (INT x = _x0; x<_x1 ; x++)
     {
          gx[x] = gxav[x] + gxar[x];
          gy[x] = gyav[x] + gyar[x];
     }


     _y_cpt = (_y_cpt +1) %_per_reaff;
}


Deriche_OPB_Comp::Deriche_OPB_Comp
(
     const Arg_Fonc_Num_Comp & arg,
     Fonc_Num                f0,
     REAL                    alpha,
     INT                     d0,
     INT                     delta_ar,
     INT                     per_reaf
)      :
       Fonc_Num_OPB_TPL<REAL>
       (
              arg,
              2,
              Arg_FNOPB
              (
                 f0,
                 Box2di
                 (
                       Pt2di(-d0,-d0),
                       Pt2di(d0,per_reaf+d0+delta_ar)
                 ),
                 GenIm::real4
              ),
              Arg_FNOPB::def,
              Arg_FNOPB::def
       ),
      _per_reaff  (per_reaf),
      _y_cpt      (0),
      _tx         (_x1_buf-_x0_buf),
      _tdcd       (alpha,_tx),
      _p0_ar      (_x0_buf,0),
      _p1_ar      (_x1_buf,per_reaf),
      _buf_gx_Ar  (NEW_MATRICE(_p0_ar,_p1_ar,REAL4)),
      _buf_gy_Ar  (NEW_MATRICE(_p0_ar,_p1_ar,REAL4))
{
}


      /***********************************************************/
      /*                                                         */
      /*         Deriche_OPB_Not_Comp                            */
      /*                                                         */
      /***********************************************************/


class Deriche_OPB_Not_Comp : public Fonc_Num_Not_Comp
{
      public :
          Deriche_OPB_Not_Comp
          (
                Fonc_Num                   f,
                REAL                       alpha,
                INT                        d0
          );

      private :
          virtual bool  integral_fonc (bool) const
          {
               return false;
          }
          virtual INT dimf_out() const { return _f.dimf_out(); }
          
         void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}


          Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg)
          {

               INT delta_reaf = 20;
               INT per_reaf = delta_reaf * 2;

               return new Deriche_OPB_Comp
                          (
                                  arg,
                                  _f,
                                  _alpha,
                                  _d0,
                                  per_reaf,
                                  delta_reaf
                           );
          }

          Fonc_Num     _f;
          REAL         _alpha;
          INT          _d0;
};



Deriche_OPB_Not_Comp::Deriche_OPB_Not_Comp
(
        Fonc_Num                   f,
        REAL                       alpha,
        INT                        d0 
)    :
     _f      (f),
     _alpha  (alpha),
     _d0     (d0)
{
}


/*******************************************************************/
/*                                                                 */
/*      Interface Functions                                        */
/*                                                                 */
/*******************************************************************/


Fonc_Num deriche(Fonc_Num f,REAL alpha,INT d0)
{
     Tjs_El_User.ElAssert
     (
        f.dimf_out() == 1,
        EEM0 << "Need 1-dim out fonct for Canny-Deriche"
     );

     Symb_FNum SF(Rconv(f));
     return new Deriche_OPB_Not_Comp(Virgule(SF,SF),alpha,ElMax(d0,0));
}





void TD_Canny_Deriche::reset_matr(REAL4 **m,Pt2di p1,Pt2di p2)
{
   for (int y=p1.y ; y<p2.y ; y++)
       set_cste(m[y]+p1.x,(REAL4)0.0,p2.x-p1.x);
}

REAL4 ** TD_Canny_Deriche::matrice(Pt2di p1,Pt2di p2)
{
   REAL4 ** m = NEW_MATRICE(p1,p2,REAL4);
   reset_matr(m,p1,p2);
   return m;
}

void TD_Canny_Deriche::reset_recur_ar()
{
     reset_matr(Buf_gx_ar,Pt2di(0,0),Pt2di(_tx,3));
     reset_matr(Buf_V_XP,Pt2di(0,0),Pt2di(_tx,3));

     reset_matr(Buf_gy_ar,Pt2di(0,0),Pt2di(_tx,3));
     reset_matr(Buf_V_YP,Pt2di(0,0),Pt2di(_tx,3));
}


TD_Canny_Deriche::~TD_Canny_Deriche()
{
    DELETE_VECTOR(buf_P,0);
    DELETE_VECTOR(buf_M,0);

    DELETE_MATRICE(Buf_gx_av,Pt2di(0,-1),Pt2di(_tx,1));
    DELETE_MATRICE(Buf_V_XM,Pt2di(0,-2),Pt2di(_tx,1));

    DELETE_MATRICE(Buf_gx_ar,Pt2di(0,0),Pt2di(_tx,3));
    DELETE_MATRICE(Buf_V_XP,Pt2di(0,0),Pt2di(_tx,3));

    DELETE_MATRICE(Buf_gy_av,Pt2di(0,-1),Pt2di(_tx,1));
    DELETE_MATRICE(Buf_V_YM,Pt2di(0,-2),Pt2di(_tx,1));

    DELETE_MATRICE(Buf_gy_ar,Pt2di(0,0),Pt2di(_tx,3));
    DELETE_MATRICE(Buf_V_YP,Pt2di(0,0),Pt2di(_tx,3));
}



TD_Canny_Deriche::TD_Canny_Deriche(REAL alpha,int tx)
{
   REAL exp_alpha = (float) exp( (double) - alpha);
   REAL exp_2alpha = exp_alpha * exp_alpha;

   REAL kd = - (1-exp_alpha)*(1-exp_alpha)/exp_alpha;
   REAL ks = (1-exp_alpha)*(1-exp_alpha)/(1+ 2*alpha*exp_alpha - exp_2alpha); 
   
   a = kd * exp_alpha;
   b1 = 2 * exp_alpha;
   b2 = exp_2alpha;

   a1 = ks;  
   a2= ks * exp_alpha * (alpha-1) ;
   a3 = ks * exp_alpha * (alpha+1); 
   a4 = ks * exp_2alpha;

   mAmpl = (2*(b1-a-2*b2))/(1-b1+b2);

   _tx = tx;
   buf_P = NEW_VECTEUR(0,tx,REAL4);
   buf_M = NEW_VECTEUR(0,tx,REAL4);

   Buf_gx_av = matrice(Pt2di(0,-1),Pt2di(_tx,1));
   Buf_V_XM =  matrice(Pt2di(0,-2),Pt2di(_tx,1));

   Buf_gx_ar = matrice(Pt2di(0,0),Pt2di(_tx,3));
   Buf_V_XP =  matrice(Pt2di(0,0),Pt2di(_tx,3));

   Buf_gy_av = matrice(Pt2di(0,-1),Pt2di(_tx,1));
   Buf_V_YM =  matrice(Pt2di(0,-2),Pt2di(_tx,1));

   Buf_gy_ar = matrice(Pt2di(0,0),Pt2di(_tx,3));
   Buf_V_YP =  matrice(Pt2di(0,0),Pt2di(_tx,3));
}


void TD_Canny_Deriche::filtr_H_gx(REAL4 * gx)
{
     buf_M[0] = buf_M[1] = 0;
	 INT x;

     for (x=2; x<_tx ; x++)
         buf_M[x] = (float) ( a   * gx[x-1]
                    + b1  * buf_M[x-1]
                    - b2  * buf_M[x-2]);

     buf_P[_tx-1] = buf_P[_tx-2] = 0;

     for (x= _tx-3; x >= 0; x--)
         buf_P[x] =  (float) (- a  * gx[x+1] 
                         + b1 * buf_P[x+1] 
                         - b2 * buf_P[x+2]);

     for (x=0; x< _tx; x++) 
         gx[x] = (float)( (buf_M[x] + buf_P[x]) / mAmpl );

}

REAL4 *  TD_Canny_Deriche::filtr_V_av_gx(REAL4 *gx)
{
     rotate_plus_data(Buf_gx_av,-1,1);
     convert(Buf_gx_av[0],gx,_tx);
     rotate_plus_data(Buf_V_XM,-2,1);

     REAL4 * B0 = Buf_V_XM[0];
     REAL4 * B1 = Buf_V_XM[-1];
     REAL4 * B2 = Buf_V_XM[-2];
     REAL4 * G0 = Buf_gx_av[0];
     REAL4 * G1 = Buf_gx_av[-1];

     for (INT x=0; x< _tx; x++) 
         B0[x]  = (float) ( a1*G0[x]+a2*G1[x]+b1*B1[x]-b2*B2[x] ); 
    return B0;
}

REAL4 *  TD_Canny_Deriche::filtr_V_ar_gx(REAL4 *gx)
{
     rotate_moins_data(Buf_gx_ar,0,3);
     convert(Buf_gx_ar[0],gx,_tx);
     rotate_moins_data(Buf_V_XP,0,3);

     REAL4 * B0 = Buf_V_XP[0];
     REAL4 * B1 = Buf_V_XP[1];
     REAL4 * B2 = Buf_V_XP[2];
     REAL4 * G1 = Buf_gx_ar[1];
     REAL4 * G2 = Buf_gx_ar[2];

     for (INT x=0; x<_tx; x++)
         B0[x] =  (float) ( a3*G1[x]-a4*G2[x]+b1*B1[x]-b2*B2[x]); 

     return B0;
}




void TD_Canny_Deriche::filtr_H_gy(REAL4 * gy)
{
     buf_M[0] = buf_M[1] = 0;
	 INT x;

     for (x= 2; x< _tx; x++)              
         buf_M[x] =  (float) (    a1 * gy[x]
                     +  a2 * gy[x-1]
                     +  b1 * buf_M[x-1]
                     -  b2 * buf_M[x-2]);

     buf_P[_tx-1] = buf_P[_tx-2] =0;
     for (x= _tx-3; x>= 0; x--) 
           buf_P[x] =  (float) (  a3 * gy[x+1] 
                      - a4 * gy[x+2]
                      + b1 * buf_P[x+1]
                      - b2 * buf_P[x+2]);

      for (x= 0; x< _tx; x++)
          gy[x] = (float)( (buf_P[x] + buf_M[x]) / mAmpl );
}

REAL4 *  TD_Canny_Deriche::filtr_V_av_gy(REAL4 *gy)
{
     rotate_plus_data(Buf_gy_av,-1,1);
     convert(Buf_gy_av[0],gy,_tx);
     rotate_plus_data(Buf_V_YM,-2,1);

     REAL4 * B0 = Buf_V_YM[0];
     REAL4 * B1 = Buf_V_YM[-1];
     REAL4 * B2 = Buf_V_YM[-2];
     REAL4 * G1 = Buf_gy_av[-1];

     for (INT x=0; x< _tx; x++) 
         B0[x] = (float) ( a*G1[x]+b1*B1[x]-b2*B2[x]);
     return B0;
}

REAL4 *  TD_Canny_Deriche::filtr_V_ar_gy(REAL4 *gy)
{
     rotate_moins_data(Buf_gy_ar,0,3);
     convert(Buf_gy_ar[0],gy,_tx);
     rotate_moins_data(Buf_V_YP,0,3);

     REAL4 * B0 = Buf_V_YP[0];
     REAL4 * B1 = Buf_V_YP[1];
     REAL4 * B2 = Buf_V_YP[2];
     REAL4 * G1 = Buf_gy_ar[1];

     for (INT x=0; x< _tx; x++)  
        B0[x] = (float) ( -a*G1[x]+b1*B1[x]-b2*B2[x]);
     return B0;
}


//   Just an interface function for the old code 
//   still using the non-functional version

void deriche_uc
     (
           Im2D_REAL4  gx,
           Im2D_REAL4  gy,
           Im2D_U_INT1 i,
           REAL4       alpha
     )
{
     ELISE_COPY
     (
         i.all_pts(),
         deriche(i.in(0),alpha,5),
         Virgule(gx.out(),gy.out())
     );
}


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
