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





/*********************************************************************/
/*********************************************************************/
/*********************************************************************/
/*                                                                   */
/*         CANNY EXPONENTIAL FILTER                                  */
/*                                                                   */
/*********************************************************************/
/*********************************************************************/
/*********************************************************************/
/*********************************************************************/

            /* -  -  -  -  -  -  -  -  -  -  -  -  -  - */
            /*                                          */
            /*           Red_Ass_OPB_Comp               */
            /*                                          */
            /* -  -  -  -  -  -  -  -  -  -  -  -  -  - */

void exponential_filter
     (
        REAL * out,
        REAL * buf,
        const REAL * in,
        INT          nb,
        REAL         fact,
        bool         double_sens
     )
{
     if (! nb)
        return;

     REAL cum = 0.0;

     if (double_sens)
     {
         for (INT i=0; i<nb ; i++)
         {
             cum = fact * cum  + in[i];
             buf[i] = cum;
         }

         cum = 0.0;
		 {
			for (INT i = nb-2; i>=0 ; i--)
			{
				 cum = fact * (cum  + in[i+1]);
				 buf[i]+=cum;
			}
		 }
         convert(out,buf,nb);
     }
     else
         for (INT i=0; i<nb ; i++)
         {
             cum = fact * cum  + in[i];
             out[i] = cum;
         }
     
}


class Can_Exp_OPB_Comp : public Fonc_Num_OPB_TPL<REAL>
{
         public :

               Can_Exp_OPB_Comp
               (
                    const Arg_Fonc_Num_Comp & arg,
                    INT                     dim_out,
                    Fonc_Num                f0,
                    REAL                    fx,
                    REAL                    fy,
                    INT                     nb,
                    bool                    rec_ar,
                    bool                    double_sens
               );

               virtual ~Can_Exp_OPB_Comp();

         private :
            virtual void post_new_line(bool);
            void  exp_line(INT yloc);

            REAL _fx;
            REAL _fy;
            INT  _nb;
            bool _rec_arr;
            bool _double_sens;
            INT      _cptl;


            REAL *   _buf_exp_x;
            REAL **  _cumul_line_av;

            REAL *** _buf_rec_arr;
            REAL **  _correc_arr;

};


Can_Exp_OPB_Comp::Can_Exp_OPB_Comp
(
                    const Arg_Fonc_Num_Comp & arg,
                    INT                     dim_out,
                    Fonc_Num                f0,
                    REAL                    fx,
                    REAL                    fy,
                    INT                     nb,
                    bool                    rec_ar,
                    bool                    double_sens
) :
       Fonc_Num_OPB_TPL<REAL>
       (
             arg,
             dim_out,
             Arg_FNOPB(f0,Box2di(Pt2di(0,0),Pt2di(0,nb)))
        ),
       _fx (fx),
       _fy (fy),
       _nb (nb),
       _rec_arr (rec_ar),
       _double_sens (double_sens),
       _cptl    (0)
{

     _buf_exp_x      = NEW_VECTEUR(_x0,_x1,REAL);
     _cumul_line_av  = NEW_MATRICE(Pt2di(_x0,0),Pt2di(_x1,_dim_out),REAL);
     if (_rec_arr)
     {
        _buf_rec_arr    = NEW_TAB_MATRICE(_dim_out,Pt2di(_x0,1),Pt2di(_x1,_nb+1),REAL);
        _correc_arr     = NEW_MATRICE(Pt2di(_x0,0),Pt2di(_x1,_dim_out),REAL);
     }
}


Can_Exp_OPB_Comp::~Can_Exp_OPB_Comp()
{
     DELETE_VECTOR(_buf_exp_x,_x0);
     DELETE_MATRICE(_cumul_line_av,Pt2di(_x0,0),Pt2di(_x1,_dim_out));
     if (_rec_arr)
     {
        DELETE_MATRICE(_correc_arr,Pt2di(_x0,0),Pt2di(_x1,_dim_out));
        DELETE_TAB_MATRICE(_buf_rec_arr,_dim_out,Pt2di(_x0,1),Pt2di(_x1,_nb+1));
     }
}

void  Can_Exp_OPB_Comp::exp_line(INT yloc)
{
      for (INT d=0; d<_dim_out; d++)
      {
           REAL * line = kth_buf((REAL *)0,0)[d][yloc]; 
           exponential_filter
           (
                line+_x0,
                _buf_exp_x+_x0,
                line+_x0,
                _x1-_x0,
                _fx,
                _double_sens
           );
      }
}



void Can_Exp_OPB_Comp::post_new_line(bool first)
{
        if (first)
        {
           for (INT y = 0 ; y < _nb; y++)
               exp_line(y);
        }
        exp_line(_nb);

        for (INT d =0; d < _dim_out ; d++)
        {
            REAL ** lines = kth_buf((REAL *)0,0)[d];
            REAL *    bav = _cumul_line_av[d];
            REAL * res = _buf_res[d];

            if (first)
               set_cste(bav+_x0,0.0,_x1-_x0);

            {
               REAL * l0     = lines[0];
               for (INT x=_x0 ; x<_x1 ; x++)
                   res[x] = bav[x] = _fy * bav[x] + l0[x];
             }

            if (! _nb)
               return;

            if (_rec_arr)
            {
                REAL ** brecar = _buf_rec_arr[d];
                REAL *  corar  = _correc_arr[d];

                if (! _cptl)
                {
                   convert
                   (
                      brecar[_nb]+_x0,
                      lines[_nb] +_x0,
                      _x1-_x0
                   );
                   for (INT y=_nb-1 ; y>=1 ; y--)
                   {
                      REAL * ly   = lines[y];
                      REAL * cur  = brecar[y];
                      REAL * prec = brecar[y+1];
                      for (INT x=_x0 ; x<_x1 ; x++)
                          cur[x] = ly[x] + _fy * prec[x];
                   }
                   set_cste(corar+_x0,0.0,_x1-_x0);
                }
                else
                {
                   REAL * lnb   = lines[_nb];
                   REAL fact_cor    = Pow(_fy,_cptl-1);
                   for (INT x = _x0 ; x<_x1 ; x++)
                      corar[x] +=  fact_cor * lnb[x];
                }
                REAL * lar = brecar[_cptl+1];
                REAL fact_cor = Pow(_fy,_nb+1-_cptl);
                for (INT x = _x0 ; x<_x1 ; x++)
                    res[x] +=  _fy*lar[x] + fact_cor * corar[x];
            }
            else
            {
                 REAL   fNy = 1.0;
                 for (INT y=1 ; y<=_nb ; y++)
                 {
                      fNy *= _fy;
                      REAL * ly = lines[y];
                      for (INT x = _x0 ; x<_x1 ; x++)
                          res[x] += ly[x] * fNy ;
                 }
            }

       }
       _cptl    = (_cptl+1) % _nb;
}


            /* -  -  -  -  -  -  -  -  -  -  -  -  -  - */
            /*                                          */
            /*           Red_Ass_OPB_No_Comp            */
            /*                                          */
            /* -  -  -  -  -  -  -  -  -  -  -  -  -  - */

class Can_Exp_OPB_Not_Comp : public Fonc_Num_Not_Comp
{
      public :
          Can_Exp_OPB_Not_Comp
          (
                    Fonc_Num                f0,
                    REAL                    fx,
                    REAL                    fy,
                    INT                     nb,
                    bool                    rec_ar,
                    bool                    double_sens
          );

      private :

          virtual bool  integral_fonc (bool) const
          {
               return false;
          }

          virtual INT dimf_out() const
          {
              return _f.dimf_out();
          }
          void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}


          Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg);

          Fonc_Num                 _f;
          REAL                     _fx;
          REAL                     _fy;
          INT                      _nb;
          bool                     _rec_ar;
          bool                     _double_sens;
};

Can_Exp_OPB_Not_Comp::Can_Exp_OPB_Not_Comp
(
      Fonc_Num                f,
      REAL                    fx,
      REAL                    fy,
      INT                     nb,
      bool                    rec_ar,
      bool                    double_sens
)   :
    _f      (Rconv(f)),
    _fx     (fx),
    _fy     (fy),
    _nb     (nb),
    _rec_ar (rec_ar),
    _double_sens (double_sens)
{
}


Fonc_Num_Computed * Can_Exp_OPB_Not_Comp::compute
                    (const Arg_Fonc_Num_Comp & arg)
{
          return new Can_Exp_OPB_Comp
                      (  arg,dimf_out(),_f,_fx,_fy,
                        _nb,_rec_ar,_double_sens
                      );
}


Fonc_Num canny_exp_filt(Fonc_Num f,REAL fx,REAL fy,INT nb)
{
    if (nb < 0)
    {
       nb = ElMax(2,ElMin(100,round_up(log(1e-9)/log(fy))));
    }
    return new Can_Exp_OPB_Not_Comp(f,fx,fy,nb,nb>4,true);
}

Fonc_Num semi_cef(Fonc_Num f,REAL fx,REAL fy)
{
         return new Can_Exp_OPB_Not_Comp(f,fx,fy,0,false,false);
}

Fonc_Num inv_semi_cef(Fonc_Num f,REAL fx,REAL fy)
{
    Im2D_REAL8 Im(2,2);
    Im.data()[0][0] = fx * fy;
    Im.data()[0][1] = -fx;
    Im.data()[1][0] = -fy;
    Im.data()[1][1] = 1;
    return som_masq(f,Im,Pt2di(-1,-1));
}

static REAL CoeffICEF(REAL fact,INT I)
{
    return ((I==1) ?(1+ElSquare(fact)) : (-fact)) /(1-ElSquare(fact));
}

Fonc_Num  inv_canny_exp_filt(Fonc_Num f,REAL fx,REAL fy)
{
    Im2D_REAL8 Im(3,3);
    for (INT y=0; y<3 ; y++)
    {
        for (INT x=0; x<3 ; x++)
        {
            Im.data()[y][x] = CoeffICEF(fy,y)*CoeffICEF(fx,x);
        }
    }
    return som_masq(f,Im);
}

Fonc_Num inv_canny_exp_filt(Fonc_Num f,REAL fxy)
{
   return inv_canny_exp_filt(f,fxy,fxy);
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
