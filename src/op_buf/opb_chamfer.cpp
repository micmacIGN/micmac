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

#define CH16Bits 1

#if CH16Bits
#define TY_CHAMF U_INT2
#define TY_IM_CHMF GenIm::u_int2
#else
#define TY_CHAMF U_INT1
#define TY_IM_CHMF GenIm::u_int1
#endif


/*********************************************************************/
/*                                                                   */
/*         BINARIZATION ADAPTATER                                    */
/*                                                                   */
/*********************************************************************/

class Binarize_Fonc_Comp : public  Fonc_Num_Comp_TPL<INT>
{
       public :

         Binarize_Fonc_Comp
         (
              const Arg_Fonc_Num_Comp &,
              Fonc_Num_Computed *,
              INT val,
              bool neg
         );

         ~Binarize_Fonc_Comp() {delete  _f;}

      private :

            const Pack_Of_Pts * values(const Pack_Of_Pts * pts)
            {
                INT nb = pts->nb();
                INT **  v_in =
                    SAFE_DYNC(Std_Pack_Of_Pts<INT> *,const_cast<Pack_Of_Pts *>(_f->values(pts)))->_pts;

                INT **  v_out =
                      SAFE_DYNC(Std_Pack_Of_Pts<INT> *,_pack_out)->_pts; 

                for (INT d = 0; d < _dim_out; d++)
                    if (_neg)
                       neg_binarise(v_out[d],v_in[d],_val,nb);
                    else
                       binarise(v_out[d],v_in[d],_val,nb);

                _pack_out->set_nb(nb);
                return _pack_out;
            }



           Fonc_Num_Computed * _f;
           INT _val;
           bool _neg;
};

Binarize_Fonc_Comp::Binarize_Fonc_Comp
(
     const Arg_Fonc_Num_Comp & arg,
     Fonc_Num_Computed *       f,
     INT                       val,
     bool                      neg
)  :
     Fonc_Num_Comp_TPL<INT> (arg,f->idim_out(),arg.flux()),
     _f (f),
     _val (val),
     _neg (neg)
{
}



class Binarize_Not_Comp : public  Fonc_Num_Not_Comp
{
      public :
          Binarize_Not_Comp(Fonc_Num,INT,bool);

      private :


          Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg)
          {
              Fonc_Num_Computed *f =_f.compute(arg);
              return new Binarize_Fonc_Comp(arg,f,_val,_neg);
          }

           int dimf_out() const { return _f.dimf_out();}
           void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}
           bool integral_fonc(bool) const { return true;}

          Fonc_Num _f;
          INT _val;
          bool _neg;
};

Binarize_Not_Comp::Binarize_Not_Comp
(
     Fonc_Num f,
     INT      val,
     bool     neg
)  :
   _f  (Iconv(f)),
   _val (val),
   _neg (neg)
{
}


Fonc_Num binarize(Fonc_Num f,INT val,bool neg)
{
     return new Binarize_Not_Comp(f,val,neg);
}



/*********************************************************************/
/*********************************************************************/
/*********************************************************************/
/*                                                                   */
/*         CHAMFER                                                   */
/*                                                                   */
/*********************************************************************/
/*********************************************************************/
/*********************************************************************/
/*********************************************************************/


      /***********************************************************/
      /*                                                         */
      /*         Chamfer_OPB_Comp                                */
      /*                                                         */
      /***********************************************************/



class Chamfer_OPB_Comp : public Fonc_Num_OPB_TPL<INT>
{
         public :


               typedef void (Chamfer_OPB_Comp:: * post_tr) 
                            (INT *,const TY_CHAMF *,INT nb);

               void post_conv (INT * out,const TY_CHAMF * in,INT nb)
               {
                    convert(out,in,nb);
               }

               void post_erod (INT * out,const TY_CHAMF * in,INT nb);
               void post_dilat (INT * out,const TY_CHAMF * in,INT nb);

               Chamfer_OPB_Comp
               (
                    const Arg_Fonc_Num_Comp & arg,
                    INT                     dim_out,
                    Fonc_Num                f0,
                    const Chamfer &         chamf,
                    INT                     r,
                    INT                     max_d,
                    INT                     delta,
                    INT                     per_reaf,
                    post_tr                 post
               );

         private :


            virtual void post_new_line(bool);

            void prop_av(INT yloc);
            void prop_ar(INT yloc);
            void prop
            (
                   INT yloc,
                   const Pt2di * v,const INT * p,INT nbv,
                   INT x0,INT x1,INT dx
            );

            INT  _nb;
            INT  _delta;
            INT  _per_reaff;
            const Chamfer & _chamf;

            INT   _max_d;
            INT  _y_cpt;
            post_tr  _post;

};

void Chamfer_OPB_Comp::post_erod (INT * out,const TY_CHAMF * in,INT nb)
{
     for (INT i=0 ; i<nb ; i++)
         out[i] = (in[i] == _max_d);
}

void Chamfer_OPB_Comp::post_dilat (INT * out,const TY_CHAMF * in,INT nb)
{
     for (INT i=0 ; i<nb ; i++)
         out[i] = (in[i] != _max_d);
}


Chamfer_OPB_Comp::Chamfer_OPB_Comp
(
    const Arg_Fonc_Num_Comp & arg,
    INT                     dim_out,
    Fonc_Num                f0,
    const Chamfer &         chamf,
    INT                      r,
    INT                     max_d,
    INT                     delta,
    INT                     per_reaf,
    post_tr                 post
)   :
       Fonc_Num_OPB_TPL<INT>
       (
              arg,
              dim_out,
              Arg_FNOPB
              (
                 f0,
                 Box2di
                 (
                       Pt2di(-(r+delta),-(r+delta)),
                       Pt2di(r+delta,per_reaf+delta+r-1)
                 ),
                 TY_IM_CHMF
              )
       ),
      _nb         (per_reaf+delta),
      _delta      (delta),
      _per_reaff  (per_reaf),
      _chamf      (chamf),
      _max_d      (max_d),
      _y_cpt      (0),
      _post       (post)
{
}

void Chamfer_OPB_Comp::prop
     (
           INT yloc,
           const Pt2di * v,
           const INT   * pds,
           INT           nbv,
           INT           x0,
           INT           x1,
           INT           dx
     )
{
      for (INT d=0; d<_dim_out; d++)
      {
           TY_CHAMF ** l = kth_buf((TY_CHAMF *)0,0)[d]+yloc;
           TY_CHAMF * l0 = l[0];

           for (INT x= x0 ; x != x1 ; x+= dx)
               if (l0[x])
               {
                  for (INT k=0 ; k<nbv ; k++)
                      l0[x] = ElMin((INT)l0[x],l[v[k].y][x+v[k].x]+pds[k]);
               }
      }
}

void Chamfer_OPB_Comp::prop_av(INT yloc)
{
     prop
     (
         yloc,
         _chamf.neigh_yn(),_chamf.pds_yn(),_chamf.nbv_yn(),
         _x0-_delta,_x1+_delta,1
     );
}

void Chamfer_OPB_Comp::prop_ar(INT yloc)
{
     prop
     (
         yloc,
         _chamf.neigh_yp(),_chamf.pds_yp(),_chamf.nbv_yp(),
         _x1+_delta-1,_x0-_delta-1,-1
     );
}



void Chamfer_OPB_Comp::post_new_line(bool first)
{
      if (first)
      {
           for (INT y = -_delta ; y < _nb; y++)
               prop_av(y);
      }
      prop_av(_nb);
     
     if (! _y_cpt)
         for (INT y = _nb-1 ; y >= 0; y--)
               prop_ar(y);

     for (INT d =0; d < _dim_out ; d++)
     {
          TY_CHAMF * l0 = kth_buf((TY_CHAMF *)0,0)[d][0];
          INT * res = _buf_res[d];
		  (this->*_post)(res+_x0,l0+_x0,_x1-_x0);
     }


     _y_cpt = (_y_cpt +1) %_per_reaff;

}


      /***********************************************************/
      /*                                                         */
      /*         Chamfer_OPB_Not_Comp                            */
      /*                                                         */
      /***********************************************************/


class Chamfer_OPB_Not_Comp : public Fonc_Num_Not_Comp
{
      public :
          Chamfer_OPB_Not_Comp
          (
                Fonc_Num                   f0,
                const Chamfer &            chamf,
                INT                        max_d,
                Chamfer_OPB_Comp::post_tr       ,
                bool                        neg = false,
                bool                        binarise_input = true
          );

      private :
          virtual bool  integral_fonc (bool) const
          {
               return true;
          }
          virtual INT dimf_out() const { return _f.dimf_out(); }
          
          void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}


          Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg)
          {
               Box2di b (Pt2di(0,0),Pt2di(0,0));

               ASSERT_TJS_USER
               (
                    arg.flux()->is_rect_2d(b),
                    "Need Rect Fluc for chamfer dist operator"
               );


               INT delta =  (_max_d + _chamf.p_0_1()-1)/_chamf.p_0_1();
               INT per_reaf = (INT) (2.5 * delta) + 5;
               per_reaf = adjust_nb_pack_y(per_reaf,b._p1.y-b._p0.y);
                
               Fonc_Num f = _f;
               if (_binarise_input)
                  f = binarize(f,_max_d,_neg);
               else
                  f = Iconv(f);

               return new Chamfer_OPB_Comp
                          (
                                  arg,
                                  f.dimf_out(),
                                  f,
                                  _chamf,
                                  _chamf.radius(),
                                  _max_d,
                                  delta,
                                  per_reaf,
                                  _post
                           );
          }

          Fonc_Num                      _f;
          const Chamfer &               _chamf;
          INT                           _max_d;
          Chamfer_OPB_Comp::post_tr     _post;
          bool                          _neg;
          bool                          _binarise_input;
};

Chamfer_OPB_Not_Comp::Chamfer_OPB_Not_Comp
(
        Fonc_Num                      f0,
        const Chamfer &               chamf,
        INT                           max_d,
        Chamfer_OPB_Comp::post_tr     post,
        bool                          neg,
        bool                          binarise_input
)    :
     _f                (f0),
     _chamf            (chamf), 
     _max_d            (CH16Bits ? max_d : (ElMin(255,max_d)) ),
     _post             (post),
     _neg              (neg),
     _binarise_input   (binarise_input)
{
}


/*******************************************************************/
/*                                                                 */
/*      Interface Functions                                        */
/*                                                                 */
/*******************************************************************/


Fonc_Num EnvKLipshcitz(Fonc_Num f,const Chamfer & chamf,INT max_dif)
{
         return new Chamfer_OPB_Not_Comp
                    (
                         f,
                         chamf,
                         max_dif,
                         &Chamfer_OPB_Comp::post_conv,
                         false,
                         false
                    );
}

Fonc_Num EnvKLipshcitz_5711(Fonc_Num f,INT max_dif)
{
         return EnvKLipshcitz(f,Chamfer::d5711,max_dif);
}
Fonc_Num EnvKLipshcitz_32(Fonc_Num f,INT max_dif)
{
         return EnvKLipshcitz(f,Chamfer::d32,max_dif);
}
Fonc_Num EnvKLipshcitz_d8(Fonc_Num f,INT max_dif)
{
         return EnvKLipshcitz(f,Chamfer::d8,max_dif);
}
Fonc_Num EnvKLipshcitz_d4(Fonc_Num f,INT max_dif)
{
         return EnvKLipshcitz(f,Chamfer::d4,max_dif);
}




              // =========== extinc ==============

Fonc_Num extinc(Fonc_Num f,const Chamfer & chamf,INT max_d)
{
         return new Chamfer_OPB_Not_Comp
                    (f,chamf,max_d,&Chamfer_OPB_Comp::post_conv);
}

Fonc_Num extinc_5711(Fonc_Num f,INT max_d)
{
         return extinc(f,Chamfer::d5711,max_d);
}
Fonc_Num extinc_32(Fonc_Num f,INT max_d)
{
         return extinc(f,Chamfer::d32,max_d);
}
Fonc_Num extinc_d8(Fonc_Num f,INT max_d)
{
         return extinc(f,Chamfer::d8,max_d);
}
Fonc_Num extinc_d4(Fonc_Num f,INT max_d)
{
         return extinc(f,Chamfer::d4,max_d);
}


              // =========== erod  ==============


Fonc_Num erod(Fonc_Num f,const Chamfer & chamf,INT max_d)
{
         return new Chamfer_OPB_Not_Comp
                    (f,chamf,max_d+1,&Chamfer_OPB_Comp::post_erod);
}

Fonc_Num erod_5711(Fonc_Num f,INT max_d)
{
         return erod(f,Chamfer::d5711,max_d);
}
Fonc_Num erod_32(Fonc_Num f,INT max_d)
{
         return erod(f,Chamfer::d32,max_d);
}
Fonc_Num erod_d8(Fonc_Num f,INT max_d)
{
         return erod(f,Chamfer::d8,max_d);
}
Fonc_Num erod_d4(Fonc_Num f,INT max_d)
{
         return erod(f,Chamfer::d4,max_d);
}


              // =========== dilat  ==============


Fonc_Num dilat(Fonc_Num f,const Chamfer & chamf,INT max_d)
{
         return new Chamfer_OPB_Not_Comp
                    (f,chamf,max_d+1,&Chamfer_OPB_Comp::post_dilat,true);
}

Fonc_Num dilat_5711(Fonc_Num f,INT max_d)
{
         return dilat(f,Chamfer::d5711,max_d);
}
Fonc_Num dilat_32(Fonc_Num f,INT max_d)
{
         return dilat(f,Chamfer::d32,max_d);
}
Fonc_Num dilat_d8(Fonc_Num f,INT max_d)
{
         return dilat(f,Chamfer::d8,max_d);
}
Fonc_Num dilat_d4(Fonc_Num f,INT max_d)
{
         return dilat(f,Chamfer::d4,max_d);
}


              // =========== opening  ============

Fonc_Num open(Fonc_Num f, const Chamfer & chamf,INT max_d,INT delta)
{
         return dilat(erod(f,chamf,max_d),chamf,max_d+delta);
}

Fonc_Num open_5711(Fonc_Num f,INT max_d,INT delta)
{
         return dilat_5711(erod_5711(f,max_d),max_d+delta);
}
Fonc_Num open_32(Fonc_Num f,INT max_d,INT delta)
{
         return dilat_32(erod_32(f,max_d),max_d+delta);
}
Fonc_Num open_d8(Fonc_Num f,INT max_d,INT delta)
{
         return dilat_d8(erod_d8(f,max_d),max_d+delta);
}
Fonc_Num open_d4(Fonc_Num f,INT max_d,INT delta)
{
         return dilat_d4(erod_d4(f,max_d),max_d+delta);
}

              // =========== closing  ============

Fonc_Num close(Fonc_Num f, const Chamfer & chamf,INT max_d,INT delta)
{
         return erod(dilat(f,chamf,max_d),chamf,max_d+delta);
}

Fonc_Num close_5711(Fonc_Num f,INT max_d,INT delta)
{
         return erod_5711(dilat_5711(f,max_d),max_d+delta);
}
Fonc_Num close_32(Fonc_Num f,INT max_d,INT delta)
{
         return erod_32(dilat_32(f,max_d),max_d+delta);
}
Fonc_Num close_d8(Fonc_Num f,INT max_d,INT delta)
{
         return erod_d8(dilat_d8(f,max_d),max_d+delta);
}
Fonc_Num close_d4(Fonc_Num f,INT max_d,INT delta)
{
         return erod_d4(dilat_d4(f,max_d),max_d+delta);
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
