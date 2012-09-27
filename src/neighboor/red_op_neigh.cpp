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
/*                                                                   */
/*         Red_Op_Neigh_Comp                                         */
/*                                                                   */
/*********************************************************************/

template <class Type> class Red_Op_Neigh_Comp : public Fonc_Num_Comp_TPL<Type>
{
      public :

           Red_Op_Neigh_Comp();

           Red_Op_Neigh_Comp
           (
                   const Arg_Fonc_Num_Comp  & argf,
                   Neigh_Rel_Compute *,
                   Fonc_Num_Computed *,
                   Flux_Pts_Computed *flx_interf,
                   const OperAssocMixte &  _op
           );
           virtual ~Red_Op_Neigh_Comp();

      private :


           const Pack_Of_Pts * values(const Pack_Of_Pts *);

           const OperAssocMixte &  _op;
           Pt2di                   _sbuf;
           Type ***                _v_buf;
           char **                 _is_neigh;
           Neigh_Rel_Compute *     _n;
           Fonc_Num_Computed *     _f;
           Flux_Pts_Computed *     _flx_interf;
           INT                     _nb_neigh;
           INT                     _dim_out;
           Type                    _neutre;
};


template <class Type> Red_Op_Neigh_Comp<Type>::~Red_Op_Neigh_Comp()
{
     DELETE_TAB_MATRICE(_v_buf,_f->idim_out(),Pt2di(0,0),_sbuf);
     DELETE_MATRICE(_is_neigh,Pt2di(0,0),_sbuf);
     delete _n;
     delete _f;
     delete _flx_interf;
}

template <class Type> Red_Op_Neigh_Comp<Type>::Red_Op_Neigh_Comp
                      (
                            const Arg_Fonc_Num_Comp  & argf,
                            Neigh_Rel_Compute * n,
                            Fonc_Num_Computed * f,
                            Flux_Pts_Computed *flx_interf,
                            const OperAssocMixte &  op

                       )  : 
     Fonc_Num_Comp_TPL<Type>(argf,f->idim_out(),argf.flux()),
     _op   (op),
     _sbuf  (argf.flux()->sz_buf(),n->nb_neigh()+1),
     _v_buf (NEW_TAB_MATRICE(f->idim_out(),Pt2di(0,0),_sbuf,Type)),
     _is_neigh  (NEW_MATRICE(Pt2di(0,0),_sbuf,char)),
     _n            (n),
     _f            (f),
     _flx_interf  (flx_interf),
     _nb_neigh     (n->nb_neigh()),
     _dim_out      (_f->idim_out())
{
     _op.set_neutre(_neutre);
}

template <class Type> const Pack_Of_Pts * 
         Red_Op_Neigh_Comp<Type>::values(const Pack_Of_Pts * pack_pts)
{
        
   INT nb_pts = pack_pts->nb();
   INT ndir = 0;
   while (ndir !=  _nb_neigh)
   {
       INT last_dir = ndir;
       const Pack_Of_Pts * neigh = 
           _n->neigh_in_num_dir(pack_pts,_is_neigh,ndir);

       const Std_Pack_Of_Pts<Type> * vals 
              = _f->values(neigh)->std_cast((Type *)0);
       

       for (INT d_out = 0; d_out < _dim_out ; d_out ++)
       {
          Type * _vf = vals->_pts[d_out];
          INT i_tot = 0;

          for (INT dir = last_dir; dir <ndir ; dir++)
          {
              Type * vb = _v_buf[d_out][dir];
              char * sel = _is_neigh[dir];
              for (INT ipt=0 ; ipt<nb_pts ; ipt++)
              {
                   vb[ipt] = (sel[ipt]) ? _vf[i_tot++] : _neutre;
              }
          }
          El_Internal.ElAssert
          (
              i_tot == neigh->nb(),
              EEM0 << "Bad assertion in"
                   << " Sel_Func_Neigh_Rel_Comp::neigh_in_num_dir"
           );
       }
   }

   for (INT d_out = 0; d_out < _dim_out ; d_out ++)
   {
        Type * o = this->_pack_out->_pts[d_out];
        Type ** vb = _v_buf[d_out];
        set_cste(o,_neutre,nb_pts);
        for (INT dir=0;  dir<_nb_neigh; dir++)
            _op.t0_eg_t1_op_t2(o,o,vb[dir],nb_pts);
   }
   this->_pack_out->set_nb(nb_pts);
   return this->_pack_out;
}

/*********************************************************************/
/*                                                                   */
/*         Red_Op_Neigh_Not_Comp                                     */
/*                                                                   */
/*********************************************************************/

class Red_Op_Neigh_Not_Comp : public  Fonc_Num_Not_Comp
{
      public :

          Red_Op_Neigh_Not_Comp(const OperAssocMixte &,Neigh_Rel,Fonc_Num);
          

          Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg);

      private :

           const OperAssocMixte &  _op;
           Neigh_Rel               _n;
           Fonc_Num                _f;


         virtual bool  integral_fonc (bool iflx) const
         {
               return _f.integral_fonc(iflx);
         }

         virtual INT dimf_out() const
         {
                return _f.dimf_out();
         }
         void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}
};

Red_Op_Neigh_Not_Comp::Red_Op_Neigh_Not_Comp
(
         const OperAssocMixte &  op,
         Neigh_Rel n,
         Fonc_Num f
)                 :
       _op (op),
       _n  (n),
       _f  (f)
{
}


Fonc_Num_Computed * Red_Op_Neigh_Not_Comp::compute
                   (const Arg_Fonc_Num_Comp & arg)

{
   Neigh_Rel_Compute * n = _n.compute(Arg_Neigh_Rel_Comp(arg.flux(),false));
   Flux_Pts_Computed *flx_inter
                              = flx_interface
                                (   arg.flux()->dim(),
                                    n->type_pack(),
                                    n->sz_buf()
                                );

    Fonc_Num_Computed * f = _f.compute(Arg_Fonc_Num_Comp(flx_inter));

    if (f->type_out() ==  Pack_Of_Pts::integer)
       return new Red_Op_Neigh_Comp<INT>(arg,n,f,flx_inter,_op);
    else
       return new Red_Op_Neigh_Comp<REAL>(arg,n,f,flx_inter,_op);

}

Fonc_Num Neigh_Rel::red_op(const OperAssocMixte& op,Fonc_Num f)
{
     return new Red_Op_Neigh_Not_Comp(op,*this,f);
}

Fonc_Num Neigh_Rel::red_sum (Fonc_Num f)
{
    return red_op(OpSum,f);
}

Fonc_Num Neigh_Rel::red_max (Fonc_Num f)
{
    return red_op(OpMax,f);
}

Fonc_Num Neigh_Rel::red_min (Fonc_Num f)
{
    return red_op(OpMin,f);
}

Fonc_Num Neigh_Rel::red_mul (Fonc_Num f)
{
    return red_op(OpMul,f);
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
