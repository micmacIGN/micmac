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
/*         Flaged_Neigh_Rel_Comp                                     */
/*                                                                   */
/*********************************************************************/

class Flaged_Neigh_Rel_Comp : public Neigh_Rel_Compute
{
  public :

      Flaged_Neigh_Rel_Comp
      (
           const Arg_Neigh_Rel_Comp &,
           Neigh_Rel_Compute *,
           Fonc_Num_Computed *
      );

     virtual ~Flaged_Neigh_Rel_Comp()
     {
         DELETE_VECTOR(_sel,0);
         DELETE_MATRICE_ORI(_is_neigh,_sbuf.x,_sbuf.y);
         delete _f;
         delete _n;
     }

  private :

      virtual const Pack_Of_Pts * neigh_in_num_dir
                           ( const Pack_Of_Pts * pack_0,
                             char ** is_neigh,
                             INT & num_dir
                           );
 
      const Std_Pack_Of_Pts<INT> * _val_flag;
      Neigh_Rel_Compute *    _n;
      Fonc_Num_Computed *    _f;
      Pt2di                  _sbuf;
      char **                _is_neigh;
      INT  *                 _sel;


      void set_reflexif(bool refl){_n->set_reflexif(refl);}
};


Flaged_Neigh_Rel_Comp::Flaged_Neigh_Rel_Comp
      (
           const Arg_Neigh_Rel_Comp & arg,
           Neigh_Rel_Compute * n,
           Fonc_Num_Computed * f
      ) :
       Neigh_Rel_Compute
       (  arg,
          n->neigh(),
         (arg.flux()->type() == Pack_Of_Pts::real)     ?
                                 Pack_Of_Pts::real     :
                                 Pack_Of_Pts::integer  ,
         n->sz_buf()            
     ),
     _val_flag  (0),
     _n         (n),
     _f         (f),
     _sbuf      (arg.flux()->sz_buf(),n->nb_neigh()+1),
     _is_neigh  (NEW_MATRICE_ORI(_sbuf.x,_sbuf.y,char)),
     _sel       (NEW_VECTEUR(0,_sbuf.x*_sbuf.y,INT))
{
}

const Pack_Of_Pts * Flaged_Neigh_Rel_Comp::neigh_in_num_dir
                    ( 
                         const Pack_Of_Pts *  pack_0,
                         char **              is_neigh_sup,
                         INT &                num_dir
                    )
{
   if (num_dir == 0)
       _val_flag = _f->values(pack_0)->int_cast();

   char ** is_neigh = (is_neigh_sup != 0) ? is_neigh_sup : _is_neigh;

   INT d0 = num_dir;
   INT nb_pts = pack_0->nb();
   const Pack_Of_Pts * pts_tr = 
        _n->neigh_in_num_dir(pack_0,is_neigh,num_dir);


   INT * flag = _val_flag->_pts[0];
   INT nbcur = 0;
   for (INT d = d0; d<num_dir ; d++)
   {
        INT fld = (1<<d);
        char * isn = is_neigh[d];

        for (INT ip =0; ip <nb_pts ; ip++)
        {
            int flp = (flag[ip] &  fld) != 0;
            if (isn[ip])
               _sel[nbcur++] = flp;
            isn[ip] = isn[ip] & flp;
        }
   }


   El_Internal.ElAssert
   (  
          nbcur == pts_tr->nb(),
          EEM0 << "Bad assertion in"
               << " Flaged_Neigh_Rel_Comp::neigh_in_num_dir"
   );

                      
    _pack->set_nb(0);
    pts_tr->select_tab(_pack,_sel);
    return _pack;
}

/*********************************************************************/
/*                                                                   */
/*         Flaged_Neigh_Rel_Not_Comp                                 */
/*                                                                   */
/*********************************************************************/

class Flaged_Neigh_Rel_Not_Comp : public Neigh_Rel_Not_Comp
{
   public :
       Flaged_Neigh_Rel_Not_Comp(Neigh_Rel,Fonc_Num);

   private :

      Neigh_Rel_Compute * compute (const Arg_Neigh_Rel_Comp & arg)
      {
            Neigh_Rel_Compute * n  = _n.compute(arg);
            Fonc_Num_Computed * f =  _f.compute(Arg_Fonc_Num_Comp(arg.flux()));

            Tjs_El_User.ElAssert
            (
                 f->idim_out() == 1,
                 EEM0 << "dim out should be 1 for neighbooring filter"
                      <<"(got dim=" << f->idim_out() << ")\n"
            );

            return new Flaged_Neigh_Rel_Comp(arg,n,f);
      };

      Neigh_Rel   _n;
      Fonc_Num    _f;
};


Flaged_Neigh_Rel_Not_Comp::Flaged_Neigh_Rel_Not_Comp 
        (Neigh_Rel n,Fonc_Num f) :
        _n (n),
        _f (Iconv(f))
{
}


Neigh_Rel sel_flag(Neigh_Rel n,Fonc_Num f)
{
     return new Flaged_Neigh_Rel_Not_Comp(n,f);
}




/*********************************************************************/
/*                                                                   */
/*         Sel_Func_Neigh_Rel_Comp                                   */
/*                                                                   */
/*********************************************************************/

class Sel_Func_Neigh_Rel_Comp : public Neigh_Rel_Compute
{
  public :

      Sel_Func_Neigh_Rel_Comp
      (
           const Arg_Neigh_Rel_Comp &,
           Neigh_Rel_Compute *,
           Fonc_Num_Computed *,
           Flux_Pts_Computed *flx_interf
      );

     virtual ~Sel_Func_Neigh_Rel_Comp()
     {
         delete _f;
         delete _flx_interf;
         delete _n;
     }

  private :

      virtual const Pack_Of_Pts * neigh_in_num_dir
                           ( const Pack_Of_Pts * pack_0,
                             char ** is_neigh,
                             INT & num_dir
                           );
 
      Neigh_Rel_Compute * _n;
      Fonc_Num_Computed * _f;
      Flux_Pts_Computed *_flx_interf;


      void set_reflexif(bool refl){_n->set_reflexif(refl);}

};

const Pack_Of_Pts * Sel_Func_Neigh_Rel_Comp::neigh_in_num_dir
                    ( 
                         const Pack_Of_Pts *  pack_0,
                         char **              is_neigh,
                         INT &                num_dir
                    )
{
   INT d0 = num_dir;
   INT nb_pts = pack_0->nb();
   const Pack_Of_Pts * pts_tr = 
        _n->neigh_in_num_dir(pack_0,is_neigh,num_dir);

   Std_Pack_Of_Pts<INT> * val =
          SAFE_DYNC(Std_Pack_Of_Pts<INT> *,const_cast<Pack_Of_Pts *>(_f->values(pts_tr)));

   if (is_neigh)
   {
       INT * v = val->_pts[0];
       INT nbcur = 0;
       for (INT d = d0; d<num_dir ; d++)
       {
           for (INT ip =0; ip <nb_pts ; ip++)
               if (is_neigh[d][ip])
                   is_neigh[d][ip] = (v[nbcur++] != 0);
       }
       El_Internal.ElAssert
       (  
          nbcur == pts_tr->nb(),
          EEM0 << "Bad assertion in"
               << " Sel_Func_Neigh_Rel_Comp::neigh_in_num_dir"
       );
   }

                      
    _pack->set_nb(0);
    pts_tr->select_tab(_pack,val->_pts[0]);
    return _pack;
}

Sel_Func_Neigh_Rel_Comp::Sel_Func_Neigh_Rel_Comp
      (
           const Arg_Neigh_Rel_Comp & arg,
           Neigh_Rel_Compute * n,
           Fonc_Num_Computed * f,
           Flux_Pts_Computed * flx_interf
      ) :
       Neigh_Rel_Compute
      (  arg,
         n->neigh(),
         (arg.flux()->type() == Pack_Of_Pts::real)     ?
                                 Pack_Of_Pts::real     :
                                 Pack_Of_Pts::integer  ,
         n->sz_buf()            
     ),
     _n (n),
     _f (f),
     _flx_interf (flx_interf)
{
}



/*********************************************************************/
/*                                                                   */
/*         Sel_Func_Neigh_Rel_Not_Comp                               */
/*                                                                   */
/*********************************************************************/

class Sel_Func_Neigh_Rel_Not_Comp : public Neigh_Rel_Not_Comp
{
   public :
       Sel_Func_Neigh_Rel_Not_Comp(Neigh_Rel,Fonc_Num);

   private :

      Neigh_Rel_Compute * compute 
             (const Arg_Neigh_Rel_Comp & arg)
      {
            Neigh_Rel_Compute * n  = _n.compute(arg);
  
            Flux_Pts_Computed *flx_inter 
                              = flx_interface
                                (   arg.flux()->dim(),
                                    n->type_pack(),
                                    n->sz_buf()
                                );

            Fonc_Num_Computed * f;
            Arg_Fonc_Num_Comp arg_fonc = Arg_Fonc_Num_Comp(flx_inter);
            f= _f.compute(arg_fonc);

            Tjs_El_User.ElAssert
            (
                 f->idim_out() == 1,
                 EEM0 << "dim out should be 1 for neighbooring filter"
                      <<"(got dim=" << f->idim_out() << ")\n"
            );

           f= convert_fonc_num(arg_fonc,f,flx_inter,Pack_Of_Pts::integer);

           return new Sel_Func_Neigh_Rel_Comp(arg,n,f,flx_inter);
      };

   // -----------------------------data----------------------

        Neigh_Rel   _n;
        Fonc_Num    _f;
};


Sel_Func_Neigh_Rel_Not_Comp::Sel_Func_Neigh_Rel_Not_Comp 
        (Neigh_Rel n,Fonc_Num f) :
        _n (n),
        _f (f)
{
}


Neigh_Rel sel_func(Neigh_Rel n,Fonc_Num f)
{
     return new Sel_Func_Neigh_Rel_Not_Comp(n,f);
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
