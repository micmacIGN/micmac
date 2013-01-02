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
/****************                   **********************************/
/****************    DILATATION     **********************************/
/****************                   **********************************/
/*********************************************************************/
/*********************************************************************/
/*********************************************************************/


/*********************************************************************/
/*                                                                   */
/*         Dilat_Compute                                             */
/*                                                                   */
/*********************************************************************/

class Dilat_Compute : public Flux_Pts_Computed
{
   public :
      Dilat_Compute(Flux_Pts_Computed *,Neigh_Rel_Compute *);

      ~Dilat_Compute()
      {
            delete _nrel;
            delete _flx;
      }

      virtual const Pack_Of_Pts * next(void)
      {
          if (_num_neigh_cur == _nb_neigh)
          {
             _last_pck = _flx->next();
             if (! _last_pck)
                return 0;
              _num_neigh_cur = 0;
          }

          const Pack_Of_Pts * res = 
               _nrel->neigh_in_num_dir(_last_pck,(char **)0,_num_neigh_cur);

          return res;
      }


      Flux_Pts_Computed * _flx;
      Neigh_Rel_Compute * _nrel;
      INT                _num_neigh_cur;
      INT                _nb_neigh;

   private :
         const Pack_Of_Pts *      _last_pck;
      
};


Dilat_Compute::Dilat_Compute(Flux_Pts_Computed * flx,Neigh_Rel_Compute * nrel) :
       Flux_Pts_Computed
       (   flx->dim(),
           nrel->type_pack(),
           nrel->sz_buf()
        ),
       _flx           (flx),
       _nrel          (nrel),
       _num_neigh_cur (nrel->nb_neigh()),
       _nb_neigh      (nrel->nb_neigh()),
       _last_pck      (0)
{
}





/*********************************************************************/
/*                                                                   */
/*         Conc_Compute                                              */
/*                                                                   */
/*********************************************************************/

class Conc_Compute :  public Dilat_Compute
{
      public :

          Conc_Compute(Flux_Pts_Computed * flx,Neigh_Rel_Compute * nrel,bool refl,INT nb_step_max);

           ~Conc_Compute()
           {
                 delete _next_set;
                 delete _curent_set;
                 delete _curser;
           }

      private :

           virtual const Pack_Of_Pts * next(void);
           INT             _num_gen;
           Curser_on_PoP * _curser;
           Std_Pack_Of_Pts<INT> * _last_pck;
           Std_Pack_Of_Pts<INT> * _curent_set;
           Std_Pack_Of_Pts<INT> * _next_set;
           bool                   _reflexif;
           INT                    _nb_step_max;
};


Conc_Compute::Conc_Compute(Flux_Pts_Computed * flx,Neigh_Rel_Compute * nrel,bool refl,INT nb_step_max):
     Dilat_Compute(flx,nrel),
     _num_gen      (0),
     _curser       (Curser_on_PoP::new_curs
                        (flx->dim(),flx->sz_buf(),nrel->type_pack())
                   ),
     _curent_set   (Std_Pack_Of_Pts<INT>::new_pck(flx->dim(),flx->sz_buf())),
     _next_set     (Std_Pack_Of_Pts<INT>::new_pck(flx->dim(),flx->sz_buf())),
     _reflexif     (refl),
     _nb_step_max  (nb_step_max)
{
    if (_reflexif)
    {
        _num_neigh_cur++;
        _nb_neigh++;
    }
}


const Pack_Of_Pts * Conc_Compute::next(void)
{
   for(;;)
   {
      if (_num_neigh_cur == _nb_neigh)
      {
          _last_pck  = 
                 (_num_gen == 0)                                     ?
                 SAFE_DYNC(Std_Pack_Of_Pts<INT> *,const_cast<Pack_Of_Pts *>( _flx->next()))     :
                 SAFE_DYNC(Std_Pack_Of_Pts<INT> *,const_cast<Pack_Of_Pts *>( _curser->next()))  ;

          if (! _last_pck)
          {
              if ((_num_gen==0) && _reflexif)
              {
                   _num_neigh_cur--;
                   _nb_neigh--;
                   _nrel->set_reflexif(false);
              }
              _num_gen++;
              ElSwap(_curent_set,_next_set);
              if ((! _curent_set->nb()) || (_num_gen == _nb_step_max))
                 return 0;
              _curser->re_start(_curent_set);
              _next_set->set_nb(0);
          }
          else
              _num_neigh_cur = 0;
      }
      else
      {
           Std_Pack_Of_Pts<INT> * res  =
                SAFE_DYNC
                (        Std_Pack_Of_Pts<INT> *, 
                          const_cast<Pack_Of_Pts *>
                          (
                               _nrel->neigh_in_num_dir
                               (_last_pck,(char **) 0,_num_neigh_cur)
                          )
                );
           bool chang;
           _next_set =
              res->cat_and_grow(_next_set,2*_next_set->pck_sz_buf(),chang);
           return res;

      }
  }
}

/*********************************************************************/
/*                                                                   */
/*         Conc_Compute                                              */
/*                                                                   */
/*********************************************************************/

class DilConc_Not_Comp : public Flux_Pts_Not_Comp
{
    public :

        DilConc_Not_Comp(Flux_Pts,Neigh_Rel,bool,bool,int nb_step_max = 1000000000);

    private :

       Flux_Pts     _flx;
       Neigh_Rel    _nrel;
       bool         _dil;
       bool         _reflex;
       INT          _nb_step_max;

        Flux_Pts_Computed * compute(const Arg_Flux_Pts_Comp & arg)
        {
             Flux_Pts_Computed * flx =   _flx.compute(arg);
             Neigh_Rel_Compute * neigh = _nrel.compute(Arg_Neigh_Rel_Comp(flx,_reflex));

             Flux_Pts_Computed * res = 
                 _dil                                     ?
                     new Dilat_Compute(flx,neigh)         :
                     new Conc_Compute(flx,neigh,_reflex,_nb_step_max)  ;

             return split_to_max_buf(res,arg);
       }
};


DilConc_Not_Comp::DilConc_Not_Comp(Flux_Pts flx,Neigh_Rel nrel,bool dil,bool reflex,int nb_step_max) :
     _flx         (flx) ,
     _nrel        (nrel),
     _dil         (dil),
     _reflex      (reflex),
     _nb_step_max (nb_step_max)
{
}


Flux_Pts dilate(Flux_Pts flx,Neigh_Rel nrel)
{
    return new DilConc_Not_Comp(flx,nrel,true,false);
}

Flux_Pts conc(Flux_Pts flx,Neigh_Rel nrel,INT nb_step_max)
{
    return new DilConc_Not_Comp(flx,nrel,false,true,nb_step_max);
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
