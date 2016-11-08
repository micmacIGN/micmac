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
/*         Std_Flux_Of_Points<Type>                                  */
/*                                                                   */
/*********************************************************************/


template <class Type> Std_Flux_Of_Points<Type>::Std_Flux_Of_Points
                      (INT dim,INT sz_buf,bool sz_buf_0 ) :

       Flux_Pts_Computed (dim,Std_Flux_Of_Points<Type>::_type_pack,sz_buf),
       _pack (Std_Pack_Of_Pts<Type>::new_pck(dim,sz_buf_0 ? 0 : sz_buf))
{
}

template <class Type> Std_Flux_Of_Points<Type>::~Std_Flux_Of_Points()
{
     delete _pack;
}


template <> const Pack_Of_Pts::type_pack Std_Flux_Of_Points<INT>::_type_pack = Pack_Of_Pts::integer;
template <> const Pack_Of_Pts::type_pack Std_Flux_Of_Points<REAL>::_type_pack = Pack_Of_Pts::real;




/*********************************************************************/
/*                                                                   */
/*         Std_Flux_Interface<Type>                                  */
/*                                                                   */
/*********************************************************************/

template <class Type> Std_Flux_Interface<Type>::Std_Flux_Interface(INT dim,INT sz_buf) :
        Std_Flux_Of_Points<Type>(dim,sz_buf,true)
{
}



template <class Type> const Pack_Of_Pts * Std_Flux_Interface<Type>::next()
{
     elise_fatal_error
    (  "should never be her :Std_Flux_Interface<Type>::next()",
       __FILE__,__LINE__);

    return 0;
}

#if ElTemplateInstantiation
    template class Std_Flux_Of_Points<INT>;
    template class Std_Flux_Of_Points<REAL>;
#endif

template class Std_Flux_Interface<INT>;
template class Std_Flux_Interface<REAL>;

/*********************************************************************/
/*                                                                   */
/*         RLE_Flux_Interface                                        */
/*                                                                   */
/*********************************************************************/


RLE_Flux_Interface::RLE_Flux_Interface(INT dim,INT sz_buf) :
        RLE_Flux_Pts_Computed(dim,sz_buf)
{
}



const Pack_Of_Pts * RLE_Flux_Interface::next()
{
     elise_fatal_error
    (  "should never be here RLE_Flux_Interface::next()",
       __FILE__,__LINE__);

    return 0;
}


Flux_Pts_Computed * flx_interface(INT dim, Pack_Of_Pts::type_pack type,INT sz_buf)
{
      switch (type)
      {
          case  Pack_Of_Pts::rle :
                return new RLE_Flux_Interface (dim,sz_buf);

          case  Pack_Of_Pts::integer :
                return new Std_Flux_Interface<INT>(dim,sz_buf);

          case  Pack_Of_Pts::real :
                return new Std_Flux_Interface<REAL>(dim,sz_buf);

          default :
               elise_fatal_error("incoherence in flx_interface",__FILE__,__LINE__);
      }
      return 0;
}

Flux_Pts_Computed * flx_interface(Flux_Pts_Computed * flx)
{
    return  flx_interface(flx->dim(),flx->type(),flx->sz_buf());
}

Flux_Pts_Computed * interface_flx_chc(Flux_Pts_Computed * flx,Fonc_Num_Computed * f)
{
    return  flx_interface(f->idim_out(),f->type_out(),flx->sz_buf());
}

Flux_Pts_Computed * tr_flx_interface(Flux_Pts_Computed * flx,INT * tr_coord)
{
      Box2di b(Pt2di(0,0),Pt2di(0,0));

      if (flx->is_rect_2d(b))
      {
          Pt2di  tr (tr_coord[0],tr_coord[1]);

          return RLE_Flux_Pts_Computed::rect_2d_interface
                 (
                     b._p0 + tr,
                     b._p1 + tr,
                     flx->sz_buf()
                 );
      }
      else
          return flx_interface(flx);
}

/*********************************************************************/
/*                                                                   */
/*         RLE SELECT                                                */
/*                                                                   */
/*********************************************************************/

class select_RLE_Computed : public  Flux_Pts_Computed
{
    public :
       select_RLE_Computed(Flux_Pts_Computed *,Fonc_Num_Computed *,bool rebuf);
       const Pack_Of_Pts * next();
       virtual ~select_RLE_Computed(void);

    private :
        Flux_Pts_Computed      * _flx;
        Fonc_Num_Computed      * _fonc;
        INT                    _delta_buf;
        static const INT       _fact_over_buf;
        bool                   _finished;
        Pack_Of_Pts *          _pack;
        bool                   _rebuf;
};

select_RLE_Computed::~select_RLE_Computed(void)
{
     delete _fonc;
     delete _flx;
     delete _pack;
}



const int select_RLE_Computed::_fact_over_buf = 3;

const Pack_Of_Pts * select_RLE_Computed::next()
{

    if (_rebuf)
    {
        if (_finished)
           return 0;;

        _pack->set_nb(0);
        while(_pack->nb() <= _delta_buf)
        {
             const Pack_Of_Pts * pack_gen = _flx->next();
             if (! pack_gen)
             {
                _finished = true;
                return _pack;
             }
             Std_Pack_Of_Pts<INT> *pack_vals  =
                   SAFE_DYNC(Std_Pack_Of_Pts<INT>  *,const_cast<Pack_Of_Pts *>(_fonc->values(pack_gen)));

             pack_gen->select_tab(_pack,pack_vals->_pts[0]);
        }
        return _pack;
   }
   else
   {
        const Pack_Of_Pts * pack_gen = _flx->next();
        if (! pack_gen)
           return 0;

        Std_Pack_Of_Pts<INT> *pack_vals  =
             SAFE_DYNC(Std_Pack_Of_Pts<INT>*,const_cast<Pack_Of_Pts *>(_fonc->values(pack_gen)));

        _pack->set_nb(0);
        pack_gen->select_tab(_pack,pack_vals->_pts[0]);
        return _pack;

   }
}



select_RLE_Computed::select_RLE_Computed
      (         Flux_Pts_Computed * flx,
                Fonc_Num_Computed * fonc,
                bool                rebuf
      )     :
              Flux_Pts_Computed
              (      flx->dim(),
                      (flx->type() == Pack_Of_Pts::real) ?
                                    Pack_Of_Pts::real  :
                                    Pack_Of_Pts::integer  ,
                      flx->sz_buf()*(rebuf ? _fact_over_buf : 1)
              )                ,
              _flx(flx)        ,
              _fonc (fonc)     ,
              _finished (false),
              _rebuf    (rebuf)
{
     if (flx->type() == Pack_Of_Pts::real)
        _pack =  Std_Pack_Of_Pts<REAL>::new_pck(flx->dim(),sz_buf());
     else
        _pack =  Std_Pack_Of_Pts<INT>::new_pck(flx->dim(),sz_buf());

    _delta_buf = sz_buf() - flx->sz_buf();
}

/*********************************************************************/
/*                                                                   */
/*         select_Not_Comp                                           */
/*                                                                   */
/*********************************************************************/


class select_Not_Comp : public  Flux_Pts_Not_Comp
{
      public :
          select_Not_Comp(Flux_Pts,Fonc_Num,bool rebuf_rle,bool rebuf_not_rle);
          Flux_Pts_Computed * compute(const Arg_Flux_Pts_Comp & arg);


      private :
          Flux_Pts  _flux;
          Fonc_Num  _fonc;
          bool _rebuf_rle;
          bool _rebuf_not_rle;
};



select_Not_Comp::select_Not_Comp(Flux_Pts flx,Fonc_Num fonc,bool rebuf_rle,bool rebuf_not_rle) :
    _flux             (flx),
    _fonc            (fonc),
   _rebuf_rle        (rebuf_rle),
   _rebuf_not_rle    (rebuf_not_rle)
{
}


Flux_Pts_Computed * select_Not_Comp::compute(const Arg_Flux_Pts_Comp & arg)
{
    Flux_Pts_Computed * flx;
    Fonc_Num_Computed * fonc;

    flx =  _flux.compute(arg);
    Arg_Fonc_Num_Comp arg_fonc = Arg_Fonc_Num_Comp(flx);
    fonc = _fonc.compute(arg_fonc);

    // accept REAL function; dangerous ? to reexamine latter
    fonc = convert_fonc_num(arg_fonc,fonc,flx,Pack_Of_Pts::integer);

    ASSERT_TJS_USER
    (     fonc->idim_out() == 1,
          "select can handle only 1_dimensional fonc"
    );

    bool rebuf = (flx->type() == Pack_Of_Pts::rle) ? _rebuf_rle : _rebuf_not_rle;
    Flux_Pts_Computed * res =
           new  select_RLE_Computed(flx,fonc,rebuf);

    return  rebuf                                      ?
            split_to_max_buf (res,Arg_Flux_Pts_Comp()) :
            res
    ;
}


Flux_Pts select(Flux_Pts flx,Fonc_Num fonc)
{
     return new select_Not_Comp(flx,fonc,true,false);
}


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
