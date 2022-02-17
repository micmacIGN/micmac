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



/***********************************************************************/
/*                                                                     */
/*       Data_Liste_Pts_Gen                                            */
/*                                                                     */
/***********************************************************************/



Data_Liste_Pts_Gen::~Data_Liste_Pts_Gen(){}

Data_Liste_Pts_Gen::Data_Liste_Pts_Gen(const DataGenIm * gi,INT dim) :
       _gi       (gi),
       _dim      (dim),
       _nb_last  (0),
       _sz_el    (gi->sz_el()),
       _sz_base  (gi->sz_base_el()),
       _nb_by_el (el_liste_pts::Elise_SZ_buf_lpts / (dim * gi->sz_el())),
       _free_for_out (true),
       _first     (new el_liste_pts()),
       _is_copie  (false)
{
      _last = _first;

      Tjs_El_User.ElAssert
      (  (dim <= Elise_Std_Max_Dim),
         EEM0 << "in Elise, dimension of List of points must be <= "
              << (int) Elise_Std_Max_Dim << "\n"
              << "|         requested dimension : " << dim
      );

};


void Data_Liste_Pts_Gen::free_el()
{
    while(_first != _last)
    {
         el_liste_pts * before = _first;
         _first = _first->next;
         delete before;
    }
    delete _last;
}


bool   Data_Liste_Pts_Gen::empty() const
{
   return (_first == _last) && (_nb_last == 0);
}

INT   Data_Liste_Pts_Gen::card() const
{
   INT nb = 0;

   for (el_liste_pts * el = _first; el != _last; el = el->next)
       nb ++;

   return nb * _nb_by_el + _nb_last;
}

INT   Data_Liste_Pts_Gen::dim() const
{
    return _dim;
}




void Data_Liste_Pts_Gen::cat_pts(char ** c,INT nb_tot)
{
    char * coord[Elise_Std_Max_Dim];

    for (int j = 0; j < _dim ; j++)
         coord[j] = c[j];
    while (nb_tot)
    {

         if (_nb_last == _nb_by_el)
         {
             _last->next = new el_liste_pts();
             _last = _last->next;
             _nb_last = 0;
         }
         INT nb = ElMin(nb_tot,_nb_by_el-_nb_last);


         _gi->striped_output_rle
         (
             _last->buf(), // +_dim*_nb_last*_sz_el,
             nb,
             _dim,
             coord,
             _nb_last
          );


         for (int j = 0; j < _dim ; j++)
             coord[j] +=  nb * _sz_base;
         _nb_last += nb;
         nb_tot -= nb;
    }
}






INT Data_Liste_Pts_Gen::next_pts(char ** c,INT nb_max)
{

     ASSERT_INTERNAL
     (   nb_max >= _nb_by_el,
         "Incoherence in data_Liste_Pts_Gen::next_pts"
     );

    char * coord[Elise_Std_Max_Dim];

    for (int j = 0; j < _dim ; j++)
         coord[j] = c[j];

    INT nb = 0;

    for (;;)
    {
         INT delta_nb = ((_first == _last) ? _nb_last : _nb_by_el);

         if ((nb+delta_nb > nb_max) || (! _first))
             return nb;

         _gi->striped_input_rle
         (
             coord,
             delta_nb,
             _dim,
             _first->buf(),
             0
          );
          nb += delta_nb;

         for (int j = 0; j < _dim ; j++)
             coord[j] +=  delta_nb * _sz_base;

         _first = (_first != _last) ? _first->next : 0;
    }
}


/***********************************************************************/
/*                                                                     */
/*       Liste_Pts_Out                                                 */
/*                                                                     */
/***********************************************************************/

class Liste_Pts_Out_Comp : public Output_Computed
{
      Data_Liste_Pts_Gen *  _lpts;  // a copy to allow simultanate flux and out

       virtual void update
       (
                const Pack_Of_Pts * pts,
                const Pack_Of_Pts *
       )
       {
            _lpts->cat_pts
            (
               C_CAST(char **,pts->adr_coord()),
               pts->nb()
            );
       }

   public :

       virtual ~Liste_Pts_Out_Comp() {_lpts->_free_for_out = true;}

       Liste_Pts_Out_Comp(Data_Liste_Pts_Gen *);
};


Liste_Pts_Out_Comp::Liste_Pts_Out_Comp(Data_Liste_Pts_Gen * pl) :
        Output_Computed   (0),
        _lpts             (pl)
{
}


class Liste_Pts_Out_Not_Comp : public Output_Not_Comp
{
      Liste_Pts_Gen _l;
      Data_Liste_Pts_Gen * _lpts;


      Output_Computed * compute(const Arg_Output_Comp & arg)
      {
         Tjs_El_User.ElAssert
         (
            _lpts->_dim == arg.flux()->dim(),
            EEM0
               << "Writing in Liste_Pts : dim of flux should equal dim of liste\n"
               <<"|   dim of flux : "  << arg.flux()->dim()
               << " , dim of liste : " << _lpts->_dim
         );

            Tjs_El_User.ElAssert
            ( _lpts->_free_for_out,
              EEM0 << "the liste of point, used as output, was locked \n"
                   << "|   (probably smth like : \n"
                   << "|          copy(l.all_pts(),1,l)   or  \n"
                   << "|          copy(flux,1,l|l)               )"
            );
            _lpts->_free_for_out = false;

            Output_Computed * res = new Liste_Pts_Out_Comp(_lpts);

            return out_adapt_type_pts
                   (
                       arg,
                       res,
                       ( _lpts->_gi->integral_type()   ?
                          Pack_Of_Pts::integer         :
                          Pack_Of_Pts::real
                       )
                   );
      };

    public :
      Liste_Pts_Out_Not_Comp(Liste_Pts_Gen,Data_Liste_Pts_Gen *);
};


Liste_Pts_Out_Not_Comp::Liste_Pts_Out_Not_Comp
         (Liste_Pts_Gen l,Data_Liste_Pts_Gen * lpts) :
              _l    (l),
              _lpts (lpts)
{
}

Output::Output(Liste_Pts_Gen l) :
     PRC0
     (  new Liste_Pts_Out_Not_Comp
            (l,SAFE_DYNC(Data_Liste_Pts_Gen *,l._ptr))
     )
{
}

/***********************************************************************/
/*                                                                     */
/*       Liste_Pts_in_Comp                                             */
/*                                                                     */
/***********************************************************************/

template <class Type> class Liste_Pts_in_Comp : public Std_Flux_Of_Points<Type>
{
      Data_Liste_Pts_Gen   _lpts;
      Data_Liste_Pts_Gen   * _lpts_init;

      const Pack_Of_Pts * next()
      {
           INT nb = _lpts.next_pts(C_CAST(char **,this->_pack->_pts),this->sz_buf());
           this->_pack->set_nb(nb);
           return nb   ? this->_pack : 0;
      }

   public:
      virtual ~Liste_Pts_in_Comp()
      {
          _lpts_init->_free_for_out = true;
      }

      Liste_Pts_in_Comp(Data_Liste_Pts_Gen *,const Arg_Flux_Pts_Comp &);
};

template <class Type> Liste_Pts_in_Comp<Type>::Liste_Pts_in_Comp
                          (Data_Liste_Pts_Gen *lpts,const Arg_Flux_Pts_Comp & arg) :
             Std_Flux_Of_Points<Type>(lpts->_dim,arg.sz_buf()),
            _lpts (*lpts),
            _lpts_init (lpts)
{
}

class Liste_Pts_in_Not_Comp : public Flux_Pts_Not_Comp
{
      Liste_Pts_Gen _l;
      Data_Liste_Pts_Gen *_lpts;


      Flux_Pts_Computed * compute(const Arg_Flux_Pts_Comp & arg)
      {

            Tjs_El_User.ElAssert
            ( _lpts->_free_for_out,
              EEM0 << "the liste of point, used as flux, was locked \n"
            );

            _lpts->_free_for_out = false;

            if( _lpts->_gi->integral_type())
               return  new Liste_Pts_in_Comp<INT>(_lpts,arg);
            else
               return  new Liste_Pts_in_Comp<REAL>(_lpts,arg);
      }


  public :
      Liste_Pts_in_Not_Comp(Liste_Pts_Gen,Data_Liste_Pts_Gen *);
};

Liste_Pts_in_Not_Comp::Liste_Pts_in_Not_Comp
     (Liste_Pts_Gen l,Data_Liste_Pts_Gen * lpts) :
           _l     (l),
           _lpts  (lpts)
{
}

/***********************************************************************/
/*                                                                     */
/*        Data_Liste_Pts<Type,TyBase>                                  */
/*                                                                     */
/***********************************************************************/

template  <class Type,class TyBase>
            void Data_Liste_Pts<Type,TyBase>::add_pt(Type * pt)
{
     if (_nb_last == _nb_by_el)
     {
        _last->next = new el_liste_pts();
        _last = _last->next;
        _nb_last = 0;
     }

     memcpy((reinterpret_cast<Type *>(_last->buf()))+_dim*_nb_last,pt,_dim*sizeof(Type));
     _nb_last ++;
}



template  <class Type,class TyBase>
            Data_Liste_Pts<Type,TyBase>::Data_Liste_Pts
                 (INT dim)   :

             Data_Liste_Pts_Gen( &DataIm1D<Type,TyBase>::The_Bitm,dim)
{
}

template <class Type,class TyBase> Data_Liste_Pts<Type,TyBase>::~Data_Liste_Pts()
{
   if (! _is_copie)
      free_el();
}

template <class Type,class TyBase>
         Im2D<Type,TyBase> Data_Liste_Pts<Type,TyBase>::image()
{
     auto dup = *this;
     dup._is_copie = true;
     INT nb = card();
     Im2D<Type,TyBase> res (nb,_dim);
     Im2D<TyBase,TyBase> tmp (_nb_by_el,_dim);

     TyBase ** _tdata = tmp.data();
     Type   ** _rdata = res.data();

     for (INT ipt=0; ipt<nb; ipt+= _nb_by_el)
     {
         INT nb_loc = ElMin(nb-ipt,_nb_by_el);
         dup.next_pts((char **) tmp.data(),ElMax(nb_loc,_nb_by_el));
         for (INT d=0; d<_dim; d++)
             convert(_rdata[d]+ipt,_tdata[d],nb_loc);
     }

     return res;
}


/*
template  <class Type,class TyBase>
            Data_Liste_Pts<Type,TyBase>::Data_Liste_Pts
                 (INT dim)   :

             Data_Liste_Pts_Gen( &DataIm1D<Type,TyBase>::The_Bitm,dim)
{
}
*/

/***********************************************************************/
/*                                                                     */
/*                  Liste_Pts<Type,TyBase>                             */
/*                                                                     */
/***********************************************************************/

template  <class Type,class TyBase>
            void Liste_Pts<Type,TyBase>::add_pt(Type * pt)
{
     // ((Data_Liste_Pts<Type,TyBase> *)_ptr)->add_pt(pt);
    dlpt()->add_pt(pt);
}

template <class Type,class TyBase>
         Data_Liste_Pts<Type,TyBase>  * Liste_Pts<Type,TyBase>::dlpt()  const
{
   return (Data_Liste_Pts<Type,TyBase> *)_ptr;
}


template <class Type,class TyBase> Liste_Pts<Type,TyBase>::Liste_Pts(INT dim) :
          Liste_Pts_Gen ( new Data_Liste_Pts<Type,TyBase> (dim))
{
}

template <class Type,class TyBase>
         Im2D<Type,TyBase> Liste_Pts<Type,TyBase>::image() const
{
/*
   return ((Data_Liste_Pts<Type,TyBase> *)_ptr)->image();
*/
   Data_Liste_Pts<Type,TyBase> * p = dlpt();
   return p->image();
}


template <class Type,class TyBase> Liste_Pts<Type,TyBase>::Liste_Pts(INT dim,Type ** data,INT nb) :
          Liste_Pts_Gen ( new Data_Liste_Pts<Type,TyBase> (dim))
{
    Data_Liste_Pts<Type,TyBase> * p = dlpt();
    INT nb_by_el = p->_nb_by_el;

    Im2D<TyBase,TyBase> I(nb_by_el,dim);

    for (INT k =0; k<nb ; k+=nb_by_el)
    {
        INT nb_loc = ElMin(nb_by_el,nb-k);
        for (INT d =0; d<dim ; d++)
            convert(I.data()[d],data[d]+k,nb_loc);
        p->cat_pts (C_CAST(char **,I.data()),nb_loc);
    }
}


template <class Type,class TyBase> Liste_Pts<Type,TyBase>::Liste_Pts(Type *x,Type * y,INT nb) :
          Liste_Pts_Gen (0)
{
   Type * d[2];
    d[0] = x;
    d[1] = y;

   *this =  Liste_Pts<Type,TyBase>(2,d,nb);
}


template class Liste_Pts<U_INT1,INT>;

template class Liste_Pts<INT1,INT>;

#if ElTemplateInstantiation
   template class Data_Liste_Pts<U_INT1,INT>;
   template class Data_Liste_Pts<INT1,INT>;

   template class Data_Liste_Pts<U_INT2,INT>;
   template class Liste_Pts<U_INT2,INT>;

   template class Data_Liste_Pts<INT2,INT>;
   template class Liste_Pts<INT2,INT>;

   template class Data_Liste_Pts<INT4,INT>;
   template class Liste_Pts<INT4,INT>;

   template class Data_Liste_Pts<REAL4,REAL8>;
   template class Data_Liste_Pts<REAL8,REAL8>;
#endif  // ElTemplateInstantiation

template class Liste_Pts<REAL4,REAL8>;

template class Liste_Pts<REAL8,REAL8>;


/***********************************************************************/
/*                                                                     */
/*       Liste_Pts_Gen                                                 */
/*                                                                     */
/***********************************************************************/

Liste_Pts_Gen::Liste_Pts_Gen(Data_Liste_Pts_Gen * lpts) :
      PRC0(lpts)
{
}


Flux_Pts  Liste_Pts_Gen::all_pts()
{
     return new Liste_Pts_in_Not_Comp
                (*this,SAFE_DYNC(Data_Liste_Pts_Gen *,_ptr));
}


bool Liste_Pts_Gen::empty() const
{
   return SAFE_DYNC(Data_Liste_Pts_Gen *,_ptr)->empty();
}


INT Liste_Pts_Gen::card() const
{
   return SAFE_DYNC(Data_Liste_Pts_Gen *,_ptr)->card();
}


INT Liste_Pts_Gen::dim() const
{
   return SAFE_DYNC(Data_Liste_Pts_Gen *,_ptr)->dim();
}


/* Footer-MicMac-eLiSe-25/06/2007

   Ce logiciel est un programme informatique servant a  la mise en
   correspondances d'images pour la reconstruction du relief.

   Ce logiciel est regi par la licence CeCILL-B soumise au droit francais et
   respectant les principes de diffusion des logiciels libres. Vous pouvez
   utiliser, modifier et/ou redistribuer ce programme sous les conditions
   de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
   sur le site "http://www.cecill.info".

   En contrepartie de l'accessibilite au code source et des droits de copie,
   de modification et de redistribution accordes par cette licence, il n'est
   offert aux utilisateurs qu'une garantie limitee.  Pour les memes raisons,
   seule une responsabilite restreinte pese sur l'auteur du programme,  le
   titulaire des droits patrimoniaux et les concedants successifs.

   A cet egard  l'attention de l'utilisateur est attiree sur les risques
   associes au chargement, a l'utilisation, a la modification et/ou au
   developpement et a la reproduction du logiciel par l'utilisateur etant
   donne sa specificite de logiciel libre, qui peut le rendre complexe a
   manipuler et qui le reserve donc a des developpeurs et des professionnels
   avertis possedant  des  connaissances  informatiques approfondies.  Les
   utilisateurs sont donc invites a charger  et  tester  l'adequation  du
   logiciel a leurs besoins dans des conditions permettant d'assurer la
   securite de leurs systemes et ou de leurs donnees et, plus generalement,
   a l'utiliser et l'exploiter dans les memes conditions de securite.

   Le fait que vous puissiez acceder a cet en-tete signifie que vous avez
   pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
   termes.
   Footer-MicMac-eLiSe-25/06/2007/*/
