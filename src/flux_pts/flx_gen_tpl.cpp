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
/*         Std_Pack_Of_Pts_Gen                                       */
/*                                                                   */
/*********************************************************************/


Std_Pack_Of_Pts_Gen::Std_Pack_Of_Pts_Gen(INT dim,INT sz_buf,type_pack type) :
     Pack_Of_Pts(dim,sz_buf,type)
{
}


/*********************************************************************/
/*                                                                   */
/*         Std_Pack_Of_Pts<Type>                                     */
/*                                                                   */
/*********************************************************************/


template <class Type> Std_Pack_Of_Pts<Type>::Std_Pack_Of_Pts
                      (INT dim,INT sz_buf)  :
       Std_Pack_Of_Pts_Gen(dim,sz_buf,type_glob)
{

    _tprov_ptr    =  AllocTprov<Type>::alloc_small_ptr_tprov(dim);
    _pts = _tprov_ptr->coord();

    if (sz_buf > 0)
    {
          _tprov_tprov  = AllocTprov<Type>::alloc_small_tprov_tprov(dim);
          Tab_Prov<Type> **  tpr = _tprov_tprov->coord();
          for (int i=0 ; i<dim ; i++)
          {
// cout << "AAA " << i << " " <<  sz_buf << "\n";
              tpr[i] =  AllocTprov<Type>::alloc_big_tprov(sz_buf);
// cout << "BBBB  \n";
              _pts[i] =  tpr[i]->coord();
          }
    }
    else
    {
         for (int i=0 ; i<dim ; i++)
            _pts[i] = 0;
        _tprov_tprov = 0;
    }
}

template <class Type> Std_Pack_Of_Pts<Type>::~Std_Pack_Of_Pts(void)
{

    if(_tprov_tprov)
    {
        Tab_Prov<Type> **  tpr = _tprov_tprov->coord();
        for (int i=_dim-1 ; i>=0 ; i--)
             delete tpr[i];

        delete _tprov_tprov;
    }
    delete _tprov_ptr;
}


template <class Type> void Std_Pack_Of_Pts<Type>::show(void) const
{

   for (int d=0 ; d<_dim ; d++)
   {
       for (int n=0 ; n<_nb ; n++)
           cout << _pts[d][n] << " ";
       cout << "\n";
   }
}

template <class Type> void Std_Pack_Of_Pts<Type>::show_kth(INT k) const
{

   cout << "[" << _pts[0][k] ;
   for (int d=1 ; d<_dim ; d++)
       cout << " " << _pts[d][k] ;
   cout << "]";
}

template <class Type> Std_Pack_Of_Pts<Type> 
         * Std_Pack_Of_Pts<Type>::new_pck(INT dim,INT sz_buf) 
{
   return new Std_Pack_Of_Pts<Type>(dim,sz_buf);
}




template <class Type> void Std_Pack_Of_Pts<Type>::select_tab
                        (   Pack_Of_Pts * pack_gen,
                            const INT * tab_sel
                        ) const
{
// static int aCpt=0; aCpt++;
// std::cout << "XXXXX SELECT TAB " << aCpt << "\n";
  Std_Pack_Of_Pts<Type> * pck = 
             SAFE_DYNC( Std_Pack_Of_Pts<Type> *,pack_gen);
  Type ** pts_pack = pck->_pts;

  INT nb= pck->_nb;
  INT d = pck->_dim;
// std::cout << "1111111111\n";
  El_Internal.ElAssert
  (
          d<=_dim,
          EEM0<<"Std_Pack_Of_Pts<Type>::select_tab"
  );

// std::cout << "222222\n";
  for (int i=0; i<_nb; i++)
  {
      if (tab_sel[i])
      {
          for(int j=0 ; j<d ; j++)
             pts_pack[j][nb]=_pts[j][i];
          nb++;
      }
  }
// std::cout << "33333\n";
  pck->_nb = nb;
}

template <class Type> void Std_Pack_Of_Pts<Type>::convert_from_int
                           (const  Std_Pack_Of_Pts<INT> * pck2)
{
    set_nb(pck2->nb());
    for (int d=0; d<dim() ; d++)
        convert(_pts[d],pck2->_pts[d],_nb);
}

template <class Type> void Std_Pack_Of_Pts<Type>::convert_from_real
                           (const  Std_Pack_Of_Pts<REAL> * pck2)
{
    set_nb(pck2->nb());
    for (int d=0; d<dim() ; d++)
        convert(_pts[d],pck2->_pts[d],_nb);
}


template <class Type> void Std_Pack_Of_Pts<Type>::convert_from_rle
                           (const  RLE_Pack_Of_Pts * pck2) 
{
    set_nb(pck2->nb());
    const INT * pt0 = pck2->pt0();

    set_fonc_id(_pts[0],(Type) pt0[0],_nb);

    for (int d=1; d<dim() ; d++)
        set_cste(_pts[d],(Type) pt0[d],_nb);
}





template <class Type> void Std_Pack_Of_Pts<Type>::interv
                           (const  Std_Pack_Of_Pts_Gen * pck_gen,INT n1,INT n2)
{
   const  Std_Pack_Of_Pts<Type> * pck = 
          SAFE_DYNC( Std_Pack_Of_Pts<Type> *,const_cast<Std_Pack_Of_Pts_Gen *>(pck_gen));
    set_nb(ElMax(0,n2-n1));
    for (int d=0; d<dim() ; d++)
       _pts[d] = pck->_pts[d] + n1;
}

template <class Type> void * Std_Pack_Of_Pts<Type>::adr_coord() const
{
    return _pts;
}


template <class Type> void Std_Pack_Of_Pts<Type>::trans
              (const Pack_Of_Pts * pack, const INT * tr)
{
     Std_Pack_Of_Pts<Type> * std_pack = SAFE_DYNC(Std_Pack_Of_Pts<Type> *,const_cast<Pack_Of_Pts *>(pack));

     ASSERT_INTERNAL
     (    
          (std_pack->_dim == _dim) && (_sz_buf >= std_pack->_nb),
          "incoherence in RLE_Pack_Of_Pts::trans"
     );
     _nb = std_pack->_nb;

     for (int j =0; j <_dim ; j++)
     {
         Type * l = _pts[j];
         Type * stdl = std_pack->_pts[j];
         INT t = tr[j];

         for (int n =0; n <_nb ; n++)
             l[n] = stdl[n] + t;
     }
}


template <class Type> void Std_Pack_Of_Pts<Type>::push(Type * coord)
{
     ASSERT_INTERNAL
    (    not_full(),
         "Push in a full Std_Pack_Of_Pts"
    );

    for (int j = 0; j<_dim ; j++)
        _pts[j][_nb] = coord[j];
    _nb++;
}

template <class Type> void Std_Pack_Of_Pts<Type>::copy
                                   (const Pack_Of_Pts * gen)
{
    const Std_Pack_Of_Pts<Type> * p = gen->std_cast((Type *)0);

    _nb = p->_nb;
    ASSERT_INTERNAL(_sz_buf>=_nb,"insufficient buf in copy");
    for (int d=0 ; d<_dim ; d++)
        convert(_pts[d],p->_pts[d],_nb);
}

template <class Type> Pack_Of_Pts * Std_Pack_Of_Pts<Type>::dup(INT sz_buf) const
{
    if (sz_buf == -1)
       sz_buf = ElMax(_nb,1);

    Std_Pack_Of_Pts<Type> * res = Std_Pack_Of_Pts<Type>::new_pck(_dim,sz_buf);
    res->copy(this);

/*
    for (int d=0 ; d<_dim ; d++)
        convert(res->_pts[d],_pts[d],_nb);
    res->set_nb(_nb);
*/
    return res;
}


template <class Type>  void Std_Pack_Of_Pts<Type>::cat 
                       (const Std_Pack_Of_Pts<Type> * to_cat)
{
     ASSERT_INTERNAL(_nb+to_cat->_nb <= _sz_buf,"Overflow in  Std_Pack_Of_Pts<Type>::cat");

     for (int d = 0; d<_dim; d++)
         convert(_pts[d]+_nb,to_cat->_pts[d],to_cat->_nb);

     _nb += to_cat->_nb;
}

template <class Type>  void Std_Pack_Of_Pts<Type>::cat_gen
                       (const Std_Pack_Of_Pts_Gen * to_cat)
{
      ASSERT_INTERNAL(to_cat->type() == type(),"incoherence in cat_gen");
      cat(SAFE_DYNC(Std_Pack_Of_Pts<Type> *,const_cast<Std_Pack_Of_Pts_Gen *>(to_cat)));
}


template <class Type>  Std_Pack_Of_Pts<Type> * Std_Pack_Of_Pts<Type>::cat_and_grow
                       (Std_Pack_Of_Pts<Type> *  res,INT new_sz_buf,bool & chang) const
{
    ASSERT_INTERNAL(_dim == res->_dim,"incoherence in  Std_Pack_Of_Pts<Type>::cat_and_grow");

    // if res is unssufficient to contains 
    if (_nb+res->_nb  >  res->_sz_buf)
    {
          Std_Pack_Of_Pts<Type> * new_res = 
                  SAFE_DYNC(Std_Pack_Of_Pts<Type> *, res->dup(ElMax(_nb +res->_nb,new_sz_buf)));
          delete res;
          res = new_res;
          chang = true;
    }
    else
        chang = false;
    res->cat(this);
    return res;
}

template <class Type>  void Std_Pack_Of_Pts<Type>::auto_reverse()
{
     for (int d = 0; d<_dim; d++)
         auto_reverse_tab(_pts[d],_nb);
}

template <class Type>  void Std_Pack_Of_Pts<Type>::rgb_bgr
                            (const Std_Pack_Of_Pts_Gen * pck_gen)
{
     const Std_Pack_Of_Pts<Type> * pck = pck_gen->std_cast((Type *) 0);

     _nb = pck->_nb;
     _pts[0] = pck->_pts[2];
     _pts[1] = pck->_pts[1];
     _pts[2] = pck->_pts[0];
}


template <class Type>  void Std_Pack_Of_Pts<Type>::rgb_bgr()
{
     auto_reverse_tab(_pts,_dim);
}



template <class Type>  void 
          Std_Pack_Of_Pts<Type>::kth_pts(INT *p,INT k) const
{
     ASSERT_INTERNAL
     (
         (k>=0) && (k<_nb),
         "RLE_Pack_Of_Pts::kth_pts"
     );
     for(INT d=0 ; d<_dim ; d++)
        p[d] = (INT)_pts[d][k];

}

template <class Type>  INT
          Std_Pack_Of_Pts<Type>::proj_brd
             (
                      const Pack_Of_Pts * pck_gen,
                      const INT * p0,
                      const INT * p1,
                      INT         rab    // 0 for INT, 1 for real
             )
{
    const Std_Pack_Of_Pts<Type> * pck = pck_gen->std_cast((Type *) 0);

    set_nb(pck->nb());
    for (INT d=0; d<_dim ; d++)
       proj_in_seg
       (
           _pts[d],
           pck->_pts[d],
           (Type) p0[d]+rab,
           (Type) p1[d]-rab,
           _nb
       );
   return 0;
}

template <class Type> void Std_Pack_Of_Pts<Type>::verif_inside 
     (
         const INT * pt_min,
         const INT * pt_max,
         Type        rab_p0,
         Type        rab_p1
      )  const
{
   for (int d=0 ; d<_dim ; d++)
   {
       INT index =
            index_values_out_of_range
            (
                   _pts[d],
                   _nb,
                   pt_min[d]+rab_p0,
                   pt_max[d]-rab_p1
            );

         Tjs_El_User.ElAssert
         (
             index == INDEX_NOT_FOUND,
             EEM0 << "Out of bitmap domain while writing or reading\n"
                  << "|    point = " << ElEM(this,index) << "\n"
                  << "|    box , p_min = " << ElEM(pt_min,_dim)
                  <<        ", p_max = " << ElEM(pt_max,_dim)   << "\n"
                  << "|     (interpolation witdh  = " 
                  << rab_p0 << " " << rab_p1 << ")\n"
         );
   }
}

template <> CONST_STAT_TPL Pack_Of_Pts::type_pack Std_Pack_Of_Pts<INT>::type_glob = Pack_Of_Pts::integer;
template <> CONST_STAT_TPL Pack_Of_Pts::type_pack Std_Pack_Of_Pts<REAL>::type_glob = Pack_Of_Pts::real;


#if ElTemplateInstantiation
	template class Std_Pack_Of_Pts<INT>;
	template class Std_Pack_Of_Pts<REAL>;
#endif



Std_Pack_Of_Pts<INT> * lpt_to_pack(ElList<Pt2di> l)
{
    Std_Pack_Of_Pts<INT> * pack = Std_Pack_Of_Pts<INT>::new_pck(2,l.card());

    for (int i =0; !(l.empty()) ; i++)
    {
         Pt2di p = l.pop();
         pack->_pts[0][i] = p.x;
         pack->_pts[1][i] = p.y;
    }
    pack->set_nb(pack->pck_sz_buf());
    return pack;
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
