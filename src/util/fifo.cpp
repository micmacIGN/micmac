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


// TEST SVN COMMIT
// TEST SVN COMMIT

    //==============================================================
    //==============================================================
    //==============================================================

#if (0)


static INT  Fils(INT i) { return   i/2;}
static INT  Pere1(INT i){ return   i*2;}
static INT  Pere2(INT i){ return   i*2+1;}




template <class Type,class Compare>   ElHeap<Type,Compare>::ElHeap(Type infty,Compare inferior,INT capa) :
    ElFifo<Type> (capa),
    _infty    (infty),
    _inferior (inferior)
{
    pushlast(_infty);
}

template <class Type,class Compare>  void ElHeap<Type,Compare>::heap_up(INT k)
{
    for(;;)
    {
        INT ind_plus_petit = k;

        INT p1 = Pere1(k);
        if ((p1 < _nb) && _inferior(_tab[p1],_tab[ind_plus_petit]))
            ind_plus_petit = p1;
        
        INT p2 = Pere2(k);
        if ((p2 < _nb) && _inferior(_tab[p2],_tab[ind_plus_petit]))
            ind_plus_petit = p2;

        if (ind_plus_petit == k)
           return;
        else
        {
            ElSwap(_tab[ind_plus_petit],_tab[k]);
            k = ind_plus_petit;
        }
    }
}


template <class Type,class Compare>  void ElHeap<Type,Compare>::heap_down(INT k)
{
    while ((k>1)&& _inferior(_tab[k],_tab[Fils(k)]))
    {
          ElSwap(_tab[k],_tab[Fils(k)]);
          k = Fils(k);
    }
}


template <class Type,class Compare>  void ElHeap<Type,Compare>::push(Type v)
{
   if (_inferior(v,_infty))
      Tjs_El_User.assert
      (
          false,
          EEM0 << " ! (_inferior(v,_infty)) in ElHeap<Type,Compare>::push"
      );

    pushlast(v);
    heap_down(_nb-1);
}

template <class Type,class Compare>  Type ElHeap<Type,Compare>::top()
{
   if (_nb <= 1) 
      Tjs_El_User.assert
      (
          false,
          EEM0 << " empty ElHeap<Type,Compare> in top"
      );
   return _tab[1];
}

template <class Type,class Compare>  bool ElHeap<Type,Compare>::empty()
{
    return _nb <= 1;
}

template <class Type,class Compare>  INT ElHeap<Type,Compare>::nb()
{
    return _nb - 1;
}


template <class Type,class Compare>  void ElHeap<Type,Compare>::set_empty()
{
   _nb = 1;
}


template <class Type,class Compare>  void ElHeap<Type,Compare>::pop()
{
   if (_nb <= 1) 
      Tjs_El_User.assert
      (
          false,
          EEM0 << " empty ElHeap<Type,Compare> in top"
      );
   _tab[1] = _tab[_nb-1];
   _nb--;
   heap_up(1);
}

template class  ElHeap<Pt3di,ElCmpZ>;
template class  ElHeap<Pt4di,ElCmp4Z>;




/***********************************************/
/***********************************************/
/**************  ElIntegerHeap  ****************/
/***********************************************/
/***********************************************/


template <class Type> ElIntegerHeap<Type>::~ElIntegerHeap()
{
    ElRawList<Type>::rec_delete(_lfree);
    while(! _objs.empty())
    {
         ElRawList<Type>::rec_delete(_objs.poplast());
    }
}

template <class Type> void ElIntegerHeap<Type>::push(Type val,INT index)
{
    if (_nb_objet == 0)
    {
         _ind_min  = index; 
    }
    _nb_objet ++;

    INT ind_rel = index-_ind_min;

    if (ind_rel < 0)
    {
        for (INT i= ind_rel ; i<0; i++)
           _objs.pushfirst(0);
       _ind_min = index;
       ind_rel = 0;
    }
    else
    {
       while(_objs.nb() <= ind_rel)
           _objs.pushlast(0);
    }

    ElRawList<Type> * el = get_el();
    el->_o = val;
    el->_next = _objs[ind_rel];
    _objs[ind_rel]  = el;
}

template <class Type> Type ElIntegerHeap<Type>::top(INT & index)
{
   if ( _nb_objet == 0)
      Tjs_El_User.assert 
      (
          false,
          EEM0 << "Empty ElIntegerHeap in pop"
      );
    index  = _ind_min;
    return _objs[0]->_o;
}

template <class Type> void ElIntegerHeap<Type>::pop()
{
   if ( _nb_objet == 0)
      Tjs_El_User.assert 
      (
          false,
          EEM0 << "Empty ElIntegerHeap in pop"
      );

   _nb_objet--;
    ElRawList<Type> * el = _objs[0];
    _objs[0] = _objs[0]->_next;
    put_el_free(el);

    if (! _objs[0])
    {
        while(_objs.nb() && (_objs[0] == 0))
        {
              _objs.popfirst();
              _ind_min++;
              _nb_incr++;
        }
    }
}

template <class Type> bool ElIntegerHeap<Type>::pop(Type & val,INT & ind)
{
   if ( _nb_objet == 0) return false;
   val = top(ind);
   pop();
   return true;
}

template <class Type> bool ElIntegerHeap<Type>::empty()
{
     return _nb_objet == 0;
}

template <class Type> INT ElIntegerHeap<Type>::nb()
{
     return _nb_objet ;
}



template <class Type> ElIntegerHeap<Type>::ElIntegerHeap(INT capa) :
    _nb_objet (0),
    _objs     (capa),
    _lfree    (0),
    _nb_incr  (0)
{
}



template class ElIntegerHeap<Pt4di>;
template class ElIntegerHeap<Pt3di>;

/*****************************************************/
/*****************************************************/
/**************  ElBornedIntegerHeap  ****************/
/*****************************************************/
/*****************************************************/



template <class Type> ElBornedIntegerHeap<Type>::ElBornedIntegerHeap(INT max_dif) :
    _nb_objet (0),
    _max_dif   (max_dif),
    _objs     (new  ElFifo<Type> [max_dif])
{
}

template <class Type> void ElBornedIntegerHeap<Type>::pop()
{
  if ( _nb_objet == 0)
     Tjs_El_User.assert 
      (
          false,
          EEM0 << "Empty ElBornedIntegerHeap in pop"
      );
   _nb_objet--;
    while(_objs[_ind_min%_max_dif].empty()) _ind_min++;
   _objs[_ind_min%_max_dif].poplast();

}

template <class Type> Type ElBornedIntegerHeap<Type>::top(INT & ind)
{
  if ( _nb_objet == 0)
     Tjs_El_User.assert 
      (
          false,
          EEM0 << "Empty ElBornedIntegerHeap in pop"
      );
   while(_objs[_ind_min%_max_dif].empty()) _ind_min++;
   ind = _ind_min;
   return _objs[_ind_min%_max_dif].top();
}


template <class Type> bool ElBornedIntegerHeap<Type>::pop(Type & val,INT & ind)
{
  if ( _nb_objet == 0)
     return false;

  _nb_objet--;
  while(_objs[_ind_min%_max_dif].empty()) _ind_min++;
  ind = _ind_min;
  val = _objs[_ind_min%_max_dif].poplast();

  return true;
}

template <class Type> void ElBornedIntegerHeap<Type>::push(Type val,INT index)
{
    if (_nb_objet == 0)
    {
        _ind_min  = index; 
        _ind_max = index + 1;
        Tjs_El_User.assert 
        (
           index>=0,
           EEM0 << "bad index (<0) in ElBornedIntegerHeap"
        );
    }
    else
    {
         if (index < _ind_min) _ind_min = index;
         else if (index>=_ind_max) _ind_max = index+1;
    }
   _nb_objet ++;

   while((_ind_max-_ind_min>_max_dif) && _objs[_ind_min%_max_dif].empty())
         _ind_min++;

   if (_ind_max-_ind_min>_max_dif)
        Tjs_El_User.assert 
        (
           false,
           EEM0 << "bad index in ElBornedIntegerHeap" 
        );
   _objs[index%_max_dif].pushlast(val);
}


template <class Type> bool ElBornedIntegerHeap<Type>::empty()
{
     return _nb_objet == 0;
}

template <class Type> INT ElBornedIntegerHeap<Type>::nb()
{
     return _nb_objet ;
}


template <class Type> ElBornedIntegerHeap<Type>::~ElBornedIntegerHeap()
{
    delete [] _objs;
}

template <class Type> INT ElBornedIntegerHeap<Type>::capa_tot() const
{
    INT res =0;
    for (INT k=0 ; k<_max_dif; k++)
        res += _objs[k].capa();
   return res;
}


template class ElBornedIntegerHeap<Pt4di>;
template class ElBornedIntegerHeap<Pt3di>;
#endif

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
