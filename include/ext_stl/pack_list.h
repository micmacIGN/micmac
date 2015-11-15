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



#ifndef  _ELISE_EXT_STL_PACK_LIST
#define  _ELISE_EXT_STL_PACK_LIST




#include <list>



template <class T,const int NB> class  ElPackList
{
      public :

           struct el
           {
                 T _vals[NB];
           };

      private :

           INT  _sz;
           INT  _i0;
           INT  _i1;
           ElSTDNS list<el >       _l;
           ElSTDNS list<el > *     _rl;


           void set_reserv_non_empty()
           {
                if (! _rl->size())
                   _rl->push_front(el());
           }

           void _push_front()
           {
               if(_rl)
               {
                  set_reserv_non_empty();
                  _l.splice(_l.begin(),*_rl,_rl->begin());
               }
               else
                  _l.push_front(el());
           }

           void _pop_front()
           {
                if(_rl)
                   _rl->splice(_rl->begin(),_l,_l.begin());
                else
                   _l.pop_front();
           }

           void _push_back()
           {
                if(_rl)
                {
                    set_reserv_non_empty();
                    _l.splice(_l.end(),*_rl,--_rl->end());
                }
                else
                    _l.push_back(el());
           }

           void _pop_back()
           {
                if(_rl)
                   _rl->splice(_rl->end(),_l,--_l.end());
                else
                   _l.pop_back();
           }

      public :
		

           class iterator
           {
               friend  class ElPackList<T,NB>;
               private :
                 ElTyName ElSTDNS list<el>::iterator _iter;
                 INT                _i;
                 INT                _pabs;

                 iterator(ElTyName ElSTDNS list<el>::iterator iter,INT i,INT pos_abs) :
                      _iter (iter),
                      _i    (i),
                      _pabs (pos_abs)
                 {
                 }
                 
               public :

                 bool operator==(const iterator& x) const 
                 { 
                      return (_pabs == x._pabs);
                 }
                 bool operator!=(const iterator& x) const 
                 { 
                      return !(*this == x);
                 }


                 T & operator * () { return (*_iter)._vals[_i];}

                 void operator++(int) 
                 {
                      if (_i == NB-1) { _i=0; _iter++; }
                      else _i++;
                      _pabs++;
                 }
                 void operator--(int) 
                 {
                      if (_i == 0) { _i=NB-1; _iter--; }
                      else _i--;
                      _pabs--;
                 }
           };

           T  & first()  { return (*_l.begin())._vals[_i0] ;}
           T  & last()   { iterator it = end(); it--; return *it;}

           iterator begin() {return iterator(_l.begin(),_i0,0);}
           iterator end()   {return iterator(--_l.end(),_i1,_sz);}

            ElPackList(ElPackList* reserve = 0) : 
                   _sz      (0), 
                   _i0      (NB/2), 
                   _i1      (_i0),
                   _rl      (reserve ? &(reserve->_l) : 0) 
            {}

            void set_reserve(ElPackList* reserve)
            {
                 _rl = &(reserve->_l);
            }

            T & front()
            {
                return _l.front()._vals[_i0];
            }


            void push_front(const T & v)
            {
                 if ((_i0==0)|| (_sz ==0))
                 {
                    _push_front();
                    if (_sz == 0)
                       _i0 = _i1 = 1;
                    else
                       _i0 = NB;
                 }
                 _sz ++;
                 _l.front()._vals[--_i0] = v;
            }

            T pop_front()
            {
                 ELISE_ASSERT(_sz!=0,"empty in ElPackList::pop");
                 T res =  _l.front()._vals[_i0];
                 _sz--;
                 _i0++;

                 if ((_i0==NB) || (_sz ==0))
                 {
                    _pop_front();
                    _i0 = 0 ;
                    if (_sz == 0) _i1 = 0;
                 }
                 return res;
            }

            void push_back(const T & v)
            {
                 if ((_i1==NB)|| (_sz ==0))
                 {
                    _push_back();
                    if (_sz == 0)
                       _i0 = _i1 = 0;
                    else
                       _i1 = 0;
                 }
                 _sz ++;
                 _l.back()._vals[_i1++] = v;
            }

            T pop_back()
            {
                 ELISE_ASSERT(_sz!=0,"empty in ElPackList::pop");
                 T res =  _l.back()._vals[_i1-1];
                 _sz--;
                 _i1--;

                 if ((_i1==0) || (_sz ==0))
                 {
                    _pop_back();
                    _i1 = NB;
                    if (_sz == 0) _i0 = NB;
                 }
                 return res;
            }


            T& back()
            {
                 return  _l.back()._vals[_i1-1];
            }


            INT size() const{return _sz;}
            bool empty() const{return _sz == 0;}

};

#endif //  _ELISE_EXT_STL_PACK_LIST


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
