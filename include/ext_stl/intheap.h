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



#ifndef _ELISE_EXT_STL_INT_HEAP
#define _ELISE_EXT_STL_INT_HEAP

#include "ext_stl/pack_list.h"

template <class Type,INT NB> class ElBornedIntegerHeap
{
      private :

          ElBornedIntegerHeap(const ElBornedIntegerHeap<Type,NB> &);


          INT     _nb_objet;
          INT     _ind_min;
          INT     _ind_max;
          INT     _max_dif;

          ElPackList<Type,NB>     * _objs;
          ElPackList<Type,NB>      _reserve;

          void verif_delta_index()
          {
              ELISE_ASSERT
              (
                   _ind_max-_ind_min<=_max_dif,
                    "bad index in ElBornedIntegerHeap"
              );
          }

      public :

          INT nb() { return _nb_objet;}
          bool empty() { return _nb_objet==0;}

          ElBornedIntegerHeap(INT max_dif) :
             _nb_objet (0),
             _max_dif  (max_dif),
             _objs     (new   ElPackList<Type,NB> [max_dif])
          {
               for (int KKK=0; KKK<max_dif; KKK++)
                   _objs[KKK].set_reserve(&_reserve);
          }

          virtual ~ElBornedIntegerHeap(){ delete [] _objs;};
          void push(const Type & val,INT index)
          {
              if (_nb_objet == 0)
              {
                  _ind_min = index;
                  _ind_max = index + 1;
              }
              else
              {
                if (index < _ind_min) 
                {
                   _ind_min = index;
                   verif_delta_index();
                }
                else if (index>=_ind_max) 
                {
                   _ind_max = index+1;
                    while
                    (
                         (_ind_max-_ind_min>_max_dif) 
                      && _objs[mod(_ind_min,_max_dif)].empty()
                    )
                      _ind_min++;
                   verif_delta_index();
                }
              }
              _nb_objet ++;

              _objs[mod(index,_max_dif)].push_back(val);
          }
          bool pop(Type & val,INT & index)
          {
            if ( _nb_objet == 0)
               return false;

            _nb_objet--;
            while(_objs[mod(_ind_min,_max_dif)].empty()) _ind_min++;
            index = _ind_min;
            val = _objs[mod(_ind_min,_max_dif)].pop_front();

            return true;
          }

};

//  Idem ElBornedIntegerHeap, mais sort l'element max

template <class Type,INT NB> class ElMaxBornedIntegerHeap  : public ElBornedIntegerHeap<Type,NB>
{
      public :

         ElMaxBornedIntegerHeap(INT max_dif) :  ElBornedIntegerHeap<Type,NB> (max_dif) {}

         void push(const Type & val,INT index)
         {
                ElBornedIntegerHeap<Type,NB>::push(val,-index);
         }

          bool pop(Type & val,INT & index)
          {
               bool res = ElBornedIntegerHeap<Type,NB>::pop(val,index);
               if (res) index = -index;
               return res;
          }

      private :
};


#endif /* ! _ELISE_EXT_STL_INT_HEAP */








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
