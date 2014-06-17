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


#ifndef _ELISE_EXT_STL_EL_SLIST_H
#define _ELISE_EXT_STL_EL_SLIST_H

template <class Obj>  class ElSlist : public  Mcheck
{
      private :

         // pour l'instant pas besoin de copie, donc private
         ElSlist(const ElSlist &);

      public :
          class node
          {
              public :
                  node (const Obj & obj,node * NEXT) :
                       _obj(obj),
                       _next (NEXT)
                  {}
                  Obj          _obj;
                  node  *    _next;

                  void recycler(ElSlist * Reserve)
                  {
                      if (Reserve)
                         Reserve->push_front(this);
                      else
                         DELETE_ONE(this);
                  }
                  void  suppr_next(ElSlist * Reserve)
                  {
                       node * next = _next;
                       ELISE_ASSERT(next,"ElSlist::node::suppr_next");
                       _next = next->_next;
                       next->recycler(Reserve);
                  }
          };
      private :
          friend  class node;

          node * _node;

          void push_front(node * n)
          {
               n->_next = _node;
              _node = n;
          }

         void pop_front (ElSlist * Reserve)
         {
             node * next = _node->_next;
             _node->recycler(Reserve);
             _node = next;
         }

      public :


         class iterator
         {
               public :
                  friend  class ElSlist;
                  bool operator!=(const iterator & it)
                  {
                       return _node != it._node;
                  }
                  bool operator==(const iterator & it)
                  {
                       return _node == it._node;
                  }
                  void  operator ++(int)
                  {
                      _node = _node->_next;
                  }


                  Obj & operator * () { return _node->_obj;}

               private :

                  node * _node;
                  iterator(node * N) : _node(N){}
         };

         iterator begin() {return iterator(_node);}
         iterator end() {return iterator(0);}

         bool  empty() {return _node != 0;}
         void  push_front(const Obj & obj,ElSlist * Reserve =0)
         {
              if (Reserve && (Reserve->_node))
              {
                 node * newN = Reserve->_node;
                 Reserve->_node = newN->_next;
                 _node = new (newN) node(obj,_node);
              }
              else
                  _node = CLASS_NEW_ONE(node,(obj,_node));
         }

         void clear (ElSlist * Reserve =0)
         {
             while(_node)
                   pop_front(Reserve);
         }

         INT remove(const Obj & obj,ElSlist * Reserve =0)
         {
              INT res =0;
              while (_node && (_node->_obj == obj))
              {
                   pop_front(Reserve);
                   res++;
              }
              node * prec = _node; 
              while (prec && (prec->_next))
              {
                   if (prec->_next->_obj == obj)
                   {
                      res ++;
                      prec->suppr_next(Reserve);
                   }
                   else
                      prec = prec->_next;
              }
              return res;
         }

         ~ElSlist(){clear();}

         ElSlist() : _node (0) {}

};


#endif //  _ELISE_EXT_STL_EL_SLIST_H


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
