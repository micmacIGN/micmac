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

#ifndef _ELISE_ALGO_GEOM_INDEX_PTS
#define _ELISE_ALGO_GEOM_INDEX_PTS

#include "StdAfx.h"

template <class Type,class Fpt> class IndexPts
{
     public :
        IndexPts
        (
             Fpt   FPT,
             INT   NBHASH
        );
        ~IndexPts();

        Type * get(Pt2di pt);
        void   add(const Type & val);

     private :
  
        INT           hash(Pt2di pt);
        Fpt                     _fpt;
        INT                 _nb_hash;
        ElFilo<Type>   **       _mat;
};

template <class Type,class Fpt>  INT IndexPts<Type,Fpt>::hash(Pt2di pt)
{
     return  mod(pt.x + 17 * pt.y +pt.x*pt.y, _nb_hash);
}


template <class Type,class Fpt>  
         IndexPts<Type,Fpt>::IndexPts
         (
             Fpt   FPT,
             INT   NBHASH  
         )  :
            _fpt      (FPT),
            _nb_hash  (ElMax(NBHASH,1)),
            _mat      (NEW_TAB(_nb_hash,ElFilo<Type> *))
{
    for (INT k=0 ; k<_nb_hash ; k++)
        _mat[k] =0;
}

template <class Type,class Fpt> Type * IndexPts<Type,Fpt>::get(Pt2di pt)
{
   ElFilo<Type> * el = _mat[hash(pt)];
   if (! el)
      return 0;
   for (INT k =0; k<el->nb() ; k++)
       if (_fpt((*el)[k]) == pt)
         return & ((*el)[k]);
   return 0;
}

template <class Type,class Fpt>  void IndexPts<Type,Fpt>::add(const Type & val)
{
     INT ind = hash(_fpt(val));
     if (!  _mat[ind])
        _mat[ind] = CLASS_NEW_ONE(ElFilo<Type>,(2));

     _mat[ind]->pushlast(val);
}

template <class Type,class Fpt> IndexPts<Type,Fpt>::~IndexPts()
{
    for (INT k=0 ; k<_nb_hash ; k++)
        if (_mat[k])
           DELETE_ONE(_mat[k]);
   DELETE_TAB(_mat);
}



#endif // _ELISE_ALGO_GEOM_INDEX_PTS







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
