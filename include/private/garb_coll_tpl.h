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



#ifndef _ELISE_GAR_COLL_TPL_H
#define _ELISE_GAR_COLL_TPL_H


template <class Type> class Tab_Prov
                      : public Mcheck
{
    public :
          virtual ~Tab_Prov() ;
          virtual Type   *   coord() = 0;
};

template <class Type> class AllocTprov 
{
    public :
        static  Tab_Prov<Type> *   alloc_big_tprov(INT sz_buf);
        static  Tab_Prov<Type *>             * alloc_small_ptr_tprov(INT sz_buf);
        static  Tab_Prov<Tab_Prov<Type> *>   * alloc_small_tprov_tprov(INT sz_buf);
};




typedef Tab_Prov<INT>   Tprov_INT;
Tprov_INT * alloc_tab_prov_int_small(INT sz_buf);

typedef Tab_Prov<REAL>   Tprov_REAL;
typedef Tab_Prov<char>   Tprov_char;




Tprov_char * dup_name_std(const char * name);
Tprov_INT  * dup_sz_std(const int * sz,int dim);

template <class Type> class Elise_Pile;

template <class Type> class Elise_Pile
{
    // sometime qpush is defined to solve some problems with inlining template
    public  :
       INT nb();
       Type * ptr();
       void  destr();
       void push(Type);
       void reset(INT);
       Elise_Pile();
       Elise_Pile(INT capa);

         void qpush(const Type & v){_ptr[_nb++] = v;}
    private :

       Type       * _ptr;
       INT          _nb;
       INT          _capa;
};



#endif /*! _ELISE_GAR_COLL_TPL_H */

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
