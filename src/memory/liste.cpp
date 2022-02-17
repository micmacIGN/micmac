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



   /**************************************************/

template <class Type>  ElList<Type>  operator +(ElList<Type> l,Type e)
{
     return   new liste_phys<Type>(e,l);
}


template <class Type> ElList<Type>::ElList(RC_Object * el) :
     PRC0(el)
{
}

template <class Type>  ElList<Type>::ElList() :
      PRC0(0)
{
}



template <class Type>  ElList<Type>  SumElListe (ElList<Type> l,Type e)
{
     return   new liste_phys<Type>(e,l);
}






template <class Type>  Type ElList<Type>::car() const
{
    ASSERT_TJS_USER(_ptr!= 0," attempt to take car of empty list");
    return ((liste_phys<Type> *) _ptr)->_el;
}

template <class Type>  ElList<Type> ElList<Type>::cdr() const
{
    ASSERT_TJS_USER(_ptr!= 0," attempt to take car of empty list");
    return ((liste_phys<Type> *) _ptr)->_next;
}


template <class Type>  Type ElList<Type>::pop()
{
    ASSERT_TJS_USER(_ptr!= 0," attempt to take car of empty list");
    Type el =  ((liste_phys<Type> *) _ptr)->_el;
    *this =  ElList<Type>(  ((liste_phys<Type> *) _ptr)->_next);
    //  *this =  ((liste_phys<Type> *) _ptr)->_next;  CA PLANTE depuis 22/10/2013 ?? !!!  MPD : aucune idee pourquoi 
    // *this = this->cdr();
    return el;
}


template <class Type>  bool ElList<Type>::empty() const
{
    return _ptr == 0;
}

template <class Type>  INT ElList<Type>::card() const
{
    INT n = 0;

     auto l = *this;
     while (l._ptr)
     {
           l = ((liste_phys<Type> *) l._ptr)->_next;
           n ++;
     }
     return n;
}

template <class Type> ElList<Type> ElList<Type>::reverse()
{
     ElList<Type> res ;

     for (ElList<Type> l=*this ; ! l.empty() ; l=l.cdr())
         res = res + l.car();
     return res;
}

template <class Type>  Type ElList<Type>::last() const
{
    ASSERT_TJS_USER(_ptr!= 0," attempt to take car of empty list");
     Type res = ((liste_phys<Type> *) _ptr)->_el;

     auto l = *this;
     while (l._ptr)
     {
           res = ((liste_phys<Type> *) l._ptr)->_el;
           l = ((liste_phys<Type> *) l._ptr)->_next;
     }
     return res;
}


   /**************************************************/

template <class Type> liste_phys<Type>::liste_phys(Type e,ElList<Type> n) :
    _el   (e),
    _next (n)
{
} 




template <class Type> liste_phys<Type>::~liste_phys() {}


/*
template <class Type> void instantiate_liste(Type v)
{
     ElList<Type> l;
     l.car();
     l.card();
     l.last();
     l.cdr();
     l.pop();
     l.empty();
}
*/

#define INSTANTIATE_LISTE()\
l.car();\
l.card();\
l.last();\
l.cdr();\
l.pop();\
l.empty();


/*
#if (0)
    instantiate_liste(Hplan());
    instantiate_liste(Hdroite());
    instantiate_liste(Hregion());
    instantiate_liste(Batiment());
    instantiate_liste(Facette_3d());
    instantiate_liste(Facette_2d());
#endif
*/

/*
#define MACRO_INSTANTATIATE_LISTE(Type)\
template class ElList<Type>;

MACRO_INSTANTATIATE_LISTE(Pt2dr)
MACRO_INSTANTATIATE_LISTE(INT)
MACRO_INSTANTATIATE_LISTE(Pt3dr)
MACRO_INSTANTATIATE_LISTE(Pt2di)
MACRO_INSTANTATIATE_LISTE(Pt3di)
MACRO_INSTANTATIATE_LISTE(Elise_Palette)
MACRO_INSTANTATIATE_LISTE(Disp_Set_Of_Pal)
MACRO_INSTANTATIATE_LISTE(Elise_colour)
MACRO_INSTANTATIATE_LISTE(Arg_Opt_Plot1d)
MACRO_INSTANTATIATE_LISTE(Gif_Im)
MACRO_INSTANTATIATE_LISTE(Arg_Tiff)
MACRO_INSTANTATIATE_LISTE(ArgSkeleton)
*/

void instatiate_liste()
{
/*
    instantiate_liste(Pt2dr(3.,3.));
    instantiate_liste((INT)3);
    instantiate_liste(Pt3dr(3.,3.,3.));
    instantiate_liste(Pt2di(3,3));
    instantiate_liste(Pt3di(3,3,3));
    instantiate_liste(Elise_Palette(Gray_Pal(2)));
    instantiate_liste(Disp_Set_Of_Pal((Data_Disp_Set_Of_Pal *)0));
    instantiate_liste(Elise_colour::rgb(1.0,1.0,1.0));
    instantiate_liste(Arg_Opt_Plot1d(PlBox(Pt2dr(0,0),Pt2dr(0,0))));
    instantiate_liste(Gif_Im((char *)0,ELISE_fp(),(class Data_Giff *)0));
    instantiate_liste(Arg_Tiff(Tiff_Im::ANoStrip()));
    instantiate_liste(ArgSkeleton(AngSkel(0.0)));
*/
}

// template class liste_phys<aTINS >;
//template ElList<aTINS >  SumElListe <aTINS> (ElList<aTINS > ,aTINS );
//template class ElList<aTINS >;
/*
typedef Pt2dr aTINS;
template ElList<aTINS >  operator +<aTINS> (ElList<aTINS > ,aTINS );
template ElList<TYPE >  operator +<TYPE> (ElList<TYPE > ,TYPE );\
template <class Type>  ElList<Type>  operator +(ElList<Type> l,Type e)


`ElList<Elise_Palette> operator+<Elise_Palette>(ElList<Elise_Palette>, Elise_Palette)'

*/

 //ElList<int>  operator+ <int> (ElList<int> ,int );

/*template <class Type>  ElList<Type> Test  (ElList<Type> l,Type e)
{
     return   new liste_phys<Type>(e,l);
}
template <> ElList<int>  Test <int> (ElList<int> ,int );
template ElList<int>  operator+ (ElList<int> ,int );
*/


#if ElTemplateInstantiation
#if (1)
#define INSTANTIATE_EL_LISTE(TYPE) \
template ElList<TYPE>  operator+ (ElList<TYPE> ,TYPE ); \
template class liste_phys<TYPE >;\
template ElList<TYPE >  SumElListe <TYPE > (ElList<TYPE > ,TYPE );\
template class ElList<TYPE >;

#if (0)
INSTANTIATE_EL_LISTE(Hplan)
INSTANTIATE_EL_LISTE(Hdroite)
INSTANTIATE_EL_LISTE(Hregion)
INSTANTIATE_EL_LISTE(Hdroite_2D<INT>)
INSTANTIATE_EL_LISTE(Batiment)
INSTANTIATE_EL_LISTE(Facette_3d)
INSTANTIATE_EL_LISTE(Facette_2d)
#endif

INSTANTIATE_EL_LISTE(INT)
INSTANTIATE_EL_LISTE(REAL)
INSTANTIATE_EL_LISTE(ElList<Pt2di>)
INSTANTIATE_EL_LISTE(Pt3dr)
INSTANTIATE_EL_LISTE(Pt2dr)
INSTANTIATE_EL_LISTE(Pt2di)
INSTANTIATE_EL_LISTE(Facette_2D)
INSTANTIATE_EL_LISTE(Pt3di)
INSTANTIATE_EL_LISTE(Elise_Palette);
INSTANTIATE_EL_LISTE(Disp_Set_Of_Pal);
INSTANTIATE_EL_LISTE(Elise_colour);
INSTANTIATE_EL_LISTE(Arg_Opt_Plot1d);
INSTANTIATE_EL_LISTE(Gif_Im);
INSTANTIATE_EL_LISTE(Arg_Tiff);
INSTANTIATE_EL_LISTE(ArgSkeleton);
#endif

#endif


   /**************************************************/

// test capa : this is not required by NEW_VECTEUR and
// DELETE_VECTOR (who handle corrcly empty allocation)
// but this allos user not to destroy the stack created
// by implicit construction.


template <class Type> void Elise_Pile<Type>::destr()
{
    if (_capa)
       DELETE_VECTOR(_ptr,0);
}

template <class Type> Elise_Pile<Type>::Elise_Pile(INT capa)
{
   _nb   = 0;
   _capa = capa;
   _ptr = capa ?  NEW_VECTEUR(0,capa,Type) : 0;
}

template <class Type> Elise_Pile<Type>::Elise_Pile()
{
    *this = Elise_Pile<Type>(0);
}

template <class Type> INT Elise_Pile<Type>::nb()
{
    return _nb;
}

template <class Type> Type * Elise_Pile<Type>::ptr()
{
    return _ptr;
}


template <class Type> void Elise_Pile<Type>::push(Type v)
{
    ASSERT_INTERNAL(_nb <_capa,"Stack over flow");
    _ptr[_nb++] = v;
}

template <class Type> void Elise_Pile<Type>::reset(INT nb)
{
    ASSERT_INTERNAL( (nb >= 0) && (nb <= _nb),"Stack invalid reset");
    _nb = nb;
}


#if ElTemplateInstantiation
template class Elise_Pile<El_RW_Point>;
template class Elise_Pile<El_RW_Rectangle>;
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
