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



#ifndef _ELISE_GARB_COLL_H
#define _ELISE_GARB_COLL_H



template <class Type>  class ElList; 
template <class Type>  class liste_phys;
template <class Type>  ElList <Type> operator  + (ElList<Type>,Type);

template <class Type>  ElList <Type> SumElListe (ElList<Type>,Type);
template <class Type>  ElList <Type> SumElListe (ElList<Type>,Type,Type);


template <class Type>  class ElList :  public  PRC0
{
   public :

       ElList<Type>();

       Type        car() const;  // Fatal error when empty
       ElList<Type>  cdr() const;  // Fatal error when empty
       Type        last() const;  // Fatal error when empty
       Type        pop();  // Fatal error when empty

       ElList<Type> reverse();  // only for ElList<Pt2di> for today

       // friend ElList <Type> ::operator  + (ElList<Type>,Type);

       bool                 empty() const;
       INT                  card() const;
       void                 clear() {while(!empty()) pop();}

      ElList<Type>(RC_Object*);
};

template <class Type>  class liste_phys : public RC_Object
{

         // friend  ElList<Type> ::operator +(ElList<Type> l,Type e);
         // friend  ElList<Type>;

         public :


               virtual ~liste_phys();
               liste_phys(Type, ElList<Type>);
               Type  _el;
               ElList<Type> _next;
};
     


typedef  ElList<Pt2di> LPt2di;
typedef  ElList<Pt3di> LPt3di;
template  <class Type> class Data_Tab_CPT_REF;





/*
const INT Elise_Std_Max_Dim = 20;
const INT Elise_Std_Max_Buf = 500;
*/
enum {
    Elise_Std_Max_Dim = 20,
    Elise_Std_Max_Buf = 500
};


template  <class Type> class Tab_CPT_REF : public PRC0
{
     public :

           Tab_CPT_REF(Type p0,Type p1,Type p2);
           Tab_CPT_REF(Type p0,Type p1,Type p2,Type p3);
           Tab_CPT_REF(const Type *objects,INT nb);
           Tab_CPT_REF(INT nb);

           INT   nb();
           Type & operator [](INT i);
           void  push(Type);

     private :

          inline  Data_Tab_CPT_REF<Type> * dtd();
};


typedef Tab_CPT_REF<Pt2dr>   Facette_2D;

extern ElList<Facette_2D> read_cadastre(const char * name);

ElList<Pt2di> NewLPt2di(Pt2di);

/**************************************************/
/**************************************************/
/**************************************************/

#endif /*! _ELISE_GARB_COLL_H*/

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
