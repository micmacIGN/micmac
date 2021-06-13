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


#ifndef _ELISE_IM_TPL_FONC_OPERATOR
#define _ELISE_IM_TPL_FONC_OPERATOR

template <class Type> class  TFoncCste
{
    public :
         typedef  Type  OutputFonc;

         TFoncCste(Type v) : _v(v) {}
         Type get(const Pt2di &)   { return _v;}
                                                                                                                private :
                                                                                                                   Type _v;
};
template <class Type> TFoncCste<Type>  TCste(Type v) { return TFoncCste<Type>(v);}            

class TFX
{
    public :

        typedef INT OutputFonc;
        TFX(){}
        INT get(const Pt2di & p) {return p.x;}
};

class TFY
{
    public :

        typedef INT OutputFonc;
        TFY(){}
        INT get(const Pt2di & p) {return p.y;}
};



template <class F1,class F2> class TClPlusFonc
{
     public :
          typedef ElTyName  F1::OutputFonc  OutputFonc;
          TClPlusFonc(F1 f1,F2 f2) : _f1 (f1), _f2 (f2) {}
          OutputFonc   get(const Pt2di & p) {return _f1.get(p) + _f2.get(p);}

     private :
          F1 _f1;
          F2 _f2;
};
template <class F1,class F2>  
TClPlusFonc<F1,F2> TPlus(F1 f1,F2 f2) { return TClPlusFonc<F1,F2>(f1,f2);}


template <class F1,class F2> class TClModFonc
{
     public :
          typedef ElTyName  F1::OutputFonc  OutputFonc;
          TClModFonc(F1 f1,F2 f2) : _f1 (f1), _f2 (f2) {}
          OutputFonc   get(const Pt2di & p) {return _f1.get(p) % _f2.get(p);}

     private :
          F1 _f1;
          F2 _f2;
};
template <class F1,class F2>  
TClModFonc<F1,F2> TMod(F1 f1,F2 f2) { return TClModFonc<F1,F2>(f1,f2);}


template <class T1,class T2> class ElPair
{
    public :
       ElPair(T1 v1,T2 v2):  _v1(v1),_v2(v2) {}
   
       const T1 & v1() const {return _v1;}
       const T2 & v2() const {return _v2;}

    private :
       T1  _v1;
       T2  _v2;
};

template <class F1,class F2> class TClCatFonc
{
     public :
          typedef ElPair<ElTyName  F1::OutputFonc,ElTyName  F2::OutputFonc>   OutputFonc;
          TClCatFonc (F1 f1,F2 f2) : _f1 (f1), _f2 (f2) {}
          OutputFonc   get(const Pt2di & p) {return OutputFonc(_f1.get(p),_f2.get(p));}

     private :
          F1 _f1;
          F2 _f2;
};

template <class F1,class F2>  
TClCatFonc<F1,F2> TCatF(F1 f1,F2 f2) { return TClCatFonc<F1,F2>(f1,f2);}





#endif  //  _ELISE_IM_TPL_FONC_OPERATOR












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
