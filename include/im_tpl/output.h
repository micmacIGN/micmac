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


#ifndef _ELISE_IM_TPL_OUTPUT
#define _ELISE_IM_TPL_OUTPUT

template <class O1,class O2> class TClCatOut
{
     public :
          typedef ElPair<ElTyName O1::ValOut,ElTyName O2::ValOut>  ValOut;

          TClCatOut (O1 o1,O2 o2) : _o1 (o1), _o2 (o2) {}
          void   oset(const Pt2di & p,const ValOut & pair) 
          {
               _o1.oset(p,pair.v1());
               _o2.oset(p,pair.v2());
          }

     private :
          O1 _o1;
          O2 _o2;
};                 

template <class O1,class O2>
TClCatOut<O1,O2> TCatO(O1 o1,O2 o2) { return TClCatOut<O1,O2>(o1,o2);}  


template <class O1,class O2> class TClPipeOut
{
     public :
          typedef ElTyName O1::ValOut  ValOut;

          TClPipeOut (O1 o1,O2 o2) : _o1 (o1), _o2 (o2) {}
          void   oset(const Pt2di & p,ValOut v) 
          {
               _o1.oset(p,v);
               _o2.oset(p,v);
          }

     private :
          O1 _o1;
          O2 _o2;
};                 

template <class O1,class O2>
TClPipeOut<O1,O2> TPipe(O1 o1,O2 o2) { return TClPipeOut<O1,O2>(o1,o2);}  





template <class Type> class TClSigma
{
     public :
          typedef Type  ValOut;

          TClSigma (Type & v) :  _v (v) { v=0;}
          void   oset(const Pt2di &,Type v) { _v += v; }
     private :
          Type  & _v;
};                 

template <class Type> TClSigma<Type> TSigma(Type & v) { return TClSigma<Type>(v);}  




template <class F,class O> class TClRedirOut
{
     public :
          typedef ElTyName O::ValOut  ValOut;

          TClRedirOut (O o,F f) : _f (f), _o (o) {}
          void   oset(const Pt2di & p,ValOut) 
          {
               _o.oset(p,_f.get(p));
          }

     private :
          F _f;
          O _o;
};                 

template <class F,class O>
TClRedirOut<F,O> TRedir(O o,F f) { return TClRedirOut<F,O>(o,f);}  



template <class Type> class TClPushPt
{
     public :
        typedef INT ValOut;

        TClPushPt( ElFilo<Type> & f) : _f(f) {}

        void   oset(const Type & p,ValOut)
        {
               _f.pushlast(p);
        }

     private :
        ElFilo<Type> & _f;
};

template <class Type>
TClPushPt<Type> TPushPt(ElFilo<Type> & f) {return TClPushPt<Type>(f);}


#endif  //  _ELISE_IM_TPL_OUTPUT












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
