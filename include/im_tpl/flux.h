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


#ifndef _ELISE_IM_TPL_FLUX
#define _ELISE_IM_TPL_FLUX

class TFlux_Rect2d
{
      public :

          typedef Pt2di  OutFlux;

          Pt2di  PtsInit()
          {
                return Pt2di(_p0.x-1,_p0.y);
          }

          bool next(Pt2di & p)
          {
               p.x++;       
               if (p.x == _p1.x)
               {
                  p.x = _p0.x;
                  p.y++;
               }
               return p.y != _p1.y;
          }

          TFlux_Rect2d(Pt2di p0,Pt2di p1) : _p0(Inf(p0,p1)), _p1 (Sup(p0,p1)) {}

      private :
          Pt2di _p0;
          Pt2di _p1;
};


class TFlux_BordRect2d
{
    public :
       typedef Pt2di  OutFlux;

       Pt2di  PtsInit()
       {
                return Pt2di(_p0.x-1,_p0.y);
       }

       bool next(Pt2di & p)
       {
            p +=TAB_4_NEIGH[_k];
            while  (! p.in_box(_p0,_p1))
            {
                p -= TAB_4_NEIGH[_k];
                _k++;
                if (_k == 4)
                   return false;
                p +=TAB_4_NEIGH[_k];
            }
            return true;
       }

       TFlux_BordRect2d(Pt2di p0,Pt2di p1) : _p0(Inf(p0,p1)), _p1 (Sup(p0,p1)), _k(0) {}

      private :
          Pt2di _p0;
          Pt2di _p1;
          INT   _k;
};


class TFlux_Line2d
{
    public :
       typedef Pt2di  OutFlux;

       Pt2di  PtsInit()
       {
             return Pt2di(_tdl.pcur());
       }

       TFlux_Line2d(Pt2di p1,Pt2di p2,bool conx_8 = true,bool include_p2 = true) :
           _tdl(p1,p2,conx_8,include_p2)
       {
       }

       bool next(Pt2di & p)
       {
           if (_tdl.nb_residu() !=0)
           {
                p  = _tdl.pcur();
                _tdl.next_pt();
                return true;
           }
           else
                return false;
       }


    private :

       Trace_Digital_line  _tdl;

};



template <class TypeFlux,class TypeFonc> class TFluxSelect
{
         public :
            typedef ElTyName TypeFlux::OutFlux   OutFlux;

            TFluxSelect(TypeFlux flux,TypeFonc fonc)  :
                 _flux (flux), 
                 _fonc(fonc) 
            {}

            OutFlux PtsInit()  {return _flux.PtsInit();}

            bool next(OutFlux &p)
            {
                 while (_flux.next(p))
                      if (_fonc.get(p)) 
                         return true;
                 return false;
            }

         private :

              TypeFlux  _flux;
              TypeFonc  _fonc;
};

template <class TypeFlux,class TypeFonc>  
TFluxSelect<TypeFlux,TypeFonc> TSelect(TypeFlux flux,TypeFonc fonc)
{
    return TFluxSelect<TypeFlux,TypeFonc>(flux,fonc);
}



#endif  //  _ELISE_IM_TPL_FLUX











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
