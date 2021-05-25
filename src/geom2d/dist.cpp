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



REAL SegComp::square_dist_droite(Pt2dr pt) const
{
    return ElSquare(ordonnee(pt));
}
REAL SegComp::dist_droite(Pt2dr pt) const
{
    return ElAbs(ordonnee(pt));
}


REAL SegComp::square_dist_demi_droite(Pt2dr pt) const
{
    REAL a = abscisse(pt);
    REAL o2 =  ElSquare(ordonnee(pt));
 
    if (a < 0)
       return o2 + ElSquare(a);
    else
       return o2;
}
REAL SegComp::dist_demi_droite(Pt2dr pt) const
{
    return sqrt(square_dist_demi_droite(pt));
}




REAL SegComp::square_dist_seg(Pt2dr pt) const
{
    REAL a = abscisse(pt);
    REAL o2 =  ElSquare(ordonnee(pt));
 
    if (a < 0)
    {
       return o2 + ElSquare(a);
    }
    else if (a < _a1)
    {
       return o2;
    }
    else
    {
       return o2 + ElSquare(a-_a1);
    }
}
REAL SegComp::dist_seg(Pt2dr pt) const
{
    return sqrt(square_dist_seg(pt));
}

REAL  SegComp::square_dist(ModePrim  mode,Pt2dr   pt) const
{
      switch (mode)
      {
           case droite :
                return square_dist_droite(pt);

           case demi_droite :
                return square_dist_demi_droite(pt);

           default :
                return square_dist_seg(pt);
      }
}
REAL  SegComp::dist(ModePrim  mode,Pt2dr   pt) const
{
    return sqrt(square_dist(mode,pt));
}

REAL  SegComp::_square_dist
      (
          ModePrim m0,
          const SegComp & s1,
          ModePrim m1
      ) const
{
    REAL res = DBL_MAX;
    for (INT k=0; k<(int)m1; k++)
        res = ElMin
              (
                  res,
                  square_dist(m0,s1.kpts(k))
              );
   return res;
}




REAL  SegComp::square_dist
      (
          ModePrim m0,
          const SegComp & s1,
          ModePrim m1
      ) const
{

   bool got;
   inter(m0,s1,m1,got);
   if (got) return 0.0;

   if ((m0== droite) && (m1 == droite)) 
      return square_euclid(p0()-s1.proj_ortho_droite(p0()));

   return ElMin
          (
             _square_dist(m0,s1,m1),
             s1._square_dist(m1,*this,m0)
          );
}

REAL  SegComp::dist
      (
          ModePrim m0,
          const SegComp & s1,
          ModePrim m1
      ) const
{
    return sqrt(square_dist(m0,s1,m1));
}


   /*********************************************************/
   /*                                                       */
   /*        HAUSSDORF                                      */
   /*                                                       */
   /*********************************************************/

REAL SegComp::square_haussdorf_seg_assym(const SegComp & s2) const
{
     return ElMax
            (
                s2.square_dist_seg(p0()),
                s2.square_dist_seg(p1())
            );
}

REAL SegComp::square_haussdorf_seg_sym(const SegComp & s2) const
{
     return ElMax
            (
                this->square_haussdorf_seg_assym(s2),
                s2.square_haussdorf_seg_assym(*this)
            );
}


REAL SegComp::square_haussdorf_droite_assym(const SegComp & s2) const
{
     return ElMax
            (
                s2.square_dist_droite(p0()),
                s2.square_dist_droite(p1())
            );
}

REAL SegComp::square_haussdorf_droite_sym(const SegComp & s2) const
{
     return ElMax
            (
                this->square_haussdorf_droite_assym(s2),
                s2.square_haussdorf_droite_assym(*this)
            );
}








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
