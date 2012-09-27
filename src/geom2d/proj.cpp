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



Pt2dr  SegComp::proj_ortho_droite(Pt2dr pt) const
{
     return p0()+_tangente * abscisse(pt);
}

Pt2dr  SegComp::proj_ortho_demi_droite(Pt2dr pt) const
{
     REAL a = abscisse(pt);
     return (a <= 0 )               ?
            p0()                    :
            p0()+_tangente *      a ;
}

Pt2dr  SegComp::proj_ortho_seg(Pt2dr pt) const
{
     REAL a = abscisse(pt);
     if (a <= 0) return p0();
     if (a >= _a1) return p1();

     return p0()+_tangente *  a ;
}

REAL  SegComp::abscisse_proj_seg(Pt2dr pt) const
{
   return ElMax(0.0,ElMin(_a1,abscisse(pt)));
}


REAL   SegComp::recouvrement_seg(const Seg2d & s) const
{
    return ElAbs(abscisse_proj_seg(s.p0())-abscisse_proj_seg(s.p1()));
}

Pt2dr  SegComp::proj_ortho(ModePrim mode,Pt2dr pt) const
{
      switch(mode)
      {
           case droite :
                return proj_ortho_droite(pt);

           case demi_droite :
                return proj_ortho_demi_droite(pt);

           default :
                return proj_ortho_seg(pt);
      }
}


void SegComp::proj_ortho
      (
         ModePrim          m0,
         const SegComp &   s1,
         ModePrim          m1,
         REAL &            dmin,
         Pt2dr &           p0min,
         Pt2dr &           p1min
       ) const
{
      for (INT k =0; k<(INT) m1 ; k++)
      {
           Pt2dr p1 = s1.kpts(k);
           Pt2dr p0 = proj_ortho(m0,p1);
           REAL d01 = square_euclid(p0-p1);
           if (d01 < dmin)
           {
               dmin  = d01;
               p0min = p0;
               p1min = p1;
           }
      }
}

Seg2d  SegComp::proj_ortho(ModePrim m0,const SegComp & s1,ModePrim m1) const
{
     bool ok;
     Pt2dr pint = inter(m0,s1,m1,ok);
     if (ok)
        return Seg2d(pint,pint);

     if ((m0 == droite) && (m1 == droite))
        return  Seg2d(p0(),s1.proj_ortho_droite(p0()));

      REAL dmin  = DBL_MAX;
      Pt2dr p0min;
      Pt2dr p1min;

      proj_ortho   (m0,   s1,m1,dmin,p0min,p1min);
      s1.proj_ortho(m1,*this,m0,dmin,p1min,p0min);

      return Seg2d(p0min,p1min);
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
