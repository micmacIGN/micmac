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
#ifndef _ELISE_HASSAN_REECHANTILLONNAGE_H
#define _ELISE_HASSAN_REECHANTILLONNAGE_H
// #include<values.h>

namespace Reechantillonnage
{
   template<class Type> Type plus_proche_voisin(Type** im, INT tx, INT ty, Pt2dr p)
   {
      INT x = (INT)(p.x + .5);
      INT y = (INT)(p.y + .5);
      if( x <  0 || x >= tx || y <  0 || y >= ty )
        return 0;
      return im[y][x];
   }

/************************************************************************************/

   template<class Type> Type biline (Type** im, INT tx, INT ty, Pt2dr p)//tx ty la taille de l'image en x et y
   {
      INT x_min = (INT)(p.x);
      INT y_min = (INT)(p.y);
      INT x_max = x_min+1;
      INT y_max = y_min+1;

      if( x_min < 0 || y_min < 0  || x_max > tx-1 || y_max > ty-1 )
          return 0;

      REAL p_x_x_min = p.x  - x_min;
      REAL p_x_x_max = p.x  - x_max;
      REAL p_y_y_min = p.y  - y_min;
      REAL p_y_y_max = p.y  - y_max;

      REAL r =   p_x_x_min * p_y_y_min * im[y_max][x_max]
               + p_x_x_max * p_y_y_max * im[y_min][x_min]
               - p_x_x_min * p_y_y_max * im[y_min][x_max]
               - p_x_x_max * p_y_y_min * im[y_max][x_min];

      return (Type)r;
   }

/************************************************************************************/

   template<class Type> Type bicube (Type** im, INT tx, INT ty, Pt2dr p)
   {
                                                           //attention la valeur peut etre negative : probleme apres casting
      if(     (p.x < 1.0) || (p.x >= (REAL)(tx - 2))
           || (p.y < 1.0) || (p.y >= (REAL)(ty - 2))
        )
           return  0;

      INT xc = (INT) ( p.x ) ;
      INT xc__1 = xc - 1;
      INT xc_1  = xc + 1;
      INT xc_2  = xc + 2;
      INT yc = (INT) ( p.y ) ;

      REAL dc  = p.x - xc;
      REAL dc2 = dc * dc;
      REAL dl  = p.y - yc;
      REAL dl2 = dl * dl;

      REAL coefc0 =   -dc * (1 - dc) * (1 - dc);
      REAL coefc1 =   (1 - (2 - dc) * dc2);
      REAL coefc2 =   dc * (1 + dc -  dc2);
      REAL coefc3 =   -dc2 * (1 - dc);

                                        // interpolation sur les colonnes
      int yv = yc - 1;
      REAL valcol0 = coefc0 * im[yv][xc__1]     + coefc1 * im[yv][xc] + coefc2 * im[yv][xc_1] + coefc3 * im[yv][xc_2];
      REAL valcol1 = coefc0 * im[(++yv)][xc__1] + coefc1 * im[yv][xc] + coefc2 * im[yv][xc_1] + coefc3 * im[yv][xc_2];
      REAL valcol2 = coefc0 * im[(++yv)][xc__1] + coefc1 * im[yv][xc] + coefc2 * im[yv][xc_1] + coefc3 * im[yv][xc_2];
      REAL valcol3 = coefc0 * im[(++yv)][xc__1] + coefc1 * im[yv][xc] + coefc2 * im[yv][xc_1] + coefc3 * im[yv][xc_2];

                                       // interpolation sur la ligne


      REAL somme =   (   -dl * (1 - dl) * (1 - dl) * valcol0
                       + (1 - (2 - dl) * dl2)      * valcol1
                       + dl * (1 + dl - dl2)       * valcol2
                       - dl2 * (1 - dl)            * valcol3
                     );

     REAL max; 
     REAL min; // = (REAL) std::numeric_limits<Type>::min();

     if(sizeof(Type) == sizeof(U_INT1) ) { max = 255; min = 0; }
                                        

     if( somme > max ) somme = max;
     else if( somme < min ) somme = min;

      return (Type)(somme);

   }
};

#endif // _ELISE_HASSAN_REECHANTILLONNAGE_H

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
