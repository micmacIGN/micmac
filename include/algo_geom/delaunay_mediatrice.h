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

#ifndef _ELISE_ALGO_GEOM_DELAUNAY_MEDIATRICE
#define _ELISE_ALGO_GEOM_DELAUNAY_MEDIATRICE

#include "StdAfx.h"


typedef long  double tRationnelDelaunay;

inline int CmpRat(const tRationnelDelaunay & D1,const  tRationnelDelaunay & D2)
{
   tRationnelDelaunay aDif = D1-D2;
   if (aDif > 0) return 1;
   if (aDif < 0) return -1;
   return 0;
}

template <class Iterator,class Fpt,class Act,class Type>  void  
          Delaunay_Mediatrice
          (
                 Iterator begin,
                 Iterator end,
                 Fpt   fpt,
                 Act & act,
                 REAL  dist,
                 Type  *
          )
{
    int I,J,K;
    int continuer_K;
    tRationnelDelaunay A_x,A_y,AB_x,AB_y,AC_x,AC_y,norme_AB,norme_AC,det_AB_AC,scal_AB_AC;
    tRationnelDelaunay alpha,beta,nouv_val;
    int etat,valeur_K[2],dec_etat[2];

    std::vector<Type *> F;
    {
        for (Iterator it = begin ; it != end ; it++)
            F.push_back(&(*it));
    }
    Type ** vals =&(F[0]);
    INT nb = (int)F.size();


    {
        ElComparPtOn_x_then_y<Type ,Fpt> Cmp(fpt);
        STDSORT(vals,vals+nb,Cmp);                       
    }
   
    {
        ElComparPts<Type ,Fpt> CmpEq(fpt);
        Type ** newlast = STDUNIQUE(vals,vals+nb,CmpEq);
        nb = (int)(newlast - vals);
    }

    std::vector<INT> MARQUEUR_K;

    for (I = 0 ; I < nb ; I++)
    {
        /* calcul des coordonnees du point A */
        Pt2dr pi = fpt(*vals[I]);
        A_x = pi.x;
        A_y = pi.y;
        MARQUEUR_K.clear();
        for ( J = 0; J < nb ; J++)
            MARQUEUR_K.push_back(0);
        REAL x_lim = A_x + dist;
        for(J = I+1 ; (J < nb) && ( fpt(*vals[J]).x <x_lim) ; J++)
        {
            Pt2dr pj = fpt(*vals[J]);
            if (euclid(pi,pj) < dist)
            {
               /* calcul des coordonnees et de la norme du vecteur AB */
               AB_x = pj.x - A_x;
               AB_y = pj.y - A_y;
               norme_AB = AB_x * AB_x + AB_y * AB_y;
               /* initialisation de l'intervalle */
               alpha = -1e60;
               beta = 1e60;
               continuer_K = nb - 2;
               etat = 0;
               valeur_K[0] = valeur_K[1] = (I+J)/2;
               dec_etat[0] = 1;
               dec_etat[1] = -1;
               MARQUEUR_K[I] = MARQUEUR_K[J] = J;
               while(continuer_K)
               {
                  continuer_K --;
                  K = valeur_K[etat];
                  if (MARQUEUR_K[K] != J)
                  {
                     MARQUEUR_K[K] = J;
                     /* calcul des coordonnees du vecteur AC */
                     Pt2dr pk = fpt(*vals[K]);
                     AC_x = pk.x - A_x; 
                     AC_y = pk.y - A_y;
                     /* calcul des produits scalaires et vectoriels de AB et AC */
                     det_AB_AC = AB_x * AC_y - AB_y * AC_x;	
                     scal_AB_AC = AB_x * AC_x + AC_y * AB_y;	
                     /* si A,B,C aligne */
                     if (det_AB_AC == 0)  
                     {
                        /* si C appartient au segment [A,B] */
                        if ( (scal_AB_AC > 0) && (scal_AB_AC < norme_AB))
                        {
                           /* alors AB ne sera pas un arc de la triangulation */
                           alpha = 1;
                           beta = -1;
                        }
                           /* sinon C ne refute aucune partie de la mediatrice de AB
                              donc on ne fait rien */
                     }
                     else
                     {
                         norme_AC = AC_x * AC_x + AC_y * AC_y;
                         /* si C est a "gauche" du vecteur AB on va
                          eventuellement modifier beta */
                        if (det_AB_AC > 0)
                        {
                           /* calcul de l'abscisse de l'intersection des mediatrices */
                           nouv_val = (norme_AC - scal_AB_AC) /det_AB_AC;
                           /* beta = min (beta,nouv_val) */
                           if (1 == CmpRat(beta,nouv_val))
                           {
                              beta = nouv_val;
                           }
                        }
                         /* si C est a "droite" du vecteur AB on va
                          eventuellement modifier alpha */
                        else
                        {
                           /* calcul de l'abscisse de l'intersection des mediatrices */
                           nouv_val = (- norme_AC + scal_AB_AC)
                                        / ( - det_AB_AC);
                           /* alpha = max (alpha,nouv_val) */
                           if ( -1 ==  CmpRat(alpha,nouv_val))
                           {
                              alpha = nouv_val;
                           }
                        }
                     }
                     /* si alpha > beta il est inutile de continuer */
                    if ( 1 == CmpRat(alpha,beta))
                        continuer_K = 0;
                  }	
                  /* on memorise le prochain K pour dans 2 etapes */
                  valeur_K[etat] += dec_etat[etat];
                  /* si on atteint les bords il vaudrait mieux s'arreter la */
                  if ((valeur_K[etat] < 0) || (valeur_K[etat] == nb))
                  {
                     valeur_K[etat] -= dec_etat[etat];
                     dec_etat[etat] = 0;
                  }
                  /* si on explorait K dans les x croissants alors on va explorer dans les x decroissants
                     et lycee de Versailles */
                  etat = (etat + 1) % 2;
               }
               /* si alpha > beta IJ est un arc de la triangulation */
               INT cmp = CmpRat (alpha,beta);
               if ( cmp <=0)
                  act(*vals[I],*vals[J],cmp==0);
           }
        }
    }
}





template <class Type,class Fpt,class Act>  void  
          Delaunay_Mediatrice
          (
                 Type * vals,
                 INT   nb,
                 Fpt   fpt,
                 Act & act,
                 REAL  dist
          )
{
    Delaunay_Mediatrice(vals,vals+nb,fpt,act,dist,(Type *)0);
}

                
#endif // _ELISE_ALGO_GEOM_DELAUNAY_MEDIATRICE
               

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
