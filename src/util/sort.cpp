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


void Rset_min_max(REAL & v1, REAL & v2)
{
    if (v1>v2)
    {
       REAL tmp = v1;
       v1 = v2;
       v2 = tmp;
    }
}


template <class Type> void  elise_sort
      (
            Type * v,
            INT   nb
      )
{
    switch(nb)
    {
        case 0:   
        case 1: return;
        case 2:
                 Rset_min_max(v[0],v[1]);
               return;
        case 3:
                  Rset_min_max(v[0],v[1]);
                  Rset_min_max(v[0],v[2]);
                  Rset_min_max(v[1],v[2]);
            return;
        case 4:
                  Rset_min_max(v[0],v[1]);
                  Rset_min_max(v[2],v[3]);

                  Rset_min_max(v[0],v[2]);
                  Rset_min_max(v[1],v[3]);
                  Rset_min_max(v[1],v[2]);
            return;
        default :
        {
              INT l,j,ir,i;
              Type rra;

              v--;
              l=(nb >> 1)+1;
              ir=nb;
              for (;;) 
              {
                  if (l > 1)
                     rra=v[--l];
                  else 
                  {
                     rra=v[ir];
                     v[ir]=v[1];
                     if (--ir == 1) 
                     {
                        v[1]=rra;
                        return;
                     }
                  }
                  i=l;
                  j=l << 1;
                  while (j <= ir) 
                  {
                     if (j < ir && (v[j]<=v[j+1])) 
                        ++j;
                     if (rra<=v[j]) 
                     {
                        v[i]=v[j];
                        j += (i=j);
                     }
                     else 
                        j=ir+1;
                  }
                  v[i]=rra;
              }
        }
    }
}

template <class Type>  void elise_indexe_sort
     (
            Type *   v,
            INT  *   indexe,
            INT      nb
     )

{
        INT l,j,ir,i;
        INT rra;

        if (nb <= 1)
                return;

        for (INT k=0; k<nb; k++)
            indexe[k] = k;

        indexe--;
        l=(nb >> 1)+1;
        ir=nb;
        for (;;) 
        {
            if (l > 1)
                    rra=indexe[--l];
            else 
            {
                    rra=indexe[ir];
                    indexe[ir]=indexe[1];
                    if (--ir == 1) 
                    {
                       indexe[1]=rra;
                       return;
                    }
            }
            i=l;
            j=l << 1;
            while (j <= ir) 
            {
                 if ((j<ir)&&(v[indexe[j]]<=v[indexe[j+1]]))
                    ++j;
                 if (v[rra]<=v[indexe[j]])
                 {
                    indexe[i]=indexe[j];
                    j += (i=j);
                 }
                    else j=ir+1;
            }
            indexe[i]=rra;
        }
}


template  void  elise_sort(REAL * v,INT  nb);
template  void elise_indexe_sort(REAL *   v,INT *  indexe,INT  nb);



void RecGetSubset(std::vector<std::vector<int> > & aRes,int aNb,int aMax)
{
   if (aNb==0)
   {
       std::vector<int> aVide;
       aRes.push_back(aVide);
       return;
   }

   if (aNb==aMax)
   {
       std::vector<int> aFull;
       for (int aK=0 ; aK<aNb; aK++)
           aFull.push_back(aK);
       aRes.push_back(aFull);
       return;
   }

   // Ceux qui n'ont pas aMax
   RecGetSubset(aRes,aNb,aMax-1);
   int aNbRes = (int)aRes.size();
   // Ceux qui ont  aMax
   RecGetSubset(aRes,aNb-1,aMax-1);
   for (int aK=aNbRes ; aK<int(aRes.size()) ; aK++)
       aRes[aK].push_back(aMax-1);
}

void GetSubset(std::vector<std::vector<int> > & aRes,int aNb,int aMax)
{
    aRes.clear();
    if ((aNb<0) || (aMax<0) || (aNb>aMax))
       return;
    RecGetSubset(aRes,aNb,aMax);
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
