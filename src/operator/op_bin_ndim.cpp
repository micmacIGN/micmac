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





/***************************************************************/
/***************************************************************/
/****                                                       ****/
/****                                                       ****/
/****        DIAGONALISATION DE MATRICES                    ****/
/****                                                       ****/
/****                                                       ****/
/***************************************************************/
/***************************************************************/

void op_diag_m2_sym(REAL **o,REAL **in,INT nb)
{
    REAL * taxx = in[0];
    REAL * taxy = in[1];
    REAL * tayy = in[2];


    REAL * tvp1 = o[0];
    REAL * tvp2 = o[1];
    REAL * tang = o[2];

    
    for (INT i=0 ; i< nb ; i++)
    {
        REAL sxx = taxx[i];
        REAL sxy = taxy[i];
        REAL syy = tayy[i];

        REAL vp1,vp2,ang;

        if ((sxx == syy) && (sxy == 0.0))
        {
            ang = 0.0;
            vp1 = vp2 = sxx;
        }
        else
        {
             REAL  phi = atan2(sxy,(syy-sxx)/2.0);
             ang = (PI-phi)/2.0;

             REAL  c = cos(ang);
             REAL  s = sin(ang);


             if (ElAbs(c)>ElAbs(s))
                vp1 = (sxx*c+sxy*s)/c;
             else
                vp1 = (sxy*c+syy*s)/s;
    
             vp2 = sxx+syy-vp1;

             if (vp1 < vp2)
             {
                ElSwap(vp1,vp2);
                ang += PI/2.0;
             }
        }

        tvp1[i] = vp1;
        tvp2[i] = vp2;
        tang[i] = ang;
    }
}


Fonc_Num diag_m2_sym(Fonc_Num f)
{
   return op_un_3d_real(f,op_diag_m2_sym);
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
