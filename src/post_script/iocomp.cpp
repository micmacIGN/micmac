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


void Ps_Hex_Code_Int(ofstream & fp,INT i)
{
    i = ElMax(0,ElMin(255,(INT) i));
    fp << hex << (i/16) << (i%16);
    fp << dec;
}


void Ps_Hex_Code_Col(ofstream & fp,REAL v)
{
    Ps_Hex_Code_Int(fp,(INT)(v*256));
}

void Ps_Real_Prec(ofstream & fp,REAL v,INT nb_apr_virg)
{
    if ((nb_apr_virg > 6) || (v > 2e9) || (v < -2e9))
    {
      fp << v;
      return;
    }

    if (v<0)
    {
       v = -v;
       fp << "-";
    }


    REAL a =  floor(v);
    fp << a;

    INT pow10 = 1;
    for (INT i = 0; i<nb_apr_virg; i++)
        pow10 *= 10;
    INT  b =  (INT)((v-a)*pow10);

    INT nbc = nb_apr_virg;
    {
        INT p10 = 10;

        while (nbc && (b%p10 ==0))
        {
            nbc--;
            p10 *= 10;
        }
    }

    if (nbc)
    {
        fp << ".";
        pow10  /= 10;
        while (nbc)
        {
            nbc --;
            fp << (b/pow10);
            b = b % pow10;
            pow10 /= 10;
        }
    }

}

void Ps_Real_Prec(ofstream & fp,Elise_colour c,INT nb_apr_virg)
{

    Ps_Real_Prec(fp,c.r(),nb_apr_virg);
    fp << " ";
    Ps_Real_Prec(fp,c.g(),nb_apr_virg);
    fp << " ";
    Ps_Real_Prec(fp,c.b(),nb_apr_virg);
}

void Ps_Real_Col_Prec(ofstream & fp,INT r,INT g,INT b,INT nb_apr_virg)
{
    Ps_Real_Prec(fp,Elise_colour::rgb(r/256.0,g/260.0,b/256.0),nb_apr_virg);
}

void Ps_Pts(ofstream & fp,Pt2dr pt,INT nb_apr_virg)
{
   Ps_Real_Prec(fp,pt.x,nb_apr_virg);
   fp << " ";
   Ps_Real_Prec(fp,pt.y,nb_apr_virg);
   fp << " ";
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
