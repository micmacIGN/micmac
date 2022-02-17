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

#include "all_phot.h"




#define FUSEAU ASkHeureLocaleVraie


/*--------------------------------------------------------------------------------------*/
void orvecteur_soleil_terre (  int *annee, int *jour, int *mois, int *heure, int *minute, 
								double *latitude, double *longitude,
			      				double soleil[3] )
{
int Year ;
double HmsTU ;
double JJ0h, JJ ;
double AlphaSol, DeltaSol ;
double TS ;
double AzNord, Hauteur ;
const double Pi_ = atan(1.0)*4.0 ;

Year = *annee ;
if ( Year < 100 ) 
	{
	if ( Year < 50 ) 
		{ Year = Year + 2000 ; }
	else
		{ Year = Year + 1900 ; }
	}

JJ0h = ASJourJulien ( Year, *mois, *jour ) ;
/* 1) Heure TU (Heure Moyenne a Grennwitch) */
HmsTU = (double)*heure+ ((double)*minute) / 60.0 ;
HmsTU = ASTempsUniversel ( JJ0h, HmsTU, - *longitude/15.0, FUSEAU ) ;

/* 2) Ascension droite et declinaison du soleil */
JJ = JJ0h + HmsTU/24.0 ;
ASAlphaDeltaSoleil ( JJ, &AlphaSol, &DeltaSol ) ;

/* 3) Temps sideral du lieu */
TS = ASTempsSideralMoyen ( JJ0h, HmsTU, - *longitude/15.0 ) ;
TS = TS*Pi_ / 12.0 ;

/* 4) Azimut et Hauteur dans le repere local */
ASAlphaDeltaToAzHt ( TS, *latitude * Pi_/180.0, AlphaSol, DeltaSol, &AzNord, &Hauteur ) ;

/* 5) Vecteur en coordonnees cartesiennes */
soleil[2] = - sin ( Hauteur ) ;
soleil[1] = - cos ( Hauteur ) * cos ( AzNord ) ;
soleil[0] = - cos ( Hauteur ) * sin ( AzNord ) ;
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
