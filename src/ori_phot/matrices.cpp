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




#ifndef _abs
#define _abs( a ) ( (a) < 0 ? -(a) : (a) )
#endif



void MATOrdonneLignes ( double**, int, int ) ;
int MATNormaliseLigne ( double*, int ) ;
int MATPivot ( double**, int , int ) ;
int MATDiagInvert ( double**, int, int, double* ) ;

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
int MATSolve ( int dim, double *MM, double *YY, double *XX )
{
/* Resolution de MM.XX = YY, ou dim est la dimension du systeme */
double		**AA ;
double		*Copie ;
int			ii, jj ;
int			status ;

status = 1 ;
Copie = (double*) malloc ( (dim+1)*dim*sizeof(double) ) ;
if ( Copie == 0 ) return 0 ;
AA = (double**) malloc ( dim*sizeof(double*) ) ;
if ( AA == 0 ) { free ( Copie ) ; return 0 ; }

/* Remplissage des pointeurs */
for ( jj = 0 ; jj < dim ; jj++ )
	{
	AA[jj] = &(Copie[jj*(dim+1)]) ;
	}
/* Copie des donnees d'entree */
for ( jj = 0 ; jj < dim ; jj++ )
	{
	/* Contenu de la matrice */
	for ( ii = 0 ; ii < dim ; ii++ ) { AA[jj][ii] = MM[ii+jj*dim] ; }
	/* Second membre */
	AA[jj][dim] = YY[jj] ;
	}

/* Normalisation */
for ( jj = 0 ; jj < dim ; jj++ )
	{
	status = MATNormaliseLigne ( AA[jj], dim ) ;
	if ( status != 1 ) { jj = dim ; } /* sortie de boucle en cas d'echec */
	}

/* Reduction a un systeme diagonal */
if ( status == 1 ) 
	{
	for ( jj = 0 ; jj < dim-1 ; jj++ )
		{
		MATOrdonneLignes ( AA, dim, jj ) ;
		status = MATPivot ( AA, dim , jj ) ;
		if ( status != 1 ) { jj = dim ; } /* sortie de boucle en cas d'echec */
		}
	}

/* Resolution du systeme diagonal */
if ( status == 1 ) 
	{
	for ( jj = dim-1 ; jj >= 0 ; jj-- )
		{ 
		status = MATDiagInvert ( AA, dim, jj, XX ) ;
		if ( status != 1 ) { jj = -1 ; } /* sortie de boucle en cas d'echec */
		}
	}

free ( AA ) ;
free ( Copie ) ;
return status ;
}
/*----------------------------------------------------------------------------*/
int MATDiagInvert ( double **AA, int dim, int pos, double *Result )
{
/* 	Result est deja rempli pour les lignes superieures a pos
	Les coef d'indice inferieur a pos sur la ligne pos sont nuls
 */
double 		VV ;
int			ii ;

if ( AA[pos][pos] == 0.0 ) return 0 ;
VV = 0.0 ;
for ( ii = pos+1 ; ii < dim ; ii++ ) { VV = VV + AA[pos][ii]*Result[ii] ; }
Result[pos] = ( AA[pos][dim] - VV ) / AA[pos][pos] ;
return 1 ;
}
/*----------------------------------------------------------------------------*/
void MATOrdonneLignes ( double **AA, int dim, int pos )
/* 
 La matrice est stockee dans les dim premiers elements de chaque ligne
 Le dernier element contient la valeur du second membre
 Fonction :
 Recherche le pivot le plus fort en colonne pos dans les lignes d'indice
 superieur a pos et place cette ligne en ligne pos
 */
{
int		jmax, jj ;
double	VV, Vmax ;
double* tmp ;

jmax = pos ;
Vmax = _abs ( AA[pos][pos] ) ;
for ( jj = pos+1 ; jj < dim ; jj++ )
	{
	VV = _abs ( AA[jj][pos] ) ;
	if ( VV > Vmax ) { Vmax = VV ; jmax = jj ; }
	}
tmp = AA[jmax] ;
AA[jmax] = AA[pos] ;
AA[pos] = tmp ;
}
/*----------------------------------------------------------------------------*/
int MATNormaliseLigne ( double *Ligne, int dim ) 
{
int 	ii ;
double 	Vmax, VV ;
Vmax = 0.0 ;
for ( ii = 0 ; ii < dim ; ii++ )
	{
	VV = _abs(Ligne[ii]) ;
	if ( VV > Vmax ) { Vmax = VV ; }
	}
if ( Vmax == 0.0 ) return 0 ;
for ( ii = 0 ; ii < dim+1 ; ii++ ) { Ligne[ii] = Ligne[ii] / Vmax ; }
return 1 ;
}
/*----------------------------------------------------------------------------*/
int MATPivot ( double **AA, int dim , int pos )
{
/* 
 La matrice est stockee dans les dim premiers elements de chaque ligne
 Le dernier element contient la valeur du second membre
 Fonction :
 elimine l'element ii sur tte les lignes d'indice superieur a pos 
 les elements d'indice colonne < pos sont suposes nuls sur ces lignes
 */
double 	coef1, coef2 ;
int 	ii,jj ;
int 	MaxNonNul, status ;

coef1 = AA[pos][pos] ;
if ( coef1 == 0.0 ) return 0 ;

status = 1 ;
for ( jj = pos+1 ; jj < dim ; jj++ )
	{
	coef2 = AA[jj][pos] ;
	AA[jj][pos] = 0.0 ;
	for ( ii = pos+1 ; ii < dim+1 ; ii++ )
		{
		AA[jj][ii] = coef1*AA[jj][ii] - coef2*AA[pos][ii] ;
		}
	MaxNonNul = MATNormaliseLigne ( AA[jj], dim ) ;
	if ( MaxNonNul != 1 ) { status = 0 ; }
	}
return status ;
}
/*----------------------------------------------------------------------------*/




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
