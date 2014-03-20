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







/*------------------------------------------------------------------------------------*/
void FDSupposedWidth ( 	double RealMarks[16], 
						double *Larg, double *Haut )
{
/* si on a perdu le certif de calibration, on peut prendre la largeur moyenne
   de la chambre arrondie */
double dx, dy ;
double l1, l2, l3, l4 ;

/* en microns */
dx = RealMarks[2] - RealMarks[0] ;
dy = RealMarks[3] - RealMarks[1] ;
l1 = sqrt(dx*dx+dy*dy) ;

dx = RealMarks[4] - RealMarks[0] ;
dy = RealMarks[5] - RealMarks[1] ;
l2 = sqrt(dx*dx+dy*dy) ;

dx = RealMarks[2] - RealMarks[6] ;
dy = RealMarks[3] - RealMarks[7] ;
l3 = sqrt(dx*dx+dy*dy) ;

dx = RealMarks[4] - RealMarks[6] ;
dy = RealMarks[5] - RealMarks[7] ;
l4 = sqrt(dx*dx+dy*dy) ;

*Larg = (l1+l2+l3+l4)/4.0 ;

dx = RealMarks[8] - RealMarks[14] ;
dy = RealMarks[9] - RealMarks[15] ;
l1 = sqrt(dx*dx+dy*dy) ;

dx = RealMarks[10] - RealMarks[12] ;
dy = RealMarks[11] - RealMarks[13] ;
l2 = sqrt(dx*dx+dy*dy) ;

*Haut = (l1+l2)/2.0 ;
}
/*------------------------------------------------------------------------------------*/
double FDRotCorrection ( double Marks[16] )
{
int		ii ;
double	*MM[8] ;
double teta1, teta2, teta ;

for ( ii = 0 ; ii < 8 ; ii++ )
	{
	MM[ii] = &(Marks[2*ii]) ;
	}
teta1 = atan ( (MM[3][1]-MM[0][1])/(MM[3][0]-MM[0][0]) ) ;
teta2 = atan ( (MM[1][1]-MM[2][1])/(MM[1][0]-MM[2][0]) ) ;
teta = (teta1+teta2)/2.0 ;
return teta ;
}
/*------------------------------------------------------------------------------------*/
void FDComputeIdealMarks ( double Centered[16], double RealMarks[16], 
						   double IdealMarks[16] )
{
/* La position des marques ideales n'a aucune importance.
   On calcule une position proche des vraies marques par seul souci de confort
   (coordonnees non corrigees proches des coordonnees corrigees) 
   Largeur est donnee en pixels
 */
double teta ;
double xc,yc ;
double cc, ss ;
double *RM[8], *IM[8], *CM[8] ;
int ii ;

for ( ii = 0 ; ii < 8 ; ii++ )
	{
	RM[ii] = &(RealMarks[2*ii]) ;
	IM[ii] = &(IdealMarks[2*ii]) ;
	CM[ii] = &(Centered[2*ii]) ;
	}

/* centre de gravite des marques */
xc = 0.0 ; yc = 0.0 ;
for ( ii = 0 ; ii < 8 ; ii++ )
	{ xc = xc + RM[ii][0] ; yc = yc + RM[ii][1] ; }
xc = xc / 8.0 ;
yc = yc / 8.0 ;
/* angle */
teta = FDRotCorrection ( RealMarks ) ;

/* Marques ideales */
cc = cos(teta) ; ss = sin(teta) ;
for ( ii = 0 ; ii < 8 ; ii++ )
	{
	IM[ii][0] = xc + cc*CM[ii][0] - ss*CM[ii][1] ;
	IM[ii][1] = yc + ss*CM[ii][0] + cc*CM[ii][1] ;
	}
}
/*------------------------------------------------------------------------------------*/
void FDBilinearTransform ( 	double teta, double *XY1, double *XY2, int NPt,
							double PolyCoefs[8] )
{
/* Calcule la transformation ax+by+cxy+d qui transforme les XY1 en XY2 */
double xy ;
double sx, sy, sxy, sx2, sy2, sx2y, sxy2, sx2y2 ;
double sxz, syz , sxyz, sz ;
double* coefs[2] ;
double AA[16] ;
double BB[4] ;
double x1,y1 ,zz ;
double cc,ss,vx,vy ;
int status ; GccUse(status);

int ii, coord ;
double rr, residu ;
int	NCoefs ;

cc = cos(teta) ; ss = sin(teta) ;
sx = 0.0 ; sy = 0.0 ; sxy = 0.0 ; sx2 = 0.0 ; sy2 = 0.0 ;
sx2y = 0.0 ; sxy2 = 0.0 ; sx2y2 = 0.0 ;
sxz = 0.0 ; syz = 0.0 ; sxyz = 0.0 ; sz = 0.0 ;
for ( ii = 0 ; ii < NPt ; ii++ )
	{
	x1 = XY1[2*ii]*cc + XY1[2*ii+1]*ss ;
	y1 = -XY1[2*ii]*ss + XY1[2*ii+1]*cc ;
	xy = x1*y1 ;
	sx = sx + x1 ;
	sy = sy + y1 ;
	sxy = sxy + xy ;
	sx2 = sx2 + x1*x1 ;
	sy2 = sy2 + y1*y1 ;
	sx2y = sx2y + xy*x1 ;
	sxy2 = sxy2 + xy*y1 ;
	sx2y2 = sx2y2 + xy*xy ;
	}
sx = sx / (double) NPt ;
sy = sy / (double) NPt ;
sx2 = sx2 / (double) NPt ;
sy2 = sy2 / (double) NPt ;
sxy = sxy / (double) NPt ;
sx2y = sx2y / (double) NPt ;
sxy2 = sxy2 / (double) NPt ;
sx2y2 = sx2y2 / (double) NPt ;

/* calcul des transfos X et Y */
coefs[0] = &(PolyCoefs[0]) ;
coefs[1] = &(PolyCoefs[4]) ;
for ( coord = 0 ; coord < 2 ; coord++ )
	{
	if ( coord == 0 )
		{
		vx = cc ;
		vy = ss ;
		}
	else
		{
		vx = -ss ;
		vy =  cc ;
		}
	sxz = 0.0 ; syz = 0.0 ; sxyz = 0.0 ; sz = 0.0 ;
	for ( ii = 0 ; ii < NPt ; ii++ )
		{
		x1 = XY1[2*ii]*cc + XY1[2*ii+1]*ss ;
		y1 = -XY1[2*ii]*ss + XY1[2*ii+1]*cc ;
		zz = XY2[2*ii]*vx + XY2[2*ii+1]*vy ;
		sxz = sxz + x1*zz ;
		syz = syz + y1*zz ;
		sxyz = sxyz + x1*y1*zz ;
		sz = sz + zz ;
		}
	sxz = sxz / (double)NPt ;
	syz = syz / (double)NPt ;
	sxyz = sxyz / (double)NPt ;
	sz = sz / (double)NPt ;

	if ( NPt >= 8 )
		{
		NCoefs = 4 ;
		BB[0] = sxz ;
		BB[1] = syz ;
		BB[2] = sxyz ;
		BB[3] = sz ;

		AA[0] = sx2 ; 	AA[1] = sxy ; 	AA[2] = sx2y ; 		AA[3] = sx ;
		AA[4] = sxy ; 	AA[5] = sy2 ; 	AA[6] = sxy2 ; 		AA[7] = sy ;
		AA[8] = sx2y ; 	AA[9] = sxy2 ; 	AA[10] = sx2y2 ; 	AA[11] = sxy ;
		AA[12] = sx ; 	AA[13] = sy ; 	AA[14] = sxy ; 		AA[15] = 1.0 ;

		status = MATSolve ( NCoefs, AA, BB, coefs[coord] ) ;
		}
	else
		{
		NCoefs = 3 ;
		BB[0] = sxz ;
		BB[1] = syz ;
		BB[2] = sz ;

		AA[0] = sx2 ;	AA[1] = sxy ;	AA[2] = sx ;
		AA[3] = sxy ;	AA[4] = sy2 ; 	AA[5] = sy ;
		AA[6] = sx ;	AA[7] = sy ;	AA[8] = 1.0 ;

		/*status = */MATSolve ( NCoefs, AA, BB, coefs[coord] ) ;
		coefs[coord][3] = coefs[coord][2] ;
		coefs[coord][2] = 0.0 ;
		}
	/* Residus */
	residu = 0 ;
/*	for ( ii= 0 ; ii < NPt ; ii++ )*/
		{
		x1 = XY1[2*ii]*cc + XY1[2*ii+1]*ss ;
		y1 = -XY1[2*ii]*ss + XY1[2*ii+1]*cc ;
		zz = XY2[2*ii]*vx + XY2[2*ii+1]*vy ;
		rr = coefs[coord][0]*x1 + coefs[coord][1]*y1 +
			 coefs[coord][2]*x1*y1 + coefs[coord][3] - zz ;
		residu = residu + rr*rr ;
		}
	residu = sqrt(residu/(double)NPt) ;
/* 	printf ( "\nApproximation coordonnee %d : residu = %lf", coord, residu ) ; */
	}
}
/*------------------------------------------------------------------------------------*/
void FDTransformPoint ( double inXY[2], double teta, double Coefs[8], double outXY[2] )
{
double XY0[2], XY1[2] ;
double cc, ss ;
/* Pour memoire, histoire de ne pas se planter dans l'utilisation des polynomes */
cc = cos(teta) ;
ss = sin(teta) ;
XY0[0] =  cc * inXY[0] + ss * inXY[1] ;
XY0[1] = -ss * inXY[0] + cc * inXY[1] ;
XY1[0] = XY0[0]*Coefs[0] + XY0[1]*Coefs[1] + XY0[0]*XY0[1]*Coefs[2] + 
		Coefs[3] ;
XY1[1] = XY0[0]*Coefs[4] + XY0[1]*Coefs[5] + XY0[0]*XY0[1]*Coefs[6] + 
		Coefs[7] ;
outXY[0] = cc * XY1[0] - ss * XY1[1] ;
outXY[1] = ss * XY1[0] + cc * XY1[1] ;
}
/*------------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------------*/
int FDReadMRK ( char *FileName, double IdealMarks[16], double RealMarks[16] )
{
char ligne[255] ;
FILE *fp ;
int ii ;

fp = ElFopen ( FileName, "r" ) ;
if ( fp != 0 )
	{
	VoidFscanf ( fp, "%s", ligne ) ;	/* Commentaires */
	for ( ii = 0 ; ii < 16 ; ii = ii+2 )
		{
		VoidFscanf ( fp, "%lf%lf", &(IdealMarks[ii]), &(IdealMarks[ii+1]) ) ;
		}
	VoidFscanf ( fp, "%s", ligne ) ;	/* Commentaires */
	for ( ii = 0 ; ii < 16 ; ii = ii+2 )
		{
		VoidFscanf ( fp, "%lf%lf", &(RealMarks[ii]), &(RealMarks[ii+1]) ) ;
		}
	ElFclose ( fp ) ;
	return 1 ;
	}
else
	{ return 0 ; }
}
/*------------------------------------------------------------------------------------*/
int FDWriteMRK( char *FileName, double IdealMarks[16], double RealMarks[16] )
{
FILE *fp ;
int ii ;

fp = ElFopen ( FileName, "w" ) ;
if ( fp != 0 )
	{
	fprintf ( fp, "%s\n", " //Marques-a-utiliser-pour-la-mise-en-place-(en-pixels)" ) ;	/* Commentaires */
	for ( ii = 0 ; ii < 16 ; ii = ii+2 )
		{
/* MPD-MODIF */
		fprintf ( fp, " %10.3f %10.3f\n", IdealMarks[ii], IdealMarks[ii+1] ) ;
		}
	fprintf ( fp, "%s\n", " //Position-reelle-des-marques-(en-pixels)" ) ;	/* Commentaires */
	for ( ii = 0 ; ii < 16 ; ii = ii+2 )
		{
/* MPD-MODIF */
		fprintf ( fp, " %10.3f %10.3f\n", RealMarks[ii], RealMarks[ii+1] ) ;
		}
	ElFclose ( fp ) ;
	return 1 ;
	}
else
	{ return 0 ; }
}
/*------------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------------*/
int FDDimsPix ( char *FileName, 
				int *IdPhot1, double *Focal1, double *Pix1, double *Unit1, int *NPt1, 
				int *IdPhot2, double *Focal2, double *Pix2, double *Unit2, int *NPt2 )
{
FILE *fp ;
char ligne[80] ;
int  xx, ii ;

fp = ElFopen ( FileName, "r" ) ;
if ( fp != 0 )
	{
	VoidFscanf ( fp, "%s%d%s%lf%s%lf%s%lf\n",
			 &(ligne[0]),IdPhot1,&(ligne[30]),Focal1,&(ligne[35]),Pix1,&(ligne[40]),Unit1 ) ;
	*NPt1 = 0 ;
	VoidFscanf ( fp, "%d%d%d", &ii, &xx, &xx ) ;
	while ( ii != -1 )
		{
		VoidFscanf ( fp, "%d%d%d\n", &ii, &xx, &xx ) ;
		*NPt1 = *NPt1 + 1 ;
		}
	VoidFscanf ( fp, "%s%d%s%lf%s%lf%s%lf\n",
			 &(ligne[0]),IdPhot2,&(ligne[30]),Focal2,&(ligne[35]),Pix2,&(ligne[40]),Unit2 ) ;
	*NPt2 = 0 ;
	VoidFscanf ( fp, "%d%d%d", &ii, &xx, &xx ) ;
	while ( ii != -1 )
		{
		VoidFscanf ( fp, "%d%d%d", &ii, &xx, &xx ) ;
		*NPt2 = *NPt2 + 1 ;
		}
	ElFclose ( fp ) ;
	return 1 ;
	}
else
	{ return 0 ; }
}
/*------------------------------------------------------------------------------------*/
int FDReadPix ( char *FileName, 
				int NPt1, double Unit1, int *IdPoint1, double *XY1,
				int NPt2, double Unit2, int *IdPoint2, double *XY2 )
{
FILE *fp ;
char ligne[80] ;
int ii, xx, yy ;
int Itoto ;
double Rtoto ;

fp = ElFopen ( FileName, "r" ) ;
if ( fp != 0 )
	{
	VoidFscanf ( fp, "%s%d%s%lf%s%lf%s%lf\n",
			 &(ligne[0]),&Itoto,&(ligne[30]),&Rtoto,&(ligne[35]),&Rtoto,&(ligne[40]),&Rtoto ) ;
	for ( ii = 0 ; ii < NPt1 ; ii++ ) 
		{
		VoidFscanf ( fp, "%d%d%d", &(IdPoint1[ii]), &xx, &yy ) ;
		XY1[2*ii] = Unit1*(double)xx ;
		XY1[2*ii+1] = Unit1*(double)yy ;
		}
	VoidFscanf ( fp, "%d%d%d\n", &ii,&xx,&yy ) ;
	VoidFscanf ( fp, "%s%d%s%lf%s%lf%s%lf\n",
			 &(ligne[0]),&Itoto,&(ligne[30]),&Rtoto,&(ligne[35]),&Rtoto,&(ligne[40]),&Rtoto ) ;
	for ( ii = 0 ; ii < NPt2 ; ii++ ) 
		{
		VoidFscanf ( fp, "%d%d%d", &(IdPoint2[ii]), &xx, &yy ) ;
		XY2[2*ii] = Unit2*(double)xx ;
		XY2[2*ii+1] = Unit2*(double)yy ;
		}
	VoidFscanf ( fp, "%d%d%d\n", &ii,&xx,&yy ) ;
	ElFclose ( fp ) ;
	return 1 ;
	}
else
	{ return 0 ; }
}
/*------------------------------------------------------------------------------------*/
int FDWritePix( char *FileName, 
				int IdPhot1, double Focal1, double Pix1, double Unit1, int NPt1, 
				int *IdPoint1, double *XY1,
				int IdPhot2, double Focal2, double Pix2, double Unit2, int NPt2,
				int *IdPoint2, double *XY2 )
{
FILE *fp ;
int ii, xx, yy ;

fp = ElFopen ( FileName, "w" ) ;
if ( fp != 0 )
	{ 
/* MPD-MODIF */
	fprintf ( fp, " %s%5d%s%7.0f%s%5.0f%5s%5.2f\n" ,
			 	"                  Cliche     ",IdPhot1,
			 	"  Foc",Focal1,
			 	".  Pix",Pix1,
			 	".  Uni",Unit1 ) ;
	for ( ii = 0 ; ii < NPt1 ; ii++ ) 
		{
		xx = (int)(XY1[2*ii]/Unit1 +0.5) ;
		yy = (int)(XY1[2*ii+1]/Unit1 +0.5) ;
		fprintf ( fp, " %6d%8d%8d\n", IdPoint1[ii], xx, yy ) ;
		}
	ii = -1 ; xx = 0 ; yy = 0 ;
	fprintf ( fp, " %6d%8d%8d\n", ii, xx, yy ) ;
	
/* MPD-MODIF */
	fprintf ( fp, " %s%5d%s%7.0f%s%5.0f%5s%5.2f\n" ,
			 	"                  Cliche     ",IdPhot2,
			 	"  Foc",Focal2,
			 	".  Pix",Pix2,
			 	".  Uni",Unit2 ) ;
	for ( ii = 0 ; ii < NPt2 ; ii++ ) 
		{
		xx = (int)(XY2[2*ii]/Unit2 +0.5) ;
		yy = (int)(XY2[2*ii+1]/Unit2 +0.5) ;
		fprintf ( fp, " %6d%8d%8d\n", IdPoint2[ii], xx, yy ) ;
		}
	ii = -1 ; xx = 0 ; yy = 0 ;
	fprintf ( fp, " %6d%8d%8d\n", ii, xx, yy ) ;
	ElFclose ( fp ) ;
	return 1 ;
	}
else
	{ return 0 ; }
}



/*------------------------------------------------------------------------------------*/

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
