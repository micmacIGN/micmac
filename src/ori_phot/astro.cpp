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



/* Formules astronomiques classiques - equinoxe moyen 1900 - precession et nutation
   non prises en compte ; 
   
 validees par comparaison des azimut et hauteur obtenus avec les valeurs publiees
 par le BDL (3615 BDL)
 
 La precision (ERREUR MAX) est meilleure que la demi minute d'arc sur les azimuts
 
 Elle reste meilleure que la demi minute pour les hauteur superieures a 14 60 degres
 Lorsque le soleil est tres bas, le modele de refraction est sans doute trop pauvre : 
 la precision reste toutefois (a priori) meilleure que quelques (2 a 3) minutes d'arcs 
 (les deux ou trois tests effectues n'ont pas montre d'erreur superieure a la demi-minute)
 
	Echantillon de reference (source 3615 BDL) :
	--------------------------------------------
	Lieu du calcul : Paris (-2¡20',48¡50')
	Annee du calcul: 1996
	Jour  du calcul: 15 de chaque mois
	Heure du calcul: 10h00 (TU)
	Azimuts comptes vers l'Est a partir du Sud en degres-minutes
	Hauteur au dessus de l'horizon en degres-minutes
	
	Donnees BDL :
			Mois		Azimut Est		Hauteur
			
			1			28¡49'			15¡5'
			2			33¡8'			22¡28'
			3			36¡26'			33¡2'
			4			40¡6'			44¡47'
			5			45¡16'			53¡12'
			6			50¡40'			56¡24'
			7			50¡42'			54¡8'
			8			44¡5'			47¡50'
			9			34¡46'			38¡47'
			10			27¡24'			28¡44'
			11			23¡56'			19¡21'
			12			24¡59'			14¡10'
	Pour les deux angles, emq = 0.25 minutes d'arc sur l'echantillon ci-dessus (compatible
	avec la simple erreur de quantification)
	
 ===============================
 NON TESTE dans l'hemisphere SUD
 ===============================
 */
 
/*=========================================================================================*/

#include "StdAfx.h"
#include "all_phot.h"

/* Les deux Soleils programmes S1 et S2 donnent pratiquement les memes resultats 
   S2 semble legerment plus precis */
#define _SOLEILS2_

/* On utilise un modele de refraction simplifie : couche fine refractrice (d'environ 3km8)
   autour de la sphere terrestre.
   Deux parametres : _REFRAC_ = indice de refraction
   					 _RATM_ = ratio entre le rayon de l'enveloppe externe de la couche et
   					 		 le rayon terrestre
   Ces parametres ont ete calibres pour :
   		- avoir une hauteur quasi nulle au coucher du soleil (Hauteur vraie = -36'36'')
   		- donner des hauteurs compatibles avec celles du BDL (echantillon de 12 dates a 10h)
*/
#define _REFRAC_ 	0.999689334064
#define _RATM_  	1.000595218750



/*=================================================================================*/
/*	Routines Internes */
typedef struct
	{
	double T ;	/* Jour Julien en Siecles Juliens */
	double L ; /* Longitude du perihelie + Anomalie Moyenne */
	double M ; /* anomalie moyennes */
	double V ; /* anomalie vraie */
	double LambdaApp ; /* longitude geometrique apparente */
	double LambdaMoy ; /* longitude geometrique moyenne */
	double e ; /* excentricite */
	double EpsilonApp ; /* obliquite apparente */
	double EpsilonMoy ; /* obliquite moyenne */
	double Abberation ; /* distance horaire du soleil */
	} TPositionSoleil ;
void 	ASCalculeSoleil ( double /*JourJulien*/, TPositionSoleil* /*SunRecord*/ ) ;
double 	ASCorrectionHoraire ( TPositionSoleil* /*SunRecord*/ ) ;
void 	ASCorrectionPlanete ( TPositionSoleil* /*SunRecord*/ ) ;
double	ASSiecleJulien ( double /*JourJulien*/ ) ;
double 	ASCorrigeHauteur ( 	double /*Hauteur*/, 
							double /*RayonAtmo*/, double /*Refraction*/ ) ;
double 	ASDHEntreeAtmosphere ( double /*Hauteur*/, double /*RayonAtmo*/ ) ;
#ifndef _abs
#define _abs( a ) ( (a) < 0 ? -(a) : (a) )
#endif

/*-----------------------------------------------------------------------------------*/
void 	ASAlphaDeltaSoleil ( double JourJulien, double *AlphaSoleil, double *DeltaSoleil )
{
TPositionSoleil Sun ;
double 			LgEclSoleil ;
const double Pi_ = 4.0*atan(1.0) ;

ASCalculeSoleil ( JourJulien, &Sun ) ;
/* 1) Longitude Ecliptique du soleil */
LgEclSoleil = - Sun.LambdaApp ;
if ( LgEclSoleil < 0.0 ) LgEclSoleil = LgEclSoleil + 2.0*Pi_ ;

/* 2) Ascension Droite et declinaison soleil */
*AlphaSoleil = cos(Sun.EpsilonApp)*tan(LgEclSoleil) ;
if ( (*AlphaSoleil > -1.0)&&(*AlphaSoleil<1.0) )
	{ *AlphaSoleil = atan (*AlphaSoleil) ; }
else
	{ *AlphaSoleil = Pi_/2.0 -atan(1.0 / *AlphaSoleil) ; }
if ( *AlphaSoleil < 0.0 ) *AlphaSoleil = *AlphaSoleil + 2.0*Pi_ ;
if ( ElAbs(*AlphaSoleil-LgEclSoleil) > Pi_/2.0 ) 
	{
	*AlphaSoleil = *AlphaSoleil + Pi_ ;
	if ( *AlphaSoleil > 2.0*Pi_ ) { *AlphaSoleil = *AlphaSoleil - 2.0*Pi_ ; }
	}
*DeltaSoleil = -sin(Sun.EpsilonApp)*sin(LgEclSoleil) ;
if ( (*DeltaSoleil > -0.5) && (*DeltaSoleil < 0.5) ) 
	{ *DeltaSoleil = asin ( *DeltaSoleil ) ; }
else
	{ *DeltaSoleil = Pi_/2.0 -  acos ( *DeltaSoleil ) ; }
}
/*-----------------------------------------------------------------------------------*/
void	ASAlphaDeltaToAzHt ( double TempsSideralMoyen, double LatitudeNord, 
								double AscensionDroite, double Declinaison, 
								double *AzimutNord, double *Hauteur )
/* tout en radians ; les azimuts par rapport au Nord et vers l'Est */
{
double Teta ;
const double Pi_ = 4.0*atan(1.0) ;
double vecteur[3], av0, av1 ;

Teta = TempsSideralMoyen + AscensionDroite ;
if ( Teta > 2.0*Pi_ ) { Teta = Teta - 2.0*Pi_ ; }
vecteur[0] = -cos(Declinaison)*sin(Teta) ;
vecteur[1] = sin(Declinaison)*cos(LatitudeNord) - cos(Declinaison)*sin(LatitudeNord)*cos(Teta) ;
vecteur[2] = sin(Declinaison)*sin(LatitudeNord) + cos(Declinaison)*cos(LatitudeNord)*cos(Teta) ;
/* Azimut */
av0 = _abs ( vecteur[0] ) ;
av1 = _abs ( vecteur[1] ) ;
if ( av0 >= av1 )
	{
	if ( av0 == 0 ) 
		{
		*AzimutNord = 0.0 ;
		}
	else
		{
		*AzimutNord = atan ( av0 / av1 ) ;
		}
	}
else
	{
	*AzimutNord = Pi_/2.0 - atan ( av1 / av0 ) ;
	}
if ( vecteur[1] < 0 ) { *AzimutNord = Pi_ - *AzimutNord ; }
if ( vecteur[0] < 0 ) { *AzimutNord = - *AzimutNord ; }

/* Hauteur avec correction de refraction */
if ( vecteur[2] < 0.707 )
	{ *Hauteur = asin (vecteur[2]) ; }
else
	{ *Hauteur = Pi_/2.0 - acos(vecteur[2]) ; }
*Hauteur = ASCorrigeHauteur ( *Hauteur, _RATM_, _REFRAC_ ) ;
}
/*-----------------------------------------------------------------------------------*/
double ASCorrectionHoraire ( TPositionSoleil *Sun )
/* Retourne la valeur a rajouter a l'heure vraie pour obtenir l'heure moyenne 
   Equation du Temps p. 55
 */
{
double ET ;

/* en Secondes */
ET = 459.74*sin((*Sun).M) + 4.80*sin(2.0*(*Sun).M) - 
	 591.89*sin(2.0*((*Sun).L)) + 12.74*sin(4.0*((*Sun).L)) ;
/*printf ( "\n------> Correction horaire en minutes = %lf", ET/60.0 ) ;
 En Heures */
ET = ET/3600.0 ;
return ET ;
}
/*-----------------------------------------------------------------------------------*/
double ASTempsSideralMoyen ( double JourJulien0Heure, double HmsTU,
							 double LongitudeOuestEnHeures )
{
/* methode n.2 p.58 - temps sideral rapporte a l'equinoxe moyen 1900 */
double T0, Sideral ;

T0 = ASSiecleJulien ( JourJulien0Heure ) ;
/* Temps sideral greenwitch a 0 h */
Sideral = 6.64606556 + 2400.051262*T0 + 0.0000258*T0*T0 ;
if ( abs((int)Sideral)>=24 ) { Sideral = Sideral - (double)(24*(int)(Sideral/24.0)) ; }
/* Prise en compte de la longitude Ouest du lieu :
LA LONGITUDE EST PRISE EN COMPTE SI ON DONNE L'HEURE A GREENWITCH */
Sideral = Sideral - LongitudeOuestEnHeures + HmsTU*1.0027379093 ;
if ( abs((int)Sideral)>=24 ) { Sideral = Sideral - (double)(24*(int)(Sideral/24.0)) ; }
if ( Sideral < 0.0 ) Sideral = Sideral + 24.0 ;
return Sideral ;
}
/*-----------------------------------------------------------------------------------*/
double ASJourJulien ( int Annee, int Mois, int Jour )
/* 
Methode de A.RICHARD :
Mois compris entre 1 et 12
Jour (Quantieme) compris entre 1 et 31
*/
{
/*
   Les ElapsedDays sont le tableau de la methode de A.RICHARD 
   Lorsque l'annee est bissextile*, il faut enlever 1 ssi le Mois est Janv ou Fevr.
   *On compte, POUR CETTE METHODE, toute Annee divisible par 4 comme bissextile
   (cf. Formulaire Astro p.46)
*/
const int ElapsedDays[12] = { 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334 } ;
const unsigned long SpecialDay = 4 + 32*10 + 32*16*1582 ;
unsigned long FullDate ;
int NJours ;
double JJ ;

/* Nombre de jours ecoules  */
NJours = ElapsedDays[Mois-1] ;
if ( (4*(Annee/4)==Annee) && (Mois<=2) ) NJours = NJours - 1 ;

/* selection de la methode de calcul */
FullDate = Jour + 32*Mois + 32*16*Annee ;
if ( FullDate < SpecialDay )
	{
	JJ = (double)(1721057 + (int)(365.25*(double)Annee) + NJours + Jour) + 0.5 ;
	}
else
	{
	JJ = (double)(1721059 + (int)(365.2425*(double)Annee) + NJours + Jour ) + 0.5 ;
	}
return JJ ;
}
/*-----------------------------------------------------------------------------------*/
#ifdef _SOLEILS1_
void ASCalculeSoleil ( double JourJulien, TPositionSoleil* Sun )
{
double UU, XX, CC ;
const double Pi_ = 4.0*atan(1.0) ;

/* siecle julien equinoxe 1900 */
(*Sun).T = ASSiecleJulien (JourJulien) ;
/* Excentricite */
(*Sun).e = 0.01675104 - 0.0000418*(*Sun).T*(*Sun).T 
			- 0.000000126*(*Sun).T*(*Sun).T*(*Sun).T ;
/* Parametres */
(*Sun).L = 279.696668 + 36000.76892*(*Sun).T + 0.0003025*(*Sun).T*(*Sun).T ;
(*Sun).L = (*Sun).L - (double)(360*(int)((*Sun).L/360.0)) ;
(*Sun).M = 358.475830 + 35999.04975*(*Sun).T - 0.00015*(*Sun).T*(*Sun).T
			- 0.0000033*(*Sun).T*(*Sun).T*(*Sun).T ;
(*Sun).M = (*Sun).M - (double)(360*(int)((*Sun).M/360.0)) ;
/* en radians */
(*Sun).L = (*Sun).L * Pi_ / 180.0 ;
(*Sun).M = (*Sun).M * Pi_ / 180.0 ;

/* Anomalie Vraie */
UU = (*Sun).M ;
XX = UU - (*Sun).e*sin(UU) ;
while ( abs(1000000.0*((*Sun).M-XX)) > 5.0 )		/* pour une precis. de l'ordre de la seconde d'arc */
	{
	UU = UU + ((*Sun).M-XX)/(1.0-(*Sun).e*cos(UU)) ;
	XX = UU - (*Sun).e*sin(UU) ;
	}
(*Sun).V = 2.0 * atan ( sqrt((1.0+(*Sun).e)/(1.0-(*Sun).e))*tan(UU/2.0) ) ;
/* Longitude Ecliptique */
(*Sun).LambdaMoy = ( (*Sun).L + (*Sun).V - (*Sun).M ) ;

(*Sun).LambdaApp = (*Sun).LambdaMoy ;
/* Obliquite */
(*Sun).EpsilonMoy = 23.4522944 - 0.0130125*(*Sun).T - 0.00000164*(*Sun).T*(*Sun).T ;
(*Sun).EpsilonMoy = (*Sun).EpsilonMoy*Pi_ / 180.0 ;
(*Sun).EpsilonApp = (*Sun).EpsilonMoy ;
/* Correction des planetes */
ASCorrectionPlanete ( Sun ) ;
}
#endif
/*-----------------------------------------------------------------------------------*/
#ifdef _SOLEILS2_
void ASCalculeSoleil ( double JourJulien, TPositionSoleil* Sun )
{
double Omega, CC ;
const double Pi_ = 4.0*atan(1.0) ;

/* siecle julien equinoxe 1900 */
(*Sun).T = ASSiecleJulien (JourJulien) ;
/* Excentricite */
(*Sun).e = 0.01675104 - 0.0000418*(*Sun).T*(*Sun).T 
			- 0.000000126*(*Sun).T*(*Sun).T*(*Sun).T ;
/* Parametres */
(*Sun).L = 279.696668 + 36000.76892*(*Sun).T + 0.0003025*(*Sun).T*(*Sun).T ;
(*Sun).L = (*Sun).L - (double)(360*(int)((*Sun).L/360.0)) ;
(*Sun).M = 358.475830 + 35999.04975*(*Sun).T - 0.00015*(*Sun).T*(*Sun).T
			- 0.0000033*(*Sun).T*(*Sun).T*(*Sun).T ;
(*Sun).M = (*Sun).M - (double)(360*(int)((*Sun).M/360.0)) ;
/* en radians */
(*Sun).L = (*Sun).L * Pi_ / 180.0 ;
(*Sun).M = (*Sun).M * Pi_ / 180.0 ;

/* Equation au centre */
CC = (1.91946 - 0.004789*(*Sun).T - 0.000014*(*Sun).T*(*Sun).T)*sin((*Sun).M) +
	 (0.020094 - 0.00001*(*Sun).T)*sin(2.0*(*Sun).M) + 0.000293*sin(3.0*(*Sun).M) ;
CC = CC*Pi_ / 180.0 ;

(*Sun).LambdaMoy = (*Sun).L + CC ;
(*Sun).V = (*Sun).M + CC ;

Omega = 259.180 - 1934.142*(*Sun).T ;
Omega = Omega - (double)(360*(int)(Omega/360.0)) ;
Omega = Omega*Pi_/180.0 ;
(*Sun).LambdaApp = (*Sun).LambdaMoy - Pi_*(0.005690 + 0.004790*sin(Omega))/180.0 ;

(*Sun).EpsilonMoy = 23.4522944 - 0.0130125*(*Sun).T - 0.00000164*(*Sun).T*(*Sun).T ;
(*Sun).EpsilonApp = (*Sun).EpsilonMoy + 0.002560*cos(Omega) ;
(*Sun).EpsilonMoy = (*Sun).EpsilonMoy*Pi_ / 180.0 ;
(*Sun).EpsilonApp = (*Sun).EpsilonApp*Pi_ / 180.0 ;

/* Correction des planetes */
ASCorrectionPlanete ( Sun ) ;
}
#endif
/*-----------------------------------------------------------------------------------*/
double ASDHEntreeAtmosphere ( double Hauteur, double RAtm )
{
double SinH ;
double xx,aa,bb,cc,delta ;
double teta ;

SinH = sin (Hauteur) ;
/*
 pour y = Sin(hauteur)**2 et x = cos(teta)
 On resoud en x : 
 	a**2.x**2 - 2a(1-y)x + 1 - y -a**2.y
 */
aa = RAtm*RAtm ;
bb = -2.0*RAtm*(1.0-SinH*SinH) ;
cc = 1.0 - SinH*SinH*(RAtm*RAtm+1.0) ;
delta = bb*bb - 4.0*aa*cc ;
if ( delta < 0.0 ) 
	{
	/* delta < 0 = pb de precision
		(le probleme a toujours une solution */
	xx = -bb/(2.0*aa) ;
	}
else
	{
	if ( Hauteur >= 0.0 )
		{
		xx = (-bb + sqrt(delta)) / (2.0*aa) ;
		}
	else
		{
		xx = (-bb - sqrt(delta)) / (2.0*aa) ;
		}
	}

teta = acos ( xx ) ;
return ( teta ) ;
}
/*-----------------------------------------------------------------------------------*/
double ASCorrigeHauteur ( double Hauteur, double RAtmosphere, double Refrac )
{
double DH, HCorrige, Result ;
const double Pi_ = 4.0 * atan (1.0) ;

HCorrige = Hauteur ;
Result = HCorrige + 1 ;
while ( ElAbs(1000000.0*(HCorrige-Result))> 5.0 )
	{
	Result = HCorrige ;
	DH = ASDHEntreeAtmosphere ( HCorrige, RAtmosphere ) ;
	HCorrige = Refrac*cos(DH+Hauteur) ;
	if ( HCorrige < 0.707 )
		{ HCorrige = acos(HCorrige) ; }
	else
		{ HCorrige = Pi_/2.0 - asin(HCorrige) ; }
	HCorrige = HCorrige - DH ;
	}
return HCorrige ;
}
/*-----------------------------------------------------------------------------------*/
void ASCorrectionPlanete ( TPositionSoleil* Sun )
{
/* Soleil - corrections p. 74 : a rajouter a la longitude ecliptique moyenne */
double DLg ;
double AVenus, BVenus, CJupiter, DLune, EE ;
const double Pi_ = 4.0 * atan (1.0) ;

AVenus = 153.23 + 22518.7541*(*Sun).T ;
BVenus = 216.57 + 45037.5082*(*Sun).T ;
CJupiter = 312.69 + 32964.3577*(*Sun).T ;
DLune = 350.74 + 445267.1142*(*Sun).T - 0.00144*(*Sun).T*(*Sun).T ;
EE = 231.19 + 20.20*(*Sun).T ;
DLg = 0.00134*cos(AVenus) + 0.00154*cos(BVenus) + 0.002*cos(CJupiter) +
	  0.00179*sin(DLune) + 0.00178*sin(EE) ;
DLg = Pi_*DLg / 180.0 ;
(*Sun).LambdaMoy = (*Sun).LambdaMoy + DLg ;
(*Sun).LambdaApp = (*Sun).LambdaApp + DLg ;
}
/*-----------------------------------------------------------------------------------*/
double 	ASTempsUniversel ( double JourJulien0h,
						   double HMS, double LongitudeOuestEnHeures, int Fuseau )
{
double			HmsTU ;
double			JJ ;
TPositionSoleil Sun ;

HmsTU =  HMS ;
if ( (Fuseau>-12)&&(Fuseau<12) ) 
	{
	/*Heure a GreenWitch*/ 
	HmsTU = HmsTU - (double)(Fuseau) ; 
	} 
else
	{
	/* correction Horaire */
	HmsTU = HmsTU + LongitudeOuestEnHeures ; /* Heure a Greenwitch */
	if ( Fuseau == ASkHeureLocaleVraie )
		{
		JJ = JourJulien0h + HmsTU/24.0 ;
		ASCalculeSoleil ( JJ, &Sun ) ;
		HmsTU = HmsTU + ASCorrectionHoraire ( &Sun ) ;
		}
	}
return HmsTU ;
}
/*-----------------------------------------------------------------------------------*/
double	ASSiecleJulien ( double JourJulien )
{
return ((JourJulien - 2415020.0)/ 36525.0) ;
}
/*-----------------------------------------------------------------------------------*/



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant Ã  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est rÃ©gi par la licence CeCILL-B soumise au droit franÃ§ais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusÃ©e par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilitÃ© au code source et des droits de copie,
de modification et de redistribution accordÃ©s par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitÃ©e.  Pour les mÃªmes raisons,
seule une responsabilitÃ© restreinte pÃ¨se sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concÃ©dants successifs.

A cet Ã©gard  l'attention de l'utilisateur est attirÃ©e sur les risques
associÃ©s au chargement,  Ã  l'utilisation,  Ã  la modification et/ou au
dÃ©veloppement et Ã  la reproduction du logiciel par l'utilisateur Ã©tant 
donnÃ© sa spÃ©cificitÃ© de logiciel libre, qui peut le rendre complexe Ã  
manipuler et qui le rÃ©serve donc Ã  des dÃ©veloppeurs et des professionnels
avertis possÃ©dant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invitÃ©s Ã  charger  et  tester  l'adÃ©quation  du
logiciel Ã  leurs besoins dans des conditions permettant d'assurer la
sÃ©curitÃ© de leurs systÃ¨mes et ou de leurs donnÃ©es et, plus gÃ©nÃ©ralement, 
Ã  l'utiliser et l'exploiter dans les mÃªmes conditions de sÃ©curitÃ©. 

Le fait que vous puissiez accÃ©der Ã  cet en-tÃªte signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez acceptÃ© les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
