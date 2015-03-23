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
#ifndef _ORILIB_
#define _ORILIB_

//#ifndef __USE_ORILIB_ELISE__
//#else

/*
#ifndef UNIX
#define UNIX
#endif
*/
/*
	Fichier de declaration des routines de photogrammetrie.
	======================================================

Fonctionnalites generales des routines :

1) Allouer une/des structure(s) d'orientation (voir ALLOC & FREE)
2) Lire un/des fichier(s) d'orientation (voir LECTURE & ECRITURE)
3) Travailler !

	Les fonctions disponibles concernent :
	- les calculs de coordonnees (carto <-> terrain <-> images)
	- les calculs d'epipolaire
	- les calculs d'orientation d'images extraites des images initiales

4) Ecrire eventuellement les nouvelles orientation
5) desallouer et aller se coucher


Contenu de cette documentation :

Ce fichier est la seule documentation disponible. Il contient les prototypes
des fonctions disponibles en C (et peut etre utilise comme include pour les
declarer), et des commentaires sur leurs fonctionnalites.

Table des matieres :

	- 1.) UTILISATION pratique et CODES DE RETOUR des fonctions
	- 2.) GLOSSAIRE
	- 3.) ALLOC & FREE
		or$alloc_orientation
		or$free_orientation
	- 4.) LECTURE & ECRITURE
		or$lit_orientation
		or$lit_orientation_texte
		or$ecrit_orientation
		or$ecrit_orientation_texte
		or$lit_image_orientation	(MATIS uniquement)
		or$ecrit_image_orientation	(MATIS uniquement)
	- 5.) COPIE & MODIF en MEMOIRE
		or$copy_orientation
		or$fenetrage_orientation 
	- 6.) PHOTOGRAMMETRIE (Coordonnees)
		or$carte_to_terrain
		or$terrain_to_carte
		or$terrain_to_photo
		or$photos_to_terrain
		or$photo_et_z_to_terrain
	- 7.) PHOTOGRAMMETRIE (Infos annexes) 
		or$point_fuite_soleil
		or$point_fuite_verticale
	- 8.) EPIPOLAIRES 
		or$orient_epipolaires
		or$projette_image
		or$projette_epipolaire
		or$repere_3D_image
	- 9.) CORRELATION
		or$epipolaire
		or$minmax_paral
	- 10.) INFOS DIVERSES
		or$origine_terrain
		or$emprise_carte
		or$emprise_terrain
		or$sommet_de_pdv_carto
		or$sommet_de_pdv_terrain
		or$vecteur_solaire


*************************************************************************
1.) UTILISATION pratique et CODES DE RETOUR des fonctions	*********

Compilation : 
  Declaration des fonction = d$jamet:[jamet.these.phot]orilib.h
Link :
  Bibliotheque a jour : D$JAMET:[jamet.axp.lib]utilib.olb


TOUTES les fonctions renvoient un code retour entier, egal a 1 si tout va
bien, et a 0 sinon, SAUF la fonction photos_to_terrain qui retourne la
distance d'intersection des rayons perspectifs.

TOUTES les coordonnees Carte et Terrain sont en metres.
TOUTES les coordonnees Image sont en PIXELS.

TOUTES les fonctions ont des arguments passes par adresse (qu'ils soient
en input ou en output), pour faciliter la compatibilite avec FORTRAN ( c'est
moins fatiguant pour les C-istes de taper &machin, que pour les FORTRANNEUX
d'ecrire %val(machin) ).

*************************************************************************
2.) GLOSSAIRE:							*********

Ce court glossaire presente les noms de variable les plus utilises de facon a 
limiter l'explication de chaque fonction.

> phot, phot-in, phot-input, phot-out, ...
	Nom choisi pour le pointeur (entier 4) sur la structure contenant
	en memoire les infos d'orientation. La dite structure est
	"transparente" pour l'utilisateur, qui manipule toujours ce
	pointeur, passe a toutes les fonctions de la bibliotheque.
> xcarte, ycarte, zcarte :
	Coordonnees Cartographiques (=LAMBERT dans notre cas)
> xterre, yterre, zterre :
	Coordonnees Terrain = dans le repere Euclidien associe a la mise
	en place.
> col, colonne et leurs variantes :
	Offset Colonne dans l'image en PIXELS
	Lorsqu'une valeur col est retournee par une fonction, la valeur de
	colonne dans l'image utilisee est col ssi votre origine de tableau
	est 0; ce sera 1+col si votre origine est 1 .
> lig, ligne et leurs variantes :
	Offset Colonne dans l'image en PIXELS
	Meme remarque que col sur l'origine des lignes.


Pour les anti-C :
long* 		= integer*4
int*  		= integer*4
double* 	= double
char*		= %VAL(%REF(character*x))	(C'est une facon de dire)




*************************************************************************
3.) ALLOC & FREE						*********/


/*===============================*/
 int oralloc_orientation ( 	void**		/*phot*/ ) ;
/*===============================*/
/*
Alloue la structure d'orientation. Operation OBLIGATOIRE avant toute
utilisation d'autres routines utilisant "phot" (sinon ca plante!)
*/


/*===============================*/
 int orfree_orientation ( 	void** 		/*phot*/ ) ;
/*===============================*/
/*
Libere la structure allouee
*/


/************************************************************************
4.) LECTURE & ECRITURE						*********

Deux formats de stokage possibles :

Format binaire (moins encombrant et conseille) - les routines images
		propres au MATIS ne supportent que ce format

Format texte ( a but essentiellement de format d'echange : routines se 
	       terminant par _texte)					*/






/*
Lit un fichier d'orientation (.ORI).
Fichier specifie le nom complet avec extension.
En fortran, passer %VAL(%REF(Fichier)), Fichier etant declare comme chaine de
caracteres.
*/

			/*------------------------*/
#ifndef UNIX
/*===============================*/
 int orlit_image_orientation ( 	void** 		/*ima*/, 
/*===============================*/	char* 		/*chantier*/, 
					void** 		/*phot*/ ) ;
/*
"ima" est un pointeur sur une structure image (de la librairie IMG).
"chantier" est une chaine de caractere (voire Rqs precedentes sur les chaines)

La fonction lit le fichier d'orientation correspondant a l'image de la facon 
suivantes : si NomIma.IMG est le nom de l'image, si la chaine de caractere
"chantier" contient le mot "chantier", la routine va chercher le fichier
			NomIma-chantier.ORI
Par ailleurs, elle gere le fenetrage de l'orientation de l'image si "ima" a 
ete recupere par IMG_CLI_GETIMAGE.
*/


/*===============================*/
 int orecrit_image_orientation ( 	void** 		/*photo*/, 
/*===============================*/	void** 		/*ima*/, 
					char* 		/*chantier*/ ) ;
/*
Memes arguments et memes conventions que "lit_image_orientation"
*/
#endif
/************************************************************************
5.) COPIE & MODIF en MEMOIRE					*********/

/*===============================*/
 int orcopy_orientation ( 		void** 		/*phot-in*/, 
/*===============================*/	void** 		/*phot-out*/ );
/* 
simple recopie du contenu de la structure pointee par "phot-in" dans
celle pointee par "phot-out"
*/

/*===============================*/
 int orfenetrage_orientation ( 	void** 		/*phot-in*/, 
/*===============================*/	double* 	/*left*/, 
					double* 	/*top*/,
					double* 	/*right*/, 
					double* 	/*bottom*/,
					double* 	/*hstep*/, 
					double* 	/*vstep*/,
					void**		/*phot-out*/ ) ;
/*
"left", "top", "right", "bottom" definissent l'emprise d'une sous-image dans
l'image correspondant a "phot-in". "hstep" et "vstep" definissent un pas
d'echantillonnage (le tout en PIXELS).
"Phot-out" sera remplie par l'orientation de la sous-image definie par ces
parametres.

ATTENTION: meme convention de coordonnees que plus haut; "left", "top", "right",
et "bottom" sont des offsets par rapport a l'origine d'image que vous
choississez ( 0,0 ou 1,1 ou autre ).

Nota: on peut utiliser cette fonction avec "phot-out" = "phot-in". 
      "phot-in" est alors bien entendu ecrase.
*/

/************************************************************************
6.) PHOTOGRAMMETRIE (Coordonnees)				*********/

/*===============================*/
 int orcarte_to_terrain ( 		void** 		/*phot*/,
/*===============================*/	double* 	/*xcarte*/, 
					double* 	/*ycarte*/, 
					double* 	/*zcarte*/,
			    		double* 	/*xterre*/, 
					double* 	/*yterre*/, 
					double* 	/*zterre*/ ) ;

/*===============================*/
 int orterrain_to_carte ( 		void** 		/*phot*/,
/*===============================*/	double* 	/*xterre*/, 
					double* 	/*yterre*/, 
					double* 	/*zterre*/,
			    		double* 	/*xcarte*/, 
					double* 	/*ycarte*/, 
					double* 	/*zcarte*/ ) ;

/*===============================*/
 double orphotos_to_terrain ( 		void** 		/*phot-1*/, 
/*===============================*/	double* 	/*colonne-1*/, 
					double* 	/*ligne-1*/,
				     	void** 		/*phot-2*/, 
					double*		/*colonne-2*/, 
					double* 	/*ligne-2*/,
				     	double* 	/*xterre*/, 
					double* 	/*yterre*/, 
					double* 	/*zterre*/ ) ;
/*
Retourne la distance d'intersection entre les deux rayons perspectifs (en
metres)
NB : retourne -1.0 si les rayons perpectifs ne se coupent pas
*/

/*===============================*/
 int orphoto_et_zCarte_to_terrain (	void**	/*phot*/, 
/*===============================*/		double*	/*colonne*/,
					double* 	/*ligne*/, 
					double* 	/*zCarte*/,
					double* 	/*xterre*/,
					double* 	/*yterre*/,
					double* 	/*zterre*/ ) ;
/* 
meme fonction que la precedente, mais le z en entree est un z carte.
La fonction renvoie x,y,z terrain correspondant.

Nota : Cette fonction est approchee (la terre est assimilee a un plan
sur l'emprise de la photo). Ne s'en servir donc que pour avoir des
ordres de grandeur.
*/

/************************************************************************
7.) PHOTOGRAMMETRIE (Infos annexes)				*********/

/*===============================*/
 int orpoint_fuite_soleil ( 		void** 		/*phot*/, 
/*===============================*/	double* 	/*fuite[2]*/ ) ;
/*
"fuite" est un tableau de deux reels doubles donnant l'offset colonne-ligne du
point de fuite dans l'image (en PIXELS)
*/

/*===============================*/
 int orpoint_fuite_verticale (		void** 		/*phot*/, 
/*===============================*/	double* 	/*fuite[2]*/ ) ;
/*
"fuite" est un tableau de deux reels doubles donnant l'offset colonne-ligne du
point de fuite dans l'image (en PIXELS)
*/


/************************************************************************
8.) EPIPOLAIRES							*********/

/*
Quelques mots sur la fabrication des epipolaires :

   a) Methode standart (pour cette bibliotheque).

   a.1 : Allocation des orientations des epipolaires
   a.2 : Appel de la fonction ororient_epipolaire
   a.3 : Calcul du reechantillonnage :
	 - soit on fait l'image d'un seul coup : appel direct a 
	   orprojette_image
	 - soit on fait l'image par pave : 
	   . Allouer une orientation pour un pave ;
	   . Utiliser orfenetrage_orientation pour definir l'emprise
	     du pave (avec l'orientation epipolaire en entree et 
	     l'orientation pave en sortie)
	   . orprojette_image pour chaque pave

   b) Methode economique (moins d'appels de fonction).

   b.1 : Allocation des orientations des epipolaires
   b.2 : Appel de la fonction ororient_epipolaire
   b.3 : Calcul du reechantillonnage :
	- appel de orrepere_3D_image pour recuperer le vecteur coin[3],
	  coordonnees terrain de l'origine de l'image (sur le plan de 
	  projection), et les vecteurs 3D uu[3] et vv[3] de deplacement d'un 
	  pas en colonne et en ligne ;
	- avec une convention d'origine a (0,0), on definit les coordonnees
	  Terrain de chaque pixel (i,j) de l'image epipolaire par
		xterre = coin[0] + i * uu[0] + j*vv[0] ;
		yterre = coin[1] + i * uu[1] + j*vv[1] ;
		zterre = coin[2] + i * uu[2] + j*vv[2] ;
	- on peut alors utiliser orterrain_to_photo pour retrouver la position
	  du point xterre,yterre,zterre dans l'image brute.
*/


/*===============================*/
 int ororient_epipolaires(		void** 		/*phot-1*/, 
/*===============================*/	void** 		/*phot-2*/,
					double* 	/*right*/, 
					double* 	/*top*/, 
					double* 	/*left*/, 
					double* 	/*bottom*/, 
					double* 	/*zcarte-min*/, 
					double* 	/*zcarte-max*/, 
					void** 		/*phot-epipo-1*/, 
					void** 		/*phot-epipo-2*/,
					int* 		/*ncol-image-epipo*/, 
					int* 		/*nlig-image-epipo*/ ) ;
/*
Calcule les orientations des images epipolaires definies sur le couple
"phot-1", "phot-2" par l'emprise "left", "top", "right", "bottom" sur "phot-1".
L'emprise est exprimee en PIXELS avec la meme convention d'origine que dans les
autres focntions (0,0 : "left", "top", "right" et "bottom" sont des offsets par
rapport a l'origine).
L'emprise des epipolaires est egalement conditionnee par le domaine de
variation de Z ("zcarte-min", zcarte-max", en coordonnees carte).

La fonction retourne egalement la dimension des images epipolaires correpondant
a cette emprise ( "ncol-image-epipo", nlig-image-epipo", en PIXELS).

NOTA : si vous souhaitez donner des dimensions differentes a l'epipolaire
gauche, de facon a ce que ses homologues soient strictement inclus dans
l'epipolaire droite, vous pouvez utiliser la fonction "minmax_paral", en
deduire la sous-image de l'epipolaire gauche, et utiliser
"fenetrage_orientation" pour calculer l'orientation de la sous-image epipolaire
gauche.
*/



/*===============================*/
 int orprojette_image ( 		void** 		/*phot*/, 
/*===============================*/	unsigned char * /*data-in*/,
					int* 		/*ncol-in*/,
					int* 		/*nlign-in*/,
						void** 		/*phot-epipo*/,
		   			unsigned char(*)( unsigned char *, int*, int*, double*, double* )/*interpolation()*/,
			   		unsigned char * 	/*data-out*/, 
					int* 		/*ns-out*/, 
					int* 		/*nl-out*/ ) ;
/*
Calcule l'image epipolaire (valeurs radiometrique des pixels) correspondant
a "phot-epipo" par projection des radiometries de "phot", et en remplit le
buffer "data-out".
Les radiometrie de l'image brute sont stockees dans "data-in", de dimensions
"ncol-in"x"nlig-in" pixels.
Le buffer "data-out" est de dimension "ncol-out"x"nlig-out". Nominalement, ces
valeurs sont celles qui sortent de "orient_peipolaires".

"interpolation" est un pointeur de finction. La fonction correspondante  
doit avoir le prototype suivant:
  unsigned char interpolation ( unsigned char *data-in, 
				int *ncol-in, int *nlig-in, 
			  	double *xx, double *yy ) ;
  ( la valeur retournee est la valeur interpolee a la position (xx,yy), en
   PIXELS, avec la meme convention d'origine (0,0) que precedemment )
*/

/*===============================*/
 int orprojette_epipolaire ( 		void** 		/*phot*/, 
/*===============================*/	unsigned char* /*data-in*/, 
					int* 		/*ncol-in*/, 
					int* 		/*nlign-in*/, 
			   		void** 		/*phot-epipo*/, 
		   			unsigned char(*)( unsigned char*, int*, int*, double*, double* )/*interpolation()*/,
			   		unsigned char* 	/*data-out*/, 
					int* 		/*ns-out*/, 
					int* 		/*nl-out*/ ) ;
/* 
Meme chose que projette image, un peu plus rapide, MAIS valide uniquement
pour les epipolaires - ou les images SANS DISTORTIONS
*/



/*===============================*/
 int orrepere_3D_image ( 		void** 		/*phot*/, 
/*===============================*/	double* 	/*coin[3]*/, 
				    	double* 	/*uu[3]*/, 
					double* 	/*vv[3]*/ ) ;
/*
Donne la position terrain coin[3] de l'origine d'une image (i.e. la position 
sur le plan de projection situe en 3D) et les vecteurs de deplacement en
colonne et en ligne correspondant respectivement a un pas de 1 PIXELS en
colonne et en ligne
*/

/*===============================*/
int oremprise_photo_epipolaire ( 		void**		/*orientation epi*/,
/*===============================*/		void**		/* orientation photo*/,
								 		double*		/* col min*/, 
										double*		/* lig min*/,
								 		double*		/* col max*/, 
										double*		/* lig max*/ ) ;
/*
Donne l'emprise colonne ligne d'une image epipolaire definie par son
orientation dans l'image brute.
Renvoie le status 0 si les deux orientations fournies ne correspondent
pas a deux images ayant strictement le meme sommet.
*/

/*===============================*/
int oremprise_photo_carte ( 		void**		/* photo */, 
/*===============================*/	double*		/* xmin */,
							 		double*		/* ymin */,
							 		double*		/* xmax */, 
									double* 	/* ymax */,
							 		double*		/* zmax */, 
									double*		/* zmax */,
							 		double*		/* col min */, 
									double*		/* lig min */,
							 		double*		/* col max */, 
									double*		/* lig max*/ ) ;
/*
Donne l'emprise colonne ligne d'une zone carte definie par
	xmin, ymin, xmax, ymax, zmin, zmax (en coord carte, bien sur)
*/

/************************************************************************
9.) CORRELATION							*********/

/*===============================*/
 int orepipolaire ( 			void** 		/*phot-1*/, 
/*===============================*/	void** 		/*phot-2*/ ) ;
/*
Renvoie 1 si le couple est epipolaire, 0 sinon.
(La definition choisie pour epipolaire est : les lignes de meme indice sont 
 homologues et la resolution en colonne est la meme)
*/


/*===============================*/
 int orminmax_paral ( 			void** 		/*phot-epipo-1*/, 
/*===============================*/	void** 		/*phot-epipo-2*/,
			       		double* 	/*zcarte-min*/, 
					double* 	/*zcarte-max*/,
			       		double* 	/*paralaxe-min*/, 
					double* 	/*paralaxe-max*/ ) ;
/*
Donne les parallaxes min et max en colonne, correspondant a l'intervalle 
[zcarte-min, zcarte-max] (en coordonnees terrain)
NOTA : dans la precedente version, cette fonction prenait des Z terrain en
argument ; j'ai juge plus commode de prendre des Z carte (les intervalles
de variations sont toujours plus petits.
*/

/************************************************************************
10.) INFOS DIVERSES						*********/

/*===============================*/
 int ororigine_terrain ( 		void** 		/*phot*/, 
/*===============================*/	double* 	/*origine_carte[2]*/ ) ;
/*
origine_carte[2] est l'origine du repere terrain en coordonnes carte
(point de tangeance entre le plan (OXY) du repere terrain et la Terre, situe
 a l'altitude zero)
*/

/*===============================*/
 int orsommet_de_pdv_carto ( 		void** 		/*phot*/, 
/*===============================*/	double* 	/*xcarte*/, 
					double* 	/*ycarte*/, 
					double* 	/*zcarte*/ ) ;
/*
Renvoie les coordonnees du sommet de prise de vue en coordonnees Carte
*/

/*===============================*/
 int orsommet_de_pdv_terrain (		void** 		/*phot*/, 
/*===============================*/	double* 	/*xterre*/, 
					double* 	/*yterre*/, 
					double* 	/*zterre*/ ) ;


void orDirI(void**,double*,double*,double*);
void orDirJ(void**,double*,double*,double*);
void orDirK(void**,double*,double*,double*);

void orSetSommet ( void* *phot, double *x, double *y,double * z);
void orSetDirI ( void* *phot, double *x, double *y,double * z);
void orSetDirJ ( void* *phot, double *x, double *y,double * z);
void orSetDirK ( void* *phot, double *x, double *y,double * z);
void orSetAltiSol ( void* *phot, double *AltiSol);


/*
Renvoie les coordonnees du sommet de prise de vue en coordonnees Terrain
*/


/*===============================*/
 int oremprise_carte ( 		void** 		/*phot*/, 
/*===============================*/	double*		/*zmin*/,
					double*		/*zmax*/,
					int*		/*marge*/,
					double* 	/*xcarte-min*/, 
					double* 	/*ycarte-min*/,
				     	double* 	/*xcarte-max*/, 
					double* 	/*ycarte-max*/ ) ;
/*
Donne une emprise de l'image sur la carte. Cette emprise est calculee
pour une altitude variant entre zmin et zmax.
inputs:
-------
zmin, zmax : bornes de l'altitude sur la zone en coord carte
marge	   : marge en pixels autour de l'image correspondante
*/

/*===============================*/
 int oremprise_terrain ( 		void** 		/*phot*/, 
/*===============================*/	double*		/*zmin*/,
					double*		/*zmax*/,
					int*		/*marge*/,
					double* 	/*xterrain-min*/, 
					double* 	/*yterrain-min*/,
				     	double* 	/*xterrain-max*/, 
					double* 	/*yterrain-max*/ ) ;
/*
idem que emprise_carte, mais avec des output en coordonnees terrain
*/


/*===============================*/
 int orvecteur_solaire ( 		void** 		/*phot*/, 
/*===============================*/	double* 	/*soleil-terre[3]*/ ) ;
/*
Renvoie le vecteur unitaire definissant la direction Soleil->Terre dans le
repere Terrain.
*/

/*===============================*/
double orresolution_sol ( void** /*photo*/ ) ;
/*===============================*/
/*
Renvoie l'ordre de grandeur de la resolution au sol
*/

/*===============================*/
double oraltitude_sol ( void** /*photo*/ ) ;
/*===============================*/
/*
Renvoie l'altitude du sol stockee dans le fichier d'orientation
*/


/*===============================*/
 int orlit_orientation (const 		char* 		/*fichier*/, 
/*===============================*/	void** 		/*phot*/ ) ;




/*===============================*/
 int orphoto_et_z_to_terrain (		void** 		/*phot*/, 
/*===============================*/	const double* 	/*colonne*/, 
					const double* 	/*ligne*/, 
					const double* 	/*zterre*/,
					double* 	/*xterre*/,
					double* 	/*yterre*/ ) ;


 int orphoto_et_prof_to_terrain (      void** 		/*phot*/, 
	                                const double* 	/*colonne*/, 
					const double* 	/*ligne*/, 
					const double* 	/*prof*/,
					double* 	/*xterre*/,
					double* 	/*zterre*/ ,
					double* 	/*yterre*/ ) ;


int orphoto1_et_prof2_to_terrain ( void* *phot1,
                                 const double *col1,
                                 const double *lig1,
                                 void* *phot2,
                                 const double *prof2,
                                 double *xterre,
                                 double *yterre ,
                                 double *zterre
                               );

int  or_prof(  void* *phot,
                 const double *xterre,
                 const double *yterre ,
                 const double *zterre,
                 double       *prof
            );



/* 
ici zterre est en input, et la fonction ne retourne que (xterre, yterre)
*/

/*===============================*/
 int orecrit_orientation_texte ( 	void** 		/*photo*/, 
/*===============================*/const	char* 		/*fichier*/ ) ;
/*
Ecrit un fichier d'orientation (.ORI).
Memes arguments que "Lit"
Meme chose que le precedent, mais dans un format texte.
*/

/*===============================*/
 int orecrit_orientation ( 		void** 		/*photo*/, 
/*===============================*/const	char* 		/*fichier*/ ) ;
/*
Ecrit un fichier d'orientation (.ORI).
Memes arguments que "Lit"
*/

/*===============================*/
 int orlit_orientation_texte ( 	const char* 		/*fichier*/, 
/*===============================*/	void** 		/*phot*/,
                                    bool QuickGrid) ;
/*
Lit un fichier d'orientation (.ORI).
Fichier specifie le nom complet avec extension.
En fortran, passer %VAL(%REF(Fichier)), Fichier etant declare comme chaine de
caracteres.
Meme chose que le precedent, mais dans un format texte.
*/

/*===============================*/
 int orterrain_to_photo ( 		void** 		/*phot*/,
				const	double*   /*xterre*/, 
			        const	double* 	/*yterre*/, 
				const	double* 	/*zterre*/,
			   		double* 	/*colonne*/, 
					double* 	/*ligne*/ ) ;



//#endif

#endif



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
