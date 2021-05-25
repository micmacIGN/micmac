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



#ifndef _ORILIB_DEV_
#define _ORILIB_DEV_
	/*=================================================*/

#define _NS_GRILLE 36




typedef struct
    {
	/* grille de distorsion : on prevoit 1 point tout les cm au maximum */
	int ns, nl, pas ;
	double dx[_NS_GRILLE*_NS_GRILLE] ;
	double dy[_NS_GRILLE*_NS_GRILLE] ;
    } or_grille ;


typedef struct or_orientation
    {
	/* mode de calcul de la mise en place */
	int distor ;    /*   !=0 ? on prend en compte des distortions
				   refraction et/ou distortion chambre
				   et/ou distortion scanner  */
	/* pour info */
	int refraction ;
	char chambre[8] ;
	/* donnees temporelles */
	int	jour, mois, annee, heure, minute, seconde ;
	/* pour la correction atmospherique : altitude moyenne du sol */
	double	altisol ;
	double	mProf ; // Remplacera altisol en terrestre, <0 quand non def
	/* coordonnees Lambert de l'origine du repere Euclidien local */
	double origine[2] ;
	int    lambert ;	/* numero de la zone lambert */
	/* coordonnees en metres dans le repere euclidien local */
	double sommet[3] ;	/* sommet de prise de vue */
	double focale ;		/* en metres */
	double vi[3] ;		/* vecteurs definissant le plan image */
	double vj[3] ;
	double vk[3] ;
	double pix[2] ;		/* pas d'echantillonnage */
	/* taille de l'image associee */
	int ins, inl ;
	/* coordonnees pixel du point principal */
	double ipp[2] ;		/* coord pixels point principal 
		(offset en pixels par rapport a l'origine de l'image) */
	or_grille gt2p ;
	or_grille gp2t ;

	cOrientationConique * mOC;
	// cDbleGrid                                      * mCorrDistM2C; // 
	//  MODIF MPD, pour des camera commme les fishe eye, une grille standard ne
	//  peut pas faire l'affaire. Pour l'instant on remplace simplement par
	//  la classe mere cDbleGrid => ElDistortion22_Gen et on verifie que ca ne
	//  pose pas de pb
	//
	ElDistortion22_Gen                                      * mCorrDistM2C; // 
        std::string mName;


        bool   mDontUseDist;
        void InitNewParam();
	int CorrigeDist_T2P(double *col, double *lig);
	int CorrigeDist_P2T(double *col, double *lig);
    } or_orientation ;


/*
 * STRUCTURE DE STOCKAGE :
 * copie conforme de la precedente, sans les grilles
 *
 * Cette structure est exclusivement utilisee par lit_fic_orientation 
 * et ecrit_fic_orientation
 */

typedef struct
    {
	/* mode de calcul de la mise en place */
	int distor ;    /*   !=0 ? on prend en compte des distortions
				   refraction et/ou distortion chambre
				   et/ou distortion scanner  */
	/* pour info */
	int refraction ;
	char chambre[8] ;
	/* donnees temporelles */
	int	jour, mois, annee, heure, minute, seconde ;
	/* pour la correction atmospherique : altitude moyenne du sol */
	double	altisol ;
	/* coordonnees Lambert de l'origine du repere Euclidien local */
	double origine[2] ;
	int    lambert ;	/* numero de la zone lambert */
	/* coordonnees en metres dans le repere euclidien local */
	double sommet[3] ;	/* sommet de prise de vue */
	double focale ;		/* en metres */
	double vi[3] ;		/* vecteurs definissant le plan image */
	double vj[3] ;
	double vk[3] ;
	double pix[2] ;		/* pas d'echantillonnage */
	/* taille de l'image associee */
	int ins, inl ;
	/* coordonnees pixel du point principal */
	double ipp[2] ;		/* coord pixels point principal 
		(offset en pixels par rapport a l'origine de l'image) */
    } or_file_orientation ;

	/*=================================================*/

#include "orilib.h"

	/*=================================================*/

int orlit_fictexte_orientation (const char*	/*fic*/, 
				or_orientation*	/*ori*/,
				bool QuickGrid) ;
int orecrit_fictexte_orientation (const char*	/*fic*/, 
				or_orientation*	/*ori*/ ) ;

int orlit_fic_orientation (const 	char* 		/*fic*/, 
				or_orientation*	/*ori*/ ) ;

int NEW_orlit_fic_orientation (const 	char* 		/*fic*/, 
				or_orientation*	/*ori*/ ) ;

int orecrit_fic_orientation (const	char* 		/*fic*/, 
				or_orientation*	/*ori*/ ) ;

void orinters_SM_photo ( 	void** 		/*phot*/, 
				double* 	/*SM[3]*/, 
			   	double* 	/*colonne*/, 
				double* 	/*ligne*/ ) ;

void orDirLoc_to_photo ( 	void** 		/*phot*/, 
				double* 	/*SM[3]*/, 
			   	double* 	/*colonne*/, 
				double* 	/*ligne*/ ) ;

void orSM ( 			void** 		/*phot*/, 
				double* 	/*colonne*/, 
				double* 	/*ligne*/, 
				double* 	/*SM[3]*/ ) ;

void orPhoto_to_DirLoc (        void** 		/*phot*/, 
				double* 	/*colonne*/, 
				double* 	/*ligne*/, 
				double* 	/*SM[3]*/ ) ;



void orrepere_epipolaire ( 	double* 	/*S1[3]*/, 
				double* 	/*S2[3]*/, 
			     	double* 	/*vi[3]*/, 
				double* 	/*vj[3]*/, 
				double* 	/*vk[3]*/ ) ;

void oremprise_epipo_gauche (	void**		/*phot*/, 
				double* 	/*ph_c0*/, 
				double* 	/*ph_l0*/,
			      	double*		/*ph_c1*/, 
				double* 	/*ph_l1*/, 
			      	double* 	/*zmin*/, 
				double* 	/*zmax*/,
			      	void** 		/*epiphot*/ ) ;

void oremprise_epipo_droite (	void** 		/*epiphot1*/, 
				void** 		/*epiphot2*/ ) ;

void orremplit_orient_epipolaire ( void** 	/*phot*/, 
				double* 	/*MM[3]*/, 
				double* 	/*vi[3]*/, 
				double* 	/*vj[3]*/, 
				double* 	/*vk[3]*/,
				double* 	/*pas*/,
				void**		/*epiphot*/ ) ;

double orbest_resol ( 	void** 		/*phot1*/, 
				void** 		/*phot2*/, 
				double* 	/*zmin*/ ) ;

void orfenetrage_grille ( 	or_grille* 	/*gr1*/ ,
			   	double* 	/*left*/, 
				double* 	/*top*/, 
			   	double* 	/*hstep*/, 
				double* 	/*vstep*/,
			   	int* 		/*ins*/, 
				int* 		/*inl*/,
			   	or_grille* 	/*gr2*/ ) ;

int orcorrige_distortion( 	double* 	/*col*/, 
				double* 	/*lig*/, 
				or_grille* 	/*gr*/ ) ;

void orinverse_grille ( 	or_grille* 	/*gin*/, 
				or_grille* 	/*gout*/ ) ;

void make_std_inversion_grille(or_orientation *);

void orAddFilmDistortions ( 
				or_grille* /*grille T>P*/,
				or_grille* /*grille P>T*/, 
				double[16] /*IdealMarks*/, 
				double[16] /*RealMarks*/,
				int* /*NMarks*/ ) ;

void orgrille_refraction ( 	or_orientation* /*ori*/, 
				or_grille* 	/*gr*/ ) ;

void ordim_grille ( 		int* 		/*ins*/, 
				int* 		/*inl*/, 
				int* 		/*gns*/, 
				int* 		/*gnl*/, 
				int* 		/*gpas*/ ) ;

void orinit_grille_0 ( 	void** 		/*phot*/ ) ;

int orinit_distortions ( char* /*chambre*/, 
				int* /*refrac*/, 
				double[8] /*IdealMarks*/,
				double[16] /*RealMarks*/,
				int* /*NMarks*/,
				void** /*phot*/ ) ;


	/*=================================================*/
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
