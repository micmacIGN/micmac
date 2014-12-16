/*
		Methode des gradients conjugues

*/

#include <stdio.h>
#include <math.h>
#include "gradconj.h"

/*		Algorithme de descente du gradient, cas non quadratique 
		ou courbure negative.
*/

void Grad_Conj::Descente() {
	double u0, u, lu0, lpt, pt, mgr, ech;
	
	lpt = potsol;
	printf( "Descente\n" );
	u0 = 0;
	mgr = GradC.abs().max();
	ech = echelle/mgr;
	u = ech;
	while( 1 ) {
		pt = Potent( Sol - u * GradC );
		fprintf( stderr, "desc: u=%g pt=%g\n", u, pt );
		if( pt > lpt ) {
			if( fabs( u0-u ) < ech /100 ) break;
			u = (u0+u)/2;
		} else {
			lu0 = u0;
			lpt = pt;
			u0 = u;
			u += (u-lu0)*1.5;
		}
	}
	Sol -= u0 * GradC;
	potsol = lpt;
}

/*		Calcul de la derivee et de la courbure le long d'une direction	*/

void Grad_Conj::CalcDer( double &G, double &C, double &gm ) {
	double e, v2, v4;

	gm = GradC.abs().max();
	e = echelle / gm;
	v2 = Potent( Sol - e*GradC );
	v4 = Potent( Sol + e*GradC );
	C = (v4+v2)/2 - potsol;
	G = (v4-v2)/2;
	C /= e*e;
	G /= e;
}

/*		Minimisation de la fonction par la methode des gradients conjugues
*/
void Grad_Conj::Minimize() {
	double gdg, cbg, gmx, k, modagr, modngr;
	
	potsol = Gradient();
	GradC = -Grad;
	modngr = GradC/GradC;
	iterations = 0;
	while( 1 ) {
		iterations++;
/*			Calcul de la nouvelle solution approchee	*/
		CalcDer( gdg, cbg, gmx );
		if( cbg <= 0 ) {
			fprintf( stderr, "Trouve courbure negative.\n" );
			GradC = -Grad;
			Descente();
		} else Sol -= gdg/2/cbg*GradC;
		modifmax = -gdg*gmx/2/cbg;
		echelle = modifmax/4;

		if( Controle() ) break;

/*	Calcul du potentiel pour cette solution		*/
/*			Calcul du nouveau gradient		*/
		potsol = Gradient();
		modagr = modngr;
		modngr = Grad/Grad;

/*		Calcul de la nouvelle direction a explorer		*/
		k = modngr/modagr;
		GradC *= k;
		GradC -= Grad;
	}
}

