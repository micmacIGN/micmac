/*
		Methode des gradients conjugues

*/
#define VECT_IN_LINE
#include "vect.h"

class Grad_Conj {
public:
	double echelle;		//	dimension en dessous de laquelle on s'attend a un comportement quadratique
	int Dim;			//	Dimension du Pb
	Vect 		Sol		//	Solution courante
				, GradC	//	Gradient Conjugue
				, Grad;	//	Gradient courant
	double	modifmax,	//	Modification maximale apportee aux parametres
			potsol;		//	Potentiel de la solution courante
	int		iterations;	//	Nombre d'iterations effectuees
	Grad_Conj( int d, double e ) : echelle( e ), Dim(d), Sol(d), GradC(d), Grad(d) {}
	virtual ~Grad_Conj(){}
//		Fonction fournie par la librairie : minimisation
	void Minimize();
	void Descente();
	void CalcDer(double &G, double &C, double &gm );
//	Fonctions a fournir par le client :
//	Initialisation de Sol
	virtual void Init() = 0;
//	Calcul du gradient en une solution possible, retourne la valeur du potentiel en Sol
	virtual double Gradient() = 0;

//	Calcule le potentiel en Sol
	virtual double Potent( const Vect &t) = 0;

//	Communication du resultat courant au prog principal.
//	Si celui-ci decide d'arreter la fonction retourne 1, sinon 0.
	virtual int Controle() = 0;
};


