#include <assert.h>
#include "Observation.h"

// ---------------------------------------------------------------
// Méthode permettant de récupérer un type d'observable
// ---------------------------------------------------------------
double Observation::getChannel(std::string obs_type){
	
	if (obs_type == "C1")  {return this->c1;}
	if (obs_type == "L1")  {return this->l1;}
	if (obs_type == "L2")  {return this->l2;}
	if (obs_type == "P1")  {return this->p1;}
	if (obs_type == "P2")  {return this->p2;}
	if (obs_type == "D1")  {return this->d1;}
	if (obs_type == "D2")  {return this->d2;}
     	
	std::cout << "ERROR: type " << obs_type << " is not a valid type of observable" << std::endl;
	assert(false);
	return -1;
	
}

// ---------------------------------------------------------------
// Méthode permettant de récupérer l'indicater de LOSS OF LOCK
//  un type d'observable
// ---------------------------------------------------------------
int Observation::getChannel_LLI(std::string obs_type){
	
	if (obs_type == "C1")  {return this->c1_lli;}
	if (obs_type == "L1")  {return this->l1_lli;}
	if (obs_type == "L2")  {return this->l2_lli;}
	if (obs_type == "P1")  {return this->p1_lli;}
	if (obs_type == "P2")  {return this->p2_lli;}
	if (obs_type == "D1")  {return this->d1_lli;}
	if (obs_type == "D2")  {return this->d2_lli;}
     	
	std::cout << "ERROR: type " << obs_type << " is not a valid type of observable" << std::endl;
	assert(false);
	return -1;
	
}

// ---------------------------------------------------------------
// Méthode permettant de récupérer l'indicater de SIGNAL STRENGTH
//  un type d'observable
// ---------------------------------------------------------------
int Observation::getChannel_SGS(std::string obs_type){
	
	if (obs_type == "C1")  {return this->c1_sgs;}
	if (obs_type == "L1")  {return this->l1_sgs;}
	if (obs_type == "L2")  {return this->l2_sgs;}
	if (obs_type == "P1")  {return this->p1_sgs;}
	if (obs_type == "P2")  {return this->p2_sgs;}
	if (obs_type == "D1")  {return this->d1_sgs;}
	if (obs_type == "D2")  {return this->d2_sgs;}
     	
	std::cout << "ERROR: type " << obs_type << " is not a valid type of observable" << std::endl;
	assert(false);
	return -1;
}

// ---------------------------------------------------------------
// Méthode permettant de tester un type d'observable
// ---------------------------------------------------------------
bool Observation::hasChannel(std::string obs_type){
	
	if (obs_type == "C1")  {return this->c1_valid;}
	if (obs_type == "L1")  {return this->l1_valid;}
	if (obs_type == "L2")  {return this->l2_valid;}
	if (obs_type == "P1")  {return this->p1_valid;}
	if (obs_type == "P2")  {return this->p2_valid;}
	if (obs_type == "D1")  {return this->d1_valid;}
	if (obs_type == "D2")  {return this->d2_valid;}
     	
	std::cout << "ERROR: type " << obs_type << " is not a valid type of observable" << std::endl;
	assert(false);
	return -1;
	
}

// ---------------------------------------------------------------
// Méthode permettant d'enregistrer la position d'un type donné 
// d'observations dans une ligne d'observables.
// ---------------------------------------------------------------
void Observation::setChannel(std::string obs_type, double value, int lli, int sgs){
	
	if (obs_type == "C1")  {this->setC1(value); this->setC1_LLI(lli); this->setC1_SGS(sgs); return;}
	if (obs_type == "L1")  {this->setL1(value); this->setL1_LLI(lli); this->setL1_SGS(sgs); return;}
	if (obs_type == "L2")  {this->setL2(value); this->setL2_LLI(lli); this->setL2_SGS(sgs); return;}
	if (obs_type == "P1")  {this->setP1(value); this->setP1_LLI(lli); this->setP1_SGS(sgs); return;}
	if (obs_type == "P2")  {this->setP2(value); this->setP2_LLI(lli); this->setP2_SGS(sgs); return;}
	if (obs_type == "D1")  {this->setD1(value); this->setD1_LLI(lli); this->setD1_SGS(sgs); return;}
	if (obs_type == "D2")  {this->setD2(value); this->setD2_LLI(lli); this->setD2_SGS(sgs); return;}
     	
	std::cout << "ERROR: type " << obs_type << " is not a valid type of observable" << std::endl;
	assert(false);
	
}

// ---------------------------------------------------------------
// Méthode permettant d'enregistrer la position d'un type donné 
// d'observations dans une ligne d'observables.
// ---------------------------------------------------------------
void Observation::setChannel(std::string obs_type, double value){
	setChannel(obs_type, value, -1, -1);
}


// ---------------------------------------------------------------
// Méthode permettant de supprimer un type d'observation
// ---------------------------------------------------------------
void Observation::removeChannel(std::string obs_type){
	
	if (obs_type == "C1")  {this->setC1(-1); this->c1_valid=false; return;}
	if (obs_type == "L1")  {this->setL1(-1); this->l1_valid=false; return;}
	if (obs_type == "L2")  {this->setL2(-1); this->l2_valid=false; return;}
	if (obs_type == "P1")  {this->setP1(-1); this->p1_valid=false; return;}
	if (obs_type == "P2")  {this->setP2(-1); this->p2_valid=false; return;}
	if (obs_type == "D1")  {this->setD1(-1); this->d1_valid=false; return;}
	if (obs_type == "D2")  {this->setD2(-1); this->d2_valid=false; return;}
     	
	std::cout << "ERROR: type " << obs_type << " is not a valid type of observable" << std::endl;
	assert(false);
	
}

