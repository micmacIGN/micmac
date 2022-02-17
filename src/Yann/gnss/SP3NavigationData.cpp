#include "SP3NavigationData.h"

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <stdio.h>
#include <string>
#include <vector>
#include <assert.h>

// -------------------------------------------------------------------------------
// Calcul de l'interpolateur de lagrange des données (X,Y) au point x
// -------------------------------------------------------------------------------
double SP3NavigationData::lagrange_interpolation(double x, std::vector<double>& X, std::vector<double>& Y){

	double y = 0;
	double l;
	size_t order = X.size();

	for (unsigned i=0; i<order; i++){
		l = 1;
		for (unsigned j=0; j<order; j++){
			if (i==j) continue;
			l = l*(x-X.at(j))/(X.at(i)-X.at(j));
		}
		y = y+Y.at(i)*l;
	}

	return y;

}

// -------------------------------------------------------------------------------
// Méthode privée de sélection des indices d'interpolation
// -------------------------------------------------------------------------------
std::vector<int> SP3NavigationData::select_indices_for_lagrange(GPSTime time){

	unsigned int idx;
	std::vector<int> indices;

	if ((time-this->slots.front().time < 0) || (time-this->slots.back().time > SP3_INTERVAL_SEC)){
		std::cout << "ERROR: GPS time [" << time << "] is out of SP3 file range" << std::endl;
		assert (false);
		return indices;
	}

	for (idx=1; idx<this->slots.size(); idx++){
		if (this->slots.at(idx).time - time > 0) break;
	}

	int id_avant = ORDER_LAGRANGE/2-1;
	int id_apres = ORDER_LAGRANGE/2-1;
	size_t N = this->slots.size()-idx;

	if (time - this->slots.at(idx-1).time <= SP3_INTERVAL_SEC/2){
		id_avant ++;
	}else{
		id_apres ++;
	}

	id_avant = std::min(id_avant, (int)idx-1);
	id_apres = std::min(id_apres, static_cast<int>(N));

	for (unsigned int i=idx-1-id_avant; i<=idx+1+id_apres; i++){
		indices.push_back(i);
	}

	return indices;

}

// -------------------------------------------------------------------------------
// Méthode pour tester si le PRN appartient aux données (pas de temps nécessaire)
// -------------------------------------------------------------------------------
bool SP3NavigationData::hasEphemeris(std::string prn){
    std::vector<std::string> PRN = this->slots.at(0).PRN;
    for (unsigned i=0; i<PRN.size(); i++){
        if (Utils::prn_equal(PRN.at(i), prn)) return true;
    }
    return false;
}

// -------------------------------------------------------------------------------
// Méthode privée de récupération de l'indice de ligne d'un PRN
// Cette méthode suppose que tous les slots sont ordonnancés de la même manière
// -------------------------------------------------------------------------------
int SP3NavigationData::lineIndexFromPRN(std::string prn, std::vector<std::string> PRN){

	for (unsigned i=0; i<PRN.size(); i++){
        if (Utils::prn_equal(PRN.at(i), prn)) return i;
    }

    std::cout << "ERROR: [" << prn << "] can't be found in sp3 navigation file" << std::endl;
    assert (false);
	return -1;

}


// -------------------------------------------------------------------------------
// Calcul de l'erreur d'horloge (en s) du satellite prn à partir d'un slot sp3
// L'argument pseudorange permet de déduire le temps de propagation du signal
// Le calcul est effectué par interpolation lagrangienne d'ordre ORDER_LAGRANGE.
// -------------------------------------------------------------------------------
double SP3NavigationData::computeSatelliteClockError(std::string prn_code, GPSTime time, double pseudorange){

	// --------------------------------------------------------
	// Sélection des slots de navigation "encadrants"
	// --------------------------------------------------------
	std::vector<int> indices = this->select_indices_for_lagrange(time);

	// --------------------------------------------------------
	// Récupération du numéro de ligne du PRN
	// --------------------------------------------------------
    int prn = lineIndexFromPRN(prn_code, this->slots.at(0).PRN);

	// --------------------------------------------------------
	// Formation des données
	// --------------------------------------------------------
    std::vector<double> Dt;
	std::vector<double> T;

	for (unsigned int i=0; i<indices.size(); i++){

		int id_sp3 = indices.at(i);

		if (this->slots.at(id_sp3).T.at(prn) == Utils::SP3_NO_DATA_VAL) continue;

		Dt.push_back(this->slots.at(id_sp3).T.at(prn));
		T.push_back(this->slots.at(id_sp3).time.convertToAbsTime());

	}

	// --------------------------------------------------------
	// Interpolation
	// --------------------------------------------------------
	double t = time.convertToAbsTime() - pseudorange/Utils::C;

    if (Dt.size() == 0) {
        return Utils::SP3_NO_DATA_VAL;
    }

    // --------------------------------------------------------
	// Correction relativiste
	// --------------------------------------------------------
	double X1 = this->slots.at(indices.front()).X.at(prn);
	double Y1 = this->slots.at(indices.front()).Y.at(prn);
	double Z1 = this->slots.at(indices.front()).Z.at(prn);
	double T1 = T.front();

    double X2 = this->slots.at(indices.back()).X.at(prn);
	double Y2 = this->slots.at(indices.back()).Y.at(prn);
	double Z2 = this->slots.at(indices.back()).Z.at(prn);
	double T2 = T.back();

    ECEFCoords r1(X1, Y1, Z1); r1.scalar(1e3);
    ECEFCoords r2(X2, Y2, Z2); r2.scalar(1e3);

    ECEFCoords r12 = r1+r2; r12.scalar(0.5);
    ECEFCoords v12 = r2-r1; v12.scalar(1/(T2-T1));

    double relativistic = -2*r12.dot(v12)/(Utils::C*Utils::C);

	return lagrange_interpolation(t, T, Dt)*1e-6 + relativistic;

}

// -------------------------------------------------------------------------------
// Calcul de l'erreur d'horloge du satellite prn à partir d'un slot sp3
// Le calcul est effectué par interpolation lagrangienne d'ordre ORDER_LAGRANGE.
// -------------------------------------------------------------------------------
double SP3NavigationData::computeSatelliteClockError(std::string prn_code, GPSTime time){
	return computeSatelliteClockError(prn_code, time, 0);
}


// -------------------------------------------------------------------------------
// Calcul de la position du satellite prn à partir d'un slot de données sp3
// L'argument pseudorange permet de déduire le temps de propagation du signal
// Le calcul est effectué par interpolation lagrangienne d'ordre ORDER_LAGRANGE.
// -------------------------------------------------------------------------------
ECEFCoords SP3NavigationData::computeSatellitePos(std::string prn_code, GPSTime time, double pseudorange){

	ECEFCoords xyz;

	// --------------------------------------------------------
	// Sélection des slots de navigation "encadrants"
	// --------------------------------------------------------
	std::vector<int> indices = this->select_indices_for_lagrange(time);

	// --------------------------------------------------------
	// Récupération du numéro de ligne du PRN
	// --------------------------------------------------------
    int prn = lineIndexFromPRN(prn_code, this->slots.at(0).PRN);

	// --------------------------------------------------------
	// Formation des données
	// --------------------------------------------------------
	std::vector<double> X;
	std::vector<double> Y;
	std::vector<double> Z;
	std::vector<double> T;

	for (unsigned int i=0; i<indices.size(); i++){

		int id_sp3 = indices.at(i);

		X.push_back(this->slots.at(id_sp3).X.at(prn));
		Y.push_back(this->slots.at(id_sp3).Y.at(prn));
		Z.push_back(this->slots.at(id_sp3).Z.at(prn));
		T.push_back(this->slots.at(id_sp3).time.convertToAbsTime());

	}

	// --------------------------------------------------------
	// Interpolation
	// --------------------------------------------------------
	double t = time.convertToAbsTime() - pseudorange/Utils::C;
	xyz.X = lagrange_interpolation(t, T, X)*1e3;
	xyz.Y = lagrange_interpolation(t, T, Y)*1e3;
	xyz.Z = lagrange_interpolation(t, T, Z)*1e3;

	// --------------------------------------------------------
	// Rotation dans le repère ECI
	// --------------------------------------------------------
	xyz.shiftECEFRef(pseudorange/Utils::C);

	return xyz;

}


// -------------------------------------------------------------------------------
// Calcul de la position du satellite prn à partir d'un slot de données sp3
// Le calcul est effectué par interpolation lagrangienne d'ordre ORDER_LAGRANGE.
// -------------------------------------------------------------------------------
ECEFCoords SP3NavigationData::computeSatellitePos(std::string prn, GPSTime time){
	return this->computeSatellitePos(prn, time, 0);
}


