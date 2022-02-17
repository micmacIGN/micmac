#include <cassert>

#include "GloNavigationData.h"
#include "GloNavigationSlot.h"

// -------------------------------------------------------------------------------
// Ajout d'un slot de navigation aux données
// -------------------------------------------------------------------------------
void GloNavigationData::addNavigationSlot(GloNavigationSlot &slot){

	// Add slot to nav data
	this->navSlots.push_back(slot);

	// Increment PRN number
	this->PRN_COUNT[slot.getPRN()-1]++;

}

// -------------------------------------------------------------------------------
// Test si les éphémérides contiennent une données respectant les 3 conditions :
//  (1) Le PRN correspond
//  (2) La date du paquet est < à la date requise
//  (3) La date du paquet est la plus récente possible
// -------------------------------------------------------------------------------
bool GloNavigationData::hasEphemeris(int PRN, GPSTime time){

    if ((PRN < 1) || (PRN > 32)){
       return false;
    }

	size_t idx;
	bool found = false;

	GloNavigationSlot slot;

	for (idx=this->navSlots.size()-1; idx>=0; idx--){
		if (this->navSlots.at(idx).getPRN() != PRN) continue;        // Contrainte 1
		if (this->navSlots.at(idx).getTime() > time) continue;	     // Contrainte 2
		found = true; break;									     // Contrainte 3
	}

	if ((!found) || (time - this->navSlots.at(idx).getTime() > GLO_COEFF_SECURITY*GLO_RINEX_NAV_INTERVAL_SEC)){
        return false;
	}

	return true;

}

// -------------------------------------------------------------------------------
// Test si les éphémérides contiennent une données respectant les 3 conditions :
//  (1) Le PRN correspond
//  (2) La date du paquet est < à la date requise
//  (3) La date du paquet est la plus récente possible
// -------------------------------------------------------------------------------
bool GloNavigationData::hasEphemeris(std::string PRN, GPSTime time){
    if (PRN.substr(0,1) != "R") return false;
    return this->hasEphemeris(std::stoi(PRN.substr(1,2)), time);
}


// -------------------------------------------------------------------------------
// Récupération du paquet d'éphéméride respectant les 3 conditions suivantes :
//  (1) Le PRN correspond
//  (2) La date du paquet est < à la date requise
//  (3) La date du paquet est la plus récente possible
// -------------------------------------------------------------------------------
GloNavigationSlot& GloNavigationData::getNavigationSlot(int PRN, GPSTime time){

    if ((PRN < 1) || (PRN > 24)){
        std::cout << "ERROR: [R" << Utils::formatNumber(PRN,"%02d") << "] is not a valid GLONASS PRN number" << std::endl;
		assert (false);
    }

	size_t idx;
	bool found = false;

	GloNavigationSlot slot;

	for (idx=this->navSlots.size()-1; idx>=0; idx--){
		if (this->navSlots.at(idx).getPRN() != PRN) continue;                                            // Contrainte 1
		if (this->navSlots.at(idx).getTime().addSeconds(this->getLeapSeconds()) > time) continue;	     // Contrainte 2
		found = true; break;									                                         // Contrainte 3
	}

	if ((!found) || (time - this->navSlots.at(idx).getTime() > GLO_COEFF_SECURITY*GLO_RINEX_NAV_INTERVAL_SEC)){
        std::cout << PRN << std::endl;
		std::cout << "ERROR: GPS time [" << time << "] is out of rinex nav file range for PRN [R";
		std::cout << Utils::formatNumber(PRN,"%02d") << "]" << std::endl;
		assert (false);
	}

	return this->navSlots.at(idx);

}


// -------------------------------------------------------------------------------
// Récupération du paquet d'éphéméride respectant les 3 conditions suivantes :
//  (1) Le PRN correspond
//  (2) La date du paquet est < à la date requise
//  (3) La date du paquet est la plus récente possible
// -------------------------------------------------------------------------------
GloNavigationSlot& GloNavigationData::getNavigationSlot(std::string PRN, GPSTime time){
	return this->getNavigationSlot(std::stoi(PRN.substr(1,2)), time);
}

// -------------------------------------------------------------------------------
// Calcul des positions des satellite à partir d'un slot de données de rinex .nav
// L'argument pseudorange permet de déduire le temps de propagation du signal
// -------------------------------------------------------------------------------
std::vector<ECEFCoords> GloNavigationData::computeSatellitePos(std::vector<std::string> PRN, GPSTime t, std::vector<double> psr){
    std::vector<ECEFCoords> XYZ;
    for (unsigned i=0; i<PRN.size(); i++){
        XYZ.push_back(computeSatellitePos(PRN.at(i), t, psr.at(i)));
    }
    return XYZ;
}

// -------------------------------------------------------------------------------
// Calcul des erreur d'horloge de tous les satellites
// L'argument pseudorange permet de déduire le temps de propagation du signal
// -------------------------------------------------------------------------------
std::vector<double> GloNavigationData::computeSatelliteClockError(std::vector<std::string> PRN, GPSTime t, std::vector<double> psr){
    std::vector<double> T;
    for (unsigned i=0; i<PRN.size(); i++){
        T.push_back(computeSatelliteClockError(PRN.at(i), t, psr.at(i)));
    }
    return T;
}

// -------------------------------------------------------------------------------
// Calcul d'une position de satellite à partir d'un slot de données de rinex .nav
// L'argument pseudorange permet de déduire le temps de propagation du signal
// -------------------------------------------------------------------------------
ECEFCoords GloNavigationData::computeSatellitePos(int PRN, GPSTime time, double pseudorange){
    GPSTime time2(time.convertToAbsTime()-this->leap_seconds);
    return this->getNavigationSlot(PRN, time).computeSatellitePos(time2, pseudorange);
}

// -------------------------------------------------------------------------------
// Calcul d'une position de satellite à partir d'un slot de données de rinex .nav
// L'argument pseudorange permet de déduire le temps de propagation du signal
// -------------------------------------------------------------------------------
ECEFCoords GloNavigationData::computeSatellitePos(std::string PRN, GPSTime time, double pseudorange){
    return this->computeSatellitePos(std::stoi(PRN.substr(1,2)), time, pseudorange);
}

// -------------------------------------------------------------------------------
// Calcul d'une position de satellite à partir d'un slot de données de rinex .nav
// -------------------------------------------------------------------------------
ECEFCoords GloNavigationData::computeSatellitePos(int PRN, GPSTime time){
	return this->computeSatellitePos(PRN, time, 0);
}

// -------------------------------------------------------------------------------
// Calcul d'une position de satellite à partir d'un slot de données de rinex .nav
// -------------------------------------------------------------------------------
ECEFCoords GloNavigationData::computeSatellitePos(std::string PRN, GPSTime time){
	return this->computeSatellitePos(std::stoi(PRN.substr(1,2)), time);
}

// -------------------------------------------------------------------------------
// Calcul de l'erreur d'horloge d'un satellite
// L'argument pseudorange permet de déduire le temps de propagation du signal
// -------------------------------------------------------------------------------
double GloNavigationData::computeSatelliteClockError(int PRN, GPSTime time, double pseudorange){
    GPSTime time2(time.convertToAbsTime()-this->leap_seconds);
    return this->getNavigationSlot(PRN, time).computeSatelliteClockError(time2, pseudorange);
}

// -------------------------------------------------------------------------------
// Calcul de l'erreur d'horloge d'un satellite
// L'argument pseudorange permet de déduire le temps de propagation du signal
// -------------------------------------------------------------------------------
double GloNavigationData::computeSatelliteClockError(std::string PRN, GPSTime time, double pseudorange){
    return this->computeSatelliteClockError(std::stoi(PRN.substr(1,2)), time, pseudorange);
}

// -------------------------------------------------------------------------------
// Calcul de l'erreur d'horloge d'un satellite
// -------------------------------------------------------------------------------
double GloNavigationData::computeSatelliteClockError(int PRN, GPSTime time){
    return this->computeSatelliteClockError(PRN, time, 0);
}


// -------------------------------------------------------------------------------
// Calcul de l'erreur d'horloge d'un satellite
// -------------------------------------------------------------------------------
double GloNavigationData::computeSatelliteClockError(std::string PRN, GPSTime time){
    GPSTime time2(time.convertToAbsTime()-this->leap_seconds);
    return this->computeSatelliteClockError(std::stoi(PRN.substr(1,2)), time2);
}
