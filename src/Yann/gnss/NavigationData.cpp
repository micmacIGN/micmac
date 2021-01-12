#include <assert.h>

#include "NavigationData.h"


// -------------------------------------------------------------------------------
// Ajout d'un slot de navigation aux données
// -------------------------------------------------------------------------------
void NavigationData::addNavigationSlot(NavigationSlot &slot){

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
bool NavigationData::hasEphemeris(int PRN, GPSTime time){

    if ((PRN < 1) || (PRN > 36)){
       return false;
    }

	size_t idx;
	bool found = false;

	NavigationSlot slot;

	for (idx=this->navSlots.size()-1; idx>=0; idx--){
		if (this->navSlots.at(idx).getPRN() != PRN) continue;        // Contrainte 1
		if (this->navSlots.at(idx).getTime() > time) continue;	     // Contrainte 2
		found = true; break;									     // Contrainte 3
	}

	if ((!found) || (time - this->navSlots.at(idx).getTime() > COEFF_SECURITY*RINEX_NAV_INTERVAL_SEC)){
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
bool NavigationData::hasEphemeris(std::string PRN, GPSTime time){
    if (PRN.substr(0,1) != this->constellation) return false;
    return this->hasEphemeris(std::stoi(PRN.substr(1,2)), time);
}


// -------------------------------------------------------------------------------
// Récupération du paquet d'éphéméride respectant les 3 conditions suivantes :
//  (1) Le PRN correspond
//  (2) La date du paquet est < à la date requise
//  (3) La date du paquet est la plus récente possible
// -------------------------------------------------------------------------------
NavigationSlot& NavigationData::getNavigationSlot(int PRN, GPSTime time){

    if ((PRN < 1) || (PRN > 32)){
        std::cout << "ERROR: [" << this->constellation << Utils::formatNumber(PRN,"%02d");
        std::cout << "] is not a valid GNSS PRN number" << std::endl;
		assert (false);
    }

	size_t idx;
	bool found = false;

	NavigationSlot slot;

	for (idx=this->navSlots.size()-1; idx>=0; idx--){
		if (this->navSlots.at(idx).getPRN() != PRN) continue;        // Contrainte 1
		if (this->navSlots.at(idx).getTime() > time) continue;	     // Contrainte 2
		found = true; break;									     // Contrainte 3
	}

	if ((!found) || (time - this->navSlots.at(idx).getTime() > COEFF_SECURITY*RINEX_NAV_INTERVAL_SEC)){
        std::cout << PRN<< std::endl;
		std::cout << "ERROR: GNSS time [" << time << "] is out of rinex nav file range for PRN [";
		std::cout << this->constellation << Utils::formatNumber(PRN,"%02d") << "]" << std::endl;
		assert (false);
	}

	return this->navSlots.at(idx);

}

// -------------------------------------------------------------------------------
// Calcul des positions des satellite à partir d'un slot de données de rinex .nav
// L'argument pseudorange permet de déduire le temps de propagation du signal
// -------------------------------------------------------------------------------
std::vector<ECEFCoords> NavigationData::computeSatellitePos(std::vector<std::string> PRN, GPSTime t, std::vector<double> psr){
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
std::vector<double> NavigationData::computeSatelliteClockError(std::vector<std::string> PRN, GPSTime t, std::vector<double> psr){
    std::vector<double> T;
    for (unsigned i=0; i<PRN.size(); i++){
        T.push_back(computeSatelliteClockError(PRN.at(i), t, psr.at(i)));
    }
    return T;
}

// -------------------------------------------------------------------------------
// Récupératuion du paquet d'éphéméride respectant les 3 conditions suivantes :
//  (1) Le PRN correspond
//  (2) La date du paquet est < à la date requise
//  (3) La date du paquet est la plus récente possible
// -------------------------------------------------------------------------------
NavigationSlot& NavigationData::getNavigationSlot(std::string PRN, GPSTime time){
	return this->getNavigationSlot(std::stoi(PRN.substr(1,2)), time);
}

// -------------------------------------------------------------------------------
// Calcul d'une position de satellite à partir d'un slot de données de rinex .nav
// L'argument pseudorange permet de déduire le temps de propagation du signal
// -------------------------------------------------------------------------------
ECEFCoords NavigationData::computeSatellitePos(int PRN, GPSTime time, double pseudorange){
    return this->getNavigationSlot(PRN, time).computeSatellitePos(time, pseudorange);
}

// -------------------------------------------------------------------------------
// Calcul d'une position de satellite à partir d'un slot de données de rinex .nav
// L'argument pseudorange permet de déduire le temps de propagation du signal
// -------------------------------------------------------------------------------
ECEFCoords NavigationData::computeSatellitePos(std::string PRN, GPSTime time, double pseudorange){
    return this->computeSatellitePos(std::stoi(PRN.substr(1,2)), time, pseudorange);
}

// -------------------------------------------------------------------------------
// Calcul d'une position de satellite à partir d'un slot de données de rinex .nav
// -------------------------------------------------------------------------------
ECEFCoords NavigationData::computeSatellitePos(int PRN, GPSTime time){
	return this->computeSatellitePos(PRN, time, 0);
}

// -------------------------------------------------------------------------------
// Calcul d'une position de satellite à partir d'un slot de données de rinex .nav
// -------------------------------------------------------------------------------
ECEFCoords NavigationData::computeSatellitePos(std::string PRN, GPSTime time){
	return this->computeSatellitePos(std::stoi(PRN.substr(1,2)), time);
}

// -------------------------------------------------------------------------------
// Calcul de l'erreur d'horloge d'un satellite
// L'argument pseudorange permet de déduire le temps de propagation du signal
// -------------------------------------------------------------------------------
double NavigationData::computeSatelliteClockError(int PRN, GPSTime time, double pseudorange){
    return this->getNavigationSlot(PRN, time).computeSatelliteClockError(time, pseudorange);
}

// -------------------------------------------------------------------------------
// Calcul de l'erreur d'horloge d'un satellite
// L'argument pseudorange permet de déduire le temps de propagation du signal
// -------------------------------------------------------------------------------
double NavigationData::computeSatelliteClockError(std::string PRN, GPSTime time, double pseudorange){
    return this->computeSatelliteClockError(std::stoi(PRN.substr(1,2)), time, pseudorange);
}

// -------------------------------------------------------------------------------
// Calcul de l'erreur d'horloge d'un satellite
// -------------------------------------------------------------------------------
double NavigationData::computeSatelliteClockError(int PRN, GPSTime time){
    return this->computeSatelliteClockError(PRN, time, 0);
}


// -------------------------------------------------------------------------------
// Calcul de l'erreur d'horloge d'un satellite
// -------------------------------------------------------------------------------
double NavigationData::computeSatelliteClockError(std::string PRN, GPSTime time){
    return this->computeSatelliteClockError(std::stoi(PRN.substr(1,2)), time);
}

// -------------------------------------------------------------------------------
// Vitesse (ECEF) d'un satellite par différence finie centrée
// -------------------------------------------------------------------------------
ECEFCoords NavigationData::computeSatelliteSpeed(std::string PRN, GPSTime time){
	
	// Différence centrée en O(h^2)
	if ((this->hasEphemeris(PRN, time.addSeconds(-1))) && (this->hasEphemeris(PRN, time.addSeconds(+1)))){
    	ECEFCoords sat_pos_bwd = this->computeSatellitePos(PRN, time.addSeconds(-1));
    	ECEFCoords sat_pos_fwd = this->computeSatellitePos(PRN, time.addSeconds(+1));
    	ECEFCoords speed = sat_pos_fwd - sat_pos_bwd;
    	speed.scalar(0.5);
		return speed;
	}
		
	// Différence finie en O(h)	
	ECEFCoords sat_pos_median = this->computeSatellitePos(PRN, time);	
		
	// Différence avant 
	if (this->hasEphemeris(PRN, time.addSeconds(+1))){
		ECEFCoords sat_pos_fwd = this->computeSatellitePos(PRN, time.addSeconds(+1));
		ECEFCoords speed = sat_pos_fwd - sat_pos_median;
		return speed;
	}	
		
	// Différence arrière 
	if (this->hasEphemeris(PRN, time.addSeconds(-1))){
		ECEFCoords sat_pos_bwd = this->computeSatellitePos(PRN, time.addSeconds(-1));
		ECEFCoords speed = sat_pos_median - sat_pos_bwd;
		return speed;
	}	
	
	ECEFCoords null;
	return null;
   
}

// -------------------------------------------------------------------------------
// Vitesses (ECEF) d'un groupe de satellites par différence finie centrée
// -------------------------------------------------------------------------------
std::vector<ECEFCoords> NavigationData::computeSatelliteSpeed(std::vector<std::string> PRN, GPSTime time){
    std::vector<ECEFCoords> speeds;
    for (unsigned i=0; i<PRN.size(); i++){
        speeds.push_back(this->computeSatelliteSpeed(PRN.at(i), time));
    }
    return speeds;
}
