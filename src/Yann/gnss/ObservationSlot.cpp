#include "ObservationSlot.h"

// Test d'accès aux données d'un satellite PRN
bool ObservationSlot::hasObservation(std::string sat_name){
	std::map<std::string, Observation>::iterator i = this->slot.find(sat_name);
	return (i != this->slot.end());
}

// Accès aux données d'un satellite PRN
Observation& ObservationSlot::getObservation(std::string sat_name){
	std::map<std::string, Observation>::iterator i = this->slot.find(sat_name);
	if (!hasObservation(sat_name)){
		std::cout << "ERROR: no observation for SVN [" << sat_name << "] in slot " << timestamp << std::endl;
		assert(false);
	}
	return i->second;
}

// Liste des satellites visibles
std::vector<std::string> ObservationSlot::getSatellites(){
    std::vector<std::string> list_of_sats;
    for (auto const& element : this->slot) {
        std::string prn = element.first;
        list_of_sats.push_back(Utils::makeStdSatName(prn));
    }
    return list_of_sats;
}

// Liste de satellites d'une constellation
std::vector<std::string> ObservationSlot::getSatellitesConstellation(std::string constellation_name){
	std::vector<std::string> list_of_sats = this->getSatellites();
	std::vector<std::string> list_of_sats_const;
	for (unsigned int i=0; i<list_of_sats.size(); i++){
		if (list_of_sats.at(i).substr(0,1) == constellation_name){
			list_of_sats_const.push_back(list_of_sats.at(i));
		}
	}
	return list_of_sats_const;
}

// Récupération des observables d'un ensemble de satellites
std::vector<double> ObservationSlot::getObservables(std::vector<std::string> prn, std::string channel){
    std::vector<double> observables;
    for (unsigned i=0; i<prn.size(); i++){
        observables.push_back(this->getObservation(prn.at(i)).getChannel(channel));
    }
    return observables;
}

// Suppression d'un satellite
void ObservationSlot::removeSatellite(std::string sat_name){
	if (this->hasObservation(sat_name)){
		this->slot.erase(sat_name);
	}
}

// Suppression d'une constellation
void ObservationSlot::removeConstellation(std::string constellation_name){
	std::vector<std::string> sat_names = this->getSatellitesConstellation(constellation_name);
	for (unsigned int i=0; i<sat_names.size(); i++){
		this->removeSatellite(sat_names.at(i));
	}
}

