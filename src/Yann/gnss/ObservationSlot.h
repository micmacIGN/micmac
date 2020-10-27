#ifndef OBSERVATIONSLOT_H
#define OBSERVATIONSLOT_H

#include <map>
#include <cassert>
#include <string>
#include <vector>

#include "Utils.h"
#include "GPSTime.h"
#include "Observation.h"

// ---------------------------------------------------------------
// Classe contenant un paquet d'observations GPS
// ---------------------------------------------------------------
class ObservationSlot {

	// Metadata
	GPSTime timestamp;

	// Contents
	std::map<std::string, Observation> slot;

	public:

		bool hasObservation(std::string);
		Observation& getObservation(std::string);
		GPSTime& getTimestamp(){return this->timestamp;}

		// Mutateurs
		void setTimestamp(GPSTime& timestamp){this->timestamp = timestamp;}

		// Méthodes
		size_t getNumberOfObservations(){return this->slot.size();}
		void addObservation(Observation& obs){this->slot[obs.getSatName()] = obs;}
		void removeSatellite(std::string);
		void removeConstellation(std::string);

		std::vector<std::string> getSatellites();
	    std::vector<std::string> getSatellitesConstellation(std::string);

        std::vector<double> getObservables(std::vector<std::string>, std::string);

};


// toString override
inline std::ostream & operator<<(std::ostream & Str, ObservationSlot& slot) {
	std::vector<std::string> list_of_sats = slot.getSatellites();
	Str << "Data slot at epoch: " << slot.getTimestamp() << " (" << list_of_sats.size() << " satellites)" << std::endl;
	for (unsigned int i=0; i<list_of_sats.size(); i++){
		Str << slot.getObservation(list_of_sats.at(i)) << std::endl;
	}
	return Str;
}

#endif
