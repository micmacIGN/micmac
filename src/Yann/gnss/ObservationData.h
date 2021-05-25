#ifndef OBSERVATIONDATA_H
#define OBSERVATIONDATA_H

#include <string>
#include <vector>

#include "Utils.h"
#include "ENUCoords.h"
#include "ECEFCoords.h"
#include "ObservationSlot.h"

// ---------------------------------------------------------------
// Classe contenant les données d'observations GPS
// ---------------------------------------------------------------
class ObservationData {

	// Meta data
	std::string source_file_path = "";

	// Header parameters
	ENUCoords delta_enu;
	ECEFCoords approximate_position;
	std::string receiver_type="";
	std::string antenna_type="";
	std::string marker_name="";
	std::string observation_type="G";
	double interval = 1;
	int leap_seconds;

	bool c1_valid = true;
	bool l1_valid = true;
	bool l2_valid = true;
	bool p1_valid = true;
	bool p2_valid = true;
	bool d1_valid = true;
	bool d2_valid = true;

	// Contents
	std::vector<std::string> comments;
	std::vector<ObservationSlot> ObservationSlots;


	public:

		// Accesseurs
		std::string getSourceFilePath(){return this->source_file_path;};
		ENUCoords getDeltaEnu(){return this->delta_enu;}
		ECEFCoords& getApproxMarkerPosition(){return this->approximate_position;}
		std::string getReceiverType(){return this->receiver_type;}
		std::string getAntennaType(){return this->antenna_type;}
		std::string getMarkerName(){return this->marker_name;}
		std::string getObservationType(){return this->observation_type;}
		std::vector<ObservationSlot>& getObservationSlots(){return this->ObservationSlots;}
		std::vector<std::string> getActivatedChannels();
		size_t getNumberOfObservationSlots(){return this->ObservationSlots.size();}
		int getLeapSeconds(){return this->leap_seconds;}
		double getInterval(){return this->interval;}
		GPSTime getTimeOfFirstObs(){return this->ObservationSlots.front().getTimestamp();}
		GPSTime getTimeOfLastObs(){return this->ObservationSlots.back().getTimestamp();}

		ECEFCoords getApproxAntennaPosition();

		// Mutateurs
		void setSourceFilePath(std::string source_file_path){this->source_file_path = source_file_path;};
		void setDeltaEnu(ENUCoords delta_enu){this->delta_enu = delta_enu;}
		void setApproximatePosition(ECEFCoords position){this->approximate_position = position;}
		void setReceiverType(std::string receiver_type){this->receiver_type = receiver_type;}
		void setAntennaType(std::string antenna_type){this->antenna_type = antenna_type;}
		void setMarkerName(std::string marker_name){this->marker_name = marker_name;}
		void setObservationType(std::string observation_type){this->observation_type = observation_type;}
		void setInterval(double interval){this->interval = interval;}
		void setLeapSeconds(int leap_seconds){this->leap_seconds = leap_seconds;}
		void setObservationSlots(std::vector<ObservationSlot> slots){this->ObservationSlots = slots;}

		// Méthodes
		void addObservationSlot(ObservationSlot& slot){this->ObservationSlots.push_back(slot);}
		void addComment(std::string comment){this->comments.push_back(comment);}
		void removeObservationSlot(int i){this->ObservationSlots.erase(this->ObservationSlots.begin()+i);}
		void removeConstellation(std::string);
		void removeSatellite(std::string);
		void printRinexFile(std::string);
		void activateChannel(std::string);
		void deactivateChannel(std::string);
		void extractSpanTime(GPSTime start, GPSTime end);
		void slice(int);

		ObservationSlot& lookForEpoch(GPSTime);


};


#endif
