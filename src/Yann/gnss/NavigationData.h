#ifndef NAVIGATIONDATA_H
#define NAVIGATIONDATA_H

#include <string>
#include <vector>

#include "Utils.h"
#include "NavigationSlot.h"

// ---------------------------------------------------------------
// Classe contenant les données de navigation GPS
// ---------------------------------------------------------------
class NavigationData {

	// Meta data
	std::string source_file_path = "";
	std::string constellation = "G";      // For v3.0 and later

	// Header parameters
	int leap_seconds = 0;
	std::vector<double> ion_alpha;
	std::vector<double> ion_beta;
	std::vector<double> utc_parameters;

	// Contents
	int PRN_COUNT[36] = {0};
	std::vector<NavigationSlot> navSlots;


	// Méthodes
	bool hasEphemeris(int, GPSTime);
	NavigationSlot& getNavigationSlot(int, GPSTime);
	double computeSatelliteClockError(int, GPSTime);
    double computeSatelliteClockError(int, GPSTime, double);
    ECEFCoords computeSatellitePos(int, GPSTime, double);
    ECEFCoords computeSatellitePos(int, GPSTime);


	public:

		// Accesseurs
		std::string getSourceFilePath(){return this->source_file_path;};
		int getLeapSeconds(){return this->leap_seconds;};
		std::vector<double> getIonoAlpha(){return this->ion_alpha;};
		std::vector<double> getIonoBeta(){return this->ion_beta;};
		std::vector<double> getUTCParameters(){return this->utc_parameters;};
		std::vector<NavigationSlot>& getNavigationSlots(){return this->navSlots;}
		std::string getConstellation(){return this->constellation;}
		NavigationSlot& getNavigationSlot(int i){return this->navSlots.at(i);};
		size_t getNumberOfNavigationSlots(){return this->navSlots.size();};
		int* getPRNCount(){return this->PRN_COUNT;}

		// Mutateurs
		void setSourceFilePath(std::string source_file_path){this->source_file_path = source_file_path;};
		void setLeapSeconds(int leap_seconds){this->leap_seconds = leap_seconds;};
		void setIonoAlpha(std::vector<double> ion_alpha){this->ion_alpha = ion_alpha;};
		void setIonoBeta(std::vector<double> ion_beta){this->ion_beta = ion_beta;};
		void setConstellation(std::string constellation){this->constellation = constellation;}
		void setUTCParameters(std::vector<double> utc_parameters){this->utc_parameters = utc_parameters;};
		void setNavigationSlots(std::vector<NavigationSlot>& navSlots){this->navSlots = navSlots;};

		// Méthodes
		bool hasEphemeris(std::string, GPSTime);
		void addNavigationSlot(NavigationSlot&);
		NavigationSlot& getNavigationSlot(std::string, GPSTime);
		ECEFCoords computeSatellitePos(std::string, GPSTime, double);
		ECEFCoords computeSatellitePos(std::string, GPSTime);
        double computeSatelliteClockError(std::string PRN, GPSTime);
        double computeSatelliteClockError(std::string, GPSTime, double);
        ECEFCoords computeSatelliteSpeed(std::string, GPSTime);

        std::vector<ECEFCoords> computeSatellitePos(std::vector<std::string>, GPSTime, std::vector<double>);
        std::vector<double> computeSatelliteClockError(std::vector<std::string>, GPSTime, std::vector<double>);
        std::vector<ECEFCoords> computeSatelliteSpeed(std::vector<std::string>, GPSTime);

};


#endif
