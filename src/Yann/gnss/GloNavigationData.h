#ifndef GLONAVIGATIONDATA_H
#define GLONAVIGATIONDATA_H

#include <vector>
#include <string>

#include "Utils.h"
#include "GPSTime.h"
#include "ECEFCoords.h"
#include "GloNavigationSlot.h"

class GloNavigationData{


    private:

        // Meta data
        int leap_seconds = 0;
        std::string source_file_path = "";
        GPSTime time_ref_corr_time;
        double corr_time;

        // Data
        std::vector<GloNavigationSlot> navSlots;

        // Contents
        int PRN_COUNT[24] = {0};

        // Methods
        bool hasEphemeris(int, GPSTime);
        GloNavigationSlot& getNavigationSlot(int, GPSTime);
        ECEFCoords computeSatellitePos(int, GPSTime, double);
		ECEFCoords computeSatellitePos(int, GPSTime);
        double computeSatelliteClockError(int, GPSTime);
		double computeSatelliteClockError(int, GPSTime, double);

    public:

        // Accesseurs
		std::string getSourceFilePath(){return this->source_file_path;};
		std::vector<GloNavigationSlot>& getNavigationSlots(){return this->navSlots;}
		GloNavigationSlot& getNavigationSlot(int i){return this->navSlots.at(i);};
        GPSTime getTRefCorr(){return this->time_ref_corr_time;}
		size_t getNumberOfNavigationSlots(){return this->navSlots.size();};
		int* getPRNCount(){return this->PRN_COUNT;}
		int getLeapSeconds(){return this->leap_seconds;};
		double getCorrectionTime(){return this->corr_time;}

		// Mutateurs
		void setSourceFilePath(std::string source_file_path){this->source_file_path = source_file_path;};
		void setLeapSeconds(int leap_seconds){this->leap_seconds = leap_seconds;};
		void setNavigationSlots(std::vector<GloNavigationSlot>& navSlots){this->navSlots = navSlots;};
		void setTRefCorr(GPSTime time){this->time_ref_corr_time = time;}
		void setCorrectionTime(double corr_time){this->corr_time = corr_time;}

		// Méthodes
		bool hasEphemeris(std::string, GPSTime);
		void addNavigationSlot(GloNavigationSlot&);
		GloNavigationSlot& getNavigationSlot(std::string, GPSTime);
		ECEFCoords computeSatellitePos(std::string, GPSTime, double);
		ECEFCoords computeSatellitePos(std::string, GPSTime);
        double computeSatelliteClockError(std::string PRN, GPSTime);
        double computeSatelliteClockError(std::string, GPSTime, double);

        std::vector<ECEFCoords> computeSatellitePos(std::vector<std::string>, GPSTime, std::vector<double>);
        std::vector<double> computeSatelliteClockError(std::vector<std::string>, GPSTime, std::vector<double>);

};

#endif // GLONAVIGATIONDATA_H
