#ifndef SOLUTION_H
#define SOLUTION_H

#include "Utils.h"
#include "GPSTime.h"
#include "ECEFCoords.h"

class Solution{

    private:

        ECEFCoords position;
        ECEFCoords speed;

        double delta_time;
        double clock_drift;
        GPSTime timestamp;

        int nb_satellites_visible;
        int nb_satellites_used;

        double pdop;
        double vdop;
        double hdop;
        double tdop;
        double gdop;

        std::vector<std::string> used_satellites;
		std::vector<double> residuals;

        // -------------------------------------
        // Code de réussite du calcul
        // -------------------------------------
        // 0 : estimation réussie
        // 1 : pas assez de satellites visibles
        // -------------------------------------
        int code = 0;


    public:

        // Mutateurs
        void setPosition(ECEFCoords& position){this->position = position;}
        void setSpeed(ECEFCoords& speed){this->speed = speed;}
        void setDeltaTime(double time){this->delta_time = time;}
        void setClockDrift(double drift){this->clock_drift = drift;}
        void setNumberOfVisibleSatellites(int nb){this->nb_satellites_visible = nb;}
        void setTimestamp(GPSTime timestamp){this->timestamp = timestamp;}
        void addUsedSatellite(std::string sat_name){this->used_satellites.push_back(sat_name);}
        void setUsedSatellites(std::vector<std::string>& used_sats){this->used_satellites = used_sats;}
		void setResiduals(std::vector<double> residuals){this->residuals = residuals;}

        void setPDOP(double pdop){this->pdop = pdop;}
        void setVDOP(double vdop){this->vdop = vdop;}
        void setHDOP(double hdop){this->hdop = hdop;}
        void setTDOP(double tdop){this->tdop = tdop;}
        void setGDOP(double gdop){this->gdop = gdop;}

        int getCode(){return this->code;}

        // Accesseurs
        ECEFCoords& getPosition(){return this->position;}
        ECEFCoords& getSpeed(){return this->speed;}
        double getDeltaTime(){return this->delta_time;}
        double getClockDrift(){return this->clock_drift;}
        int getNumberOfVisibleSatellites(){return this->nb_satellites_visible;}
        size_t getNumberOfUsedSatellites(){return this->used_satellites.size();}
        std::vector<std::string> getUsedSatellites(){return this->used_satellites;}
		std::vector<double> getResiduals(){return this->residuals;}

        GPSTime getTimestamp(){return this->timestamp;}

        double getPDOP(){return this->pdop;}
        double getVDOP(){return this->vdop;}
        double getHDOP(){return this->hdop;}
        double getTDOP(){return this->tdop;}
        double getGDOP(){return this->gdop;}

        void setCode(int code){this->code = code;}

};

// toString override
inline std::ostream & operator<<(std::ostream & Str, Solution solution) {
    ENUCoords v = solution.getSpeed().toENUSpeed(solution.getPosition());
	Str <<  solution.getTimestamp() << " " << solution.getPosition().toGeoCoords() << " ";
    Str <<  "[Ve=" << Utils::formatNumber(v.E, "%6.3f");
	Str << ", Vn=" << Utils::formatNumber(v.N, "%6.3f");
	Str << ", Vu=" << Utils::formatNumber(v.U, "%6.3f") << "] ";
	Str << solution.getNumberOfVisibleSatellites() << " ";
	Str << solution.getNumberOfUsedSatellites() << " ";
	Str << solution.getPDOP();
	return Str;
}

#endif // SOLUTION_H
