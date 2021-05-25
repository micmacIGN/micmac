#ifndef ATMOSPHERICMODEL_H
#define ATMOSPHERICMODEL_H

#include "Utils.h"
#include "NavigationData.h"

class AtmosphericModel{

    public:

        AtmosphericModel(NavigationData);
        AtmosphericModel(std::vector<double>, std::vector<double>);

        void setTemperature(double T) {this->T = T;}
        void setHumidity(double Rh) {this->Rh = Rh;}
        void setTime(GPSTime time) {this->time = time;}

        double tropo_correction(ECEFCoords, ECEFCoords);
        double tropo_correction_saastomonian(ECEFCoords, ECEFCoords);
        double iono_correction_raw(ECEFCoords);
        double iono_correction(ECEFCoords, ECEFCoords);
        double all_corrections(ECEFCoords, ECEFCoords);

    private:

        GPSTime time;

        std::vector<double> alpha{0.0, 0.0, 0.0, 0.0};
        std::vector<double> beta{0.0, 0.0, 0.0, 0.0};

        double T = 293;
        double P = 1023;
        double Rh = 0.6;

};

#endif // ATMOSPHERICMODEL_H
