#ifndef STATISTICS_H
#define STATISTICS_H

#include "Utils.h"
#include "ECEFCoords.h"

class Statistics {

    public:

        static double mean(std::vector<double>);
        static double meanAbs(std::vector<double>);
        static double mse(std::vector<double>);
        static double rmse(std::vector<double>);
        static double sd(std::vector<double>);
        static double min(std::vector<double>);
        static double max(std::vector<double>);
        static int argmin(std::vector<double>);
        static int argmax(std::vector<double>);

        static ECEFCoords mean(std::vector<ECEFCoords>);
        static ECEFCoords meanAbs(std::vector<ECEFCoords>);
        static ECEFCoords mse(std::vector<ECEFCoords>);
        static ECEFCoords rmse(std::vector<ECEFCoords>);
        static ECEFCoords sd(std::vector<ECEFCoords>);
        static ECEFCoords min(std::vector<ECEFCoords>);
        static ECEFCoords max(std::vector<ECEFCoords>);

        static ENUCoords mean(std::vector<ENUCoords>);
        static ENUCoords meanAbs(std::vector<ENUCoords>);
        static ENUCoords mse(std::vector<ENUCoords>);
        static ENUCoords rmse(std::vector<ENUCoords>);
        static ENUCoords sd(std::vector<ENUCoords>);
        static ENUCoords min(std::vector<ENUCoords>);
        static ENUCoords max(std::vector<ENUCoords>);

};

#endif // STATISTICS_H
