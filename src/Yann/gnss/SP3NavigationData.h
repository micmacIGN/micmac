#ifndef SP3NAVIGATIONDATA_H
#define SP3NAVIGATIONDATA_H

#include "Utils.h"
#include "GPSTime.h"
#include "ECEFCoords.h"
#include "SP3NavigationSlot.h"

#include <vector>

// ---------------------------------------------------------------
// Classe contenant des données de navigation au format SP3
// ---------------------------------------------------------------
class SP3NavigationData {

	private:

        int ORDER_LAGRANGE = 9;
        static int lineIndexFromPRN(std::string, std::vector<std::string>);
		static double lagrange_interpolation(double, std::vector<double>&, std::vector<double>&);
		std::vector<int> select_indices_for_lagrange(GPSTime);

	public:

        bool hasEphemeris(std::string PRN);

        void setLagrangeOrder(int order){this->ORDER_LAGRANGE = order;}

		std::vector<SP3NavigationSlot> slots;

		ECEFCoords computeSatellitePos(std::string, GPSTime, double);
		ECEFCoords computeSatellitePos(std::string, GPSTime);

        double computeSatelliteClockError(std::string PRN, GPSTime);
        double computeSatelliteClockError(std::string, GPSTime, double);

};

#endif
