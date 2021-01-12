#ifndef SP3NAVIGATIONSLOT_H
#define SP3NAVIGATIONSLOT_H

#include "Utils.h"
#include "GPSTime.h"
#include "ECEFCoords.h"

#include <vector>

// ---------------------------------------------------------------
// Classe contenant un paquet de données de navigation SP3
// ---------------------------------------------------------------
class SP3NavigationSlot {

	public:

		GPSTime time;

		std::vector<std::string> PRN;

		std::vector<double> X;
		std::vector<double> Y;
		std::vector<double> Z;
		std::vector<double> T;

		SP3NavigationSlot(GPSTime time){
			this->time = time;
		}

		void addData(std::string prn, double x, double y, double z, double t){
			PRN.push_back(prn); X.push_back(x); Y.push_back(y); Z.push_back(z); T.push_back(t);
		}

};

#endif
