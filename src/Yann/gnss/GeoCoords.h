#ifndef GEOCOORDS_H
#define GEOCOORDS_H

#include <iostream>

#include "Utils.h"
#include "ENUCoords.h"
#include "ECEFCoords.h"

// ---------------------------------------------------------------
// Classe contenant les coordonnées géographiques
// ---------------------------------------------------------------

class ENUCoords;
class ECEFCoords;

class GeoCoords {

	public:

		GeoCoords(double lon=0, double lat=0, double h=0){
			this->longitude = lon;
			this->latitude = lat;
			this->height = h;
		}

		GeoCoords(std::vector<double> vec){
			this->longitude = vec.at(0);
			this->latitude = vec.at(1);
			this->height = vec.at(2);
		}

		double longitude = 0;
		double latitude = 0;
		double height = 0;

		ECEFCoords toECEFCoords();
	
		double distanceTo(GeoCoords);
		
		static std::string makeWKT(std::vector<GeoCoords>);

};

// toString override
inline std::ostream & operator<<(std::ostream & Str, GeoCoords coords) {
    Str <<  "[lon=" << Utils::formatNumber(coords.longitude, "%11.8f");
	Str << ", lat=" << Utils::formatNumber(coords.latitude, "%10.8f");
	Str << ", hgt=" << Utils::formatNumber(coords.height, "%6.3f") << "]";
	return Str;
}


#endif
