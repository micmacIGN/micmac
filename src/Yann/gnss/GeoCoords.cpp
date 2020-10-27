#include <cmath>

#include "Utils.h"
#include "GeoCoords.h"

ECEFCoords GeoCoords::toECEFCoords(){

	ECEFCoords xyz;

	double n, lat, lon, hgt;

	double f = Utils::Fe;
	double a = Utils::Re;
	double e = sqrt(f*(2-f));

	lon = Utils::deg2rad(this->longitude);
	lat = Utils::deg2rad(this->latitude);
	hgt = this->height;
	n = a/sqrt(1-pow(e*sin(lat),2));

	xyz.X = (n+hgt)*cos(lat)*cos(lon);
	xyz.Y = (n+hgt)*cos(lat)*sin(lon);
	xyz.Z = ((1-e*e)*n+hgt)*sin(lat);

	return xyz;

}

// -----------------------------------------------------------------
// Distance à un autre point de coordonnées géographiques
// -----------------------------------------------------------------
double GeoCoords::distanceTo(GeoCoords pt){
	
	return  this->toECEFCoords().distanceTo(pt.toECEFCoords());
	
}

// -----------------------------------------------------------------
// Conversion d'une liste de coordonnées géographiques
// en une chaîne de caractères WKT
// -----------------------------------------------------------------
std::string GeoCoords::makeWKT(std::vector<GeoCoords> trajectory){

    std::string wkt = "LINESTRING((";

    for (unsigned i=0; i<trajectory.size(); i++){
        wkt += Utils::formatNumber(trajectory.at(i).longitude, "%10.10f") + " ";
        wkt += Utils::formatNumber(trajectory.at(i).latitude, "%10.10f");
        if (i != trajectory.size()-1) wkt += ",";
    }

    wkt += "))";

    return wkt;

}
