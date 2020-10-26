#ifndef ECEFCOORDS_H
#define ECEFCOORDS_H

#include <string>
#include <iostream>

#include "Utils.h"
#include "GeoCoords.h"
#include "ENUCoords.h"

// ---------------------------------------------------------------
// Classe contenant les coordonnées dans le repère ECEF
// ---------------------------------------------------------------

class GeoCoords;
class ENUCoords;

class ECEFCoords {

	public:

		ECEFCoords(double X=0, double Y=0, double Z=0){
			this->X = X;
			this->Y = Y;
			this->Z = Z;
		}

		ECEFCoords(std::vector<double> vec){
			this->X = vec.at(0);
			this->Y = vec.at(1);
			this->Z = vec.at(2);
		}

		double X = 0;
		double Y = 0;
		double Z = 0;

		// Conversions
		GeoCoords toGeoCoords();
		ENUCoords toENUCoords(ECEFCoords);
		void shiftECEFRef(double);
		void rotate(double);
		void scalar(double);
		void normalize();

		// Calculs
		double norm();
		double dot(ECEFCoords);
		double distanceTo(ECEFCoords);
		double elevationTo(ECEFCoords);
		double azimuthTo(ECEFCoords);

        ENUCoords toENUSpeed(ECEFCoords);

		// Vecteur entre points
		ECEFCoords operator- (ECEFCoords p){
			ECEFCoords output;
			output.X = this->X - p.X;
			output.Y = this->Y - p.Y;
			output.Z = this->Z - p.Z;
			return output;
		}

		// Somme de vecteurs
		ECEFCoords operator+ (ECEFCoords p){
			ECEFCoords output;
			output.X = this->X + p.X;
			output.Y = this->Y + p.Y;
			output.Z = this->Z + p.Z;
			return output;
		}


};


// toString override
inline std::ostream & operator<<(std::ostream & Str, ECEFCoords coords) {
	Str <<  "[X=" << Utils::formatNumber(coords.X, "%14.3f");
	Str << ", Y=" << Utils::formatNumber(coords.Y, "%14.3f");
	Str << ", Z=" << Utils::formatNumber(coords.Z, "%14.3f") << "]";
	return Str;
}


#endif
