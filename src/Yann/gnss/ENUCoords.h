#ifndef ENUCOORDS_H
#define ENUCOORDS_H

#include <cmath>
#include <iostream>

#include "Utils.h"
#include "ECEFCoords.h"

// ---------------------------------------------------------------
// Classe contenant les coordonnées planes dans un repère RTL
// ---------------------------------------------------------------

class ECEFCoords;

class ENUCoords {

	public:

		ENUCoords(double E=0, double N=0, double U=0){
			this->E = E;
			this->N = N;
			this->U = U;
		}

		ENUCoords(std::vector<double> vec){
			this->E = vec.at(0);
			this->N = vec.at(1);
			this->U = vec.at(2);
		}

		double E = 0;
		double N = 0;
		double U = 0;

		// Conversion
		ECEFCoords toECEFCoords(ECEFCoords ref);

		// Calculs
		double norm2D();
		double norm3D();
		double dot(ENUCoords);
		double distanceTo(ENUCoords);
		double elevationTo(ENUCoords);
		double azimuthTo(ENUCoords);

		ENUCoords operator- (ENUCoords p){
			ENUCoords output;
			output.E = this->E - p.E;
			output.N = this->N - p.N;
			output.U = this->U - p.U;
			return output;
		}

		ENUCoords operator+ (ENUCoords p){
			ENUCoords output;
			output.E = this->E + p.E;
			output.N = this->N + p.N;
			output.U = this->U + p.U;
			return output;
		}

};

// toString override
inline std::ostream & operator<<(std::ostream & Str, ENUCoords coords) {
    Str <<  "[E=" << Utils::formatNumber(coords.E, "%14.3f");
	Str << ", N=" << Utils::formatNumber(coords.N, "%14.3f");
	Str << ", U=" << Utils::formatNumber(coords.U, "%14.3f") << "]";
	return Str;
}




#endif
