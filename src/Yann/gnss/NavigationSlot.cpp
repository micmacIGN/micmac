#include <assert.h>

#include "NavigationSlot.h"
#include "RinexReader.h"

void NavigationSlot::fillLine(std::vector<double> val, int line_number){

	switch (line_number){
		case 1:
			this->setSvClockBias(val[0]);
			this->setSvClockDrift(val[1]);
			this->setSvClockDriftRate(val[2]);
			break;
		case 2:
			this->setIODE(val[0]);
			this->setCrs(val[1]);
			this->setDeltaN(val[2]);
			this->setM0(val[3]);
			break;
		case 3:
			this->setCuc(val[0]);
			this->setEcc(val[1]);
			this->setCus(val[2]);
			this->setSqrtA(val[3]);
			break;
		case 4:
			this->setToe(val[0]);
			this->setCic(val[1]);
			this->setOMEGA(val[2]);
			this->setCis(val[3]);
			break;
		case 5:
			this->setI0(val[0]);
			this->setCrc(val[1]);
			this->setOmega(val[2]);
			this->setOMEGADOT(val[3]);
			break;
		case 6:
			this->setIDOT(val[0]);
			this->setCodeL2(val[1]);
			this->setGpsWeek(val[2]);
			this->setL2P(val[3]);
			break;
		case 7:
			this->setSVAccuracy(val[0]);
			this->setSVHealth(val[1]);
			this->setTgd(val[2]);
			this->setIODC(val[3]);
			break;
		default:
			break;
	}
}

void NavigationSlot::print(){

	std::cout << "------------------------------------------------------" << std::endl;
	std::cout << "PRN " << this->prn << ": " << this->getTime()           << std::endl;
	std::cout << "------------------------------------------------------" << std::endl;

	std::cout << "SV clock bias       (seconds)       " << sv_clock_bias << std::endl;
	std::cout << "SV clock drift      (sec/sec)       " << sv_clock_drift << std::endl;
	std::cout << "SV clock drift rate (sec/sec2)      " << sv_clock_drift_rate << std::endl;
	std::cout << "IODE Issue of Data, Ephemeris       " << iode << std::endl;
	std::cout << "Crs                 (meters)        " << crs << std::endl;
	std::cout << "Delta n             (radians/sec)   " << delta_n << std::endl;
	std::cout << "M0                  (radians)       " << m0 << std::endl;
	std::cout << "Cuc                 (radians)       " << cuc << std::endl;
	std::cout << "e Eccentricity                      " << ecc << std::endl;
	std::cout << "Cus                 (radians)	    " << cus << std::endl;
	std::cout << "sqrt(A)             (sqrt(m))       " << sqrt_a << std::endl;
	std::cout << "Toe         (sec of GPS week)       " << toe << std::endl;
	std::cout << "Cic                 (radians)       " << cic << std::endl;
	std::cout << "OMEGA               (radians)       " << OMEGA << std::endl;
	std::cout << "CIS                 (radians)       " << cis << std::endl;
	std::cout << "i0                  (radians)       " << i0 << std::endl;
	std::cout << "Crc                 (meters)        " << crc << std::endl;
	std::cout << "omega               (radians)       " << omega << std::endl;
	std::cout << "OMEGA DOT           (radians/sec)   " << OMEGA_DOT << std::endl;
	std::cout << "IDOT                (radians/sec)   " << IDOT << std::endl;
	std::cout << "Codes on L2 channel                 " << code_L2 << std::endl;
	std::cout << "GPS Week # (to go with TOE)         " << gps_week << std::endl;
	std::cout << "L2 P data flag                      " << L2P << std::endl;
	std::cout << "SV accuracy         (meters)        " << sv_accuracy << std::endl;
	std::cout << "SV health           (MSB only)      " << sv_health << std::endl;
	std::cout << "TGD                 (seconds)       " << tgd << std::endl;
	std::cout << "IODC Issue of Data, Clock           " << iodc << std::endl;

}

// -------------------------------------------------------------------------------
// Calcul de l'anomalie excentrique par la méthode :
// (1) du point fixe ou (2) de Newton-Raphson
// -------------------------------------------------------------------------------
double NavigationSlot::eccentricAnomaly(double meanAnomaly, double eccentricity, int method){

	double epsilon = 1e-12;
	double Ek = meanAnomaly;
	bool convergence = false;

	if (method == FIXED_POINT){
		double save = Ek;
		while(!convergence) {
			Ek = meanAnomaly + eccentricity*sin(Ek);
			convergence = (std::abs(save - Ek) < epsilon);
			save = Ek;
		}
		return Ek;
	}
	if (method == NEWTON_RAPHSON){
		double dEk = 10;
		while(!convergence) {
			dEk = (Ek - eccentricity*sin(Ek) - meanAnomaly) / (1-eccentricity*cos(Ek));
			convergence = (std::abs(dEk) < epsilon);
			Ek = Ek - dEk;
		}
		return Ek;
	}


	std::cout << "ERROR: unknown numerical method for eccentric anomaly computation " << std::endl;
	assert (false);
	return -1;

}

// -------------------------------------------------------------------------------
// Calcul d'une position de satellite à partir d'un slot de données de rinex .nav
// L'argument pseudorange permet de déduire le temps de propagation du signal
// -------------------------------------------------------------------------------
ECEFCoords NavigationSlot::computeSatellitePos(GPSTime time, double pseudorange){

	// Calcul de l'écart au temps de référence
	double tk0 = (time.wt_week-this->time.wt_week)*SECONDS_WEEK;
	tk0 += time.wt_sec-this->toe;
	tk0 += time.ms/1000.0;
	tk0 = tk0 - pseudorange/Utils::C;

	// Calcul des paramètres de base
	double a = this->sqrt_a*this->sqrt_a;
	double e = this->ecc;
	double n = sqrt(Utils::Mue/pow(a, 3)) + this->delta_n;
	double Mk = this->m0 + n*tk0;

	// Calcul de l'anomalie excentrique
	double Ek = eccentricAnomaly(Mk, e,  NEWTON_RAPHSON);

	// Calcul de la position d'orbite
	double rk = a*(1.0-e*cos(Ek));
	double vk = atan2(sqrt(1.0-e*e)*sin(Ek), cos(Ek)-e);
	double pk = vk + this->omega;
	double uk = pk;

	// Inclinaison du plan oribital
	double ik = this->i0 + this->IDOT*tk0;

	// Longitude du noeud ascendant
	double Omegak = this->OMEGA + (this->OMEGA_DOT-Utils::dOMEGAe)*tk0 - Utils::dOMEGAe*this->toe;

	// Corrections harmoniques
	uk = pk + this->cus*sin(2.0*pk) + this->cuc*cos(2.0*pk);
	rk = rk + this->crs*sin(2.0*pk) + this->crc*cos(2.0*pk);
	ik = ik + this->cis*sin(2.0*pk) + this->cic*cos(2.0*pk);

	// Coordonnées dans le plan orbital
	double xk = rk*cos(uk);
	double yk = rk*sin(uk);

	// Rotation dans le repère ECEF
	ECEFCoords xyz;
	xyz.X = xk*cos(Omegak) - yk*sin(Omegak)*cos(ik);
	xyz.Y = xk*sin(Omegak) + yk*cos(Omegak)*cos(ik);
	xyz.Z =                  yk*sin(ik);

	// Rotation dans l'ECI
    xyz.shiftECEFRef(pseudorange/Utils::C);

	return xyz;

}

// -------------------------------------------------------------------------------
// Calcul d'une position de satellite à partir d'un slot de données de rinex .nav
// -------------------------------------------------------------------------------
ECEFCoords NavigationSlot::computeSatellitePos(GPSTime time){
	return computeSatellitePos(time, 0);
}


// -------------------------------------------------------------------------------
// Calcul de l'erreur d'horloge d'un satellite
// L'argument pseudorange permet de déduire le temps de propagation du signal
// -------------------------------------------------------------------------------
double NavigationSlot::computeSatelliteClockError(GPSTime time, double pseudorange){

    // Calcul de l'écart au temps de référence
	double tk0 = (time.wt_week-this->time.wt_week)*SECONDS_WEEK;
	tk0 += time.wt_sec-this->toe;
	tk0 = tk0 - pseudorange/Utils::C;

	// Calcul des paramètres de base
	double c = Utils::C;
	double e = this->ecc;
    double a =  this->sqrt_a*this->sqrt_a;
	double n = sqrt(Utils::Mue/pow(a, 3)) + this->delta_n;
	double Mk = this->m0 + n*tk0;

	// Calcul de l'anomalie excentrique
	double Ek = eccentricAnomaly(Mk, e,  NEWTON_RAPHSON);

    // Calcul de l'écart au temps de référence
	tk0 = (time.wt_week-this->time.wt_week)*SECONDS_WEEK;
	tk0 += time.wt_sec-this->getTime().wt_sec;
	tk0 = tk0 - pseudorange/c;

	// Calcul de la correction relativiste due à l'excentricité
    double dtr = -2.0*sqrt(Utils::Mue)/(c*c)*this->sqrt_a*e*sin(Ek);

    // Calcul de la correction totale
    double error = this->sv_clock_bias;
    error += (this->sv_clock_drift + this->sv_clock_drift_rate*tk0)*tk0;
    error += dtr - this->tgd;

    return error;
}


// -------------------------------------------------------------------------------
// Calcul de l'erreur d'horloge d'un satellite
// -------------------------------------------------------------------------------
double NavigationSlot::computeSatelliteClockError(GPSTime time){
    return computeSatelliteClockError(time, 0);
}

