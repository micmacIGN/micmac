#ifndef NAVIGATIONSLOT_H
#define NAVIGATIONSLOT_H

#include <string>
#include <vector>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <stdio.h>
#include <cmath>

#include "Utils.h"
#include "ECEFCoords.h"
#include "GPSTime.h"

// ---------------------------------------------------------------
// Classe contenant un paquet de données de navigation
// ---------------------------------------------------------------
class NavigationSlot {

	int prn;
	double sv_clock_bias, sv_clock_drift, sv_clock_drift_rate;
	double iode, crs, delta_n, m0;
	double cuc, ecc, cus, sqrt_a;
	double toe, cic, OMEGA, cis;
	double i0, crc, omega, OMEGA_DOT;
	double IDOT, code_L2, gps_week, L2P;
	double sv_accuracy, sv_health, tgd, iodc;

	GPSTime time;

	public:

		// Accesseurs
		int getPRN(){return this->prn;}
		GPSTime& getTime(){return this->time;}

		double getSvClockBias() {return this->sv_clock_bias;};
		double getSvClockDrift() {return this->sv_clock_drift;};
		double getSvClockDriftRate() {return this->sv_clock_drift_rate;};

		double getIODE() {return this->iode;};
		double getCrs() {return this->crs;};
		double getDeltaN() {return this->delta_n;};
		double getM0() {return this->m0;};

		double getCuc() {return this->cuc;};
		double getEcc() {return this->ecc;};
		double getCus() {return this->cus;};
		double getSqrtA() {return this->sqrt_a;};

		double getToe() {return this->toe;};
		double getCic() {return this->cic;};
		double getOMEGA() {return this->OMEGA;};
		double getCis() {return this->cis;};

		double getI0() {return this->i0;};
		double getCrc() {return this->crc;};
		double getOmega() {return this->omega;};
		double getOMEGADOT() {return this->OMEGA_DOT;};

		double getIDOT() {return this->IDOT;};
		double getCodeL2() {return this->code_L2;};
		double getGpsWeek() {return this->gps_week;};
		double getL2P() {return this->L2P;};

		double getSVAccuracy() {return this->sv_accuracy;};
		double getSVHealth() {return this->sv_health;};
		double getTgd() {return this->tgd;};
		double getIODC() {return this->iodc;};

		// Mutateurs
		void setPRN(int prn){this->prn=prn;}
		void setTime(GPSTime time){this->time=time;}

		void setSvClockBias(double sv_clock_bias) {this->sv_clock_bias=sv_clock_bias;};
		void setSvClockDrift(double sv_clock_drift) {this->sv_clock_drift=sv_clock_drift;};
		void setSvClockDriftRate(double sv_clock_drift_rate) {this->sv_clock_drift_rate=sv_clock_drift_rate;};

		void setIODE(double iode) {this->iode=iode;};
		void setCrs(double crs) {this->crs=crs;};
		void setDeltaN(double delta_n) {this->delta_n=delta_n;};
		void setM0(double m0) {this->m0=m0;};

		void setCuc(double cuc) {this->cuc=cuc;};
		void setEcc(double ecc) {this->ecc=ecc;};
		void setCus(double cus) {this->cus=cus;};
		void setSqrtA(double sqrt_a) {this->sqrt_a=sqrt_a;};

		void setToe(double toe) {this->toe=toe;};
		void setCic(double cic) {this->cic=cic;};
		void setOMEGA(double OMEGA) {this->OMEGA=OMEGA;};
		void setCis(double cis) {this->cis=cis;};

		void setI0(double i0) {this->i0=i0;};
		void setCrc(double crc) {this->crc=crc;};
		void setOmega(double omega) {this->omega=omega;};
		void setOMEGADOT(double OMEGA_DOT) {this->OMEGA_DOT=OMEGA_DOT;};

		void setIDOT(double IDOT) {this->IDOT=IDOT;};
		void setCodeL2(double code_L2) {this->code_L2=code_L2;};
		void setGpsWeek(double gps_week) {this->gps_week=gps_week;};
		void setL2P(double L2P) {this->L2P=L2P;};

		void setSVAccuracy(double sv_accuracy) {this->sv_accuracy=sv_accuracy;};
		void setSVHealth(double sv_health) {this->sv_health=sv_health;};
		void setTgd(double tgd) {this->tgd=tgd;};
		void setIODC(double iodc) {this->iodc=iodc;};

		// Méthodes
		ECEFCoords computeSatellitePos(GPSTime);
		ECEFCoords computeSatellitePos(GPSTime, double);
		double computeSatelliteClockError(GPSTime);
		double computeSatelliteClockError(GPSTime, double);
		void fillLine(std::vector<double>, int);
		void print();

	private:
		static double eccentricAnomaly(double, double, int);

};
#endif
