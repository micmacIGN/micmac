#include "GloNavigationSlot.h"

void GloNavigationSlot::fillLine(std::vector<double> val, int line_number){

    // +--------------------+------------------------------------------+------------+
    // |                    | - SV clock bias (sec)             (-TauN)|   D19.12,  |
    // |                    | - SV relative frequency bias    (+GammaN)|   D19.12,  |
    // |                    | - message frame time                 (tk)|   D19.12   |
    // +--------------------+------------------------------------------+------------+
    // | BROADCAST ORBIT - 1| - Satellite position X      (km)         | 3X,4D19.12 |
    // |                    | -           velocity X dot  (km/sec)     |            |
    // |                    | -           X acceleration  (km/sec2)    |            |
    // |                    | -           health (0=OK)            (Bn)|            |
    // +--------------------+------------------------------------------+------------+
    // | BROADCAST ORBIT - 2| - Satellite position Y      (km)         | 3X,4D19.12 |
    // |                    | -           velocity Y dot  (km/sec)     |            |
    // |                    | -           Y acceleration  (km/sec2)    |            |
    // |                    | -           frequency number (1-24)      |            |
    // +--------------------+------------------------------------------+------------+
    // | BROADCAST ORBIT - 3| - Satellite position Z      (km)         | 3X,4D19.12 |
    // |                    | -           velocity Z dot  (km/sec)     |            |
    // |                    | -           Z acceleration  (km/sec2)    |            |
    // |                    | - Age of oper. information  (days)   (E) |            |
    // +--------------------+------------------------------------------+------------+

	switch (line_number){
		case 1:
            this->setSvClockBias(-val[0]);
            this->setSvRelFreqBias(val[1]);
            this->setMessageFrameTime(val[2]);
			break;
		case 2:
            this->setX(val[0]);
            this->setXdot(val[1]);
            this->setXacc(val[2]);
            this->setHealth(val[3]);
			break;
		case 3:
            this->setY(val[0]);
            this->setYdot(val[1]);
            this->setYacc(val[2]);
            this->setFreqNb(val[3]);
			break;
		case 4:
            this->setZ(val[0]);
            this->setZdot(val[1]);
            this->setZacc(val[2]);
            this->setAge(val[3]);
			break;
		default:
			break;
	}
}

void GloNavigationSlot::print(){

	std::cout << "------------------------------------------------------" << std::endl;
	std::cout << "PRN R" << Utils::formatNumber(this->prn, "%02d");
	std::cout << ": " << this->getTime() << std::endl;
	std::cout << "------------------------------------------------------" << std::endl;


	std::cout << "X                       (km)      " << Utils::formatNumber(this->x, "% 19.12f")                  << std::endl;
	std::cout << "Y                       (km)      " << Utils::formatNumber(this->y, "% 19.12f")                  << std::endl;
	std::cout << "Z                       (km)      " << Utils::formatNumber(this->z, "% 19.12f")                  << std::endl;
	std::cout << "dX/dt                 (km/s)      " << Utils::formatNumber(this->xdot, "% 19.12f")               << std::endl;
	std::cout << "dY/dt                 (km/s)      " << Utils::formatNumber(this->ydot, "% 19.12f")               << std::endl;
	std::cout << "dZ/dt                 (km/s)      " << Utils::formatNumber(this->zdot, "% 19.12f")               << std::endl;
	std::cout << "d2X/dt2              (km/s2)      " << Utils::formatNumber(this->xacc, "% 19.12f")               << std::endl;
	std::cout << "d2Y/dt2              (km/s2)      " << Utils::formatNumber(this->yacc, "% 19.12f")               << std::endl;
	std::cout << "d2Z/dt2              (km/s2)      " << Utils::formatNumber(this->zacc, "% 19.12f")               << std::endl;
    std::cout << "SV clock bias         (tauN)      " << Utils::formatNumber(this->SvClockBias, "% 19.12f")        << std::endl;
	std::cout << "SV freq bias        (gammaN)      " << Utils::formatNumber(this->SvRelFreqBias, "% 19.12f")      << std::endl;
	std::cout << "Message time            (tk)      " << Utils::formatNumber(this->messageFrameTime, "% 19.12f")   << std::endl;
	std::cout << "Health                (0=OK)      " << Utils::formatNumber(this->health, "% 19.12f")             << std::endl;
	std::cout << "Age                   (days)      " << Utils::formatNumber(this->age, "% 19.12f")                << std::endl;
	std::cout << "Frequency number      (1-24)      " << Utils::formatNumber(this->freq_nb, "% 19.12f")            << std::endl;

	std::cout << "------------------------------------------------------" << std::endl;

}


// -------------------------------------------------------------------------------
// Intégration numérique de la position d'un satellite par Runge-Kutta d'ordre 4
// Sortie : vecteur cinématique [x,y,z,vx,vy,vz]
// -------------------------------------------------------------------------------
std::vector<double> GloNavigationSlot::runge_kutta_4(ECEFCoords pos_ini, ECEFCoords v_ini, ECEFCoords a_ms, double t0, double h){

    double a = 6378136.0;
	double mu = 3.9860044e14;
	double C20 = -1.08263e-3;

	double r;
	double mub, xb, yb, zb, rob;

    double k1x, k1y, k1z;
	double k2x, k2y, k2z;
	double k3x, k3y, k3z;
	double k4x, k4y, k4z;

	double k1vx, k1vy, k1vz;
	double k2vx, k2vy, k2vz;
	double k3vx, k3vy, k3vz;
    double k4vx, k4vy, k4vz;

	double x,y,z,vx,vy,vz;
	double kx,ky,kz,kvx,kvy,kvz;

	double ax_eci = a_ms.X;
	double ay_eci = a_ms.Y;
	double az_eci = a_ms.Z;

    int T = t0/h;
    double remain = t0-T*h;

    // Intégration
    for (int i=0; i<=T; i++){

        if (i == T) h = remain;
        if (h == 0) break;

        //----------------------------------------------
        // Coeff k1 : pente en 0
        //----------------------------------------------
        x=pos_ini.X;
        y=pos_ini.Y;
        z=pos_ini.Z;
        vx=v_ini.X;
        vy=v_ini.Y;
        vz=v_ini.Z;

        r = sqrt(x*x + y*y + z*z);
        mub = mu/(r*r); xb = x/r; yb = y/r; zb = z/r; rob = a/r;
        k1x = vx;
        k1y = vy;
        k1z = vz;
        k1vx = -mub*xb + 1.5*C20*mub*xb*rob*rob*(1-5*zb*zb) + ax_eci;
        k1vy = -mub*yb + 1.5*C20*mub*yb*rob*rob*(1-5*zb*zb) + ay_eci;
        k1vz = -mub*zb + 1.5*C20*mub*zb*rob*rob*(3-5*zb*zb) + az_eci;

        //----------------------------------------------
        // Coeff k2 : pente en h/2 calculée avec k1
        //----------------------------------------------
        x = pos_ini.X + k1x*h/2;
        y = pos_ini.Y + k1y*h/2;
        z = pos_ini.Z + k1z*h/2;
        vx = v_ini.X + k1vx*h/2;
        vy = v_ini.Y + k1vy*h/2;
        vz = v_ini.Z + k1vz*h/2;

        r = sqrt(x*x + y*y + z*z);
        mub = mu/(r*r); xb = x/r; yb = y/r; zb = z/r; rob = a/r;
        k2x = vx;
        k2y = vy;
        k2z = vz;
        k2vx = -mub*xb + 1.5*C20*mub*xb*rob*rob*(1-5*zb*zb) + ax_eci;
        k2vy = -mub*yb + 1.5*C20*mub*yb*rob*rob*(1-5*zb*zb) + ay_eci;
        k2vz = -mub*zb + 1.5*C20*mub*zb*rob*rob*(3-5*zb*zb) + az_eci;


        //----------------------------------------------
        // Coeff k3 : pente en h/2 recalculée avec k2
        //----------------------------------------------
        x = pos_ini.X + k2x*h/2;
        y = pos_ini.Y + k2y*h/2;
        z = pos_ini.Z + k2z*h/2;
        vx = v_ini.X + k2vx*h/2;
        vy = v_ini.Y + k2vy*h/2;
        vz = v_ini.Z + k2vz*h/2;

        r = sqrt(x*x + y*y + z*z);
        mub = mu/(r*r); xb = x/r; yb = y/r; zb = z/r; rob = a/r;
        k3x = vx;
        k3y = vy;
        k3z = vz;
        k3vx = -mub*xb + 1.5*C20*mub*xb*rob*rob*(1-5*zb*zb) + ax_eci;
        k3vy = -mub*yb + 1.5*C20*mub*yb*rob*rob*(1-5*zb*zb) + ay_eci;
        k3vz = -mub*zb + 1.5*C20*mub*zb*rob*rob*(3-5*zb*zb) + az_eci;


        //----------------------------------------------
        // Coeff k4 : pente en h calculée avec k3
        //----------------------------------------------
        x = pos_ini.X + k3x*h;
        y = pos_ini.Y + k3y*h;
        z = pos_ini.Z + k3z*h;
        vx = v_ini.X + k3vx*h;
        vy = v_ini.Y + k3vy*h;
        vz = v_ini.Z + k3vz*h;

        r = sqrt(x*x + y*y + z*z);
        mub = mu/(r*r); xb = x/r; yb = y/r; zb = z/r; rob = a/r;
        k4x = vx;
        k4y = vy;
        k4z = vz;
        k4vx = -mub*xb + 1.5*C20*mub*xb*rob*rob*(1-5*zb*zb) + ax_eci;
        k4vy = -mub*yb + 1.5*C20*mub*yb*rob*rob*(1-5*zb*zb) + ay_eci;
        k4vz = -mub*zb + 1.5*C20*mub*zb*rob*rob*(3-5*zb*zb) + az_eci;

        //----------------------------------------------
        // Pente moyenne : (k1+2*k2+2*k3+k4)/6
        //----------------------------------------------
        kx = (k1x + 2*k2x + 2*k3x + k4x)/6.0;
        ky = (k1y + 2*k2y + 2*k3y + k4y)/6.0;
        kz = (k1z + 2*k2z + 2*k3z + k4z)/6.0;
        kvx = (k1vx + 2*k2vx + 2*k3vx + k4vx)/6.0;
        kvy = (k1vy + 2*k2vy + 2*k3vy + k4vy)/6.0;
        kvz = (k1vz + 2*k2vz + 2*k3vz + k4vz)/6.0;

        //----------------------------------------------
        // Estimation en h
        //----------------------------------------------
        pos_ini.X += kx*h;
        pos_ini.Y += ky*h;
        pos_ini.Z += kz*h;
        v_ini.X += kvx*h;
        v_ini.Y += kvy*h;
        v_ini.Z += kvz*h;

    }

    std::vector<double> output;
    output.push_back(pos_ini.X); output.push_back(pos_ini.Y); output.push_back(pos_ini.Z);
    output.push_back(v_ini.X); output.push_back(v_ini.Y); output.push_back(v_ini.Z);

    return output;

}

// -------------------------------------------------------------------------------
// Calcul d'une position de satellite à partir d'un slot de données de rinex .nav
// L'argument pseudorange permet de déduire le temps de propagation du signal
// -------------------------------------------------------------------------------
ECEFCoords GloNavigationSlot::computeSatellitePos(GPSTime time, double pseudorange){

    // Calcul du temps orbital
    double t0 = time - this->getTime() - pseudorange/Utils::C;

    // Calcul heure sidérale
    double TSL0 = this->getTime().gast();

    TSL0 = 22.94054032666641;                          // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    // Transformation des positions/vitesses dans le repère ECI
    double theta_g0 = TSL0*PI/12 + Utils::dOMEGAe*900;

    ECEFCoords pxyz_eci(this->x   , this->y   , this->z);
    ECEFCoords vxyz_eci(this->xdot, this->ydot, this->zdot);
    ECEFCoords a_ms_eci(this->xacc, this->yacc, this->zacc);

    // Conversion en m
    pxyz_eci.scalar(1e3);
    vxyz_eci.scalar(1e3);
    a_ms_eci.scalar(1e3);

    pxyz_eci.rotate(-theta_g0);
    vxyz_eci.rotate(-theta_g0);
    a_ms_eci.rotate(-theta_g0);

    vxyz_eci.X -= Utils::dOMEGAe*pxyz_eci.Y;
    vxyz_eci.Y += Utils::dOMEGAe*pxyz_eci.X;

    // Intégration numérique RK4
    std::vector<double> pv = runge_kutta_4(pxyz_eci, vxyz_eci, a_ms_eci, t0, RUNGE_KUTTA_INTEGRATION_STEP);
    ECEFCoords  sat(pv.at(0), pv.at(1), pv.at(2));
    ECEFCoords vsat(pv.at(3), pv.at(4), pv.at(5));

    // Retour en coordonnées ECEF PZ90
    theta_g0 += Utils::dOMEGAe*t0;
    sat.rotate(+theta_g0);

    // Transformation PZ90 -> WGS84
    sat.X += -0.36;
    sat.Y +=  0.08;
    sat.Z +=  0.18;

    // Rotation dans l'ECI
    sat.shiftECEFRef(pseudorange/Utils::C);

	return sat;

}

// -------------------------------------------------------------------------------
// Calcul d'une position de satellite à partir d'un slot de données de rinex .nav
// -------------------------------------------------------------------------------
ECEFCoords GloNavigationSlot::computeSatellitePos(GPSTime time){
	return computeSatellitePos(time, 0);
}


// -------------------------------------------------------------------------------
// Calcul de l'erreur d'horloge d'un satellite GLONASS
// L'argument pseudorange permet de déduire le temps de propagation du signal
// -------------------------------------------------------------------------------
double GloNavigationSlot::computeSatelliteClockError(GPSTime time, double pseudorange){

     // Calcul du temps orbital
    double t0 = time - this->getTime() - pseudorange/Utils::C;

    // Calcul heure sidérale
    double TSL0 = this->getTime().gast();

    TSL0 = 22.94054032666641;                          // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    // Transformation des positions/vitesses dans le repère ECI
    double theta_g0 = TSL0*PI/12 + Utils::dOMEGAe*900;

    ECEFCoords pxyz_eci(this->x   , this->y   , this->z);
    ECEFCoords vxyz_eci(this->xdot, this->ydot, this->zdot);
    ECEFCoords a_ms_eci(this->xacc, this->yacc, this->zacc);

    // Conversion en m
    pxyz_eci.scalar(1e3);
    vxyz_eci.scalar(1e3);
    a_ms_eci.scalar(1e3);

    pxyz_eci.rotate(-theta_g0);
    vxyz_eci.rotate(-theta_g0);
    a_ms_eci.rotate(-theta_g0);

    vxyz_eci.X -= Utils::dOMEGAe*pxyz_eci.Y;
    vxyz_eci.Y += Utils::dOMEGAe*pxyz_eci.X;

    // Intégration numérique RK4
    std::vector<double> pv = runge_kutta_4(pxyz_eci, vxyz_eci, a_ms_eci, t0, RUNGE_KUTTA_INTEGRATION_STEP);
    ECEFCoords  sat(pv.at(0), pv.at(1), pv.at(2));
    ECEFCoords vsat(pv.at(3), pv.at(4), pv.at(5));

    // Calcul de la correction relativiste due à l'excentricité
    double dtr = -2*sat.dot(vsat)/(Utils::C*Utils::C);

    // Calcul de la correction totale
    double error = dtr + this->SvClockBias + this->SvRelFreqBias*t0;

    return -error;

}


// -------------------------------------------------------------------------------
// Calcul de l'erreur d'horloge d'un satellite GLONASS
// -------------------------------------------------------------------------------
double GloNavigationSlot::computeSatelliteClockError(GPSTime time){
    return computeSatelliteClockError(time, 0);
}
