#include "NavigationDataSet.h"

// -------------------------------------------------------------------------------
// Test générique pour savoir si le jeu de données contient un satellite prn
// -------------------------------------------------------------------------------
bool NavigationDataSet::hasEphemeris(std::string prn, GPSTime time){

    bool check = false;

    if (prn.substr(0,1) == "G"){
        if (this->gps_loaded) check = check || this->GPS_BRDC_NAV_DATA.hasEphemeris(prn, time);
        if (this->gps_precise_sp3) check = check || this->GPS_SP3_NAV_DATA.hasEphemeris(prn);
    }

    if (prn.substr(0,1) == "R"){
        if (this->glo_loaded) check = check || this->GLO_BRDC_NAV_DATA.hasEphemeris(prn, time);
        if (this->glo_precise_sp3) check = check || this->GLO_SP3_NAV_DATA.hasEphemeris(prn);
    }


    if (prn.substr(0,1) == "E"){
        if (this->gal_loaded) check = check || this->GAL_BRDC_NAV_DATA.hasEphemeris(prn, time);
        if (this->gal_precise_sp3) check = check || this->GAL_SP3_NAV_DATA.hasEphemeris(prn);
    }

    return check;

}

// -------------------------------------------------------------------------------
// Calcul générique de l'erreur d'horloge (en s) du satellite prn
// -------------------------------------------------------------------------------
double NavigationDataSet::computeSatelliteClockError(std::string prn_code, GPSTime time, double pseudorange){

    // GPS
    if (prn_code.substr(0,1) == "G"){
        if (this->gps_precise_sp3){
            if (this->GPS_SP3_NAV_DATA.hasEphemeris(prn_code)){
                return this->GPS_SP3_NAV_DATA.computeSatelliteClockError(prn_code, time, pseudorange);
            }
        }
        if (this->gps_loaded){
            if (this->GPS_BRDC_NAV_DATA.hasEphemeris(prn_code, time)){
                return this->GPS_BRDC_NAV_DATA.computeSatelliteClockError(prn_code, time, pseudorange);
            }
        }
    }

     // GLONASS
    if (prn_code.substr(0,1) == "R"){
        if (this->glo_precise_sp3){
         if (this->GLO_SP3_NAV_DATA.hasEphemeris(prn_code)){
                return this->GLO_SP3_NAV_DATA.computeSatelliteClockError(prn_code, time, pseudorange);
            }
        }
        if (this->glo_loaded){
            if (this->GLO_BRDC_NAV_DATA.hasEphemeris(prn_code, time)){
                return this->GLO_BRDC_NAV_DATA.computeSatelliteClockError(prn_code, time, pseudorange);
            }
        }
    }

     // GALILEO
    if (prn_code.substr(0,1) == "E"){
        if (gal_precise_sp3){
            if (this->GAL_SP3_NAV_DATA.hasEphemeris(prn_code)){
                return this->GAL_SP3_NAV_DATA.computeSatelliteClockError(prn_code, time, pseudorange);
            }
        }
        if (gal_loaded){
        if (this->GAL_BRDC_NAV_DATA.hasEphemeris(prn_code, time)){
                return this->GAL_BRDC_NAV_DATA.computeSatelliteClockError(prn_code, time, pseudorange);
            }
        }
    }

    return 1e100;
}


// -------------------------------------------------------------------------------
// Calcul générique de l'erreur d'horloge (en s) du satellite prn
// -------------------------------------------------------------------------------
double NavigationDataSet::computeSatelliteClockError(std::string prn_code, GPSTime time){
	return computeSatelliteClockError(prn_code, time, 0);
}

// -------------------------------------------------------------------------------
//  Calcul générique de la position ECEF du satellite prn
// -------------------------------------------------------------------------------
ECEFCoords NavigationDataSet::computeSatellitePos(std::string prn_code, GPSTime time, double pseudorange){


    // GPS
    if (prn_code.substr(0,1) == "G"){
        if (this->gps_precise_sp3){
            if (this->GPS_SP3_NAV_DATA.hasEphemeris(prn_code)){
                return this->GPS_SP3_NAV_DATA.computeSatellitePos(prn_code, time, pseudorange);
            }
        }
        if (this->gps_loaded){
            if (this->GPS_BRDC_NAV_DATA.hasEphemeris(prn_code, time)){
                return this->GPS_BRDC_NAV_DATA.computeSatellitePos(prn_code, time, pseudorange);
            }
        }
    }

     // GLONASS
    if (prn_code.substr(0,1) == "R"){
        if (this->glo_precise_sp3){
         if (this->GLO_SP3_NAV_DATA.hasEphemeris(prn_code)){
                return this->GLO_SP3_NAV_DATA.computeSatellitePos(prn_code, time, pseudorange);
            }
        }
        if (this->glo_loaded){
            if (this->GLO_BRDC_NAV_DATA.hasEphemeris(prn_code, time)){
                return this->GLO_BRDC_NAV_DATA.computeSatellitePos(prn_code, time, pseudorange);
            }
        }
    }

     // GALILEO
    if (prn_code.substr(0,1) == "E"){
        if (gal_precise_sp3){
            if (this->GAL_SP3_NAV_DATA.hasEphemeris(prn_code)){
                return this->GAL_SP3_NAV_DATA.computeSatellitePos(prn_code, time, pseudorange);
            }
        }
        if (gal_loaded){
        if (this->GAL_BRDC_NAV_DATA.hasEphemeris(prn_code, time)){
                return this->GAL_BRDC_NAV_DATA.computeSatellitePos(prn_code, time, pseudorange);
            }
        }
    }
    ECEFCoords pos;
    return pos;

}

// -------------------------------------------------------------------------------
//  Calcul générique de la position ECEF du satellite prn
// -------------------------------------------------------------------------------
ECEFCoords NavigationDataSet::computeSatellitePos(std::string prn, GPSTime time){
	return this->computeSatellitePos(prn, time, 0);
}
