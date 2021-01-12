#include "AtmosphericModel.h"

AtmosphericModel::AtmosphericModel(NavigationData nav_data){
    this->alpha = nav_data.getIonoAlpha();
    this->beta = nav_data.getIonoBeta();
}

AtmosphericModel::AtmosphericModel(std::vector<double> alpha, std::vector<double> beta){
    this->alpha = alpha;
    this->beta = beta;
}


// ---------------------------------------------------------
// Algorithme de correction du retard troposphérique
// Modèle de Takeyasu SAKAI (GPS programming)
// ---------------------------------------------------------
double AtmosphericModel::tropo_correction(ECEFCoords usr, ECEFCoords sat) {

    // Elévation du satellite
    double el = usr.elevationTo(sat);

    // Altitude du récepteur
    double h = usr.toGeoCoords().height;

    // Calcul de la correction
    double d = 1.0;
    double scale = 1.0/2.3*1e5;
    if (h > 0.0) {
        if (h < scale) {
            d = 2.47 * pow((1-h/scale), 5);
        }else {
            d = 0.0;
        }
    }

    // Facteur d'obliquité
    d /= sin(el) + 0.0121;

    return -d;

}


// ---------------------------------------------------------
// Algorithme de correction du retard troposphérique
// Utilisation du modèle de Saastomonien
// ---------------------------------------------------------
double AtmosphericModel::tropo_correction_saastomonian(ECEFCoords usr, ECEFCoords sat) {

    // Elévation du satellite
    double el = usr.elevationTo(sat);

    double e = Rh*exp(-37.2465+0.213166*T-0.000256908*T*T);
    double delta = 0.002277/cos(PI/2-el)*(P+(1255.0/T+0.05)*e);

    return -delta;

}


// ---------------------------------------------------------
// Algorithme de correction du retard ionosphérique avec le
// modèle de Klobuchar au zénith
// ---------------------------------------------------------
double AtmosphericModel::iono_correction_raw(ECEFCoords usr) {

    // Azimut/élévation du satellite
    double az = 0;
    double el = PI/2;
    double el_sc = el/PI;

    // Coordonnées géodésiques du récepteur
    GeoCoords geo_usr = usr.toGeoCoords();

    // Coordonnées du point d'entrée iono
    double psi = 0.0137/(el_sc+0.11) - 0.022;
    double phi_i0 = geo_usr.latitude/180.0 + psi*cos(az);
    double phi_i = std::max(std::min(phi_i0, 0.416), -0.416);
    double lambda_i = geo_usr.longitude/180.0 + psi*sin(az)/cos(phi_i*PI);
    double phi_m = phi_i + 0.064*cos((lambda_i-1.617)*PI);

    // Heure locale au niveau du point d'entrée
    double tl = 12*3600*lambda_i + this->time.wt_sec;
    while(tl < 0) tl = tl + 86400;
    while(tl >= 86400) tl = tl - 86400;

    // Amplitude et periode du modèle
    double amp = ((this->alpha.at(3)*phi_m + this->alpha.at(2))*phi_m + this->alpha.at(1))*phi_m + this->alpha.at(0);
    double per = ( (this->beta.at(3)*phi_m +  this->beta.at(2))*phi_m +  this->beta.at(1))*phi_m +  this->beta.at(0);
    amp = std::max(amp, 0.0);
    per = std::max(per, 20*3600.0);

    // Variable polynomiale du modèle
    double x = (2*PI*(tl-14*3600))/per;
    while (x > +PI) x = x - 2*PI;
    while (x < -PI) x = x + 2*PI;

    // Calcul du retard
    double t_iono = 5*1e-9;
    if (std::abs(x) < PI/2) {
        t_iono += amp*(1 - pow(x, 2)/2.0 + pow(x, 4)/24.0);
    }

    // Facteur d'obliquité
    t_iono *= 1+16*pow(0.53-el/PI, 3);

    return -t_iono*Utils::C;

}


// ---------------------------------------------------------
// Algorithme de correction du retard ionosphérique avec le
// modèle de Klobuchar
// ---------------------------------------------------------
double AtmosphericModel::iono_correction(ECEFCoords usr, ECEFCoords sat) {

    // Azimut/élévation du satellite
    double az = usr.azimuthTo(sat);
    double el = usr.elevationTo(sat);
    double el_sc = el/PI;

    // Coordonnées géodésiques du récepteur
    GeoCoords geo_usr = usr.toGeoCoords();

    // Coordonnées du point d'entrée iono
    double psi = 0.0137/(el_sc+0.11) - 0.022;
    double phi_i0 = geo_usr.latitude/180.0 + psi*cos(az);
    double phi_i = std::max(std::min(phi_i0, 0.416), -0.416);
    double lambda_i = geo_usr.longitude/180.0 + psi*sin(az)/cos(phi_i*PI);
    double phi_m = phi_i + 0.064*cos((lambda_i-1.617)*PI);

    // Heure locale au niveau du point d'entrée
    double tl = 12*3600*lambda_i + this->time.wt_sec;
    while(tl < 0) tl = tl + 86400;
    while(tl >= 86400) tl = tl - 86400;

    // Amplitude et periode du modèle
    double amp = ((this->alpha.at(3)*phi_m + this->alpha.at(2))*phi_m + this->alpha.at(1))*phi_m + this->alpha.at(0);
    double per = ( (this->beta.at(3)*phi_m +  this->beta.at(2))*phi_m +  this->beta.at(1))*phi_m +  this->beta.at(0);
    amp = std::max(amp, 0.0);
    per = std::max(per, 20*3600.0);

    // Variable polynomiale du modèle
    double x = (2*PI*(tl-14*3600))/per;
    while (x > +PI) x = x - 2*PI;
    while (x < -PI) x = x + 2*PI;

    // Calcul du retard
    double t_iono = 5*1e-9;
    if (std::abs(x) < PI/2) {
        t_iono += amp*(1 - pow(x, 2)/2.0 + pow(x, 4)/24.0);
    }

    // Facteur d'obliquité
    t_iono *= 1+16*pow(0.53-el/PI, 3);

    return -t_iono*Utils::C;

}

// ---------------------------------------------------------
// Algorithme de correction de tous les retards atmo.
// ---------------------------------------------------------
double AtmosphericModel::all_corrections(ECEFCoords usr, ECEFCoords sat) {
    double trop = this->tropo_correction(usr, sat);
    double iono = this->iono_correction(usr, sat);
    return trop + iono;
}
