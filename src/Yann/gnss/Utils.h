#ifndef UTILS_H
#define UTILS_H

#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <vector>

// ----------------------------------------------------
// Paramètres globaux du programme
// ----------------------------------------------------
#define GPS_CONST                           true     //
#define GLONASS_CONST                       true     //
#define GALILEO_CONST                      false     //
// ----------------------------------------------------
#define MASK_ELEV_DEG                         10     //
#define MASK_PSR_ERROR                        10     //
#define WEIGHTED_OLS                        true     //
#define PSR_ABERRANT                       1.5e6     //
// ----------------------------------------------------
#define NB_GPS_SATS                           32     //
#define NB_GLONASS_SATS                       32     //
#define NB_QZSS_SATS                           4     //
#define NB_GALILEO_SATS                       36     //
#define NB_BEIDOU_SATS                        32     //
// ----------------------------------------------------
#define COEFF_SECURITY                       1.1     //
#define RINEX_NAV_INTERVAL_SEC              7200     //
// ----------------------------------------------------
#define GLO_COEFF_SECURITY                   1.1     //
#define GLO_RINEX_NAV_INTERVAL_SEC          7200     //
// ----------------------------------------------------
#define SP3_INTERVAL_SEC                     900     //
// ----------------------------------------------------
#define RUNGE_KUTTA_INTEGRATION_STEP          60     //
// ----------------------------------------------------
#define FIXED_POINT                            1     //
#define NEWTON_RAPHSON                         2     //
// ----------------------------------------------------
#define L1_FREQ                        1575.42e6     //
#define L2_FREQ                        1227.60e6     //
// ----------------------------------------------------
#define TIME_DIFF_TOLERANCE_DGPS             900     //
// ----------------------------------------------------
#define SECONDS_WEEK              (3600L*24L*7L)     //
#define TIME_T_BASE_YEAR                    1970     //
#define TIME_T_ORIGIN                 315964800L     //
// ----------------------------------------------------


// ---------------------------------------------------------------
// Classe contenant les fonctions utilitaires
// - Constantes physiques
// - Conversions géodésiques
// - Systèmes de temps
// ---------------------------------------------------------------


class Utils {

	public:

		constexpr static double C = 2.99792458e8;                  // Vitesse de la lumière dans le vide (m/s)
		constexpr static double Mue = 3.986005e14;                 // Constante fondamentale GMt (m^3/s^2)
		constexpr static double dOMEGAe = 7.2921151467e-5;         // Vitesse de rotation de la Terre (rad/s)
		constexpr static double Re = 6378137.0;				       // Demi-grand axe de la Terre (m)
		constexpr static double Fe = 1.0/298.257223563;            // Excentricité de l'ellipsoïde terrestre
		constexpr static double TSREF = 18.697374558;              // Heure sidérale au 01/01/2000 à 12:00:00
		constexpr static double DJS = 24.06570982441908;           // Durée d'un jour solaire en heures sidérales

		constexpr static double SP3_NO_DATA_VAL = 999999.999999;   // Error in sp3 navigation file


		inline static double deg2rad(double degree){return degree*PI/180.0;}
		inline static double rad2deg(double radian){return radian*180.0/PI;}

		static std::string formatNumber(int, std::string);
		static std::string formatNumber(double, std::string);
		static std::string trim(std::string);
		static std::string blank(int);
		static double atof2(std::string);
		static bool prn_equal(std::string, std::string);
		static std::vector<double> parseLine(std::string&, int, int, int);
		static std::string makeStdSatName(std::string);
		static std::vector<std::string> getSupportedSatellites();
		static std::vector<std::string> tokenize(std::string, std::string);

};


#endif
