#include "GPSTime.h"

#include <math.h>

double GPSTime::convertToAbsTime(){

    int days_month[] = {31,28,31,30,31,30,31,31,30,31,30,31};

	long days=0L;

	// Nombre de jours depuis le 01/01/1970
	for (int i=TIME_T_BASE_YEAR; i<this->year; i++){
		days += (i%4==0)?366:365;
	}

	// Nombre de jours de l'année courante
	for (int i=1; i<this->month; i++){
		days += days_month[i-1];
		if ((i == 2) && (this->year%4==0)){
			days ++;
		}
	}

	// Nombre de jours du mois courant
	days += this->day-1;

	// Conversion en secondes
	return days*86400 + this->hour*3600 + this->min*60 + this->sec + this->ms/1000.0;

}

// Conversion en heure sidérale (à greenwich)
double GPSTime::gast(){
    GPSTime ref(2000,01,01,12,00,00);
    double TSL0 = (*this - ref)*Utils::DJS + Utils::TSREF;
    double factor = floor(TSL0/24);
    TSL0 = TSL0 - factor*24;
    return TSL0;
}

// Conversion to string
std::string GPSTime::to_string() {
    std::string output = "";
	output += Utils::formatNumber(this->day,"%02d");
	output += "/" + Utils::formatNumber(this->month,"%02d");
	output += "/" + Utils::formatNumber(this->year,"%04d") + " ";
	output += Utils::formatNumber(this->hour,"%02d") + ":";
	output += Utils::formatNumber(this->min,"%02d") + ":";
	output += Utils::formatNumber(this->sec,"%02d");
	return output;
}

// Conversion to string
std::string GPSTime::to_formal_string() {
    std::string output = "";
	output += Utils::formatNumber(this->day,"%02d");
	output += Utils::formatNumber(this->month,"%02d");
	output += Utils::formatNumber(this->year,"%04d");
	output += Utils::formatNumber(this->hour,"%02d");
	output += Utils::formatNumber(this->min,"%02d");
	output += Utils::formatNumber(this->sec,"%02d");
	output += "." + Utils::formatNumber(this->ms,"%03d");
	return output;
}

// Conversion to string with ms
std::string GPSTime::to_complete_string() {
    std::string output = "";
	output += Utils::formatNumber(this->day,"%02d");
	output += "/" + Utils::formatNumber(this->month,"%02d");
	output += "/" + Utils::formatNumber(this->year,"%04d") + " ";
	output += Utils::formatNumber(this->hour,"%02d") + ":";
	output += Utils::formatNumber(this->min,"%02d") + ":";
	output += Utils::formatNumber(this->sec,"%02d") + ".";
	output += Utils::formatNumber(this->ms,"%03d");
	return output;
}
