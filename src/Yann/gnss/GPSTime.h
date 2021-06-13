#ifndef GPSTIME_H
#define GPSTIME_H

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <stdio.h>
#include <string>
#include <time.h>

#include "Utils.h"


// ---------------------------------------------------------------
// Classe permettant de gérer le système de temps GPS
// ---------------------------------------------------------------

class GPSTime {

	public:

		int day = 0;
		int month = 0;
		int year = 0;
		int hour = 0;
		int min = 0;
		int sec = 0;
		int ms = 0;
		int wt_week = 0;
		int wt_sec = 0;

		GPSTime(int year, int month, int day, int hour, int min, int sec, int ms){

			this->year = year;
			this->month = month;
			this->day = day;
			this->hour = hour;
			this->min = min;
			this->sec = sec;
			this->ms = ms;

			this->wt_week = ((int)(this->convertToAbsTime())-TIME_T_ORIGIN)/SECONDS_WEEK;
			this->wt_sec = ((int)(this->convertToAbsTime())-TIME_T_ORIGIN)%SECONDS_WEEK;

		}
	
		GPSTime(int year, int month, int day, int hour, int min, int sec){
			new (this) GPSTime(year, month, day, hour, min, sec, 0);
		}

		GPSTime(long seconds=0){

			time_t t = (long)seconds;
			struct tm * ptm = gmtime(&t);

			new (this) GPSTime(1900+ptm->tm_year,ptm->tm_mon+1,ptm->tm_mday,ptm->tm_hour,ptm->tm_min,ptm->tm_sec);

		}

		GPSTime(int week, int sec){

			long t = (long)week*SECONDS_WEEK + TIME_T_ORIGIN + (long)((sec>0.0)?sec+0.5:sec-0.5);

			new (this) GPSTime(t);

		}
	
	
		// Lecture d'une chaîne de caractères
        // Format : "dd*MM*yyyy*hh*mm*ss
        GPSTime(std::string time) {

            int day = std::stoi(time.substr(0,2));
            int mon = std::stoi(time.substr(3,2));
            int yea = std::stoi(time.substr(6,4));
            int hou = std::stoi(time.substr(11,2));
            int min = std::stoi(time.substr(14,2));
            int sec = std::stoi(time.substr(17,2));

            new (this)  GPSTime(yea, mon, day, hou, min, sec);

        }



		double convertToAbsTime();

		// Comparaisons de dates
		bool operator< (GPSTime t){return this->convertToAbsTime() < t.convertToAbsTime();}
		bool operator> (GPSTime t){return this->convertToAbsTime() > t.convertToAbsTime();}
		bool operator== (GPSTime t){return this->convertToAbsTime() == t.convertToAbsTime();}
		bool operator<= (GPSTime t){return this->convertToAbsTime() <= t.convertToAbsTime();}
		bool operator>= (GPSTime t){return this->convertToAbsTime() >= t.convertToAbsTime();}

		// Différences entre temps en secondes
		double operator- (GPSTime t){return this->convertToAbsTime() - t.convertToAbsTime();}

		// Conversion en heure sidérale (à greenwich)
        double gast();
	
		// Ajout de secondes
        GPSTime addSeconds(int sec){
			GPSTime tps(this->convertToAbsTime() + sec); 
			tps.ms = this->ms;
			return tps;
		}

		// Conversion en texte
		std::string to_string();
		std::string to_formal_string();
		std::string to_complete_string();

};

// toString override
inline std::ostream & operator<<(std::ostream & Str, GPSTime tps) {
	Str << tps.to_string();
	return Str;
}

#endif
