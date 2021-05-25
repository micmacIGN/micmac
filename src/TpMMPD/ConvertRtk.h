#include "StdAfx.h"

const double JD2000 = 2451545.0; 	// J2000 in jd
const double J2000 = 946728000.0; 	// J2000 in seconds starting from 1970-01-01T00:00:00
const double MJD2000 = 51544.5; 	// J2000 en mjd
const double GPS0 = 315964800.0; 	// 1980-01-06T00:00:00 in seconds starting from 1970-01-01T00:00:00
const int LeapSecond = 18;			// GPST-UTC=18s

//class
class cRPG_Appli;

//struct
struct PosGPS{
	
	//Positions
    Pt3dr Pos;
    
    //Name or Number
    std::string Name;
    
    //vector of Quality
    int Flag;
    
    //Time expressed in Modified Julian day
    double Time;
    
    //Uncertainty
    Pt3dr Ic;
    
    //Correclation terms of Var-CoVar Matrix
    Pt3dr VarCoVar;
    
    //Number of satellites
    int NS;
    
    //Age of Age
    double Age;
    
    //Ratio Factor
    double Ratio;
};

//struct
struct hmsTime{
	double Year;
	double Month;
	double Day;
	double Hour;
	double Minute;
	double Second;
};

//struct
struct towTime{
	double GpsWeek;
	double Tow; //or wsec
};

class cRPG_Appli
{
     public :
          cRPG_Appli(int argc,char ** argv);
//          double hmsTime2MJD(const hmsTime & Time, const std::string & TimeSys);
//          double towTime2MJD(const towTime & Time, const std::string & TimeSys);
//          void ShowHmsTime(const hmsTime & Time);
//          void ShowTowTime(const towTime & Time);
     private :
		std::string mDir; 			
		std::string mFile;
		std::string mOut;
		std::string mStrChSys;
};

template <typename T> string NumberToString(T Number)
{
	ostringstream ss;
    ss << Number;
    return ss.str();
}

double hmsTime2MJD(const hmsTime & Time, const std::string & TimeSys);
double towTime2MJD(const towTime & Time, const std::string & TimeSys);
hmsTime ElDate2hmsTime(const cElDate & aDate);
void ShowHmsTime(const hmsTime & Time);
void ShowTowTime(const towTime & Time);
