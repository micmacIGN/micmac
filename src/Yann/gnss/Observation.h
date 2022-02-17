#ifndef OBSERVATION_H
#define OBSERVATION_H

#include <string>
#include <vector>

#include "Utils.h"
#include "GPSTime.h"

// ---------------------------------------------------------------
// Classe contenant une mesure GPS
// ---------------------------------------------------------------
class Observation {
	
	// Metadata
	GPSTime timestamp;
	std::string satellite_name="";
	
	// Contents
	double c1 = -1;
	double l1 = -1;
	double l2 = -1;
	double p1 = -1;
	double p2 = -1;
	double d1 = -1;
	double d2 = -1;
		
	bool c1_valid = false;
	bool l1_valid = false;
	bool l2_valid = false;
	bool p1_valid = false;
	bool p2_valid = false;
	bool d1_valid = false;
	bool d2_valid = false;
	
	// Loss of lock
	int c1_lli = -1;
	int l1_lli = -1;
	int l2_lli = -1;
	int p1_lli = -1;
	int p2_lli = -1;
	int d1_lli = -1;
	int d2_lli = -1;
	
	// Signal strenth
	int c1_sgs = -1;
	int l1_sgs = -1;
	int l2_sgs = -1;
	int p1_sgs = -1;
	int p2_sgs = -1;
	int d1_sgs = -1;
	int d2_sgs = -1;
	
	
	public:
	
		Observation(std::string satellite_name=""){
			this->satellite_name = satellite_name;
		}
	
		// Accesseurs
		double getC1(){return this->c1;}
		double getL1(){return this->l1;}
		double getL2(){return this->l2;}
		double getP1(){return this->p1;}
		double getP2(){return this->p2;}
		double getD1(){return this->d1;}
		double getD2(){return this->d2;}
		
		double getC1_LLI(){return this->c1_lli;}
		double getL1_LLI(){return this->l1_lli;}
		double getL2_LLI(){return this->l2_lli;}
		double getP1_LLI(){return this->p1_lli;}
		double getP2_LLI(){return this->p2_lli;}
		double getD1_LLI(){return this->d1_lli;}
		double getD2_LLI(){return this->d2_lli;}
		
		double getC1_SS(){return this->c1_sgs;}
		double getL1_SS(){return this->l1_sgs;}
		double getL2_SS(){return this->l2_sgs;}
		double getP1_SS(){return this->p1_sgs;}
		double getP2_SS(){return this->p2_sgs;}
		double getD1_SS(){return this->d1_sgs;}
		double getD2_SS(){return this->d2_sgs;}
		
		
		double getChannel(std::string);
		int getChannel_LLI(std::string);
		int getChannel_SGS(std::string);
	
		bool hasC1(){return this->c1_valid;}
		bool hasL1(){return this->l1_valid;}
		bool hasL2(){return this->l2_valid;}
		bool hasP1(){return this->p1_valid;}
		bool hasP2(){return this->p2_valid;}
		bool hasD1(){return this->d1_valid;}
		bool hasD2(){return this->d2_valid;}
		bool hasChannel(std::string);
		
	
		GPSTime getTimestamp(){return this->timestamp;}
		std::string getSatName(){return this->satellite_name;}
		
		// Mutateurs
		void setC1(double c1_obs){this->c1=c1_obs; c1_valid=true;}
		void setL1(double l1_obs){this->l1=l1_obs; l1_valid=true;}
		void setL2(double l2_obs){this->l2=l2_obs; l2_valid=true;}
		void setP1(double p1_obs){this->p1=p1_obs; p1_valid=true;}
		void setP2(double p2_obs){this->p2=p2_obs; p2_valid=true;}
		void setD1(double d1_obs){this->d1=d1_obs; d1_valid=true;}
		void setD2(double d2_obs){this->d2=d2_obs; d2_valid=true;}
		
		void setC1_LLI(double c1_obs){this->c1_lli=c1_obs;}
		void setL1_LLI(double l1_obs){this->l1_lli=l1_obs;}
		void setL2_LLI(double l2_obs){this->l2_lli=l2_obs;}
		void setP1_LLI(double p1_obs){this->p1_lli=p1_obs;}
		void setP2_LLI(double p2_obs){this->p2_lli=p2_obs;}
		void setD1_LLI(double d1_obs){this->d1_lli=d1_obs;}
		void setD2_LLI(double d2_obs){this->d2_lli=d2_obs;}
		
		void setC1_SGS(double c1_obs){this->c1_sgs=c1_obs;}
		void setL1_SGS(double l1_obs){this->l1_sgs=l1_obs;}
		void setL2_SGS(double l2_obs){this->l2_sgs=l2_obs;}
		void setP1_SGS(double p1_obs){this->p1_sgs=p1_obs;}
		void setP2_SGS(double p2_obs){this->p2_sgs=p2_obs;}
		void setD1_SGS(double d1_obs){this->d1_sgs=d1_obs;}
		void setD2_SGS(double d2_obs){this->d2_sgs=d2_obs;}
	
		void setTimestamp(GPSTime timestamp){this->timestamp = timestamp;}
		void setSatName(std::string satellite_name){this->satellite_name=satellite_name;}

		// Méthodes
		void setChannel(std::string, double);
		void setChannel(std::string, double, int, int);
		void removeChannel(std::string);

};

// toString override
inline std::ostream & operator<<(std::ostream & Str, Observation obs) {
	Str << "[" << obs.getSatName() << "]  " << obs.getTimestamp() << "  "; 
	Str << "C1=" << Utils::formatNumber(obs.getC1(),"%7.3f") << "  ";
	Str << "D1=" << Utils::formatNumber(obs.getD1(),"%7.3f") << " ";
	Str << "D2=" << Utils::formatNumber(obs.getD2(),"%7.3f") << " ";
	Str << std::endl;
	Str << "L1=" << Utils::formatNumber(obs.getL1(),"%7.3f") << " ";
	Str << "L2=" << Utils::formatNumber(obs.getL2(),"%7.3f") << "  ";
	Str << "P1=" << Utils::formatNumber(obs.getP1(),"%7.3f") << " ";
	Str << "P2=" << Utils::formatNumber(obs.getP2(),"%7.3f") << " ";
	return Str;
}

#endif