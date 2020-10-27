#ifndef GLONAVIGATIONSLOT_H
#define GLONAVIGATIONSLOT_H

#include "Utils.h"
#include "GPSTime.h"
#include "ECEFCoords.h"

class GloNavigationSlot{

    private:

        int prn;
        GPSTime time;

        double x;
        double y;
        double z;

        double xdot;
        double ydot;
        double zdot;

        double xacc;
        double yacc;
        double zacc;

        double health;
        double freq_nb;
        double age;

        double SvClockBias;
        double SvRelFreqBias;
        double messageFrameTime;

        static std::vector<double> runge_kutta_4(ECEFCoords, ECEFCoords, ECEFCoords, double, double);

    public:

        // Accesseurs
		int getPRN(){return this->prn;}
		GPSTime& getTime(){return this->time;}
		double getX(){return this->x;}
		double getY(){return this->y;}
		double getZ(){return this->z;}
        double getXdot(){return this->xdot;}
		double getYdot(){return this->ydot;}
		double getZdot(){return this->zdot;}
        double getXacc(){return this->xacc;}
		double getYacc(){return this->yacc;}
		double getZacc(){return this->zacc;}
        double getHealth(){return this->health;}
        double getFreqNb(){return this->freq_nb;}
        double getAge(){return this->age;}
        double getSvClockBias(){return this->SvClockBias;}
        double getSvRelFreqBias(){return this->SvRelFreqBias;}
        double getMessageFrameTime(){return this->messageFrameTime;}


		// Mutateurs
		void setPRN(int prn){this->prn=prn;}
		void setTime(GPSTime time){this->time=time;}
        void setX(double x){this->x=x;}
		void setY(double y){this->y=y;}
		void setZ(double z){this->z=z;}
        void setXdot(double xdot){this->xdot=xdot;}
		void setYdot(double ydot){this->ydot=ydot;}
		void setZdot(double zdot){this->zdot=zdot;}
        void setXacc(double xacc){this->xacc=xacc;}
		void setYacc(double yacc){this->yacc=yacc;}
		void setZacc(double zacc){this->zacc=zacc;}
        void setHealth(double health){this->health=health;}
        void setFreqNb(double freq_nb){this->freq_nb=freq_nb;}
        void setAge(double age){this->age=age;}
        void setSvClockBias(double clockBias){this->SvClockBias=clockBias;}
        void setSvRelFreqBias(double freqBias){this->SvRelFreqBias=freqBias;}
        void setMessageFrameTime(double time){this->messageFrameTime=time;}


		// Méthodes
		ECEFCoords computeSatellitePos(GPSTime);
		ECEFCoords computeSatellitePos(GPSTime, double);
		double computeSatelliteClockError(GPSTime);
		double computeSatelliteClockError(GPSTime, double);
		void fillLine(std::vector<double>, int);
		void print();



};

#endif // GLONAVIGATIONSLOT_H
