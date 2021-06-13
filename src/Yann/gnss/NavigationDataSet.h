#ifndef NAVIGATIONDATASET_H
#define NAVIGATIONDATASET_H

#include <vector>

#include "Utils.h"
#include "SP3Reader.h"
#include "RinexReader.h"
#include "NavigationData.h"
#include "SP3NavigationData.h"

// ---------------------------------------------------------------
// Classe permettant de gérer plusieurs fichiers de navigation
// ---------------------------------------------------------------
class NavigationDataSet {

    private:

        bool gps_precise_sp3 = false;
        bool gal_precise_sp3 = false;
        bool glo_precise_sp3 = false;

        bool gps_loaded = false;
        bool gal_loaded = false;
        bool glo_loaded = false;

        NavigationData GPS_BRDC_NAV_DATA;
        NavigationData GAL_BRDC_NAV_DATA;
        GloNavigationData GLO_BRDC_NAV_DATA;

        SP3NavigationData GPS_SP3_NAV_DATA;
        SP3NavigationData GAL_SP3_NAV_DATA;
        SP3NavigationData GLO_SP3_NAV_DATA;


    public:

        bool hasEphemeris(std::string prn, GPSTime);

        void addGpsPreciseEphemeris(SP3NavigationData sp3)    {this->GPS_SP3_NAV_DATA = sp3; gps_precise_sp3=true;}
        void addGalileoPreciseEphemeris(SP3NavigationData sp3){this->GAL_SP3_NAV_DATA = sp3; gal_precise_sp3=true;}
        void addGlonassPreciseEphemeris(SP3NavigationData sp3){this->GLO_SP3_NAV_DATA = sp3; glo_precise_sp3=true;}

        void loadGpsPreciseEphemeris(std::string sp3_file)    {this->addGpsPreciseEphemeris(SP3Reader::readNavFile(sp3_file));}
        void loadGalileoPreciseEphemeris(std::string sp3_file){this->addGalileoPreciseEphemeris(SP3Reader::readNavFile(sp3_file));}
        void loadGlonassPreciseEphemeris(std::string sp3_file){this->addGlonassPreciseEphemeris(SP3Reader::readNavFile(sp3_file));}

        void addGpsEphemeris(NavigationData nav)       {this->GPS_BRDC_NAV_DATA = nav; gps_loaded=true;}
        void addGalileoEphemeris(NavigationData nav)   {this->GAL_BRDC_NAV_DATA = nav; gal_loaded=true;}
        void addGlonassEphemeris(GloNavigationData nav){this->GLO_BRDC_NAV_DATA = nav; glo_loaded=true;}

        void loadGpsEphemeris(std::string nav_file)    {this->addGpsEphemeris(RinexReader::readNavFile(nav_file));}
        void loadGalileoEphemeris(std::string nav_file){this->addGalileoEphemeris(RinexReader::readNavFile(nav_file));}
        void loadGlonassEphemeris(std::string nav_file){this->addGlonassEphemeris(RinexReader::readGloNavFile(nav_file));}

        double computeSatelliteClockError(std::string PRN, GPSTime);
        double computeSatelliteClockError(std::string, GPSTime, double);
        ECEFCoords computeSatellitePos(std::string, GPSTime, double);
		ECEFCoords computeSatellitePos(std::string, GPSTime);

};

#endif // NAVIGATIONDATASET_H
