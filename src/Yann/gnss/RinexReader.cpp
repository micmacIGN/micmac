#include "RinexReader.h"
#include "ENUCoords.h"
#include "ECEFCoords.h"
#include "ObservationSlot.h"

std::string sep  = "-----------------------------------------------------------------------";
std::string sep2 = "=======================================================================";

// Rinex files specifications : https://www.ngs.noaa.gov/CORS/RINEX-2.txt

// ---------------------------------------------------------------
// Fonction de récupération du tag de commentaire de ligne
// ---------------------------------------------------------------
std::string RinexReader::getComment(std::string &line){
	if (line.length() < 60){
		return "";
	}
	return line.substr(60,line.length()-60);
}


// ---------------------------------------------------------------
// Fonction de lecture d'un fichier de navigation rinex v2 et v3.
// ---------------------------------------------------------------
NavigationData RinexReader::readNavFile(std::string nav_file_path){

    NavigationData navData;

    std::ifstream in(nav_file_path.c_str());

	if (!in) {
		std::cout << "Cannot open navigation input file [" << nav_file_path << "]" << std::endl;
		assert(false);
		return navData;
	}

	std::string line;
	std::getline(in, line);

    in.close();

	double version = std::stof(line.substr(0,9));
	std::string constellation = line.substr(40,1);

	std::cout << sep2 << std::endl;

	if ((version < 2) || (version >= 4)){
		 std::cout << "Error: rinex version " << version << " is not supported" << std::endl;
   		 assert(false);
	}

	std::cout << "Reading rinex (" << Utils::formatNumber(version,"%3.2f") << ") navigation file " << nav_file_path << std::endl;
	std::cout << sep2 << std::endl;

	if ((constellation != " ") || (floor(version) == 3)){
		return RinexReader::readNavFileV3(nav_file_path);
	}else{
		return RinexReader::readNavFileV2(nav_file_path);
	}

	std::cout << std::endl;

    return navData;

}



// ---------------------------------------------------------------
// Fonction de lecture d'un fichier de navigation rinex GPS v2
// https://www.glonass-iac.ru/en/GLONASS/docs/rinex2.htm
// ---------------------------------------------------------------
NavigationData RinexReader::readNavFileV2(std::string nav_file_path){

	NavigationData navData;
	navData.setSourceFilePath(nav_file_path);

	std::ifstream in(nav_file_path.c_str());

	if (!in) {
		std::cout << "Cannot open navigation input file [" << nav_file_path << "]" << std::endl;
		assert(false);
		return navData;
	}

	std::string line;
	std::string comment;

	std::vector<double> ion_alpha;
	std::vector<double> ion_beta;
	std::vector<double> delta_utc;
	std::vector<NavigationSlot> NavSlots;

	int REDUNDANT = 0;

	// -----------------------------------------------------
	// Reading header
	// -----------------------------------------------------

	while (std::getline(in, line)) {

        //  +--------------------+------------------------------------------+------------+
        //  |RINEX VERSION / TYPE| - Format version (2.10)                  | F9.2,11X,  |#
        //  |                    | - File type ('N' = GPS nav mess data)    |   A1,39X   |
        //  +--------------------+------------------------------------------+------------+
        if (getComment(line).rfind("RINEX VERSION / TYPE", 0) == 0){
            if (line.substr(20,1) != "N"){
                std::cout << "Error: file " << nav_file_path << " is not a GPS rinex navigation file" << std::endl;
                assert(false);
            }
        }

		// +--------------------+------------------------------------------+------------+
		// |LEAP SECONDS        | Delta time due to leap seconds           |     I6     |
		// +--------------------+------------------------------------------+------------+
		if (getComment(line).rfind("LEAP SECONDS", 0) == 0){
			navData.setLeapSeconds(std::stoi(line.substr(0,6)));
		}
		// +--------------------+------------------------------------------+------------+
		// |ION ALPHA           | Ionosphere parameters A0-A3 of almanac   |  2X,4D12.4 |
		// |                    | (page 18 of subframe 4)                  |            |
		// +--------------------+------------------------------------------+------------+

		if (getComment(line).rfind("ION ALPHA", 0) == 0){
			navData.setIonoAlpha(Utils::parseLine(line,2,4,12));
		}

		// +--------------------+------------------------------------------+------------+
		// |ION BETA            | Ionosphere parameters B0-B3 of almanac   |  2X,4D12.4 |
		// +--------------------+------------------------------------------+------------+

		if (getComment(line).rfind("ION BETA", 0) == 0){
			navData.setIonoBeta(Utils::parseLine(line,2,4,12));
		}

		// +--------------------+------------------------------------------+------------+
		// |DELTA-UTC: A0,A1,T,W| Almanac parameters to compute time in UTC| 3X,2D19.12,|
		// |                    | (page 18 of subframe 4)                  |     2I9    |
		// |                    | A0,A1: terms of polynomial               |            |
		// |                    | T    : reference time for UTC data       |            |
		// |                    | W    : UTC reference week number         |            |
		// +--------------------+------------------------------------------+------------+

		if (getComment(line).rfind("DELTA-UTC:", 0) == 0){
			navData.setUTCParameters(Utils::parseLine(line,3,2,19));
		}

		// -----------------------------------------------------
		// End of header
		// -----------------------------------------------------
		if (getComment(line).rfind("END OF HEADER", 0) == 0){
			break;
		}

	}

	// -----------------------------------------------------
	// Reading main part of file
	// -----------------------------------------------------

	std::vector<double> line_values;


	while (std::getline(in, line)) {

		if (getComment(line).rfind("COMMENT", 0) == 0){
			continue;
		}

		NavigationSlot slot;

		slot.setPRN(std::stoi(line.substr(0,2)));

		// Year conversion to 4 digits
		int year = std::stoi(line.substr(2+0*3,3));
		year += (year >= 70)?1900:2000;

		GPSTime time   (year,                               // Years
						std::stoi(line.substr(2+1*3,3)),    // Months
						std::stoi(line.substr(2+2*3,3)),    // Days
						std::stoi(line.substr(2+3*3,3)),    // Hours
						std::stoi(line.substr(2+4*3,3)),    // Minutes
						std::stoi(line.substr(2+5*3,3)));   // Seconds

		slot.setTime(time);


		line_values = Utils::parseLine(line,22,3,19);
		slot.fillLine(line_values, 1);

		for (int num_line=2; num_line<8; num_line++){
			std::getline(in, line);
			line_values = Utils::parseLine(line,3,4,19);
			slot.fillLine(line_values, num_line);
		}

		std::getline(in, line);

		// -----------------------------------------------------
		// Contrôle de redondance
		// -----------------------------------------------------

		bool tps_different = true;
		bool prn_different = true;

        for (unsigned i=0; i<navData.getNavigationSlots().size(); i++){
            tps_different = (navData.getNavigationSlots().at(i).getTime() - time != 0);
            prn_different = (navData.getNavigationSlots().at(i).getPRN() != slot.getPRN());
            if ((!tps_different) && (!prn_different)) break;
        }

        if ((tps_different || prn_different)) {
            navData.addNavigationSlot(slot);
        } else{
            REDUNDANT ++;
        }

	}

	in.close();

	// -----------------------------------------------------
	// Data summary
	// -----------------------------------------------------
	std::cout << navData.getNumberOfNavigationSlots() << " data slots loaded with success" << std::endl;
	if (navData.getNumberOfNavigationSlots() > 0){
		std::cout << "Date of first navigation slot:  " << navData.getNavigationSlot(0).getTime() << std::endl;
		std::cout << "Date of last navigation slot :  " << navData.getNavigationSlot(navData.getNumberOfNavigationSlots()-1).getTime() << std::endl;
	}

	std::cout << sep << std::endl;

	int prn = 0;
	for (int i=0; i<8; i++){
		for (int j=0; j<4; j++){
			prn = i*4+j;
            std::cout << "PRN G" << Utils::formatNumber(prn+1, "%02d") << "  ";
			std::cout << Utils::formatNumber(navData.getPRNCount()[prn],"% 3d") << "      ";
		}
		std::cout << std::endl;
	}

	std::vector<int> MISSING_SATS;
	for (int i=0; i<37; i++){
		if (navData.getPRNCount()[i] == 0){
			MISSING_SATS.push_back(i+1);
		}
	}

	if (MISSING_SATS.size() > 0){
        std::cout << sep << std::endl;
		std::cout << "WARNING: MISSING SATELLITES ";
		for (unsigned int i=0; i<MISSING_SATS.size(); i++){
			std::cout << MISSING_SATS.at(i) << " ";
		}
		std::cout << std::endl;
	}

	if (REDUNDANT > 0){
        std::cout << sep << std::endl;
        std::cout << "WARNING: " << REDUNDANT << " REDUNDANT DATA SLOT(S)" << std::endl;
	}

	return navData;

}

// ---------------------------------------------------------------
// Fonction de lecture d'un fichier de navigation rinex GNSS v3
// ftp://igs.org/pub/data/format/rinex303.pdf
// ---------------------------------------------------------------
NavigationData RinexReader::readNavFileV3(std::string nav_file_path){

	NavigationData navData;
	navData.setSourceFilePath(nav_file_path);

	std::ifstream in(nav_file_path.c_str());

	if (!in) {
		std::cout << "Cannot open navigation input file [" << nav_file_path << "]" << std::endl;
		return navData;
	}

	std::string line;
	std::string comment;
	std::string constellation;

	std::vector<NavigationSlot> NavSlots;

	int REDUNDANT = 0;

	// -----------------------------------------------------
	// Reading header
	// -----------------------------------------------------

	while (std::getline(in, line)) {

        //  +--------------------+------------------------------------------+------------+
        //  |RINEX VERSION / TYPE| - Format version                         | F9.2,11X,  |#
        //  |                    | - File type ('N' = GPS nav mess data)    |   A1,19X   |
        //  |                    | - Satellite system (G, E, J, C)          |   A1,19X   |
        //  +--------------------+------------------------------------------+------------+
        if (getComment(line).rfind("RINEX VERSION / TYPE", 0) == 0){
            if (line.substr(20,1) != "N"){
                std::cout << "Error: file " << nav_file_path << " is not a GNSS rinex navigation file" << std::endl;
                assert(false);
            }
            constellation = line.substr(40,1);
            navData.setConstellation(constellation);

            if (constellation == "R") {
                std::cout << "Error: GLONASS rinex version " << line.substr(0,9) << " is not supported yet" << std::endl;
                assert(false);
            }

            if (constellation == "G") std::cout << "GPS ";
            if (constellation == "C") std::cout << "Beidou ";
            if (constellation == "E") std::cout << "Galileo ";
            if (constellation == "J") std::cout << "QZSS ";
            std::cout << "(" << constellation << ") constellation navigation data" << std::endl;

        }

		// -----------------------------------------------------
		// End of header
		// -----------------------------------------------------
		if (getComment(line).rfind("END OF HEADER", 0) == 0){
			break;
		}
	}

	// -----------------------------------------------------
	// Reading main part of file
	// -----------------------------------------------------

	std::vector<double> line_values;


	while (std::getline(in, line)) {

		if (getComment(line).rfind("COMMENT", 0) == 0){
			continue;
		}

		NavigationSlot slot;

		slot.setPRN(std::stoi(line.substr(1,2)));

		GPSTime time   (std::stoi(line.substr( 4,4)),    // Years
						std::stoi(line.substr( 9,2)),    // Months
						std::stoi(line.substr(12,2)),    // Days
						std::stoi(line.substr(15,2)),    // Hours
						std::stoi(line.substr(18,2)),    // Minutes
						std::stoi(line.substr(21,2)));   // Seconds

		slot.setTime(time);

		line_values = Utils::parseLine(line,23,3,19);
		slot.fillLine(line_values, 1);

		for (int num_line=2; num_line<8; num_line++){
			std::getline(in, line);
			line_values = Utils::parseLine(line,4,4,19);
			slot.fillLine(line_values, num_line);
		}

		std::getline(in, line);

		// -----------------------------------------------------
		// Contrôle de redondance
		// -----------------------------------------------------

		bool tps_different = true;
		bool prn_different = true;

        for (unsigned i=0; i<navData.getNavigationSlots().size(); i++){
            tps_different = (navData.getNavigationSlots().at(i).getTime() - time != 0);
            prn_different = (navData.getNavigationSlots().at(i).getPRN() != slot.getPRN());
            if ((!tps_different) && (!prn_different)) break;
        }

        if ((tps_different || prn_different)) {
            navData.addNavigationSlot(slot);
        } else{
            REDUNDANT ++;
        }

	}

	in.close();

	// -----------------------------------------------------
	// Data summary
	// -----------------------------------------------------
	std::cout << navData.getNumberOfNavigationSlots() << " data slots loaded with success" << std::endl;
	if (navData.getNumberOfNavigationSlots() > 0){
		std::cout << "Date of first navigation slot:  " << navData.getNavigationSlot(0).getTime() << std::endl;
		std::cout << "Date of last navigation slot :  " << navData.getNavigationSlot(navData.getNumberOfNavigationSlots()-1).getTime() << std::endl;
	}

	std::cout << sep << std::endl;

	int nb_sat = 0;
	if (constellation == "G") nb_sat = NB_GPS_SATS;
	if (constellation == "E") nb_sat = NB_GALILEO_SATS;
	if (constellation == "Q") nb_sat = NB_QZSS_SATS;
	if (constellation == "C") nb_sat = NB_BEIDOU_SATS;

	int ilim = nb_sat/4;

	int prn = 0;
	for (int i=0; i<ilim; i++){
		for (int j=0; j<4; j++){
			prn = i*4+j;
            std::cout << "PRN " << constellation << Utils::formatNumber(prn+1, "%02d") << "  ";
			std::cout << Utils::formatNumber(navData.getPRNCount()[prn],"% 3d") << "      ";
		}
		std::cout << std::endl;
	}

	std::vector<int> MISSING_SATS;
	for (int i=0; i<32; i++){
		if (navData.getPRNCount()[i] == 0){
			MISSING_SATS.push_back(i+1);
		}
	}

	if (MISSING_SATS.size() > 0){
        std::cout << sep << std::endl;
		std::cout << "WARNING: MISSING SATELLITES ";
		for (unsigned int i=0; i<MISSING_SATS.size(); i++){
			std::cout << MISSING_SATS.at(i) << " ";
			if (i % 20 == 16) std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	if (REDUNDANT > 0){
        std::cout << sep << std::endl;
        std::cout << "WARNING: " << REDUNDANT << " REDUNDANT DATA SLOT(S)" << std::endl;
	}

	return navData;

}



// ---------------------------------------------------------------
// Fonction de lecture d'un fichier de navigation rinex GLONASS
// https://www.glonass-iac.ru/en/GLONASS/docs/rinex2.htm
// ---------------------------------------------------------------
GloNavigationData RinexReader::readGloNavFile(std::string nav_file_path){

    GloNavigationData navData;
    navData.setSourceFilePath(nav_file_path);

    std::ifstream in(nav_file_path.c_str());

    if (!in) {
		std::cout << "Cannot open navigation input file [" << nav_file_path << "]" << std::endl;
		return navData;
	}

	std::string line;
	std::string comment;

	std::vector<NavigationSlot> NavSlots;

	int REDUNDANT = 0;

    // -----------------------------------------------------
	// Reading header
	// -----------------------------------------------------

	while (std::getline(in, line)) {

        //  +--------------------+------------------------------------------+------------+
        //  |RINEX VERSION / TYPE| - Format version (2.10)                  | F9.2,11X,  |#
        //  |                    | - File type ('G' = GLONASS nav mess data)|   A1,39X   |
        //  +--------------------+------------------------------------------+------------+
        if (getComment(line).rfind("RINEX VERSION / TYPE", 0) == 0){
            if (line.substr(20,1) != "G"){
                std::cout << "Error: file " << nav_file_path << " is not a GLONASS rinex navgation file" << std::endl;
                assert(false);
            }
        }

        // +--------------------+------------------------------------------+------------+
		// |LEAP SECONDS        | Delta time due to leap seconds           |     I6     |
		// +--------------------+------------------------------------------+------------+
		if (getComment(line).rfind("LEAP SECONDS", 0) == 0){
			navData.setLeapSeconds(std::stoi(line.substr(0,6)));
		}

        // +--------------------+------------------------------------------+------------+
		//*|CORR TO SYSTEM TIME | - Time of reference for system time corr |            |*
		// |                    |   (year, month, day)                     |     3I6,   |
		// |                    | - Correction to system time scale (sec)  |  3X,D19.12 |
		// |                    |   to correct GLONASS system time to      |            |
		// |                    |   UTC(SU)                         (-TauC)|            |
		// +--------------------+------------------------------------------+------------+
		if (getComment(line).rfind("CORR TO SYSTEM TIME", 0) == 0){

            int year  = std::stoi(line.substr( 0, 6));
            int month = std::stoi(line.substr( 6,12));
            int day   = std::stoi(line.substr(12,18));
            double ct = std::stof(line.substr(21,19));

            GPSTime time(year, month, day, 0, 0, 0);
            navData.setTRefCorr(time); navData.setCorrectionTime(ct);

		}

        // +--------------------+------------------------------------------+------------+
        // |END OF HEADER       | Last record in the header section.       |    60X     |
        // +--------------------+------------------------------------------+------------+
		if (getComment(line).rfind("END OF HEADER", 0) == 0){
			break;
		}

	}

	// -----------------------------------------------------
	// Reading main part of file
	// -----------------------------------------------------
	std::vector<double> line_values;
	while (std::getline(in, line)) {

		if (getComment(line).rfind("COMMENT", 0) == 0){
			continue;
		}

		GloNavigationSlot slot;

		 // +--------------------+------------------------------------------+------------+
         // |PRN / EPOCH / SV CLK| - Satellite number:                      |     I2,    |
         // |                    |       Slot number in sat. constellation  |            |
         // |                    | - Epoch of ephemerides             (UTC) |            |
         // |                    |     - year (2 digits, padded with 0,     |   1X,I2.2, |
         // |                    |                if necessary)             |            |
         // |                    |     - month,day,hour,minute,             |  4(1X,I2), |
         // |                    |     - second                             |    F5.1,   |
         // |                    | - SV clock bias (sec)             (-TauN)|   D19.12,  |
         // |                    | - SV relative frequency bias    (+GammaN)|   D19.12,  |
         // |                    | - message frame time                 (tk)|   D19.12   |
         // |                    |   (0 .le. tk .lt. 86400 sec of day UTC)  |            |
         // +--------------------+------------------------------------------+------------+

		slot.setPRN(std::stoi(line.substr(0,2)));

		// Year conversion to 4 digits
		int year = std::stoi(line.substr(2+0*3,3));
		year += (year >= 70)?1900:2000;

		GPSTime time   (year,                               // Years
						std::stoi(line.substr(2+1*3,3)),    // Months
						std::stoi(line.substr(2+2*3,3)),    // Days
						std::stoi(line.substr(2+3*3,3)),    // Hours
						std::stoi(line.substr(2+4*3,3)),    // Minutes
						std::stoi(line.substr(2+5*3,3)));   // Seconds

		slot.setTime(time);

        line_values = Utils::parseLine(line,22,3,19);
        slot.fillLine(line_values, 1);

        for (int num_line=2; num_line<5; num_line++){
			std::getline(in, line);
			line_values = Utils::parseLine(line,3,4,19);
			slot.fillLine(line_values, num_line);
		}

        // -----------------------------------------------------
		// Contrôle de redondance
		// -----------------------------------------------------

		bool tps_different = true;
		bool prn_different = true;

        for (unsigned i=0; i<navData.getNavigationSlots().size(); i++){
            tps_different = (navData.getNavigationSlots().at(i).getTime() - time != 0);
            prn_different = (navData.getNavigationSlots().at(i).getPRN() != slot.getPRN());
            if ((!tps_different) && (!prn_different)) break;
        }

        if ((tps_different || prn_different)) {
            navData.addNavigationSlot(slot);
        } else{
            REDUNDANT ++;
        }

	}

	in.close();

	// -----------------------------------------------------
	// Data summary
	// -----------------------------------------------------
	std::cout << navData.getNumberOfNavigationSlots() << " data slots loaded with success" << std::endl;
	if (navData.getNumberOfNavigationSlots() > 0){
		std::cout << "Date of first navigation slot:  " << navData.getNavigationSlot(0).getTime() << std::endl;
		std::cout << "Date of last navigation slot :  " << navData.getNavigationSlot(navData.getNumberOfNavigationSlots()-1).getTime() << std::endl;
	}

	std::cout << sep << std::endl;

	int prn = 0;
	for (int i=0; i<8; i++){
		for (int j=0; j<3; j++){
			prn = i*3+j;
			std::cout << "PRN R" << Utils::formatNumber(prn+1, "%02d") << "  ";
			std::cout << Utils::formatNumber(navData.getPRNCount()[prn],"% 3d") << "      ";
		}
		std::cout << std::endl;
	}

	std::vector<int> MISSING_SATS;
	for (int i=0; i<24; i++){
		if (navData.getPRNCount()[i] == 0){
			MISSING_SATS.push_back(i+1);
		}
	}

	if (MISSING_SATS.size() > 0){
        std::cout << sep << std::endl;
		std::cout << "WARNING: MISSING SATELLITES ";
		for (unsigned int i=0; i<MISSING_SATS.size(); i++){
			std::cout << MISSING_SATS.at(i) << " ";
		}
		std::cout << std::endl;
	}

	if (REDUNDANT > 0){
        std::cout << sep << std::endl;
        std::cout << "WARNING: " << REDUNDANT << " REDUNDANT DATA SLOT(S)" << std::endl;
	}

	std::cout << std::endl;

	return navData;

}

// ---------------------------------------------------------------
// Fonction de lecture d'un fichier d'observations rinex
// ---------------------------------------------------------------
ObservationData RinexReader::readObsFile(std::string obs_file_path){

	ObservationData obsData;
	obsData.setSourceFilePath(obs_file_path);

	std::ifstream in(obs_file_path.c_str());

	if (!in) {
		std::cout << "Cannot open observation input file [" << obs_file_path << "]" << std::endl;
		assert(false);
		return obsData;
	}

	std::cout << sep2 << std::endl;
	std::cout << "Reading rinex observation file " << obs_file_path << std::endl;
	std::cout << sep2 << std::endl;

	std::string line;
	std::string comment;

	double nb_sat_avg = 0;
	int number_of_observables_in_file = 0;
	std::vector <std::string> TYPES_OF_OBS;

	// -----------------------------------------------------
	// Reading header
	// -----------------------------------------------------
	while (std::getline(in, line)) {

		// +--------------------+------------------------------------------+------------+
		// |RINEX VERSION / TYPE| - Format version : 3.01                  | F9.2,11X,  |
		// |                    | - File type: O for Observation Data      |   A1,19X,  |
		// |                    | - Satellite System: G: GPS               |   A1,19X   |
		// |                    |                     R: GLONASS           |            |
		// |                    |                     E: Galileo           |            |
		// |                    |                     S: SBAS payload      |            |
		// |                    |                     M: Mixed             |            |
		// +--------------------+------------------------------------------+------------+
		if (getComment(line).rfind("RINEX VERSION / TYPE", 0) == 0){
			obsData.setObservationType(line.substr(40,1));
		}

		// +--------------------+------------------------------------------+------------+
		// |MARKER NAME         | Name of antenna marker                   |     A60    |
		// +--------------------+------------------------------------------+------------+
		if (getComment(line).rfind("MARKER NAME", 0) == 0){
			obsData.setMarkerName(Utils::trim(line.substr(0,60)));
		}

		// +--------------------+------------------------------------------+------------+
		// |REC # / TYPE / VERS | Receiver number, type, and version       |    3A20    |
		// |                    | (Version: e.g. Internal Software Version)|            |
		// +--------------------+------------------------------------------+------------+
		if (getComment(line).rfind("REC # / TYPE / VERS", 0) == 0){
			obsData.setReceiverType(Utils::trim(line.substr(20,20)));
		}

		// +--------------------+------------------------------------------+------------+
		// |ANT # / TYPE        | Antenna number and type                  |    2A20    |
		// +--------------------+------------------------------------------+------------+
		if (getComment(line).rfind("ANT # / TYPE", 0) == 0){
			obsData.setAntennaType(Utils::trim(line.substr(20,20)));
		}

		// +--------------------+------------------------------------------+------------+
		// |APPROX POSITION XYZ | Geocentric approximate marker position   |   3F14.4   |
		// |                    | (Units: Meters, System: ITRS recommended)|            |
		// |                    | Optional for moving platforms            |            |
		// +--------------------+------------------------------------------+------------+
		if (getComment(line).rfind("APPROX POSITION XYZ", 0) == 0){
			ECEFCoords approx(Utils::parseLine(line, 0, 3, 14));
			obsData.setApproximatePosition(approx);
		}

		// +--------------------+------------------------------------------+------------+
		// |ANTENNA: DELTA H/E/N| - Antenna height: Height of the antenna  |    F14.4,  |
		// |                    |   reference point (ARP) above the marker |            |
		// |                    | - Horizontal eccentricity of ARP         |   2F14.4   |
		// |                    |   relative to the marker (east/north)    |            |
		// |                    | All units in meters                      |            |
		// +--------------------+------------------------------------------+------------+
		if (getComment(line).rfind("ANTENNA: DELTA H/E/N", 0) == 0){
			std::vector<double> hen = Utils::parseLine(line, 0, 3, 14);
			ENUCoords enh(hen.at(1), hen.at(2), hen.at(0));
			obsData.setDeltaEnu(enh);
		}

		// +--------------------+------------------------------------------+------------+
		// |INTERVAL            | Observation interval in seconds          |   F10.3    |
		// +--------------------+------------------------------------------+------------+
		if (getComment(line).rfind("INTERVAL", 0) == 0){
			obsData.setInterval(Utils::atof2(line.substr(0,10)));
		}

		// +--------------------+------------------------------------------+------------+
		// |LEAP SECONDS        | - Number of leap seconds since 6-Jan-1980|     I6,    |
		// +--------------------+------------------------------------------+------------+
		if (getComment(line).rfind("LEAP SECONDS", 0) == 0){
			obsData.setLeapSeconds(std::stoi(line.substr(0,6)));
		}

		// +--------------------+------------------------------------------+------------+
		// |# / TYPES OF OBSERV | - Number of different observation types  |     I6,    |
		// |                    |   stored in the file                     |            |
		// |                    | - Observation types                      |            |
		// |                    |   - Observation code                     | 9(4X,A1,   |
		// |                    |   - Frequency code                       |         A1)|
		// |                    |   If more than 9 observation types:      |            |
		// |                    |     Use continuation line(s) (including  |6X,9(4X,2A1)|
		// |                    |     the header label in cols. 61-80!)    |            |
		// +--------------------+------------------------------------------+------------+
		if (getComment(line).rfind("# / TYPES OF OBSERV", 0) == 0){
            if (TYPES_OF_OBS.size() == 0){
                number_of_observables_in_file = std::stoi(line.substr(3,3));
            }
			line = Utils::trim(line.substr(10,50));
			int nb_rec_line = (line.size()+4)/6.0;
			for (int i=0; i<nb_rec_line; i++){
				std::string obs_type = line.substr(6*i,2);
				TYPES_OF_OBS.push_back(obs_type);
			}
		}

		// -----------------------------------------------------
		// End of header
		// -----------------------------------------------------
		if (getComment(line).rfind("END OF HEADER", 0) == 0){
			break;
		}

	}

	int lli, sgs;
	int nb_line_per_sat = ceil(number_of_observables_in_file/5.0);

	// -----------------------------------------------------
	// Reading data lines
	// -----------------------------------------------------
	while (std::getline(in, line)) {

		if (line.substr(0,10) == "          ") continue;
		if (getComment(line).rfind("COMMENT", 0) == 0) continue;

		ObservationSlot slot;

		// +-------------+-------------------------------------------------+------------+
		// | EPOCH/SAT   | - Epoch :                                       |            |
		// |     or      |   - year (2 digits, padded with 0 if necessary) |  1X,I2.2,  |
		// | EVENT FLAG  |   - month,day,hour,min,                         |  4(1X,I2), |
		// |             |   - sec                                         |   F11.7,   |
		// +-------------+-------------------------------------------------+------------+

		// Year conversion to 4 digits
		int year = std::stoi(line.substr(0,3));
		year += (year >= 70)?1900:2000;

		GPSTime time   (year,                               // Years
						std::stoi(line.substr(1+1*3,3)),    // Months
						std::stoi(line.substr(1+2*3,3)),    // Days
						std::stoi(line.substr(1+3*3,3)),    // Hours
						std::stoi(line.substr(1+4*3,3)),    // Minutes
						std::stoi(line.substr(1+5*3,3)),    // Seconds
                        std::stoi(line.substr(1+6*3,3)));   // Milisec


		slot.setTimestamp(time);

		// +-------------+-------------------------------------------------+------------+
		// |             | - Number of satellites in current epoch         |     I3,    |
		// |             | - List of PRNs (sat.numbers with system         | 12(A1,I2), |
		// |             |   identifier, see 5.1) in current epoch         |            |
		// |             | - receiver clock offset (seconds, optional)     |   F12.9    |
		// |             |                                                 |            |
		// |             |   If more than 12 satellites: Use continuation  |    32X,    |
		// |             |   line(s)                                       | 12(A1,I2)  |
		// +-------------+-------------------------------------------------+------------+

		int nb_sats = std::stoi(line.substr(29,3));

		int nb_sat_lines = nb_sats/12;
		int nb_sat_remaining = nb_sats % 12;

		std::vector<std::string> LIST_OF_SATS;

		for (int i=0; i<nb_sat_lines; i++){
			for (int j=0; j<12; j++) LIST_OF_SATS.push_back(line.substr(32+3*j,3));
			if ((nb_sat_remaining != 0) || (i < nb_sat_lines-1)) std::getline(in, line);
		}

		for (int j=0; j<nb_sat_remaining; j++){
			LIST_OF_SATS.push_back(line.substr(32+3*j,3));
		}

		// Standardisation des noms de PRN (e.g. 'G 1' => "G01")
		for (unsigned i=0; i<LIST_OF_SATS.size(); i++) LIST_OF_SATS[i] = Utils::makeStdSatName(LIST_OF_SATS[i]);

		// +-------------+-------------------------------------------------+------------+
		// |OBSERVATIONS | - Observation      | rep. within record for     |  m(F14.3,  |
		// |             | - LLI              | each obs.type (same seq    |     I1,    |
		// |             | - Signal strength  | as given in header)        |     I1)    |
		// |             |                                                 |            |
		// |             | If more than 5 observation types (=80 char):    |            |
		// |             | continue observations in next record.           |            |
		// +-------------+-------------------------------------------------+------------+
		for (unsigned int i=0; i<LIST_OF_SATS.size(); i++){

			Observation obs(LIST_OF_SATS.at(i));
			obs.setTimestamp(time);

			std::string line_of_obs = "";

			for (int j=0; j<nb_line_per_sat; j++){
				std::getline(in, line);
				while (line.size() < 78) line += " ";
				line_of_obs += line;
			}

			std::vector<double> observables = Utils::parseLine(line_of_obs, 0, number_of_observables_in_file, 16);

			for (unsigned int l=0; l<observables.size(); l++){
				bool condition = false;
				condition = condition || (TYPES_OF_OBS.at(l) == "L1");
				condition = condition || (TYPES_OF_OBS.at(l) == "L2");
				condition = condition || (TYPES_OF_OBS.at(l) == "C1");
				condition = condition || (TYPES_OF_OBS.at(l) == "P1");
				condition = condition || (TYPES_OF_OBS.at(l) == "P2");
				condition = condition || (TYPES_OF_OBS.at(l) == "D1");
				condition = condition || (TYPES_OF_OBS.at(l) == "D2");
				if (condition){
					std::string lli_str = line_of_obs.substr(16*l+14, 1);
					std::string sgs_str = line_of_obs.substr(16*l+15, 1);
					lli = (lli_str != " ")? std::stoi(lli_str):-1;
					sgs = (sgs_str != " ")? std::stoi(sgs_str):-1;
					obs.setChannel(TYPES_OF_OBS.at(l), observables.at(l), lli, sgs);
				}
			}

			slot.addObservation(obs);

		}

		obsData.addObservationSlot(slot);
		nb_sat_avg += LIST_OF_SATS.size();

	}

	in.close();

	// -----------------------------------------------------
	// Data summary
	// -----------------------------------------------------
	std::cout << obsData.getNumberOfObservationSlots() << " epoch(s) loaded with success" << std::endl;
	if (obsData.getNumberOfObservationSlots() > 0){
		std::cout << "Date of first observation:  " << obsData.getObservationSlots().front().getTimestamp() << std::endl;
		std::cout << "Date of last observation:   " << obsData.getObservationSlots().back().getTimestamp() << std::endl;
	}

	std::cout << sep << std::endl;
	std::cout << "Recorded observations: ";
	for (unsigned int l=0; l<TYPES_OF_OBS.size(); l++){
		bool condition = false;
		condition = condition || (TYPES_OF_OBS.at(l) == "L1");
		condition = condition || (TYPES_OF_OBS.at(l) == "L2");
		condition = condition || (TYPES_OF_OBS.at(l) == "C1");
		condition = condition || (TYPES_OF_OBS.at(l) == "P1");
		condition = condition || (TYPES_OF_OBS.at(l) == "P2");
		condition = condition || (TYPES_OF_OBS.at(l) == "D1");
		condition = condition || (TYPES_OF_OBS.at(l) == "D2");
		if (condition){
			std::cout << TYPES_OF_OBS.at(l) << " ";
		}
	}

	nb_sat_avg /= obsData.getNumberOfObservationSlots();

	std::cout << std::endl;
	std::cout << "Average number of satellites: " << Utils::formatNumber(nb_sat_avg, "%4.2f") << std::endl;
	std::cout << "Point name: " << obsData.getMarkerName();
	std::cout << " (Receiver/Antenna: " << Utils::trim(obsData.getReceiverType().substr(0,20));
	std::cout << "/" << Utils::trim(obsData.getAntennaType().substr(0,15)) << ")" << std::endl;
	std::cout << "Approx. antenna: " << obsData.getApproxAntennaPosition() << std::endl;
	std::cout << sep << std::endl;
	std::cout << std::endl;

	return obsData;

}

















