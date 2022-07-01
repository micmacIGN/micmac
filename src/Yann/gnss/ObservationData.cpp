#include <assert.h>
#include <iostream>
#include <fstream>

#include "ObservationData.h"

// -------------------------------------------------------------------------------
// Fonction de recherche d'une époque dans un fichier d'observations
// Retourne l'époque la plus proche de celle demandée
// -------------------------------------------------------------------------------
ObservationSlot& ObservationData::lookForEpoch(GPSTime time){

    double min = 1e30; int argmin = 0;
    for (unsigned i=0; i<this->getNumberOfObservationSlots(); i++){
        GPSTime time_epoch = this->getObservationSlots().at(i).getTimestamp();
        double diff = std::abs(time.convertToAbsTime() - time_epoch.convertToAbsTime());
        if (diff < min){
            argmin = i;
            min = diff;
        }
    }

    return this->getObservationSlots().at(argmin);

}

void ObservationData::removeSatellite(std::string sat_name){
    sat_name = Utils::makeStdSatName(sat_name);
	for (unsigned int i=0; i<ObservationSlots.size(); i++){
		ObservationSlots.at(i).removeSatellite(sat_name);
	}
	this->addComment("Edited: all "+sat_name+" observations excluded");
}

void ObservationData::removeConstellation(std::string constellation_name){
	for (unsigned int i=0; i<ObservationSlots.size(); i++){
		ObservationSlots.at(i).removeConstellation(constellation_name);
	}
	std::string code = "";
	if (constellation_name == "G") code = "GPS";
	if (constellation_name == "E") code = "GALILEO";
	if (constellation_name == "R") code = "GLONASS";
	if (constellation_name == "Q") code = "QZSS";
	if (constellation_name == "S") code = "SBAS";
	if (code != ""){
		this->addComment("Edited: all "+code+" satellites excluded");
	}
}

void ObservationData::extractSpanTime(GPSTime start, GPSTime end){

	std::vector<ObservationSlot> slots;

	for (unsigned int i=0; i<ObservationSlots.size(); i++){
		if (ObservationSlots.at(i).getTimestamp() < start) continue;
		if (ObservationSlots.at(i).getTimestamp() > end)  continue;
		slots.push_back(ObservationSlots.at(i));
	}

	this->ObservationSlots = slots;
	this->addComment("Edited: all obs. before " + start.to_string() + " excluded");
    this->addComment("Edited: all obs. after  " + end.to_string()   + " excluded");
}

void ObservationData::slice(int subsampling_factor){

	std::vector<ObservationSlot> slots;

	for (unsigned int i=0; i<ObservationSlots.size(); i+=subsampling_factor){
		slots.push_back(ObservationSlots.at(i));
	}

	this->interval = this->interval*subsampling_factor;
	this->ObservationSlots = slots;
	this->addComment("Forced Modulo Decimation to "+
                  std::to_string(subsampling_factor)+" seconds");}

void ObservationData::deactivateChannel(std::string channel){

	if (channel == "C1")  {this->c1_valid = false; return;}
	if (channel == "L1")  {this->l1_valid = false; return;}
	if (channel == "L2")  {this->l2_valid = false; return;}
	if (channel == "P1")  {this->p1_valid = false; return;}
	if (channel == "P2")  {this->p2_valid = false; return;}
	if (channel == "D1")  {this->d1_valid = false; return;}
	if (channel == "D2")  {this->d2_valid = false; return;}

	std::cout << "ERROR: type " << channel << " is not a valid type of observable" << std::endl;
	assert(false);

}

void ObservationData::activateChannel(std::string channel){

	if (channel == "C1")  {this->c1_valid = true; return;}
	if (channel == "L1")  {this->l1_valid = true; return;}
	if (channel == "L2")  {this->l2_valid = true; return;}
	if (channel == "P1")  {this->p1_valid = true; return;}
	if (channel == "P2")  {this->p2_valid = true; return;}
	if (channel == "D1")  {this->d1_valid = true; return;}
	if (channel == "D2")  {this->d2_valid = true; return;}

	std::cout << "ERROR: type " << channel << " is not a valid type of observable" << std::endl;
	assert(false);
}

std::vector<std::string> ObservationData::getActivatedChannels(){

	std::vector<std::string> channels;

	if (this->c1_valid)  {channels.push_back("C1");}
	if (this->l1_valid)  {channels.push_back("L1");}
	if (this->l2_valid)  {channels.push_back("L2");}
	if (this->p1_valid)  {channels.push_back("P1");}
	if (this->p2_valid)  {channels.push_back("P2");}
	if (this->d1_valid)  {channels.push_back("D1");}
	if (this->d2_valid)  {channels.push_back("D2");}

	return channels;

}

// -------------------------------------------------------------------------------
// Récupération de la position d'anntenne : marker position - delta ENU
// -------------------------------------------------------------------------------
ECEFCoords ObservationData::getApproxAntennaPosition(){
    ECEFCoords ref = this->approximate_position;
    ref.X -= 1;
    ref.Y -= 1;
    ref.Z -= 1;
    ENUCoords approx_enu = this->approximate_position.toENUCoords(ref) + this->delta_enu;
    return approx_enu.toECEFCoords(ref);
}

// -------------------------------------------------------------------------------
// Fonction d'écriture d'un fichier rinex (format 2.11)
// -------------------------------------------------------------------------------
void ObservationData::printRinexFile(std::string rinex_file_path){

	std::cout << "Printing rinex file " << rinex_file_path << "..." << "\r";

	std::ofstream output;
	output.open (rinex_file_path);

	// +--------------------+------------------------------------------+------------+
	// |RINEX VERSION / TYPE| - Format version : 3.01                  | F9.2,11X,  |
	// |                    | - File type: O for Observation Data      |   A1,19X,  |
	// |                    | - Satellite System: G: GPS               |   A1,19X   |
	// |                    |                     R: GLONASS           |            |
	// |                    |                     E: Galileo           |            |
	// |                    |                     S: SBAS payload      |            |
	// |                    |                     M: Mixed             |            |
	// +--------------------+------------------------------------------+------------+
	output << Utils::formatNumber(2.11,"%9.2f") << Utils::blank(11);
	output << "OBSERVATION DATA" << Utils::blank(4);
	output << this->getObservationType() << Utils::blank(19);
	output << "RINEX VERSION / TYPE" << std::endl;

	// +--------------------+------------------------------------------+------------+
	// |COMMENT             | Comment (suppression, add, notice...)    |     A60    |
	// +--------------------+------------------------------------------+------------+
	for (unsigned int i=0; i<this->comments.size(); i++){
		int length = comments.at(i).size();
		int nl = length/59; int remaining = length % 59;
		for (int j=0; j<nl; j++) {
			output << comments.at(i).substr(j*59,59) << " COMMENT" << std::endl;
		}
		output << comments.at(i).substr(nl*59, remaining);
		output << Utils::blank(60-remaining) << "COMMENT" << std::endl;
	}

	// +--------------------+------------------------------------------+------------+
	// |MARKER NAME         | Name of antenna marker                   |     A60    |
	// +--------------------+------------------------------------------+------------+
	int temp1 = static_cast<int>(60-this->getMarkerName().size());
	output << this->getMarkerName() << Utils::blank(temp1);
	output << "MARKER NAME" << std::endl;

	// +--------------------+------------------------------------------+------------+
	// |REC # / TYPE / VERS | Receiver number, type, and version       |    3A20    |
	// |                    | (Version: e.g. Internal Software Version)|            |
	// +--------------------+------------------------------------------+------------+
	output << Utils::blank(20);
	int temp2 = static_cast<int>(20-this->getReceiverType().size());
	output << this->getReceiverType() << Utils::blank(temp2);
	output << Utils::blank(20) << "REC # / TYPE / VERS" << std::endl;

	// +--------------------+------------------------------------------+------------+
	// |ANT # / TYPE        | Antenna number and type                  |    2A20    |
	// +--------------------+------------------------------------------+------------+
	output << Utils::blank(20);
	int temp3 = static_cast<int>(20-this->getAntennaType().size());
	output << this->getAntennaType() << Utils::blank(temp3);
	output << Utils::blank(20) << "ANT # / TYPE" << std::endl;

	// +--------------------+------------------------------------------+------------+
	// |APPROX POSITION XYZ | Geocentric approximate marker position   |   3F14.4   |
	// |                    | (Units: Meters, System: ITRS recommended)|            |
	// |                    | Optional for moving platforms            |            |
	// +--------------------+------------------------------------------+------------+
	output << Utils::formatNumber(this->approximate_position.X, "%14.4f");
	output << Utils::formatNumber(this->approximate_position.Y, "%14.4f");
	output << Utils::formatNumber(this->approximate_position.Z, "%14.4f");
	output << Utils::blank(60-3*14) << "APPROX POSITION XYZ" << std::endl;

	// +--------------------+------------------------------------------+------------+
	// |ANTENNA: DELTA H/E/N| - Antenna height: Height of the antenna  |    F14.4,  |
	// |                    |   reference point (ARP) above the marker |            |
	// |                    | - Horizontal eccentricity of ARP         |   2F14.4   |
	// |                    |   relative to the marker (east/north)    |            |
	// |                    | All units in meters                      |            |
	// +--------------------+------------------------------------------+------------+
	output << Utils::formatNumber(this->getDeltaEnu().U, "%14.4f");
	output << Utils::formatNumber(this->getDeltaEnu().E, "%14.4f");
	output << Utils::formatNumber(this->getDeltaEnu().N, "%14.4f");
	output << Utils::blank(60-3*14) << "ANTENNA: DELTA H/E/N" << std::endl;

	// +--------------------+------------------------------------------+------------+
	// |INTERVAL            | Observation interval in seconds          |   F10.3    |
	// +--------------------+------------------------------------------+------------+
	output << Utils::formatNumber(this->getInterval(), "%10.3f");
	output << Utils::blank(50) << "INTERVAL" << std::endl;

	// +--------------------+------------------------------------------+------------+
	// |LEAP SECONDS        | - Number of leap seconds since 6-Jan-1980|     I6,    |
	// +--------------------+------------------------------------------+------------+
	output << Utils::formatNumber(this->getLeapSeconds(), "%6d");
	output << Utils::blank(54) << "LEAP SECONDS" << std::endl;

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
	std::vector<std::string> codes = getActivatedChannels();
	int nb_channels = static_cast<int>(codes.size());

	output << Utils::formatNumber(nb_channels, "%6d");

	for (int i=0; i<nb_channels; i++){
		output << Utils::blank(4) << codes.at(i);
	}

	output << Utils::blank(60-6-nb_channels*6) << "# / TYPES OF OBSERV" << std::endl;

	// +--------------------+------------------------------------------+------------+
	// |TIME OF FIRST OBS   | - Time of first observation record       | 5I6,F13.7, |
	// |                    |	(4-digit-year, month,day,hour,min,sec) |            |
	// +--------------------+------------------------------------------+------------+
	GPSTime& start = this->ObservationSlots.front().getTimestamp();
	output << Utils::formatNumber(start.year,      "%6d");
	output << Utils::formatNumber(start.month,     "%6d");
	output << Utils::formatNumber(start.day,       "%6d");
	output << Utils::formatNumber(start.hour,      "%6d");
	output << Utils::formatNumber(start.min,       "%6d");
	output << Utils::formatNumber(start.sec+0., "%13.7f");
	output << Utils::blank(60-5*6-13) << "TIME OF FIRST OBS" << std::endl;

	// +--------------------+------------------------------------------+------------+
	// |TIME OF LAST OBS    | - Time of first observation record       | 5I6,F13.7, |
	// |                    |	(4-digit-year, month,day,hour,min,sec) |            |
	// +--------------------+------------------------------------------+------------+
	GPSTime& end = this->ObservationSlots.back().getTimestamp();
	output << Utils::formatNumber(end.year,      "%6d");
	output << Utils::formatNumber(end.month,     "%6d");
	output << Utils::formatNumber(end.day,       "%6d");
	output << Utils::formatNumber(end.hour,      "%6d");
	output << Utils::formatNumber(end.min,       "%6d");
	output << Utils::formatNumber(end.sec+0., "%13.7f");
	output << Utils::blank(60-5*6-13) << "TIME OF LAST OBS" << std::endl;

	output << Utils::blank(60) << "END OF HEADER" << std::endl;

	for (unsigned int i=0; i<this->ObservationSlots.size(); i++){

		ObservationSlot& slot = this->ObservationSlots.at(i);
		
		// Empty slot
        if (slot.getSatellites().size() == 0) continue;
       

		// +-------------+-------------------------------------------------+------------+
		// | EPOCH/SAT   | - Epoch :                                       |            |
		// |     or      |   - year (2 digits, padded with 0 if necessary) |  1X,I2.2,  |
		// | EVENT FLAG  |   - month,day,hour,min,                         |  4(1X,I2), |
		// |             |   - sec                                         |   F11.7,   |
		// +-------------+-------------------------------------------------+------------+
		GPSTime& tps = slot.getTimestamp();
		output << Utils::blank(1) << Utils::formatNumber(tps.year,      "%4d").substr(2,2);
		output << Utils::blank(1) << Utils::formatNumber(tps.month,     "%2d");
		output << Utils::blank(1) << Utils::formatNumber(tps.day,       "%2d");
		output << Utils::blank(1) << Utils::formatNumber(tps.hour,      "%2d");
		output << Utils::blank(1) << Utils::formatNumber(tps.min,       "%2d");
		output << Utils::formatNumber(tps.sec+0., "%11.7f");

		// +-------------+-------------------------------------------------+------------+
		// |             | - Number of satellites in current epoch         |     I3,    |
		// |             | - List of PRNs (sat.numbers with system         | 12(A1,I2), |
		// |             |   identifier, see 5.1) in current epoch         |            |
		// |             | - receiver clock offset (seconds, optional)     |   F12.9    |
		// |             |                                                 |            |
		// |             |   If more than 12 satellites: Use continuation  |    32X,    |
		// |             |   line(s)                                       | 12(A1,I2)  |
		// +-------------+-------------------------------------------------+------------+
		std::vector<std::string> sat_list = slot.getSatellites();

		output << "  0" << Utils::formatNumber((int) sat_list.size(), "%3d");

		for (unsigned int sat=0; sat<sat_list.size(); sat++) {
			if ((sat%12 == 0) && (sat>0)) output << std::endl << Utils::blank(32);
			output << sat_list.at(sat);
		}

		output << std::endl;

		for (unsigned int sat=0; sat<sat_list.size(); sat++) {

			Observation& obs = slot.getObservation(sat_list.at(sat));

			// +-------------+-------------------------------------------------+------------+
			// |OBSERVATIONS | - Observation      | rep. within record for     |  m(F14.3,  |
			// |             | - LLI              | each obs.type (same seq    |     I1,    |
			// |             | - Signal strength  | as given in header)        |     I1)    |
			// |             |                                                 |            |
			// |             | If more than 5 observation types (=80 char):    |            |
			// |             | continue observations in next record.           |            |
			// +-------------+-------------------------------------------------+------------+
			for (unsigned int k=0; k<codes.size(); k++){
				if ((k%5 == 0) && (k>0)) output << std::endl;
				if (obs.hasChannel(codes.at(k))){

					output << Utils::formatNumber(obs.getChannel(codes.at(k)),"%14.3f");

					if (obs.getChannel_LLI(codes.at(k)) == -1){
						output << Utils::blank(1);
					} else{
						output << obs.getChannel_LLI(codes.at(k));
					}

					if (obs.getChannel_SGS(codes.at(k)) == -1){
						output << Utils::blank(1);
					} else{
						output << obs.getChannel_SGS(codes.at(k));
					}

				} else {
					output << Utils::blank(16);
				}
			}

			output << std::endl;

		}

	}

	output.close();

	std::cout << "Writing rinex file " << rinex_file_path << "... done" << std::endl;


}
