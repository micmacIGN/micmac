#include "SP3Reader.h"
#include "SP3NavigationData.h"
#include "SP3NavigationSlot.h"

// ---------------------------------------------------------------
// Fonction de lecture d'un fichier de navigation SP3
// ---------------------------------------------------------------
SP3NavigationData SP3Reader::readNavFile(std::string nav_file_path){

	SP3NavigationData navData;

	std::string sep  = "-----------------------------------------------------------------------";
	std::string sep2 = "=======================================================================";

	std::ifstream in(nav_file_path.c_str());

	if (!in) {
		std::cout << "Cannot open navigation input file [" << nav_file_path << "]" << std::endl;
		return navData;
	}

	std::cout << sep2 << std::endl;
	std::cout << "Reading sp3 navigation file " << nav_file_path << std::endl;
	std::cout << sep2 << std::endl;

	std::string line;
	std::string comment;

	// -----------------------------------------------------
	// Reading header
	// -----------------------------------------------------

	std::vector<double> line_values;
	std::vector<std::string> clock_errrors;


	while (std::getline(in, line)) {

		// Header lines
		if (line.rfind("#", 0) == 0) continue;
		if (line.rfind("+", 0) == 0) continue;
		if (line.rfind("%", 0) == 0) continue;
		if (line.rfind("/", 0) == 0) continue;

		// End of file
		if (line.rfind("EOF", 0) == 0) break;

		// New slot
		if (line.rfind("*", 0) == 0){

			// Year conversion to 4 digits
			int year = std::stoi(line.substr(4,4));
			year += (year >= 70)?1900:2000;


			GPSTime time   (year,                               // Years
							std::stoi(line.substr(4+1*3,3)),    // Months
							std::stoi(line.substr(4+2*3,3)),    // Days
							std::stoi(line.substr(4+3*3,3)),    // Hours
							std::stoi(line.substr(4+4*3,3)),    // Minutes
							std::stoi(line.substr(4+5*3,3)));   // Seconds

			SP3NavigationSlot slot(time);
			navData.slots.push_back(slot);



		}else{

			std::string prn = Utils::makeStdSatName(line.substr(1,3));
			navData.slots.back().PRN.push_back(prn);

			line_values = Utils::parseLine(line,4,4,14);
			navData.slots.back().X.push_back(line_values.at(0));
			navData.slots.back().Y.push_back(line_values.at(1));
			navData.slots.back().Z.push_back(line_values.at(2));
			navData.slots.back().T.push_back(line_values.at(3));

			if (line_values[3] == Utils::SP3_NO_DATA_VAL){
				clock_errrors.push_back(line.substr(0,4));
			}

		}

	}

	in.close();

	// -----------------------------------------------------
	// Data summary
	// -----------------------------------------------------
	size_t n = navData.slots.size()*navData.slots.at(0).PRN.size();
	std::cout << navData.slots.size() << " data slots loaded with success (" << n << " records)" << std::endl;
	if (navData.slots.size() > 0){
		std::cout << "Date of first navigation slot:  " << navData.slots.at(0).time << std::endl;
		std::cout << "Date of last navigation slot:   " << navData.slots.back().time << std::endl;

		std::cout << sep << std::endl;

		if (clock_errrors.size() > 0){
			size_t pc = clock_errrors.size()*100/n;
			std::cout << "WARNING: Satellite clock unknown for " << clock_errrors.size() << " (" << pc << " %) record(s)" << std::endl;
		}
	}

	return navData;

}
