#include "Utils.h"

// ---------------------------------------------------------------
// Function to format an integer number in string
// ---------------------------------------------------------------
std::string Utils::formatNumber(int number, std::string code) {

	char buffer [100];
  	snprintf (buffer, 100, code.c_str(), number);
	std::string output = buffer;
    return output;
}

// ---------------------------------------------------------------
// Function to format a floating point number in string
// ---------------------------------------------------------------
std::string Utils::formatNumber(double number, std::string code) {

	char buffer [100];
  	snprintf (buffer, 100, code.c_str(), number);
	std::string output = buffer;
    return output;
}


// ---------------------------------------------------------------
// Function to convert rinex char (format with 'D') to double
// ---------------------------------------------------------------
double Utils::atof2(std::string entry){

    for (unsigned i=0; i<entry.length(); ++i){
        if ((entry.at(i) == 'D') || (entry.at(i) == 'd')){
            entry.at(i) = 'e';
        }
	}

    // Standard conversion
    return std::atof(entry.c_str());

}

// ---------------------------------------------------------------
// Function to read formatted line based on a specification code
// ---------------------------------------------------------------
std::vector<double> Utils::parseLine(std::string& line, int offset, int nb, int length){

	std::vector<double> output;

	for (int i=0; i<nb; i++){
		output.push_back(atof2(line.substr(offset+i*length, length)));
	}

	return output;

}

// ---------------------------------------------------------------
// Function to trim whitespaces in a string
// ---------------------------------------------------------------
std::string Utils::trim(std::string str){
    size_t first = str.find_first_not_of(' ');
    if (first == std::string::npos){
        return "";
	}
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last-first+1));
}

// ---------------------------------------------------------------
// Function to make blank string
// ---------------------------------------------------------------
std::string Utils::blank(int length){
    std::string line = "";
	for (int i=0; i<length; i++) line += " ";
    return line;
}

// ---------------------------------------------------------------
// Test d'égalité de noms de PRN (e.g. 'G01' = 'G 1')
// ---------------------------------------------------------------
bool Utils::prn_equal(std::string prn1, std::string prn2){
    bool prefix = (prn1.substr(0,1) == prn2.substr(0,1));
    bool number = (std::stoi(prn1.substr(1,2)) == std::stoi(prn2.substr(1,2)));
    return (prefix && number);
}

// ---------------------------------------------------------------
// Conversion des noms de satellites (e.g. 'G 1' => 'G01')
// ---------------------------------------------------------------
std::string Utils::makeStdSatName(std::string prn){
    prn = prn.substr(0,1) + Utils::formatNumber(std::stoi(prn.substr(1,2)),"%02d");
    return prn;
}


// ---------------------------------------------------------------
// Function to get supported satellites by gnss.cpp lib
// ---------------------------------------------------------------
std::vector<std::string> Utils::getSupportedSatellites(){

	std::vector<std::string> SATS;

	if (GPS_CONST)     for (int i=1; i<=32; i++) SATS.push_back("G"+Utils::formatNumber(i,"%02d"));
	if (GLONASS_CONST) for (int i=1; i<=26; i++) SATS.push_back("R"+Utils::formatNumber(i,"%02d"));
	if (GALILEO_CONST) for (int i=1; i<=24; i++) SATS.push_back("E"+Utils::formatNumber(i,"%02d"));

	return SATS;

}

// ---------------------------------------------------------------
// Découpage d'une chaîne en tokens élémentaires
// ---------------------------------------------------------------
std::vector<std::string> Utils::tokenize(std::string texte, std::string delimiter){

	size_t delimiter_pos = 0;
	std::string token = "";
	std::vector<std::string> tokens;

	delimiter_pos = texte.find(delimiter.c_str());
   	while ((delimiter_pos > 0) && (delimiter_pos < texte.size()+1)){
		std::string token = texte.substr(0, delimiter_pos);
		texte = texte.substr(delimiter_pos+1, texte.size()-delimiter_pos);
		delimiter_pos = texte.find(delimiter.c_str());
		tokens.push_back(token);
	}

	tokens.push_back(texte);

	return tokens;

}
