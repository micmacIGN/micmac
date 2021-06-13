#ifndef RINEXREADER_H
#define RINEXREADER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>

#include "Utils.h"
#include "NavigationData.h"
#include "ObservationData.h"
#include "GloNavigationData.h"

// ---------------------------------------------------------------
// Classe contenant les fonctions de lecture de fichiers rinex
// (observations et navigation).
// ---------------------------------------------------------------
class RinexReader {


	private:
		static std::string getComment(std::string&);

	public:

        static ObservationData readObsFile(std::string);
		static NavigationData readNavFile(std::string);
		static NavigationData readNavFileV2(std::string);
		static NavigationData readNavFileV3(std::string);
		static GloNavigationData readGloNavFile(std::string);


};

#endif

