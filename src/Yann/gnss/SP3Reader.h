#ifndef SP3READER_H
#define SP3READER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>

#include "Utils.h"
#include "SP3NavigationData.h"

// ---------------------------------------------------------------
// Classe contenant les fonctions de lecture de fichiers sp3
// ---------------------------------------------------------------
class SP3Reader {

	public:
		static SP3NavigationData readNavFile(std::string);

};

#endif


