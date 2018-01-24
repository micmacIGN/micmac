#ifndef MERGEHOMOL_H
#define MERGEHOMOL_H
#include "StdAfx.h"
#include <fstream>
#include <algorithm>
#include <iterator>
#include <map>
//#include <boost/filesystem.hpp>


vector<string> getFilesList(string ndir);
vector<string> getSubDirList(string ndir);
set<string> getFilesSet(string ndir);
vector<string> getDirListRegex(string pattern);


#endif // MERGEHOMOL_H

