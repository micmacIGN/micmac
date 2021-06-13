#ifndef __POINT_MATCH__
#define __POINT_MATCH__

#include <vector>
#include <list>
#include <iostream>
#include <utility>

typedef std::pair<Pt2dr,Pt2dr> PointMatch;

// read/write a list of Digeo points
bool writeMatchesFile( const std::string &aFilename, const std::list<PointMatch> &aList ); // use last version of the format and processor's byte order
bool writeMatchesFile( const std::string &aFilename, const std::vector<PointMatch> &aList ); // use last version of the format and processor's byte order

bool writeMatchesFile( const string &aFilename, const list<PointMatch> &oList, U_INT4 aVersion, bool aWriteBigEndian );
bool writeMatchesFile( const string &aFilename, const vector<PointMatch> &oList, U_INT4 aVersion, bool aWriteBigEndian );

bool readMatchesFile( const string &aFilename, vector<PointMatch> &aVector, VersionedFileHeader *oHeader=NULL );

#endif
