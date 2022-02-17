#include "StdAfx.h"

using namespace std;

//----
// v0 functions
//----

void writePointMatch_v0( ostream &aOutput, const list<PointMatch> &aList )
{
	list<PointMatch>::const_iterator it = aList.begin();
	while ( it!=aList.end() )
	{
		const Pt2dr &p0 = it->first, &p1 = (*it++).second;
		aOutput << p0.x << ' ' << p0.y << ' ' << p1.x << ' ' << p1.y << endl;
	}
}

void writePointMatch_v0( ostream &aOutput, const vector<PointMatch> &aList )
{
	const PointMatch *it = aList.data();
	size_t i = aList.size();
	while ( i-- )
	{
		const Pt2dr &p0 = it->first, &p1 = (*it++).second;
		aOutput << p0.x << ' ' << p0.y << ' ' << p1.x << ' ' << p1.y << endl;
	}
}

void readPointMatch_v0( istream &aInput, vector<PointMatch> &oVector )
{
	list<PointMatch> readList;

	REAL8 x0, y0, x1, y1;
	size_t nbMatches = 0;
	while ( !aInput.eof() )
	{
		aInput >> x0 >> y0 >> x1 >> y1;
		readList.push_back( PointMatch( Pt2dr(x0,y0), Pt2dr(x1,y1) ) );
		nbMatches++;
	}
	readList.pop_back();
	nbMatches--;

	oVector.resize(nbMatches);
	list<PointMatch>::const_iterator itSrc = readList.begin();
	PointMatch *itDst = oVector.data();
	while ( nbMatches-- ) *itDst++ = *itSrc++;
}


//----
// v1 functions
//----

void reverseRealVector( vector<REAL8> &aVector )
{
	REAL8 *itReal = aVector.data();
	size_t iReal = aVector.size();
	while ( iReal-- ) byte_inv_8(itReal++);
}

void toRealVector( const list<PointMatch> &aMatches, bool aReverseByteOrder, vector<REAL8> &oVector )
{
	size_t iMatch = aMatches.size();
	oVector.resize(4*iMatch);
	list<PointMatch>::const_iterator itMatch = aMatches.begin();
	REAL8 *itReal = oVector.data();
	while ( itMatch!=aMatches.end() )
	{
		const Pt2dr &p0 = itMatch->first, &p1 = (*itMatch++).second;
		*itReal++ = p0.x;
		*itReal++ = p0.y;
		*itReal++ = p1.x;
		*itReal++ = p1.y;
	}

	if (aReverseByteOrder) reverseRealVector(oVector);
}

void toRealVector( const vector<PointMatch> &aMatches, bool aReverseByteOrder, vector<REAL8> &oVector )
{
	size_t iMatch = aMatches.size();
	oVector.resize(4*iMatch);
	const PointMatch *itSrc = aMatches.data();
	REAL8 *itDst = oVector.data();
	while ( iMatch-- )
	{
		const Pt2dr &p0 = itSrc->first, &p1 = (*itSrc++).second;
		*itDst++ = p0.x;
		*itDst++ = p0.y;
		*itDst++ = p1.x;
		*itDst++ = p1.y;
	}

	if (aReverseByteOrder) reverseRealVector(oVector);
}

void fromRealVector( vector<REAL8> &aVector, bool aReverseByteOrder, vector<PointMatch> &oMatches )
{
	if (aReverseByteOrder) reverseRealVector(aVector);

	oMatches.resize(aVector.size() / 4);
	size_t iMatch = oMatches.size();
	const REAL8 *itSrc = aVector.data();
	PointMatch *itDst = oMatches.data();
	while ( iMatch-- )
	{
		Pt2dr &p0 = itDst->first, &p1 = (*itDst++).second;
		p0.x = *itSrc++;
		p0.y = *itSrc++;
		p1.x = *itSrc++;
		p1.y = *itSrc++;
	}

}

void writePointMatch_v1( std::ostream &aOutput, bool aReverseByteOrder, const vector<PointMatch> &aMatches )
{
	U_INT4 nbMatches = (U_INT4)aMatches.size();
	if (aReverseByteOrder) byte_inv_4(&nbMatches);
	aOutput.write((const char *)&nbMatches, 4);

	vector<REAL8> reals;
	toRealVector(aMatches, aReverseByteOrder, reals);
	aOutput.write((const char *)reals.data(), reals.size()*8);
}

void writePointMatch_v1( std::ostream &aOutput, bool aReverseByteOrder, const list<PointMatch> &aMatches )
{
	U_INT4 nbMatches = (U_INT4)aMatches.size();
	if (aReverseByteOrder) byte_inv_4(&nbMatches);
	aOutput.write((const char *)&nbMatches, 4);

	vector<REAL8> reals;
	toRealVector(aMatches, aReverseByteOrder, reals);
	aOutput.write((const char *)reals.data(), reals.size()*8);
}

void readPointMatch_v1( std::istream &aInput, bool aReverseByteOrder, vector<PointMatch> &oVector )
{
	U_INT4 nbMatches;
	aInput.read((char *)&nbMatches, 4);
	if (aReverseByteOrder) byte_inv_4(&nbMatches);

	vector<REAL8> reals(4 * (size_t)nbMatches);
	aInput.read((char *)reals.data(), reals.size() * 8);
	fromRealVector(reals, aReverseByteOrder, oVector);
}


//----
// all versions functions
//----

// read/write a list of Digeo points
bool writeMatchesFile( const string &aFilename, const list<PointMatch> &aList ) // use last version of the format and processor's byte order
{
	return writeMatchesFile(aFilename, aList, g_versioned_headers_list[VFH_PointMatch].last_handled_version, MSBF_PROCESSOR());
}

bool writeMatchesFile( const string &aFilename, const vector<PointMatch> &aList ) // use last version of the format and processor's byte order
{
	return writeMatchesFile(aFilename, aList, g_versioned_headers_list[VFH_PointMatch].last_handled_version, MSBF_PROCESSOR());
}

bool writeMatchesFile( const string &aFilename, const list<PointMatch> &oList, U_INT4 aVersion, bool aWriteBigEndian )
{
	ofstream f( aFilename.c_str(), ios::binary );
	if (!f) return false;

	if ( aVersion>0 )
	{
		VersionedFileHeader header(VFH_PointMatch, aVersion, aWriteBigEndian);
		header.write(f);
	}
	bool reverseByteOrder = ( aWriteBigEndian!=MSBF_PROCESSOR() );

	switch( aVersion )
	{
	case 0: writePointMatch_v0(f, oList); break;
	case 1: writePointMatch_v1(f, reverseByteOrder, oList); break;
	default: return false;
	};

	return true;
}

bool writeMatchesFile( const string &aFilename, const vector<PointMatch> &oVector, U_INT4 aVersion, bool aWriteBigEndian )
{
	ofstream f( aFilename.c_str(), ios::binary );
	if (!f) return false;

	if ( aVersion>0 )
	{
		VersionedFileHeader header(VFH_PointMatch, aVersion, aWriteBigEndian);
		header.write(f);
	}
	bool reverseByteOrder = ( aWriteBigEndian!=MSBF_PROCESSOR() );

	switch( aVersion )
	{
	case 0: writePointMatch_v0(f, oVector); break;
	case 1: writePointMatch_v1(f, reverseByteOrder, oVector); break;
	default: return false;
	};

	return true;
}

bool readMatchesFile( const string &aFilename, vector<PointMatch> &oVector, VersionedFileHeader *oHeader )
{
	ifstream f(aFilename.c_str(), ios::binary);
	if (!f) return false;

	VersionedFileHeader header;
	header.read_known(VFH_PointMatch, f);
	if ( oHeader!=NULL ) *oHeader=header;
	bool reverseByteOrder = ( header.isMSBF()!=MSBF_PROCESSOR() );

	try
	{
		switch (header.version())
		{
		case 0: readPointMatch_v0(f, oVector); break;
		case 1: readPointMatch_v1(f, reverseByteOrder, oVector); break;
		default: return false;
		}
	}
	catch (const bad_alloc &)
	{
		ELISE_DEBUG_ERROR(true, "readMatchesFile", "not enough memory to read file [" << aFilename << "]");
		return false;
	}

	return true;
}

