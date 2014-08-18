#ifndef __DIGEO_POINT__
#define __DIGEO_POINT__

#include <string>
#include <vector>
#include <list>
#include <iostream>
#include "general/sys_dep.h"

#define DIGEO_FILEFORMAT_MAGIC_NUMBER_LSBF 989008845ul
#define DIGEO_FILEFORMAT_MAGIC_NUMBER_MSBF 3440636730ul
#define DIGEO_FILEFORMAT_CURRENT_VERSION 1
#define DIGEO_DESCRIPTOR_SIZE 128
#define DIGEO_MAX_NB_ANGLES 4

extern const U_INT4 digeo_fileformat_magic_number;
extern const U_INT4 digeo_fileformat_current_version;

typedef struct
{
	unsigned char byteOrder;
	U_INT4        version;
	U_INT4        nbPoints;
	U_INT4        descriptorSize;

	bool reverseByteOrder;
} DigeoFileHeader;

class DigeoPoint
{
public:
	typedef enum
	{
		DETECT_UNKNOWN,
		DETECT_LOCAL_MIN,
		DETECT_LOCAL_MAX
	} DetectType;

	REAL8      x, y,
	           scale;
	DetectType type;
	std::vector<std::pair<REAL8,REAL8[DIGEO_DESCRIPTOR_SIZE]> > descriptors; // first is the angle, second is the descriptor itself
	//REAL8      angles[DIGEO_MAX_NB_ANGLES];
	//REAL8      descriptors[DIGEO_MAX_NB_ANGLES][DIGEO_DESCRIPTOR_SIZE];

	static unsigned char sm_uchar_descriptor[DIGEO_DESCRIPTOR_SIZE];
	static REAL8 sm_real8_descriptor[DIGEO_DESCRIPTOR_SIZE];

	DigeoPoint();

	bool operator ==( const DigeoPoint &i_b ) const;
	bool operator !=( const DigeoPoint &i_b ) const;

	void write_v0( std::ostream &output ) const;
	void read_v0( std::istream &output );

	void write_v1( std::ostream &output, bool reverseByteOrder ) const;
	void read_v1( std::istream &output, bool reverseByteOrder );

	// read/write a list of Digeo points
	static bool writeDigeoFile( const std::string &i_filename, const std::vector<DigeoPoint> &i_list, U_INT4 i_version=DIGEO_FILEFORMAT_CURRENT_VERSION, bool i_writeBigEndian=MSBF_PROCESSOR() );
	static bool writeDigeoFile( const std::string &i_filename, const std::list<DigeoPoint> &i_list, U_INT4 i_version=DIGEO_FILEFORMAT_CURRENT_VERSION, bool i_writeBigEndian=MSBF_PROCESSOR() );
	// this reading function detects the fileformat and can be used with old siftpp_tgi files
	// if o_header is not null, addressed variable is filled
	static bool readDigeoFile( const std::string &i_filename, bool i_storeMultipleAngles, std::vector<DigeoPoint> &o_list, DigeoFileHeader *o_header=NULL );

	static void multipleToUniqueAngle( std::vector<DigeoPoint> &io_points );
	static void uniqueToMultipleAngles( std::vector<DigeoPoint> &io_points );

	static void removePointsOfType( DetectType i_type, std::vector<DigeoPoint> &io_points );
};

std::ostream & operator <<( std::ostream &s, const DigeoPoint &p );





//----
// DigeoPoint inline methods
//----


inline DigeoPoint::DigeoPoint():type( DETECT_UNKNOWN ){}

#endif // __DIGEO_POINT__
