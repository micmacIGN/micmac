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

	typedef union{
		struct{
			REAL8 angle;
			REAL8 descriptor[DIGEO_DESCRIPTOR_SIZE];
		};
		REAL8 all[DIGEO_DESCRIPTOR_SIZE+1];
	} Entry;

	REAL8      x, y,
	           scale;
	DetectType type;
	std::vector<Entry> entries; // first is the angle, second is the descriptor itself
	//REAL8      angles[DIGEO_MAX_NB_ANGLES];
	//REAL8      descriptors[DIGEO_MAX_NB_ANGLES][DIGEO_DESCRIPTOR_SIZE];

	static unsigned char sm_uchar_descriptor[DIGEO_DESCRIPTOR_SIZE];
	static REAL8 sm_real8_descriptor[DIGEO_DESCRIPTOR_SIZE];

	inline DigeoPoint();

	void addDescriptor( REAL8 i_angle );
	void addDescriptor( REAL8 i_angle, const REAL8 *i_descriptor );
	void addDescriptor( const REAL8 *i_descriptor );

	bool operator ==( const DigeoPoint &i_b ) const;
	bool operator !=( const DigeoPoint &i_b ) const;

	// getters
	inline size_t nbAngles() const;
	inline REAL8 angle( size_t i_index ) const;
	inline const REAL8 * descriptor( size_t i_index ) const;
	inline REAL8 * descriptor( size_t i_index );
	inline const Entry & entry( size_t i_index ) const;
	inline Entry & entry( size_t i_index );
	
	// setters
	inline void setAngle( size_t i_index, REAL8 i_angle );
	inline void setDescriptor( size_t i_index, const REAL8 *i_descriptor );

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

inline size_t DigeoPoint::nbAngles() const { return entries.size(); }

inline REAL8 DigeoPoint::angle( size_t i_index ) const
{
	#ifdef __DEBUG_DIGEO
		ELISE_ASSERT( i_index<entries.size(), ( string("angle: index out of range : ")+ToString(i_index)+" >= "+ToString(entries.size()) ).c_str() );
	#endif
	return entries[i_index].angle;
}

inline void DigeoPoint::setAngle( size_t i_index, REAL8 i_angle )
{
	#ifdef __DEBUG_DIGEO
		ELISE_ASSERT( i_index<entries.size(), ( string("setAngle: index out of range : ")+ToString(i_index)+" >= "+ToString(entries.size()) ).c_str() );
	#endif
	entries[i_index].angle = i_angle;
}

inline const DigeoPoint::Entry & DigeoPoint::entry( size_t i_index ) const
{
	#ifdef __DEBUG_DIGEO
		ELISE_ASSERT( i_index<entries.size(), ( string("entry const: index out of range : ")+ToString(i_index)+" >= "+ToString(entries.size()) ).c_str() );
	#endif
	return entries[i_index];
}

inline DigeoPoint::Entry & DigeoPoint::entry( size_t i_index )
{
	#ifdef __DEBUG_DIGEO
		ELISE_ASSERT( i_index<entries.size(), ( string("entry: index out of range : ")+ToString(i_index)+" >= "+ToString(entries.size()) ).c_str() );
	#endif
	return entries[i_index];
}

inline void DigeoPoint::setDescriptor( size_t i_index, const REAL8 *i_descriptor )
{
	#ifdef __DEBUG_DIGEO
		ELISE_ASSERT( i_index<entries.size(), ( string("setDescriptor: index out of range : ")+ToString(i_index)+" >= "+ToString(entries.size()) ).c_str() );
	#endif
	memcpy( entries[i_index].descriptor, i_descriptor, 8*DIGEO_DESCRIPTOR_SIZE );
}

inline const REAL8 * DigeoPoint::descriptor( size_t i_index ) const
{
	#ifdef __DEBUG_DIGEO
		ELISE_ASSERT( i_index<entries.size(), ( string("descriptor const: index out of range : ")+ToString(i_index)+" >= "+ToString(entries.size()) ).c_str() );
	#endif
	return entries[i_index].descriptor;
}

inline REAL8 * DigeoPoint::descriptor( size_t i_index )
{
	#ifdef __DEBUG_DIGEO
		ELISE_ASSERT( i_index<entries.size(), ( string("descriptor: index out of range : ")+ToString(i_index)+" >= "+ToString(entries.size()) ).c_str() );
	#endif
	return entries[i_index].descriptor;
}

#endif // __DIGEO_POINT__
