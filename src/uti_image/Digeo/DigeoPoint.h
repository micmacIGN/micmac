#ifndef __DIGEO_POINT__
#define __DIGEO_POINT__

#include <string>
#include <vector>
#include <list>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include "general/sys_dep.h"
#include "private/VersionedFileHeader.h"

//#define __DEBUG_DIGEO_POINT

#define DIGEO_DESCRIPTOR_SIZE 128
#define DIGEO_MAX_NB_ANGLES 4

class DigeoPoint
{
public:
	typedef enum
	{
		DETECT_UNKNOWN = 0,
		DETECT_LOCAL_MIN = 1,
		DETECT_LOCAL_MAX = 2
	} DetectType; // do not forget to change DigeoPoint::nbDetectTypes in DigeoPoint.cpp when adding a new type

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
	static const unsigned int nbDetectTypes;

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
	static bool writeDigeoFile( const std::string &i_filename, const std::vector<DigeoPoint> &i_list ); // use last version of the format and processor's byte order
	static bool writeDigeoFile( const std::string &i_filename, const std::list<DigeoPoint> &i_list );
	static bool writeDigeoFile( const std::string &i_filename, const std::vector<DigeoPoint> &i_list, U_INT4 i_version, bool i_writeBigEndian );
	static bool writeDigeoFile( const std::string &i_filename, const std::list<DigeoPoint> &i_list, U_INT4 i_version, bool i_writeBigEndian );
	// this reading function detects the fileformat and can be used with old siftpp_tgi files
	// if o_header is not null, addressed variable is filled
	static bool readDigeoFile( const std::string &i_filename, bool i_storeMultipleAngles, std::vector<DigeoPoint> &o_list, VersionedFileHeader *o_header=NULL );

	static void multipleToUniqueAngle( std::vector<DigeoPoint> &io_points );
	static void uniqueToMultipleAngles( std::vector<DigeoPoint> &io_points );

	static void removePointsOfType( DetectType i_type, std::vector<DigeoPoint> &io_points );

	void removeDuplicatedEntries();

	// remove duplicated points and duplicated entries for the same point
	static void removeDuplicates( std::vector<DigeoPoint> &io_points, int i_gridSize=100 );

	double minDescriptorDistance2( const DigeoPoint &aPoint ) const;
};

std::ostream & operator <<( std::ostream &s, const DigeoPoint &p );

std::string DetectType_to_string( DigeoPoint::DetectType i_type );



//----
// DigeoPoint inline methods
//----


inline DigeoPoint::DigeoPoint(): x(0), y(0), scale(0), type( DETECT_UNKNOWN ){}

inline size_t DigeoPoint::nbAngles() const { return entries.size(); }

inline REAL8 DigeoPoint::angle( size_t i_index ) const
{
	#ifdef __DEBUG_DIGEO
		if ( i_index>=entries.size() ){ std::cerr << "angle: index out of range : " << i_index << " >= " << entries.size() << std::endl; exit(EXIT_FAILURE); }
	#endif
	return entries[i_index].angle;
}

inline void DigeoPoint::setAngle( size_t i_index, REAL8 i_angle )
{
	#ifdef __DEBUG_DIGEO
		if ( i_index>=entries.size() ){ std::cerr << "setAngle: index out of range : " << i_index << " >= " << entries.size() << std::endl; exit(EXIT_FAILURE); }
	#endif
	entries[i_index].angle = i_angle;
}

inline const DigeoPoint::Entry & DigeoPoint::entry( size_t i_index ) const
{
	#ifdef __DEBUG_DIGEO
		if ( i_index>=entries.size() ){ std::cerr << "entry const: index out of range : " << i_index << " >= " << entries.size() << std::endl; exit(EXIT_FAILURE); }
	#endif
	return entries[i_index];
}

inline DigeoPoint::Entry & DigeoPoint::entry( size_t i_index )
{
	#ifdef __DEBUG_DIGEO
		if ( i_index>=entries.size() ){ std::cerr << "entry: index out of range : " << i_index << " >= " << entries.size() << std::endl; exit(EXIT_FAILURE); }
	#endif
	return entries[i_index];
}

inline void DigeoPoint::setDescriptor( size_t i_index, const REAL8 *i_descriptor )
{
	#ifdef __DEBUG_DIGEO
		if ( i_index>=entries.size() ){ std::cerr << "setDescriptor: index out of range : " << i_index << " >= " << entries.size() << std::endl; exit(EXIT_FAILURE); }
	#endif
	memcpy( entries[i_index].descriptor, i_descriptor, 8*DIGEO_DESCRIPTOR_SIZE );
}

inline const REAL8 * DigeoPoint::descriptor( size_t i_index ) const
{
	#ifdef __DEBUG_DIGEO
		if ( i_index>=entries.size() ){ std::cerr << "descriptor const: index out of range : " << i_index << " >= " << entries.size() << std::endl; exit(EXIT_FAILURE); }
	#endif
	return entries[i_index].descriptor;
}

inline REAL8 * DigeoPoint::descriptor( size_t i_index )
{
	#ifdef __DEBUG_DIGEO
		if ( i_index>=entries.size() ){ std::cerr << "descriptor: index out of range : " << i_index << " >= " << entries.size() << std::endl; exit(EXIT_FAILURE); }
	#endif
	return entries[i_index].descriptor;
}

#endif // __DIGEO_POINT__
