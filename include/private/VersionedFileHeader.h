#ifndef __VERSIONED_HEADER__
#define __VERSIONED_HEADER__

#include <stdint.h>
#include <string>
#include <iostream>

#include "general/sys_dep.h"
#include "private/util.h"

typedef enum{
   VFH_Digeo=0,
   VFH_TracePack,
   VFH_PointMatch,
   VFH_RawIm2D,
   VFH_Unknown // this one must stay at the end of the list
} VFH_Type;

typedef struct 
{
   uint32_t magic_number_MSBF;
   uint32_t magic_number_LSBF;
   uint32_t last_handled_version;
   std::string name;
} versioned_file_header_t;

extern versioned_file_header_t g_versioned_headers_list[];

void byte_inv_4(void * t);
bool MSBF_PROCESSOR();

class VersionedFileHeader
{
private:
   char     m_isMSBF;
   uint32_t m_magicNumber;
   uint32_t m_version;

public:
   inline VersionedFileHeader();
          VersionedFileHeader( VFH_Type i_type );
          VersionedFileHeader( VFH_Type i_type, uint32_t i_version, bool i_isMSBF );

   inline bool     isMSBF() const;
   inline uint32_t version() const;
   inline uint32_t magicNumber() const;

   bool read_raw( std::istream &io_stream );
   bool read_known( VFH_Type i_type, std::istream &io_stream ); // read the header of a known type, return false if the type is different
   bool read_unknown( std::istream &io_stream, VFH_Type &o_type );
   void write( std::ostream &io_stream ) const;
};

//--------------------------------------------
// related functions
//--------------------------------------------

inline int nbVersionedTypes();

bool typeFromMagicNumber( uint32_t i_magic, VFH_Type &o_type, bool &o_isMSBF );

VFH_Type versionedFileType( const std::string &i_filename );

// generate a new random number than is not equal to itself in reverse byte order and that is not already used
void generate_new_magic_number( uint32_t &o_direct, uint32_t &o_reverse );

void inverseByteOrder( U_INT1 * ); // useless except for templates
void inverseByteOrder( INT1 * ); // useless except for templates
void inverseByteOrder( U_INT2 * );
void inverseByteOrder( INT2 * );
//void inverseByteOrder( U_INT4 * );
void inverseByteOrder( INT4 * );
//void inverseByteOrder( U_INT8 * );
//void inverseByteOrder( INT8 * );
void inverseByteOrder( REAL4 * );
void inverseByteOrder( REAL8 * );
void inverseByteOrder( REAL16 * );

#include "VersionedFileHeader.inline.h"

#endif
