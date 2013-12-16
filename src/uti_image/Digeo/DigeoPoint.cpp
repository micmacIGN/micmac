#include "StdAfx.h"

#include <fstream>

using namespace std;

unsigned char DigeoPoint::sm_uchar_descriptor[DIGEO_DESCRIPTOR_SIZE];

// v0 is the same format as siftpp_tgi (not endian-wise)
// REAL8 values are cast to float
// descriptor is cast to unsigned char values d[i]->(unsigned char)(512*d[i])
void DigeoPoint::write_v0( ostream &output ) const
{
   REAL4 float_value = (REAL4)x; output.write( (char*)&float_value, 4 );
   float_value = (REAL4)y; output.write( (char*)&float_value, 4 );
   float_value = (REAL4)scale; output.write( (char*)&float_value, 4 );
   float_value = (REAL4)angle; output.write( (char*)&float_value, 4 );
   
   // cast descriptor elements to unsigned char before writing
   unsigned char *it_uchar = sm_uchar_descriptor;
   const Real_ *itReal = descriptor;
   size_t i = DIGEO_DESCRIPTOR_SIZE;
   while (i--) (*it_uchar++)=(unsigned char)( 512*(*itReal++) );
   
   output.write( (char*)sm_uchar_descriptor, m_descriptorSize );
}

void DigeoPoint::write_v1( ostream &output, bool reverseByteOrder ) const
{
   REAL4 float_values[4] = { (REAL4)x,
			     (REAL4)y,
			     (REAL4)scale,
			     (REAL4)angle };
   INT2 int2 = type;
   if ( reverseByteOrder )
   {
      byte_inv_4( float_values );
      byte_inv_4( float_values+1 );
      byte_inv_4( float_values+2 );
      byte_inv_4( float_values+3 );
      byte_inv_2( &int2 );
   }
   output.write( (char*)float_values, 16 );
   
   // cast descriptor elements to unsigned char before writing
   unsigned char *it_uchar = sm_uchar_descriptor;
   const REAL8 *itReal = descriptor;
   size_t i = DIGEO_DESCRIPTOR_SIZE;
   while (i--) (*it_uchar++)=(unsigned char)( 512*(*itReal++) );
   
   output.write( (char*)sm_uchar_descriptor, m_descriptorSize );
   
   output.write( (char*)&int2, 2 );
}

void DigeoPoint::read_v0( std::istream &output )
{
   REAL4 float_values[4];
   output.read( (char*)float_values, 16 );
   x 	 = (REAL8)float_values[0];
   y 	 = (REAL8)float_values[1];
   scale = (REAL8)float_values[2];
   angle = (REAL8)float_values[3];
   
   output.read( (char*)sm_uchar_descriptor, m_descriptorSize );
   
   // cast descriptor elements to REAL4 after reading
   unsigned char *it_uchar = sm_uchar_descriptor;
   REAL8 *itReal = descriptor;
   size_t i = DIGEO_DESCRIPTOR_SIZE;
   while (i--) (*itReal++)=( (REAL8)(*it_uchar++)/512 );
}

void DigeoPoint::read_v1( std::istream &output, bool reverseByteOrder )
{
   REAL4 float_values[4];
   output.read( (char*)float_values, 16 );
      
   output.read( (char*)sm_uchar_descriptor, m_descriptorSize );
   
   // cast descriptor elements to REAL4 after reading
   unsigned char *it_uchar = sm_uchar_descriptor;
   REAL8 *itReal = descriptor;
   size_t i = DIGEO_DESCRIPTOR_SIZE;
   while (i--) (*itReal++)=( (REAL8)(*it_uchar++)/512 );
      
   INT2 int2 = type;
   output.read( (char*)&int2, 2 );
   
   if ( reverseByteOrder )
   {
      byte_inv_4( float_values );
      byte_inv_4( float_values+1 );
      byte_inv_4( float_values+2 );
      byte_inv_4( float_values+3 );
      byte_inv_2( &int2 );
   }
   
   x 	 = (REAL8)float_values[0];
   y 	 = (REAL8)float_values[1];
   scale = (REAL8)float_values[2];
   angle = (REAL8)float_values[3];
   type  = (DetectType)int2;
}

//----
// reading functions
//----

void readDigeoFile_v0( istream &stream, const DigeoFileHeader &header, vector<DigeoPoint> &o_vector )
{   
   // read points
   o_vector.resize( header.nbPoints );
   if ( header.nbPoints==0 ) return;
   DigeoPoint *itPoint = &o_vector[0];
   U_INT4 iPoint = header.nbPoints;
   while ( iPoint-- ) ( *itPoint++ ).read_v0( stream );
}

void readDigeoFile_v1( istream &stream, const DigeoFileHeader &header, vector<DigeoPoint> &o_vector )
{
   // read points
   o_vector.resize( header.nbPoints );
   if ( header.nbPoints==0 ) return;
   DigeoPoint *itPoint = &o_vector[0];
   U_INT4 iPoint = header.nbPoints;
   while ( iPoint-- ) ( *itPoint++ ).read_v1( stream, header.reverseByteOrder );
}

// o_mustInverse is true if file's byte order and system's byte order are different
bool readDigeoFileHeader( istream &stream, DigeoFileHeader &header )
{
   U_INT4 magic_number;
   stream.read( (char*)&magic_number, 4 );
   if ( magic_number==DIGEO_FILEFORMAT_MAGIC_NUMBER_LSBF ||
        magic_number==DIGEO_FILEFORMAT_MAGIC_NUMBER_MSBF )
   {
      stream.read( (char*)&header.byteOrder, 1 );
      char cpuByteOrder = ( MSBF_PROCESSOR()?2:1 ),
	   magicNumberByteOrder = ( magic_number==DIGEO_FILEFORMAT_MAGIC_NUMBER_MSBF?2:1 );
	   
      if ( magicNumberByteOrder!=header.byteOrder )
      {
	 cerr << "ERROR: DigeoFileHeader::read_v1 : endianness and magic number are inconsistent, endianness = " << (int)header.byteOrder << endl;
	 return false;
      }

      header.reverseByteOrder = ( cpuByteOrder!=header.byteOrder );
      stream.read( (char*)&header.version, 4 );
      stream.read( (char*)&header.nbPoints, 4 );
      stream.read( (char*)&header.descriptorSize, 4 );
      if ( header.reverseByteOrder )
      {
	 byte_inv_4( &header.version );
	 byte_inv_4( &header.nbPoints );
	 byte_inv_4( &header.descriptorSize );
      }
   }
   else
   {
      stream.seekg( 0, stream.beg );
      
      stream.read( (char*)&header.nbPoints, 4 );
      stream.read( (char*)&header.descriptorSize, 4 );
      
      header.version = 0;
      header.byteOrder = 0;
      header.reverseByteOrder = false; // endianness is unknown in v0, default is to keep data as it is 
   }
      
   if ( header.descriptorSize!=DIGEO_DESCRIPTOR_SIZE )
   {
      cerr << "ERROR: DigeoFileHeader::read_v0 : descriptor's dimension is " << header.descriptorSize << " but should be " << DIGEO_DESCRIPTOR_SIZE << endl;
      return false;
   }
   
   return true;
}

// this reading function detects the fileformat and can be used with old siftpp_tgi files
// if o_header is not null, addressed variable is filled
bool readDigeoFile( const string &i_filename, vector<DigeoPoint> &o_list, DigeoFileHeader *o_header )
{
   ifstream f( i_filename.c_str(), ios::binary );
   if ( !f ) return false;
   
   DigeoFileHeader header;
   if ( !readDigeoFileHeader( f, header ) ) return false;
   if ( o_header!=NULL ) *o_header=header;
   
   switch ( header.version )
   {
   case 0: readDigeoFile_v0( f, header, o_list ); break;
   case 1: readDigeoFile_v1( f, header, o_list ); break;
   default: cerr << "ERROR: writeDigeoFile : unkown version number " << header.version << endl; return false;
   }
   return true;
}


//----
// writing functions
//----

void writeDigeoFile_v0( std::ostream &stream, const vector<DigeoPoint> &i_list )
{
   U_INT4 nbPoints = (U_INT4)i_list.size(),
	  descriptorSize = (U_INT4)DIGEO_DESCRIPTOR_SIZE;
   
   stream.write( (char*)&nbPoints, 4 );
   stream.write( (char*)&descriptorSize, 4 );
   
   // write points
   if ( i_list.size()==0 ) return;
   const DigeoPoint *itPoint = &i_list[0];
   while ( nbPoints-- ) ( *itPoint++ ).write_v0( stream );
}

inline void writeDigeoFile_versioned_header( std::ostream &stream, U_INT4 version, bool reverseByteOrder )
{
   // this header is common to all version > 0
   unsigned char endianness = ( MSBF_PROCESSOR()?2:1 );
   if ( reverseByteOrder ) endianness=3-endianness;
   U_INT4 magic_number = ( endianness==1?DIGEO_FILEFORMAT_MAGIC_NUMBER_LSBF:DIGEO_FILEFORMAT_MAGIC_NUMBER_MSBF );
   stream.write( (char*)&magic_number, 4 );
   stream.write( (char*)&endianness, 1 );
   if ( reverseByteOrder ) byte_inv_4( &version );
   stream.write( (char*)&version, 4 );
}

void writeDigeoFile_v1( std::ostream &stream, const vector<DigeoPoint> &i_list, bool i_writeBigEndian )
{
   bool reverseByteOrder = ( MSBF_PROCESSOR()!=i_writeBigEndian );
   
   writeDigeoFile_versioned_header( stream, 1, reverseByteOrder );
   
   // same has v0
   U_INT4 nbPoints = (U_INT4)i_list.size(),
	  descriptorSize = (U_INT4)DIGEO_DESCRIPTOR_SIZE;
   stream.write( (char*)&nbPoints, 4 );
   stream.write( (char*)&descriptorSize, 4 );
   
   // write points
   if ( i_list.size()==0 ) return;
   const DigeoPoint *itPoint = &i_list[0];
   while ( nbPoints-- ) ( *itPoint++ ).write_v1( stream, reverseByteOrder );
}

bool writeDigeoFile( const string &i_filename, const vector<DigeoPoint> &i_list, U_INT4 i_version, bool i_writeBigEndian )
{
   ofstream f( i_filename.c_str(), ios::binary );
   if ( !f ) return false;
   
   switch( i_version )
   {
   case 0: writeDigeoFile_v0( f, i_list ); break;
   case 1: writeDigeoFile_v1( f, i_list, i_writeBigEndian ); break;
   default: cerr << "ERROR: writeDigeoFile : unkown version number " << i_version << endl; return false;
   };
   
   return true;
}
