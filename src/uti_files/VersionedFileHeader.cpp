#include <cstdlib>
#include <fstream>

#include "private/VersionedFileHeader.h"

#include <ctime>

using namespace std;

versioned_file_header_t g_versioned_headers_list[] = 
{
   { 3440636730, 989008845, 1, "Digeo : points of interest" },
   { 1950933035, 736118900, 1, "TracePack" },
   { 3156841558, 1452550588, 1, "Matched pair of points" },
   { 3071145667, 3287682487, 1, "Raw Im2D" },
   { 0, 0, 0, "unknown" } // designate the end of the list
};


//--------------------------------------------
// class VersionedFileHeader
//--------------------------------------------

VersionedFileHeader::VersionedFileHeader( VFH_Type i_type ):
   m_version( g_versioned_headers_list[i_type].last_handled_version )
{
   if ( MSBF_PROCESSOR() )
   {
      m_isMSBF = 1;
      m_magicNumber = g_versioned_headers_list[i_type].magic_number_MSBF;
   }
   else
   {
      m_isMSBF = 0;
      m_magicNumber = g_versioned_headers_list[i_type].magic_number_LSBF;
   }
}

VersionedFileHeader::VersionedFileHeader( VFH_Type i_type, uint32_t i_version, bool i_isMSBF ):
   m_version( i_version )
{
   if ( i_isMSBF )
   {
      m_isMSBF = 1;
      m_magicNumber = g_versioned_headers_list[i_type].magic_number_MSBF;
   }
   else
   {
      m_isMSBF = 0;
      m_magicNumber = g_versioned_headers_list[i_type].magic_number_LSBF;
   }
   if ( i_isMSBF!=MSBF_PROCESSOR() ) byte_inv_4( &i_version );
}

bool VersionedFileHeader::read_raw( std::istream &io_istream )
{
	char c;
	io_istream.get(c);
	if ( c!=-70 ) return false;

	io_istream.read( (char*)this, sizeof(VersionedFileHeader) );
	if ( m_isMSBF!=0 && m_isMSBF!=1 ) return false;

	if ( MSBF_PROCESSOR()!=isMSBF() ) byte_inv_4( &m_version );

	return true;
}

bool VersionedFileHeader::read_unknown( std::istream &io_istream, VFH_Type &o_id )
{
	// save stream state and position
	streampos pos   = io_istream.tellg();
	ios_base::iostate state = io_istream.rdstate();

	bool isMSBF_; // from magic number
	if ( !read_raw( io_istream ) ||
	     !typeFromMagicNumber(m_magicNumber, o_id, isMSBF_) ||
	     isMSBF_!=isMSBF() )
	{
		// reading failed, rewind to starting point and reset error flags
		m_magicNumber = m_version = 0;
		m_isMSBF = ( MSBF_PROCESSOR()?1:0 );
		io_istream.seekg( pos );
		io_istream.setstate( state );
		return false;
	}
	return true;
}

bool VersionedFileHeader::read_known( VFH_Type i_type, std::istream &io_istream )
{
	// save stream state and position
	streampos pos   = io_istream.tellg();
	ios_base::iostate state = io_istream.rdstate();

	if ( !read_raw(io_istream) || m_magicNumber!=(isMSBF()?g_versioned_headers_list[i_type].magic_number_MSBF:g_versioned_headers_list[i_type].magic_number_LSBF) )
	{
		// reading failed, rewind to starting point and reset error flags
		m_magicNumber = m_version = 0;
		m_isMSBF = ( MSBF_PROCESSOR()?1:0 );
		io_istream.seekg(pos);
		io_istream.setstate(state);
		return false;
	}

	return true;
}

void VersionedFileHeader::write( std::ostream &io_ostream ) const
{   
   io_ostream.put(-70);
   io_ostream.write( (char*)this, sizeof(VersionedFileHeader) );
}


//--------------------------------------------
// related functions
//--------------------------------------------

bool typeFromMagicNumber( uint32_t i_magic, VFH_Type &o_type, bool &o_isMSBF )
{
	versioned_file_header_t *itHeader = g_versioned_headers_list;
	for ( int i=0; i<nbVersionedTypes(); i++ )
	{
		if ( i_magic==itHeader->magic_number_MSBF )
		{
			o_type = (VFH_Type)i;
			o_isMSBF = true;
			return true;
		}
		if ( i_magic==itHeader->magic_number_LSBF )
		{
			o_type = (VFH_Type)i;
			o_isMSBF = false;
			return true;
		}
		itHeader++;
	}
	return false;
}

// generate a new random number than is not equal to itself in reversed byte order and that is not already used
void generate_new_magic_number( uint32_t &o_direct, uint32_t &o_reverse )
{
	srand( (unsigned int)time(NULL) );
	uint32_t magic, magic_inverse;
	unsigned char *m = (unsigned char*)&magic;
	versioned_file_header_t *itHeader;
	while ( true ){
		m[0] = rand()%256;
		m[1] = rand()%256;
		m[2] = rand()%256;
		m[3] = rand()%256;
		magic_inverse = magic;
		byte_inv_4( &magic_inverse );
		if ( magic_inverse==magic ) continue;
		itHeader = g_versioned_headers_list;
		while ( itHeader->magic_number_MSBF!=0 ){
			if ( itHeader->magic_number_MSBF==magic ||
			     itHeader->magic_number_LSBF==magic )
				continue;
			itHeader++;
		}
		o_direct  = magic;
		o_reverse = magic_inverse;
		return;
	}
}

VFH_Type versionedFileType( const string &i_filename )
{
   ifstream f( i_filename.c_str(), ios::binary );
   VersionedFileHeader header;
   VFH_Type fileType;
   if ( !header.read_unknown( f, fileType ) ) return VFH_Unknown;
   return fileType;
}

#ifdef NO_ELISE
   void byte_inv_4(void * t)
   {
      std::swap( ((char *) t)[0], ((char *) t)[3] );
      std::swap( ((char *) t)[1], ((char *) t)[2] );
   }

   bool MSBF_PROCESSOR()
   {
       static bool init = false;
       static bool res  = true;

       if ( !init )
       {
	   uint16_t ui2=0;
	   char * c = (char *) &ui2;
	   c[0] = 0;
	   c[1] = 1;
	   res  = (ui2 == 1);
	   init = true;
      }
      
      return res;
   }
#endif
