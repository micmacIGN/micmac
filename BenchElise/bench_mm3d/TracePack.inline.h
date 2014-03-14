// this file is supposed to be included only in TracePack.h

//--------------------------------------------
// class TracePack::Registry::Item
//--------------------------------------------

inline TracePack::Registry::Item::Item():m_date(cElDate::NoDate){}
	 
TracePack::Registry::Item::Item( const cElFilename &i_filename,
                                 TD_Type i_type,
                                 const cElDate &i_date,
                                 mode_t i_rights,
	                              U_INT8 i_dataLength,
		                           const ChunkStream::FileItem &i_data ):
   m_filename( i_filename ),
   m_type( i_type ),
   m_date( i_date ),
   m_rights( i_rights ),
   m_dataLength( i_dataLength ),
   m_data( i_data )
{
}

bool TracePack::Registry::Item::hasData() const
{
   return ( m_type==TracePack::Registry::TD_State ||
	         m_type==TracePack::Registry::TD_Added ||
	         m_type==TracePack::Registry::TD_Modified );
}


//--------------------------------------------
// class TracePack::Registry
//--------------------------------------------

size_t TracePack::Registry::size() const { return m_items.size(); }


//--------------------------------------------
// class TracePack
//--------------------------------------------

unsigned int TracePack::nbStates() const { return m_registries.size(); }

const cElDate & TracePack::date() const { return m_date; }

U_INT4 TracePack::getNbPackFiles() const { return ChunkStream( m_filename, maxPackFileSize, true ).getNbFiles(); }

cElFilename TracePack::packFilename( const U_INT4 &i_iFile ) const { return ChunkStream( m_filename, maxPackFileSize, true ).getFilename(i_iFile); }

const list<cElFilename> & TracePack::getIgnoredFiles() const { return m_ignoredFiles; }

const list<ctPath> & TracePack::getIgnoredDirectories() const { return m_ignoredDirectories; }

void TracePack::setAnchor( const ctPath &i_path ) { m_anchor=i_path; }


//--------------------------------------------
// related functions
//--------------------------------------------

void write_string( const std::string &i_str, bool i_reverseByteOrder, std::ostream &io_fOut )
{
   U_INT4 len = i_str.length();
   if ( i_reverseByteOrder ) byte_inv_4( &len );
   io_fOut.write( (char*)&len, 4 );
   io_fOut.write( i_str.c_str(), i_str.length() );
}

void read_string( std::istream &io_fIn, bool i_reverseByteOrder, std::string &o_str )
{
   U_INT4 len;
   io_fIn.read( (char*)&len, 4 );
   if ( i_reverseByteOrder ) byte_inv_4( &len );
   char *buffer = new char[len];
   io_fIn.read( buffer, len );
   o_str.assign( buffer, len );
   delete [] buffer;
}

// uint8<->raw
void uint8_to_raw_data( const U_INT8 &i_v, bool i_reverseByteOrder, char *&o_rawData )
{
   memcpy( o_rawData, &i_v, 8 );
   if ( i_reverseByteOrder ) byte_inv_8( &o_rawData );
   o_rawData += 8;
}

void uint8_from_raw_data( const char *&io_rawData, bool i_reverseByteOrder, U_INT8 &o_v )
{
   memcpy( &o_v, io_rawData, 8 );
   if ( i_reverseByteOrder ) byte_inv_8( &o_v );
   io_rawData += 8;
}
