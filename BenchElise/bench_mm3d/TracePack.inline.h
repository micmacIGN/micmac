// this file is supposed to be included only in TracePack.h

//--------------------------------------------
// class FileChunk
//--------------------------------------------

FileChunk::FileChunk(){}

FileChunk::FileChunk( FileChunkType i_type, const std::vector<char> &i_rawData ):
	m_type(i_type),
	m_contentSize(i_rawData.size()),
	m_rawData(i_rawData)
{}
   
FileChunk::FileChunk( const string &i_filename, bool i_hasMore, U_INT4 i_dataSize ):
   m_type(FCT_Data),
   m_contentSize(i_dataSize+dataHeaderSize),
   m_isFirst(true),
   m_filename(i_filename),
   m_filenameRawSize( string_raw_size(i_filename) ),
   m_hasMore(i_hasMore),
   m_dataSize(i_dataSize)
{
	#ifdef __DEBUG_TRACE_PACK
		if ( m_dataSize<string_raw_size(m_filename) )
		{
			cerr << RED_DEBUG_ERROR << "FileChunk::FileChunk: m_dataSize = " << m_dataSize << " < string_raw_size(m_filename) = " << string_raw_size(m_filename) << endl;
			exit(EXIT_FAILURE);
		}
	#endif
}

FileChunk::FileChunk( bool i_hasMore, U_INT4 i_dataSize ):
   m_type(FCT_Data),
   m_contentSize(i_dataSize+dataHeaderSize),
   m_isFirst(false),
   m_hasMore(i_hasMore),
   m_dataSize(i_dataSize)
{
}

void FileChunk::_set_dataChunk( U_INT i_fullSize )
{
   #ifdef __DEBUG_TRACE_PACK
      if ( i_fullSize<=(headerSize+dataHeaderSize) )
      {
			cerr << RED_DEBUG_ERROR << "FileChunk::FileChunk: i_fullSize = " << i_fullSize << " <= headerSize+dataHeaderSize = " << headerSize+dataHeaderSize << endl;
			exit( EXIT_FAILURE );
      }
   #endif
   m_type = FCT_Data;
   m_contentSize = i_fullSize-headerSize;
   m_dataSize = i_fullSize-(headerSize+dataHeaderSize);
}

void FileChunk::set_dataChunk( U_INT i_fullSize )
{
   _set_dataChunk( i_fullSize );
   m_isFirst = false;
}

void FileChunk::set_dataChunk( U_INT i_fullSize, const std::string &i_filename )
{
   _set_dataChunk( i_fullSize );
   m_isFirst = true;
   m_filename = i_filename;
}

bool FileChunk::read_chunk( std::istream &io_in )
{
   io_in.read( m_rawData.data(), m_rawData.size() );
   return true;
}

unsigned int FileChunk::fullSize() const
{
   unsigned int res = m_contentSize+5; // 5 = m_size's size(4) + m_type's size(1)
   return res;
}

U_INT4 FileChunk::onDiskData() const
{	
   #ifdef __DEBUG_TRACE_PACK
      if ( m_type!=FCT_Data )
      {
			cerr << RED_DEBUG_ERROR << "FileChunk::dataChunk_copy: chunk is not of type FCT_Data" << endl;
			exit(EXIT_FAILURE);
      }
      if ( m_isFirst && m_filenameRawSize>m_dataSize )
      {
			cerr << RED_DEBUG_ERROR << "FileChunk::dataChunk_copy: m_filenameRawSize>m_dataSize" << endl;
			exit(EXIT_FAILURE);
      }
   #endif
	return ( m_isFirst?m_dataSize-m_filenameRawSize:m_dataSize );
}

U_INT4 FileChunk::toSkip() const
{
	if ( m_type==FCT_Data ) return onDiskData();
	return 0;
}


//--------------------------------------------
// class TracePack::Registry::Item
//--------------------------------------------

inline TracePack::Registry::Item::Item():m_date(cElDate::NoDate){}
	 
TracePack::Registry::Item::Item( const cElFilename &i_filename,
                                 TD_Type i_type,
                                 const cElDate &i_date,
                                 mode_t i_rights,
                                 streampos i_dataOffset,
                                 unsigned int i_dataSize ):
   m_filename( i_filename ),
   m_type( i_type ),
   m_date( i_date ),
   m_rights( i_rights ),
   m_dataOffset( i_dataOffset ),
   m_dataSize( i_dataSize )
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


//--------------------------------------------
// related functions
//--------------------------------------------

void write_uint4( U_INT4 i_ui4, bool i_reverseByteOrder, std::ostream &io_fOut )
{
   if ( i_reverseByteOrder ) byte_inv_4( &i_ui4 );
   io_fOut.write( (char*)&i_ui4, 4 );
}

void read_uint4( std::istream &io_fIn, bool i_reverseByteOrder, U_INT4 &i_ui4 )
{
   io_fIn.read( (char*)&i_ui4, 4 );
   if ( i_reverseByteOrder ) byte_inv_4( &i_ui4 );
}

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
