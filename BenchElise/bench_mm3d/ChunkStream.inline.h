//--------------------------------------------
// class ChunkStream::Item
//--------------------------------------------

ChunkStream::Item::~Item(){}

template <class T> T & ChunkStream::Item::specialize(){ return *((T*)this); }


//--------------------------------------------
// class ChunkStream::FileItem::Chunk
//--------------------------------------------

bool ChunkStream::FileItem::Chunk::operator ==( const Chunk &i_b ) const { return m_filename==i_b.m_filename && m_offset==i_b.m_offset && m_length==i_b.m_length; }

bool ChunkStream::FileItem::Chunk::operator !=( const Chunk &i_b ) const { return !( (*this)==i_b ); }


//--------------------------------------------
// class ChunkStream::FileItem
//--------------------------------------------

bool ChunkStream::FileItem::operator !=( const FileItem &i_b ) const { return !( (*this)==i_b ); }


//--------------------------------------------
// class ChunkStream
//--------------------------------------------

bool ChunkStream::prepare_next_read(){ return ( m_remaining>chunkHeaderSize || open_next_input_file() ); }

bool ChunkStream::prepare_next_write(){ return ( m_remaining>chunkHeaderSize || open_next_output_file() ); }

U_INT8 ChunkStream::maxFileSize() const { return m_maxFileSize; }

void ChunkStream::setReverseByteOrder( bool i_isReverse ){ m_reverseByteOrder=i_isReverse; }

bool ChunkStream::getReverseByteOrder() const{ return m_reverseByteOrder; }


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

void write_uint8( U_INT8 i_ui8, bool i_reverseByteOrder, std::ostream &io_fOut )
{
	if ( i_reverseByteOrder ) byte_inv_8( &i_ui8 );
   io_fOut.write( (char*)&i_ui8, 8 );
}

void read_uint8( std::istream &io_fIn, bool i_reverseByteOrder, U_INT8 &i_ui8 )
{
   io_fIn.read( (char*)&i_ui8, 8 );
   if ( i_reverseByteOrder ) byte_inv_8( &i_ui8 );
}
