#include "ChunkStream.h"

#include <cstdlib>
#include <sstream>

using namespace std;

//--------------------------------------------
// class ChunkStream::FileItem::Chunk
//--------------------------------------------

ChunkStream::FileItem::Chunk::Chunk( const cElFilename &i_filename, streampos i_offset, U_INT8 i_length ):
	m_filename( i_filename ),
	m_offset( i_offset ),
	m_length( i_length ){}


//--------------------------------------------
// class ChunkStream::FileItem
//--------------------------------------------

ChunkStream::FileItem::FileItem( const cElFilename &i_storedFilename, U_INT8 i_length ):
	m_storedFilename( i_storedFilename ),
	m_length( i_length )
{
}
	
bool ChunkStream::FileItem::isFileItem() const { return true; }

bool ChunkStream::FileItem::copyToFile( const cElFilename &i_dstFilename ) const
{
	if ( !i_dstFilename.m_path.exists() && !i_dstFilename.m_path.create() ){
		#ifdef __DEBUG_CHUNK_STREAM
			cerr << RED_DEBUG_ERROR << "ChunkStream::FileItem::copyToFile: create directory [" << i_dstFilename.m_path.str() << "]" << endl;
			exit(EXIT_FAILURE);
		#endif
	}
	
	ofstream dst( i_dstFilename.str_unix().c_str(), ios::binary );

	if ( !dst ){
		#ifdef __DEBUG_CHUNK_STREAM
			cerr << RED_DEBUG_ERROR << "ChunkStream::FileItem::copyToFile: cannot open destination file [" << i_dstFilename.str_unix() << "]" << endl;
			exit(EXIT_FAILURE);
		#endif
		return false;
	}
	
	ifstream src;
	cElFilename currentSrcFile;
	list<Chunk>::const_iterator itChunk = m_chunks.begin();
	while ( itChunk!=m_chunks.end() )
	{
		// set Chunk's file as input
		if ( currentSrcFile!=itChunk->m_filename ){
			if ( src.is_open() ) src.close();
			currentSrcFile = itChunk->m_filename;
			src.open( currentSrcFile.str_unix().c_str(), ios::binary );
			if ( !src ){
				#ifdef __DEBUG_CHUNK_STREAM
					cerr << RED_DEBUG_ERROR << "ChunkStream::FileItem::copyToFile: unable to open source file [" << currentSrcFile.str_unix() << "]" << endl;
				#endif
				return false;
			}
		}
		// move to Chunk's offset inside source file
		src.seekg( itChunk->m_offset );
		if ( !stream_copy( src, dst, itChunk->m_length ) ) return false;
		itChunk++;
	}
	return true;
}

bool ChunkStream::FileItem::operator ==( const FileItem &i_b ) const
{
	if ( m_storedFilename!=i_b.m_storedFilename || m_length!=i_b.m_length || m_chunks.size()!=i_b.m_chunks.size() ) return false;
	list<Chunk>::const_iterator itA = m_chunks.begin(),
	                            itB = i_b.m_chunks.begin();
	while ( itA!=m_chunks.end() )
		if ( *itA++!=*itB++ ) return false;
	return true;
}


//--------------------------------------------
// class ChunkStream::BufferItem
//--------------------------------------------

ChunkStream::BufferItem::BufferItem( unsigned char i_type ):
	m_type(i_type)
{
	#ifdef __DEBUG_CHUNK_STREAM
		if ( (i_type&dataHeaderFlag)!=0 ){
			cout << RED_DEBUG_ERROR << "ChunkStream::BufferItem::BufferItem: constructing a BufferItem with the FileItem type " << (int)i_type << endl;
			exit(EXIT_FAILURE);
		}
	#endif
}

bool ChunkStream::BufferItem::isFileItem() const { return false; }


//--------------------------------------------
// class ChunkStream
//--------------------------------------------

ChunkStream::ChunkStream( const cElFilename &i_filename, U_INT8 i_maxFileSize, bool i_reverseByteOrder ):
	m_filename( i_filename ),
   m_maxFileSize( i_maxFileSize ),
   m_reverseByteOrder( i_reverseByteOrder ){}

bool ChunkStream::open_next_output_file()
{
	if ( !isWriteOpen() ){
		#ifdef __DEBUG_CHUNK_STREAM
			cerr << RED_DEBUG_ERROR << "ChunkStream::open_next_output_file: stream is not open for writing" << endl;
			exit(EXIT_FAILURE);
		#endif
		return false;
	}
	m_outputStream.close();
	m_outputStream.open( getFilename( ++m_currentFileIndex ).str_unix().c_str(), ios::binary );
	m_remaining = m_maxFileSize;
	return true;
}

bool ChunkStream::open_next_input_file()
{
	if ( !isReadOpen() ){
		#ifdef __DEBUG_CHUNK_STREAM
			cerr << RED_DEBUG_ERROR << "ChunkStream::open_next_input_file: stream is not open for reading" << endl;
			exit(EXIT_FAILURE);
		#endif
		return false;
	}
	if ( isEndOfStream() ){
		#ifdef __DEBUG_CHUNK_STREAM
			cerr << RED_DEBUG_ERROR << "ChunkStream::open_next_input_file: end of stream is already reached" << endl;
			exit(EXIT_FAILURE);
		#endif
		return false;
	}
	const cElFilename filename = getFilename( ++m_currentFileIndex );
	m_inputStream.close();
	m_remaining = filename.getSize();
	m_inputStream.open( filename.str_unix().c_str(), ios::binary );
	return true;
}

bool ChunkStream::readOpen( U_INT4 i_iStartIndex, U_INT8 i_startOffset )
{	
	if ( isOpen() ) close();
	
	m_lastFileIndex = getNbFiles();
	if ( m_lastFileIndex==0 ){
		m_currentFileIndex = 0;
		m_remaining = 0;
		return true;
	}
	m_lastFileIndex--;
	
	if ( i_iStartIndex>m_lastFileIndex ){
		#ifdef __DEBUG_CHUNK_STREAM
			cerr << RED_DEBUG_ERROR << "ChunkStream::readOpen: trying to start at file index " << i_iStartIndex << " but last file index is " << m_lastFileIndex << endl;
			exit(EXIT_FAILURE);
		#endif
		return false;
	}
	
	for ( U_INT4 iFile=0; iFile<=i_iStartIndex; iFile++ )
		if ( !getFilename(iFile).exists() ){
			#ifdef __DEBUG_CHUNK_STREAM
				cerr << RED_DEBUG_ERROR << "ChunkStream::readOpen: first file is said to be [" << getFilename(i_iStartIndex).str_unix() << " but file [" << getFilename(iFile).str_unix()
				     << "] does not exist" << endl;
				exit(EXIT_FAILURE);
			#endif
			return false;
		}

	cElFilename startFile = getFilename(i_iStartIndex);
	U_INT8 fileSize = startFile.getSize();
	if ( i_startOffset>fileSize ){
		#ifdef __DEBUG_CHUNK_STREAM
			cerr << RED_DEBUG_ERROR << "ChunkStream::readOpen: an offset of size " << i_startOffset << " is too high for a file of size " << startFile.getSize() << endl;
			exit(EXIT_FAILURE);
		#endif
		return false;
	}
	m_remaining = fileSize-i_startOffset;
	
	m_inputStream.open( startFile.str_unix().c_str(), ios::binary );
	if ( !m_inputStream ){
		#ifdef __DEBUG_CHUNK_STREAM
			cerr << RED_DEBUG_ERROR << "ChunkStream::readOpen: cannot open file [" << startFile.str_unix() << "]" << endl;
			exit(EXIT_FAILURE);
		#endif
		return false;
	}
	m_currentFileIndex = i_iStartIndex;
	m_inputStream.seekg( (streampos)i_startOffset );
	
	return true;
}

bool ChunkStream::writeOpen()
{
	if ( isOpen() ) close();
	m_currentFileIndex = getNbFiles();
	cElFilename filename;
	if ( m_currentFileIndex==0 ){
		// the stream is empty, creating first file
		filename = getFilename(0);
		m_remaining = m_maxFileSize;
		m_outputStream.open( filename.str_unix().c_str(), ios::binary );
	}
	else{
		filename = getFilename( --m_currentFileIndex );
		if ( filename.getSize()+chunkHeaderSize>=m_maxFileSize ){
			// last stream file is already full, use next the next one
			m_remaining = m_maxFileSize;
			filename = getFilename( ++m_currentFileIndex );
			m_outputStream.open( filename.str_unix().c_str(), ios::binary );
		}
		else{
			// there is room left in last file, open it in append mode
			m_remaining = m_maxFileSize-filename.getSize(); // which is enough to put at least one chunk
			m_outputStream.open( filename.str_unix().c_str(), ios::binary|ios::app );
		}
	}
	
	#ifdef __DEBUG_CHUNK_STREAM
		if ( !m_outputStream ){
			cerr << RED_DEBUG_ERROR << "ChunkStream::writeOpen: cannot open last stream file [" << filename.str_unix() << "]" << endl;
			exit(EXIT_FAILURE);
		}
	#endif
	
	return !m_outputStream.fail();
}

void ChunkStream::close()
{
	if ( m_outputStream.is_open() ) m_outputStream.close();
	if ( m_inputStream.is_open() ) m_inputStream.close();
}

bool ChunkStream::isEndOfStream() const { return m_currentFileIndex==m_lastFileIndex && m_remaining==0; }

bool ChunkStream::isReadOpen() const { return m_inputStream.is_open(); }

bool ChunkStream::isWriteOpen() const { return m_outputStream.is_open(); }

bool ChunkStream::isOpen() const { return ( isReadOpen()||isWriteOpen() ); }

bool ChunkStream::write_file( const cElFilename &i_srcFilename, const cElFilename &i_toStoreFilename )
{
	FileItem item;
	return write_file( i_srcFilename, i_toStoreFilename, item );
}

int g_iwrite;

void ChunkStream::write_header( unsigned char i_type, bool i_hasMore, U_INT8 i_chunkSize )
{
	#ifdef __DEBUG_CHUNK_STREAM_OUTPUT_HEADERS
		cout << "--- write_header file=[" << getFilename(m_currentFileIndex).str_unix() << "] offset=" << m_outputStream.tellp() << " type=" << (int)i_type << " hasMore="
			  << (i_hasMore?"true":"false") << " chunkSize=" << i_chunkSize << " remaining=" << m_remaining << endl;
	#endif
	
	#ifdef __CHUNK_STREAM
		if ( !isWriteOpen() ){ cerr << RED_DEBUG_ERROR << "ChunkStream::write_header: stream is not writeOpen" << endl; exit(EXIT_FAILURE); }
		if ( m_remaining<=chunkHeaderSize ){
			cerr << RED_DEBUG_ERROR << "ChunkStream::write_header: there's no room for a chunk in output file [" << getFilename(m_currentFileIndex).str_unix() << "]" << endl;
			exit(EXIT_FAILURE);
		}
	#endif
	m_outputStream.put( (char)i_type );
	m_outputStream.put( i_hasMore?1:0 );
	write_uint8( i_chunkSize, m_reverseByteOrder, m_outputStream );
}

bool ChunkStream::write_buffer( unsigned char i_type, const std::vector<char> &i_buffer )
{
	U_INT8 length = i_buffer.size();
	const char *itBuffer = i_buffer.data();
	while ( length!=0 ){
		if ( !prepare_next_write() ) return false;
		U_INT8 sizeToWrite = std::min( m_remaining-chunkHeaderSize, length );
		length -= sizeToWrite;
		write_header( i_type, length!=0, sizeToWrite );
		m_outputStream.write( itBuffer, sizeToWrite );
		itBuffer += sizeToWrite;
		
		#ifdef __DEBUG_CHUNK_STREAM
			if ( m_remaining<sizeToWrite+chunkHeaderSize ){
				cerr << RED_DEBUG_ERROR << "ChunkStream::write_buffer: " << sizeToWrite+chunkHeaderSize << " but " << m_remaining << " remained" << endl;
				exit(EXIT_FAILURE);
			}
		#endif
		
		m_remaining -= sizeToWrite+chunkHeaderSize;
	}
	
	return true;
}

bool ChunkStream::write_file( const cElFilename &i_srcFilename, const cElFilename &i_toStoreFilename, FileItem &o_fileItem )
{
	U_INT8 fileSize = i_srcFilename.getSize();	
	unsigned char chunkType = dataHeaderFlag;
	
	o_fileItem.m_storedFilename = i_toStoreFilename;
	o_fileItem.m_length = fileSize;
	o_fileItem.m_chunks.clear();

	ifstream src( i_srcFilename.str_unix().c_str(), ios::binary );
	if ( !src ){
		#ifdef __DEBUG_CHUNK_STREAM
			cerr << RED_DEBUG_ERROR << "ChunkStream::write_file: cannot open source file [" << i_srcFilename.str_unix() << "]" << endl;
			exit(EXIT_FAILURE);
		#endif
		return false;
	}
	
	// write filename in separate chunks
	string toStoredFilename = i_toStoreFilename.str_unix();
	vector<char> buffer( (size_t)string_raw_size(toStoredFilename) );
	char *itBuffer = buffer.data();
	string_to_raw_data( toStoredFilename, m_reverseByteOrder, itBuffer );
	write_buffer( chunkType|dataFilenameHeaderFlag, buffer );
	
	// write file's data
	while ( fileSize!=0 ){
		if ( !prepare_next_write() ) return false;
		U_INT8 sizeToWrite = std::min( m_remaining-chunkHeaderSize, fileSize );
		fileSize -= sizeToWrite;
		write_header( chunkType, fileSize!=0, sizeToWrite );
		o_fileItem.m_chunks.push_back( FileItem::Chunk( getFilename( m_currentFileIndex ), m_outputStream.tellp(), sizeToWrite ) );
		stream_copy( src, m_outputStream, sizeToWrite );
		
		#ifdef __DEBUG_CHUNK_STREAM
			if ( m_remaining<sizeToWrite+chunkHeaderSize ){
				cerr << RED_DEBUG_ERROR << "ChunkStream::write_file: " << sizeToWrite+chunkHeaderSize << " but " << m_remaining << " remained" << endl;
				exit(EXIT_FAILURE);
			}
		#endif
		
		m_remaining -= sizeToWrite+chunkHeaderSize;
	}
	
	return true;
}

bool ChunkStream::remove()
{
   close();
	U_INT4 iFilename = 0;
	cElFilename filename = getFilename(iFilename++);
	while ( filename.exists() ){
		if ( !filename.remove() ){
			#ifdef __DEBUG_CHUNK_STREAM
				cerr << RED_DEBUG_ERROR << "ChunkStream::clear:: cannot remove file [" << filename.str_unix() << "]" << endl;
				exit(EXIT_FAILURE);
			#endif
			return false;
		}
		filename = getFilename( iFilename++ );
	}
	return true;
}

cElFilename ChunkStream::getFilename( U_INT4 i_fileIndex ) const
{
	if ( i_fileIndex==0 ) return m_filename;
	stringstream ss;
	ss << m_filename.m_basename << '.' << i_fileIndex;
	return cElFilename( m_filename.m_path, ss.str() );
}

U_INT4 ChunkStream::getNbFiles() const
{
	U_INT4 iFilename = 0;
	cElFilename filename = getFilename(0);
	while ( filename.exists() ) filename=getFilename( ++iFilename );
	return iFilename;
}

bool ChunkStream::read_header( unsigned char &o_type, bool &o_hasMore, U_INT8 &o_chunkSize )
{
	if ( !isReadOpen() ){
		#ifdef __DEBUG_CHUNK_STREAM
			cerr << RED_DEBUG_ERROR << "ChunkStream::read_header: stream is not readOpen" << endl;
			exit(EXIT_FAILURE);
		#endif
		return false;
	}
	o_type = m_inputStream.get();
	o_hasMore = (m_inputStream.get()==1);
	read_uint8( m_inputStream, m_reverseByteOrder, o_chunkSize );
	
	#ifdef __DEBUG_CHUNK_STREAM
		if ( o_chunkSize>1e8 ){
			cerr << RED_DEBUG_ERROR << "ChunkStream::read_header: chunkSize seems a bit high = " << o_chunkSize << endl;
			exit(EXIT_FAILURE);
		}
	#endif

	#ifdef __DEBUG_CHUNK_STREAM_OUTPUT_HEADERS
		cout << "--- read_header file=[" << getFilename(m_currentFileIndex).str_unix() << "] offset=" << (U_INT8)m_inputStream.tellg()-chunkHeaderSize << " type=" << (int)o_type << " hasMore="
			  << (o_hasMore?"true":"false") << " chunkSize=" << o_chunkSize << endl;
	#endif
	
	return true;
}

bool ChunkStream::read_buffer( unsigned char i_itemType, vector<char> &o_buffer )
{
	// read all chunk of the buffer
	unsigned char chunkType;
	bool hasMore;
	U_INT8 totalSize = 0,
	       chunkSize;
	list<vector<char> > buffers;
	do{
		if ( !prepare_next_read() )	return false;
		if ( !read_header( chunkType, hasMore, chunkSize ) ) return false;
		// check types are the same
		if ( chunkType!=i_itemType ){
			#ifdef __DEBUG_CHUNK_STREAM
				cerr << RED_DEBUG_ERROR << "ChunkStream::read_buffer: a chunk of type " << (int)chunkType << " has been found but a chunk of type " << (int)i_itemType << " is expected" << endl;
				exit(EXIT_FAILURE);
			#endif
			return false;
		}

		buffers.push_back( vector<char>() );
		vector<char> &buffer = buffers.back();
		buffer.resize( (size_t)chunkSize );
		m_inputStream.read( buffer.data(), chunkSize );

		#ifdef __DEBUG_CHUNK_STREAM
			if ( m_inputStream.gcount()<0 || (U_INT8)m_inputStream.gcount()!=chunkSize ){
				cerr << RED_DEBUG_ERROR << "ChunkStream::read_buffer: tried to read " << chunkSize << " byte but " << m_inputStream.gcount() << " were read correctly" << endl;
				exit(EXIT_FAILURE);
			}
			if ( m_remaining<chunkSize+chunkHeaderSize ){
				cerr << RED_DEBUG_ERROR << "ChunkStream::read_buffer: " << chunkSize+chunkHeaderSize << " but " << m_remaining << " remained" << endl;
				exit(EXIT_FAILURE);
			}
		#endif
		
		m_remaining -= chunkSize+chunkHeaderSize;		
		totalSize += chunkSize;
	} while ( hasMore );
	
	// create a single buffer with all chunks
	o_buffer.resize( (size_t)totalSize );
	list<vector<char> >::const_iterator itBuffer = buffers.begin();
	char *itItemData = o_buffer.data();
	while ( itBuffer!=buffers.end() ){
		const vector<char> &buffer = *itBuffer++;
		memcpy( itItemData, buffer.data(), buffer.size() );
		itItemData += buffer.size();
	}
	
	return true;
}

ChunkStream::Item * ChunkStream::read()
{
	if ( !prepare_next_read() )	return NULL;
	unsigned char itemType = m_inputStream.peek();
	if ( (itemType&dataHeaderFlag)==0 ){
		// current item is a BufferItem
		BufferItem *pItem = new BufferItem( itemType );
		if ( !read_buffer( itemType, pItem->m_buffer ) ) return NULL;
		return pItem;
	}
	else{
		// read stored filename
		vector<char> buffer;
		if ( !read_buffer( itemType, buffer ) ) return NULL;
		#ifdef __DEBUG_CHUNK_STREAM
			if ( (itemType&dataFilenameHeaderFlag)==0 ){
				cerr << RED_DEBUG_ERROR << "ChunkStream::read: next chunk is data-file chunk whereas a filename-file chunk is expected" << endl;
				exit(EXIT_FAILURE);
			}
		#endif
		const char *itRaw = (const char *)buffer.data();
		string filename_str;
		string_from_raw_data( itRaw, m_reverseByteOrder, filename_str );
		
		FileItem *pItem = new FileItem( cElFilename(filename_str), 0 );
		if ( isEndOfStream() ) return pItem;
		if ( !prepare_next_read() ) return NULL;
		
		U_INT8 chunkSize;
		unsigned char expectedType = (itemType^dataFilenameHeaderFlag),
		              chunkType = (unsigned char)m_inputStream.peek();
		bool hasMore = true;
		if ( chunkType==expectedType ){
			while ( hasMore ){
				if ( !prepare_next_read() ) return NULL;
				if ( !read_header( chunkType, hasMore, chunkSize ) ) return NULL;				
				if ( chunkType!=expectedType ){
					#ifdef __DEBUG_CHUNK_STREAM
						cerr << RED_DEBUG_ERROR << "ChunkStream::read: a chunk of type " << (int)chunkType << " has been found whereas a type " << (int)expectedType << " is expected" << endl;
						exit(EXIT_FAILURE);
					#endif
					return NULL;
				}
				
				pItem->m_chunks.push_back( FileItem::Chunk( getFilename(m_currentFileIndex), m_inputStream.tellg(), chunkSize ) );
				pItem->m_length += chunkSize;
				m_inputStream.seekg( chunkSize, ios::cur );
				
				#ifdef __DEBUG_CHUNK_STREAM
					if ( m_remaining<chunkSize+chunkHeaderSize ){
						cerr << RED_DEBUG_ERROR << "ChunkStream::read: " << chunkSize+chunkHeaderSize << " but " << m_remaining << " remained" << endl;
						exit(EXIT_FAILURE);
					}
				#endif
				
				m_remaining -= chunkSize+chunkHeaderSize;
			}
		}
		return pItem;
	}
}

bool ChunkStream::read( U_INT4 i_iStartIndex, U_INT8 i_startOffset, std::list<Item*> &o_items )
{
	clear_item_list( o_items );
	if ( !readOpen( i_iStartIndex, i_startOffset ) ) return false;
	Item *pItem;
	while ( !isEndOfStream() ){
		if ( ( pItem=read() )==NULL ) return false;
		o_items.push_back( pItem );
	}
	return true;
}


//--------------------------------------------
// related functions
//--------------------------------------------

bool stream_copy( istream &io_src, ostream &io_dst, U_INT8 i_length )
{
   #ifdef __DEBUG_CHUNK_STREAM
      if ( !io_src ){
			cerr << RED_DEBUG_ERROR << "stream_copy: io_src is not ready for reading" << endl;
			exit(EXIT_FAILURE);
      }
      if ( !io_dst ){
			cerr << RED_DEBUG_ERROR << "stream_copy: io_dst is not ready for writing" << endl;
			exit(EXIT_FAILURE);
      }
   #endif
   
   const unsigned int buffer_size = 1000000;
   U_INT8 remaining = i_length;
   vector<char> buffer( buffer_size );
   while ( !io_src.eof() && remaining ){
      if ( buffer_size>remaining )
			io_src.read( buffer.data(), remaining );
      else
			io_src.read( buffer.data(), buffer_size );
      streamsize nbRead = io_src.gcount();
      if ( nbRead<0 ){	 
			#ifdef __DEBUG_CHUNK_STREAM
				cerr << RED_DEBUG_ERROR << "write_file: unable to read in input stream" << endl;
				exit(EXIT_FAILURE);
			#endif
			return false;
      }
      io_dst.write( buffer.data(), nbRead );
      remaining -= (unsigned int)nbRead;
   }
   
   #ifdef __DEBUG_CHUNK_STREAM
      if ( remaining!=0 ){
			cerr << RED_DEBUG_ERROR << "write_file: " << remaining << " bytes need to be read but end-of-file is reached" << endl;
			exit(EXIT_FAILURE);
      }
   #endif
   
   return remaining==0;
}

void clear_item_list( list<ChunkStream::Item*> io_items )
{
	list<ChunkStream::Item*>::iterator itItem = io_items.begin();
	while ( itItem!=io_items.end() ){
		#ifdef __DEBUG_CHUNK_STREAM
			if ( *itItem==NULL ){ cerr << RED_DEBUG_ERROR << "clear_item_list: trying to delete a NULL pointer" << endl; exit(EXIT_FAILURE); }
		#endif
		delete *itItem++;
	}
	io_items.clear();
}
