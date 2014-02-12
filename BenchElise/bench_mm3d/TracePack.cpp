#include "TracePack.h"

#include "StdAfx.h"

#include "general/cElCommand.h"

using namespace std;

#ifdef __DEBUG_TRACE_PACK
   const string __difference_illegal_item_message = "TracePack::Registry::difference: illegal action item";
   const string __apply_illegal_action_message = "TracePack::Registry::apply: illegal action item";
   const string __apply_illegal_state_message = "TracePack::Registry::apply: illegal state item";
   const string __apply_illegal_no_add_message = __apply_illegal_action_message+" (TD_Added expected)";
#endif

//--------------------------------------------
// class FileChunk
//--------------------------------------------
   
bool FileChunk::dataChunk_copy( std::istream &io_src, std::ostream &io_dst ) const
{
   write_file( io_src, io_dst, onDiskData() );
   return true;
}

bool FileChunk::write_dataChunk( const cElFilename &i_inStreamFilename, std::istream &io_in, bool i_reverseByteOrder, std::ostream &io_out )
{      
   // write header
   io_out.put( (char)m_type );
   write_uint4( m_contentSize, i_reverseByteOrder, io_out );
   
   // write data chunk header
   if ( m_isFirst )
   {
      io_out.put( 1 );
      write_string( m_filename, io_out, i_reverseByteOrder );
   }
   else
      io_out.put( 0 );
   io_out.put( (char)(m_hasMore?1:0) );
   
   m_inStreamFilename = i_inStreamFilename;
   m_inStreamOffset = io_out.tellp();
   
   // write file data for this chunk
   return dataChunk_copy( io_in, io_out );
}

void FileChunk::write_chunk( bool i_reverseByteOrder, std::ostream &io_out ) const
{
   #ifdef __DEBUG_TRACE_PACK
      if ( m_type==FCT_Data )
      {
	 cerr << RED_DEBUG_ERROR << "FileChunk::write: chunk is of type FCT_Data" << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
   // write header
   io_out.put( (char)m_type );
   write_uint4( m_contentSize, i_reverseByteOrder, io_out );
   // write rawData for non-data chunks
   io_out.write( m_rawData.data(), m_rawData.size() );
}

bool FileChunk::read( const cElFilename &i_filename, std::istream &io_in, bool i_reverseByteOrder )
{
   m_type = (FileChunkType)io_in.get();
   read_uint4( io_in, i_reverseByteOrder, m_contentSize );
   
   switch ( m_type )
   {
   case FCT_Data:
      return read_dataChunk( io_in, i_reverseByteOrder, i_filename );
   case FCT_Ignore:
   case FCT_Registry:
      m_rawData.resize( m_contentSize );
      return read_chunk( io_in );
   default:
      #ifdef __DEBUG_TRACE_PACK
	 cerr << RED_DEBUG_ERROR << "FileChunk::read: unkown type (from int value " << (int)m_type << ")" << endl;
	 exit(EXIT_FAILURE);
      #endif
      return false;
   }
}

bool FileChunk::read_dataChunk( std::istream &io_in, bool i_reverseByteOrder, const cElFilename &i_filename )
{
   #ifdef __DEBUG_TRACE_PACK
      if ( m_type!=FCT_Data )
      {
	 cerr << RED_DEBUG_ERROR << "FileChunk::read_dataChunk_header: chunk is not of type FCT_Data" << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
   
   m_isFirst = ( io_in.get()==1 );
      
   if ( m_isFirst )
   {
      read_string( io_in, i_reverseByteOrder, m_filename );
      m_filenameRawSize = string_raw_size( m_filename );
      #ifdef __DEBUG_TRACE_PACK
	 if ( m_contentSize<( dataHeaderSize+string_raw_size(m_filename) ) )
	 {
	    cerr << RED_DEBUG_ERROR << "FileChunk::read_dataChunk_header: negative m_dataSize, m_size = " << m_contentSize << " < dataHeaderSize+raw_size(m_filename) = "
	         << dataHeaderSize+string_raw_size(m_filename) << endl;
	    exit(EXIT_FAILURE);
	 }
      #endif
   }
   
   m_dataSize = m_contentSize-dataHeaderSize;
   m_hasMore = ( io_in.get()==1 );
   
   m_inStreamFilename = i_filename;
   m_inStreamOffset = io_in.tellg();
   
   return true;
}

void FileChunk::trace( ostream &io_dst ) const
{
   io_dst << "fullSize = " << fullSize() << ' ' << FileChunkType_to_string(m_type) << " contentSize = " << m_contentSize << ' ';
   if ( m_isFirst ) io_dst << m_filename << "(" << m_filenameRawSize << ") ";
   if ( m_hasMore ) io_dst << "hasMore ";
   if ( m_type==FCT_Data )
      io_dst  << "dataSize = " << m_dataSize << endl;
   else
      io_dst  << "size(rawData) = " << m_rawData.size() << endl;
}

bool FileChunk::readAllChunks( const cElFilename &i_filename, streampos i_offset, bool i_reverseByteOrder, std::list<FileChunk> &o_chunks )
{   
   ifstream fsrc( i_filename.str_unix().c_str(), ios::binary );
   
   if ( !fsrc ) return false;
   
   fsrc.seekg( 0, ios::end );
   streampos lastPost = fsrc.tellg();
   fsrc.seekg( i_offset );

   FileChunk chunk;
   FileChunkType type;
   while ( fsrc.tellg()<lastPost ) // seekg do not set eof() to true, we need another test
   {      
      if ( !chunk.read( i_filename, fsrc, i_reverseByteOrder ) ) return false;
      fsrc.seekg( chunk.toSkip(), ios::cur );      
      o_chunks.push_back( chunk );
   }
   return true;
}

bool FileChunk::readAllChunks( const std::list<cElFilename> &i_filenames, streampos i_offset, bool i_reverseByteOrder, std::list<FileChunk> &o_chunks )
{
   o_chunks.clear();
   if ( i_filenames.size()==0 ) return true;
   list<cElFilename>::const_iterator itFilename = i_filenames.begin();
   if ( !readAllChunks( *itFilename++, i_offset, i_reverseByteOrder, o_chunks ) ) return false;
   while ( itFilename!=i_filenames.end() )      
      if ( !readAllChunks( *itFilename++, 0, i_reverseByteOrder, o_chunks ) ) return false;
   return true;
}

bool FileChunk::outputData( std::ostream &io_dst ) const
{
   ifstream fsrc( m_inStreamFilename.str_unix().c_str(), ios::binary );
   
   if ( !fsrc )
   {
      #ifdef __DEBUG_TRACE_PACK
	 cerr << RED_DEBUG_ERROR << "FileChunk::outputData: unable to open source file [" << m_inStreamFilename.str_unix() << ']' << endl;
	 exit(EXIT_FAILURE);
      #endif
      return false;
   }
   
   fsrc.seekg( m_inStreamOffset );
   dataChunk_copy( fsrc, io_dst );
   return true;
}


//--------------------------------------------
// class ChunkStream
//--------------------------------------------

ChunkStream::ChunkStream( const cElFilename &i_filename, U_INT4 i_maxFileSize, bool i_reverseByteOrder ):
   m_filename( i_filename ),
   m_maxFileSize( i_maxFileSize ),
   m_reverseByteOrder( i_reverseByteOrder ),
   m_iFirstChunkFile(0),
   m_offsetInFirstFile(0)
{
}

cElFilename ChunkStream::filename( unsigned int i_iFile ) const
{
   if ( i_iFile==0 ) return m_filename;
   stringstream ss;
   ss << m_filename.m_basename << '.' << i_iFile;
   return cElFilename( m_filename.m_path, ss.str() );
}

unsigned int ChunkStream::getNbFiles() const
{
   unsigned int i = m_iFirstChunkFile;
   while ( cElFilename( filename(i) ).exists() ) i++;
   return i;
}

bool ChunkStream::setOffset( U_INT4 i_iFirstChunkFile, streampos i_offset )
{
   if ( m_maxFileSize<=(U_INT4)i_offset )
   {
      #ifdef __DEBUG_TRACE_PACK
	 cerr << RED_DEBUG_ERROR << "ChunkStream::setOffset: i_offset = " << (U_INT4)i_offset << " >= m_maxFileSize = " << m_maxFileSize << endl;
	 exit(EXIT_FAILURE);
      #endif
      return false;
   }
   i_iFirstChunkFile = i_iFirstChunkFile;
   m_offsetInFirstFile = i_offset;
   return true;
}
   
bool ChunkStream::readChunks( std::list<FileChunk> &o_chunks )
{
   // construct file list
   const unsigned int iFileEnd = m_iFirstChunkFile+getNbFiles();
   list<cElFilename> filenames;
   for ( unsigned int i=m_iFirstChunkFile; i<iFileEnd; i++ )
      filenames.push_back( filename(i) );
   return FileChunk::readAllChunks( filenames, m_offsetInFirstFile, m_reverseByteOrder, o_chunks );
}

bool ChunkStream::open_new_or_append( unsigned int &io_iLastKnownFile, unsigned int i_minRemainingSize, cElFilename &o_filename, unsigned int &o_remainingSize, std::ofstream &io_fdst )
{
   o_filename = filename(io_iLastKnownFile);
   unsigned int fileSize = o_filename.getSize();
   
   o_remainingSize = ( fileSize>=m_maxFileSize?0:m_maxFileSize-fileSize );
   if ( o_remainingSize<i_minRemainingSize )
   {
      o_remainingSize = m_maxFileSize;
      o_filename = filename(++io_iLastKnownFile); // use next file
   }
   
   if ( !o_filename.exists() )
      io_fdst.open( o_filename.str_unix().c_str(), ios::binary );
   else
   {
      if ( io_iLastKnownFile==m_iFirstChunkFile && fileSize<(unsigned int)m_offsetInFirstFile )
      {
	 #ifdef __DEBUG_TRACE_PACK
	    cerr << RED_DEBUG_ERROR << "ChunkStream::open_new_or_append: first stream file [" << o_filename.str_unix() << "] is supposed to have an offset of " 
	         << (unsigned int)m_offsetInFirstFile << " but has a size of " << fileSize << endl;
	    exit(EXIT_FAILURE);
	 #endif
	 return false;
      }
      io_fdst.open( o_filename.str_unix().c_str(), ios::binary|ios::app );
   }
   
   #ifdef __DEBUG_TRACE_PACK
      if ( !io_fdst )
      {
	 cerr << RED_DEBUG_ERROR << "ChunkStream::open_new_or_append: unable to open file [" << o_filename.str_unix() << ']' << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
   
   return (bool)io_fdst;
}

bool ChunkStream::writeChunks( std::list<FileChunk> &io_chunks )
{
   if ( io_chunks.size()==0 ) return true;
   
   unsigned int iLastFile = m_iFirstChunkFile+getNbFiles();
   if ( iLastFile!=m_iFirstChunkFile ) iLastFile--;
   
   cElFilename file;
   unsigned int remainingSize;
   ofstream fdst;
   list<FileChunk>::iterator itChunk = io_chunks.begin();
   
   if ( !open_new_or_append( iLastFile, itChunk->fullSize(), file, remainingSize, fdst ) ) return false;
   
   ifstream fsrc;
   while ( itChunk!=io_chunks.end() )
   {
      FileChunk &chunk = *itChunk++;
      const unsigned int fullSize = chunk.fullSize();
      
      if ( fullSize>m_maxFileSize )
      {
	 #ifdef __DEBUG_TRACE_PACK
	    cerr << RED_DEBUG_ERROR << "ChunkStream::writeChunks: chunk full size = " << fullSize << " > m_maxFileSize = " << m_maxFileSize << endl;
	    exit(EXIT_FAILURE);
	 #endif
	 return false;
      }
      
      // open new file if need
      if ( remainingSize<fullSize )
      {
	 fdst.close();
	 file = filename( ++iLastFile );
	 fdst.open( file.str_unix().c_str(), ios::binary );
	 remainingSize = m_maxFileSize;
      }
      remainingSize -= fullSize;
      
      if ( chunk.m_type==FileChunk::FCT_Data )
      {
	 if ( chunk.m_isFirst ) fsrc.open( chunk.m_filename.c_str(), ios::binary );
	 if ( !chunk.write_dataChunk( file, fsrc, m_reverseByteOrder, fdst ) ) return false;
	 if ( !chunk.m_hasMore ) fsrc.close();
      }
      else
	 chunk.write_chunk( m_reverseByteOrder, fdst );
   }
   return true;
}

bool ChunkStream::remove() const
{
   unsigned int iFile = 0;
   cElFilename file = filename(iFile);
   while ( file.exists() )
   {
      if ( !file.remove() ) return false;
      file = filename(++iFile);
   }
   return true;
}


//--------------------------------------------
// class TracePack::Registry::Item
//--------------------------------------------

void TracePack::Registry::Item::reset_date()
{
   ELISE_fp::lastModificationDate( m_filename.str_unix(), m_date );
}

void TracePack::Registry::Item::dump( ostream &io_ostream, const string &i_prefix ) const
{
   io_ostream << i_prefix << "filename   : [" << m_filename.str_unix() << ']' << endl;
   io_ostream << i_prefix << "type       : " << TD_Type_to_string(m_type) << endl;
   io_ostream << i_prefix << "rights     : " << file_rights_to_string(m_rights) << " (" << m_rights << ")" << endl;
   io_ostream << i_prefix << "date       : " << m_date << endl;
   io_ostream << i_prefix << "dataOffset : " << m_dataOffset << endl;
   io_ostream << i_prefix << "dataSize   : " << m_dataSize << endl;
}

bool TracePack::Registry::Item::copyToDirectory( const cElFilename &i_packName, const ctPath &i_directory ) const
{
   const cElFilename filename( i_directory, m_filename );
   
   if ( !filename.m_path.exists() && !filename.m_path.create() )
   {
      #ifdef __DEBUG_TRACE_PACK
	 cerr << RED_DEBUG_ERROR << "TracePack::Registry::Item::copyToDirectory: cannot create directory [" << filename.m_path.str() << ']' << endl;
	 exit(EXIT_FAILURE);
      #endif
      return false;
   }
   
   ofstream fOut( filename.str_unix().c_str(), ios::binary );
   ifstream fIn( i_packName.str_unix().c_str(), ios::binary );
   
   fIn.seekg( m_dataOffset );
   
   if ( !fOut || !fIn )
   {
      #ifdef __DEBUG_TRACE_PACK
	 if ( !fIn )
	 {
	    cerr << RED_DEBUG_ERROR << "TracePack::Registry::Item::copyToDirectory: unable to open [" << i_packName.str_unix() << "] for reading at position " << m_dataOffset << endl;
	    exit(EXIT_FAILURE);
	 }
	 if ( !fOut )
	 {
	    cerr << RED_DEBUG_ERROR << "TracePack::Registry::Item::copyToDirectory: unable to open [" << filename.str_unix() << "] for writing" << endl;
	    exit(EXIT_FAILURE);
	 }
      #endif
      return false;
   }
   
   return write_file( fIn, fOut, m_dataSize );
}

bool TracePack::Registry::Item::trace_compare( const TracePack::Registry::Item &i_b ) const
{
   if ( m_filename!=i_b.m_filename )
      cout << "Items have different filenames : " << m_filename.str_unix() << " != " << i_b.m_filename.str_unix() << endl;
   else if ( m_type!=i_b.m_type )
      cout << "Items have different types : " << TD_Type_to_string(m_type) << " != " << TD_Type_to_string(i_b.m_type) << endl;
   else if ( m_date!=i_b.m_date )
      cout << "Items have different dates : " << m_date << " != " << i_b.m_date << endl;
   else if ( m_dataOffset!=i_b.m_dataOffset )
      cout << "Items have different data offsets : " << m_dataOffset << " != " << i_b.m_dataOffset << endl;
   else if ( m_dataSize!=i_b.m_dataSize )
      cout << "Items have different data sizes : " << m_dataSize << " != " << i_b.m_dataSize << endl;
   else
      return true;
   return false;
}

bool TracePack::Registry::Item::applyToDirectory( const cElFilename &i_packname, const ctPath &i_path ) const
{
   switch ( m_type )
   {
   case TD_Added:
   case TD_Modified:
      if ( !copyToDirectory( i_packname, i_path ) ) return false;
      return m_filename.setRights( m_rights );
   case TD_Removed:
      return cElFilename( i_path, m_filename ).remove();
   case TD_State:
      #ifdef __DEBUG_TRACE_PACK
	 cerr << RED_DEBUG_ERROR << "TracePack::Registry::Item::applyToDirectory: cannot apply state item [" << m_filename.str_unix() << ']';
	 exit(EXIT_FAILURE);
      #endif
      break;
   }
   return false;
}

// input/output methods
unsigned int TracePack::Registry::Item::raw_size() const
{
   unsigned int res = string_raw_size( m_filename.str_unix() )+4/*type*/;
   if ( m_type==TD_Added ||
        m_type==TD_Modified ||
	m_type==TD_State )
      res += cElDate::raw_size()+4/*rights*/+4/*file size*/;
   return res;
}

void TracePack::Registry::Item::to_raw_data( bool i_reverseByteOrder, char *&o_rawData ) const
{
   #ifdef __DEBUG_TRACE_PACK
      char *rawData = o_rawData;
   #endif
   
   // copy filename
   string_to_raw_data( m_filename.str_unix(), i_reverseByteOrder, o_rawData );
   
   // copy type
   INT4 i = (INT4)m_type;
   if ( i_reverseByteOrder ) byte_inv_4( &i );
   memcpy( o_rawData, &i, 4 );
   o_rawData += 4;
   
   if ( m_type==TracePack::Registry::TD_State ||
	m_type==TracePack::Registry::TD_Added ||
	m_type==TracePack::Registry::TD_Modified )
   {
      // copy last modification date
      m_date.to_raw_data( i_reverseByteOrder, o_rawData );
      
      // copy rights on file
      U_INT4 ui = (U_INT4)m_rights;
      if ( i_reverseByteOrder ) byte_inv_4( &ui );
      memcpy( o_rawData, &i, 4 );
      o_rawData += 4;
      
      // copy file's size
      ui = (U_INT4)m_dataSize;
      if ( i_reverseByteOrder ) byte_inv_4( &ui );
      memcpy( o_rawData, &i, 4 );
      o_rawData += 4;
   }
   
   #ifdef __DEBUG_TRACE_PACK
      unsigned int nbCopied = o_rawData-rawData;
      if ( nbCopied!=raw_size() )
      {
	 cerr << RED_DEBUG_ERROR << "TracePack::Registry::Item::to_raw_data: " << nbCopied << " copied bytes, but raw_size() = " << raw_size() << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
}

void TracePack::Registry::Item::from_raw_data( char *&io_rawData, bool i_reverseByteOrder )
{
   #ifdef __DEBUG_TRACE_PACK
      char *rawData = io_rawData;
   #endif
      
   // copy filename
   string s;
   string_from_raw_data( io_rawData, i_reverseByteOrder, s );
   m_filename = cElFilename(s);
   
   // copy type
   INT4 i;
   memcpy( &i, io_rawData, 4 );
   if ( i_reverseByteOrder ) byte_inv_4( &i );
   io_rawData += 4;
   m_type = (TD_Type)i;
   
   if ( hasData() )
   {
      // copy last modification date
      m_date.from_raw_data( io_rawData, i_reverseByteOrder );
      
      // copy rights on file
      U_INT4 ui;
      memcpy( &ui, io_rawData, 4 );
      if ( i_reverseByteOrder ) byte_inv_4( &ui );
      io_rawData += 4;
      
      // copy file size
      memcpy( &ui, io_rawData, 4 );
      if ( i_reverseByteOrder ) byte_inv_4( &ui );
      m_dataSize = (unsigned int)ui;
      io_rawData += 4;
   }
   
   #ifdef __DEBUG_TRACE_PACK
      unsigned int nbCopied = io_rawData-rawData;
      if ( nbCopied!=raw_size() )
      {
	 cerr << RED_DEBUG_ERROR << "TracePack::Registry::Item::to_raw_data: " << nbCopied << " copied bytes, but raw_size() = " << raw_size() << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
}

bool TracePack::Registry::Item::computeData( unsigned int i_remainingDst )
{
   #ifdef __DEBUG_TRACE_PACK
      if ( m_data.size()>0 )
      {
	 cerr << RED_DEBUG_ERROR << "TracePack::Registry::Item::computeData: computing data a second time" << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
   return file_to_chunk_list( m_filename.str_unix(), m_dataSize, i_remainingDst, TracePack::maxPackFileSize, m_data );
}


//--------------------------------------------
// class TracePack::Registry
//--------------------------------------------

TracePack::Registry::Registry():
   m_hasBeenWritten( false )
{
}

void TracePack::Registry::difference( const Registry &i_a, const Registry &i_b )
{
   m_items.clear();
   list<TracePack::Registry::Item>::const_iterator itA = i_a.m_items.begin(),
                                                   itB = i_b.m_items.begin();
   while ( itA!=i_a.m_items.end() && itB!=i_b.m_items.end() )
   {
      #ifdef __DEBUG_TRACE_PACK
	 if ( itA->m_type!=TD_State ||
              itB->m_type!=TD_State )
	 {
	    cerr << RED_DEBUG_ERROR << __difference_illegal_item_message << endl;
	    exit(EXIT_FAILURE);
	 }
      #endif
      
      int compare = itA->m_filename.compare( itB->m_filename );
      if ( compare==0 )
      {
	 if ( itA->m_date!=itB->m_date )
	    m_items.push_back( Item( itA->m_filename, TD_Modified, itB->m_date, itB->m_rights, itB->m_dataOffset, itB->m_dataSize ) );
	 itA++; itB++;
      }
      else
      {
	 if ( compare<0 )
	    m_items.push_back( Item( (*itA++).m_filename, TD_Removed, cElDate::NoDate, 0/*rights*/, 0/*data offset*/, 0/*data length*/ ) );
	 else
	 {
	    m_items.push_back( Item( itB->m_filename, TD_Added, itB->m_date, itB->m_rights, itB->m_dataOffset, itB->m_dataSize ) );
	    itB++;
	 }
      }
   }
   while ( itA!=i_a.m_items.end() )
   {
      #ifdef __DEBUG_TRACE_PACK
	 if ( itA->m_type!=TD_State )
	 {
	    cerr << RED_DEBUG_ERROR << __difference_illegal_item_message << endl;
	    exit(EXIT_FAILURE);
	 }
      #endif
      m_items.push_back( Item( (*itA++).m_filename, TD_Removed, cElDate::NoDate, 0/*rights*/, 0/*data offset*/, 0/*data length*/ ) );
   }
   while ( itB!=i_b.m_items.end() )
   {
      #ifdef __DEBUG_TRACE_PACK
	 if ( itB->m_type!=TD_State )
	 {
	    cerr << RED_DEBUG_ERROR << __difference_illegal_item_message << endl;
	    exit(EXIT_FAILURE);
	 }
      #endif
      m_items.push_back( Item( itB->m_filename, TD_Added, itB->m_date, itB->m_rights, itB->m_dataOffset, itB->m_dataSize ) );
      itB++;
   }
}

void TracePack::Registry::compare_states( const TracePack::Registry &i_a, const TracePack::Registry &i_b,
                                          list<Registry::Item> &o_onlyInA, list<Registry::Item> &o_onlyInB, list<Registry::Item> &o_inBoth )
{  
   o_onlyInA.clear();
   o_onlyInB.clear();
   o_inBoth.clear();
   list<TracePack::Registry::Item>::const_iterator itA = i_a.m_items.begin(),
                                                   itB = i_b.m_items.begin();
   while ( itA!=i_a.m_items.end() && itB!=i_b.m_items.end() )
   {
      int compare = itA->m_filename.compare( itB->m_filename );
      if ( compare==0 )
      {
	 if ( itA->m_type==itB->m_type )
	    o_inBoth.push_back( *itA );
	 else
	 {
	    o_onlyInA.push_back( *itA );
	    o_onlyInB.push_back( *itB );
	 }
	 itA++; itB++;
      }
      else if ( compare<0 )
	 o_onlyInA.push_back( *itA++ );
      else if ( compare>0 )
	 o_onlyInB.push_back( *itB++ );
   }
   while ( itA!=i_a.m_items.end() )
      o_onlyInA.push_back( *itA++ );
   while ( itB!=i_b.m_items.end() )
      o_onlyInB.push_back( *itB++ );
}

bool TracePack::Registry::write_v1( ofstream &io_stream, const ctPath &i_anchor, bool )
{   
   U_INT4 ui = (U_INT4)m_items.size(),
          dataSize;
   INT4 i;
   string s;
   
   m_command.write( io_stream, false/*__DEL*/ );
   io_stream.write( (char*)(&ui), 4 );
   list<Item>::iterator itItem = m_items.begin();
   while ( itItem!=m_items.end() )
   {
      // write filename
      s = itItem->m_filename.str_unix();
      write_string( s, io_stream, false/*__DEL*/ );
      // write type
      i = (INT4)itItem->m_type;
      io_stream.write( (char*)(&i), 4 );
      if ( itItem->m_type==TracePack::Registry::TD_State ||
           itItem->m_type==TracePack::Registry::TD_Added ||
           itItem->m_type==TracePack::Registry::TD_Modified )
      {
	 // write last modification date
	 itItem->m_date.write_raw( io_stream );
	 // write rights on file
	 ui = (U_INT4)itItem->m_rights;
	 io_stream.write( (char*)(&ui), 4 );
	 // writing file's data
	 dataSize = (U_INT4)itItem->m_dataSize;
	 io_stream.write( (char*)(&dataSize), 4 );
	 itItem->m_dataOffset = io_stream.tellp();
	 if ( !write_file( cElFilename( i_anchor, itItem->m_filename ), io_stream, dataSize ) )
	 {
	    #ifdef __DEBUG_TRACE_PACK
	       cerr << RED_DEBUG_ERROR << "TracePack::Registry::write_v1: unable to copy data of file [" << itItem->m_filename.str_unix() << ']' << endl;
	       exit(EXIT_FAILURE);
	    #endif
	    return false;
	 }
      }
      itItem++;
   }
   m_hasBeenWritten = true;
   return true;
}

void TracePack::Registry::read_v1( istream &io_stream, bool )
{
   m_items.clear();
   U_INT4 nbFiles, dataSize;
   streampos offset;
   INT4 i;
   vector<char> buffer;
   cElFilename itemFilename;
   cElDate d( cElDate::NoDate );
   U_INT4 rights;
   m_command.read( io_stream, false/*__DEL*/ );
   io_stream.read( (char*)&nbFiles, 4 );
   
   while ( nbFiles-- )
   {
      // read filename
      itemFilename = cElFilename( read_string( io_stream, buffer, false/*__DEL*/ ) );
      // write type
      io_stream.read( (char*)(&i), 4 );
      // add item
      if ( i==TracePack::Registry::TD_State ||
           i==TracePack::Registry::TD_Added ||
           i==TracePack::Registry::TD_Modified )
      {
	 // read last modification date
	 d.read_raw( io_stream );
	 // read rights on file
	 io_stream.read( (char*)(&rights), 4 );
	 // read file's raw data
	 io_stream.read( (char*)(&dataSize), 4 );
	 offset = io_stream.tellg();
	 io_stream.seekg( dataSize, ios::cur );
      }
      else
      {
	 // TD_Remove
	 offset = dataSize = 0;
	 rights = 0;
	 d = cElDate::NoDate;
      }
      m_items.push_back( Item( itemFilename, (TD_Type)i, d, (mode_t)rights, offset, dataSize ) );
   }
   m_hasBeenWritten = true;
}

void TracePack::Registry::reset_dates()
{
   list<Item>::iterator itItem = m_items.begin();
   while ( itItem!=m_items.end() )
      ( *itItem++ ).reset_date();
}

TracePack::Registry::Item & TracePack::Registry::add( const Item &i_item )
{
   const cElFilename &filename = i_item.m_filename;
   list<Item>::iterator it = m_items.begin();
   while ( it!=m_items.end() && it->m_filename<filename )
      it++;
   if ( it==m_items.end() )
   {
      m_items.push_back( i_item );
      return m_items.back();
   }
   if ( it->m_filename==filename ) return *it;
   return *m_items.insert( it, i_item );
}

void TracePack::Registry::stateDirectory( const ctPath &i_path )
{
   ctPath path( i_path );
   path.toAbsolute( getWorkingDirectory() );
   
   #ifdef __DEBUG_TRACE_PACK
      if ( !path.exists() )
      {
	 cerr << RED_DEBUG_ERROR << "TracePack::Registry::stateDirectory: directory ["+i_path.str()+"] does not exist" << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
   
   m_items.clear();
   cElDate date = cElDate::NoDate;
   int fileLength;
   mode_t rights;
   list<string> files = RegexListFileMatch( path.str()+ctPath::sm_unix_separator, ".*", numeric_limits<INT>::max(), false );
   list<string>::iterator itFile = files.begin();
   while ( itFile!=files.end() )
   {
      cElFilename attachedFile( i_path, *itFile ), // attached filename is the filename in situ
                  detachedFile( *itFile++ ); // detached filename is the filename without source path
      const string attached_filename = attachedFile.str_unix();
      ELISE_fp::lastModificationDate( attached_filename, date );
      attachedFile.getRights( rights );
      #ifdef __DEBUG_TRACE_PACK
	 if ( !attachedFile.getRights( rights ) )
	 {
	    cerr << RED_DEBUG_ERROR << "TracePack::Registry::stateDirectory: cannot retrieve rights on file [" << attached_filename << ']' << endl;
	    exit(EXIT_FAILURE);
	 }
      #endif
      fileLength = ELISE_fp::file_length( attached_filename );
      #ifdef __DEBUG_TRACE_PACK
	 if ( fileLength<0 )
	 {
	    cerr << RED_DEBUG_ERROR << "TracePack::Registry::stateDirectory: cannot read length of file [" << attached_filename << "]" << endl;
	    exit(EXIT_FAILURE);
	 }
      #endif
      add( Item( detachedFile, TD_State, date, rights, 0, (unsigned int)fileLength ) );
   }
}

void TracePack::Registry::apply( const TracePack::Registry &i_actions )
{
   list<TracePack::Registry::Item>::iterator itA = m_items.begin();
   list<TracePack::Registry::Item>::const_iterator itB = i_actions.m_items.begin();
   while ( itA!=m_items.end() && itB!=i_actions.m_items.end() )
   {      
      #ifdef __DEBUG_TRACE_PACK
	 if ( itA->m_type!=TD_State )
	 {
	    cerr << RED_DEBUG_ERROR << __apply_illegal_action_message << endl;
	    exit(EXIT_FAILURE);
	 }
	 if ( itB->m_type==TD_State )
	 {
	    cerr << RED_DEBUG_ERROR << __apply_illegal_state_message << endl;
	    exit(EXIT_FAILURE);
	 }
      #endif
      
      int compare = itA->m_filename.compare( itB->m_filename );
      if ( compare==0 )
      {	 
	 if ( itB->m_type==TD_Removed )
	 {
	    itA = m_items.erase( itA );
	    itB++;
	 }
	 else if ( itB->m_type==TD_Modified )
	 {
	    itA->m_date       = itB->m_date;
	    itA->m_rights     = itB->m_rights;
	    itA->m_dataOffset = itB->m_dataOffset;
	    itA->m_dataSize   = itB->m_dataSize;
	    itA++; itB++;
	 }
	 #ifdef __DEBUG_TRACE_PACK
	    else
	    {
	       cerr << RED_DEBUG_ERROR << "TracePack::Registry::apply: illegal action item of type TD_Added" << endl;
	       exit(EXIT_FAILURE);
	    }
	 #endif
      }
      else
      {
	 if ( compare<0 )
	    itA++;
	 else
	 {
	    #ifdef __DEBUG_TRACE_PACK
	       if ( itB->m_type!=TD_Added )
	       {
		  cerr << RED_DEBUG_ERROR << __apply_illegal_no_add_message << endl;
		  exit(EXIT_FAILURE);
	       }
	       else
	    #endif
	       {
		  m_items.insert( itA, Item( itB->m_filename, TD_State, itB->m_date, itB->m_rights, itB->m_dataOffset, itB->m_dataSize ) );
		  itB++;
	       }
	 }
      }
   }
   while ( itB!=i_actions.m_items.end() )
   {
      #ifdef __DEBUG_TRACE_PACK
	 if ( itB->m_type!=TD_Added )
	 {
	    cerr << RED_DEBUG_ERROR << __apply_illegal_no_add_message << endl;
	    exit(EXIT_FAILURE);
	 }
	 else
      #endif
	 {
	    m_items.push_back( Item( itB->m_filename, TD_State, itB->m_date, itB->m_rights, itB->m_dataOffset, itB->m_dataSize ) );
	    itB++;
	 }
   }
}

void TracePack::Registry::dump( ostream &io_ostream, const string &i_prefix ) const
{
   const string prefix = i_prefix+"\t";
   list<Item>::const_iterator itItem = m_items.begin();
   int i = 0;
   while ( itItem!=m_items.end() )
   {
      io_ostream << i_prefix << "item " << i << " :" << endl;
      itItem->dump( io_ostream, prefix );
      itItem++; i++;
   }
}

TracePack::Registry::Item * TracePack::Registry::getItem( const cElFilename &i_filename )
{
   list<TracePack::Registry::Item>::iterator itItem = m_items.begin();
   while ( itItem!=m_items.end() )
   {
      if ( itItem->m_filename==i_filename )
	 return &(*itItem);
      itItem++;
   }
   return NULL;
}

bool TracePack::Registry::trace_compare( const TracePack::Registry &i_b ) const
{
   unsigned int nbItems = m_items.size();
   if ( nbItems!=i_b.m_items.size() )
   {
      cout << "registries have different number of items : " << nbItems << " != " << i_b.m_items.size() << endl;
      return false;
   }
   if ( m_command!=i_b.m_command )
   {
      cout << "registries have different commands : " << m_command.str() << " != " << i_b.m_command.str() << endl;
      return false;
   }
   list<TracePack::Registry::Item>::const_iterator itA = m_items.begin(),
                                                   itB = i_b.m_items.begin();
   for ( unsigned int iItem=0; iItem<nbItems; iItem++ )
   {
      if ( !itA->trace_compare( *itB ) )
      {
	 cout << "registries items of index " << iItem << " are different" << endl;
	 return false;
      }
      itA++; itB++;
   }
   return true;
}
	 
bool TracePack::Registry::applyToDirectory( const cElFilename &i_filename, const ctPath &i_path ) const
{
   list<TracePack::Registry::Item>::const_iterator itItem = m_items.begin();
   while ( itItem!=m_items.end() )
   {
      if ( !( itItem->applyToDirectory( i_filename, i_path ) ) )
      {
	 #ifdef __DEBUG_TRACE_PACK
	    cerr << RED_DEBUG_ERROR << "TracePack::Registry::applyToDirectory: unable to apply item of type " 
	         << TD_Type_to_string(itItem->m_type) << " for file [" << itItem->m_filename.str_unix() << "]" << endl;
	    exit(EXIT_FAILURE);
	 #endif
	 return false;
      }
      itItem++;
   }
   return true;
}

unsigned int TracePack::Registry::raw_size() const
{
   unsigned int res = 4; // size of nbItems
   list<Item>::const_iterator itItem = m_items.begin();
   while ( itItem!=m_items.end() )
      res += ( *itItem++ ).raw_size();
   return res;
}
      
// input/output methods
void TracePack::Registry::to_raw_data( bool i_reverseByteOrder, char *&o_rawData ) const
{
   #ifdef __DEBUG_TRACE_PACK
      char *rawData = o_rawData;
   #endif
      
   // copy command
   m_command.to_raw_data( i_reverseByteOrder, o_rawData );
   
   // copy number of items
   uint4_to_raw_data( m_items.size(), i_reverseByteOrder, o_rawData );
   
   list<Item>::const_iterator itItem = m_items.begin();
   while ( itItem!=m_items.end() )
      ( *itItem++ ).to_raw_data( i_reverseByteOrder, o_rawData );
   
   #ifdef __DEBUG_TRACE_PACK
      unsigned int nbCopied = o_rawData-rawData;
      if ( nbCopied!=raw_size() )
      {
	 cerr << RED_DEBUG_ERROR << "TracePack::Registry::: " << nbCopied << " copied bytes, but raw_size() = " << raw_size() << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
}

void TracePack::Registry::from_raw_data( char *&io_rawData, bool i_reverseByteOrder )
{
   #ifdef __DEBUG_TRACE_PACK
      char *rawData = io_rawData;
   #endif
      
   // copy commande
   m_command.from_raw_data( io_rawData, i_reverseByteOrder );
   
   // copy number of items
   U_INT4 nbItems;
   uint4_from_raw_data( io_rawData, i_reverseByteOrder, nbItems );
   
   // copy items
   m_items.resize( nbItems );
   list<Item>::iterator itItem = m_items.begin();
   while ( itItem!=m_items.end() )
      ( *itItem++ ).from_raw_data( io_rawData, i_reverseByteOrder );
   
   #ifdef __DEBUG_TRACE_PACK
      unsigned int nbCopied = io_rawData-rawData;
      if ( nbCopied!=raw_size() )
      {
	 cerr << RED_DEBUG_ERROR << "TracePack::Registry::from_raw_data_v1: " << nbCopied << " copied bytes, but raw_size() = " << raw_size() << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
}

void TracePack::Registry::getChunk( bool i_reverseByteOrder, FileChunk &o_chunk ) const
{
   o_chunk.m_type = FileChunk::FCT_Registry;
   o_chunk.m_rawData.resize( o_chunk.m_contentSize=raw_size() );
   char *itData = o_chunk.m_rawData.data();
   to_raw_data( i_reverseByteOrder, itData );
}

#ifdef __DEBUG_TRACE_PACK
   void TracePack::Registry::check_sorted() const
   {
      if ( m_items.size()<2 ) return;
      list<Item>::const_iterator it1 = m_items.begin(),
                                 it2 = it1;
      it2++;
      while ( it2!=m_items.end() )
      {
	 if ( (*it1++).m_filename.compare( (*it2++).m_filename )>=0 )
	 {
	    cerr << RED_DEBUG_ERROR << "TracePack::Registry::check_sorted: items are not corretly sorted or an item appears more than once" << endl;
	    exit(EXIT_FAILURE);
	 }
      }
   }
#endif


//--------------------------------------------
// class TracePack
//--------------------------------------------
   
TracePack::TracePack( const cElFilename &i_filename, const ctPath &i_anchor ):
   m_filename( i_filename ),
   m_lastFilename( i_filename ),
   m_remainingSpace( maxPackFileSize ),
   m_anchor( i_anchor ),
   m_date( cElDate::NoDate ),
   m_writeInUpdateMode( false ),
   m_nbRegistriesOffset( 0 ),
   m_writeIgnoredItems( true )
{
   
   cElDate::getCurrentDate_UTC( m_date );
}

void TracePack::read_v1( istream &f, bool i_reverseByteOrder )
{
   U_INT4 nbRegistries;
   m_date.read_raw( f );
   m_nbRegistriesOffset = f.tellg();   
   f.read( (char*)&nbRegistries, 4 );
   if ( i_reverseByteOrder ) byte_inv_4( &nbRegistries );   
   m_registries.resize( nbRegistries );
   list<Registry>::iterator itReg = m_registries.begin();
   while ( nbRegistries-- )
      ( *itReg++ ).read_v1( f, false/*__DEL*/ );
}

bool TracePack::load()
{
   ifstream f( m_filename.str_unix().c_str(), ios::binary );
   if ( !f )
   {
      #ifdef __DEBUG_TRACE_PACK
	 cerr << RED_DEBUG_ERROR << "TracePack::load: unable to open [" << m_filename.str_unix() << "] for reading" << endl;
	 exit(EXIT_FAILURE);
      #endif
      return false;
   }
   VersionedFileHeader header( VFH_TracePack );

   bool res = header.read_known( VFH_TracePack, f );
   
   #ifdef __DEBUG_TRACE_PACK
      if ( !res )
      {
	 cerr << RED_DEBUG_ERROR << "TracePack::load: unable to read versioned file header of type VFH_TracePack from file [" << m_filename.str_unix() << ']' << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
   
   switch ( header.version() )
   {
   case 1: read_v1( f, false/*__DEL*/ ); break;
   case 2: res=read_v2( f, false/*__DEL*/ ); break;
   default: res=false;
   }
   
   #ifdef __DEBUG_TRACE_PACK
      if ( !res )
      {
	 cerr << RED_DEBUG_ERROR << "TracePack::load: unable to read versioned file of type VFH_TracePack (v" << header.version() << ' ' << (header.isMSBF()?"big-endian":"little-endian") << ") from file [" << m_filename.str_unix() << ']' << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
   
   m_writeInUpdateMode = res;
   return res;
}

bool TracePack::write_v1( ofstream &f, bool i_reverseByteOrder )
{
   if ( !m_writeInUpdateMode )
   {
      // this is a new file
      m_date.write_raw( f );
      U_INT4 nbRegistries = (U_INT4)m_registries.size();
      m_nbRegistriesOffset = f.tellp();
      if ( i_reverseByteOrder ) byte_inv_4( &nbRegistries );
      f.write( (char*)&nbRegistries, 4 );
   }

   // write ignored items lists
   //if ( m_writeIgnoredItems ) write_ignored_items( f, i_reverseByteOrder );

   list<Registry>::iterator itReg = m_registries.begin();
   while ( itReg!=m_registries.end() )
   {
      if ( !itReg->m_hasBeenWritten && !itReg->write_v1( f, m_anchor, false/*__DEL*/ ) )
	 return false;
      itReg++;
   }
   
   if ( m_writeInUpdateMode )
   {
      f.close();
      save_nbRegistries();
   }
   
   return true;
}

string TracePack::packFilename( const U_INT4 &i_iFile ) const
{ 
   if ( i_iFile==0 ) return m_filename.str_unix();
   stringstream ss;
   ss << m_filename.str_unix() << "." << i_iFile;
   return ss.str();
}

unsigned int TracePack::getNbPackFiles() const
{
   unsigned int i = 0;
   while ( cElFilename( packFilename(i) ).exists() ) i++;
   return i;
}
   
bool TracePack::newPackFile()
{
   m_lastFilename = cElFilename( packFilename( ++m_iLastFile ) );
   m_remainingSpace = maxPackFileSize;
   if ( m_packOutputStream.is_open() )
   {
      m_packOutputStream.close();
      m_packOutputStream.open( m_lastFilename.str_unix().c_str(), ios::binary );
      
      #ifdef __DEBUG_TRACE_PACK
	 if ( !m_packOutputStream )
	 {
	    cerr << RED_DEBUG_ERROR << "TracePack::newPackFile: unable to create a new pack file of name [" << m_lastFilename.str_unix() << "]" << endl;
	    exit(EXIT_FAILURE);
	 }
      #endif
      
      return (bool)m_packOutputStream;
   }
   return true;
}

bool TracePack::writeData( TracePack::Registry::Item &i_item, bool i_reverseByteOrder )
{  
   if ( !i_item.computeData( m_remainingSpace ) ) return false;
 
   // write all chunks from item's data list
   ifstream srcFile( cElFilename( m_anchor, i_item.m_filename ).str_unix().c_str(), ios::binary );  
   list<FileChunk> &chunks = i_item.m_data;
   list<FileChunk>::iterator itChunk = chunks.begin();
   while ( itChunk!=chunks.end() )
   {
      const unsigned int chunkFullSize = itChunk->fullSize();
      if ( chunkFullSize>m_remainingSpace && !newPackFile() ) return false;
      
      itChunk->write_dataChunk( m_lastFilename, srcFile, i_reverseByteOrder, m_packOutputStream );
      m_remainingSpace -= chunkFullSize;
      
      itChunk++;
   }
   return true;
}

bool TracePack::writeData( TracePack::Registry &i_registry, bool i_reverseByteOrder )
{
   list<Registry::Item>::iterator itItem = i_registry.m_items.begin();
   while ( itItem!=i_registry.m_items.end() )
   {
      if ( itItem->hasData() && !writeData( *itItem, i_reverseByteOrder ) ) return false;
      itItem++;
   }
   return true;
}
      
bool TracePack::write_v2( bool i_reverseByteOrder )
{
   m_date.write_raw( m_packOutputStream, i_reverseByteOrder );
   
   FileChunk chunk;
   list<TracePack::Registry>::iterator itReg = m_registries.begin();
   while ( itReg!=m_registries.end() )
   {
      if ( !itReg->m_hasBeenWritten )
      {
	 // write the registry itself
	 itReg->getChunk( i_reverseByteOrder, chunk );
	 const unsigned int chunkFullSize = chunk.fullSize();
	 if ( chunkFullSize>m_remainingSpace ) newPackFile();
	 
	 if (chunkFullSize>m_remainingSpace)
	 {
	    #ifdef __DEBUG_TRACE_PACK
	       cerr << RED_DEBUG_ERROR << "TracePack::write_v2: registry chunk too big (" << chunk.fullSize() << "), maxPackFileSize = " << maxPackFileSize << endl;
	       exit(EXIT_FAILURE);
	    #endif
	    return false;
	 }
	 
	 chunk.write_chunk( i_reverseByteOrder, m_packOutputStream );
	 m_remainingSpace -= chunkFullSize;
	 
	 // write data of registry's Items
	 if ( !writeData( *itReg, i_reverseByteOrder ) )
	 {
	    #ifdef __DEBUG_TRACE_PACK
	       cerr << RED_DEBUG_ERROR << "TracePack::write_v2: unable to write registry's data" << endl;
	       exit(EXIT_FAILURE);
	    #endif
	    return false;
	 }
	 
	 itReg->m_hasBeenWritten = true;
      }
      itReg++;
   }

   // write ignored items lists
   //if ( m_writeIgnoredItems ) write_ignored_items( f, i_reverseByteOrder );
   
   return true;
}

bool TracePack::read_v2( ifstream &f, bool i_reverseByteOrder )
{
   /*
   const unsigned int nbFiles = getNbPackFiles();
   
   string currentFilename = m_filename.str_unix();
   FileChunk chunk;
   list<FileChunk> dataChunks;
   for ( unsigned int iFile=0; iFile<nbFiles; iFile++ )
   {
      m_date.read_raw( fsrc, i_reverseByteOrder );
      
      while ( !fsrc.eof() )
      {
	 chunk.read(f, i_reverseByteOrder);
	 
	 if ( chunk.m_type==FCT_Data )
	 {
	    if ( chunk.m_hasMore )
	    {
	       
	       if ( chunk.m_isFirst )
	       {
		  
	       }
	    }
	    else
	    {
	       
	    }
	 }
      }
      
      // open next file if there's one
      if ( iFile!=nbFiles-1 )
      {
	 currentFilename = m_filenamepackFilename( iFile+1 );
	 f.close();
	 f.open( currentFilename.c_str(), ios::binary );
      }
   }
   */
   return false;
}


void TracePack::save_nbRegistries() const
{
   #ifdef __DEBUG_TRACE_PACK
      if ( !m_writeInUpdateMode || m_nbRegistriesOffset==0 )
      {
	 cerr << RED_DEBUG_ERROR << "TracePack::update_registry_number: TracePack is not in update mode" << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
   ofstream f( m_filename.str_unix().c_str(), ios::in|ios::out|ios::binary );
   f.seekp( m_nbRegistriesOffset );
   U_INT4 nbRegistries = (U_INT4)m_registries.size();
   f.write( (char*)&nbRegistries, 4 );
}

bool TracePack::save( unsigned int i_version, bool i_MSBF )
{
   ofstream f;
   
   bool res = false;
   switch ( i_version )
   {
   case 1: // __TO_V1
      if ( m_writeInUpdateMode )
	 f.open( m_filename.str_unix().c_str(), ios::app|ios::binary );
      else
	 f.open( m_filename.str_unix().c_str(), ios::binary );
   
      if ( !m_writeInUpdateMode )
      {
	 VersionedFileHeader header( VFH_TracePack, 1, i_MSBF );
	 header.write( f );
      }
      res = write_v1( f, false/*__DEL*/ );
      break;
   case 2:
      if ( !m_filename.exists() )
      {
	 m_packOutputStream.open( packFilename(0).c_str(), ios::binary );
	 if ( !f ) return false;
	 VersionedFileHeader header( VFH_TracePack, 2, i_MSBF );
	 header.write( f );
      }
      else
	 m_packOutputStream.open( m_filename.str_unix().c_str(), ios::app|ios::binary );
      res = write_v2( i_MSBF==MSBF_PROCESSOR() );
      break;
   default: res=false;
   }
   m_writeInUpdateMode = res; // __TO_V1
   
   return res;
}

void TracePack::getState( unsigned int i_iState, TracePack::Registry &o_state ) const
{
   #ifdef __DEBUG_TRACE_PACK
      if ( i_iState>=m_registries.size() )
      {
	 cerr << RED_DEBUG_ERROR << "TracePack::getState: index out of range " << i_iState << " >= " << m_registries.size() << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
   list<TracePack::Registry>::const_iterator itReg = m_registries.begin();
   o_state = *itReg++;
   
   if ( i_iState==0 ) return;
   
   while ( i_iState-- )      
      o_state.apply( *itReg++ );
}

void TracePack::addState( const cElCommand &i_command )
{
   #ifdef __DEBUG_TRACE_PACK
      if ( !m_anchor.exists() )
      {
	 cerr << RED_DEBUG_ERROR << "TracePack::addState: directory [" << m_anchor.str() << "] does not exist" << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
   
   if ( nbStates()==0 )
   {
      Registry reg0;
      reg0.stateDirectory( m_anchor );
      m_registries.push_back( reg0 );
      return;
   }
   
   Registry directoryState, lastState, diff;
   directoryState.stateDirectory( m_anchor );
      
   getState( nbStates()-1, lastState );
   diff.difference( lastState, directoryState );
   diff.m_command = i_command;
   m_registries.push_back( diff );
}

void TracePack::dump( ostream &io_ostream, const string &i_prefix ) const
{
   const string prefix = i_prefix+"\t";
   list<Registry>::const_iterator itRegistry = m_registries.begin();
   int i = 0;
   while ( itRegistry!=m_registries.end() )
   {
      io_ostream << i_prefix << "registry " << i << " :" << endl;
      itRegistry->dump( io_ostream, prefix );
      itRegistry++; i++;
   }
}

const TracePack::Registry & TracePack::getRegistry( unsigned int i_iRegistry ) const
{
   #ifdef __DEBUG_TRACE_PACK
      if ( i_iRegistry>=m_registries.size() )
      {
	 cerr << RED_DEBUG_ERROR << "TracePack::getRegistry: index " << i_iRegistry << " out of range (max:" << m_registries.size() << ')' << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
   list<Registry>::const_iterator itReg = m_registries.begin();
   while ( i_iRegistry-- ) itReg++;
   return *itReg;
}

bool TracePack::copyItemOnDisk( unsigned int i_iState, const cElFilename &i_itemName )
{
   TracePack::Registry reg;
   getState( (unsigned int)i_iState, reg );
   TracePack::Registry::Item *pItem = reg.getItem( i_itemName );
   if ( pItem==NULL )
   {
      #ifdef __DEBUG_TRACE_PACK
	 cerr << "TracePack::copyItemOnDisk: file [" << i_itemName.str_unix() << "] does not exist in state " << i_iState << " of pack [" << m_filename.str_unix() << ']' << endl;
      #endif
      return false;
   }
   return pItem->copyToDirectory( m_filename, m_anchor );
}

bool TracePack::trace_compare( const TracePack &i_b ) const
{	 
   if ( m_filename!=i_b.m_filename )
      cout << "packs have different filenames : " << m_filename.str_unix() << " != " << i_b.m_filename.str_unix() << endl;
   //else if ( m_anchor!=i_b.m_anchor )
   //   cout << "packs have different anchors : " << m_anchor.str_unix() << " != " << i_b.m_anchor.str_unix() << endl;
   else if ( m_date!=i_b.m_date )
      cout << "packs have different dates : " << m_date << " != " << i_b.m_date << endl;
   //else if ( m_writeInUpdateMode!=i_b.m_writeInUpdateMode )
   //   cout << "packs have different writing modes : " << (m_writeInUpdateMode?"create":"update") << " != " << (i_b.m_writeInUpdateMode?"create":"update") << endl;
   else if ( m_nbRegistriesOffset!=i_b.m_nbRegistriesOffset )
      cout << "packs have different offset for the number of registries : " << m_nbRegistriesOffset << " != " << i_b.m_nbRegistriesOffset << endl;
   else
   {
      unsigned int nbRegistries = (unsigned int)m_registries.size();      
      if ( nbRegistries!=i_b.m_registries.size() )
      {
	 cout << "packs have different number of registries : " << nbRegistries << " != " << i_b.m_registries.size() << endl;
	 return false;
      }
      list<TracePack::Registry>::const_iterator itA = m_registries.begin(),
						itB = i_b.m_registries.begin();
      for ( unsigned int iReg=0; iReg<nbRegistries; iReg++ )
      {
	 if ( !itA->trace_compare( *itB ) )
	 {
	    cout << "packs registries of index " << iReg << " are different" << endl;
	    return false;
	 }
	 itA++; itB++;
      }
      return true;
   }
   return false;
}

void TracePack::setState( unsigned int i_iState )
{
   #ifdef __DEBUG_TRACE_PACK
      const unsigned int nbRegistries = nbStates();
      if ( i_iState>=nbRegistries )
	 cerr << "TracePack::setState: state index " << i_iState << " out of range (" << nbRegistries << " registries" 
	      << (nbRegistries>1?'s':'\0') << " in pack [" << m_filename.str_unix() << "])" << endl;
   #endif
   #ifdef __DEBUG_TRACE_PACK
      if ( !ELISE_fp::IsDirectory( m_anchor.str() ) )
      {
	 cerr << RED_DEBUG_ERROR << "TracePack::setState: destination directory [" << m_anchor.str() << "] does not exist" << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
   
   Registry refReg, dirReg, dir_to_ref;
   getState( i_iState, refReg );
   dirReg.stateDirectory( m_anchor );
   dir_to_ref.difference( dirReg, refReg );
   dir_to_ref.dump();
   dir_to_ref.applyToDirectory( m_filename, m_anchor );
}

void TracePack::getAllCommands( std::vector<cElCommand> &o_commands ) const
{
   o_commands.resize( nbStates() );
   list<TracePack::Registry>::const_iterator itReg = m_registries.begin();
   for ( unsigned int iCmd=0; iCmd<o_commands.size(); iCmd++ )
      o_commands[iCmd] = ( *itReg++ ).m_command;
}
   
void TracePack::add_ignored( const ctPath &i_directory )
{
   #ifdef __DEBUG_TRACE_PACK
      if ( i_directory.isAbsolute() || i_directory.count_upward_references()!=0 )
      {
	 cerr << RED_DEBUG_ERROR << "TracePack::add_ignored(ctPath): trying to add [" << i_directory.str() << "] which is invalid (absolute or outside anchor)" << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
   
   list<ctPath>::iterator itDirectory = m_ignoredDirectories.begin();
   while ( itDirectory!=m_ignoredDirectories.end() && (*itDirectory)<i_directory ) itDirectory++;
   
   #ifdef __DEBUG_TRACE_PACK
      if ( (*itDirectory)==i_directory )
      {
	 cerr << RED_DEBUG_ERROR << "TracePack::add_ignored(ctPath): trying to add [" << i_directory.str() << "] which is already in m_ignoredDirectories" << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
   
   m_ignoredDirectories.insert( itDirectory, i_directory );
}

void TracePack::add_ignored( const cElFilename &i_filename )
{
   #ifdef __DEBUG_TRACE_PACK
      if ( i_filename.m_path.isAbsolute() || i_filename.m_path.count_upward_references()!=0 )
      {
	 cerr << RED_DEBUG_ERROR << "TracePack::add_ignored(cElFilename): trying to add [" << i_filename.str_unix() << "] which is invalid (file's path is absolute or outside anchor)" << endl;
	 exit(EXIT_FAILURE);
      }
   #endif

   list<cElFilename>::iterator itFilename = m_ignoredFiles.begin();
   while ( itFilename!=m_ignoredFiles.end() && (*itFilename)<i_filename ) itFilename++;
   
   #ifdef __DEBUG_TRACE_PACK
      if ( (*itFilename)==i_filename )
      {
	 cerr << RED_DEBUG_ERROR << "TracePack::add_ignored(cElFilename): trying to add [" << i_filename.str_unix() << "] which is already in m_ignoredFiles" << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
   
   m_ignoredFiles.insert( itFilename, i_filename );
}
   
// add directories to ignored directories list if they enlarge the ignored items set
// io_ignoredDirectories is left with directories actually added to the list
void TracePack::addIgnored( list<ctPath> &io_directoriesToIgnore )
{   
   // remove already ignored directories from io_ignoredDirectories
   list<ctPath>::iterator itDirectoryToIgnore = io_directoriesToIgnore.begin();
   while ( itDirectoryToIgnore!=io_directoriesToIgnore.end() )
   {
      if ( isIgnored( *itDirectoryToIgnore ) )
	 itDirectoryToIgnore = io_directoriesToIgnore.erase( itDirectoryToIgnore );
      else
	 itDirectoryToIgnore++;
   }
   
   // remove ignored directories contained by elements of io_directoriesToIgnore
   list<ctPath>::iterator itIgnoredDirectory = m_ignoredDirectories.begin();
   while ( itIgnoredDirectory!=m_ignoredDirectories.end() )
   {
      if ( contains( io_directoriesToIgnore, *itIgnoredDirectory ) )
	 itIgnoredDirectory = m_ignoredDirectories.erase( itIgnoredDirectory );
      else
	 itIgnoredDirectory++;
   }
   
   // add all remaining elements of io_directoriesToIgnore to m_ignoredDirectories
   itDirectoryToIgnore = io_directoriesToIgnore.begin();
   while ( itDirectoryToIgnore!=io_directoriesToIgnore.end() )
      add_ignored( *itDirectoryToIgnore );
      
   m_writeIgnoredItems = ( io_directoriesToIgnore.size()!=0 );
}

// add files to ignored files list if they enlarge the ignored items set
// io_ignoredFiles is left with files actually added to the list
void TracePack::addIgnored( list<cElFilename> &io_filesToIgnore )
{
   // remove already ignored files from io_filesToIgnore
   list<cElFilename>::iterator itFileToIgnore = io_filesToIgnore.begin();
   while ( itFileToIgnore!=io_filesToIgnore.end() )
   {
      if ( isIgnored( *itFileToIgnore ) )
	 itFileToIgnore = io_filesToIgnore.erase( itFileToIgnore );
      else
	 itFileToIgnore++;
   }
   
   // add remaining files to ignored files list
   itFileToIgnore = io_filesToIgnore.begin();
   while ( itFileToIgnore!=io_filesToIgnore.end() )
      add_ignored( *itFileToIgnore++ );
   
   m_writeIgnoredItems = ( io_filesToIgnore.size()!=0 );
}

// a directory is ignore if it is in the m_ignoredDirectories list or if it is contained by an ignored directory
bool TracePack::isIgnored( const ctPath &i_directory ) const
{
   return contains( m_ignoredDirectories, i_directory );
}

// a filename is ignore if it is in the m_ignoredFiles list or if it is contained by an ignored directory
bool TracePack::isIgnored( const cElFilename &i_file ) const
{
   // check if i_file is in m_ignoredFiles
   list<cElFilename>::const_iterator itFile = m_ignoredFiles.begin();
   while ( itFile!=m_ignoredFiles.end() && ( (*itFile)<i_file ) ) itFile++;
   if ( *itFile==i_file ) return true;
   
   // check if i_file is contained by an ignored directory
   list<ctPath>::const_iterator itDirectory = m_ignoredDirectories.begin();
   while ( itDirectory!=m_ignoredDirectories.end() && (*itDirectory)<i_file.m_path )
   {
      if ( itDirectory->contains( i_file ) ) return true;
      itDirectory++;
   }
   return ( itDirectory!=m_ignoredDirectories.end() && (*itDirectory)==i_file.m_path );
}

void TracePack::compareWithItemsOnDisk( unsigned int i_iRegistry, const ctPath &i_onDiskPath, const list<Registry::Item> &i_itemToCompare, list<Registry::Item> &o_differentFromDisk )
{
   o_differentFromDisk.clear();
   list<Registry::Item>::const_iterator itItem = i_itemToCompare.begin();
   while ( itItem!=i_itemToCompare.end() )
   {
      if ( itItem->m_type==TracePack::Registry::TD_Added ||
           itItem->m_type==TracePack::Registry::TD_Modified )
      {
	 copyItemOnDisk( i_iRegistry, itItem->m_filename );
	 cElFilename fileFromPack( m_anchor, itItem->m_filename ),
	             fileOnDisk( i_onDiskPath, itItem->m_filename );
	 if ( !is_equivalent( fileFromPack, fileOnDisk ) )
	    o_differentFromDisk.push_back( *itItem );
      }
      itItem++;
   }
}
   

//--------------------------------------------
// related functions
//--------------------------------------------

bool write_file( const cElFilename &i_filenameIn, ostream &io_fOut, unsigned int i_expectedSize )
{
   ifstream fIn( i_filenameIn.str_unix().c_str(), ios::binary );
   
   if ( !fIn )
   {
      #ifdef __DEBUG_TRACE_PACK
	 cerr << RED_DEBUG_ERROR << "write_file: unable to open [" << i_filenameIn.str_unix() << "] for reading" << endl;
	 exit(EXIT_FAILURE);
      #endif
      return false;
   }
   
   return write_file( fIn, io_fOut, i_expectedSize );
}

bool write_file( istream &io_fIn, ostream &io_fOut, unsigned int i_length )
{
   #ifdef __DEBUG_TRACE_PACK
      if ( !io_fIn )
      {
	 cerr << RED_DEBUG_ERROR << "write_file: io_fIn is not ready for reading" << endl;
	 exit(EXIT_FAILURE);
      }
      if ( !io_fOut )
      {
	 cerr << RED_DEBUG_ERROR << "write_file: io_fOut is not ready for writing" << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
   
   const unsigned int buffer_size = 1000000;
   unsigned int remaining = i_length;
   vector<char> buffer( buffer_size );
   while ( !io_fIn.eof() && remaining )
   {
      if ( buffer_size>remaining )
	 io_fIn.read( buffer.data(), remaining );
      else
	 io_fIn.read( buffer.data(), buffer_size );
      streamsize nbRead = io_fIn.gcount();
      if ( nbRead<0 )
      {	 
	 #ifdef __DEBUG_TRACE_PACK
	    cerr << RED_DEBUG_ERROR << "write_file: unable to read in input stream" << endl;
	    exit(EXIT_FAILURE);
	 #endif
	 return false;
      }
      io_fOut.write( buffer.data(), nbRead );
      remaining -= (unsigned int)nbRead;
   }
   
   #ifdef __DEBUG_TRACE_PACK
      if ( remaining!=0 )
      {
	 cerr << RED_DEBUG_ERROR << "write_file: " << remaining << " bytes need to be read but end-of-file is reached" << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
   
   return remaining==0;
}

string TD_Type_to_string( TracePack::Registry::TD_Type i_type )
{
   switch (i_type)
   {
   case TracePack::Registry::TD_Added: return "TD_Added";
   case TracePack::Registry::TD_Removed: return "TD_Removed";
   case TracePack::Registry::TD_Modified: return "TD_Modified";
   case TracePack::Registry::TD_State: return "TD_State";
   }
   return "unknown";
}

ostream & operator <<( ostream &s, const cElHour &h )
{
   return ( s << setw(2) << setfill('0') << h.H() << ':'
              << setw(2) << setfill('0') << h.M() << ':'
              << setw(2) << setfill('0') << h.S() );
}

ostream & operator <<( ostream &s, const cElDate &d )
{
   return ( s << setw(2) << setfill('0') << d.Y() << '-'
              << setw(2) << setfill('0') << d.M() << '-'
	      << setw(2) << setfill('0') << d.D() << ' '
	      << d.H() );
}

bool load_script( const cElFilename &i_filename, std::list<cElCommand> &o_commands )
{
   ifstream f( i_filename.str_unix().c_str(), ios::binary );
   if ( !f )
   {
      #ifdef __DEBUG_TRACE_PACK
	 cerr << RED_DEBUG_ERROR << "load_script: script file [" << i_filename.str_unix() << "] open for reading" << endl;
	 exit(EXIT_FAILURE);
      #endif
      return false;
   }
   string line;
   cElCommand cmd;
   while ( !f.eof() )
   {
      // read a line in the script file
      getline( f, line );
      if ( line.length()!=0 && line[1]!='#' )
      {
	 // use the string to create a command with RawString tokens and add it to the list
	 cmd.set_raw( line );	 
	 o_commands.push_back( cmd );
      }
   }
   return true;
}

bool is_equivalent( const cElFilename &i_a, const cElFilename &i_b )
{
   #if (ELISE_POSIX)
      cElCommand cmd;
      cmd.add( ctRawString("cmp") );
      cmd.add( ctRawString( "-s" ) );
      cmd.add( ctRawString( i_a.str_unix() ) );
      cmd.add( ctRawString( i_b.str_unix() ) );
      return cmd.system();
   #else
      // __TODO
      not implemented
   #endif
}

// returns if an element of i_pathList contains i_path
bool contains( const std::list<ctPath> &i_pathList, const ctPath &i_path )
{
   // directories that may contain i_directory are lesser than it
   list<ctPath>::const_iterator itPath = i_pathList.begin();
   while ( itPath!=i_pathList.end() && ( (*itPath)<i_path ) )
   {
      if ( itPath->contains( i_path ) ) return true;
      itPath++;
   }
   return ( itPath!=i_pathList.end() && ( (*itPath)==i_path ) );
}

bool file_to_chunk_list( const string &i_filename, unsigned int i_fileSize, U_INT4 &i_firstMax, U_INT4 i_newMax, std::list<FileChunk> &o_chunks )
{
   const U_INT4 fullHeaderSize = FileChunk::headerSize+FileChunk::dataHeaderSize;
   const U_INT4 filenameRawSize = string_raw_size( i_filename );
   i_fileSize += filenameRawSize; // filename is considered data
   
   if ( i_newMax<fullHeaderSize+filenameRawSize )
   {
      #ifdef __DEBUG_TRACE_PACK
	 cerr << RED_DEBUG_ERROR << "file_to_chunk_list: i_newMax = " << i_newMax << " too low, first chunk's min size = " << fullHeaderSize+filenameRawSize << endl;
      #endif
      return false;
   }
   
   // first chunk
   U_INT4 dataSize;
   if ( fullHeaderSize+filenameRawSize<=i_firstMax )
      dataSize = std::min( i_firstMax-fullHeaderSize, i_fileSize );
   else
      dataSize = std::min( i_newMax-fullHeaderSize, i_fileSize );
   o_chunks.push_back( FileChunk( i_filename, dataSize<i_fileSize/*hasMore*/, dataSize ) );
   i_fileSize -= dataSize;
   
   if ( i_fileSize==0 )
   {
      unsigned int fullSize = o_chunks.back().fullSize();
      i_firstMax = ( fullSize<i_firstMax?i_firstMax-fullSize:i_newMax-fullSize );
      return true;
   }
   
   dataSize = i_newMax-fullHeaderSize;
   unsigned int nbChunks = i_fileSize/dataSize;
   if ( nbChunks!=0 )
   {
      FileChunk chunk( true/*hasMore*/, dataSize );
      while ( nbChunks-- ) o_chunks.push_back( chunk );
   }
   dataSize = i_fileSize%dataSize;
   if ( dataSize!=0 ) o_chunks.push_back( FileChunk( false/*hasMore*/, dataSize ) );
   i_firstMax = i_newMax-o_chunks.back().fullSize();
   o_chunks.back().m_hasMore = false;
   
   return true;
}

bool chunk_list_to_file( const cElFilename &i_outputFilename, const std::list<FileChunk> &o_chunks )
{
   ofstream fdst( i_outputFilename.str_unix().c_str(), ios::binary );
   if ( !fdst ) return false;
   std::list<FileChunk>::const_iterator itChunk = o_chunks.begin();
   while ( itChunk!=o_chunks.end() )
      if ( !( *itChunk++ ).outputData( fdst ) ) return false;
   return true;
}

string FileChunkType_to_string( FileChunk::FileChunkType i_type )
{
   switch ( i_type )
   {
   case FileChunk::FCT_Registry: return string("FCT_Registry");
   case FileChunk::FCT_Ignore: return string("FCT_Ignore");
   case FileChunk::FCT_Data: return string("FCT_Data");
   default: return string("unknown");
   }
}
