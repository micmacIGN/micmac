#ifndef __CHUNK_STREAM__
#define __CHUNK_STREAM__

#include <fstream>
#include <string>
#include <list>

#include "general/cElCommand.h"

#define __DEBUG_CHUNK_STREAM
#define __DEBUG_CHUNK_STREAM_OUTPUT_HEADERS

class ChunkStream
{
private:
   cElFilename m_filename;
   U_INT8      m_maxFileSize;
   bool        m_reverseByteOrder;
   
   static const unsigned int chunkHeaderSize = 10; // unsigned char type (1), bool hasMore(1), U_INT8 chunkSize(8)
   static const unsigned char dataHeaderFlag = 128;
   static const unsigned char dataFilenameHeaderFlag = 64;
   
   // for read/write ( only set after a successful readOpen() or writeOpen() )
   U_INT4    m_currentFileIndex;
   U_INT8    m_remaining;
      // for reading ( only set after a successful readOpen() )
   std::ifstream  m_inputStream;
   U_INT8    m_lastFileIndex;
      // for writing ( only set after a successful writeOpen() )
   std::ofstream m_outputStream;
      
   void write_header( unsigned char i_type, bool i_hasMore, U_INT8 i_size );
	bool read_header( unsigned char &o_type, bool &o_hasMore, U_INT8 &o_chunkSize );
	bool read_buffer( unsigned char i_itemType, std::vector<char> &o_buffer );
   bool open_next_output_file();
   bool open_next_input_file();
   inline bool prepare_next_read();
   inline bool prepare_next_write();
public:
	class Item
	{
	public:
		virtual bool isFileItem() const = 0;
		inline virtual ~Item();
		template <class T> T & specialize();
	};
	
	class FileItem : public Item
	{
	public:
		class Chunk
		{
		public:
			cElFilename    m_filename;
			std::streampos m_offset;
			U_INT8         m_length;
			
			Chunk( const cElFilename &i_filename, std::streampos i_offset, U_INT8 i_length );
			inline bool operator ==( const Chunk &i_b ) const;
			inline bool operator !=( const Chunk &i_b ) const;
		};
		
		cElFilename      m_storedFilename;
		U_INT8           m_length;
		std::list<Chunk> m_chunks;
		
		FileItem( const cElFilename &i_storedFilename=cElFilename(), U_INT8 i_length=0 );
		bool isFileItem() const;
		bool copyToFile( const cElFilename &i_dstFilename ) const;
		bool operator ==( const FileItem &i_b ) const;
		inline bool operator !=( const FileItem &i_b ) const;
	};
	
	class BufferItem : public Item
	{
	public:
		unsigned char     m_type;
		std::vector<char> m_buffer;
		
		BufferItem( unsigned char i_type );
		bool isFileItem() const;
	};

   ChunkStream( const cElFilename &i_filename, U_INT8 i_maxFileSize, bool i_reverseByteOrder );

	// readOpen must be called before any reading attempt
   bool readOpen( U_INT4 i_iStartFile, U_INT8 i_startOffset );
	// read next Item
	Item * read();
	// readOpen and read all items until the end of the stream is reached
	// o_items's pointers must be freed by the caller
	bool read( U_INT4 i_iStartIndex, U_INT8 i_startOffset, std::list<Item*> &o_items );
	// end of the stream in reading mode (in writing mode the stream as nearly no end)
   bool isEndOfStream() const;
   // writeOpen must be called before any writing attempt
   bool writeOpen();
   void close();
   inline void setReverseByteOrder( bool i_isReverse );
   inline bool getReverseByteOrder() const;
   
   // insert file to the end of the stream
   // o_fileItem is set to the written item as if it was just read
   bool write_file( const cElFilename &i_srcFilename, const cElFilename &i_toStoreFilename, FileItem &o_fileItem );
   bool write_file( const cElFilename &i_srcFilename, const cElFilename &i_toStoreFilename );
   
   // insert a buffer to the end of the stream
   bool write_buffer( unsigned char i_type, const std::vector<char> &i_buffer );
   
   // the stream is opened for reading
   bool isReadOpen() const;
   
   // the stream is opened for writing
   bool isWriteOpen() const;
   
   // the stream is opened for reading of writing
   bool isOpen() const;
      
   // delete all files associated to the stream
   bool remove();
   
   inline U_INT8 maxFileSize() const;
   
   // get the name of the stream file of index i_fileIndex
   cElFilename getFilename( U_INT4 i_fileIndex ) const;
   
   // get the number of files currently composing the stream
   U_INT4 getNbFiles() const;
};

inline void write_uint4( U_INT4 i_ui4, bool i_reverseByteOrder, std::ostream &io_fOut );
inline void read_uint4( std::istream &io_fIn, bool i_reverseByteOrder, U_INT4 &i_ui4 );
inline void write_uint8( U_INT8 i_ui8, bool i_reverseByteOrder, std::ostream &io_fOut );
inline void read_uint8( std::istream &io_fIn, bool i_reverseByteOrder, U_INT8 &i_ui8 );

// transfer i_length bytes from io_src to io_dst
bool stream_copy( std::istream &io_src, std::ostream &io_dst, U_INT8 i_length );

// delete all pointers in the list then clear the list
void clear_item_list( std::list<ChunkStream::Item*> io_items );

#include "ChunkStream.inline.h"

#endif
