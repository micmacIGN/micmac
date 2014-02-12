#ifndef __TRACE_PACK__
#define __TRACE_PACK__

// __DEL
#define __DEBUG_TRACE_PACK

#include "StdAfx.h"

#include "general/cElCommand.h"
#include "VersionedFileHeader.h"

class FileChunk
{
private:
   inline void _set_dataChunk( U_INT i_fullSize );

   inline bool read_chunk( std::istream &io_in );
   bool read_dataChunk( std::istream &io_in, bool i_reverseByteOrder, const cElFilename &i_filename );   
public:
   typedef struct{
      U_INT4 iFile;
      U_INT4 offset;
   } offset_t;
   
   typedef enum{
      FCT_Registry = 0,
      FCT_Ignore,
      FCT_Data
   } FileChunkType;
   
   FileChunkType m_type;
   U_INT4        m_contentSize; // size of the chunk after reading type+size
   
   // for data chunk
      // for writing
   bool	           m_isFirst;
   std::string     m_filename;
   U_INT4          m_filenameRawSize;
   bool            m_hasMore;
   U_INT4          m_dataSize;
      // direct access coordinates (available when chunk has been written/read)
   cElFilename m_inStreamFilename;
   streampos   m_inStreamOffset;
   
   // for Ignore/Registry chunks
   vector<char> m_rawData;
   
   // constructor for a first data chunk
   inline FileChunk();
   inline FileChunk( FileChunkType i_type, const std::vector<char> &i_rawData );
   
   void trace( std::ostream &io_dst=std::cout ) const;
   inline U_INT4 onDiskData() const;
   inline U_INT4 toSkip() const;
   bool outputData( std::ostream &io_dst ) const;
   
   // construct FCT_Data chunk
   inline FileChunk( const std::string &i_filename, bool i_hasMore, U_INT4 i_dataSize );
   inline FileChunk( bool i_hasMore, U_INT4 i_dataSize );
   
   inline void set_dataChunk( U_INT i_fullSize );
   inline void set_dataChunk( U_INT i_fullSize, const std::string &i_filename );

   static const U_INT4 headerSize = 1/*type*/+4/*size*/;
   static const U_INT4 dataHeaderSize = 1/*isFirst*/+1/*hasMore*/;
   inline unsigned int fullSize() const;
   
   static bool readAllChunks( const cElFilename &i_filename, streampos i_offset, bool i_reverseByteOrder, std::list<FileChunk> &o_chunks );
   static bool readAllChunks( const std::list<cElFilename> &i_filenames, streampos i_offset, bool i_reverseByteOrder, std::list<FileChunk> &o_chunks );
   
   // chunk-reading process is in to steps
   bool read( const cElFilename &i_filename, std::istream &io_in, bool i_reverseByteOrder );
   void write_chunk( bool i_reverseByteOrder, std::ostream &io_out ) const;
   bool write_dataChunk( const cElFilename &i_inStreamFilename, std::istream &io_in, bool i_reverseByteOrder, std::ostream &io_out );
   bool dataChunk_copy( std::istream &io_src, std::ostream &io_dst ) const;
};

class ChunkStream
{
private:
   cElFilename  m_filename;
   U_INT4       m_maxFileSize;
   bool         m_reverseByteOrder;
   U_INT4       m_iFirstChunkFile;
   streampos    m_offsetInFirstFile;
   
   bool open_new_or_append( unsigned int &io_iLastKnownFile, unsigned int i_minRemainingSize, cElFilename &o_filename, unsigned int &o_remainingSize, std::ofstream &io_fdst );
   
public:
   ChunkStream( const cElFilename &i_filename, U_INT4 i_maxFileSize, bool i_reverseByteOrder );
   
   cElFilename filename( unsigned int i_iFile ) const;
   
   unsigned int getNbFiles() const;
   
   bool setOffset( U_INT4 i_iFirstChunkFile, streampos i_offset );
   
   bool readChunks( std::list<FileChunk> &o_chunks );
   
   bool writeChunks( std::list<FileChunk> &io_chunks );
   
   bool remove() const;
};

bool file_to_chunk_list( const std::string &i_filename, unsigned int i_fileSize, unsigned int &i_firstMax, unsigned int i_otherMax, std::list<FileChunk> &o_chunks );
bool chunk_list_to_file( const cElFilename &i_outputFilename, const std::list<FileChunk> &o_chunks );

class TracePack
{
public:
   class Registry
   {
   public:
      typedef enum
      {
	 TD_Added,
	 TD_Removed,
	 TD_Modified,
	 TD_State
      } TD_Type;

      class Item
      {
      public:
	 cElFilename         m_filename;
	 TD_Type             m_type;
	 cElDate             m_date;
	 mode_t              m_rights;
	 streampos           m_dataOffset; // __TO_V1
	 list<FileChunk>     m_data;
	 unsigned int        m_dataSize;
	 
	 inline Item();
	 inline Item( const cElFilename &i_filename,
	              TD_Type i_type,
	              const cElDate &i_date,
		      mode_t i_rights,
	              streampos i_dataOffset,
	              unsigned int i_dataSize );
	 void reset_date();
	 void apply( TracePack::Registry &o_actions );
	 void dump( std::ostream &io_ostream=std::cout, const std::string &i_prefix=string() ) const;
	 bool copyToDirectory( const cElFilename &i_packname, const ctPath &i_directory ) const;
	 bool trace_compare( const Item &i_b ) const;
	 bool applyToDirectory( const cElFilename &i_packname, const ctPath &i_path ) const;
	 
	 inline bool hasData() const;
	 bool computeData( unsigned int i_remainingDst );
	 
	 // input/output methods
	 unsigned int raw_size() const;
	 void to_raw_data( bool i_reverseByteOrder, char *&o_rawData ) const;
	 void from_raw_data( char *&o_rawData, bool i_reverseByteOrder );
      };
      
      list<Item> m_items;
      bool       m_hasBeenWritten; // __TO_V1
      cElCommand m_command;
      
	   Registry();

      // input/output methods
      unsigned int raw_size() const;
      void to_raw_data( bool i_reverseByteOrder, char *&o_rawData ) const;
      void from_raw_data( char *&io_rawData, bool i_reverseByteOrder );
      
      void reset_dates();
      Item & add( const Item &i_item );
      void stateDirectory( const ctPath &i_path );
      bool applyToDirectory( const cElFilename &i_packname, const ctPath &i_path ) const;
      void apply( const TracePack::Registry &i_actions );
      inline size_t size() const;
      void dump( std::ostream &io_ostream=std::cout, const std::string &i_prefix=string() ) const;
      Item * getItem( const cElFilename &i_filename );
      bool trace_compare( const Registry &i_b ) const;
      void difference( const TracePack::Registry &i_a, const TracePack::Registry &i_b );
      
      static void compare_states( const TracePack::Registry &i_a, const TracePack::Registry &i_b,
                                  list<Registry::Item> &o_onlyInA, list<Registry::Item> &o_onlyInB, list<Registry::Item> &o_inBoth );
      
      void getChunk( bool i_reverseByteOrder, FileChunk &o_chunk ) const;
                  
      // __TO_V1
      bool write_v1( ofstream &io_stream, const ctPath &i_anchor, bool i_reverseByteOrder );
      void read_v1( istream &io_stream, bool i_reverseByteOrder );
      
      bool write_v2( ofstream &io_stream, const ctPath &i_anchor, bool i_reverseByteOrder );
      void read_v2( istream &io_stream, bool i_reverseByteOrder );
      
      #ifdef __DEBUG_TRACE_PACK
	 void check_sorted() const;
      #endif
   };
   
private:
   static const U_INT4 maxPackFileSize = 1e9;

   cElFilename    m_filename;
   cElFilename    m_lastFilename;
   U_INT4         m_iLastFile;
   unsigned int   m_remainingSpace;
   ctPath         m_anchor;
   cElDate        m_date;
   list<Registry> m_registries;
   // ignored files/directories are considered relatively to m_anchor
   std::list<ctPath>      m_ignoredDirectories;
   std::list<cElFilename> m_ignoredFiles;
   // stream associated to m_lastFilename, open only during saving process
   std::ofstream m_packOutputStream;
   // stream associated to m_lastFilename, open only during saving process
   std::ifstream m_packInputStream;
      
   // writing synchronization
   bool     m_writeInUpdateMode; // __TO_V1
   int      m_nbRegistriesOffset; // __TO_V1
   bool     m_writeIgnoredItems;

   bool newPackFile();
   bool openNextPackFile();
   
   bool writeData( Registry &i_registry, bool i_reverseByteOrder );
   bool writeData( Registry::Item &i_item, bool i_reverseByteOrder );

   void save_nbRegistries() const; // __TO_V1
   
   void add_ignored( const ctPath &i_directory );
   void add_ignored( const cElFilename &i_filename );

   static void read_offset( std::istream &io_istream, FileChunk::offset_t &o_offset, bool i_reverseByteOrder );
   static void write_offset( std::ostream &io_ostream, FileChunk::offset_t i_offset, bool i_reverseByteOrder );
   
   // __TO_V1
   void read_v1( std::istream &f, bool i_reverseByteOrder );
   bool write_v1( std::ofstream &f, bool i_reverseByteOrder );
   bool read_v2( std::ifstream &f, bool i_reverseByteOrder );
   bool write_v2( bool i_reverseByteOrder );

public:
   TracePack( const cElFilename &i_packname, const ctPath &i_anchor );
   
   inline const cElDate & date() const;
   const Registry & getRegistry( unsigned int i_iRegistry ) const;
   
   inline unsigned int nbStates() const;
   void getState( unsigned int i_iState, Registry &o_state ) const;
   void addState( const cElCommand &i_command=cElCommand() );
   void setState( unsigned int i_iState );
   void dump( std::ostream &io_ostream=std::cout, const std::string &i_prefix=string() ) const;
   bool trace_compare( const TracePack &i_b ) const;
   void getAllCommands( std::vector<cElCommand> &o_commands ) const;
   
   bool load();
   bool save(unsigned int i_version=g_versioned_headers_list[VFH_TracePack].last_handled_version, bool i_MSBF=MSBF_PROCESSOR() );
   bool copyItemOnDisk( unsigned int i_iState, const cElFilename &i_itemName );
   void compareWithItemsOnDisk( unsigned int i_iRegistry, const ctPath &i_onDiskPath, const std::list<Registry::Item> &i_itemToCompare, std::list<Registry::Item> &o_differentFromDisk );
   
   unsigned int getNbPackFiles() const;
   
   bool isIgnored( const ctPath &i_directory ) const;
   bool isIgnored( const cElFilename &i_file ) const;
   
   void addIgnored( std::list<ctPath> &io_ignoredDirectories );
   void addIgnored( std::list<cElFilename> &io_ignoredFiles );
   
   std::string packFilename( const U_INT4 &i_iFile ) const;
};

bool load_script( const cElFilename &i_filename, std::list<cElCommand> &o_commands );

bool write_file(  const cElFilename &i_filenameIn, std::ostream &io_fOut, unsigned int i_expectedSize );
bool write_file( std::istream &io_fIn, std::ostream &io_fOut, unsigned int i_expectedSize );

inline void write_uint4( U_INT4 i_ui4, bool i_reverseByteOrder, std::ostream &io_fOut );
inline void read_uint4( std::istream &io_fIn, bool i_reverseByteOrder, U_INT4 &i_ui4 );

inline void write_string( const std::string &i_str, bool i_reverseByteOrder, std::ostream &io_fOut );
inline void read_string( std::istream &io_fIn, bool i_reverseByteOrder, std::string &o_str );

std::string TD_Type_to_string( TracePack::Registry::TD_Type i_type );
std::string FileChunkType_to_string( FileChunk::FileChunkType i_type );

bool is_equivalent( const cElFilename &i_a, const cElFilename &i_b );

ostream & operator <<( ostream &s, const cElHour &h );

ostream & operator <<( ostream &s, const cElDate &d );

// returns if an element of i_pathList contains i_path
bool contains( const std::list<ctPath> &i_pathList, const ctPath &i_path );

#include "TracePack.inline.h"

#endif
