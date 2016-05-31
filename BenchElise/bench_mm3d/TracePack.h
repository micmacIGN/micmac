#ifndef __TRACE_PACK__
#define __TRACE_PACK__

#define __DEBUG_TRACE_PACK

#include "StdAfx.h"

#include "VersionedFileHeader.h"
#include "ChunkStream.h"

class TracePack
{
public:
   class Registry
   {
   public:
		typedef enum{
			TD_Added,
			TD_Removed,
			TD_Modified,
			TD_State
		} TD_Type;

      class Item
      {
      public:
			cElFilename           m_filename;
			TD_Type               m_type;
			cElDate               m_date;
			mode_t                m_rights;
			U_INT8                m_dataLength;
			ChunkStream::FileItem m_data;
	 
			inline Item();
			inline Item( const cElFilename &i_filename,
						  TD_Type i_type,
						  const cElDate &i_date,
					mode_t i_rights,
						  U_INT8 i_dataLength,
					const ChunkStream::FileItem &i_data=ChunkStream::FileItem() );
			void reset_date();
			void apply( TracePack::Registry &o_actions );
			void dump( std::ostream &io_ostream=std::cout, const std::string &i_prefix=string() ) const;
			bool copyToDirectory( const ctPath &i_directory ) const;
			bool trace_compare( const Item &i_b ) const;
			bool applyToDirectory( const ctPath &i_path ) const;

			inline bool hasData() const;

			// Item<->raw data convertion
			U_INT8 raw_size() const;
			void to_raw_data( bool i_reverseByteOrder, char *&o_rawData ) const;
			void from_raw_data( const char *&o_rawData, bool i_reverseByteOrder );
      };
      
      list<Item> m_items;
      cElCommand m_command;
      bool       m_hasBeenWritten;
      
      Registry();

      // input/output methods
      U_INT8 raw_size() const;
      void to_raw_data( bool i_reverseByteOrder, char *&o_rawData ) const;
      void from_raw_data( const char *i_rawData, bool i_reverseByteOrder );
      
      void reset_dates();
      Item & add( const Item &i_item );
      void stateDirectory( const ctPath &i_path );
      bool applyToDirectory( const ctPath &i_path ) const;
      void apply( const TracePack::Registry &i_actions );
      inline size_t size() const;
      void dump( std::ostream &io_ostream=std::cout, const std::string &i_prefix=string() ) const;
      Item * getItem( const cElFilename &i_filename );
      bool trace_compare( const Registry &i_b ) const;
      void difference( const TracePack::Registry &i_a, const TracePack::Registry &i_b );
      // add data references to the item with the specified filename
      // return false if no item could be found with specified filename, if lengthes do not match of if item type does not accept data
      bool setItemData( const ChunkStream::FileItem &i_data );

      static bool compare_states( const TracePack::Registry &i_a, const TracePack::Registry &i_b,
                                  list<Registry::Item> &o_onlyInA, list<Registry::Item> &o_onlyInB, list<Registry::Item> &o_inBoth );

		#ifdef __DEBUG_TRACE_PACK
			void check_sorted() const;
		#endif
   };
   
private:
   typedef enum{
      SIT_Registry = 0,
      SIT_Ignore = 1
   } StreamItemType;
   static std::string StreamItemType_to_string( StreamItemType i_type );
   
   static const U_INT8 maxPackFileSize = 1e9;

   cElFilename    m_filename;
   ctPath         m_anchor;
   cElDate        m_date;
   list<Registry> m_registries;
   // ignored files/directories are considered relatively to m_anchor
   std::list<ctPath>      m_ignoredDirectories;
   std::list<cElFilename> m_ignoredFiles;

   // read/write object
   unsigned int m_versionNumber;
   bool         m_isMSBF;
   ChunkStream  m_stream;
      
   // writing synchronization
   bool     m_ignoredItemsChanged;
   
   bool writeData( Registry &i_registry, bool i_reverseByteOrder );
   bool writeData( Registry::Item &i_item, bool i_reverseByteOrder );
   
   void add_ignored( const ctPath &i_directory );
   void add_ignored( const cElFilename &i_filename );
   
   void ignored_lists_to_buffer( bool i_reverseByteOrder, std::vector<char> &o_buffer ) const;
   void ignored_lists_from_buffer( bool i_reverseByteOrder, const std::vector<char> &o_buffer );
   
   bool read_v1( U_INT8 i_offset, bool i_reverseByteOrder );
   bool write_v1( bool i_reverseByteOrder );

	void remove_ignored_items( Registry &io_registry );

public:
   TracePack( const cElFilename &i_packname, const ctPath &i_anchor, U_INT8 i_maxFileSize=maxPackFileSize );
   
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
   bool save();
   bool copyItemOnDisk( unsigned int i_iState, const cElFilename &i_itemName );
   void compareWithItemsOnDisk( unsigned int i_iRegistry, const ctPath &i_onDiskPath, const std::list<Registry::Item> &i_itemToCompare, std::list<Registry::Item> &o_differentFromDisk );
   
   inline U_INT4      getNbPackFiles() const;
   inline cElFilename packFilename( const U_INT4 &i_iFile ) const;
   
   bool isIgnored( const ctPath &i_directory ) const;
   bool isIgnored( const cElFilename &i_file ) const;
   
   bool addIgnored( std::list<ctPath> &io_ignoredDirectories );
   bool addIgnored( std::list<cElFilename> &io_ignoredFiles );

	// saving format can be set successfully only if the stream is empty
   bool setSavingFormat( unsigned int i_version, bool i_isMSBF );
   inline const list<cElFilename> & getIgnoredFiles() const;
   inline const list<ctPath> & getIgnoredDirectories() const;
   inline void setAnchor( const ctPath &i_path );
};

bool load_script( const cElFilename &i_filename, std::list<cElCommand> &o_commands );

inline void write_string( const std::string &i_str, bool i_reverseByteOrder, std::ostream &io_fOut );
inline void read_string( std::istream &io_fIn, bool i_reverseByteOrder, std::string &o_str );

std::string TD_Type_to_string( TracePack::Registry::TD_Type i_type );

bool is_equivalent( const cElFilename &i_a, const cElFilename &i_b );

ostream & operator <<( ostream &s, const cElHour &h );

ostream & operator <<( ostream &s, const cElDate &d );

// returns if an element of i_pathList contains i_path
bool oneIsAncestorOf( const std::list<ctPath> &i_pathList, const ctPath &i_path );

// uint8<->raw
inline void uint8_to_raw_data( const U_INT8 &i_v, bool i_reverseByteOrder, char *&o_rawData );
inline void uint8_from_raw_data( const char *&io_rawData, bool i_reverseByteOrder, U_INT8 &o_v );

#include "TracePack.inline.h"

#endif
