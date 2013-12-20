#ifndef __TRACE_PACK__
#define __TRACE_PACK__

// __DEL
#define __DEBUG_TRACE_PACK

#include "StdAfx.h"

#include "general/cElCommand.h"
#include "VersionedFileHeader.h"

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
	 cElFilename  m_filename;
	 TD_Type      m_type;
	 cElDate      m_date;
	 streampos    m_dataOffset;
	 unsigned int m_dataSize;
	 
	 inline Item( const cElFilename &i_filename,
	              TD_Type i_type,
	              const cElDate &i_date,
	              streampos i_dataOffset,
	              unsigned int i_dataSize );
	 void reset_date();
	 void apply( TracePack::Registry &o_actions );
	 void dump( std::ostream &io_ostream=std::cout, const std::string &i_prefix=string() ) const;
	 bool copyToDirectory( const cElFilename &i_packname, const cElPath &i_directory ) const;
	 bool compare( const Item &i_b ) const;
	 bool applyToDirectory( const cElFilename &i_packname, const cElPath &i_path ) const;
      };
      
      list<Item> m_items;
      bool       m_hasBeenWritten;
      cElCommand m_command;
      
	   Registry();
      bool write_v1( ofstream &io_stream, const cElPath &i_anchor );
      void read_v1( istream &io_stream );
      void reset_dates();
      Item & add( const Item &i_item );
      void stateDirectory( const cElPath &i_path );
      bool applyToDirectory( const cElFilename &i_packname, const cElPath &i_path ) const;
      void difference( const TracePack::Registry &i_a, const TracePack::Registry &i_b );
      void apply( const TracePack::Registry &i_actions );
      inline size_t size() const;
      void dump( std::ostream &io_ostream=std::cout, const std::string &i_prefix=string() ) const;
      Item * getItem( const cElFilename &i_filename );
      bool compare( const Registry &i_b ) const;
            
      #ifdef __DEBUG_TRACE_PACK
	 void check_sorted() const;
      #endif
   };
   
private:
   void read_v1( std::istream &f );
   bool write_v1( std::ofstream &f );
   
   cElFilename    m_filename;
   cElPath        m_anchor;
   cElDate        m_date;
   list<Registry> m_registries;
   
   // writing synchronization
   bool m_writeInUpdateMode;
   int  m_nbRegistriesOffset;

   void save_nbRegistries() const;

public:
   TracePack( const cElFilename &i_filename, const cElPath &i_anchor );
   
   inline const cElDate & date() const;
   const Registry & getRegistry( unsigned int i_iRegistry ) const;
   
   inline unsigned int nbStates() const;
   void getState( unsigned int i_iState, Registry &o_state ) const;
   void addState( const cElCommand &i_command=cElCommand() );
   void setState( unsigned int i_iState );
   void dump( std::ostream &io_ostream=std::cout, const std::string &i_prefix=string() ) const;
   bool compare( const TracePack &i_b ) const;
   void getAllCommands( std::vector<std::string> &o_commands ) const;
   
   bool load();
   bool save(unsigned int i_version=g_versioned_headers_list[VFH_TracePack].last_handled_version, bool i_MSBF=MSBF_PROCESSOR() );
   bool copyItemOnDisk( unsigned int i_iState, const cElFilename &i_itemName );
};

bool load_script( const cElFilename &i_filename, std::list<cElCommand> &o_commands );

bool write_file(  const cElFilename &i_filenameIn, std::ostream &io_fOut, unsigned int i_expectedSize );
bool write_file( std::istream &io_fIn, std::ostream &io_fOut, unsigned int i_expectedSize );

string TD_Type_to_string( TracePack::Registry::TD_Type i_type );

ostream & operator <<( ostream &s, const cElHour &h );

ostream & operator <<( ostream &s, const cElDate &d );

#include "TracePack.inline.h"

#endif
