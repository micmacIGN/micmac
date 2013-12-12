#ifndef __TRACE_PACK__
#define __TRACE_PACK__

// __DEL
#define _DEBUG

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
	 cElFilename  m_dataFile;
	 unsigned int m_dataOffset;
	 unsigned int m_dataSize;
	 
	 Item( const cElFilename &i_filename,
	       TD_Type i_type,
	       const cElDate &i_date,
	       const cElFilename &i_dataFile,
	       unsigned int i_dataOffset,
	       unsigned int i_dataSize );
	 void reset_date();
	 void apply( TracePack::Registry &o_actions );
	 void dump( std::ostream &io_ostream=std::cout, const std::string &i_prefix=string() ) const;
      };
      
      list<Item> m_items;
      
	   Registry();
      void write_v1( ostream &io_stream ) const;
      void read_v1( istream &io_stream );
      void reset_dates();
      Item & add( const Item &i_item );
      void stateDirectory( const cElPath &i_path );
      void difference( const TracePack::Registry &i_a, const TracePack::Registry &i_b );
      void apply( const TracePack::Registry &i_actions );
      inline size_t size() const;
      void dump( std::ostream &io_ostream=std::cout, const std::string &i_prefix=string() ) const;
            
      #ifdef _DEBUG
	 void check_sorted() const;
      #endif
   };
   
private:
   bool read_v1( std::istream &f, bool i_reverseByteOrder );
   void write_v1( std::ostream &f, bool i_reverseByteOrder=false ) const;
   
   cElFilename    m_filename;
   cElDate        m_date;
   list<Registry> m_registries;

public:
   TracePack();
   
   inline const cElFilename & filename() const;
   inline const cElDate & date() const;
   const Registry & getRegistry( unsigned int i_iRegistry ) const;
   
   inline unsigned int nbStates() const;
   void getState( unsigned int i_iState, Registry &o_state ) const;
   void addStateFromDirectory( const cElPath &i_path );
   void dump( std::ostream &io_ostream=std::cout, const std::string &i_prefix=string() ) const;
   
   bool load( const cElFilename &i_filename );
   bool save( const cElFilename &i_filename, unsigned int i_version=g_versioned_headers_list[VFH_TracePack].last_handled_version, bool i_MSBF=MSBF_PROCESSOR() ) const;
};

string TD_Type_to_string( TracePack::Registry::TD_Type i_type );

ostream & operator <<( ostream &s, const cElHour &h );

ostream & operator <<( ostream &s, const cElDate &d );

#include "TracePack.inline.h"

#endif
