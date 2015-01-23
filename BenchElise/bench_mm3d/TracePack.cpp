#include "TracePack.h"

#include "StdAfx.h"

#include "general/cElCommand.h"
#include "ChunkStream.h"

using namespace std;

#ifdef __DEBUG_TRACE_PACK
   const string __difference_illegal_item_message = "TracePack::Registry::difference: illegal action item";
   const string __apply_illegal_action_message = "TracePack::Registry::apply: illegal action item";
   const string __apply_illegal_state_message = "TracePack::Registry::apply: illegal state item";
   const string __apply_illegal_no_add_message = __apply_illegal_action_message+" (TD_Added expected)";
#endif


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
   io_ostream << i_prefix << "dataLength : " << m_dataLength << endl;
}

bool TracePack::Registry::Item::copyToDirectory( const ctPath &i_directory ) const
{
   return m_data.copyToFile( cElFilename( i_directory, m_filename ) );
}

bool TracePack::Registry::Item::trace_compare( const TracePack::Registry::Item &i_b ) const
{
   if ( m_filename!=i_b.m_filename )
      cout << "Items have different filenames : " << m_filename.str_unix() << " != " << i_b.m_filename.str_unix() << endl;
   else if ( m_type!=i_b.m_type )
      cout << "Items have different types : " << TD_Type_to_string(m_type) << " != " << TD_Type_to_string(i_b.m_type) << endl;
   else if ( m_date!=i_b.m_date )
      cout << "Items have different dates : " << m_date << " != " << i_b.m_date << endl;
   else if ( m_dataLength!=i_b.m_dataLength )
      cout << "Items have different data lenghtes : " << m_dataLength << " != " << i_b.m_dataLength << endl;
   else if ( !(m_data==i_b.m_data) )
      cout << "Items have different data" << endl;
   else
      return true;
   return false;
}

bool TracePack::Registry::Item::applyToDirectory( const ctPath &i_path ) const
{
   switch ( m_type ){
   case TD_Added:
   case TD_Modified:
      if ( !copyToDirectory( i_path ) ) return false;
      if ( !cElFilename( i_path, m_filename ).setRights( m_rights ) ) return false;
      return true;
      //return m_filename.setRights( m_rights );
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
U_INT8 TracePack::Registry::Item::raw_size() const
{
   U_INT8 res = string_raw_size( m_filename.str_unix() )+4/*type*/;
   if ( hasData() )
      res += cElDate::raw_size()+12; // 12 = 4(rights) + 8(data length)
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
   int4_to_raw_data( (INT4)m_type, i_reverseByteOrder, o_rawData );
   
   if ( hasData() )
   {
      // copy last modification date
      m_date.to_raw_data( i_reverseByteOrder, o_rawData );
      // copy rights on file
      uint4_to_raw_data( (U_INT4)m_rights, i_reverseByteOrder, o_rawData );
      // copy file's size
      uint8_to_raw_data( m_dataLength, i_reverseByteOrder, o_rawData );
   }
   
   #ifdef __DEBUG_TRACE_PACK
      if ( o_rawData<rawData || (U_INT8)(o_rawData-rawData)!=raw_size() )
      {
	 cerr << RED_DEBUG_ERROR << "TracePack::Registry::Item::to_raw_data: " << (U_INT8)(o_rawData-rawData) << " copied bytes, but raw_size() = " << raw_size() << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
}

void TracePack::Registry::Item::from_raw_data( const char *&io_rawData, bool i_reverseByteOrder )
{
   #ifdef __DEBUG_TRACE_PACK
      const char *rawData = io_rawData;
   #endif
      
   // copy filename
   string s;
   string_from_raw_data( io_rawData, i_reverseByteOrder, s );
   m_filename = cElFilename(s);
   
   // copy type
   INT4 i4;
   int4_from_raw_data( io_rawData, i_reverseByteOrder, i4 );
   m_type = (TD_Type)i4;
   
   if ( hasData() ){
      // copy last modification date
      m_date.from_raw_data( io_rawData, i_reverseByteOrder );
      
      // copy rights on file
      U_INT4 ui4;
      uint4_from_raw_data( io_rawData, i_reverseByteOrder, ui4 );
      #if ELISE_windows
			m_rights = cElFilename::unhandledRights;
      #else
			m_rights = (mode_t)ui4;
      #endif
      
      // copy file size
      uint8_from_raw_data( io_rawData, i_reverseByteOrder, m_dataLength );
   }
   else{
      m_date = cElDate::NoDate;
      m_rights = 0;
      m_dataLength = 0;
   }
   
   #ifdef __DEBUG_TRACE_PACK
      unsigned int nbCopied = io_rawData-rawData;
      if ( nbCopied!=raw_size() ){
			cerr << RED_DEBUG_ERROR << "TracePack::Registry::Item::to_raw_data: " << nbCopied << " copied bytes, but raw_size() = " << raw_size() << endl;
			exit(EXIT_FAILURE);
      }
   #endif
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
	    m_items.push_back( Item( itA->m_filename, TD_Modified, itB->m_date, itB->m_rights, itB->m_dataLength, itB->m_data ) );
	 itA++; itB++;
      }
      else
      {
	 if ( compare<0 )
	    m_items.push_back( Item( (*itA++).m_filename, TD_Removed, cElDate::NoDate, 0/*rights*/, 0/*data length*/ ) );
	 else
	 {
	    m_items.push_back( Item( itB->m_filename, TD_Added, itB->m_date, itB->m_rights, itB->m_dataLength, itB->m_data ) );
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
      m_items.push_back( Item( (*itA++).m_filename, TD_Removed, cElDate::NoDate, 0/*rights*/, 0/*data length*/ ) );
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
      m_items.push_back( Item( itB->m_filename, TD_Added, itB->m_date, itB->m_rights, itB->m_dataLength, itB->m_data ) );
      itB++;
   }
}

bool TracePack::Registry::compare_states( const TracePack::Registry &i_a, const TracePack::Registry &i_b,
                                          list<Registry::Item> &o_onlyInA, list<Registry::Item> &o_onlyInB, list<Registry::Item> &o_inBoth )
{  
   o_onlyInA.clear();
   o_onlyInB.clear();
   o_inBoth.clear();
   list<TracePack::Registry::Item>::const_iterator itA = i_a.m_items.begin(),
                                                   itB = i_b.m_items.begin();
   while ( itA!=i_a.m_items.end() && itB!=i_b.m_items.end() ){
      int compare = itA->m_filename.compare( itB->m_filename );
      if ( compare==0 ){
			if ( itA->m_type==itB->m_type )
				o_inBoth.push_back( *itA );
			else{
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

   return ( o_onlyInA.size()==0 && o_onlyInB.size()==0 );
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
	mode_t rights;
	list<string> files = RegexListFileMatch( path.str()+ctPath::unix_separator, ".*", numeric_limits<INT>::max(), false );
	list<string>::iterator itFile = files.begin();
	while ( itFile!=files.end() ){
		cElFilename attachedFile( i_path, *itFile ), // attached filename is the filename in situ
		            detachedFile( *itFile++ ); // detached filename is the filename relative to m_anchor
		const string attached_filename = attachedFile.str_unix();
		ELISE_fp::lastModificationDate( attached_filename, date );
		attachedFile.getRights( rights );
		add( Item( detachedFile, TD_State, date, rights, attachedFile.getSize() ) );
	}
}

void TracePack::Registry::apply( const TracePack::Registry &i_actions )
{
   list<TracePack::Registry::Item>::iterator itA = m_items.begin();
   list<TracePack::Registry::Item>::const_iterator itB = i_actions.m_items.begin();
   while ( itA!=m_items.end() && itB!=i_actions.m_items.end() ){
      #ifdef __DEBUG_TRACE_PACK
			if ( itA->m_type!=TD_State ){
				cerr << RED_DEBUG_ERROR << __apply_illegal_action_message << endl;
				exit(EXIT_FAILURE);
			}
			if ( itB->m_type==TD_State ){
				cerr << RED_DEBUG_ERROR << __apply_illegal_state_message << endl;
				exit(EXIT_FAILURE);
			}
      #endif
      
      int compare = itA->m_filename.compare( itB->m_filename );
      if ( compare==0 ){
			if ( itB->m_type==TD_Removed ){
				itA = m_items.erase( itA );
				itB++;
			}
			else if ( itB->m_type==TD_Modified ){
				itA->m_date       = itB->m_date;
				itA->m_rights     = itB->m_rights;
				itA->m_dataLength = itB->m_dataLength;
				itA->m_data       = itB->m_data;
				itA++; itB++;
			}
			#ifdef __DEBUG_TRACE_PACK
				else{
					cerr << RED_DEBUG_ERROR << "TracePack::Registry::apply: illegal action item of type TD_Added" << endl;
					exit(EXIT_FAILURE);
				}
			#endif
      }
      else{
			if ( compare<0 ) itA++;
			else{
				#ifdef __DEBUG_TRACE_PACK
					if ( itB->m_type!=TD_Added ){
						cerr << RED_DEBUG_ERROR << __apply_illegal_no_add_message << endl;
						exit(EXIT_FAILURE);
					}
					else
				#endif
				{
					m_items.insert( itA, Item( itB->m_filename, TD_State, itB->m_date, itB->m_rights, itB->m_dataLength, itB->m_data ) );
					itB++;
				}
			}
      }
   }
   while ( itB!=i_actions.m_items.end() ){
		#ifdef __DEBUG_TRACE_PACK
		if ( itB->m_type!=TD_Added ){
			cerr << RED_DEBUG_ERROR << __apply_illegal_no_add_message << endl;
			exit(EXIT_FAILURE);
		}
		else
		#endif
		{
			m_items.push_back( Item( itB->m_filename, TD_State, itB->m_date, itB->m_rights, itB->m_dataLength, itB->m_data ) );
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
	 
bool TracePack::Registry::applyToDirectory( const ctPath &i_path ) const
{
   list<TracePack::Registry::Item>::const_iterator itItem = m_items.begin();
   while ( itItem!=m_items.end() ){
      if ( !( itItem->applyToDirectory( i_path ) ) ){
			#ifdef __DEBUG_TRACE_PACK
				cerr << RED_DEBUG_ERROR << "TracePack::Registry::applyToDirectory: cannot to apply item [" << itItem->m_filename.str_unix() << "] of type " 
				     << TD_Type_to_string(itItem->m_type) << " to directory [" << i_path.str() << "]" << endl;
				exit(EXIT_FAILURE);
			#endif
			return false;
      }
      itItem++;
   }
   return true;
}

U_INT8 TracePack::Registry::raw_size() const
{
   U_INT8 res = m_command.raw_size()+4; // 4 = size of nbRegistries
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
   uint4_to_raw_data( (U_INT4)m_items.size(), i_reverseByteOrder, o_rawData );
      
   list<Item>::const_iterator itItem = m_items.begin();
   while ( itItem!=m_items.end() )
      ( *itItem++ ).to_raw_data( i_reverseByteOrder, o_rawData );
   
   #ifdef __DEBUG_TRACE_PACK
      if ( rawData>o_rawData || (U_INT8)(o_rawData-rawData)!=raw_size() )
      {
	 cerr << RED_DEBUG_ERROR << "TracePack::Registry::to_raw_data: " << (U_INT8)(o_rawData-rawData) << " copied bytes, but raw_size() = " << raw_size() << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
}

void TracePack::Registry::from_raw_data( const char *i_rawData, bool i_reverseByteOrder )
{
   #ifdef __DEBUG_TRACE_PACK
      char const *rawData = i_rawData;
   #endif
      
   // copy commande
   m_command.from_raw_data( i_rawData, i_reverseByteOrder );
   
   // copy number of items
   U_INT4 nbItems;
   uint4_from_raw_data( i_rawData, i_reverseByteOrder, nbItems );
   
   // copy items
   m_items.resize( nbItems );
   list<Item>::iterator itItem = m_items.begin();
   while ( itItem!=m_items.end() )
      ( *itItem++ ).from_raw_data( i_rawData, i_reverseByteOrder );
   
   #ifdef __DEBUG_TRACE_PACK
      unsigned int nbCopied = i_rawData-rawData;
      if ( nbCopied!=raw_size() )
      {
	 cerr << RED_DEBUG_ERROR << "TracePack::Registry::from_raw_data_v1: " << nbCopied << " copied bytes, but raw_size() = " << raw_size() << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
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

bool TracePack::Registry::setItemData( const ChunkStream::FileItem &i_data )
{
   list<Item>::iterator itItem = m_items.begin();
   while ( itItem!=m_items.end() && itItem->m_filename.compare( i_data.m_storedFilename )<0 ) itItem++;
   if ( itItem==m_items.end() ) return false;
   if ( itItem->m_filename==i_data.m_storedFilename && itItem->hasData() && itItem->m_dataLength==i_data.m_length ){
      itItem->m_data = i_data;
      return true;
   }
   return false;
}


//--------------------------------------------
// class TracePack
//--------------------------------------------

TracePack::TracePack( const cElFilename &i_filename, const ctPath &i_anchor, U_INT8 i_maxFileSize ):
   m_filename( i_filename ),
   m_anchor( i_anchor ),
   m_date( cElDate::NoDate ),
   m_versionNumber( g_versioned_headers_list[VFH_TracePack].last_handled_version ),
   m_stream( i_filename, i_maxFileSize, false ),
   m_ignoredItemsChanged( false )
{
   cElDate::getCurrentDate_UTC( m_date );
}

bool TracePack::read_v1( U_INT8 i_offset, bool i_reverseByteOrder )
{
   list<ChunkStream::Item*> items;
   if ( !m_stream.read( 0, i_offset, items ) ){
	   cerr << RED_DEBUG_ERROR << "TracePack::read_v1: cannot read list of Item from pack [" << m_filename.str_unix() << ']' << endl;
	   exit(EXIT_FAILURE);
   }

   // convert raw data from Items to TracePack objects
   list<ChunkStream::Item*>::const_iterator itItem = items.begin();
   const ChunkStream::BufferItem *lastIgnoreItem = NULL;
   Registry *lastRegistry = NULL;
   while ( itItem!=items.end() ){
		if ( (*itItem)->isFileItem() ){
			// process ChunkStream::FileItem by adding data to an item of the last found registry
			ChunkStream::FileItem &item = (*itItem)->specialize<ChunkStream::FileItem>();
			if ( lastRegistry==NULL ){
				#ifdef __DEBUG_TRACE_PACK
					cerr << RED_DEBUG_ERROR << "TracePack::read_v1: a file item is found priory to any registry item" << endl;
					exit(EXIT_FAILURE);
				#endif
				clear_item_list( items );
				return false;
			}
			if ( !lastRegistry->setItemData( item ) ){
				#ifdef __DEBUG_TRACE_PACK
					cerr << RED_DEBUG_ERROR << "TracePack::read_v1: a file item is found that is not in lastRegistry" << endl;
					exit(EXIT_FAILURE);
				#endif
				clear_item_list( items );
				return false;
			}
		}
		else{
			// process ChunkStream::BufferItems
			ChunkStream::BufferItem &item = (*itItem)->specialize<ChunkStream::BufferItem>();
			switch ( (StreamItemType)item.m_type ){
			case SIT_Registry:
				// add a new registry
				m_registries.push_back( Registry() );
				lastRegistry = &m_registries.back();
				lastRegistry->from_raw_data( item.m_buffer.data(), i_reverseByteOrder );
				lastRegistry->m_hasBeenWritten = true;
				break;
			case SIT_Ignore:
				// only last IgnoredItem is used (replacing all previous ones)
				lastIgnoreItem = &item;
				break;
			}
		}
		itItem++;
   }
   
   // load last IgnoredItem if there's one
   if ( lastIgnoreItem!=NULL ) ignored_lists_from_buffer( i_reverseByteOrder, lastIgnoreItem->m_buffer );
   
   clear_item_list( items );
   return true;
}

bool TracePack::load()
{
   ifstream f( m_filename.str_unix().c_str(), ios::binary );
   if ( !f ){
		#ifdef __DEBUG_TRACE_PACK
			cerr << RED_DEBUG_ERROR << "TracePack::load: unable to open [" << m_filename.str_unix() << "] for reading" << endl;
			exit(EXIT_FAILURE);
		#endif
		return false;
   }

	VersionedFileHeader header( VFH_TracePack );
	if ( !header.read_known( VFH_TracePack, f ) ){
		#ifdef __DEBUG_TRACE_PACK
			cerr << RED_DEBUG_ERROR << "TracePack::load: cannot read versioned file header" << endl;
			exit(EXIT_FAILURE);
		#endif
	}
	bool reverseByteOrder = header.isMSBF()!=MSBF_PROCESSOR();
	m_stream.setReverseByteOrder( reverseByteOrder );
	m_versionNumber = header.version();
	// read date
	m_date.read_raw( f, reverseByteOrder );
	U_INT8 chunkStreamOffset = (U_INT8)f.tellg();
	f.close();
   
	bool res = false;
   switch ( header.version() ){
   case 1: res=read_v1( chunkStreamOffset, reverseByteOrder ); break;
   }

   // remove ignored items
   list<Registry>::iterator itReg = m_registries.begin();
   while ( itReg!=m_registries.end() ) remove_ignored_items( *itReg++ );
   
   #ifdef __DEBUG_TRACE_PACK
		if ( !res ){
			cerr << RED_DEBUG_ERROR << "TracePack::load: unable to read versioned file of type VFH_TracePack (v" << header.version() << ' ' 
			     << (header.isMSBF()?"big-endian":"little-endian") << ") from file [" << m_filename.str_unix() << ']' << endl;
			exit(EXIT_FAILURE);
		}
   #endif
   
   return res;
}

bool TracePack::write_v1( bool i_reverseByteOrder )
{
   if ( !m_stream.writeOpen() ) return false;
   vector<char> buffer;
   
   list<Registry>::iterator itRegistry = m_registries.begin();
   while ( itRegistry!=m_registries.end() ){
      Registry &registry = *itRegistry++;
      if ( !registry.m_hasBeenWritten ){
			// write the registry itself	 
			buffer.resize( registry.raw_size() );
			char *bufferData = buffer.data();
			registry.to_raw_data( i_reverseByteOrder, bufferData );
			m_stream.write_buffer( (unsigned char)SIT_Registry, buffer );

			// write data of registry's items
			list<Registry::Item>::iterator itItem = registry.m_items.begin();
			while ( itItem!=registry.m_items.end() ){
				Registry::Item &item = *itItem++;
				if ( item.hasData() ){
					if ( !m_stream.write_file( cElFilename( m_anchor, item.m_filename ), item.m_filename, item.m_data ) ){
						#ifdef __DEBUG_TRACE_PACK
							cerr << RED_DEBUG_ERROR << "TracePack::write_v1: cannot write data of file [" << cElFilename( m_anchor, item.m_filename ).str_unix() << endl;
							exit(EXIT_FAILURE);
						#endif
						return false;
					}
				}
			}
			registry.m_hasBeenWritten = true;
      }
   }
   
   // write ignored items lists
   if ( m_ignoredItemsChanged ){
		ignored_lists_to_buffer( i_reverseByteOrder, buffer );
		m_stream.write_buffer( (unsigned char)SIT_Ignore, buffer );
   }
   
   m_stream.close();

   return true;
}

bool TracePack::save()
{
   bool reverseByteOrder = m_stream.getReverseByteOrder();
   if ( m_stream.getNbFiles()==0 ){
		// the stream is empty, write header
		ofstream f( m_stream.getFilename(0).str_unix().c_str(), ios::binary );
		if ( !f ) return false;
		VersionedFileHeader header( VFH_TracePack, m_versionNumber, reverseByteOrder?!MSBF_PROCESSOR():MSBF_PROCESSOR() );
		header.write( f );
		// write pack date
		m_date.write_raw( f, reverseByteOrder );
   }

   bool res = false;
   switch ( m_versionNumber )
   {
   case 1:
      res = write_v1(reverseByteOrder);
      break;
   default: res=false;
   }
   
   m_ignoredItemsChanged = false;
   
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
	if ( !m_anchor.exists() ){
		cerr << RED_DEBUG_ERROR << "TracePack::addState: directory [" << m_anchor.str() << "] does not exist" << endl;
		exit(EXIT_FAILURE);
	}
	#endif
   
	if ( nbStates()==0 ){
		Registry reg0;
		reg0.stateDirectory( m_anchor );
		remove_ignored_items( reg0 );
		m_registries.push_back( reg0 );
		return;
	}

	Registry directoryState, lastState, diff;
	directoryState.stateDirectory( m_anchor );

	// __DEL
	directoryState.dump();
	
	remove_ignored_items( directoryState );

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
   return pItem->copyToDirectory( m_anchor );
}

bool TracePack::trace_compare( const TracePack &i_b ) const
{	 
   if ( m_filename!=i_b.m_filename )
      cout << "packs have different filenames : " << m_filename.str_unix() << " != " << i_b.m_filename.str_unix() << endl;
   else if ( m_date!=i_b.m_date )
      cout << "packs have different dates : " << m_date << " != " << i_b.m_date << endl;
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
      if ( !ELISE_fp::IsDirectory( m_anchor.str() ) ){
			cerr << RED_DEBUG_ERROR << "TracePack::setState: destination directory [" << m_anchor.str() << "] does not exist" << endl;
			exit(EXIT_FAILURE);
      }
   #endif
   
   Registry refReg, dirReg, dir_to_ref;
   getState( i_iState, refReg );
   dirReg.stateDirectory( m_anchor );
   remove_ignored_items( dirReg );
   dir_to_ref.difference( dirReg, refReg );
   dir_to_ref.applyToDirectory( m_anchor );
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
	if ( i_directory.isAbsolute() || i_directory.count_upward_references()!=0 ){
		cerr << RED_DEBUG_ERROR << "TracePack::add_ignored(ctPath): trying to add [" << i_directory.str() << "] which is invalid (absolute or outside anchor)" << endl;
		exit(EXIT_FAILURE);
	}
	#endif
   
   list<ctPath>::iterator itDirectory = m_ignoredDirectories.begin();
   while ( itDirectory!=m_ignoredDirectories.end() && (*itDirectory)<i_directory ) itDirectory++;
   
	#ifdef __DEBUG_TRACE_PACK
	if ( (*itDirectory)==i_directory ){
		cerr << RED_DEBUG_ERROR << "TracePack::add_ignored(ctPath): trying to add [" << i_directory.str() << "] which is already in m_ignoredDirectories" << endl;
		exit(EXIT_FAILURE);
	}
	#endif
   
   m_ignoredDirectories.insert( itDirectory, i_directory );
}

void TracePack::add_ignored( const cElFilename &i_filename )
{
   #ifdef __DEBUG_TRACE_PACK
      if ( i_filename.m_path.isAbsolute() || i_filename.m_path.count_upward_references()!=0 ){
			cerr << RED_DEBUG_ERROR << "TracePack::add_ignored(cElFilename): trying to add [" << i_filename.str_unix() << "] which is invalid (file's path is absolute or outside anchor)" << endl;
			exit(EXIT_FAILURE);
      }
   #endif

   list<cElFilename>::iterator itFilename = m_ignoredFiles.begin();
   while ( itFilename!=m_ignoredFiles.end() && (*itFilename)<i_filename ) itFilename++;
   
   #ifdef __DEBUG_TRACE_PACK
      if ( (*itFilename)==i_filename ){
			cerr << RED_DEBUG_ERROR << "TracePack::add_ignored(cElFilename): trying to add [" << i_filename.str_unix() << "] which is already in m_ignoredFiles" << endl;
			exit(EXIT_FAILURE);
      }
   #endif
   
   m_ignoredFiles.insert( itFilename, i_filename );
}
   
// add directories to ignored directories list if they enlarge the ignored items set
// io_ignoredDirectories is left with directories actually added to the list
bool TracePack::addIgnored( list<ctPath> &io_directoriesToIgnore )
{   
   // remove already ignored directories from io_ignoredDirectories
   list<ctPath>::iterator itDirectoryToIgnore = io_directoriesToIgnore.begin();
   while ( itDirectoryToIgnore!=io_directoriesToIgnore.end() ){
      if ( isIgnored( *itDirectoryToIgnore ) )
			itDirectoryToIgnore = io_directoriesToIgnore.erase( itDirectoryToIgnore );
      else
			itDirectoryToIgnore++;
   }
   
   // remove ignored directories contained by elements of io_directoriesToIgnore
   list<ctPath>::iterator itIgnoredDirectory = m_ignoredDirectories.begin();
   while ( itIgnoredDirectory!=m_ignoredDirectories.end() ){
      if ( oneIsAncestorOf( io_directoriesToIgnore, *itIgnoredDirectory ) )
			itIgnoredDirectory = m_ignoredDirectories.erase( itIgnoredDirectory );
      else
			itIgnoredDirectory++;
   }
   
   // add all remaining elements of io_directoriesToIgnore to m_ignoredDirectories
   itDirectoryToIgnore = io_directoriesToIgnore.begin();
   while ( itDirectoryToIgnore!=io_directoriesToIgnore.end() )
      add_ignored( *itDirectoryToIgnore );
      
   return ( m_ignoredItemsChanged=( io_directoriesToIgnore.size()!=0 ) );
}

// add files to ignored files list if they enlarge the ignored items set
// io_ignoredFiles is left with files actually added to the list
bool TracePack::addIgnored( list<cElFilename> &io_filesToIgnore )
{
   // remove already ignored files from io_filesToIgnore
   list<cElFilename>::iterator itFileToIgnore = io_filesToIgnore.begin();
   while ( itFileToIgnore!=io_filesToIgnore.end() ){
      if ( isIgnored( *itFileToIgnore ) )
			itFileToIgnore = io_filesToIgnore.erase( itFileToIgnore );
      else
			itFileToIgnore++;
   }
   
   // add remaining files to ignored files list
   itFileToIgnore = io_filesToIgnore.begin();
   while ( itFileToIgnore!=io_filesToIgnore.end() )
      add_ignored( *itFileToIgnore++ );
   
   return ( m_ignoredItemsChanged=( io_filesToIgnore.size()!=0 ) );
}

// a directory is ignore if it is in the m_ignoredDirectories list or if it is contained by an ignored directory
bool TracePack::isIgnored( const ctPath &i_directory ) const
{
   return oneIsAncestorOf( m_ignoredDirectories, i_directory );
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
   while ( itDirectory!=m_ignoredDirectories.end() && (*itDirectory)<i_file.m_path ){
      if ( itDirectory->isAncestorOf( i_file ) ) return true;
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
           itItem->m_type==TracePack::Registry::TD_Modified ){
			copyItemOnDisk( i_iRegistry, itItem->m_filename );
			cElFilename fileFromPack( m_anchor, itItem->m_filename ), fileOnDisk( i_onDiskPath, itItem->m_filename );
			if ( !is_equivalent( fileFromPack, fileOnDisk ) ) o_differentFromDisk.push_back( *itItem );
      }
      itItem++;
   }
}

std::string TracePack::StreamItemType_to_string( StreamItemType i_type )
{
   switch ( i_type )
   {
   case TracePack::SIT_Registry: return string("SIT_Registry");
   case TracePack::SIT_Ignore: return string("SIT_Ignore");
   default: return string("unknown");
   }
}

void TracePack::ignored_lists_to_buffer( bool i_reverseByteOrder, vector<char> &o_buffer ) const
{
   // compute total size for buffer allocation
   unsigned int totalSize = 8; // size(nbPath) + size(nbFiles)
   list<ctPath>::const_iterator itPath = m_ignoredDirectories.begin();
   while ( itPath!=m_ignoredDirectories.end() ) totalSize+=string_raw_size( (*itPath++).str() );
   list<cElFilename>::const_iterator itFilename = m_ignoredFiles.begin();
   while ( itFilename!=m_ignoredFiles.end() ) totalSize+=string_raw_size( (*itFilename++).str_unix() );
   
   // fill the buffer
   o_buffer.resize( totalSize );
   char *itData = o_buffer.data();
      // copy path list's size and values
   uint4_to_raw_data( (U_INT4)m_ignoredDirectories.size(), i_reverseByteOrder, itData );
   itPath = m_ignoredDirectories.begin();
   while ( itPath!=m_ignoredDirectories.end() ) string_to_raw_data( (*itPath++).str(), i_reverseByteOrder, itData );
      // copy file list's size and values
   uint4_to_raw_data( (U_INT4)m_ignoredFiles.size(), i_reverseByteOrder, itData );
   itFilename = m_ignoredFiles.begin();
   while ( itFilename!=m_ignoredFiles.end() ) string_to_raw_data( (*itFilename++).str_unix(), i_reverseByteOrder, itData );
   
   #ifdef __DEBUG_TRACE_PACK
      if ( o_buffer.data()>itData || (unsigned int)(itData-o_buffer.data())!=totalSize ){
	 cerr << RED_DEBUG_ERROR << "TracePack::ignored_lists_to_buffer: " << itData-o_buffer.data() << " bytes have been used but buffer size is " << o_buffer.size() << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
}

void TracePack::ignored_lists_from_buffer( bool i_reverseByteOrder, const vector<char> &i_buffer )
{
   const char *itData = i_buffer.data();
   string str;
   m_ignoredDirectories.clear();
   m_ignoredFiles.clear();
   
   // extract directories from buffer and add them to ignored list
   U_INT4 nbPaths;
   uint4_from_raw_data( itData, i_reverseByteOrder, nbPaths );
   while ( nbPaths-- ){
      string_from_raw_data( itData, i_reverseByteOrder, str );
      add_ignored( ctPath( str ) );
   }
   
   // extract files from buffer and add them to ignored list
   U_INT4 nbFiles;
   uint4_from_raw_data( itData, i_reverseByteOrder, nbFiles );
   while ( nbFiles-- ){
      string_from_raw_data( itData, i_reverseByteOrder, str );
      add_ignored( cElFilename( str ) );
   }
   
   #ifdef __DEBUG_TRACE_PACK
      if ( i_buffer.data()>itData || (size_t)(itData-i_buffer.data())!=i_buffer.size() ){
			cerr << RED_DEBUG_ERROR << "TracePack::ignored_lists_from_buffer: " << itData-i_buffer.data() << " bytes have been used but buffer size is " << i_buffer.size() << endl;
			exit(EXIT_FAILURE);
      }
   #endif
}

bool TracePack::setSavingFormat( unsigned int i_version, bool i_isMSBF )
{
	if ( m_stream.getNbFiles()==0 ){
		m_versionNumber = i_version;
		m_stream.setReverseByteOrder( i_isMSBF!=MSBF_PROCESSOR() );
		return true;
	}
	else return false;
}

void TracePack::remove_ignored_items( Registry &io_registry )
{
	list<Registry::Item>::iterator itItem = io_registry.m_items.begin();
	while ( itItem!=io_registry.m_items.end() ){
		if ( isIgnored( itItem->m_filename ) )
			itItem = io_registry.m_items.erase(itItem);
		else
			itItem++;
	}
}


//--------------------------------------------
// related functions
//--------------------------------------------

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

bool file_raw_equals( const cElFilename &i_a, const cElFilename &i_b )
{
	U_INT8 size = i_a.getSize();
	if ( i_b.getSize()!=size ) return false;
	ifstream fa( i_a.str_unix().c_str(), ios::binary ),
	         fb( i_b.str_unix().c_str(), ios::binary );
	if ( !fa || !fb ) return false;
	const U_INT8 buffer_size = 1e6;
	vector<char> bufferA(buffer_size),
	             bufferB(buffer_size);
	char *dataA = bufferA.data(),
	     *dataB = bufferB.data();
	while ( size!=0 ){
		U_INT8 blockSize = std::min( size, buffer_size );
		fa.read( dataA, blockSize );
		fb.read( dataB, blockSize );
		if ( memcmp( dataA, dataB, blockSize ) ) return false;
		size -= blockSize;
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
	#elif _MSC_VER
		return file_raw_equals( i_a, i_b );
	#else
	// __TODO
	not implemented
	#endif
}

// returns if an element of i_pathList is an ancestor of i_path
bool oneIsAncestorOf( const std::list<ctPath> &i_pathList, const ctPath &i_path )
{
   // directories that may contain i_directory are lesser than it
   list<ctPath>::const_iterator itPath = i_pathList.begin();
   while ( itPath!=i_pathList.end() && ( (*itPath)<i_path ) )
   {
      if ( itPath->isAncestorOf( i_path ) ) return true;
      itPath++;
   }
   return ( itPath!=i_pathList.end() && ( (*itPath)==i_path ) );
}

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
   
   return stream_copy( fIn, io_fOut, i_expectedSize );
}
