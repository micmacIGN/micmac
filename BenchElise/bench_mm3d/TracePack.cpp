#include "TracePack.h"

#include "StdAfx.h"

#include "general/cElCommand.h"

using namespace std;

//#define __DEBUG_TRACE_PACK_PRINT_READ
//#define __DEBUG_TRACE_PACK_PRINT_WRITE

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
   io_ostream << i_prefix << "date       : " << m_date << endl;
   io_ostream << i_prefix << "dataOffset : " << m_dataOffset << endl;
   io_ostream << i_prefix << "dataSize   : " << m_dataSize << endl;
}

bool TracePack::Registry::Item::copyToDirectory( const cElFilename &i_packName, const cElPath &i_directory ) const
{
   const cElFilename filename( i_directory, m_filename );
   
   if ( !filename.m_path.exists() && !filename.m_path.create() )
   {
      #ifdef __DEBUG_TRACE_PACK
	 cerr << "DEBUG_ERROR: TracePack::Registry::Item::copyToDirectory: cannot create directory [" << filename.m_path.str_unix() << ']' << endl;
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
	    cerr << "DEBUG_ERROR: TracePack::Registry::Item::copyToDirectory: unable to open [" << i_packName.str_unix() << "] for reading at position " << m_dataOffset;
	 if ( !fOut )
	    cerr << "DEBUG_ERROR: TracePack::Registry::Item::copyToDirectory: unable to open [" << filename.str_unix() << "] for writing";
      #endif
      return false;
   }
   
   return write_file( fIn, fOut, m_dataSize );
}

bool TracePack::Registry::Item::compare( const TracePack::Registry::Item &i_b ) const
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

bool TracePack::Registry::Item::applyToDirectory( const cElFilename &i_packname, const cElPath &i_path ) const
{
   switch ( m_type )
   {
   case TD_Added:
   case TD_Modified:
      return copyToDirectory( i_packname, i_path );
      break;
   case TD_Removed:
      return cElFilename( i_path, m_filename ).remove();
      break;
   case TD_State:
      #ifdef __DEBUG_TRACE_PACK
	 cerr << "DEBUG_ERROR: TracePack::Registry::Item::applyToDirectory: cannot apply state item [" << m_filename.str_unix() << ']';
      #endif
      break;
   }
   return false;
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
	    cerr << "DEBUG_ERROR: " << __difference_illegal_item_message << endl;
      #endif
      
      int compare = itA->m_filename.compare( itB->m_filename );
      if ( compare==0 )
      {
	 if ( itA->m_date!=itB->m_date )
	 {
	    m_items.push_back( Item( itA->m_filename, TD_Modified, itB->m_date, itB->m_dataOffset, itB->m_dataSize ) );
	 }
	 itA++; itB++;
      }
      else
      {
	 if ( compare<0 )
	    m_items.push_back( Item( (*itA++).m_filename, TD_Removed, cElDate::NoDate, 0, 0 ) );
	 else
	    m_items.push_back( Item( (*itB++).m_filename, TD_Added, itB->m_date, itB->m_dataOffset, itB->m_dataSize ) );
      }
   }
   while ( itA!=i_a.m_items.end() )
   {
      #ifdef __DEBUG_TRACE_PACK
	 if ( itA->m_type!=TD_State )
	    cerr << "DEBUG_ERROR: " << __difference_illegal_item_message << endl;
      #endif
      m_items.push_back( Item( (*itA++).m_filename, TD_Removed, cElDate::NoDate, 0, 0 ) );
   }
   while ( itB!=i_b.m_items.end() )
   {
      #ifdef __DEBUG_TRACE_PACK
	 if ( itB->m_type!=TD_State )
	    cerr << "DEBUG_ERROR: " << __difference_illegal_item_message << endl;
      #endif
      m_items.push_back( Item( (*itB++).m_filename, TD_Added, itB->m_date, itB->m_dataOffset, itB->m_dataSize ) );
   }
}

bool TracePack::Registry::write_v1( ofstream &io_stream, const cElPath &i_anchor )
{   
   U_INT4 ui = (U_INT4)m_items.size(),
          dataSize;
   INT4 i;
   string s;
   
   #ifdef __DEBUG_TRACE_PACK_PRINT_WRITE
      cout << "-write--------------------- nbItems : " << ui << endl;
   #endif
   
   io_stream.write( (char*)(&ui), 4 );
   list<Item>::iterator itItem = m_items.begin();
   while ( itItem!=m_items.end() )
   {
      // write filename
      s = itItem->m_filename.str_unix();
      ui = s.size();
      io_stream.write( (char*)(&ui), 4 );
      io_stream.write( s.c_str(), ui );
      // write type
      i = (INT4)itItem->m_type;
      io_stream.write( (char*)(&i), 4 );
      if ( itItem->m_type==TracePack::Registry::TD_State ||
           itItem->m_type==TracePack::Registry::TD_Added ||
           itItem->m_type==TracePack::Registry::TD_Modified )
      {
	 itItem->m_date.write_raw( io_stream );
	 // writing file's data
	 dataSize = (U_INT4)itItem->m_dataSize;
	 io_stream.write( (char*)(&dataSize), 4 );
	 itItem->m_dataOffset = io_stream.tellp();
	 if ( !write_file( cElFilename( i_anchor, itItem->m_filename ), io_stream, dataSize ) )
	 {
	    #ifdef __DEBUG_TRACE_PACK
	       cerr << "DEBUG_ERROR: TracePack::Registry::write_v1: unable to copy data of ile [" << itItem->m_filename.str_unix() << ']' << endl;
	    #endif
	    return false;
	 }
      }
      itItem++;
   }
   m_hasBeenWritten = true;
   return true;
}

void TracePack::Registry::read_v1( istream &io_stream )
{
   m_items.clear();
   U_INT4 nbFiles, ui,
          dataSize;
   streampos offset;
   INT4 i;
   vector<char> str;
   cElFilename itemFilename, dataFilename;
   cElDate d( cElDate::NoDate );
   io_stream.read( (char*)&nbFiles, 4 );
   
   #ifdef __DEBUG_TRACE_PACK_PRINT_READ
      cout << "-read---------------------- nbFiles : " << nbFiles << endl;
   #endif
   
   while ( nbFiles-- )
   {
      // read filename
      io_stream.read( (char*)(&ui), 4 );
      str.resize(ui);
      io_stream.read( str.data(), ui );
      itemFilename = cElFilename( string( str.data(), ui ) );
      // write type
      io_stream.read( (char*)(&i), 4 );
      // add item
      if ( i==TracePack::Registry::TD_State ||
           i==TracePack::Registry::TD_Added ||
           i==TracePack::Registry::TD_Modified )
      {
	 d.read_raw( io_stream );
	 io_stream.read( (char*)(&dataSize), 4 );
	 offset = io_stream.tellg();
	 io_stream.seekg( dataSize, ios::cur );
      }
      else
      {
	 // TD_Remove
	 offset = dataSize = 0;
	 d = cElDate::NoDate;
      }
      
      #ifdef __DEBUG_TRACE_PACK_PRINT_READ
	 cout << "-read------------------------- " << itemFilename.str_unix() << ' ' << TD_Type_to_string((TD_Type)i) << ' ' << d << ' ' << offset << ' ' << dataSize << endl;
      #endif
      
      m_items.push_back( Item( itemFilename, (TD_Type)i, d, offset, dataSize ) );
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
      return *m_items.rbegin();
   }
   if ( it->m_filename==filename ) return *it;
   return *m_items.insert( it, i_item );
}

void TracePack::Registry::stateDirectory( const cElPath &i_path )
{
   cElPath path( i_path );
   path.toAbsolute();
   
   #ifdef __DEBUG_TRACE_PACK
      if ( !path.exists() )
	 cerr << "DEBUG_ERROR: TracePack::Registry::stateDirectory: directory ["+i_path.str_unix()+"] does not exist" << endl;
   #endif
   
   m_items.clear();
   cElDate date = cElDate::NoDate;
   int fileLength;
   list<string> files = RegexListFileMatch( path.str_unix()+cElPath::sm_unix_separator, ".*", numeric_limits<INT>::max(), false );
   list<string>::iterator itFile = files.begin();
   while ( itFile!=files.end() )
   {
      cElFilename attachedFile( i_path, *itFile ),
                  detachedFile( *itFile++ );
      const string attached_filename = attachedFile.str_unix();
      ELISE_fp::lastModificationDate( attached_filename, date );
      fileLength = ELISE_fp::file_length( attached_filename );
      #ifdef __DEBUG_TRACE_PACK
	 if ( fileLength<0 )
	    cerr << "DEBUG_ERROR: TracePack::Registry::stateDirectory: cannot read length of file ["+attached_filename+"]" << endl;
      #endif
      add( Item( detachedFile, TD_State, date, 0, (unsigned int)fileLength ) );
   }
}

void TracePack::Registry::apply( const TracePack::Registry &i_actions )
{
   list<TracePack::Registry::Item>::iterator itA = m_items.begin();
   list<TracePack::Registry::Item>::const_iterator itB = i_actions.m_items.begin();
   while ( itA!=m_items.end() && itB!=i_actions.m_items.end() )
   {      
      #ifdef __DEBUG_TRACE_PACK
	 if ( itA->m_type!=TD_State ) cerr << "DEBUG_ERROR: " << __apply_illegal_action_message << endl;
	 if ( itB->m_type==TD_State ) cerr << "DEBUG_ERROR: " << __apply_illegal_state_message << endl;
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
	    itA->m_dataOffset = itB->m_dataOffset;
	    itA->m_dataSize   = itB->m_dataSize;
	    itA++; itB++;
	 }
	 #ifdef __DEBUG_TRACE_PACK
	    else
	       cerr << "DEBUG_ERROR: TracePack::Registry::apply: illegal action item of type TD_Added" << endl;
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
		  cerr << "DEBUG_ERROR: " << __apply_illegal_no_add_message << endl;
	       else
	    #endif
	       {
		  m_items.insert( itA, Item( itB->m_filename, TD_State, itB->m_date, itB->m_dataOffset, itB->m_dataSize ) );
		  itB++;
	       }
	 }
      }
   }
   while ( itB!=i_actions.m_items.end() )
   {
      #ifdef __DEBUG_TRACE_PACK
	 if ( itB->m_type!=TD_Added )
	    cerr << "DEBUG_ERROR: " << __apply_illegal_no_add_message << endl;
	 else
      #endif
	 {
	    m_items.push_back( Item( itB->m_filename, TD_State, itB->m_date, itB->m_dataOffset, itB->m_dataSize ) );
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

bool TracePack::Registry::compare( const TracePack::Registry &i_b ) const
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
      if ( !itA->compare( *itB ) )
      {
	 cout << "registries items of index " << iItem << " are different" << endl;
	 return false;
      }
      itA++; itB++;
   }
   return true;
}
	 
bool TracePack::Registry::applyToDirectory( const cElFilename &i_filename, const cElPath &i_path ) const
{
   list<TracePack::Registry::Item>::const_iterator itItem = m_items.begin();
   while ( itItem!=m_items.end() )
      if ( !( *itItem++ ).applyToDirectory( i_filename, i_path ) ) return false;
   return true;
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
	    cerr << "DEBUG_ERROR: TracePack::Registry::check_sorted: items are not corretly sorted or an item appears more than once" << endl;
      }
   }
#endif


//--------------------------------------------
// class TracePack
//--------------------------------------------

TracePack::TracePack( const cElFilename &i_filename, const cElPath &i_anchor ):
   m_filename( i_filename ),
   m_anchor( i_anchor ),
   m_date( cElDate::NoDate ),
   m_writeInUpdateMode( false ),
   m_nbRegistriesOffset( 0 )
{
   cElDate::getCurrentDate_UTC( m_date );
}

void TracePack::read_v1( istream &f )
{
   U_INT4 nbRegistries;
   m_date.read_raw( f );
   m_nbRegistriesOffset = f.tellg();   
   f.read( (char*)&nbRegistries, 4 );
   
   #ifdef __DEBUG_TRACE_PACK_PRINT_READ
      cout << "-read------------------- nbRegistries : " << nbRegistries << endl;
      cout << "-read------------------- m_nbRegistriesOffset : " << m_nbRegistriesOffset << endl;
   #endif
   
   m_registries.resize( nbRegistries );
   list<Registry>::iterator itReg = m_registries.begin();
   while ( nbRegistries-- )
      ( *itReg++ ).read_v1( f );
}

bool TracePack::load()
{
   ifstream f( m_filename.str_unix().c_str(), ios::binary );
   if ( !f )
   {
      #ifdef __DEBUG_TRACE_PACK
	 cerr << "DEBUG_ERROR: TracePack::load: unable to open [" << m_filename.str_unix() << "] for reading" << endl;
      #endif
      return false;
   }
   VersionedFileHeader header( VFH_TracePack );
      
   bool res = header.read_known( VFH_TracePack, f );
   
   #ifdef __DEBUG_TRACE_PACK
      if ( !res )
	 cerr << "DEBUG_ERROR: TracePack::load: unable to read versioned file header of type VFH_TracePack from file [" << m_filename.str_unix() << ']' << endl;
      #ifdef __DEBUG_TRACE_PACK_PRINT_READ
	 else
	    cout << "-read------------------- header : VFH_TracePack, " << (header.isMSBF()?"big-endian":"little-endian") << ", v" << header.version() << endl;
      #endif
   #endif
   
   switch ( header.version() )
   {
   case 1: read_v1( f ); break;
   default: res=false;
   }
   
   #ifdef __DEBUG_TRACE_PACK
      if ( !res )
	 cerr << "DEBUG_ERROR: TracePack::load: unable to read versioned file of type VFH_TracePack (v" << header.version() << ' ' << (header.isMSBF()?"big-endian":"little-endian") << ") from file [" << m_filename.str_unix() << ']' << endl;
   #endif
   
   m_writeInUpdateMode = res;
   return res;
}

bool TracePack::write_v1( ofstream &f )
{
   if ( !m_writeInUpdateMode )
   {
      // this is a new file
      m_date.write_raw( f );
      U_INT4 nbRegistries = (U_INT4)m_registries.size();
      m_nbRegistriesOffset = f.tellp();      
      f.write( (char*)&nbRegistries, 4 );
   }

   list<Registry>::iterator itReg = m_registries.begin();
   while ( itReg!=m_registries.end() )
   {
      if ( !itReg->m_hasBeenWritten && !itReg->write_v1( f, m_anchor ) )
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

void TracePack::save_nbRegistries() const
{
   #ifdef __DEBUG_TRACE_PACK
      if ( !m_writeInUpdateMode || m_nbRegistriesOffset==0 ) cerr << "DEBUG_ERROR: TracePack::update_registry_number: TracePack is not in update mode" << endl;
   #endif
   ofstream f( m_filename.str_unix().c_str(), ios::in|ios::out|ios::binary );
   f.seekp( m_nbRegistriesOffset );
   U_INT4 nbRegistries = (U_INT4)m_registries.size();
   f.write( (char*)&nbRegistries, 4 );
}

bool TracePack::save( unsigned int i_version, bool i_MSBF )
{
   ofstream f;
   
   if ( m_writeInUpdateMode )
      f.open( m_filename.str_unix().c_str(), ios::app|ios::binary );
   else
      f.open( m_filename.str_unix().c_str(), ios::binary );
   
   if ( !f ) return false;
   if ( !m_writeInUpdateMode )
   {
      VersionedFileHeader header( VFH_TracePack, i_version, i_MSBF );
      header.write( f );
   }
   bool res = false;
   switch ( i_version )
   {
   case 1: res=write_v1( f ); break;
   default: res=false;
   }
   m_writeInUpdateMode = res;
   
   return res;
}

void TracePack::getState( unsigned int i_iState, TracePack::Registry &o_state ) const
{
   #ifdef __DEBUG_TRACE_PACK
      if ( i_iState>=m_registries.size() )
	 cerr << "DEBUG_ERROR: TracePack::getState: index out of range " << i_iState << " >= " << m_registries.size() << endl;
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
	 cerr << "DEBUG_ERROR: TracePack::addState: directory [" << m_anchor.str_unix() << "] does not exist" << endl;
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
   directoryState.m_command = i_command;
      
   getState( nbStates()-1, lastState );
   diff.difference( lastState, directoryState );
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
	 cerr << "DEBUG_ERROR: TracePack::getRegistry: index " << i_iRegistry << " out of range (max:" << m_registries.size() << ')' << endl;
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

bool TracePack::compare( const TracePack &i_b ) const
{
   if ( m_filename!=i_b.m_filename )
      cout << "packs have different filenames : " << m_filename.str_unix() << " != " << i_b.m_filename.str_unix() << endl;
   //else if ( m_anchor!=i_b.m_anchor )
   //   cout << "packs have different anchors : " << m_anchor.str_unix() << " != " << i_b.m_anchor.str_unix() << endl;
   else if ( m_date!=i_b.m_date )
      cout << "packs have different dates : " << m_date << " != " << i_b.m_date << endl;
   else if ( m_writeInUpdateMode!=i_b.m_writeInUpdateMode )
      cout << "packs have different writing modes : " << (m_writeInUpdateMode?"create":"update") << " != " << (i_b.m_writeInUpdateMode?"create":"update") << endl;
   else if ( m_nbRegistriesOffset!=i_b.m_nbRegistriesOffset )
      cout << "packs have different offset for the number of registries : " << m_nbRegistriesOffset << " != " << i_b.m_nbRegistriesOffset << endl;
   else
   {
      unsigned int nbRegistries = m_registries.size();
      if ( nbRegistries!=i_b.m_registries.size() )
      {
	 cout << "packs have different number of registries : " << nbRegistries << " != " << i_b.m_registries.size() << endl;
	 return false;
      }
      list<TracePack::Registry>::const_iterator itA = m_registries.begin(),
						itB = i_b.m_registries.begin();
      for ( unsigned int iReg=0; iReg<nbRegistries; iReg++ )
      {
	 if ( !itA->compare( *itB ) )
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
      if ( !ELISE_fp::IsDirectory( m_anchor.str_unix() ) )
	 cerr << "DEBUG_ERROR: TracePack::setState: destination directory [" << m_anchor.str_unix() << "] does not exist" << endl;
   #endif
   
   Registry refReg, dirReg, dir_to_ref;
   getState( i_iState, refReg );
   dirReg.stateDirectory( m_anchor );
   dir_to_ref.difference( dirReg, refReg );
   dir_to_ref.dump();
   dir_to_ref.applyToDirectory( m_filename, m_anchor );
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
	 cerr << "DEBUG_ERROR: write_file: unable to open [" << i_filenameIn.str_unix() << "] for reading" << endl;
      #endif
      return false;
   }
   
   return write_file( fIn, io_fOut, i_expectedSize );
}

bool write_file( istream &io_fIn, ostream &io_fOut, unsigned int i_length )
{   
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
	    cerr <<  "DEBUG_ERROR: write_file: unable to read in input stream" << endl;
	 #endif
	 return false;
      }
      io_fOut.write( buffer.data(), nbRead );
      remaining -= (unsigned int)nbRead;
   }
   
   #ifdef __DEBUG_TRACE_PACK
      if ( remaining!=0 )
	 cerr << "DEBUG_ERROR: write_file: " << remaining << " bytes need to be read but end-of-file is reached" << endl;
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
	 cerr << "DEBUG_ERROR: load_script: script file [" << i_filename.str_unix() << "] open for reading" << endl;
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
