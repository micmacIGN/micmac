#include "TracePack.h"

#include "StdAfx.h"

#include "general/cElCommand.h"

using namespace std;

#ifdef _DEBUG
   const string __difference_illegal_item_message = "TracePack::Registry::difference: illegal action item";
   const string __apply_illegal_action_message = "TracePack::Registry::apply: illegal action item";
   const string __apply_illegal_state_message = "TracePack::Registry::apply: illegal state item";
   const string __apply_illegal_no_add_message = __apply_illegal_action_message+" (TD_Added expected)";
   void debug_error( const string &i_message )
   {
      cerr << "DEBUG_ERROR: " << i_message << endl;
      exit(-1);
   }
#endif

//--------------------------------------------
// class TracePack::Registry::Item
//--------------------------------------------
      
TracePack::Registry::Item::Item( const cElFilename &i_filename,
	                         TD_Type i_type,
	                         const cElDate &i_date,
	                         const cElFilename &i_dataFile,
	                         unsigned int i_dataOffset,
	                         unsigned int i_dataSize ):
   m_filename( i_filename ),
   m_type( i_type ),
   m_date( i_date ),
   m_dataFile( i_dataFile ),
   m_dataOffset( i_dataOffset ),
   m_dataSize( i_dataOffset )
{
}

void TracePack::Registry::Item::reset_date()
{
   ELISE_fp::lastModificationDate( m_filename.str_unix(), m_date );
}

/*
TracePack::Registry::Item::Item( const cElFilename &i_filename, TD_Type i_type ):
   m_filename(i_filename),
   m_type( i_type ),
   m_date( cElDate::NoDate )
{
   reset_date();
}
*/

void TracePack::Registry::Item::dump( ostream &io_ostream, const string &i_prefix ) const
{
   io_ostream << i_prefix << "filename   : [" << m_filename.str_unix() << ']' << endl;
   io_ostream << i_prefix << "type       : " << TD_Type_to_string(m_type) << endl;
   io_ostream << i_prefix << "date       : " << m_date << endl;
   io_ostream << i_prefix << "dataFile   : " << m_dataFile.str_unix() << endl;
   io_ostream << i_prefix << "dataOffset : " << m_dataOffset << endl;
   io_ostream << i_prefix << "dataSize   : " << m_dataSize << endl;
}

//--------------------------------------------
// class TracePack::Registry
//--------------------------------------------

TracePack::Registry::Registry()
{
}
      
void TracePack::Registry::difference( const Registry &i_a, const Registry &i_b )
{   
   m_items.clear();
   cElFilename nofilename;
   list<TracePack::Registry::Item>::const_iterator itA = i_a.m_items.begin(),
                                                   itB = i_b.m_items.begin();
   while ( itA!=i_a.m_items.end() && itB!=i_b.m_items.end() )
   {
      #ifdef _DEBUG
	 if ( itA->m_type!=TD_State ||
              itB->m_type!=TD_State )
	    debug_error( __difference_illegal_item_message );
      #endif
      
      int compare = itA->m_filename.compare( itB->m_filename );
      if ( compare==0 )
      {
	 if ( itA->m_date!=itB->m_date )
	 {
	    m_items.push_back( Item( itA->m_filename, TD_Modified, itB->m_date, itB->m_dataFile, itB->m_dataOffset, itB->m_dataSize ) );
	 }
	 itA++; itB++;
      }
      else
      {
	 if ( compare<0 )
	    m_items.push_back( Item( (*itA++).m_filename, TD_Removed, cElDate::NoDate, nofilename, 0, 0 ) );
	 else
	    m_items.push_back( Item( (*itB++).m_filename, TD_Added, itB->m_date, itB->m_dataFile, itB->m_dataOffset, itB->m_dataSize ) );
      }
   }
   while ( itA!=i_a.m_items.end() )
   {
      #ifdef _DEBUG
	 if ( itA->m_type!=TD_State ) debug_error( __difference_illegal_item_message );
      #endif
      m_items.push_back( Item( (*itA++).m_filename, TD_Removed, cElDate::NoDate, nofilename, 0, 0 ) );
   }
   while ( itB!=i_b.m_items.end() )
   {
      #ifdef _DEBUG
	 if ( itB->m_type!=TD_State ) debug_error( __difference_illegal_item_message );
      #endif
      m_items.push_back( Item( (*itB++).m_filename, TD_Added, itB->m_date, itB->m_dataFile, itB->m_dataOffset, itB->m_dataSize ) );
   }
   
   // __DEL
   #ifdef _DEBUG
      check_sorted();
   #endif
}
   
void TracePack::Registry::write_v1( ostream &io_stream/*, bool m_inverseByteOrder*/ ) const
{
   U_INT4 ui = (U_INT4)m_items.size(),
          dataCoordinates[2];
   INT4 i;
   string s;
   io_stream.write( (char*)(&ui), 4 );
   list<Item>::const_iterator itItem = m_items.begin();
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
	 // write data filename
	 s = itItem->m_filename.str_unix();
	 ui = s.size();
	 io_stream.write( (char*)(&ui), 4 );
	 io_stream.write( s.c_str(), ui );
	 // write data offset and length
	 dataCoordinates[0] = (U_INT4)itItem->m_dataOffset;
	 dataCoordinates[1] = (U_INT4)itItem->m_dataSize;
	 io_stream.write( (char*)dataCoordinates, 8 );
      }
      itItem++;
   }
}

void TracePack::Registry::read_v1( istream &io_stream )
{
   m_items.clear();
   U_INT4 nbFiles, ui,
          fileCoordinates[2];
   INT4 i;
   vector<char> str;
   cElFilename itemFilename, dataFilename;
   cElDate d( cElDate::NoDate );
   io_stream.read( (char*)&nbFiles, 4 );
   while ( nbFiles-- )
   {
      // read filename
      io_stream.read( (char*)(&ui), 4 );
      str.resize(ui+1);
      io_stream.read( str.data(), ui );
      str[ui] = '\0';
      itemFilename = cElFilename( string( str.data() ) );
      // write type
      io_stream.read( (char*)(&i), 4 );
      // add item
      if ( i==TracePack::Registry::TD_State ||
           i==TracePack::Registry::TD_Added ||
           i==TracePack::Registry::TD_Modified )
      {
	 d.read_raw( io_stream );
	 // read filename
	 io_stream.read( (char*)(&ui), 4 );
	 str.resize(ui+1);
	 io_stream.read( str.data(), ui );
	 str[ui] = '\0';
	 dataFilename = cElFilename( string( str.data() ) );
	 // read offset and length
	 io_stream.read( (char*)fileCoordinates, 8 );
      }
      else
	 d = cElDate::NoDate;
      m_items.push_back( Item( itemFilename, (TD_Type)i, d, dataFilename, fileCoordinates[0], fileCoordinates[1] ) );
   }
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
   
   #ifdef _DEBUG
      if ( !path.exists() ) debug_error( "TracePack::Registry::stateDirectory: directory ["+i_path.str_unix()+"] does not exist"  );
   #endif
   
   m_items.clear();
   cElDate date = cElDate::NoDate;
   int fileLength;
   list<string> files = RegexListFileMatch( path.str_unix()+cElPath::sm_unix_separator, ".*", numeric_limits<INT>::max(), false );
   list<string>::iterator itFile = files.begin();
   while ( itFile!=files.end() )
   {
      cElFilename filename( *itFile++ );
      const string unix_filename = filename.str_unix();
      ELISE_fp::lastModificationDate( unix_filename, date );
      fileLength = ELISE_fp::file_length( unix_filename );
      #ifdef _DEBUG
	 if ( fileLength<0 )
	    debug_error( "TracePack::Registry::stateDirectory: cannot read length of file ["+unix_filename+"]"  );
      #endif
      add( Item( filename, TD_State, date, filename, 0, (unsigned int)fileLength ) );
   }
}

void TracePack::Registry::apply( const TracePack::Registry &i_actions )
{
   list<TracePack::Registry::Item>::iterator itA = m_items.begin();
   list<TracePack::Registry::Item>::const_iterator itB = i_actions.m_items.begin();
   while ( itA!=m_items.end() && itB!=i_actions.m_items.end() )
   {      
      #ifdef _DEBUG
	 if ( itA->m_type!=TD_State ) debug_error( __apply_illegal_action_message );
	 if ( itB->m_type==TD_State ) debug_error( __apply_illegal_state_message );
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
	    itA->m_dataFile   = itB->m_dataFile;
	    itA->m_dataOffset = itB->m_dataOffset;
	    itA->m_dataSize   = itB->m_dataSize;
	    itA++; itB++;
	 }
	 #ifdef _DEBUG
	    else
	       debug_error( "TracePack::Registry::apply: illegal action item of type TD_Added" );
	 #endif
      }
      else
      {
	 if ( compare<0 )
	    itA++;
	 else
	 {
	    #ifdef _DEBUG
	       if ( itB->m_type!=TD_Added )
		  debug_error( __apply_illegal_no_add_message );
	       else
	    #endif
	       {
		  m_items.insert( itA, Item( itB->m_filename, TD_State, itB->m_date, itB->m_dataFile, itB->m_dataOffset, itB->m_dataSize ) );
		  itB++;
	       }
	 }
      }
   }
   while ( itB!=i_actions.m_items.end() )
   {
      #ifdef _DEBUG
	 if ( itB->m_type!=TD_Added )
	    debug_error( __apply_illegal_no_add_message );
	 else
      #endif
	 {
	    m_items.push_back( Item( itB->m_filename, TD_State, itB->m_date, itB->m_dataFile, itB->m_dataOffset, itB->m_dataSize ) );
	    itB++;
	 }
   }
   
   // __DEL
   #ifdef _DEBUG
      check_sorted();
   #endif
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
	 
#ifdef _DEBUG
   void TracePack::Registry::check_sorted() const
   {
      if ( m_items.size()<2 ) return;
      list<Item>::const_iterator it1 = m_items.begin(),
                                 it2 = it1;
      it2++;
      while ( it2!=m_items.end() )
      {
	 if ( (*it1++).m_filename.compare( (*it2++).m_filename )>=0 )
	    debug_error( "TracePack::Registry::check_sorted: items are not corretly sorted or an item appears more than once" );
      }
   }
#endif


//--------------------------------------------
// class TracePack
//--------------------------------------------

TracePack::TracePack():
   m_date( cElDate::NoDate )
{
   cElDate::getCurrentDate_UTC( m_date );
}

bool TracePack::read_v1( istream &f, bool i_reverseByteOrder )
{
   U_INT4 nbRegistries;
   m_date.read_raw( f );
   f.read( (char*)&nbRegistries, 4 );
   m_registries.resize( nbRegistries );
   list<Registry>::iterator itReg = m_registries.begin();
   while ( nbRegistries-- )
      ( *itReg++ ).read_v1( f );
   return true;
}

bool TracePack::load( const cElFilename &i_filename )
{
   ifstream f( i_filename.str_unix().c_str(), ios::binary );
   if ( !f ) return false;
   VersionedFileHeader header( VFH_TracePack );
   header.read_known( VFH_TracePack, f );
   switch ( header.version() )
   {
   case 1:
      return read_v1( f, header.isMSBF()==MSBF_PROCESSOR() );
   default:
      return false;
   }
}

void TracePack::write_v1( ostream &f, bool i_reverseByteOrder ) const
{
   m_date.write_raw( f );
   U_INT4 nbRegistries = (U_INT4)m_registries.size();
   f.write( (char*)&nbRegistries, 4 );
   list<Registry>::const_iterator itReg = m_registries.begin();
   while ( itReg!=m_registries.end() )
      ( *itReg++ ).write_v1( f );
}

bool TracePack::save( const cElFilename &i_filename, unsigned int i_version, bool i_MSBF ) const
{
   ofstream f( i_filename.str_unix().c_str(), ios::binary );
   if ( !f ) return false;
   VersionedFileHeader header( VFH_TracePack, i_version, i_MSBF );
   header.write( f );
   switch ( i_version )
   {
   case 1:
      write_v1( f, i_MSBF!=MSBF_PROCESSOR() );
      return true;
   default:
      return false;
   }
   return true;
}

void TracePack::getState( unsigned int i_iState, TracePack::Registry &o_state ) const
{
   #ifdef _DEBUG
      if ( i_iState>=m_registries.size() )
      {
	 stringstream ss;
	 ss << "TracePack::getState: index out of range " << i_iState << " >= " << m_registries.size();
	 debug_error( ss.str() );
      }
   #endif
   list<TracePack::Registry>::const_iterator itReg = m_registries.begin();
   o_state = *itReg++;
   
   if ( i_iState==0 ) return;
   
   while ( i_iState-- )      
      o_state.apply( *itReg++ );
}

void TracePack::addStateFromDirectory( const cElPath &i_path )
{
   #ifdef _DEBUG
      if ( !i_path.exists() )
	 debug_error( (string)("TracePack::addState: directory [")+i_path.str_unix()+"] does not exist" );
   #endif
   
   if ( nbStates()==0 )
   {
      Registry reg0;
      reg0.stateDirectory( i_path );
      m_registries.push_back( reg0 );
      return;
   }
   
   Registry directoryState, lastState, diff;
   directoryState.stateDirectory( i_path );
      
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
   #ifdef _DEBUG
      if ( i_iRegistry>=m_registries.size() )
      {
	 stringstream ss;
	 ss << "TracePack::getRegistry: index " << i_iRegistry << " out of range (max:" << m_registries.size() << ')';
	 debug_error( ss.str() );
      }
   #endif
   list<Registry>::const_iterator itReg = m_registries.begin();
   while ( i_iRegistry-- ) itReg++;
   return *itReg;
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
