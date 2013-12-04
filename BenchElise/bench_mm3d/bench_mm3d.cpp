#include "StdAfx.h"

#include "general/cElCommand.h"

using namespace std;

typedef struct
{
   string name;
   int (*func)( int, char ** );
} command_t;

int help_func( int, char ** );
int snapshot_func( int, char ** );
int difference_func( int, char ** );

command_t g_commands[] = {
   { "snapshot", snapshot_func },
   { "difference", difference_func },
   { "help", help_func },
   { "", NULL } // signify the end of the list
};

class TraceTree
{
public:
   class Item
   {
   public:
      cElFilename m_filename;
      cElDate     m_lastModification;
      
      Item( const cElFilename &i_filename ):
	 m_filename(i_filename),
	 m_lastModification( -1,-1,-1,cElHour(-1,-1,-1) )
      {
	 reset_dates();
      }
      
      Item( const cElFilename &i_filename, const cElDate &i_date ):
	 m_filename(i_filename),
	 m_lastModification( i_date )
      {
      }
      
      void reset_dates()
      {
	 ELISE_fp::lastModificationDate( m_filename.str_unix(), m_lastModification );
      }
   };

   typedef enum
   {
      TT_Added,
      TT_Removed,
      TT_Modified
   } TT_Difference_Type;

   class DifferenceItem
   {
   public:
      cElFilename        m_filename;
      TT_Difference_Type m_type;
      
      DifferenceItem( const cElFilename &i_filename, TT_Difference_Type i_type ):
	 m_filename( i_filename ),
	 m_type( i_type )
      {
      }
   };
   
   list<Item> m_tree;
   
   size_t size() const { return m_tree.size(); }
   
   TraceTree()
   {
   }
   
   TraceTree( const cElPath &i_path )
   {
      cElPath path( i_path );
      path.toAbsolute();
      
      m_tree.clear();
      list<string> files = RegexListFileMatch( path.str_unix()+cElPath::sm_unix_separator, ".*", numeric_limits<INT>::max(), false );
      list<string>::iterator itFile = files.begin();
      while ( itFile!=files.end() )
	 m_tree.push_back( Item( cElFilename( *itFile++ ) ) );
   }
   
   void add( const Item &i_item )
   {
      const cElFilename &filename = i_item.m_filename;
      list<Item>::iterator it = m_tree.begin();
      while ( it!=m_tree.end() && it->m_filename<filename )
	 it++;
      if ( it==m_tree.end() )
      {
	 m_tree.push_back( i_item );
	 return;
      }
      if ( it->m_filename==filename ) return;
      m_tree.insert( it, i_item );
   }
   
   void write( ostream &io_stream ) const
   {
      U_INT4 ui = (U_INT4)m_tree.size();
      string s;
      INT4 i;
      double d;
      cElHour h(-1,-1,-1);
      io_stream.write( (char*)(&ui), 4 );
      list<Item>::const_iterator itItem = m_tree.begin();
      while ( itItem!=m_tree.end() )
      {
	 s = ( *itItem++ ).m_filename.str_unix();
	 // write filename
	 ui = s.size();
	 io_stream.write( (char*)(&ui), 4 );
	 io_stream.write( s.c_str(), ui );
	 // write last modification date
	 const cElDate &date = itItem->m_lastModification;
	 const cElHour &hour = date.H();
	 i = date.Y();
	 io_stream.write( (char*)(&i), 4 );
	 i = date.M();
	 io_stream.write( (char*)(&i), 4 );
	 i = date.D();
	 io_stream.write( (char*)(&i), 4 );
	 i = hour.H();
	 io_stream.write( (char*)(&i), 4 );
	 i = hour.M();
	 io_stream.write( (char*)(&i), 4 );
	 d = hour.S();
	 io_stream.write( (char*)(&d), sizeof(double) );
      }
   }
   
   void read( istream &io_stream )
   {
      m_tree.clear();
      U_INT4 nbFiles, ui;
      vector<char> str;
      INT4 hour, minute, year, month, day;
      double second;
      io_stream.read( (char*)&nbFiles, 4 );
      while ( nbFiles-- )
      {
	 // read filename
	 io_stream.read( (char*)(&ui), 4 );
	 str.resize(ui+1);
	 io_stream.read( str.data(), ui );
	 str[ui] = '\0';
	 // read last modification date
	 io_stream.read( (char*)(&year), 4 );
	 io_stream.read( (char*)(&month), 4 );
	 io_stream.read( (char*)(&day), 4 );
	 io_stream.read( (char*)(&hour), 4 );
	 io_stream.read( (char*)(&minute), 4 );
	 io_stream.read( (char*)(&second), sizeof(double) );
	 // add item
	 m_tree.push_back( Item( cElFilename( string( str.data() ) ), cElDate( day, month, year, cElHour(hour,minute,second ) ) ) );
      }
   }
   
   bool save( const cElFilename &i_filename ) const
   {
      ofstream f( i_filename.str_unix().c_str(), ios::binary );
      if ( !f ) return false;
      write( f );
      return true;
   }
   
   bool load( const cElFilename &i_filename )
   {
      m_tree.clear();
      ifstream f( i_filename.str_unix().c_str(), ios::binary );
      if ( !f ) return false;
      read( f );
      return true;
   }
   
   void difference( const TraceTree &i_b, list<DifferenceItem> &i_difference ) const
   {
      i_difference.clear();
      list<Item>::const_iterator itA = m_tree.begin(),
                                 itB = i_b.m_tree.begin();
      while ( itA!=m_tree.end() && itB!=i_b.m_tree.end() )
      {
	 int compare = itA->m_filename.compare( itB->m_filename );
	 if ( compare==0 )
	 {
	    if ( itA->m_lastModification!=itB->m_lastModification )
	       i_difference.push_back( DifferenceItem( itA->m_filename, TT_Modified ) );
	    itA++; itB++;
	 }
	 else
	 {
	    if ( compare<0 )
	       i_difference.push_back( DifferenceItem( (*itA++).m_filename, TT_Removed ) );
	    else
	       i_difference.push_back( DifferenceItem( (*itB++).m_filename, TT_Added ) );
	 }
      }
      while ( itA!=m_tree.end() )
	 i_difference.push_back( DifferenceItem( (*itA++).m_filename, TT_Removed ) );
      while ( itB!=i_b.m_tree.end() )
	 i_difference.push_back( DifferenceItem( (*itB++).m_filename, TT_Added ) );
   }
   
   // update last modification dates
   void update()
   {
      list<Item>::iterator itItem = m_tree.begin();
      while ( itItem!=m_tree.end() )
	 ( *itItem++ ).reset_dates();
   }
};

ostream & operator <<( ostream &s, const cElHour &h )
{
   return ( s << h.H() << 'h' << h.M() << 'm' << h.S() << 's' );
}

ostream & operator <<( ostream &s, const cElDate &d )
{
   return ( s << d.Y() << 'y' << d.M() << 'm' << d.D() << 'd' << d.H() );
}

int help_func( int argc, char **argv )
{
   cout << "help" << endl;
   return EXIT_SUCCESS;
}

int snapshot_func( int argc, char **argv )
{   
   if ( argc<2 ){ cerr << "not enough args" << endl; return EXIT_FAILURE; }
   const cElPath directory( (string)(argv[0]) ); 
   const cElFilename saveFilename( (string)(argv[1]) );
   TraceTree tree( directory );
      
   cout << "snaping [" << directory.str_unix() << "] to [" << saveFilename.str_unix() << "]" << endl;
   
   if ( !tree.save( saveFilename ) )
   {
      cerr << "ERROR: unable to save to [" << saveFilename.str_unix() << "]" << endl;
      return EXIT_FAILURE;
   }
   cerr << tree.size() << " items" << endl;
   
   return EXIT_SUCCESS;
}

// get the modifications from snapshot to directory
// 2 args : print the difference
// 3 args : write a difference file
int difference_func( int argc, char **argv )
{
   if ( argc<2 ){ cerr << "not enough args" << endl; return EXIT_FAILURE; }
   const cElFilename srcFilename( (string)(argv[0]) );
   const cElPath directory( (string)(argv[1]) ); 
   
   TraceTree srcTree;
   if ( !srcTree.load( srcFilename ) )
   {
      cerr << "ERROR: unable to load [" << srcFilename.str_unix() << "]" << endl;
      return EXIT_FAILURE;
   }
   TraceTree dstTree( directory );
   
   list<TraceTree::DifferenceItem> diff;
   srcTree.difference( dstTree, diff );
      
   int nbAdded = 0,
       nbRemoved = 0,
       nbModified = 0;
   
   list<TraceTree::DifferenceItem>::iterator itDiff = diff.begin();
   while ( itDiff!=diff.end() )
   {
      switch ( itDiff->m_type )
      {
      case TraceTree::TT_Added:
	 cout << "---> ADDED    : " << itDiff->m_filename.str_unix() << endl;
	 nbAdded++;
	 break;
      case TraceTree::TT_Removed:
	 cout << "---> REMOVED  : " << itDiff->m_filename.str_unix() << endl;
	 nbRemoved++;
	 break;
      case TraceTree::TT_Modified:
	 cElDate d(-1,-1,-1,cElHour(-1,-1,-1));
	 ELISE_fp::lastModificationDate( itDiff->m_filename.str_unix(), d );
	 cout << "---> MODIFIED : " << itDiff->m_filename.str_unix() << ' ' << d << endl;
	 nbModified++;
	 break;
      }
      itDiff++;
   }
   
   cout << nbAdded << " added file" << (nbAdded>1?'\0':'s') << endl;
   cout << nbRemoved << " removed file" << (nbRemoved>1?'\0':'s') << endl;
   cout << nbModified << " modified file" << (nbModified>1?'\0':'s') << endl;
   
   return EXIT_SUCCESS;
}

string normalizedCommand( char *i_command )
{
   vector<char> res( strlen( i_command )+1 );
   // remove starting '-'
   char *itSrc = i_command;
   while ( *itSrc!='\0' && *itSrc=='-' )
      itSrc++;
   // lower all letters
   char *itDst = res.data();
   while ( *itSrc!='\0' )
      *itDst++ = tolower( *itSrc++ );
   *itDst = '\0';
   return string( res.data() );
}

int main( int argc, char **argv )
{
   if ( argc<2 ) help_func(0,NULL);
   
   string command = normalizedCommand( argv[1] );
   command_t *itCommand = g_commands;
   while ( itCommand->func!=NULL )
   {
      if (command==itCommand->name)
	 return (*itCommand->func)(argc-2, argv+2);
      itCommand++;
   }
   
   help_func(0,NULL);
   
   return EXIT_SUCCESS;
}
