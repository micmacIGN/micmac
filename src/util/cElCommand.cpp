#include "general/cElCommand.h"

#ifndef ELISE_windows
   #ifdef _WIN32
      #define ELISE_windows 1
      #define ELISE_POSIX 0
   #else
      #define ELISE_windows 0
      #define ELISE_POSIX 1
   #endif
#endif

#if (ELISE_windows)
	#include <Windows.h>
	#ifndef S_ISDIR
	#define S_ISDIR(mode)  (((mode) & S_IFMT) == S_IFDIR)
	#endif

	#ifndef S_ISREG
	#define S_ISREG(mode)  (((mode) & S_IFMT) == S_IFREG)
	#endif
#elif (ELISE_POSIX)
	#include <unistd.h>
	#include <sys/stat.h>
#endif

#include <vector>

using namespace std;

const char   cElPath::sm_unix_separator    = '/';
const char   cElPath::sm_windows_separator = '\\';
const string cElPath::sm_all_separators    = "/\\";
   
cElPath getCurrentDirectory()
{
   #if (ELISE_windows)
      vector<char> buffer( GetCurrentDirectory( 0, NULL ) );
      GetCurrentDirectory( buffer.size(), buffer.data() );
      return cElPath( string( buffer.data() ) );
   #elif (ELISE_POSIX)
      vector<char> buffer( pathconf(".", _PC_PATH_MAX) );
      return cElPath( string( getcwd( buffer.data(), buffer.size() ) ) );
   #endif
}

//-------------------------------------------
// cElPath
//-------------------------------------------

void cElPath::append( const cElPathToken &i_token )
{
   if ( i_token.str()=="." ) return;
   if ( m_tokens.size()>0 && i_token.str()==".." ){ m_tokens.pop_back(); return; }
   m_tokens.push_back( i_token );
}
   
cElPath::cElPath( const string &i_path )
{   
   const size_t pathLength = i_path.length();
   if ( pathLength==0 ) return;
   
   string path = i_path;
   // replace all '/' and '\' by '\0' in path
   char *itPath = &path[0];
   size_t iPath = path.size();
   while ( iPath-- )
   {
      if ( *itPath==sm_unix_separator || *itPath==sm_windows_separator ) *itPath='\0';
      itPath++;
   }
   // append all strings in the token list
   itPath  = &path[0];
   const char * const pathEnd = itPath+pathLength;
   while ( itPath<pathEnd )
   {
      string tokenString(itPath);
      append( cElPathToken( tokenString ) );
      itPath += tokenString.length()+1;
   }
}

void cElPath::trace( ostream &io_stream ) const
{
   io_stream << m_tokens.size() << ' ';
   
   list<cElPathToken>::const_iterator itToken = m_tokens.begin();
   while ( itToken!=m_tokens.end() )
   {
      io_stream << "[" << itToken->str() << "]";
      itToken++;
   }
   
   io_stream << ' ' << (isAbsolute()?"absolute":"relative") << endl;
}

string cElPath::str( char i_separator ) const
{
   const string separator(1,i_separator);
   if ( !isAbsolute() && m_tokens.size()==0 ) return string(".")+separator;
   list<cElPathToken>::const_iterator itToken = m_tokens.begin();
   string res;
   while ( itToken!=m_tokens.end() )
      res.append( (*itToken++).str()+separator );
   return res;
}

int cElPath::compare( const cElPath &i_b ) const
{
   list<cElPathToken>::const_iterator itA = m_tokens.begin(),
				      itB = i_b.m_tokens.begin();
   while ( itA!=m_tokens.end() &&
	   itB!=i_b.m_tokens.end() )
   {
      int compare = ( *itA++ ).compare( *itB++ );
      if ( compare!=0 ) return compare;
   }
   if ( itA==m_tokens.end() && itB==i_b.m_tokens.end() ) return 0;
   if ( itA==m_tokens.end() ) return -1;
   return 1;
}

void cElPath::toAbsolute( const cElPath &i_relativeTo )
{
   if ( isAbsolute() ) return;
   cElPath res( i_relativeTo );
   list<cElPathToken>::iterator itToken = m_tokens.begin();
   while ( itToken!=m_tokens.end() )
      res.append( *itToken++ );
   *this = res;
}

bool cElPath::isInvalid() const { return false; }

bool cElPath::exists() const
{
   struct stat status;
   return ( stat( str_unix().c_str(), &status )==0 &&
	    S_ISDIR( status.st_mode ) );
}

//-------------------------------------------
// cElFilename
//-------------------------------------------

cElFilename::cElFilename( const std::string i_fullname )
{
   if ( i_fullname.length()==0 ) return; // this is an invalid cElFilename
   size_t pos = i_fullname.string::find_last_of( cElPath::sm_all_separators );
   if ( pos==string::npos )
   {
      m_basename = i_fullname;
      return;
   }
   m_path = cElPath( i_fullname.substr( 0, pos ) );
   if ( pos==i_fullname.length()-1 ) return; // this is an invalid cElFilename
   m_basename = i_fullname.substr( pos+1 );
}

void cElFilename::trace( std::ostream &io_stream ) const
{
   m_path.trace(io_stream);
   io_stream << '{' << m_basename << '}' << endl;
}

bool cElFilename::isInvalid() const { return false; }

int cElFilename::compare( const cElFilename &i_b ) const
{
   int compare = m_path.compare(i_b.m_path);
   if (compare!=0) return compare;
   return m_basename.compare( i_b.m_basename );
}

bool cElFilename::exists() const
{
   struct stat status;
   return ( stat( str_unix().c_str(), &status )==0 &&
	    S_ISREG( status.st_mode ) );
}
