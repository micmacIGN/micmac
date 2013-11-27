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
#elif (ELISE_POSIX)
   #include <unistd.h>
#endif

#include <vector>

using namespace std;

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
   if ( i_token.str().length()==0 || i_token.str()=="." ) return;
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
      if ( *itPath=='/' || *itPath=='\\' ) *itPath='\0';
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
   
   if ( m_tokens.size()==0 ){ m_isAbsolute=true; return; }
   
   // there is no ELISE_windows/ELISE_POSIX test here because a system may want to manipulate files for another system
   const unsigned int firstTokenSize = m_tokens.begin()->str().length();
   m_isAbsolute = ( firstTokenSize==0 ) ||  // first token is empty (unix)
                  ( (firstTokenSize==2)&&(m_tokens.begin()->str()[1]==':') ); // first token is a volume letter + ':' (windows)
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
   if ( isNull() ) return string();
   const string separator(1,i_separator);
   list<cElPathToken>::const_iterator itToken = m_tokens.begin();
   string res = (*itToken++).str();
   while ( itToken!=m_tokens.end() )
      res.append( separator+(*itToken++).str() );
   return res;
}

bool cElPath::operator ==( const cElPath &i_b ) const
{
   list<cElPathToken>::const_iterator itA = m_tokens.begin(),
				  itB = i_b.m_tokens.begin();
   while ( itA!=m_tokens.end() &&
	   itB!=i_b.m_tokens.end() )
      if ( (*itA++).str()!=(*itB++).str() ) return false;
   return ( itA==m_tokens.end() ) && ( itB==i_b.m_tokens.end() );
}
      
//-------------------------------------------
// cElFilename
//-------------------------------------------

cElFilename::cElFilename( const std::string i_fullname )
{
   if ( i_fullname.length()==0 ) return;
}
