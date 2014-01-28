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

#include "StdAfx.h"

#if (ELISE_windows)
	#include <Windows.h>
#elif (ELISE_POSIX)
	#include <unistd.h>
	#include <sys/stat.h>
#endif

#include <vector>

#define __DEBUG_C_EL_COMMAND

using namespace std;

const char   cElPath::sm_unix_separator    = '/';
const char   cElPath::sm_windows_separator = '\\';
const string cElPath::sm_all_separators    = "/\\";
   
#ifdef __DEBUG_C_EL_COMMAND
   unsigned int cElCommandToken::__nb_created = 0;
   unsigned int cElCommandToken::__nb_distroyed = 0;
#endif
   
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
// cElCommandToken
//-------------------------------------------

cElCommandToken::cElCommandToken()
{
   #ifdef __DEBUG_C_EL_COMMAND
      __nb_created++;
   #endif
}

cElCommandToken::~cElCommandToken()
{
   #ifdef __DEBUG_C_EL_COMMAND
      __nb_distroyed++;
   #endif
}

bool cElCommandToken::operator ==( const cElCommandToken &i_b ) const
{
   const CmdTokenType this_type = type();
   if ( this_type!=i_b.type() ) return false;
   
   switch ( this_type )
   {
   case CTT_RawString: return specialize<ctRawString>()==i_b.specialize<ctRawString>();
   case CTT_Path: return true;
   case CTT_Basename: return true;
   case CTT_RegEx: return true;
   case CTT_Filename: return true;
   case CTT_PathRegEx: return true;
   case CTT_Prefix: return true;
   }
   return false;
}

cElCommandToken * cElCommandToken::copy( const cElCommandToken &i_b )
{
   const CmdTokenType btype = i_b.type();
   switch ( btype )
   {
   case CTT_RawString: return new ctRawString( i_b.specialize<ctRawString>() );
   case CTT_Path: return NULL;
   case CTT_Basename: return NULL;
   case CTT_RegEx: return NULL;
   case CTT_Filename: return NULL;
   case CTT_PathRegEx: return NULL;
   case CTT_Prefix: return NULL;
   }
   return NULL;
}

cElCommandToken * cElCommandToken::allocate( CmdTokenType i_type, const std::string &i_value )
{
   switch ( i_type )
   {
   case CTT_RawString: return new ctRawString( i_value );
   case CTT_Path: return NULL;
   case CTT_Basename: return NULL;
   case CTT_RegEx: return NULL;
   case CTT_Filename: return NULL;
   case CTT_PathRegEx: return NULL;
   case CTT_Prefix: return NULL;
   }
   return NULL;
}
      
      
//-------------------------------------------
// cElCommand
//-------------------------------------------

void cElCommand::clear()
{
   list<cElCommandToken*>::iterator itToken = m_tokens.begin();
   while ( itToken!=m_tokens.end() )
      delete *itToken++;
   m_tokens.clear();
}

cElCommand & cElCommand::operator =( const cElCommand &i_b )
{
   clear();
   list<cElCommandToken*>::const_iterator itToken = i_b.m_tokens.begin();
   while ( itToken!=i_b.m_tokens.end() )
      add_a_copy( *(*itToken++) );
   return *this;
}

cElCommandToken & cElCommand::add_a_copy( const cElCommandToken &i_token )
{
   cElCommandToken *res = cElCommandToken::copy( i_token );
   
   #ifdef __DEBUG_C_EL_COMMAND
      if ( res==NULL )
      {
	 cerr <<  "cElCommand::add_a_copy: cElCommandToken::copy returned NULL" << endl;
	 ElEXIT(EXIT_FAILURE,"");
      }
   #endif
   
   m_tokens.push_back( res );
   return *res;
}

bool cElCommand::operator ==( const cElCommand &i_b ) const
{
   if ( m_tokens.size()!=i_b.m_tokens.size() ) return false;
   list<cElCommandToken*>::const_iterator itA = m_tokens.begin(),
                                          itB = i_b.m_tokens.begin();
   while ( itA!=m_tokens.end() )
      if ( *(*itA++)!=*(*itB++) ) return false;
   return true;
}

string cElCommand::str() const
{
   string res;
   list<cElCommandToken*>::const_iterator itToken = m_tokens.begin();
   if ( itToken==m_tokens.end() ) return res;
   res = (*itToken++)->str();
   const string separator = " ";
   while ( itToken!=m_tokens.end() ) res.append( separator+(*itToken++)->str() );
   return res;
}
   
bool cElCommand::write( ostream &io_ostream, bool i_inverseByteOrder ) const
{
   // write number of tokens
   U_INT4 nbTokens = m_tokens.size();
   if ( i_inverseByteOrder ) byte_inv_4( &nbTokens );
   io_ostream.write( (char*)&nbTokens, 4 );
   
   // write tokens
   string str;
   INT4 typeToWrite;
   U_INT4 length;
   list<cElCommandToken*>::const_iterator itToken = m_tokens.begin();
   while ( itToken!=m_tokens.end() )
   {
      // write token's type and the length of its string value
      typeToWrite = (INT4)(*itToken)->type();
      str = ( *itToken )->str();
      length = (U_INT4)str.length();
      io_ostream.write( (char*)(&typeToWrite), 4 );
      
      if ( i_inverseByteOrder )
      {
	 byte_inv_4( &typeToWrite );
	 byte_inv_4( &length );
      }
      
      // write token's string value
      io_ostream.write( str.c_str(), length );
      
      itToken++;
   }
   return true;
}

bool cElCommand::read( istream &io_istream, bool i_inverseByteOrder )
{
   U_INT4 nbTokens, length;
   INT4 readType;
   vector<char> buffer;
   string str;
   
   clear();
   
   // read number of tokens
   io_istream.read( (char*)&nbTokens, 4 );
   if ( i_inverseByteOrder ) byte_inv_4( &nbTokens );
   
   // read tokens
   while ( nbTokens-- )
   {
      // read the type of the token and the length of its string value
      io_istream.read( (char*)(&readType), 4 );
      io_istream.read( (char*)(&length), 4 );
      
      if ( i_inverseByteOrder )
      {
	 byte_inv_4( &readType );
	 byte_inv_4( &length );
      }
      
      // read string value
      buffer.resize( length );
      io_istream.read( buffer.data(), length );
      str.assign( buffer.data(), length );
      
      // create a new token
      m_tokens.push_back( cElCommandToken::allocate( (CmdTokenType)readType, str ) );
   }
   return true;
}

void cElCommand::set_raw( const std::string &i_str )
{
   clear();
   add_raw(i_str);
}

void cElCommand::add_raw( const std::string &i_str )
{
   clear();
   
   if ( i_str.length()==0 ) return;
   
   char *itToken = strtok( strdup( i_str.c_str() ), " " );
   ctRawString *newToken;
   while ( itToken!=NULL )
   {
      newToken = new ctRawString( string(itToken) );
      #ifdef __DEBUG_C_EL_COMMAND
	 if ( newToken==NULL )
	 {
	    cerr << "ERROR: cElCommand::cElCommand(string): new ctRawString returned NULL" << endl;
	    ElEXIT(EXIT_FAILURE,(std::string("cElCommand::add_raw with i_str=")+i_str));
	 }
      #endif
      m_tokens.push_back( newToken );
      itToken = strtok( NULL, " " );
   }
}

bool cElCommand::system() const { return ( ::System( str() )==EXIT_SUCCESS ); }

bool cElCommand::replace( const map<string,string> &i_dictionary )
{
   bool commandHasBeenModified = false;
   list<cElCommandToken*>::iterator itToken = m_tokens.begin();
   while ( itToken!=m_tokens.end() )
   {
      string str = (*itToken)->str();
      	 
      // replace expression from dictionary
      size_t pos;
      bool tokenHasBeenModified = false;
      map<string,string>::const_iterator itDico = i_dictionary.begin();
      while ( itDico!=i_dictionary.end() )
      {
	 pos=str.find( itDico->first );
	 if ( pos!=string::npos && pos!=0 && pos!=str.length()-1 )
	 {
	    str.insert( pos, itDico->second );
	    str.erase( pos+itDico->second.length(), itDico->first.length() );
	    tokenHasBeenModified = true;
	 }
	 itDico++;
      }
      
      if ( tokenHasBeenModified )
      {	 
	 // recreate a token of the same type with the new string
	 const CmdTokenType type = (*itToken)->type();
	 delete *itToken;
	 switch ( type )
	 {
	 case CTT_RawString: *itToken=new ctRawString( str ); break;
	 case CTT_Path: cerr << "cElCommand::replace: type CTT_Path not implemented" << endl; break;
	 case CTT_Basename: cerr << "cElCommand::replace: type CTT_Basename not implemented" << endl; break;
	 case CTT_RegEx: cerr << "cElCommand::replace: type CTT_RegEx not implemented" << endl; break;
	 case CTT_Filename: cerr << "cElCommand::replace: type CTT_Filename not implemented" << endl; break;
	 case CTT_PathRegEx: cerr << "cElCommand::replace: type CTT_PathRegEx not implemented" << endl; break;
	 case CTT_Prefix: cerr << "cElCommand::replace: type CTT_Prefix not implemented" << endl; break;
	 }
	 
	 commandHasBeenModified = true;
      }
      
      itToken++;
   }
   return commandHasBeenModified;
}
   

//-------------------------------------------
// cElPath
//-------------------------------------------

cElPath::cElPath( const cElPath &i_path1, const cElPath &i_path2 )
{
   *this = i_path1;
   append( i_path2 );
}

void cElPath::append( const Token &i_token )
{
   if ( i_token=="." ) return;
   if ( m_tokens.size()>0 && i_token==".." ){ m_tokens.pop_back(); return; }
   m_tokens.push_back( i_token );
}

void cElPath::append( const cElPath &i_path )
{
   if ( i_path.isAbsolute() )
   {
      #ifdef __DEBUG_C_EL_COMMAND
	 cerr << "ERROR: cElPath::append : appening absolute path [" << i_path.str_unix() << "] to a directory [" << str_unix() << "]" << endl;
      #endif
      return;
   }
   list<cElPath::Token>::const_iterator itToken = i_path.m_tokens.begin();
   while ( itToken!=i_path.m_tokens.end() )
      append( *itToken++ );
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
      append( cElPath::Token( tokenString ) );
      itPath += tokenString.length()+1;
   }
}

void cElPath::trace( ostream &io_stream ) const
{
   io_stream << m_tokens.size() << ' ';
   
   list<cElPath::Token>::const_iterator itToken = m_tokens.begin();
   while ( itToken!=m_tokens.end() )
      io_stream << "[" << (*itToken++) << "]";
   
   io_stream << ' ' << (isAbsolute()?"absolute":"relative") << endl;
}

string cElPath::str( char i_separator ) const
{
   const string separator(1,i_separator);
   if ( !isAbsolute() && m_tokens.size()==0 ) return string(".")+separator;
   list<cElPath::Token>::const_iterator itToken = m_tokens.begin();
   string res;
   while ( itToken!=m_tokens.end() )
      res.append( (*itToken++)+separator );
   return res;
}

int cElPath::compare( const cElPath &i_b ) const
{
   list<cElPath::Token>::const_iterator itA = m_tokens.begin(),
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
   list<cElPath::Token>::iterator itToken = m_tokens.begin();
   while ( itToken!=m_tokens.end() )
      res.append( *itToken++ );
   *this = res;
}

bool cElPath::isInvalid() const { return false; }

bool cElPath::exists() const { return ELISE_fp::IsDirectory( str_unix().c_str() ); }

bool cElPath::create() const
{
   ELISE_fp::MkDirRec( str_unix() );
   return exists();
}

bool cElPath::remove() const
{
   return true;
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

bool cElFilename::exists() const { return ELISE_fp::exist_file( str_unix().c_str() ); }

bool cElFilename::remove() const
{
   ELISE_fp::RmFile( str_unix() );
   return !exists();
}
   
