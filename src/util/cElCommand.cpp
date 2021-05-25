#include "general/errors.h"
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
   #include <dirent.h>
#endif

#include <vector>

#define __DEBUG_C_EL_COMMAND

using namespace std;

const char   ctPath::unix_separator    = '/';
const char   ctPath::windows_separator = '\\';
const string ctPath::all_separators    = "/\\";
const mode_t cElFilename::unhandledRights = (mode_t)std::numeric_limits<U_INT4>::max();
const mode_t cElFilename::posixMask = (mode_t)( (1<<13)-1 );

#ifdef __DEBUG_C_EL_COMMAND
   unsigned int cElCommandToken::__nb_created = 0;
   unsigned int cElCommandToken::__nb_distroyed = 0;
#endif

ctPath getWorkingDirectory()
{
   #if (ELISE_windows)
      vector<char> buffer( GetCurrentDirectory( 0, NULL ) );
      GetCurrentDirectory((int)buffer.size(), buffer.data());
      return ctPath( string( buffer.data() ) );
   #elif (ELISE_POSIX)
      vector<char> buffer( pathconf(".", _PC_PATH_MAX) );
      return ctPath( string( getcwd( buffer.data(), buffer.size() ) ) );
   #endif
}

bool setWorkingDirectory( const ctPath &i_path )
{
   #if (ELISE_windows)
      string path = i_path.str(ctPath::windows_separator);
      return (SetCurrentDirectory(path.c_str() ) != 0);
   #elif (ELISE_POSIX)
      return (chdir(i_path.str().c_str()) == 0);
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
   return ( type()==i_b.type() && str()==i_b.str() );
}

cElCommandToken * cElCommandToken::copy( const cElCommandToken &i_b )
{
   const CmdTokenType btype = i_b.type();
   switch ( btype )
   {
   case CTT_RawString: return new ctRawString( i_b.specialize<ctRawString>() );
   case CTT_Path: return new ctPath( i_b.specialize<ctPath>() );
   case CTT_Basename:
   case CTT_RegEx:
   case CTT_Filename:
   case CTT_PathRegEx:
   case CTT_Prefix:
      cerr << "ERROR: copy(cElCommandToken): unhandled CommandTokenType " << CmdTokenType_to_string(btype) << endl;
      exit(EXIT_FAILURE);
   }
   return NULL;
}

cElCommandToken * cElCommandToken::allocate( CmdTokenType i_type, const std::string &i_value )
{
   switch ( i_type )
   {
   case CTT_RawString: return new ctRawString( i_value );
   case CTT_Path: return new ctPath( i_value );
   case CTT_Basename:
   case CTT_RegEx:
   case CTT_Filename:
   case CTT_PathRegEx:
   case CTT_Prefix:
      ELISE_ERROR_EXIT("allocate(CmdTokenType,string): unhandled CmdTokenType " << CmdTokenType_to_string(i_type));
   }
   return NULL;
}

void cElCommandToken::to_raw_data( bool i_reverseByteOrder, char *&o_rawData ) const
{
   // copy type
   int4_to_raw_data( (INT4)type(), i_reverseByteOrder, o_rawData );
   // copy value
   string_to_raw_data( str(), i_reverseByteOrder, o_rawData );
}

cElCommandToken * cElCommandToken::from_raw_data( char const *&io_rawData, bool i_reverseByteOrder )
{
   // copy type
   INT4 i4;
   int4_from_raw_data( io_rawData, i_reverseByteOrder, i4 );

   #ifdef __DEBUG_C_EL_COMMAND
      if ( !isCommandTokenType(i4) ) ELISE_ERROR_EXIT("cElCommandToken::from_raw_data: invalid token type");
   #endif

   // copy string value of the token
   string value;
   string_from_raw_data( io_rawData, i_reverseByteOrder, value );

   cElCommandToken *res = allocate( (CmdTokenType)i4, value );

   #ifdef __DEBUG_C_EL_COMMAND
      if (res == NULL) ELISE_ERROR_EXIT("cElCommand::from_raw_data: allocate( " << CmdTokenType_to_string((CmdTokenType)i4) << ", [" << value << "] ) returned NULL");
   #endif

   return res;
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
      add( *(*itToken++) );
   return *this;
}

cElCommandToken & cElCommand::add( const cElCommandToken &i_token )
{
   cElCommandToken *res = cElCommandToken::copy( i_token );

   #ifdef __DEBUG_C_EL_COMMAND
      if ( res==NULL )
      {
     cerr <<  "cElCommand::add: cElCommandToken::copy returned NULL" << endl;
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

// read/write in raw binary format
void cElCommand::from_raw_data( char const *&io_rawData, bool i_reverseByteOrder )
{
   #ifdef __DEBUG_C_EL_COMMAND
      const char *rawData = io_rawData;
   #endif

   clear();

   // write number of tokens
   U_INT4 nbTokens;
   uint4_from_raw_data( io_rawData, i_reverseByteOrder, nbTokens );

   // write tokens
   while ( nbTokens-- )
      m_tokens.push_back( cElCommandToken::from_raw_data( io_rawData, i_reverseByteOrder ) );

	#ifdef __DEBUG_C_EL_COMMAND
		if ( rawData>io_rawData || (U_INT8)(io_rawData-rawData)!=raw_size() )
			ELISE_ERROR_EXIT("cElCommand::from_raw_data: " << (U_INT8)(io_rawData-rawData) << " copied bytes, but raw_size() = " << raw_size());
	#endif
}

void cElCommand::to_raw_data( bool i_reverseByteOrder, char *&o_rawData ) const
{
   #ifdef __DEBUG_C_EL_COMMAND
      char *rawData = o_rawData;
   #endif

   // copy number of tokens
   uint4_to_raw_data( (U_INT4)m_tokens.size(), i_reverseByteOrder, o_rawData );

   // copy tokens
   list<cElCommandToken*>::const_iterator itToken = m_tokens.begin();
   while ( itToken!=m_tokens.end() )
      ( *itToken++ )->to_raw_data( i_reverseByteOrder, o_rawData );

	#ifdef __DEBUG_C_EL_COMMAND
		if ( rawData>o_rawData || (U_INT8)(o_rawData-rawData)!=raw_size() ) ELISE_ERROR_EXIT("cElCommand::to_raw_data: " << (U_INT8)(o_rawData-rawData) << " copied bytes, but raw_size() = " << raw_size());
	#endif
}

U_INT8 cElCommand::raw_size() const
{
   U_INT8 totalSize = 4; // size of nb token
   std::list<cElCommandToken*>::const_iterator itToken = m_tokens.begin();
   while ( itToken!=m_tokens.end() )
      totalSize += ( *itToken++ )->raw_size();
   return totalSize;
}

void cElCommand::write( ostream &io_ostream, bool i_inverseByteOrder ) const
{
   // write number of tokens
   U_INT4 nbTokens = (U_INT4)m_tokens.size();
   if ( i_inverseByteOrder ) byte_inv_4( &nbTokens );
   io_ostream.write( (char*)&nbTokens, 4 );

   // write tokens
   string str;
   INT4 typeToWrite;
   list<cElCommandToken*>::const_iterator itToken = m_tokens.begin();
   while ( itToken!=m_tokens.end() )
   {
      // write token's type
      typeToWrite = (INT4)(*itToken)->type();
      if ( i_inverseByteOrder ) byte_inv_4( &typeToWrite );
      io_ostream.write( (char*)(&typeToWrite), 4 );

      // write token's string value
      write_string( ( *itToken )->str(), io_ostream, i_inverseByteOrder );

      itToken++;
   }
}

void cElCommand::read( istream &io_istream, bool i_inverseByteOrder )
{
   U_INT4 nbTokens;
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
      // read token's type
      io_istream.read( (char*)(&readType), 4 );
      if ( i_inverseByteOrder ) byte_inv_4( &readType );

      // read token's string value
      str = read_string( io_istream, buffer, i_inverseByteOrder );

      // create a new token
      m_tokens.push_back( cElCommandToken::allocate( (CmdTokenType)readType, str ) );
   }
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

bool cElCommand::system() const
{
   const string command = str();
   int res;
   #if (ELISE_windows)
      if ( command.size()!=0 && command[0]=='\"' )
     res = system_call( ( string("\"")+command+"\"" ).c_str() );
      else
     res = system_call( command.c_str() );
   #elif (ELISE_POSIX)
      res = system_call( command.c_str() );
   #else
      not implemented
   #endif
   return ( res==EXIT_SUCCESS );
}

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

void cElCommand::trace( std::ostream &io_ostream ) const
{
   io_ostream << "nbTokens = " << m_tokens.size() << endl;
   list<cElCommandToken*>::const_iterator itToken = m_tokens.begin();
   while ( itToken!=m_tokens.end() ){
      io_ostream << '\t';
      ( *itToken++ )->trace(io_ostream);
   }
   io_ostream << endl;
}


//-------------------------------------------
// ctPath
//-------------------------------------------

ctPath::ctPath( const ctPath &i_path1, const ctPath &i_path2 )
{
	*this = i_path1;
	append(i_path2);
	update_normalized_name();
}

ctPath::ctPath( const string &i_path )
{
	const size_t pathLength = i_path.length();
	if (pathLength != 0)
	{
		string path = i_path;
		// replace all '/' and '\' by '\0' in path
		char *itPath = &path[0];
		size_t iPath = path.size();
		while ( iPath-- )
		{
			if (*itPath == unix_separator || *itPath == windows_separator) *itPath = '\0';
			itPath++;
		}
		// append all strings in the token list
		itPath  = &path[0];
		const char * const pathEnd = itPath + pathLength;
		while (itPath < pathEnd)
		{
			string tokenString(itPath);
			append(ctPath::Token(tokenString));
			itPath += tokenString.length() + 1;
		}
	}
	update_normalized_name();
}

ctPath::ctPath( const cElFilename &i_filename )
{
   *this = i_filename.m_path;
   append( ctPath::Token( i_filename.m_basename ) );
	update_normalized_name();
}

void ctPath::append( const Token &i_token )
{
	if ( i_token=="." ) return;
	if ( m_tokens.size()>0 && i_token==".." && (*m_tokens.rbegin())!=".." )
	{
		#ifdef __DEBUG_C_EL_COMMAND
			if ( m_tokens.size()==1 && m_tokens.begin()->isRoot() )
			{
				cerr << "ERROR: ctPath::append(Token): path is higher than root" << endl;
				exit(EXIT_FAILURE);
			}
		#endif
		m_tokens.pop_back();
		return;
	}
	m_tokens.push_back(i_token);
	update_normalized_name();
}

void ctPath::append( const ctPath &i_path )
{
	if ( i_path.isAbsolute() )
	{
		#ifdef __DEBUG_C_EL_COMMAND
			cerr << "ERROR: ctPath::append : appening absolute path [" << i_path.str() << "] to a directory [" << str() << "]" << endl;
		#endif
		return;
	}
	list<ctPath::Token>::const_iterator itToken = i_path.m_tokens.begin();
	while ( itToken!=i_path.m_tokens.end() ) append( *itToken++ );
	update_normalized_name();
}

void ctPath::trace( ostream &io_stream ) const
{
   io_stream << m_tokens.size() << ' ';

   list<ctPath::Token>::const_iterator itToken = m_tokens.begin();
   while ( itToken!=m_tokens.end() )
      io_stream << "[" << (*itToken++) << "]";

   io_stream << ' ' << (isAbsolute() ? "absolute" : "relative") << endl;
}

string ctPath::str( char i_separator ) const
{
   const string separator(1,i_separator);
   if ( m_tokens.size()==0 ) return string(".")+separator;
   list<ctPath::Token>::const_iterator itToken = m_tokens.begin();
   string res;
   while ( itToken!=m_tokens.end() )
      res.append( (*itToken++)+separator );
   return res;
}

int ctPath::compare( const ctPath &i_b ) const
{
   list<ctPath::Token>::const_iterator itA = m_tokens.begin(),
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

void ctPath::toAbsolute( const ctPath &i_relativeTo )
{
	if (isAbsolute()) return;
	ctPath res(i_relativeTo);
	list<ctPath::Token>::iterator itToken = m_tokens.begin();
	while (itToken != m_tokens.end()) res.append(*itToken++);
	*this = res;
	update_normalized_name();
}

bool ctPath::isInvalid() const { return false; }

bool ctPath::exists() const
{
    #ifdef _MSC_VER
        string s = str();
        s.resize( s.length()-1 );
        return ELISE_fp::IsDirectory(s); // windows' stat function does not accept trailing '\' or '/'
    #endif
    return ELISE_fp::IsDirectory( str() );
}

bool ctPath::create() const
{
    if ( m_tokens.size()==0 || ( m_tokens.size()==1 && m_tokens.front().isRoot() ) ) return true;
    #ifdef _MSC_VER
        string path = str(windows_separator);
        #ifdef __DEBUG_C_EL_COMMAND
            if ( path.length()>248 ) ELISE_ERROR_EXIT("ctPath::create: paths are limited to 248 characters for creation");
        #endif
        CreateDirectory( path.c_str(), NULL );
    #else
        ELISE_fp::MkDirRec( str() );
    #endif
    return exists();
}

bool ctPath::getContent( list<cElFilename> &o_files ) const
{
    o_files.clear();
    #if ELISE_POSIX
        string normalizedName = m_normalizedName;

        DIR *d;
        struct dirent *dir;
        d = opendir( normalizedName.c_str() );

        if ( d==NULL ) return false;

        while ( ( dir=readdir(d) )!=NULL ){
            if ( strcmp( dir->d_name, "." )!=0 && strcmp( dir->d_name, ".." ) )
            o_files.push_back( cElFilename( *this, string( dir->d_name ) ) );
        }

        closedir(d);
        return true;
    #elif _MSC_VER
        WIN32_FIND_DATA ffd;
        HANDLE hFind = FindFirstFile( (str(windows_separator)+"*").c_str(), &ffd );
        string filename;

        if ( hFind==INVALID_HANDLE_VALUE ) return false;

        // while we keep finding new files
        do{
            // do not add parent directory nor current directory symbols
            if ( ( strcmp(ffd.cFileName, ".")!=0 ) && ( strcmp(ffd.cFileName, "..")!=0 ) ) o_files.push_back( cElFilename(*this,string( ffd.cFileName )) );
        } while ( FindNextFile( hFind, &ffd )!=0 );

        FindClose(hFind);
        return true;
    #else
        ELISE_ERROR_EXIT("ctPath::getContent: not implemented");
        return false;
    #endif
}

bool ctPath::getContent( std::list<cElFilename> &o_files, std::list<ctPath> &o_directories, bool i_isRecursive ) const
{
   std::list<cElFilename> unspecialized_filenames;
   if ( !getContent( unspecialized_filenames ) ) return false;
   specialize_filenames( unspecialized_filenames, o_files, o_directories );

   if (i_isRecursive)
   {
      list<ctPath>::iterator itPath = o_directories.begin();
      while ( itPath!=o_directories.end() )
     ( *itPath++ ).getContent( o_files, o_directories, false );
   }

   return true;
}

bool ctPath::removeEmpty() const
{
   if ( isWorkingDirectory() ) return false; // cannot remove working directory

    #if ELISE_POSIX
        if ( rmdir( str().c_str() )!=0 ) return false;
    #else
        string path = str(windows_separator);
        #ifdef __DEBUG_C_EL_COMMAND
            if ( path.length()>MAX_PATH ) ELISE_ERROR_EXIT("ctPath::remove_empty: paths are limited to " << MAX_PATH << " characters");
        #endif
        RemoveDirectory( path.c_str() );
    #endif

    return !exists();
}

static bool __ctPath_sup( const ctPath &i_a, const ctPath &i_b ){ return i_a>i_b; }

bool ctPath::removeContent( bool aRecursive ) const
{
   list<cElFilename> contentFiles;
   list<ctPath> contentPaths;
   if ( !getContent(contentFiles, contentPaths, aRecursive))
   {
        #ifdef __DEBUG_C_EL_COMMAND
            ELISE_ERROR_EXIT("ctPath::removeContent(): cannot get content of directory [" << str() <<']');
        #endif
       return false;
   }

    // remove all files
    list<cElFilename>::const_iterator itFile = contentFiles.begin();
    while ( itFile!=contentFiles.end() ){
        if ( !itFile->remove() ){
            #ifdef __DEBUG_C_EL_COMMAND
                ELISE_ERROR_EXIT("ctPath::removeContent(): cannot remove file [" << itFile->str_unix() <<']');
            #endif
            return false;
        }
        itFile++;
    }

	if ( !aRecursive) return true;

    // remove all empty directories
    contentPaths.sort( __ctPath_sup ); // sort in reverse order so that subdirectories come before their parents
    list<ctPath>::const_iterator itPath = contentPaths.begin();
    while ( itPath!=contentPaths.end() ){
        if ( !itPath->removeEmpty() ){
            #ifdef __DEBUG_C_EL_COMMAND
                ELISE_ERROR_EXIT("ctPath::removeContent(): cannot remove directory [" << itPath->str() << ']');
            #endif
            return false;
        }
        itPath++;
    }

   return true;
}

// count number of '..'
unsigned int ctPath::count_upward_references() const
{
   list<Token>::const_iterator it = m_tokens.begin();
   unsigned int count = 0;
   while ( it!=m_tokens.end() &&
           *it==".." )
   {
      it++;
      count++;
   }
   return count;
}

// returns if path contains i_path or is i_path
bool ctPath::isAncestorOf( const ctPath &i_path ) const
{
   ctPath absoluteParent = *this,
          absoluteChild = i_path;
   const ctPath workingDirectory = getWorkingDirectory();
   absoluteParent.toAbsolute( workingDirectory );
   absoluteChild.toAbsolute( workingDirectory );

   list<Token>::const_iterator itParent = absoluteParent.m_tokens.begin(),
                               itChild = absoluteChild.m_tokens.begin();
   while ( itParent!=absoluteParent.m_tokens.end() && itChild!=absoluteChild.m_tokens.end() && *itChild==*itParent ){
      itParent++; itChild++;
   }
   return itParent==absoluteParent.m_tokens.end();
}

bool ctPath::isAncestorOf( const cElFilename &i_filename ) const { return isAncestorOf( i_filename.m_path ); }

bool ctPath::isEmpty() const
{
   list<cElFilename> content;
   if ( !getContent( content ) ) return false;
   return ( content.size()==0 );
}

bool ctPath::isWorkingDirectory() const
{
   if ( isNull() ) return true;
   if ( (*this)==getWorkingDirectory() ) return true;
   return false;
}

void ctPath::prepend(ctPath aPath)
{
	list<Token>::const_iterator itToken = m_tokens.begin();
	while (itToken != m_tokens.end()) aPath.append(*itToken++);
	m_tokens.swap(aPath.m_tokens);
	update_normalized_name();
}

bool ctPath::replaceFront(const ctPath &aOldFront, const ctPath &aNewFront)
{
	if (m_tokens.size() < aOldFront.m_tokens.size()) return false;

	list<Token>::const_iterator itOld = aOldFront.m_tokens.begin();
	while (itOld != aOldFront.m_tokens.end())
	{
		if (*itOld++ != m_tokens.front())
		{
			ELISE_DEBUG_ERROR(true, "ctPath::replaceFront", "front != [" << aOldFront.str() << "]");
			return false;
		}
		m_tokens.pop_front();
	}

	prepend(aNewFront);
	return true;
}

bool ctPath::copy(ctPath aDst, bool aOverwrite) const
{
	if ( !aDst.exists())
	{
		aDst.create();
		if ( !cElFilename(str()).copyRights(cElFilename(aDst.str())))
		{
			ELISE_DEBUG_ERROR(true, "ctPath::copy", "failed to copy rights from [" << *this << "] to [" << aDst << "]");
			return false;
		}
	}
	else if ( !aOverwrite)
	{
		ELISE_DEBUG_ERROR(true, "ctPath::copy", "[" << *this << "] already exists");
		return false;
	}

	list<ctPath> paths;
	list<cElFilename> filenames;
	if ( !getContent(filenames, paths, true)) // true = recursive
	{
		ELISE_DEBUG_ERROR(true, "ctPath::copy", "!getContent(" << str() << ")");
		return false;
	}

	// create directories
	list<ctPath>::const_iterator itPath = paths.begin();
	while (itPath != paths.end())
	{
		ctPath dst(*itPath);
		if ( !dst.replaceFront(*this, aDst))
		{
			ELISE_DEBUG_ERROR(true, "ctPath::copy", "replaceFront (1) failed");
			return false;
		}

		if ( !dst.create())
		{
			ELISE_DEBUG_ERROR(true, "ctPath::copy", "failed to create subdir [" << dst.str() << "]")
			return false;
		}

		if ( !cElFilename(itPath->str()).copyRights(cElFilename(dst.str())))
		{
			ELISE_DEBUG_ERROR(true, "ctPath::copy", "failed to copy rights from [" << itPath->str() << "] to [" << dst.str() << "]");
			return false;
		}

		itPath++;
	}


	// copy files
	list<cElFilename>::const_iterator itFilename = filenames.begin();
	while (itFilename != filenames.end())
	{
		cElFilename dst(*itFilename);
		if ( !dst.m_path.replaceFront(*this, aDst))
		{
			ELISE_DEBUG_ERROR(true, "ctPath::copy", "replaceFront (2) failed");
			return false;
		}

		if ( !itFilename->copy(dst, aOverwrite))
		{
			ELISE_DEBUG_ERROR(true, "ctPath::copy", "failed to file [" << *itFilename << "] to [" << dst << "]");
			return false;
		}
		itFilename++;
	}

	return true;
}


//-------------------------------------------
// cElFilename
//-------------------------------------------

cElFilename::cElFilename( const std::string i_fullname )
{
   if ( i_fullname.length()==0 ) return; // this is an invalid cElFilename
   size_t pos = i_fullname.string::find_last_of( ctPath::all_separators );
   if ( pos==string::npos )
   {
      m_basename = i_fullname;
      return;
   }
   m_path = ctPath( i_fullname.substr( 0, pos ) );
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

bool cElFilename::exists() const { return ELISE_fp::exist_file( str_unix() ); }

bool cElFilename::isDirectory() const { return ctPath( *this ).exists(); }

bool cElFilename::remove() const
{
   ELISE_fp::RmFile( str_unix() );

   #ifdef __DEBUG_C_EL_COMMAND
      if (exists()) ELISE_ERROR_EXIT("cElFilename::remove: failed to remove file [" << str_unix() << ']');
   #endif

   return !exists();
}

bool cElFilename::create() const
{
   ofstream f( str_unix().c_str() );
   f.close();
   return exists();
}

U_INT8 cElFilename::getSize() const
{
   ifstream f( str_unix().c_str(), ios::binary );
   if (!f) return 0;
   f.seekg (0, f.end);
   return (U_INT8)f.tellg();
}

bool cElFilename::copy( const cElFilename &i_dst, bool i_overwrite ) const
{
	if ( !exists())
	{
		ELISE_DEBUG_ERROR(true, "cElFilename::copy", "source file [" << str() << "] does not exist");
		return false;
	}

	#if ELISE_windows
		return ELISE_fp::copy_file(str_windows(), i_dst.str_windows(), i_overwrite);
	#else
		return ELISE_fp::copy_file(str(), i_dst.str(), i_overwrite);
	#endif
}

bool cElFilename::move(const cElFilename &aDstFilename) const
{
	ELISE_DEBUG_ERROR( !exists(), "cElFilename::move", "file [" << (*this) << "] does not exist");
	ELISE_DEBUG_ERROR(aDstFilename.exists(), "cElFilename::move", "dst file [" << aDstFilename << "] already exists");
	ELISE_DEBUG_ERROR( !aDstFilename.m_path.exists(), "cElFilename::move", "dst path [" << aDstFilename.m_path << "] does not exists");

	#if ELISE_POSIX
		return rename(str().c_str(), aDstFilename.str().c_str()) == 0;
	#else
		ELISE_ERROR_EXIT("cElFilename::move: not implemented");
		return false;
	#endif
}

bool cElFilename::copyRights(const cElFilename &aDst) const
{
	mode_t rights;
	return getRights(rights) && aDst.setRights(rights);
}


//-------------------------------------------
// cElPathRegex
//-------------------------------------------

bool cElPathRegex::getFilenames( std::list<cElFilename> &o_filenames, bool aWantFiles, bool aWantDirectories ) const
{
	o_filenames.clear();

	if ( !aWantFiles && !aWantDirectories) return true;

	if ( !m_path.exists())
	{
		ELISE_DEBUG_ERROR(true, "cElPathRegex::get", "path [" << m_path.str() << "] does not exist");
		return false;
	}

	list<cElFilename> filenames;
	m_path.getContent(filenames); // false = recursive

	cElRegex regularExpression(m_basename, 1);

	list<cElFilename>::const_iterator itFilename = filenames.begin();
	while (itFilename != filenames.end())
	{
		const cElFilename &filename = (*itFilename++);
		if (regularExpression.Match(filename.m_basename) && (((aWantFiles && filename.exists()) || (aWantDirectories && filename.isDirectory()))))
			o_filenames.push_back(filename);
	}

	return true;
}

bool cElPathRegex::copy(const ctPath &aDst) const
{
	if ( !m_path.exists())
	{
		ELISE_DEBUG_ERROR(true, "cElPathRegex::copy", "path [" << m_path.str() << "] does not exist");
		return false;
	}

	cElRegex regularExpression(m_basename, 1);

	list<cElFilename> filenames;
	if ( !m_path.getContent(filenames))
	{
		ELISE_DEBUG_ERROR(true, "cElPathRegex::copy", "getContent(" << m_path.str() << ") failed");
		return false;
	}

	list<cElFilename>::const_iterator itFilename = filenames.begin();
	while (itFilename != filenames.end())
	{
		const cElFilename &filename = *itFilename++;
		if ( !regularExpression.Match(filename.m_basename)) continue;

		if (filename.exists())
		{
			if ( !filename.copy(cElFilename(aDst, filename.m_basename))) return false;
		}
		else if (filename.isDirectory())
		{
			if ( !ctPath(filename).copy(ctPath(aDst, filename.m_basename))) return false;
		}
		else
		{
			ELISE_DEBUG_ERROR(true, "cElPathRegex::copy", "[" << filename << "] is neither a file nor a directory");
			return false;
		}
	}

	return true;
}


//-------------------------------------------
// related functions
//-------------------------------------------

// take a list of filenames and split it into a list of existing filenames and existing directories
void specialize_filenames( const list<cElFilename> &i_filenames, list<cElFilename> &o_filenames, list<ctPath> &o_paths )
{
    list<cElFilename>::const_iterator itFilename = i_filenames.begin();
    while ( itFilename!=i_filenames.end() )
    {
		#ifdef __DEBUG_C_EL_COMMAND
			if ( itFilename->exists() && itFilename->isDirectory() )
				ELISE_ERROR_EXIT("specialize_filenames: filename [" << itFilename->str_unix() << "] is both an existing file and an existing directory ");
		#endif

        if ( itFilename->exists() )	o_filenames.push_back( *itFilename );
        else if ( itFilename->isDirectory() ) o_paths.push_back( ctPath(*itFilename) );

        #ifdef __DEBUG_C_EL_COMMAND
        else
            ELISE_ERROR_EXIT("specialize_filenames: filename [" << itFilename->str_unix() << "] is neither an existing file nor an existing directory ");
        #endif

        itFilename++;
    }
}

string CmdTokenType_to_string( CmdTokenType i_type )
{
   switch ( i_type )
   {
   case CTT_RawString: return "CTT_RawString";
   case CTT_Path:      return "CTT_Path";
   case CTT_Basename:  return "CTT_Basename";
   case CTT_RegEx:     return "CTT_RegEx";
   case CTT_Filename:  return "CTT_Filename";
   case CTT_PathRegEx: return "CTT_PathRegEx";
   case CTT_Prefix:    return "CTT_Prefix";
   }
   return "unknown";
}

string generate_random_string( unsigned int i_strLength )
{
   // generate a random filename
   //srand( time(NULL) );
   string randomName(i_strLength,'?');
   char c;
   for ( unsigned int i=0; i<i_strLength; i++ )
   {
      c = (char)(rand()%26);
      if ( (rand()%2)==0 )
     c += 65;
      else
     c += 97;
      randomName[i] = c;
   }
   return randomName;
}

std::string generate_random_string( unsigned int i_minLength, unsigned int i_maxLength )
{
   if ( i_minLength>=i_maxLength ) return generate_random_string( i_maxLength );
   unsigned int diff = i_maxLength-i_minLength;
   return generate_random_string( i_minLength+( rand()%(diff+1) ) );
}

void string_from_raw_data( char const *&io_rawData, bool i_reverseByteOrder, std::string &o_str )
{
   // copy length and resize output string
   U_INT4 length;
   uint4_from_raw_data( io_rawData, i_reverseByteOrder, length );
   // copy characters
   o_str.assign( io_rawData, length );
   io_rawData += length;
}

void string_to_raw_data( const std::string &i_str, bool i_reverseByteOrder, char *&o_rawData )
{
   // copy length
   const size_t length = i_str.length();
   uint4_to_raw_data( (U_INT4)length, i_reverseByteOrder, o_rawData );
   // copy characters
   memcpy( o_rawData, i_str.c_str(), length );
   o_rawData += length;
}

bool isCommandTokenType( int i_v )
{
   return ( i_v==(int)CTT_RawString ||
        i_v==(int)CTT_Path ||
            i_v==(int)CTT_Basename ||
            i_v==(int)CTT_RegEx ||
            i_v==(int)CTT_Filename ||
            i_v==(int)CTT_PathRegEx ||
            i_v==(int)CTT_Prefix );
}

std::string file_rights_to_string( mode_t i_rights )
{
    string res("___/___/___");
    #if ELISE_POSIX
       // rigths for owner
       res[0] = ( (i_rights&S_IRUSR)==0 )?'-':'r';
       res[1] = ( (i_rights&S_IWUSR)==0 )?'-':'w';
       res[2] = ( (i_rights&S_IXUSR)==0 )?'-':'x';
       // rights for group
       res[4] = ( (i_rights&S_IRGRP)==0 )?'-':'r';
       res[5] = ( (i_rights&S_IWGRP)==0 )?'-':'w';
       res[6] = ( (i_rights&S_IXGRP)==0 )?'-':'x';
       // rights for others
       res[8]  = ( (i_rights&S_IROTH)==0 )?'-':'r';
       res[9]  = ( (i_rights&S_IWOTH)==0 )?'-':'w';
       res[10] = ( (i_rights&S_IXOTH)==0 )?'-':'x';
    #endif
    return res;
}


string getShortestExtension( const string &i_basename )
{
	size_t pos = i_basename.rfind('.');
	return (pos == string::npos ? string() : i_basename.substr(pos));
}

std::ostream & operator <<(ostream &aStream, const ctPath &aPath)
{
	return aStream << aPath.str();
}

std::ostream & operator <<(ostream &aStream, const cElFilename &aFilename)
{
	return aStream << aFilename.str();
}
