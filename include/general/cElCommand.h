#ifndef __C_EL_COMMAND__
#define __C_EL_COMMAND__

#include <string>
#include <list>
#include <vector>
#include <map>
#include <iostream>
#include <cstring>
#include "../include/general/sys_dep.h"
#include "../include/private/util.h"
#include "sys/stat.h"

#ifdef _MSC_VER
	#define mode_t U_INT4
#endif

class ctPath;

#define __DEBUG_C_EL_COMMAND

#if (ELISE_POSIX)
	#define RED_ERROR "\033[1;31mERROR: \033[0m"
	#define RED_DEBUG_ERROR "\033[1;31mDEBUG_ERROR: \033[0m"
	#define RED_WARNING "\033[0;31mWARNING: \033[0m"
#else
	#define RED_ERROR "ERROR: "
	#define RED_DEBUG_ERROR "DEBUG_ERROR: "
	#define RED_WARNING "WARNING: "
#endif

//-------------------------------------------
// cElCommandToken
//-------------------------------------------

typedef enum
{
   CTT_RawString,
   CTT_Path,
   CTT_Basename,
   CTT_RegEx,
   CTT_Filename,
   CTT_PathRegEx,
   CTT_Prefix
} CmdTokenType;

// this is the interface for all elements in a cElCommand
class cElCommandToken
{
public:
   #ifdef __DEBUG_C_EL_COMMAND
      static unsigned int __nb_created;
      static unsigned int __nb_distroyed;
   #endif

   cElCommandToken();
   virtual ~cElCommandToken() = 0;
   
   virtual CmdTokenType type() const = 0;
   virtual std::string str() const = 0;
   
   static cElCommandToken * copy( const cElCommandToken &i_b );
   static cElCommandToken * allocate( CmdTokenType i_type, const std::string &i_value );
      
   template <class T> T & specialize();
   template <class T> const T & specialize() const;
   bool operator ==( const cElCommandToken &i_b ) const;
   inline bool operator !=( const cElCommandToken &i_b ) const;
   
   inline void trace( std::ostream &io_ostream=std::cerr ) const;
   
   // read/write in raw binary format
   static cElCommandToken * from_raw_data( char const *&io_rawData, bool i_reverseByteOrder );
   void to_raw_data( bool i_reverseByteOrder, char *&o_rawData ) const;
   inline U_INT8 raw_size() const;
};


//-------------------------------------------
// ctRawString
//-------------------------------------------

// nothing is known about this token, it is just a string
class ctRawString : public cElCommandToken
{
protected:
   std::string m_value;
   
public:
   inline ctRawString( const std::string &i_value );
   virtual CmdTokenType type() const { return CTT_RawString; }
   std::string str() const { return m_value; }
};


//-------------------------------------------
// cElCommand
//-------------------------------------------

class cElCommand
{
private:
   std::list<cElCommandToken*> m_tokens;

public:
   inline cElCommand();
   inline cElCommand( const std::string &i_str );
   inline cElCommand( const cElCommand &i_b );
   inline ~cElCommand();
   cElCommand & operator =( const cElCommand &i_b );
   void clear();
   inline unsigned int nbTokens() const;
   
   void set_raw( const std::string &i_str );
   void add_raw( const std::string &i_str );
   cElCommandToken & add( const cElCommandToken &i_token );

   bool operator ==( const cElCommand &i_b ) const;
   inline bool operator !=( const cElCommand &i_b ) const;
   
   // read/write in raw binary format
   void from_raw_data( char const *&i_rawData, bool i_reverseByteOrder );
   void to_raw_data( bool i_reverseByteOrder, char *&o_rawData ) const;
   U_INT8 raw_size() const;
   
   // returns is the command has been modified
   bool replace( const std::map<std::string,std::string> &i_dictionary );   
   bool system() const;
   
   std::string str() const;
   std::string str_make(); // a string that can be used in a Makefile
   std::string str_system(); // a string that can be used in a call to system
   
   void trace( std::ostream &io_ostream=std::cerr ) const;
   
   void write( std::ostream &io_ostream, bool i_inverseByteOrder ) const;
   void read( std::istream &io_istream, bool i_inverseByteOrder );
};


//-------------------------------------------
// ctPath
//-------------------------------------------

class cElFilename;

class ctPath : public cElCommandToken
{
private:
   class Token : public std::string
   {
   public:
      inline      Token( const std::string &i_value );
      inline bool isRoot() const;
   };
   
   std::list<Token> m_tokens;
   std::string m_normalizedName;

public:
   static const char        sm_unix_separator;
   static const char        sm_windows_separator;
   static const std::string sm_all_separators;

   ctPath( const std::string &i_path=std::string() );
   ctPath( const ctPath &i_path1, const ctPath &i_path2 );
   ctPath( const cElFilename &i_filename );

   // cElCommandToken methods
   inline CmdTokenType type() const;
   inline std::string str() const;
   
   void append( const Token &i_token );
   void append( const ctPath &i_token );

   bool isInvalid() const;
   void trace( std::ostream &io_stream=std::cout ) const;

   inline bool isAbsolute() const; // an absolute path begins with a root token
   inline bool isNull() const; // returns if the name is empty, an empty path is the current directory
   bool isWorkingDirectory() const;

   int compare( const ctPath &i_b ) const;
   inline bool operator < ( const ctPath &i_b ) const;
   inline bool operator > ( const ctPath &i_b ) const;
   inline bool operator ==( const ctPath &i_b ) const;
   inline bool operator !=( const ctPath &i_b ) const;
   
   std::string str( char i_separator ) const;
   
   void toAbsolute( const ctPath &i_relativeTo );
   bool exists() const;
   bool create() const;
   bool remove_empty() const;
   bool remove() const;
   // return true if directory exists and containts nothing
   bool isEmpty() const; 
   
   // returns if *this contains or is i_path
   // relative paths are considered relative to working directory
   bool isAncestorOf( const ctPath &i_path ) const;
   bool isAncestorOf( const cElFilename &i_filename ) const;
   
   unsigned int count_upward_references() const; // count number of '..'
   
   bool getContent( std::list<cElFilename> &o_files ) const;
   bool getContent( std::list<cElFilename> &o_files, std::list<ctPath> &o_directories, bool i_isRecursive ) const;
};


//-------------------------------------------
// cElFilename
//-------------------------------------------

class cElFilename
{
public:
   ctPath      m_path;
   std::string m_basename;
   
   inline cElFilename( const ctPath &i_path, const std::string i_basename );
   inline cElFilename( const ctPath &i_path, const cElFilename &i_filename );
   cElFilename( const std::string i_fullname=std::string() ); // i_fullname = path+basename
   
   bool isInvalid() const;
   
   void trace( std::ostream &io_stream=std::cout ) const;
   
   int compare( const cElFilename &i_b ) const;
   inline bool operator < ( const cElFilename &i_b ) const;
   inline bool operator > ( const cElFilename &i_b ) const;
   inline bool operator ==( const cElFilename &i_b ) const;
   inline bool operator !=( const cElFilename &i_b ) const;
   
   inline std::string str( char i_separator ) const;
   inline std::string str_unix() const;
   inline std::string str_windows() const;
   
   bool exists() const;
   bool isDirectory() const;
   bool create() const;
   bool remove() const;
   inline bool setRights( mode_t o_rights ) const;
   inline bool getRights( mode_t &o_rights ) const;
   U_INT8 getSize() const;
};


//-------------------------------------------
// cElRegEx
//-------------------------------------------

// it's a Filename but with different rules for use with ::system or make
class cElRegEx : public cElFilename
{
public:   
   inline cElRegEx( const ctPath &i_path, const std::string i_regex );
   inline cElRegEx( const std::string i_fullregex=std::string() ); // i_fullname = path+regex
};


//-------------------------------------------
// related functions
//-------------------------------------------

ctPath getWorkingDirectory();
bool setWorkingDirectory( const ctPath &i_path );

inline std::string read_string( std::istream &io_istream, std::vector<char> &io_buffer, bool i_reverseByteOrder );
inline void write_string( const std::string &str, std::ostream &io_ostream, bool i_reverseByteOrder );

// take a list of filename
void specialize_filenames( const std::list<cElFilename> &i_filenames, std::list<cElFilename> &o_filenames, std::list<ctPath> &o_path );

std::string CmdTokenType_to_string( CmdTokenType i_type );

bool isCommandTokenType( int i_v );

std::string generate_random_string( unsigned int i_strLength );
std::string generate_random_string( unsigned int i_minLength, unsigned int i_maxLength );

// string<->raw
inline U_INT8 string_raw_size( const std::string &i_str );
void string_to_raw_data( const std::string &i_str, bool i_reverseByteOrder, char *&o_rawData );
void string_from_raw_data( const char *&io_rawData, bool i_reverseByteOrder, std::string &o_str );

// int4<->raw
inline void int4_to_raw_data( const INT4 &i_v, bool i_reverseByteOrder, char *&o_rawData );
inline void int4_from_raw_data( const char *&io_rawData, bool i_reverseByteOrder, INT4 &o_v );

// uint4<->raw
inline void uint4_to_raw_data( const U_INT4 &i_v, bool i_reverseByteOrder, char *&o_rawData );
inline void uint4_from_raw_data( const char *&io_rawData, bool i_reverseByteOrder, U_INT4 &o_v );

std::string file_rights_to_string( mode_t i_rights );


//----------------------------------------------------------------------

#include "cElCommand.inline.h"

#endif // __C_EL_COMMAND__
