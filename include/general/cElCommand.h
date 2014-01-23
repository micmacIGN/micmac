#ifndef __C_EL_COMMAND__
#define __C_EL_COMMAND__

#include <string>
#include <list>
#include <map>
#include <iostream>

class cElPath;

#define __DEBUG_C_EL_COMMAND

//-------------------------------------------
// related functions
//-------------------------------------------

cElPath getCurrentDirectory();


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
   virtual ~cElCommandToken();
   
   virtual CmdTokenType type() const = 0;
   virtual std::string str() const = 0;
   
   static cElCommandToken * copy( const cElCommandToken &i_b );
   static cElCommandToken * allocate( CmdTokenType i_type, const std::string &i_value );
      
   template <class T> T & specialize();
   template <class T> const T & specialize() const;
   bool operator ==( const cElCommandToken &i_b ) const;
   inline bool operator !=( const cElCommandToken &i_b ) const;
};


//-------------------------------------------
// ctRawString
//-------------------------------------------

// nothing is known about this token, it is just a string
class ctRawString : public cElCommandToken
{
public:
   std::string m_value;
   
   inline ctRawString( const std::string &i_value );
   virtual CmdTokenType type() const { return CTT_RawString; }
   virtual std::string str() const { return m_value; }
};


//-------------------------------------------
// cElCommand
//-------------------------------------------

class cElCommand
{
private:
   std::list<cElCommandToken*> m_tokens;
   
   cElCommandToken & add_a_copy( const cElCommandToken &i_token );
   
public:
   inline cElCommand();
   inline cElCommand( const std::string &i_str );
   inline cElCommand( const cElCommand &i_b );
   inline ~cElCommand();
   cElCommand & operator =( const cElCommand &i_b );
   void clear();   

   void set_raw( const std::string &i_str );
   void add_raw( const std::string &i_str );

   bool operator ==( const cElCommand &i_b ) const;
   inline bool operator !=( const cElCommand &i_b ) const;
   
   bool write( std::ostream &io_ostream, bool i_inverseByteOrder ) const;
   bool read( std::istream &io_istream, bool i_inverseByteOrder );
   
   bool replace( const std::map<std::string,std::string> &i_dictionary );
   
   bool system() const;
   
   std::string str() const;
   std::string str_make(); // a string that can be used in a Makefile
   std::string str_system(); // a string that can be used in a call to system
};


//-------------------------------------------
// cElPath
//-------------------------------------------

class cElPath
{
private:
   class Token : public std::string
   {
   public:
      inline      Token( const std::string &i_value );
      inline bool isRoot() const;
   };
   
   std::list<Token> m_tokens;

public:
   
   static const char        sm_unix_separator;
   static const char        sm_windows_separator;
   static const std::string sm_all_separators;

   cElPath( const std::string &i_path=std::string() );
   cElPath( const cElPath &i_path1, const cElPath &i_path2 );

   void append( const Token &i_token );
   void append( const cElPath &i_token );

   bool isInvalid() const;
   void trace( std::ostream &io_stream=std::cout ) const;

   inline bool isAbsolute() const;
   inline bool isEmpty() const; // an empty path is the current directory

   int compare( const cElPath &i_b ) const;
   inline bool operator < ( const cElPath &i_b ) const;
   inline bool operator > ( const cElPath &i_b ) const;
   inline bool operator ==( const cElPath &i_b ) const;
   inline bool operator !=( const cElPath &i_b ) const;
   
          std::string str( char i_separator ) const;
   inline std::string str_unix() const;
   inline std::string str_windows() const;
   
   void toAbsolute( const cElPath &i_relativeTo=getCurrentDirectory() );
   bool exists() const;
   bool create() const;
   bool remove() const;
};


//-------------------------------------------
// cElFilename
//-------------------------------------------

class cElFilename
{
public:
   cElPath     m_path;
   std::string m_basename;
   
   inline cElFilename( const cElPath &i_path, const std::string i_basename );
   inline cElFilename( const cElPath &i_path, const cElFilename &i_filename );
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
   bool remove() const;
};


//-------------------------------------------
// cElRegEx
//-------------------------------------------

// it's a Filename but with different rules for use with ::system or make
class cElRegEx : public cElFilename
{
public:   
   inline cElRegEx( const cElPath &i_path, const std::string i_regex );
   inline cElRegEx( const std::string i_fullregex=std::string() ); // i_fullname = path+regex
};


//----------------------------------------------------------------------

#include "cElCommand.inline.h"

#endif // __C_EL_COMMAND__
