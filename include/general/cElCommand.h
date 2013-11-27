#ifndef __C_EL_COMMAND__
#define __C_EL_COMMAND__

#include <string>
#include <list>
#include <iostream>

//-------------------------------------------
// cElTokenAsIs
//-------------------------------------------

// no matter how it is used, this token will used 'as is'
class cElTokenAsIs
{
private:
   std::string m_value;
public:
   inline cElTokenAsIs( const std::string &i_value );
   inline const std::string & str() const;
}


//-------------------------------------------
// cElTokenToQuote
//-------------------------------------------

// no matter how it is used, this token will used between double quotes
class cElTokenToQuote
{
private:
   std::string m_value;
public:
   inline cElTokenToQuote( const std::string &i_value );
   inline const std::string & str() const;
}


//-------------------------------------------
// cElPathToken
//-------------------------------------------

class cElPathToken
{
private:
   std::string m_value;
public:
   inline cElPathToken( const std::string &i_value );
   inline const std::string & str() const;
};

//-------------------------------------------
// cElPath
//-------------------------------------------

class cElPath
{
private:
   std::list<cElPathToken> m_tokens;
   bool                    m_isAbsolute;

public:
   cElPath( const std::string &i_path=std::string() );
   
   void append( const cElPathToken &i_token );
   
   void trace( std::ostream &io_stream=std::cout ) const;
   
   std::string str( char i_separator ) const;
   inline std::string str_unix() const;
   inline std::string str_windows() const;
   
   inline bool isAbsolute() const;
   inline bool isNull() const;
   
   bool operator ==( const cElPath &i_b ) const;
   inline bool operator !=( const cElPath &i_b ) const;
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
   cElFilename( const std::string i_fullname=std::string() ); // i_fullname = path+basename
   
   inline bool isNull() const;
   
   inline bool operator ==( const cElFilename &i_b ) const;
   inline bool operator !=( const cElFilename &i_b ) const;
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

//-------------------------------------------
// related functions
//-------------------------------------------

cElPath getCurrentDirectory();

//----------------------------------------------------------------------

#include "cElCommand.inline.h"

#endif // __C_EL_COMMAND__
