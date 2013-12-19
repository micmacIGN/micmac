// this file is supposed to be included only in cElCommand.h

//-------------------------------------------
// cElCommandToken
//-------------------------------------------

bool cElCommandToken::operator !=( const cElCommandToken &i_b ) const { return !((*this)==i_b); }

template <class T> T & cElCommandToken::specialize() { return *((T *)this); }

template <class T> const T & cElCommandToken::specialize() const { return *((const T *)this); }


//-------------------------------------------
// ctRawString
//-------------------------------------------

ctRawString::ctRawString( const std::string &i_value ):m_value(i_value){}

//-------------------------------------------
// cElCommand
//-------------------------------------------

cElCommand::cElCommand(){}

cElCommand::cElCommand( const std::string &i_str ){ add_raw(i_str); }

cElCommand::~cElCommand(){ clear(); }

cElCommand::cElCommand( const cElCommand &i_b ){ *this=i_b; }

bool cElCommand::operator !=( const cElCommand &i_b ) const { return !( (*this)==i_b ); }


//-------------------------------------------
// cElPath::cElPathToken
//-------------------------------------------

cElPath::Token::Token( const std::string &i_token ):std::string(i_token){}

bool cElPath::Token::isRoot() const
{
   return length()==0 ||  // empty (unix)
          ( length()==2 && at(1)==':' ); // a volume letter + ':' (windows)
}

   
//-------------------------------------------
// cElPath
//-------------------------------------------

bool cElPath::isAbsolute() const { return m_tokens.size()>1 && m_tokens.begin()->isRoot(); }

bool cElPath::isEmpty() const { return m_tokens.begin()==m_tokens.end(); }

std::string cElPath::str_unix   () const { return str(sm_unix_separator); }
std::string cElPath::str_windows() const { return str(sm_windows_separator); }

bool cElPath::operator <( const cElPath &i_b ) const { return compare( i_b )<0; }

bool cElPath::operator >( const cElPath &i_b ) const { return compare( i_b )>0; }

bool cElPath::operator ==( const cElPath &i_b ) const { return compare( i_b )==0; }

bool cElPath::operator !=( const cElPath &i_b ) const { return compare( i_b )!=0; }


//-------------------------------------------
// cElFilename
//-------------------------------------------

cElFilename::cElFilename( const cElPath &i_path, const cElFilename &i_filename ):m_path(i_path,i_filename.m_path),m_basename(i_filename.m_basename){}
   
cElFilename::cElFilename( const cElPath &i_path, const std::string i_basename ):m_path(i_path),m_basename(i_basename){}

std::string cElFilename::str( char i_separator ) const
{
   if ( m_path.isEmpty() ) return m_basename;
   return m_path.str( i_separator )+m_basename;
}

std::string cElFilename::str_unix() const { return str(cElPath::sm_unix_separator); }

std::string cElFilename::str_windows() const { return str(cElPath::sm_windows_separator); }


bool cElFilename::operator <( const cElFilename &i_b ) const { return compare( i_b )<0; }

bool cElFilename::operator >( const cElFilename &i_b ) const { return compare( i_b )>0; }

bool cElFilename::operator ==( const cElFilename &i_b ) const { return compare( i_b )==0; }

bool cElFilename::operator !=( const cElFilename &i_b ) const { return compare( i_b )!=0; }

//-------------------------------------------
// cElRegEx
//-------------------------------------------

cElRegEx::cElRegEx( const cElPath &i_path, const std::string i_regex ):cElFilename(i_path,i_regex){}

cElRegEx::cElRegEx( const std::string i_fullregex ):cElFilename(i_fullregex){}
