// this file is supposed to be included only in cElCommand.h


//-------------------------------------------
// cElTokenAsIs
//-------------------------------------------

cElTokenAsIs::cElTokenAsIs( const std::string &i_token ):m_value(i_token){}

const std::string & cElTokenAsIs::str() const { return m_value; }


//-------------------------------------------
// cElTokenToQuote
//-------------------------------------------

cElTokenToQuote::cElTokenToQuote( const std::string &i_token ):m_value(i_token){}

const std::string & cElTokenToQuote::str() const { return m_value; }


//-------------------------------------------
// cElPathToken
//-------------------------------------------

cElPathToken::cElPathToken( const std::string &i_token ):m_value(i_token){}

const std::string & cElPathToken::str() const { return m_value; }


//-------------------------------------------
// cElPath
//-------------------------------------------

bool cElPath::isAbsolute() const { return m_isAbsolute; }

bool cElPath::isNull() const { return m_tokens.begin()==m_tokens.end(); }

std::string cElPath::str_unix   () const { return str('/'); }
std::string cElPath::str_windows() const { return str('\\'); }
   
bool cElPath::operator !=( const cElPath &i_b ) const { return !( (*this)==i_b ); }


//-------------------------------------------
// cElFilename
//-------------------------------------------

cElFilename::cElFilename( const cElPath &i_path, const std::string i_basename ):m_path(i_path),m_basename(i_basename){}

bool cElFilename::isNull() const{ return m_path.isNull() && (m_basename.length()==0); }

bool cElFilename::operator ==( const cElFilename &i_b ) const { return ( m_basename==i_b.m_basename ) && ( m_path==i_b.m_path ); }

bool cElFilename::operator !=( const cElFilename &i_b ) const { return !( (*this)==i_b ); }


//-------------------------------------------
// cElRegEx
//-------------------------------------------

cElRegEx::cElRegEx( const cElPath &i_path, const std::string i_regex ):cElFilename(i_path,i_regex){}

cElRegEx::cElRegEx( const std::string i_fullregex=std::string() ):cElFilename(i_fullregex){}
