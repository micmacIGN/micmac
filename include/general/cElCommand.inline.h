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

bool cElPathToken::isRoot() const
{
   return m_value.length()==0 ||  // empty (unix)
          ( m_value.length()==2 && m_value[1]==':' ); // a volume letter + ':' (windows)
}

int cElPathToken::compare( const cElPathToken &i_b ) const { return m_value.compare( m_value ); }

bool cElPathToken::operator <( const cElPathToken &i_b ) const { return compare(i_b)<0; }

bool cElPathToken::operator >( const cElPathToken &i_b ) const { return compare(i_b)>0; }

bool cElPathToken::operator ==( const cElPathToken &i_b ) const { return compare(i_b)==0; }

bool cElPathToken::operator !=( const cElPathToken &i_b ) const { return compare(i_b)!=0; }
   
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
