// this file is supposed to be included only in cElCommand.h

//-------------------------------------------
// cElCommandToken
//-------------------------------------------

bool cElCommandToken::operator !=( const cElCommandToken &i_b ) const { return !((*this)==i_b); }

template <class T> T & cElCommandToken::specialize() { return *((T *)this); }

template <class T> const T & cElCommandToken::specialize() const { return *((const T *)this); }

U_INT8 cElCommandToken::raw_size() const { return string_raw_size( str() )+4; } // 4 = type size

void cElCommandToken::trace( std::ostream &io_ostream ) const
{
   io_ostream << "(" << CmdTokenType_to_string( type() ) << "," << str() << ")" << std::endl;
}


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

unsigned int cElCommand::nbTokens() const { return m_tokens.size(); }
   
   
//-------------------------------------------
// ctPath::Token
//-------------------------------------------

ctPath::Token::Token( const std::string &i_token ):std::string(i_token){}

bool ctPath::Token::isRoot() const
{
   return length()==0 ||  // empty (unix)
          ( length()==2 && at(1)==':' ); // a volume letter + ':' (windows)
}

   
//-------------------------------------------
// ctPath
//-------------------------------------------

CmdTokenType ctPath::type() const { return CTT_Path; }

std::string ctPath::str() const { return m_normalizedName; }
   
bool ctPath::isAbsolute() const { return m_tokens.size()>1 && m_tokens.begin()->isRoot(); }

bool ctPath::isNull() const { return m_tokens.begin()==m_tokens.end(); }

bool ctPath::operator <( const ctPath &i_b ) const { return compare( i_b )<0; }

bool ctPath::operator >( const ctPath &i_b ) const { return compare( i_b )>0; }

bool ctPath::operator ==( const ctPath &i_b ) const { return compare( i_b )==0; }

bool ctPath::operator !=( const ctPath &i_b ) const { return compare( i_b )!=0; }

bool ctPath::remove() const{ return ( removeContent() && removeEmpty() ); }


//-------------------------------------------
// cElFilename
//-------------------------------------------

cElFilename::cElFilename( const ctPath &i_path, const cElFilename &i_filename ):m_path(i_path,i_filename.m_path),m_basename(i_filename.m_basename){}

cElFilename::cElFilename( const ctPath &i_path, const std::string i_basename ):m_path(i_path),m_basename(i_basename){}

std::string cElFilename::str( char i_separator ) const
{
   if ( m_path.isNull() ) return m_basename;
   return m_path.str( i_separator )+m_basename;
}

std::string cElFilename::str_unix() const { return str(ctPath::unix_separator); }

std::string cElFilename::str_windows() const { return str(ctPath::windows_separator); }

bool cElFilename::operator <( const cElFilename &i_b ) const { return compare( i_b )<0; }

bool cElFilename::operator >( const cElFilename &i_b ) const { return compare( i_b )>0; }

bool cElFilename::operator ==( const cElFilename &i_b ) const { return compare( i_b )==0; }

bool cElFilename::operator !=( const cElFilename &i_b ) const { return compare( i_b )!=0; }

bool cElFilename::setRights( mode_t o_rights ) const {
	bool res;
	#if (ELISE_POSIX)
		res = (chmod( str_unix().c_str(), o_rights )==0);
	#else
		res = true; // __TODO : windows read-only files
	#endif
	#ifdef __DEBUG_C_EL_COMMAND
		if ( !res ){
			std::cerr << RED_DEBUG_ERROR << "cannot set rights on [" << str_unix() << ']' << std::endl;
			exit(EXIT_FAILURE);
		}
	#endif
	return res;
}

bool cElFilename::getRights( mode_t &o_rights ) const
{
	bool res;
	#if (ELISE_POSIX)
		struct stat s;
		res = ( stat( str_unix().c_str(), &s )==0 );
		o_rights = s.st_mode;
		o_rights &= posixMask; // restrain rights to documented posix bits (the twelve less significant bits)
	#else
		o_rights = unhandledRights;
		res = true; // __TODO : windows read-only files
	#endif
	#ifdef __DEBUG_C_EL_COMMAND
		if ( !res ){
			std::cerr << RED_DEBUG_ERROR << "cannot get rights on [" << str_unix() << ']' << std::endl;
			exit(EXIT_FAILURE);
		}
	#endif
	return res;
}


//-------------------------------------------
// cElRegEx
//-------------------------------------------

cElRegEx::cElRegEx( const ctPath &i_path, const std::string i_regex ):cElFilename(i_path,i_regex){}

cElRegEx::cElRegEx( const std::string i_fullregex ):cElFilename(i_fullregex){}


//-------------------------------------------
// related functions
//-------------------------------------------

U_INT8 string_raw_size( const std::string &i_str ){ return 4+i_str.length(); }

std::string read_string( std::istream &io_istream, std::vector<char> &io_buffer, bool i_reverseByteOrder )
{
   U_INT4 ui;
   io_istream.read( (char*)(&ui), 4 );
   if ( i_reverseByteOrder ) byte_inv_4( &ui );
   io_buffer.resize(ui);
   io_istream.read( io_buffer.data(), ui );
   return std::string( io_buffer.data(), ui );
}

void write_string( const std::string &str, std::ostream &io_ostream, bool i_reverseByteOrder )
{
   U_INT4 ui = str.size();
   if ( i_reverseByteOrder ) byte_inv_4( &ui );
   io_ostream.write( (char*)(&ui), 4 );
   io_ostream.write( str.c_str(), ui );
}

// int4<->raw
void int4_to_raw_data( const INT4 &i_v, bool i_reverseByteOrder, char *&o_rawData )
{
   memcpy( o_rawData, &i_v, 4 );
   if ( i_reverseByteOrder ) byte_inv_4( &o_rawData );
   o_rawData += 4;
}

void int4_from_raw_data( char const *&io_rawData, bool i_reverseByteOrder, INT4 &o_v )
{
   memcpy( &o_v, io_rawData, 4 );
   if ( i_reverseByteOrder ) byte_inv_4( &o_v );
   io_rawData += 4;
}

// uint4<->raw
void uint4_to_raw_data( const U_INT4 &i_v, bool i_reverseByteOrder, char *&o_rawData )
{
   memcpy( o_rawData, &i_v, 4 );
   if ( i_reverseByteOrder ) byte_inv_4( &o_rawData );
   o_rawData += 4;
}

void uint4_from_raw_data( const char *&io_rawData, bool i_reverseByteOrder, U_INT4 &o_v )
{
   memcpy( &o_v, io_rawData, 4 );
   if ( i_reverseByteOrder ) byte_inv_4( &o_v );
   io_rawData += 4;
}
