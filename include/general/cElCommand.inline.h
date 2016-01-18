// this file is supposed to be included only in cElCommand.h

//-------------------------------------------
// cElCommandToken
//-------------------------------------------

inline bool cElCommandToken::operator !=(const cElCommandToken &aB) const
{
	return !((*this) == aB);
}

template <class T>
inline T & cElCommandToken::specialize()
{
	return *((T *)this);
}

template <class T>
inline const T & cElCommandToken::specialize() const
{
	return *((const T *)this);
}

inline U_INT8 cElCommandToken::raw_size() const
{
	return string_raw_size(str()) + 4; // 4 = type size
}

inline void cElCommandToken::trace(std::ostream &aOstream) const
{
	aOstream << "(" << CmdTokenType_to_string(type()) << "," << str() << ")" << std::endl;
}


//-------------------------------------------
// ctRawString
//-------------------------------------------

inline ctRawString::ctRawString(const std::string &aValue):
	m_value(aValue)
{
}

inline CmdTokenType ctRawString::type() const
{
	return CTT_RawString;
}

inline std::string ctRawString::str() const
{
	return m_value;
}


//-------------------------------------------
// cElCommand
//-------------------------------------------

inline cElCommand::cElCommand()
{
}

inline cElCommand::cElCommand(const std::string &aStr)
{
	add_raw(aStr);
}

inline cElCommand::~cElCommand()
{
	clear();
}

inline cElCommand::cElCommand(const cElCommand &aB)
{
	*this = aB;
}

inline bool cElCommand::operator !=(const cElCommand &aB) const
{
	return !((*this) == aB);
}

inline unsigned int cElCommand::nbTokens() const
{
	return (unsigned int)m_tokens.size();
}


//-------------------------------------------
// ctPath::Token
//-------------------------------------------

inline ctPath::Token::Token(const std::string &aToken):
	std::string(aToken)
{
}

inline bool ctPath::Token::isRoot() const
{
	return length() == 0 ||  // empty (unix)
		( length()==2 && at(1)==':' ); // a volume letter + ':' (windows)
}

   
//-------------------------------------------
// ctPath
//-------------------------------------------

inline CmdTokenType ctPath::type() const
{
	return CTT_Path;
}

inline std::string ctPath::str() const
{
	return m_normalizedName;
}

inline std::string ctPath::str_unix() const
{
	return str(ctPath::unix_separator);
}

inline std::string ctPath::str_windows() const
{
	return str(ctPath::windows_separator);
}
   
inline bool ctPath::isAbsolute() const
{
	return !m_tokens.empty() && m_tokens.begin()->isRoot();
}

inline bool ctPath::isNull() const
{
	return m_tokens.begin() == m_tokens.end();
}

inline bool ctPath::operator <(const ctPath &aB) const
{
	return compare(aB) < 0;
}

inline bool ctPath::operator >(const ctPath &aB) const
{
	return compare(aB) > 0;
}

inline bool ctPath::operator ==(const ctPath &aB) const
{
	return compare(aB) == 0;
}

inline bool ctPath::operator !=(const ctPath &aB) const
{
	return compare(aB) != 0;
}

inline bool ctPath::remove() const
{
	return removeContent() && removeEmpty();
}

inline void ctPath::update_normalized_name()
{
   m_normalizedName = str(unix_separator);
}


//-------------------------------------------
// cElFilename
//-------------------------------------------

inline cElFilename::cElFilename(const ctPath &aPath, const cElFilename &aFilename):
	m_path(aPath, aFilename.m_path),
	m_basename(aFilename.m_basename)
{
}

inline cElFilename::cElFilename(const ctPath &aPath, const std::string aBasename):
	m_path(aPath),
	m_basename(aBasename)
{
}

inline std::string cElFilename::str(char aSeparator) const
{
   if (m_path.isNull()) return m_basename;
   return m_path.str(aSeparator) + m_basename;
}

inline std::string cElFilename::str_unix() const
{
	return str(ctPath::unix_separator);
}

inline std::string cElFilename::str_windows() const
{
	return str(ctPath::windows_separator);
}

inline std::string cElFilename::str() const
{
	return str_unix();
}

inline bool cElFilename::operator <(const cElFilename &aB) const
{
	return compare(aB) < 0;
}

inline bool cElFilename::operator >(const cElFilename &aB) const
{
	return compare(aB) > 0;
}

inline bool cElFilename::operator ==(const cElFilename &aB) const
{
	return compare(aB) == 0;
}

inline bool cElFilename::operator !=(const cElFilename &aB) const
{
	return compare(aB) != 0;
}

inline bool cElFilename::setRights(mode_t aRights) const
{
	bool res;
	#if (ELISE_POSIX)
		res = (chmod(str_unix().c_str(), aRights) == 0);
	#else
		res = true; // __TODO : windows read-only files
	#endif
	#ifdef __DEBUG_C_EL_COMMAND
		if ( !res) ELISE_ERROR_EXIT("cannot set rights on [" << str_unix() << ']');
	#endif
	return res;
}

inline bool cElFilename::getRights(mode_t &oRights) const
{
	bool res;
	#if (ELISE_POSIX)
		struct stat s;
		res = (stat(str_unix().c_str(), &s) == 0);
		oRights = s.st_mode;
		oRights &= posixMask; // restrain rights to documented posix bits (the twelve less significant bits)
	#else
		oRights = unhandledRights;
		res = true; // __TODO : windows read-only files
	#endif

	ELISE_DEBUG_ERROR( !res, "cElFilename::getRights", "cannot get rights on [" << str() << ']');

	return res;
}


//-------------------------------------------
// cElPathRegex
//-------------------------------------------

inline cElPathRegex::cElPathRegex(const ctPath &aPath, const std::string aRegex):
	cElFilename(aPath, aRegex)
{
}

inline cElPathRegex::cElPathRegex(const std::string aFullregex):
	cElFilename(aFullregex)
{
}


//-------------------------------------------
// related functions
//-------------------------------------------

inline U_INT8 string_raw_size(const std::string &aStr)
{
	return (U_INT8)(aStr.length() + 4);
}

inline std::string read_string(std::istream &aIStream, std::vector<char> &oBuffer, bool aReverseByteOrder)
{
	U_INT4 ui;
	aIStream.read((char *)( &ui), 4);
	if (aReverseByteOrder) byte_inv_4( &ui);
	oBuffer.resize(ui);
	aIStream.read(oBuffer.data(), ui);
	return std::string(oBuffer.data(), ui);
}

inline void write_string(const std::string &aStr, std::ostream &aOStream, bool aReverseByteOrder)
{
	U_INT4 ui = (U_INT4)aStr.size();
	if (aReverseByteOrder) byte_inv_4( &ui);
	aOStream.write((char *)(&ui), 4);
	aOStream.write(aStr.c_str(), ui);
}

// int4<->raw
inline void int4_to_raw_data(const INT4 &aV, bool aReverseByteOrder, char *&aRawData)
{
	memcpy(aRawData, &aV, 4);
	if (aReverseByteOrder) byte_inv_4( &aRawData);
	aRawData += 4;
}

inline void int4_from_raw_data(char const *&aRawData, bool aReverseByteOrder, INT4 &oV)
{
	memcpy( &oV, aRawData, 4);
	if (aReverseByteOrder) byte_inv_4( &oV);
	aRawData += 4;
}

// uint4<->raw
inline void uint4_to_raw_data(const U_INT4 &aV, bool aReverseByteOrder, char *&aRawData)
{
	memcpy(aRawData, &aV, 4);
	if (aReverseByteOrder) byte_inv_4( &aRawData);
	aRawData += 4;
}

inline void uint4_from_raw_data(const char *&aRawData, bool aReverseByteOrder, U_INT4 &oV)
{
	memcpy(&oV, aRawData, 4);
	if (aReverseByteOrder) byte_inv_4( &oV);
	aRawData += 4;
}
