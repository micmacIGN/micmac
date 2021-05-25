#ifndef __C_EL_COMMAND__
#define __C_EL_COMMAND__

#include <string>
#include <list>
#include <vector>
#include <map>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include "../include/general/sys_dep.h"
#include "../include/private/util.h"
#include "sys/stat.h"

#ifdef _MSC_VER
	#define mode_t U_INT4
#endif

class ctPath;

#define __DEBUG_C_EL_COMMAND

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
	// virtual ~cElCommandToken() = 0;
	virtual ~cElCommandToken() ; // MPD => il est defini  dans le CPP, donc par purement virtuel ?

	virtual CmdTokenType type() const = 0;
	virtual std::string str() const = 0;

	static cElCommandToken * copy(const cElCommandToken &aB);
	static cElCommandToken * allocate(CmdTokenType aType, const std::string &aValue);

	template <class T> T & specialize();
	template <class T> const T & specialize() const;
	bool operator ==(const cElCommandToken &aB) const;
	bool operator !=(const cElCommandToken &aB) const;

	void trace(std::ostream &aOStream = std::cerr) const;

	// read/write in raw binary format
	static cElCommandToken * from_raw_data(char const *&aRawData, bool aReverseByteOrder);
	void to_raw_data(bool aReverseByteOrder, char *&oRawData) const;
	U_INT8 raw_size() const;
};


//-------------------------------------------
// ctRawString
//-------------------------------------------

// nothing is known about this token, it is just a string
class ctRawString: public cElCommandToken
{
protected:
	std::string m_value;

public:
	ctRawString( const std::string &aValue );
	virtual CmdTokenType type() const;
	std::string str() const;
};


//-------------------------------------------
// cElCommand
//-------------------------------------------

class cElCommand
{
private:
	std::list<cElCommandToken*> m_tokens;

public:
	cElCommand();
	cElCommand(const std::string &aStr);
	cElCommand(const cElCommand &aB);
	~cElCommand();
	cElCommand & operator =(const cElCommand &aB);
	void clear();
	unsigned int nbTokens() const;

	void set_raw(const std::string &aStr);
	void add_raw(const std::string &aStr);
	cElCommandToken & add(const cElCommandToken &aToken);

	bool operator ==(const cElCommand &aB) const;
	bool operator !=(const cElCommand &aB) const;

	// read/write in raw binary format
	void from_raw_data(char const *&aRawData, bool aReverseByteOrder);
	void to_raw_data(bool aReverseByteOrder, char *&aRawData) const;
	U_INT8 raw_size() const;

	// returns is the command has been modified
	bool replace(const std::map<std::string, std::string> &i_dictionary);   
	bool system() const;

	std::string str() const;
	std::string str_make(); // a string that can be used in a Makefile
	std::string str_system(); // a string that can be used in a call to system

	void trace(std::ostream &aOStream = std::cerr) const;

	void write(std::ostream &io_ostream, bool aReverseByteOrder) const;
	void read(std::istream &io_istream, bool aReverseByteOrder);
};


//-------------------------------------------
// ctPath
//-------------------------------------------

class cElFilename;

class ctPath: public cElCommandToken
{
private:
	class Token: public std::string
	{
	public:
		Token(const std::string &aValue);
		bool isRoot() const;
	};

	void update_normalized_name();

	std::list<Token> m_tokens;
	std::string m_normalizedName;

public:
	static const char        unix_separator;
	static const char        windows_separator;
	static const std::string all_separators;

	ctPath(const std::string &aPath = std::string());
	ctPath(const ctPath &aPath1, const ctPath &aPath2);
	ctPath(const cElFilename &aFilename);

	// cElCommandToken methods
	CmdTokenType type() const;
	std::string str() const;

	void append(const Token &aToken);
	void append(const ctPath &aPath);
	void prepend(ctPath aPath);

	bool isInvalid() const;
	void trace(std::ostream &aOStream = std::cout) const;

	bool isAbsolute() const; // an absolute path begins with a root token
	bool isNull() const; // returns if the name is empty, an empty path is the current directory
	bool isWorkingDirectory() const;

	int compare(const ctPath &aB) const;
	bool operator < (const ctPath &aB) const;
	bool operator > (const ctPath &aB) const;
	bool operator ==(const ctPath &aB) const;
	bool operator !=(const ctPath &aB) const;

	std::string str(char aSeparator) const;
	std::string str_unix() const;
	std::string str_windows() const;

	void toAbsolute(const ctPath &aRelativeTo);
	bool exists() const;
	bool create() const;
	bool removeEmpty() const;
	bool removeContent(bool aRecursive = true) const;
	bool remove() const;
	// return true if directory exists and containts nothing
	bool isEmpty() const;
	bool replaceFront(const ctPath &aOldFront, const ctPath &aNewFront);
	bool copy(ctPath aDst, bool aOverwrite = false) const;

	// returns if *this contains or is i_path
	// relative paths are considered relative to working directory
	bool isAncestorOf(const ctPath &aPath) const;
	bool isAncestorOf(const cElFilename &aFilename) const;

	unsigned int count_upward_references() const; // count number of '..'

	bool getContent(std::list<cElFilename> &oFiles) const;
	bool getContent(std::list<cElFilename> &oFiles, std::list<ctPath> &oDirectories, bool aIsRecursive) const;
};

std::ostream & operator <<(std::ostream &aStream, const ctPath &aFilename);


//-------------------------------------------
// cElFilename
//-------------------------------------------

class cElFilename
{
public:
	static const mode_t unhandledRights;
	static const mode_t posixMask;

	ctPath      m_path;
	std::string m_basename;

	cElFilename(const ctPath &aPath, const std::string aBasename);
	cElFilename(const ctPath &aPath, const cElFilename &aFilename);
	cElFilename(const std::string aFullname = std::string()); // aFullname = path + basename

	bool isInvalid() const;

	void trace(std::ostream &io_stream = std::cout) const;

	int compare(const cElFilename &aB) const;
	bool operator < (const cElFilename &aB) const;
	bool operator > (const cElFilename &aB) const;
	bool operator ==(const cElFilename &aB) const;
	bool operator !=(const cElFilename &aB) const;

	std::string str(char i_separator) const;
	std::string str() const;
	std::string str_unix() const;
	std::string str_windows() const;

	bool exists() const;
	bool isDirectory() const;
	bool create() const;
	bool remove() const;
	bool copy(const cElFilename &aDst, bool aOverwrite = false) const;
	bool setRights(mode_t oRights) const;
	bool getRights(mode_t &oRights) const;
	bool copyRights(const cElFilename &aDst) const;
	U_INT8 getSize() const;
	bool move(const cElFilename &aDstFilename) const;
};

std::ostream & operator <<(std::ostream &aStream, const cElFilename &aFilename);


//-------------------------------------------
// cElPathRegex
//-------------------------------------------

// it's a Filename but with different rules for use with ::system or make
class cElPathRegex : public cElFilename
{
public:   
	cElPathRegex(const ctPath &aPath, const std::string aRegex );
	cElPathRegex(const std::string aFullregex = std::string() ); // aFullname = path + regex
	bool getFilenames(std::list<cElFilename> &oFilenames, bool aWantFiles = true, bool aWantDirectories = false) const;
	bool copy(const ctPath &aDst) const;
};


//-------------------------------------------
// related functions
//-------------------------------------------

ctPath getWorkingDirectory();
bool setWorkingDirectory(const ctPath &aPath);

std::string read_string(std::istream &aIStream, std::vector<char> &aBbuffer, bool aReverseByteOrder);
void write_string(const std::string &aStr, std::ostream &aOStream, bool aReverseByteOrder);

// take a list of filename
void specialize_filenames(const std::list<cElFilename> &aFilenames, std::list<cElFilename> &oFilenames, std::list<ctPath> &oPath);

std::string CmdTokenType_to_string(CmdTokenType aType);

bool isCommandTokenType(int aV);

std::string generate_random_string(unsigned int aStrLength);
std::string generate_random_string(unsigned int aMinLength, unsigned int aMaxLength);

// string<->raw
U_INT8 string_raw_size(const std::string &aStr);
void string_to_raw_data(const std::string &aStr, bool aReverseByteOrder, char *&aRawData);
void string_from_raw_data(const char *&aRawData, bool aReverseByteOrder, std::string &oStr);

// int4<->raw
void int4_to_raw_data(const INT4 &aV, bool aReverseByteOrder, char *&aRawData);
void int4_from_raw_data(const char *&aRawData, bool aReverseByteOrder, INT4 &oV);

// uint4<->raw
void uint4_to_raw_data(const U_INT4 &aV, bool aReverseByteOrder, char *&oRawData);
void uint4_from_raw_data(const char *&aRawData, bool aReverseByteOrder, U_INT4 &oV);

std::string file_rights_to_string(mode_t aRights);

std::string getShortestExtension(const std::string &aBasename);


//----------------------------------------------------------------------

#include "cElCommand.inline.h"

#endif // __C_EL_COMMAND__
