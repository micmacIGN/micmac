#define EXTERNAL_TOOLS_SUBDIRECTORY "binaire-aux"
#define INTERNAL_TOOLS_SUBDIRECTORY "bin"

typedef enum {
	EXT_TOOL_UNDEF 			 = 0,
	EXT_TOOL_NOT_FOUND 		 = 1,	// the tool has been researched and not found
	EXT_TOOL_FOUND_IN_PATH 	 = 2,	// the tool has been found using the PATH environment variable
	EXT_TOOL_FOUND_IN_EXTERN = 4, 	// the tool has been found in EXTERNAL_TOOLS_SUBDIRECTORY
	EXT_TOOL_FOUND_IN_INTERN = 8, 	// the tool has been found in INTERNAL_TOOLS_SUBDIRECTORY
	EXT_TOOL_FOUND_IN_LOC 	 = 16,	// the tool has been specified with a location and was there
	EXT_TOOL_HAS_EXEC_RIGHTS = 32,	// the tool has no execution rights and they could not be granted by the process
} ExtToolStatus;

extern std::string g_externalToolItem_errors[];

class ExternalToolItem
{
private:
	std::string m_fullName;

public:	
	ExtToolStatus m_status;
	std::string m_shortName;

	ExternalToolItem( ExtToolStatus i_status=EXT_TOOL_UNDEF, const std::string &i_shortName="", const std::string &i_fullName="" );

	bool isCallable() const;
	inline string errorMessage() const;

	const std::string &callName() const;
};

// This class is an handler for external tools
// It can check the presence of a specified tool from its short name,
// It looks in :
//		- the PATH environment variable
//		- the Micmac-specific internal tool subdirectory
//		- the Micmac-specific external tool subdirectory
class ExternalToolHandler
{
public:
	ExternalToolHandler();

	// returns an ExternalToolItem for i_tool command
	const ExternalToolItem & get( const std::string &i_tool );
private:
	list<std::string> m_pathDirectories; // the list of directories in PATH environment variable
	map<std::string, ExternalToolItem> m_queriedTools; // all tools previously queried
	list<std::string> m_excludedDirectories; // directories excluded from the search

	// initialize m_pathDirectories
	void initPathDirectories();

	// return true if a the tool has been found in one of PATH's directories
	// if so io_exeTool is set to this full name otherwise it is left untouched
	bool checkPathDirectories( std::string &io_exeTool );

	// look for possible paths for this tool and store it in the map for later use
	ExternalToolItem &addTool( const std::string &i_tool );

	// exclude i_directory during the search for external tools
	void exclude( const std::string &i_directory );
};

#if (ELISE_POSIX)
	// functions for rights checking/setting (unices only)
	
	// process has execution rights on i_filename ?
	bool hasExecutionRights( const std::string &i_filename );
	
	// set execution rigths for owner, group's members and others on i_filename
	// equivalent of chmod +x i_filename
	// return true if successfull
	bool setAllExecutionRights( const std::string &i_filename, bool i_value );
	
	// process' owner owns i_filename ?
	bool isOwnerOf( std::string i_filename );
		
	// process' owner belongs to the group of i_filename ?
	bool belongsToGroupOf( const std::string &i_filename );
	
	// is process executed with super-user rights ?
	inline bool isProcessRoot(){ return geteuid()==0; }
#endif

extern ExternalToolHandler g_externalToolHandler;
extern const std::string   TheStrSiftPP;
extern const std::string   TheStrAnnPP;

std::string MMAuxilaryBinariesDirectory();

// inline methods

// ExternalToolItem

inline ExternalToolItem::ExternalToolItem( ExtToolStatus i_status,
										   const std::string &i_shortName,
										   const std::string &i_fullName ):
	m_fullName( i_fullName ), m_status( i_status ), m_shortName( i_shortName ){}

inline bool ExternalToolItem::isCallable() const{ 
	return ( m_status!=EXT_TOOL_UNDEF && m_status!=EXT_TOOL_NOT_FOUND && (m_status&EXT_TOOL_HAS_EXEC_RIGHTS)!=0 );
}

inline string ExternalToolItem::errorMessage() const{ 
	if ( m_status==EXT_TOOL_NOT_FOUND ) return g_externalToolItem_errors[0];
	if ( (m_status&EXT_TOOL_HAS_EXEC_RIGHTS)!=0 ) return g_externalToolItem_errors[1];
	return string();
}

// ExternalToolHandler

inline const ExternalToolItem & ExternalToolHandler::get( const string &i_tool )
{
	map<string,ExternalToolItem>::iterator itTool = m_queriedTools.find( i_tool );

	if ( itTool==m_queriedTools.end() )
		return addTool( i_tool );
	
	return itTool->second;
}

// exclude i_directory during the search for external tools
inline void ExternalToolHandler::exclude( const std::string &i_directory )
{
	m_excludedDirectories.push_back( filename_normalize(i_directory) );
}
