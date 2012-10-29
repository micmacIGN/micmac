#define EXTERNAL_TOOLS_SUBDIRECTORY "binaire-aux"

typedef enum {
	EXT_TOOL_UNDEF = 0,
	EXT_TOOL_NOT_FOUND = 1,
	EXT_TOOL_FOUND_IN_PATH = 2, // the tool has been found using the PATH environment variable
	EXT_TOOL_FOUND_IN_DIR = 4,  // the tool has been found in EXTERNAL_TOOLS_SUBDIRECTORY
} ExtToolStatus;

class ExternalToolItem
{
public:
	ExtToolStatus m_status;
	std::string m_shortName;
	std::string m_fullName;

	ExternalToolItem( ExtToolStatus i_status=EXT_TOOL_UNDEF, const std::string i_shortName="", const std::string i_fullName="" );

	// returns the shortest callable name (m_shortName if possible, m_fullName if not)
	const std::string callName() const;
};

// This class is an handler for external tools
// It can check the presence of a specified tool from its short name,
// It looks in :
//		- the PATH environment variable
//		- the Micmac-specific external tool subdirectory
class ExternalToolHandler
{
public:
	ExternalToolHandler();

	// returns a callable name for i_tool commande
	std::string getCallName( const std::string &i_tool );
private:
	list<std::string> m_pathDirectories; // the list of directories in PATH environment variable
	map<std::string, ExternalToolItem> m_queriedTools; // all tools previously queried

	// initialize m_pathDirectories
	void initPathDirectories();

	// return true if a the tool has been found in one of PATH's (the environment variable) directories
	// if so io_exeTool is set to this full name otherwise it is left untouched
	bool checkPathDirectories( std::string &io_exeTool );
};

extern ExternalToolHandler g_externalToolHandler;


// inline methods

// ExternalToolItem

inline ExternalToolItem::ExternalToolItem( ExtToolStatus i_status,
										   const std::string i_shortName,
										   const std::string i_fullName ):
	m_status( i_status ), m_shortName( i_shortName ), m_fullName( i_fullName ){}
	
inline const std::string ExternalToolItem::callName() const
{
	if ( ( m_status&EXT_TOOL_FOUND_IN_PATH)!=0 ) return m_shortName;
	if ( ( m_status&EXT_TOOL_FOUND_IN_DIR)!=0 ) return m_fullName;
	return "";
}

// ExternalToolHandler

inline ExternalToolHandler::ExternalToolHandler(){ initPathDirectories(); }
