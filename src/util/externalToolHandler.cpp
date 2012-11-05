#include "StdAfx.h"

ExternalToolHandler g_externalToolHandler;

using namespace std;

// ExternalToolHandler

void ExternalToolHandler::initPathDirectories()
{
	m_pathDirectories.clear();

	const char *envPath = (const char *)getenv( "PATH" );

	if ( envPath==NULL )
	{
		#if ( __VERBOSE__>1 )
				cerr << "WARNING: PATH environment variable could not be found" << endl;
		#endif
		return;
	}
	
	// copy the value for it may be the actual value of PATH used in the whole process
	char *pathCopy = new char[strlen( envPath )];
	strcpy( pathCopy, envPath );
	
	char *itPath = pathCopy;
	
	// split PATH value in a list of strings
	char *itNextDirectory = itPath;
	string directory;
	bool ok = true;

	do
	{
		if ( ( *itPath )=='\0' ){ *itPath==ELISE_CAR_ENV; ok=false; }
		if ( ( *itPath )==ELISE_CAR_ENV ) // ELISE_CAR_ENV is the separator for directories in PATH, it is system dependant
		{
			*itPath=0;
			// we got a possible new entry
			directory = (string)itNextDirectory;
			if ( directory.size()!=0 )
			{
				#if (ELISE_windows)
					replace( directory.begin(), directory.end(), '\\', '/' );
				#endif

				// make sure the path ends with a slash
				if ( ( *directory.rbegin() )!='/' ) directory.append( "/" );
				
				// it seems like a nice clean directory path, we can add it to the list
				m_pathDirectories.push_back( directory );
			}
			// next directory path begin after current character
			itNextDirectory=itPath+1;
		}

		itPath++;
	}
	while ( ok );
	
	delete [] pathCopy;
}

bool ExternalToolHandler::checkPathDirectories( string &io_exeName )
{
	list<string>::iterator itDir = m_pathDirectories.begin();
	string fullName;
	// walk through the directory list, looking for an io_exeName file
	while ( itDir!=m_pathDirectories.end() )
	{
		#if (ELISE_windows)
			if ( ( io_exeName!="convert.exe" ) || ( itDir->find("system32")==string::npos ) ) // an ugly special case but still
		#endif
		{
			fullName = (*itDir)+io_exeName;
			if ( ELISE_fp::exist_file( fullName ) )
			{
				io_exeName = fullName;
				return true;
			}
		}
		itDir++;
	}
	return false;
}

ExternalToolItem & ExternalToolHandler::addTool( const std::string &i_tool )
{
	// this tool has not been queried before, we need to check
	string exeName = i_tool,
			fullName;
	ExtToolStatus status = EXT_TOOL_UNDEF;

	#if (ELISE_windows)
		exeName.append(".exe");
	#endif

	// check EXTERNAL_TOOLS_SUBDIRECTORY directory
	fullName = MMDir()+EXTERNAL_TOOLS_SUBDIRECTORY+ELISE_CAR_DIR+exeName;
	if ( ELISE_fp::exist_file( fullName ) )
		status = EXT_TOOL_FOUND_IN_DIR;

	if ( checkPathDirectories( exeName ) )
	{
		status = ( ExtToolStatus )( status|EXT_TOOL_FOUND_IN_PATH );
		fullName = exeName;
	}

	// we searched and found nothing
	if ( status==EXT_TOOL_UNDEF ) status==EXT_TOOL_NOT_FOUND;

	// create the entry
	return ( m_queriedTools[i_tool]=ExternalToolItem( status, i_tool, fullName ) );
}

string printResult( const string &i_tool )
{
	string printLine = i_tool + ": ";
	ExternalToolItem item = g_externalToolHandler.get( i_tool );

	if ( item.m_status==EXT_TOOL_NOT_FOUND ) return ( printLine+" NOT FOUND" );

	if ( ( item.m_status&EXT_TOOL_FOUND_IN_PATH )!=0 )
	{
		printLine.append( "LOCAL");
		if ( ( item.m_status&EXT_TOOL_FOUND_IN_DIR )!=0 )
			printLine.append( ", DEFAULT -> using LOCAL");
	}
	else if ( ( item.m_status&EXT_TOOL_FOUND_IN_DIR )!=0 )
		printLine.append( "DEFAULT");

	printLine = printLine+" ("+item.m_fullName+")";
	return printLine;
}

int CheckDependencies_main(int argc,char ** argv)
{
	cout << printResult( "make" ) << endl;
	cout << printResult( "exiftool" ) << endl;
	cout << printResult( "exiv2" ) << endl;
	cout << printResult( "convert" ) << endl;

	return EXIT_SUCCESS;
}
