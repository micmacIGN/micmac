#include "StdAfx.h"

ExternalToolHandler g_externalToolHandler;

using namespace std;

// ExternalToolHandler

void ExternalToolHandler::initPathDirectories()
{
	m_pathDirectories.clear();

	char *itPath = getenv( "PATH" );

	if ( itPath==NULL )
	{
		#if ( __VERBOSE__>1 )
				cerr << "WARNING: PATH environment variable could not be found" << endl;
		#endif
		return;
	}
	
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
}

bool ExternalToolHandler::checkPathDirectories( string &io_exeName )
{
	list<string>::iterator itDir = m_pathDirectories.begin();
	string fullName;
	// walk through the directory list, looking for an io_exeName file
	while ( itDir!=m_pathDirectories.end() )
	{
		fullName = (*itDir)+io_exeName;
		if ( ELISE_fp::exist_file( fullName ) )
		{
			io_exeName = fullName;
			return true;
		}
		itDir++;
	}
	return false;
}

string ExternalToolHandler::getCallName( const string &i_tool )
{
	map<string,ExternalToolItem>::iterator itTool = m_queriedTools.find( i_tool );

	if ( itTool==m_queriedTools.end() )
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
		ExternalToolItem item = m_queriedTools[i_tool] = ExternalToolItem( status, i_tool, fullName );
		return item.callName();
	}
	else
		return itTool->second.callName();
}

int CheckDependencies_main(int argc,char ** argv)
{
	cout << "make: " << g_externalToolHandler.getCallName( "make" ) << endl;
	cout << "exiftool: " << g_externalToolHandler.getCallName( "exiftool" ) << endl;
	cout << "exiv2: " << g_externalToolHandler.getCallName( "exiv2" ) << endl;
	cout << "convert: " << g_externalToolHandler.getCallName( "convert" ) << endl;
	cout << "ann_samplekeyfiltre: " << g_externalToolHandler.getCallName( "ann_samplekeyfiltre" ) << endl;
	cout << "siftpp_tgi: " << g_externalToolHandler.getCallName( "siftpp_tgi" ) << endl;

	return EXIT_SUCCESS;
}
