#include "StdAfx.h"


ExternalToolHandler g_externalToolHandler;

using namespace std;

string g_externalToolItem_errors[] = { "cannot be found",
                                       "does not have execution rights (and process cannot grant them)" };

// ExternalToolItem

const std::string &ExternalToolItem::callName() const
{
	if ( !isCallable()) ELISE_ERROR_EXIT("tool [" << m_shortName << "] has not been found");
	return m_fullName;
}

// ExternalToolHandler

ExternalToolHandler::ExternalToolHandler()
{
#if (ELISE_windows)
    const char *windir = getenv( "windir" );
    if ( windir!=NULL )
        exclude( windir );
    else
    {
        windir = getenv( "SystemRoot" );
        if ( windir!=NULL )
            exclude( windir );
        #if (__VERBOSE__>1)
        else
            cout << "WARNING: windir environment variable cannot be found" << endl;
        #endif
    }
#endif
    initPathDirectories();
}

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
    char *pathCopy = new char[strlen( envPath )+1];
    strcpy( pathCopy, envPath );

    char *itPath = pathCopy;

    // split PATH value in a list of strings
    char *itNextDirectory = itPath;
    string directory;
    bool ok = true;

    do
    {
        if ( ( *itPath )=='\0' ){ *itPath=ELISE_CAR_ENV; ok=false; }
        if ( ( *itPath )==ELISE_CAR_ENV ) // ELISE_CAR_ENV is the separator for directories in PATH, it is system dependent
        {
            *itPath=0;
            // we got a possible new entry
            directory = (string)itNextDirectory;
            if ( directory.size()!=0 )
            {
                filename_normalize( directory );

                // do not add directories in the exclude list (including their subdirectories)
                bool addDirectory = true;
                list<string>::iterator itExcluded = m_excludedDirectories.begin();
                while ( itExcluded!=m_excludedDirectories.end() )
                {
                    if ( startWith( directory, *itExcluded++ ) )
                    {
                        addDirectory = false;
                        break;
                    }
                }

                if ( addDirectory )
                {
                    // make sure the path ends with a slash
                    if ( ( *directory.rbegin() )!='/' ) directory.append( "/" );

                    // finally, we can add it to the list
                    m_pathDirectories.push_back( directory );
                }
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

// check if i_filename has execution rigths and if not, try to grant them (process owner must own the file)
// if all this fails returns false
bool checkExecRights( const string &i_filename )
{
    #if (ELISE_POSIX)
        if ( !hasExecutionRights( i_filename ) )
        {
            cerr << "WARNING: process does not have the right to execute " << i_filename << "" << endl;
            cerr << "WARNING: trying to grant execution rights on " << i_filename << " ..." << endl;
            if ( setAllExecutionRights( i_filename, true ) )
                cerr << "WARNING: execution rights have been successfully granted on " << i_filename << endl;
            else
            {
                cerr << "WARNING: unable to grant execution rights on " << i_filename << ", try :" << endl;
                cerr << "WARNING: sudo chmod +x "+ i_filename << endl;
                cerr << "WARNING: to solve this problem" << endl;
                return false;
            }
        }
    #endif
    return true;
}

ExternalToolItem & ExternalToolHandler::addTool( const std::string &i_tool )
{
    // this tool has not been queried before, we need to check
    string exeName = i_tool,
                     fullName,
                     testName;
    ExtToolStatus status = EXT_TOOL_UNDEF;

	#if (ELISE_windows)
		if (tolower(getShortestExtension(i_tool)) != "exe") exeName.append(".exe");
	#endif

    // is there's a path in the name we don't look in other directories
    size_t pos = i_tool.find_last_of( "/\\" );
    if ( pos!=string::npos ){
        if ( ELISE_fp::exist_file( i_tool ) ){
            status = EXT_TOOL_FOUND_IN_LOC; // found in the specified location
            fullName = i_tool;
            if ( checkExecRights( fullName ) ) status=(ExtToolStatus)( status|EXT_TOOL_HAS_EXEC_RIGHTS );
            return ( m_queriedTools[i_tool]=ExternalToolItem( status, i_tool, i_tool ) );
        }
        else
            status = EXT_TOOL_UNDEF;
    }

	#if 0
		// check EXTERNAL_TOOLS_SUBDIRECTORY directory
		testName = MMDir()+EXTERNAL_TOOLS_SUBDIRECTORY+ELISE_CAR_DIR+exeName;
		if ( ELISE_fp::exist_file( testName ) ){
			status = ( ExtToolStatus )( status|EXT_TOOL_FOUND_IN_EXTERN );
			fullName = testName;
		}
	#else
		list<cElFilename> filenames;
		list<ctPath> subdirectories;
		ctPath(MMAuxilaryBinariesDirectory()).getContent(filenames, subdirectories, true); // true = aIsRecursive
		list<cElFilename>::const_iterator itFilename = filenames.begin();
		while (itFilename != filenames.end())
		{
			if (itFilename->m_basename == exeName)
			{
				status = (ExtToolStatus)(status | EXT_TOOL_FOUND_IN_EXTERN);
				fullName = itFilename->str();
			}
			itFilename++;
		}
	#endif

    // check INTERNAL_TOOLS_SUBDIRECTORY directory
    // INTERNAL_TOOLS_SUBDIRECTORY prevails upon EXTERNAL_TOOLS_SUBDIRECTORY
    testName = MMDir()+INTERNAL_TOOLS_SUBDIRECTORY+ELISE_CAR_DIR+exeName;
    if ( ELISE_fp::exist_file( testName ) ){
        status = ( ExtToolStatus )( status|EXT_TOOL_FOUND_IN_INTERN );
        fullName = testName;
    }

	#if !ELISE_windows
		// PATH directories prevails upon INTERNAL_TOOLS_SUBDIRECTORY and EXTERNAL_TOOLS_SUBDIRECTORY
		// except for excluded directories (in m_excludedDirectories) which are ignored

		// we do not use PATH under windows because all dependencies are provided in binaire-aux/windows

		if (checkPathDirectories(exeName))
		{
			status = (ExtToolStatus)(status | EXT_TOOL_FOUND_IN_PATH);
			fullName = exeName;
		}
	#endif

    if (status == EXT_TOOL_UNDEF)
    {
        // check old binaire-aux
        testName = MMDir() + EXTERNAL_TOOLS_SUBDIRECTORY + ELISE_CAR_DIR + exeName;
        __OUT("testName = [" << testName << ']');
        if (ELISE_fp::exist_file(testName))
        {
            status = (ExtToolStatus)(status | EXT_TOOL_FOUND_IN_EXTERN);
            fullName = testName;
       }
    }

    // we searched and found nothing
    if ( status==EXT_TOOL_UNDEF )
        status=EXT_TOOL_NOT_FOUND;
    else
        if ( checkExecRights( fullName ) ) status=(ExtToolStatus)( status|EXT_TOOL_HAS_EXEC_RIGHTS );

    // create the entry
    return ( m_queriedTools[i_tool]=ExternalToolItem( status, i_tool, fullName ) );
}

string MMAuxilaryBinariesDirectory()
{
	return MMDir() + EXTERNAL_TOOLS_SUBDIRECTORY + "/" + BIN_AUX_SUBDIR + "/";
}

#if (ELISE_POSIX)
    // functions for rights checking/setting (unices only)

    // process has execution rights on i_filename ?
    bool hasExecutionRights( const std::string &i_filename )
    {
        struct stat s;
        stat( i_filename.c_str(), &s );

        bool ownerCanExecute = s.st_mode&S_IXUSR,
             groupCanExecute = s.st_mode&S_IXGRP,
             otherCanExecute = s.st_mode&S_IXOTH;

        if ( isProcessRoot() && ( ownerCanExecute || groupCanExecute || otherCanExecute ) ) return true;

        bool isOwner = isOwnerOf( i_filename );
        if ( ownerCanExecute && isOwner ) return true;

        bool belongsToGroup = belongsToGroupOf( i_filename );
        if ( groupCanExecute && belongsToGroup ) return true;

        if ( otherCanExecute && !isOwner && !belongsToGroup ) return true;

        return false;
    }

    // set execution rigths for owner, group's members and others on i_filename
    // equivalent of chmod +x i_filename
    // return true if successfull
    bool setAllExecutionRights( const std::string &i_filename, bool i_value )
    {
        struct stat s;
        stat( i_filename.c_str(), &s );
        return chmod( i_filename.c_str(), s.st_mode|S_IXUSR|S_IXGRP|S_IXOTH )==0;
    }

    // process' owner owns i_filename ?
    bool isOwnerOf( std::string i_filename )
    {
        struct stat s;
        stat( i_filename.c_str(), &s );
        return s.st_uid==geteuid();
    }

    // process' owner belongs to the group of i_filename ?
    bool belongsToGroupOf( const std::string &i_filename )
    {
        struct stat file_stat;
        struct group *file_group;
        struct passwd *user;

        stat( i_filename.c_str(), &file_stat );

        // if it's impossible to retrieve file's group, there's no use in going further
        if ( ( file_group=getgrgid( file_stat.st_gid ) )==NULL ) return false;

        // compare file's group with process' effective user's primary group
        if ( ( ( user=getpwuid( geteuid() ) )!=NULL ) && ( user->pw_gid==file_stat.st_gid ) ) return true;

        // compare file's group with all other groups the user belongs to
        int i=0;
        while ( file_group->gr_mem[i] )
            if ( strcmp( file_group->gr_mem[i++], user->pw_name )==0 ) return true;

        return false;
    }
#endif
