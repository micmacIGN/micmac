#include "StdAfx.h"

// print cmd and execute ::system (helps with debugging)
int trace_system( const char *cmd )
{
	cout << " system call to [" << cmd << ']' << endl;
	return ::system( cmd );
}

#ifdef __TRACE_SYSTEM__
	int (*system_call)( const char*cmd )=trace_system;
#else
	int (*system_call)( const char*cmd )=::system;
#endif