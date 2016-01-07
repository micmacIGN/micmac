#ifndef __ELISE_ERRORS__
#define __ELISE_ERRORS__

#include "MessageHandler.h"

#include "general/sys_dep.h"

#if ELISE_unix || defined UNIX
	#define ELISE_TTY_BOLD_RED "\033[1;31m"
	#define ELISE_TTY_DEFAULT "\033[0m"
	#define ELISE_RED_ERROR "\033[1;31mERROR: \033[0m"
	#define ELISE_RED_WARNING "\033[1;31mWARNING: \033[0m"
	#define ELISE_RED_DEBUG_ERROR "\033[1;31mDEBUG_ERROR: \033[0m"
	#define ELISE_RED_DEBUG_WARNING "\033[1;31mDEBUG_WARNING: \033[0m"
#else
	#define ELISE_RED_ERROR "ERROR: "
	#define ELISE_RED_WARNING "WARNING: "
	#define ELISE_RED_DEBUG_ERROR "DEBUG_ERROR: "
	#define ELISE_RED_DEBUG_WARNING "DEBUG_WARNING: "
#endif

#ifdef __DEBUG
	extern MessageHandler *gDefaultDebugErrorHandler;
	#define ELISE_DEBUG_ERROR(condition,where,what)\
		if ( condition ){\
			std::stringstream ssWhat; ssWhat << what;\
			std::stringstream ssWhere; ssWhere << where;\
			gDefaultDebugErrorHandler->add( DebugErrorMessage(std::string(__FILE__),__LINE__,ssWhere.str(),ssWhat.str()) );\
		}

	extern MessageHandler *gDefaultDebugWarningHandler;
	#define ELISE_DEBUG_WARNING(condition,where,what)\
		if ( condition ){\
			std::stringstream ssWhat; ssWhat << what;\
			std::stringstream ssWhere; ssWhere << where;\
			gDefaultDebugWarningHandler->add( DebugErrorMessage(std::string(__FILE__),__LINE__,ssWhere.str(),ssWhat.str()) );\
		}

	#define __OUT(msg) std::cout << msg << std::endl;
#else
	#define ELISE_DEBUG_ERROR(condition,where,what)
	#define ELISE_DEBUG_WARNING(condition,where,what)

	#define __OUT(msg)
#endif

#define ELISE_ERROR_EXIT( msg ){\
	std::cerr << ELISE_RED_ERROR << msg << std::endl;\
	exit(EXIT_FAILURE);\
}

#define ELISE_ERROR_RETURN( msg ){\
	std::cerr << ELISE_RED_ERROR << msg << std::endl;\
	return EXIT_FAILURE;\
}

#define ELISE_WARNING( msg ){\
	std::cerr << ELISE_RED_WARNING << msg << std::endl;\
}

#endif
