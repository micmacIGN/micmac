#include <cstdlib>

#ifndef __DIGEO_DEBUG_H
#define __DIGEO_DEBUG_H

#ifdef NO_ELISE
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
		#define ELISE_DEBUG_ERROR( expr, where, what ){\
			if ( expr ){\
				std::cerr << ELISE_RED_DEBUG_ERROR << where << ": " << what << std::endl;\
				exit(EXIT_FAILURE);\
			}\
		}
		#define ELISE_DEBUG_WARNING( expr, where, what ){\
			if ( expr ) std::cerr << ELISE_RED_DEBUG_WARNING << where << ": " << what << std::endl;\
		}
	#else
		#define ELISE_DEBUG_ERROR( expr, where, what )
		#define ELISE_DEBUG_WARNING( expr, where, what )
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

#ifdef __DEBUG
	//~ #define __DEBUG_TIMES
	//~ #define __DEBUG_MULTI_CHANNEL
	//~ #define __DEBUG_DIGEO_CONVOLUTIONS
	//~ #define __DEBUG_CONVOLUTION_KERNEL_1D
#endif

#endif
