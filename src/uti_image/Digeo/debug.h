#include <cstdlib>

#if ELISE_unix || defined __UNIX
	#define TTY_BOLD_RED "\033[1;31m"
	#define TTY_DEFAULT "\033[0m"
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
	#define __elise_debug_error( expr, msg ){\
		if ( expr ){\
			std::cerr << ELISE_RED_DEBUG_ERROR << msg << std::endl;\
			exit(EXIT_FAILURE);\
		}\
	}
	#define __elise_debug_warning( expr, msg ){\
		if ( expr ) std::cerr << ELISE_RED_DEBUG_WARNING << msg << std::endl;\
	}
#else
	#define __elise_debug_error( expr, msg )
	#define __elise_debug_warning( expr, msg )
#endif

#define __elise_error( msg ){\
	std::cerr << ELISE_RED_ERROR << msg << std::endl;\
	exit(EXIT_FAILURE);\
}

#define __elise_warning( msg ){\
	std::cerr << ELISE_RED_WARNING << msg << std::endl;\
}

#define __DEBUG_TIMES
#define __DEBUG_MULTI_CHANNEL
#define __DEBUG_DIGEO_CONVOLUTIONS
#define __DEBUG_CONVOLUTION_KERNEL_1D
