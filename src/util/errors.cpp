#include "general/sys_dep.h"
#include "general/errors.h"

using namespace std;

#ifdef __DEBUG
	OStreamMessageHandler gDebugErrorMessageHandler( cerr, ELISE_RED_DEBUG_ERROR, MessageHandler::EXIT, EXIT_FAILURE );
	MessageHandler *gDefaultDebugErrorHandler = &gDebugErrorMessageHandler;

	OStreamMessageHandler gDebugWarningMessageHandler( cerr, ELISE_RED_DEBUG_WARNING, MessageHandler::NOTHING, EXIT_FAILURE );
	MessageHandler *gDefaultDebugWarningHandler = &gDebugWarningMessageHandler;
#endif
