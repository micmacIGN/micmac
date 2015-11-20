#ifndef __DISABLE_MSVC_WARNINGS__
#define __DISABLE_MSVC_WARNINGS__

#if _MSC_VER
	#pragma warning(disable: 4723)
	#pragma warning(disable: 4055)
	#pragma warning(disable: 4054)
	#pragma warning(disable: 4018) // signed/unsig
	#pragma warning(disable: 4244) // possible los
	#pragma warning(disable: 4305) // truncation d
	#pragma warning(disable: 4146) // unary minus
	#pragma warning(disable: 4661)
	#pragma warning(disable: 4100)
	#pragma warning(disable: 4996) // strcpy, sscanf are considered unsafe
#endif

#endif
