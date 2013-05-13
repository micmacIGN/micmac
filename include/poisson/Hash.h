#ifndef HASH_INCLUDED
	#define HASH_INCLUDED

	#if (ELISE_windows)&&(!ELISE_MinGW)
		#include <hash_map>
		using stdext::hash_map;
	#else // !WIN32
		#include <tr1/unordered_map>
		#define hash_map std::tr1::unordered_map
	#endif // WIN32
#endif // HASH_INCLUDED
