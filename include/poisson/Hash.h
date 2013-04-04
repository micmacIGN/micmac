#ifndef HASH_INCLUDED
	#define HASH_INCLUDED

	#ifdef NOWARNINGPOISSON
		#pragma GCC diagnostic push
		#pragma GCC diagnostic warning "-w"
	#endif

	#ifdef WIN32
		#include <hash_map>
		using stdext::hash_map;
	#else // !WIN32
		/*
		// #include <unordered_map>  // + MPD
		//#include <hash_multimap>  // + MPD
		#include <ext/hash_map>  // MODIF warning deprecated header => A Voir

		using namespace __gnu_cxx;

		namespace __gnu_cxx
		{
		  template<> struct hash<long long> {
			size_t operator()(long long __x) const { return __x; }
		  };
		  template<> struct hash<const long long> {
			size_t operator()(const long long __x) const { return __x; }
		  };
		  
		  
		  template<> struct hash<unsigned long long> {
			size_t operator()(unsigned long long __x) const { return __x; }
		  };
		  template<> struct hash<const unsigned long long> {
			size_t operator()(const unsigned long long __x) const { return __x; }
		  };
		}
		*/	
		#include <tr1/unordered_map>
		#define hash_map std::tr1::unordered_map
	#endif // WIN32
#endif // HASH_INCLUDED

#ifdef NOWARNINGPOISSON
	#pragma GCC diagnostic pop
#endif
