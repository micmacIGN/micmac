#ifndef HASH_INCLUDED
#define HASH_INCLUDED

    #if (defined ELISE_Darwin) && (defined __clang__) && (__clang_major__>=5)
        #include <unordered_map>
        #define hash_map std::unordered_map
    #elif (ELISE_windows)
        #ifndef __GNUC__
            #include <hash_map>
             using stdext::hash_map;
        #else
            #if __GNUC__ < 3
                    #include <hash_map>
            #else
                    #include <ext/hash_map>
                    #if __GNUC_MINOR__ == 0
                            using namespace std;       // GCC 3.0
                    #else
                            using namespace __gnu_cxx; // GCC >= 3.1
                    #endif
            #endif
        #endif
    #else // !WIN32
        #include <tr1/unordered_map>
        #define hash_map std::tr1::unordered_map
    #endif // WIN32
#endif // HASH_INCLUDED


