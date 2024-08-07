 /** 
    @file  tpp_trace.hpp    
    @brief Macros for writing traces to a file. 
           Useful when debugging a GUI app. on Windows without console!
 */

#ifndef TRPP_TRACE_TO_FILE
#define TRPP_TRACE_TO_FILE

 
#ifdef TRIANGLE_DBG_TO_FILE
#   include <cstdio>
#   include <string>

namespace tpp 
{
   extern FILE* g_debugFile;
   extern std::string g_debugFileName;
}

    // TR string
#   define TRACE(a) { if(tpp::g_debugFile) { fprintf(tpp::g_debugFile, "%s\n", a); fflush(tpp::g_debugFile); } }
    // TR string + integer 
#   define TRACE2i(a,b) { if(tpp::g_debugFile) { fprintf(tpp::g_debugFile, "%s%d\n", a, b); fflush(tpp::g_debugFile); } }
    // TR string + string 
#   define TRACE2s(a,b) { if(tpp::g_debugFile) { fprintf(tpp::g_debugFile, "%s%s\n", a, b); fflush(tpp::g_debugFile); } }
    // TR string + boolean 
#   define TRACE2b(a,b) { if(tpp::g_debugFile) { fprintf(tpp::g_debugFile, "%s%s\n", a, b ? "true " : "false"); fflush(tpp::g_debugFile); } }

#   define INIT_TRACE(a) { if (!tpp::g_debugFile) {\
                             tpp::g_debugFile = fopen(a, "w");\
                             if(!tpp::g_debugFile) std::cerr << "ERROR: Cannot open trace file: " << a << std::endl;\
                             else tpp::g_debugFileName = a; } }

#   define END_TRACE(a) { if(tpp::g_debugFile && g_debugFileName == a) {\
                               fclose(tpp::g_debugFile); \
                               tpp::g_debugFile = nullptr; tpp::g_debugFileName = ""; } }
#else
#   define TRACE(a)
#   define TRACE2i(a,b) 
#   define TRACE2s(a,b) 
#   define TRACE2b(a,b) 
#   define INIT_TRACE(a) 
#   define END_TRACE(a) 
#endif // TRIANGLE_DBG_TO_FILE

#endif
