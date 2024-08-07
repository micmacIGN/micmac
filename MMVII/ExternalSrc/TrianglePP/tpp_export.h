/**
   @file  tpp_export.hpp
   @brief Helper macros for DLL builds
*/

#pragma once

#if defined(_MSC_VER)
#  define TRPP_DECL_EXPORT __declspec(dllexport)
#  define TRPP_DECL_IMPORT __declspec(dllimport)
#else
#  define TRPP_DECL_EXPORT
#  define TRPP_DECL_IMPORT
#endif

#ifdef TRPP_BUILD_SHARED
# if defined(TRPP_TRIANGLE_LIB)
#  define TRPP_LIB_EXPORT TRPP_DECL_EXPORT
# else
#  define TRPP_LIB_EXPORT TRPP_DECL_IMPORT
# endif
#else
# define TRPP_LIB_EXPORT
#endif
