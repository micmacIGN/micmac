#######################################################
##        Global configuration
#######################################################
cmake_minimum_required(VERSION 3.15)
project(MMVII VERSION 1.0)

# strict C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use Position Independent Code for python binding
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# output MMVII binary in  (....)/micmac/MMVII/bin
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/bin>)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/bin>)

get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
# build type. Default: optimized with debug info
if(NOT is_multi_config)
  set(default_build_type "RelWithDebInfo")
  if(NOT CMAKE_BUILD_TYPE)
    message(NOTICE "-- Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
         STRING "Choose the type of build, options are: Release RelWithDebInfo Debug" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release RelWithDebInfo Debug)
  endif()
else()
	set(CMAKE_CONFIGURATION_TYPES "Release;RelWithDebInfo;Debug" CACHE STRING "" FORCE)
endif()

#######################################################
##        Dependencies
#######################################################

# Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)

# OpenMP: Optional, included in GNU C++, separate package for CLANG.
find_package(OpenMP)
if((NOT OpenMP_CXX_FOUND) AND (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
   message(STATUS "OpenMP not found: did you install libomp-dev ?")
endif()


#######################################################
##        Compilation toolchain
#######################################################

# ccache
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif(CCACHE_FOUND)

# compilers specific options : warnings, MSVC specific defines
set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
set(msvc_cxx "$<COMPILE_LANG_AND_ID:CXX,MSVC>")
set(gnu_cxx  "$<COMPILE_LANG_AND_ID:CXX,GNU>")

add_library(MMVII_compiler_flags INTERFACE)
target_compile_options(MMVII_compiler_flags INTERFACE
    "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall;-Werror>>"
    "$<${msvc_cxx}:$<BUILD_INTERFACE:/W3;/D_USE_MATH_DEFINES;/D_HAS_STD_BYTE=0;/wd4244;/wd4267;/wd4305;/wd4661;/D_CRT_SECURE_NO_WARNINGS>>"
    "$<$<CONFIG:Debug>:$<$<CXX_COMPILER_ID:MSVC>:/bigobj>>"
    "$<$<CONFIG:RelWithDebInfo>:$<$<CXX_COMPILER_ID:MSVC>:/bigobj>>"
)

# we really want -O3 instead of -O2 when available (GNU, Clang) in RelWithDebInfo mode
string(REPLACE "-O2" "-O3" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

# Keep assert() in  RelDebWithInfo
string(REGEX REPLACE "[-/]D *NDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")


#######################################################
##        Sources directories structure
#######################################################

# micmac v1 library and headers
set(mmv1_dir "${MMVII_SOURCE_DIR}/..")
set(mmv1_include_dir "${mmv1_dir}/include")
set(mmv1_lib_dir "${mmv1_dir}/lib")
find_library(elise_lib NAME elise PATHS "${mmv1_lib_dir}" NO_DEFAULT_PATH)
find_library(ANN_lib NAME ANN PATHS "${mmv1_lib_dir}" NO_DEFAULT_PATH)

# MMVII dir
set(mmv2_include_dir "${MMVII_SOURCE_DIR}/include")
set(mmv2_external_include_dir "${MMVII_SOURCE_DIR}/ExternalInclude")
set(mmv2_src_dir "${MMVII_SOURCE_DIR}/src")
set(mmv2_gencode_dir "${mmv2_src_dir}/GeneratedCodes")
set(mmv2_gencode_main_src "${mmv2_gencode_dir}/cName2CalcRegisterAll.cpp")
set(EIGEN3_INCLUDE_PATH  "${mmv2_external_include_dir}/eigen-3.4.0")

# MMVII sources: Auto generate. Not recommended, but let's try it ...
file(GLOB mmv2_libsrcs CONFIGURE_DEPENDS "${mmv2_src_dir}/*/*.cpp")
file(GLOB mmv2_headers "${mmv2_include_dir}/*.h" "${mmv2_include_dir}/*.h" "${mmv2_src_dir}/*/*.h")

# Here we initialize the generated code with en empty template if working dir is clean (first checkout or after make distclean)
list(APPEND mmv2_libsrcs "${mmv2_gencode_main_src}")   # This file may not be present (deleted after clean all), force dependance on it
list(REMOVE_DUPLICATES mmv2_libsrcs)				   # Avoid the file above being build twice
add_custom_command(OUTPUT "${mmv2_gencode_main_src}" COMMAND ${CMAKE_COMMAND} -E copy ${mmv2_include_dir}/SymbDer/cName2CalcRegisterAll.cpp.tmpl "${mmv2_gencode_main_src}" VERBATIM)


#######################################################
##        libP2007 buid (MMVII library)
#######################################################
add_library(P2007 "${mmv2_libsrcs}" "${mmv2_headers}")
target_include_directories(P2007 PRIVATE "${mmv2_include_dir};${mmv1_include_dir};${EIGEN3_INCLUDE_PATH}")
target_link_libraries(P2007 PRIVATE MMVII_compiler_flags Threads::Threads)


#######################################################
##        MMVII build (executable)
#######################################################
add_executable(MMVII src/main.cpp)

set(mmv1_libs "${elise_lib};${ANN_lib}")
if(UNIX)
	set(mmv2_extlibs -lX11)
else()	
	set(mmv2_extlibs)
endif()

target_include_directories(MMVII PRIVATE "${MMVII_SOURCE_DIR}/include"  )

target_link_libraries(MMVII PRIVATE MMVII_compiler_flags Threads::Threads P2007 ${mmv1_libs} ${mmv2_extlibs})
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION  VERSION_LESS "9.1")
  target_link_libraries(MMVII PRIVATE -lstdc++fs)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION  VERSION_LESS "9.0")
  target_link_libraries(MMVII PRIVATE -lc++fs)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(MMVII PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(P2007 PRIVATE OpenMP::OpenMP_CXX)
endif()

#######################################################
##        Compile vMMVII if Qt is found
#######################################################

option(vMMVII_BUILD "Build vMMVII application (require Qt5)" ON)

if (vMMVII_BUILD)
  find_package(QT NAMES Qt5 Qt6 QUIET OPTIONAL_COMPONENTS Widgets)
  find_package(Qt${QT_VERSION_MAJOR} QUIET OPTIONAL_COMPONENTS Widgets)

  if(Qt${QT_VERSION_MAJOR}Widgets_FOUND)
    if (Qt${QT_VERSION_MAJOR}Widgets_VERSION VERSION_LESS 5.12.8)
      message(STATUS "QT version too old (<5.12.8), vMMVII will not be compiled")
    else()
      message(STATUS "Qt${QT_VERSION_MAJOR} found. Will compile vMMVII")
      add_subdirectory(vMMVII)
    endif()
  else()
    message(WARNING "Qt not found. vMMVII will not be compiled")
  endif()
else()
  message(STATUS "vMMVII will not be compiled (config option)")
endif()

#######################################################
##        target distclean : remove generated code
#######################################################
add_custom_target(distclean
  ${CMAKE_COMMAND} --build . --target clean
  COMMAND   ${CMAKE_COMMAND} -E rm -rf ${mmv2_gencode_dir}
  VERBATIM)
  
