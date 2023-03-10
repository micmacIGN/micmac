cmake_minimum_required(VERSION 2.8.12)
project(MICMAC)

function(print_list list_name)
	message("${list_name}:")
	foreach(e ${${list_name}})
		message("\t${e}")
	endforeach()
endfunction()

function(qmake_query property_name)
	if (NOT ${property_name})
		EXECUTE_PROCESS(
				COMMAND
				${QT_QMAKE_EXECUTABLE} -query ${property_name}
				RESULT_VARIABLE qmake_result
				OUTPUT_VARIABLE ${property_name}
				ERROR_QUIET
				OUTPUT_STRIP_TRAILING_WHITESPACE)
	endif()

	if (${qmake_result})
		message(FATAL_ERROR "failed to retrieve qmake property [${property_name}]")
	endif()

	set(${property_name} ${${property_name}} PARENT_SCOPE)
endfunction()

function(include_dirs_to_flags include_dirs flags)
	foreach(dir ${include_dirs})
		list(APPEND ${flags} "-I${dir}")
	endforeach()
	set(${flags} ${${flags}} PARENT_SCOPE)
endfunction()

if (MSVC)
	function(qt_dll_to_lib qt_bins qt_libs dll_var)
		string(REPLACE "${qt_bins}" "${qt_libs}" ${dll_var} ${${dll_var}})
		string(REPLACE ".dll" ".lib" ${dll_var} ${${dll_var}})
		set(${dll_var} ${${dll_var}} PARENT_SCOPE)
	endfunction()
endif()

if (POLICY CMP0054)
	cmake_policy(SET CMP0054 NEW)
endif()

if (POLICY CMP0043)
	cmake_policy(SET CMP0043 NEW)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	set(CMAKE_CXX_COMPILER_IS_CLANG 1)
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	set(DARWIN 1)
	set(ELISE_Darwin 1)
	set(BIN_AUX_SUBDIR "darwin")
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	set(WINDOWS 1)
	set(BIN_AUX_SUBDIR "windows")
else()
	set(LINUX 1)
	set(BIN_AUX_SUBDIR "linux")
endif()

if(POLICY CMP0020)
	cmake_policy(SET CMP0020 NEW) #windows specific: Automatically link Qt executables to qtmain
endif()

set(VERBOSE 0)

set(MICMAC_VERSION_MAJOR 1)
set(MICMAC_VERSION_MINOR 0)
option(WERROR "Do warning generate error with gcc"  ON)

##############################
## 			OPTIONS			##
##############################

# construire l'interface graphique
option(WITH_INTERFACE  "build graphic interface" OFF)

# Utiliser Kakadu
option(WITH_KAKADU  "Include KAKADU Support" OFF)

# Utiliser Ign orientation
option(WITH_IGN_ORI "Include Ign orientation" OFF)

# Utiliser Ign orientation
option(WITH_IGN_IMAGE "Include Ign image" OFF)

# Utiliser les en-tetes precompilees
option(WITH_HEADER_PRECOMP "En-tetes precompilees" ON)

# Utiliser les en-tetes precompilees
option(BUILD_ONLY_ELISE_MM3D "Projet Elise et MM3D uniquement" OFF)

# construire l'interface graphique
option(WITH_ETALONPOLY  "Build etalonnage polygone" ON)

# construire les binaires Poisson
option(BUILD_POISSON "Build Poisson binaries" ON)

# construire le binaire Rnx2rtkp
option(BUILD_RNX2RTKP "Build Rnx2rtkp binaries" OFF)

# Utiliser Cuda
option(CUDA_ENABLED "Utilisation de cuda" OFF)

option(WITH_OPENCL  "Utilisation d'OpenCL" OFF)

# Multi thread Poisson avec OpenMP
option(WITH_OPEN_MP "use OpenMP" OFF)

# print system calls
option(TRACE_SYSTEM "print system calls" OFF)

# compile low-level test bench
option(BUILD_BENCH "compile low-level test bench" OFF)

# Interfaces de saisie des param�tres
option(WITH_QT5 "compile Qt interfaces" OFF)

# Graphiz library 
option(WITH_GRAPHVIZ "Use graphviz" OFF)

# Compilateur C++11
option(WITH_CPP11 "Compilateur C++11" ON)

# Graphical Interface GIMMI
OPTION(WITH_GIMMI "Graphical Interface GIMMI" OFF)

option(DEPLOY "compile for end-user" OFF)

# Utiliser Doxygen
option(WITH_DOXYGEN "Generation de documentation" OFF )

# Update QT translations
option(UPDATE_TRANSLATIONS "Update source translation translations/*.ts files (WARNING: make clean will delete the source .ts files! Danger!)" OFF)

option(WITH_CCACHE  "Use ccache to speed up recompilation" ON)

# Configure CCache if available
if(WITH_CCACHE)
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)
endif()

# fichier contenant la fonction gerant les en-tetes precompilees
include(precompiled_headers.cmake)

if(TRACE_SYSTEM)
	add_definitions(-D__TRACE_SYSTEM__=${TRACE_SYSTEM})
endif()

if(__DEBUG)
	add_definitions(-D__DEBUG=${__DEBUG})
endif()

if(WITH_OPEN_MP)
	FIND_PACKAGE(OpenMP REQUIRED)
	if(${CUDA_ENABLED})
	    message("using OpenMP for Poisson 3d shell computing and Cuda precompute")
	else()
	    message("using OpenMP for Poisson 3d shell computing")
	endif()
	list(APPEND CMAKE_C_FLAGS ${OpenMP_C_FLAGS})
	list(APPEND CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS})
	set(USE_OPEN_MP 1)
endif()

######################################
## 	Trouver les EXES et Libs	##
######################################
set(BIN_AUX_FULL_PATH ${PROJECT_SOURCE_DIR}/binaire-aux/${BIN_AUX_SUBDIR})

# DEFINITIION : des chemins binaires sous UNIX
set(UnixBinPath
		/bin
		/usr/bin
		/usr/local/bin)

set(DEPLOY 0)
set(PROJECT_SRC_DIR \"${PROJECT_SOURCE_DIR}\") #TODO: why???
set(QT_ENABLED 0) #TODO: not sure we should keep running if WITH_QT5=1 and OpenGL or QT is not found
set(GIMMI_ENABLED 0)

if(${WITH_QT5})
	set(OpenGL_GL_PREFERENCE LEGACY)
	find_package(OpenGL REQUIRED)

	if(OPENGL_FOUND)
		message(STATUS "-----> OpenGL found")

		if(MSVC)
			include("${PROJECT_SOURCE_DIR}/src/saisieQT/FindWindowsSDK.txt")
			# ajoute /lib a tous les repertoires contenant un SDK
			foreach( sdk_dir ${WINDOWSSDK_DIRS} )
				list( APPEND SDK_LIB_DIRS "${sdk_dir}/lib" )
			endforeach()
			set( CMAKE_PREFIX_PATH "${SDK_LIB_DIRS}" )
			message (" WindowsSDKDir : " ${CMAKE_PREFIX_PATH} )
		endif()

		 #add_compile_definitions(QT_NO_DEPRECATED_WARNINGS=1) # to work with both ubuntu 20.04 and 22.04...
		find_package(Qt5Widgets REQUIRED)
		find_package(Qt5Xml REQUIRED)
		find_package(Qt5Concurrent REQUIRED)
		find_package(Qt5OpenGL REQUIRED)

		if(Qt5Widgets_FOUND)
			if(POLICY CMP0045)
				cmake_policy(SET CMP0045 NEW) # target of get_target_property must exists or a FATAL_ERROR is issued
			endif()
			get_target_property(QT_QMAKE_EXECUTABLE Qt5::qmake IMPORTED_LOCATION)
			get_target_property(QT_MOC_EXECUTABLE Qt5::moc IMPORTED_LOCATION)
			get_target_property(QT_UIC_EXECUTABLE Qt5::uic IMPORTED_LOCATION)
			get_target_property(QT_RCC_EXECUTABLE Qt5::rcc IMPORTED_LOCATION)

			qmake_query(QT_VERSION)
			qmake_query(QT_INSTALL_LIBS)
			qmake_query(QT_INSTALL_PLUGINS)

			if(QT_VERSION VERSION_LESS "5.4.0")
				message(STATUS "Qt version < 5.4.0: using shortcuts patch")
				set(__QT_5_SHORTCUT_PATCH 1)
			endif()

			set(QT_CMAKE_LIBRARIES Qt5::Core Qt5::Gui Qt5::Xml Qt5::OpenGL)

			get_target_property(qt_core_lib Qt5::Core LOCATION)
			get_target_property(qt_gui_lib Qt5::Gui LOCATION)
			get_target_property(qt_xml_lib Qt5::Xml LOCATION)
			get_target_property(qt_opengl_lib Qt5::OpenGL LOCATION)
			if(MSVC)
				#~ LOCATION property points to dll files, transform to lib files path
				qmake_query(QT_INSTALL_BINS)
				qt_dll_to_lib("${QT_INSTALL_BINS}" "${QT_INSTALL_LIBS}" qt_opengl_lib)
				qt_dll_to_lib("${QT_INSTALL_BINS}" "${QT_INSTALL_LIBS}" qt_core_lib)
				qt_dll_to_lib("${QT_INSTALL_BINS}" "${QT_INSTALL_LIBS}" qt_gui_lib)
				qt_dll_to_lib("${QT_INSTALL_BINS}" "${QT_INSTALL_LIBS}" qt_xml_lib)
			endif()
			
			set(QT_LIBRARIES ${qt_core_lib} ${qt_gui_lib} ${qt_xml_lib} ${qt_opengl_lib})
			set(QT_INCLUDE_DIRS "${Qt5Core_INCLUDE_DIRS};${Qt5Widgets_INCLUDE_DIRS};${QtXml_INCLUDE_DIRS};${Qt5Concurrent_INCLUDE_DIRS};${Qt5OpenGL_INCLUDE_DIRS}")
			list(REMOVE_DUPLICATES QT_INCLUDE_DIRS)
			include_dirs_to_flags("${QT_INCLUDE_DIRS}" QT_INCLUDES)
			set(QT_DEFINITIONS "-DQT_NO_DEBUG;${Qt5Core_DEFINITIONS};${Qt5Widgets_DEFINITIONS};${QtXml_DEFINITIONS};${Qt5Concurrent_DEFINITIONS};${Qt5OpenGL_DEFINITIONS}")
			list(REMOVE_DUPLICATES QT_DEFINITIONS)
	
			message(STATUS "-----> using Qt " ${QT_VERSION})
			set(QT_ENABLED 1) #TODO: not sure we should keep running if WITH_QT5=1 and OpenGL or QT is not found

			if(DARWIN)
				set(CMAKE_INSTALL_RPATH "${QT_INSTALL_LIBS}" "${QT_INSTALL_PLUGINS}" "@executable_path/../Frameworks")
			endif()
		endif()

	else()
		message(WARNING "OpenGL not available, Qt graphical tools disabled")
	endif()
endif()

if(${DEPLOY})
    set(DEPLOY 1)
endif()

if(${WITH_CPP11})
	if(NOT MSVC)
		message(STATUS "-----> Compile with c++11 support")
		# support c++11
		if(NOT DARWIN) #TODO: if(NOT clang) => the "precompiled header bug" it tries to fix is clang related, not osx related   
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
		ELSE()
		# MODIF MPD, reverse Gregoire, car gnu++11 mulitplie par 6 le temps de compilation
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
		endif()
		message(STATUS "-- FLAG: " ${CMAKE_CXX_FLAGS})
    endif()
endif()

# CUDA
if(${CUDA_ENABLED})
	option(CUDA_LINEINFO "Compilation avec lineInfo" OFF )
	option(CUDA_FASTMATH "Compilation avec FASTMATH" OFF )
	option(CUDA_CPP11THREAD_NOBOOSTTHREAD "Use c++11 for the multi-threading" OFF )
	option(CUDA_NVTOOLS "Use NV TOOLS" OFF )

	find_package(CUDA REQUIRED)
	include(FindCUDA)

	if(CUDA_FOUND)
		set(Cuda_Sample_path_Windows_VISTA_7_8 "$ENV{AllUsersProfile}/NVIDIA Corporation/CUDA Samples/v${CUDA_VERSION}/")
		set(Cuda_Sample_path_Windows_XP "$ENV{AllUsersProfile}/Application Data/NVIDIA Corporation/CUDA Samples/v${CUDA_VERSION}/")
		set(Cuda_Sample_path_Apple "/Developer/NVIDIA/CUDA-${CUDA_VERSION}/samples/")
		set(Cuda_Sample_path_Linux "${CUDA_TOOLKIT_ROOT_DIR}/samples")

	    find_path(
			CUDA_SAMPLE_DIR
			NAMES
			    common/inc/helper_cuda.h
			PATHS
			    ${Cuda_Sample_path_Linux}
			    ${Cuda_Sample_path_Windows_VISTA_7_8}
							${Cuda_Sample_path_Windows_XP}
			    ${Cuda_Sample_path_Windows_Apple}
		     )

			#message(${CUDA_SAMPLE_DIR})

	    if(NOT CUDA_SAMPLE_DIR STREQUAL "CUDA_SAMPLE_DIR-NOTFOUND")
				#message("Cuda Samples Found")
		message("Cuda Samples Found: " ${CUDA_SAMPLE_DIR})
		include_directories(${CUDA_SAMPLE_DIR}/common/inc)
	    else()
		message("Cuda Samples Dir Not Found, install cuda samples or set CUDA_SAMPLE_DIR")
	    endif()
	endif()

	#IFNOT(${WITH_IGN_ORI})
	# ne pas oublier boost_no_system_paths
	if(NOT ${CUDA_CPP11THREAD_NOBOOSTTHREAD})
		set(Boost_USE_STATIC_LIBS ON)
		set(Boost_NO_SYSTEM_PATHS ON)
		find_package(Boost 1.48.0 COMPONENTS system serialization thread date_time chrono REQUIRED)
		#chrono
		 if(NOT ${WITH_IGN_ORI})
			  include_directories ( ${Boost_INCLUDE_DIRS} )
			  include_directories ( ${Boost_INCLUDE_DIR} )
		 endif()
	else()
		if(MSVC AND NOT MSVC12 )
			message(FATAL_ERROR "This version of visual doesn't support c++11")
		endif()
	    if(NOT ${WITH_CPP11})
			set(WITH_CPP11 ON CACHE BOOL "Compilateur C++11" FORCE)
			if(NOT MSVC)
				message("c++11")
				# support c++11
				add_definitions(-std=c++11) #+other flags? #TODO: check the precompiled header bug
			endif()
	    endif()
	endif()
endif()

# Etalonnage polygone
if (${WITH_ETALONPOLY})
	set(ETA_POLYGON 1)
endif()

# GIT
if (NOT GIT_REVISION_DIST)
	find_package(Git)
	if(GIT_FOUND)
		message(STATUS "-- Git executable found")
		set(HAS_DYN_VERSION)
        execute_process(COMMAND ${GIT_EXECUTABLE} describe --always --dirty --long
        	OUTPUT_VARIABLE GIT_REVISION_DIST
        	ERROR_VARIABLE git_rev_error
            RESULT_VARIABLE git_rev_result
            OUTPUT_STRIP_TRAILING_WHITESPACE)
	else(GIT_FOUND)
		message(STATUS "-- Git executable NOT found")
	endif(GIT_FOUND)
endif()
message(STATUS "Derniere revision git de MicMac : " ${GIT_REVISION_DIST})

set(GIT_REV ${GIT_REVISION_DIST} CACHE STRING "Revision" FORCE)

set(GIT_REVISION_DIST \"${GIT_REVISION_DIST}\" )
configure_file(
    "src/util/GIT_defines.cpp.in"
    "${CMAKE_SOURCE_DIR}/src/util/GIT_defines.cpp"
)

#~ INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR}/include)

# Utiliser le driver Kakadu pour gerer les images JP2OOO
if(${WITH_KAKADU})
	if(KAKADU_NO_FIND)
		# to use when FindKAKAKDU.cmake does not exist (KAKADU_DIR must be set)
		set(KAKADU_LIBRARY_DIRS "${KAKADU_DIR}/lib")
		set(KAKADU_INCLUDE_DIRS "${KAKADU_DIR}/include")
		set(KAKADU_LIBRARIES "kdu_v64R;kdu;kdu_aux")
		# quick check of Kakadu's headers and libraries
		if ( ( (EXISTS "${KAKADU_LIBRARY_DIRS}/kdu.lib") OR
		       (EXISTS "${KAKADU_LIBRARY_DIRS}/libkdu.a") ) AND
		     ( EXISTS "${KAKADU_DIR}/include/jp2.h" ) )
			set(KAKADU_FOUND 1)
		endif()
	else()
		find_package(KAKADU)
	endif()

	if(NOT KAKADU_FOUND)
		message("Warning! Kakadu not found. Micmac will be built without JPEG2000 support.")
	else()
		message(STATUS "Kakadu enabled")
		link_directories(${KAKADU_LIBRARY_DIRS})
		include_directories(${KAKADU_INCLUDE_DIRS})
		include_dirs_to_flags("${KAKADU_INCLUDE_DIRS}" KAKADU_INCLUDES)
		set(__USE_JP2__ 1)
	endif()
endif()

# Utiliser le socle IGN pour gerer les orientations coniques ET/OU les images JP2000
if(${WITH_IGN_ORI} OR ${WITH_IGN_IMAGE})
	set(SOCLE_COMPONENTS "")
	if(${WITH_IGN_ORI})
		set(SOCLE_COMPONENTS orientation geodesy transform math numeric tools  data xml core)
	endif()
	if(${WITH_IGN_IMAGE})
		list(APPEND SOCLE_COMPONENTS image transform math numeric tools data xml core)

		#pour le DMR sinon on peut l'enlever aisni que la dependeance a Geos
		list(APPEND SOCLE_COMPONENTS geometry)
	endif()

	find_package(IgnSocle COMPONENTS "${SOCLE_COMPONENTS}")
	find_package(TinyXML)
	find_package(PROJ4)

	if(${WITH_IGN_IMAGE})
		find_package(GDAL)
		find_package(KAKADU)
		find_package(ECWJ2)
		find_package(Geos)
	endif()

	find_package(Boost 1.48.0 COMPONENTS unit_test_framework date_time filesystem system serialization thread regex chrono REQUIRED)

	#add_definitions("-DBOOST_THREAD_USE_LIB")
	add_definitions("-DBOOST_TYPEOF_COMPLIANT")
	add_definitions("-DBoost_USE_STATIC_LIBS")
	set(Boost_USE_MULTITHREADED ON)
	set(Boost_USE_STATIC_LIBS ON)

	set(DEPENDENCIES "")
	if(IgnSocle_FOUND AND TinyXML_FOUND AND Boost_FOUND AND ${WITH_IGN_ORI})
		add_definitions(-D__USE_ORIENTATIONIGN__ )

		list(APPEND DEPENDENCIES IgnSocle TinyXML Boost)
		if(PROJ4_FOUND)
			list(APPEND DEPENDENCIES PROJ4)
		endif()
	else()
		if(${WITH_IGN_ORI})
			message("IgnSocle_FOUND : ${IgnSocle_FOUND}")
	    	message("TinyXML_FOUND : ${TinyXML_FOUND}")
	    	message("Boost_FOUND : ${Boost_FOUND}")
			message("Warning! Ign Orientation not found. Micmac will be built without Ign Orientation support.")
		endif()
	endif()

	if(IgnSocle_FOUND AND TinyXML_FOUND AND Boost_FOUND AND GDAL_FOUND AND ${WITH_IGN_IMAGE})
		add_definitions(-D__USE_IMAGEIGN__ )

		list(APPEND DEPENDENCIES IgnSocle TinyXML GDAL Boost)
		if(KAKADU_FOUND)
			list(APPEND DEPENDENCIES KAKADU)
		endif()
		if(ECWJ2_FOUND)
			list(APPEND DEPENDENCIES ECWJ2)
		endif()
		if(Geos_FOUND)
			list(APPEND DEPENDENCIES Geos)
		endif()
	else()
		if(${WITH_IGN_IMAGE})
			message("IgnSocle_FOUND : ${IgnSocle_FOUND}")
			message("TinyXML_FOUND : ${TinyXML_FOUND}")
			message("GDAL_FOUND : ${GDAL_FOUND}")
			message("KAKADU_FOUND : ${KAKADU_FOUND}")
			message("ECWJ2_FOUND : ${ECWJ2_FOUND}")
			message("Geos_FOUND : ${Geos_FOUND}")
			message("Boost_FOUND : ${Boost_FOUND}")
			message("Warning! Ign Image not found. Micmac will be built without Ign Image support.")
		endif()
	endif()

	foreach(DEPENDENCY ${DEPENDENCIES})
		link_directories(${${DEPENDENCY}_LIBRARY_DIRS})
		include_directories(${${DEPENDENCY}_INCLUDE_DIRS})
		#message("LINK_DIR ${DEPENDENCY} --> ${${DEPENDENCY}_LIBRARY_DIRS}")
		#message("INCLUDE_DIR ${DEPENDENCY} --> ${${DEPENDENCY}_INCLUDE_DIRS}")
	endforeach()
endif()

# Utiliser la librairie du MATIS pour gerer les orientations coniques
if(${WITH_MATIS_ORI})
	find_package(MatisOrientation)
	if(NOT MatisOrientation_FOUND)
		message("Warning! MatisOrientation not found. Micmac will be built without MatisOrientation support.")
	else()
		option(MICMAC_USE_PROJ4 "Use PROJ4" OFF)
		#-- recherche de boost
		set(Boost_USE_MULTITHREADED ON)
		find_package(Boost 1.48.0 COMPONENTS unit_test_framework date_time system filesystem serialization thread regex chrono REQUIRED)
		find_package(TICPP REQUIRED)
		if(TICPP_FOUND)
			link_directories(${TICPP_LIBRARY_DIRS})
			include_directories(${TICPP_INCLUDE_DIRS})
		endif()

		if(MICMAC_USE_PROJ4)
			find_package(PROJ4 REQUIRED)
			link_directories(${PROJ4_LIBRARY_DIRS})
			include_directories(${PROJ4_INCLUDE_DIRS})
		endif()

		add_definitions(-D__USE_ORIENTATIONMATIS__ )
		link_directories(${MatisOrientation_LIBRARY_DIRS})
		include_directories(${MatisOrientation_INCLUDE_DIRS})
	endif()
endif()

if(MINGW)
	set(__MINGW__ 1)
	add_definitions(-D__MSVCRT_VERSION__=0x800 -D_WIN32_WINNT=0x0500)
	set( WINSOCK2_LIBRARY ws2_32 )
endif()

if(NOT WIN32)
	# VERIFICATION si ImageMagick est installe
	find_package(ImageMagick QUIET)

	if(NOT ImageMagick_FOUND) 
		message("\n***   ATTENTION!!! Vous devez installer Image Magick. ***")
		message("***   http://www.imagemagick.org/script/index.php     ***\n")
	endif(NOT ImageMagick_FOUND)

	# VERIFICATION si Proj4 est installe
	#find_program(PROJ4_EXECUTABLE proj4 PATHS ${UnixBinPath} )
	if(PROJ4_EXECUTABLE_FOUND)
		message("Proj4 non trouve!!")
	endif()

	# VERIFICATION si X11 est present sous MacOS et Linux
	option(NO_X11  "ignoring X11 libraries even if existing" OFF)

	if(NOT NO_X11)
		#message("Trying to find X11 package�")
		FIND_PACKAGE(X11)
	else()
		message("Ignoring X11 package...")
		set(X11_LIBRARY_DIR "") # au cas ou on ait fait une passe de cmake sans actier 'NO_X11'
	endif()

	if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
		set(DYN_LIB_LIBRARY dl)
	endif()

	if(X11_FOUND)
		#message("X11 Found : ")
		#message("   include dir   : " ${X11_INCLUDE_DIR} )
		#message("   libraries     : " ${X11_LIBRARIES} )
		#message("   libraries dir : " ${X11_LIBRARY_DIR} )
		include_directories(${X11_INCLUDE_DIR})
	else()
		set(NO_X11)
		message("X11 Not Found : pas de SaisieMasq")
	endif()
else()
	set(CMAKE_BUILD_TYPE Release) #TODO: why not release for Linux/DARWIN?? 
endif()

# DEFINITION : des preprocesseurs windows
if(MSVC)
	if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
		set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}" CACHE PATH "default install path" FORCE )
	endif()

	set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE" )
	# Zm131 -> Erreur de compilation sous windows 7 avec Cuda et Boost
	if(${CUDA_ENABLED})
	    add_definitions( "/W3 /D_CRT_SECURE_NO_WARNINGS /wd4005 /wd4996 /wd4661 /wd4267 /wd4355 /nologo" )
	#else()
	    add_definitions( "/W3 /D_CRT_SECURE_NO_WARNINGS /wd4005 /wd4996 /wd4661 /wd4267 /nologo" )
	    # la valeur par defaut Zm1000 genere une erreur sous windows XP
			set(zm_value "/Zm200")
			string( FIND "${CMAKE_CXX_FLAGS}" "/Zm1000" has_zm1000 )
			if(${has_zm1000} LESS 0)
				# la valeur par defaut Zm1000 n'existe plus a partir de la version 2.8.11
				set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${zm_value}")
			else()
				# REPLACE pour eviter le conflit Zm1000 vs ZmXXX
				string(REPLACE "/Zm1000" ${zm_value} CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
			endif()
	elseif(NOT (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "19.0.0")) # 19 is visual studio 2015
		set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zm263")
	elseif(${CMAKE_SYSTEM_VERSION} MATCHES "5.1" )
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zm131") # 5.1 is windows xp
	else()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zm170")
	endif()

	if(NOT (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "8.0.0"))
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
	endif()
endif()

# DEFINITION : Dossier include
include_directories(${PROJECT_SOURCE_DIR}/include)

# DEFINITION : des preprocesseurs G++

if(CMAKE_COMPILER_IS_GNUCC)
	#~ add_compile_options(-O2 -Wall -fPIC) // pas gere par le cmake de la forge
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
if (${WERROR})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -fPIC -Werror")
else ()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -fPIC")
endif()
endif()

# DEFINITION : Windows : le projet debug est genere directement sous bin pour avoir les informations de Debug
if(MSVC)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)
endif()

set(BUILD_PATH_BIN ${PROJECT_SOURCE_DIR}/bin CACHE PATH "binaires install path")
set(BUILD_PATH_LIB ${PROJECT_SOURCE_DIR}/lib CACHE PATH "library install path")

# DEFINITION : du dossier d'installation
set(Install_Dir	${BUILD_PATH_BIN})
set(Install_Dir_lib ${BUILD_PATH_LIB})

# DEFINITION : du dossier des sources d'Elise
set(ELISE_DIR src)

# CONSTRUCTION : Librairie ANN
add_subdirectory(${PROJECT_SOURCE_DIR}/CodeExterne/ANN)
# CONSTRUCTION : Librairie POISSON
if(${BUILD_POISSON})
	add_subdirectory(${PROJECT_SOURCE_DIR}/CodeExterne/Poisson)
endif()
# CONSTRUCTION : Librairie Rnx2rtkp
if(${BUILD_RNX2RTKP})
	add_subdirectory(${PROJECT_SOURCE_DIR}/CodeExterne/rnx2rtkp)
endif()
# CONSTRUCTION : Librairie ELISE
set(code_genere_dir ${PROJECT_SOURCE_DIR}/CodeGenere)
include(${code_genere_dir}/Sources.cmake)
add_subdirectory(${ELISE_DIR})

# Doxygen
if(${WITH_DOXYGEN})
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
		configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
		add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMENT "Generating API documentation with Doxygen" VERBATIM)
    endif(DOXYGEN_FOUND)
endif()

if(${BUILD_BENCH})
	include(${PROJECT_SOURCE_DIR}/BenchElise/bench/CMakeLists.txt)
	include(${PROJECT_SOURCE_DIR}/BenchElise/bench_mm3d/CMakeLists.txt)
endif()

if(OUTPUT_ELISE_FILES)
	include("output_elise_files.cmake")
endif()

configure_file(
	"include/general/CMake_defines.h.in"
	"${CMAKE_SOURCE_DIR}/include/general/CMake_defines.h"
)

set_source_files_properties("src/tiff/el_dcraw.c" elise PROPERTIES COMPILE_FLAGS "-O2 -Wall" COMPILE_DEFINITIONS "")

if(${WITH_GRAPHVIZ})
	INCLUDE(FindPkgConfig)
    PKG_SEARCH_MODULE(gvc REQUIRED libgvc libcgraph libcdt)
    LINK_DIRECTORIES(${gvc_LIBRARY_DIRS})
    INCLUDE_DIRECTORIES(${gvc_INCLUDE_DIRS})

	message(STATUS "-----> using graphviz library")
endif()

#~ message("QT_QMAKE_EXECUTABLE = ${QT_QMAKE_EXECUTABLE}")
#~ message("QT_MOC_EXECUTABLE = ${QT_MOC_EXECUTABLE}")
#~ print_list(QT_LIBRARIES)
#~ print_list(QT_INCLUDES)
#~ print_list(QT_DEFINITIONS)
