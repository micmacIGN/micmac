# fichier contenant la fonction gerant les en-tetes précompilées
include(${CMAKE_CURRENT_SOURCE_DIR}/precompiled_headers.cmake)

cmake_minimum_required( VERSION 2.8.1 )
project(MICMAC)

set(VERBOSE 0)

set (MICMAC_VERSION_MAJOR 1)
set (MICMAC_VERSION_MINOR 0)

##############################
## 			OPTIONS			##
##############################

# construire l'interface graphique
OPTION(WITH_INTERFACE  "build graphic interface" OFF)

# Utiliser Kakadu
OPTION(WITH_KAKADU  "Include KAKADU Support" OFF)

# Utiliser Ign orientation
OPTION(WITH_IGN_ORI "Include Ign orientation" OFF )

# Utiliser les en-tetes precompilees
OPTION(WITH_HEADER_PRECOMP "En-tetes precompilees" ON )

# Utiliser les en-tetes precompilees
OPTION(BUILD_ONLY_ELISE "Projet Elise uniquement" OFF )

# construire l'interface graphique
OPTION(WITH_ETALONPOLY  "build etalonnage polygone" ON)

# Utiliser Cuda
OPTION(CUDA_ENABLED "Utilisation de cuda" OFF )

# Multi thread Poisson avec OpenMP
OPTION(WITH_OPEN_MP "use OpenMP" OFF )

if ( WITH_OPEN_MP )
	FIND_PACKAGE(OpenMP REQUIRED)
	message("using OpenMP for Poisson 3d shell computing")
	list( APPEND CMAKE_C_FLAGS ${OpenMP_C_FLAGS} )
	list( APPEND CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS} )
	add_definitions(-DUSE_OPEN_MP)
endif()

# Utiliser Cuda
OPTION(WITH_DOXYGEN "Generation de documentation" OFF )


######################################
## 	Trouver les EXES et Libs	##
######################################
IF ( ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" )
    add_definitions(-DELISE_Darwin)
endif()
# systeme 32 ou 64 bits
if (WIN32)
    set(CPU_BITNESS 64)
    if (CMAKE_CL_64)
        set(CPU_BITNESS 32)
    endif()
else()
    set(CPU_BITNESS 32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(CPU_BITNESS 64)
    endif()
endif()
message( STATUS "Systeme is " ${CMAKE_SYSTEM_NAME} ${CPU_BITNESS} )

# DEFINITION : Dossier des exe et Dll nécessaires sous windows
set(BinAuxWinDir ${PROJECT_SOURCE_DIR}/binaire-aux/BIN_AUX_WIN32)

# SUBVERSION

FIND_PACKAGE(Subversion)
IF(NOT SUBVERSION_FOUND)
	MESSAGE("Impossible de trouver la version SVN")
	MESSAGE("Impossible d'effectuer les tests sans SVN")
ENDIF()

# DEFINITIION : des chemins binaires sous UNIX
set(UnixBinPath
		/bin
		/usr/bin
		/usr/local/bin)

# CUDA
if (${CUDA_ENABLED})

	FIND_PACKAGE(CUDA)
	INCLUDE(FindCUDA)
	INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/inc)
	#IFNOT(${WITH_IGN_ORI})
	set(Boost_USE_STATIC_LIBS TRUE)
	set(BOOST_USE_SYSTEM FALSE)
	FIND_PACKAGE( Boost 1.48.0 COMPONENTS system serialization thread REQUIRED )
	#chrono
	 IF(NOT ${WITH_IGN_ORI})
		  include_directories ( ${Boost_INCLUDE_DIRS} )
		  include_directories ( ${Boost_INCLUDE_DIR} )
	 endif()
endif()

# Etalonnage polygone

if (${WITH_ETALONPOLY})

	add_definitions(-DETA_POLYGON)

endif()


# MERCURIAL
find_program(MERCURIAL_EXECUTABLE hg DOC "Emplacement de gestionnaire de versions Mercurial")

if(MERCURIAL_EXECUTABLE-NOTFOUND)
	message("Vous devez installer TortoiseHG")
else()
	# Affiche la revision MICMAC
	EXECUTE_PROCESS(
		COMMAND
		hg log -l 1 -b "default" --template "{rev}"
		WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
		OUTPUT_VARIABLE HG_REVISION_DIST
		ERROR_QUIET
		OUTPUT_STRIP_TRAILING_WHITESPACE
	)
	message( STATUS "Derniere revision mercurial de MicMac : " ${HG_REVISION_DIST})

	set(HG_REV ${HG_REVISION_DIST} CACHE STRING "Revision" FORCE)
endif()

CONFIGURE_FILE (
  "include/general/MM_InstalDir.h.in"
  "${CMAKE_SOURCE_DIR}/include/general/MM_InstalDir.h"
  )

INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR}/include)

# Utiliser Kakadu

IF(${WITH_KAKADU})
	FIND_PACKAGE( KAKADU )
	IF(NOT KAKADU_FOUND)
		MESSAGE("Warning! Kakadu not found. Micmac will be built without JPEG2000 support.")
	ELSE()
		ADD_DEFINITIONS(-D__USE_JP2__)
		LINK_DIRECTORIES( ${KAKADU_LIBRARY_DIRS})
		INCLUDE_DIRECTORIES( ${KAKADU_INCLUDE_DIRS})
		MESSAGE("KAKADU : ${KAKADU_LIBRARIES}")
	ENDIF()

ENDIF()

# Utiliser Matis orientation

IF(${WITH_IGN_ORI})
	FIND_PACKAGE( IgnSocle  COMPONENTS core data orientation geodesy transform math numeric tools xml)
	FIND_PACKAGE(TinyXML)
	#add_definitions( "-DBOOST_THREAD_USE_LIB" )
	SET(Boost_USE_MULTITHREADED ON)
	FIND_PACKAGE( Boost 1.48.0 COMPONENTS unit_test_framework date_time filesystem system serialization thread regex chrono REQUIRED)
	IF(IgnSocle_FOUND AND TinyXML_FOUND AND Boost_FOUND)
        	INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
        	LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
		message("TinyXML_LIBRARY_DIRS : ${TinyXML_LIBRARY_DIRS}")
		LINK_DIRECTORIES(${TinyXML_LIBRARY_DIRS})
		INCLUDE_DIRECTORIES(${TinyXML_INCLUDE_DIRS})
		ADD_DEFINITIONS(-D__USE_ORIENTATIONIGN__ )
		LINK_DIRECTORIES(${IgnSocle_LIBRARY_DIRS})
		INCLUDE_DIRECTORIES(${IgnSocle_INCLUDE_DIRS})
	ELSE()
		MESSAGE("IgnSocle_FOUND : ${IgnSocle_FOUND}")
                MESSAGE("TinyXML_FOUND : ${TinyXML_FOUND}")
                MESSAGE("Boost_FOUND : ${Boost_FOUND}")
		MESSAGE("Warning! Ign Orientation not found. Micmac will be built without Ign Orientation support.")
	ENDIF()
ENDIF()


IF(${WITH_MATIS_ORI})
	FIND_PACKAGE( MatisOrientation )
	IF(NOT MatisOrientation_FOUND)
		MESSAGE("Warning! MatisOrientation not found. Micmac will be built without MatisOrientation support.")
	ELSE()
		OPTION(MICMAC_USE_PROJ4 "Use PROJ4" OFF)
		#-- recherche de boost
		SET(Boost_USE_MULTITHREADED ON)
		FIND_PACKAGE( Boost 1.48.0 COMPONENTS unit_test_framework date_time system filesystem serialization thread regex chrono REQUIRED)
		FIND_PACKAGE(TICPP REQUIRED)
		IF ( TICPP_FOUND)
			LINK_DIRECTORIES(${TICPP_LIBRARY_DIRS})
			INCLUDE_DIRECTORIES(${TICPP_INCLUDE_DIRS})
		ENDIF()

		IF (MICMAC_USE_PROJ4)
			FIND_PACKAGE( PROJ4 REQUIRED )
			LINK_DIRECTORIES(${PROJ4_LIBRARY_DIRS})
			INCLUDE_DIRECTORIES(${PROJ4_INCLUDE_DIRS})
		ENDIF()

		ADD_DEFINITIONS(-D__USE_ORIENTATIONMATIS__ )
		LINK_DIRECTORIES(${MatisOrientation_LIBRARY_DIRS})
		INCLUDE_DIRECTORIES(${MatisOrientation_INCLUDE_DIRS})

	ENDIF()
ENDIF()


# VERIFICATION si ImageMagick est installé

find_package(ImageMagick REQUIRED)

if(NOT ImageMagick_FOUND)
	MESSAGE("ATTENTION!!! Vous devez installer Image Magick.")
	MESSAGE("http://www.imagemagick.org/script/index.php")
endif(NOT ImageMagick_FOUND)

# VERIFICATION si exiftool est installé

find_program(EXIF_EXECUTABLE exiftool
				PATHS 	${BinAuxWinDir} ${UnixBinPath}
				DOC "Chemin vers la commande exiftool"
				NO_DEFAULT_PATH)

if(EXIF_EXECUTABLE_NOTFOUND)

	message("Exiftool non trouvé!!")

endif()

# VERIFICATION si exiv2 est installé
find_program(EXIV2_EXECUTABLE exiv2
				PATHS 	${BinAuxWinDir} ${UnixBinPath}
				DOC "Chemin vers la commande exiv2"
				NO_DEFAULT_PATH)

if(EXIV2_EXECUTABLE_NOTFOUND)

	message("Exiv2 non trouvé!!")

endif(EXIV2_EXECUTABLE_NOTFOUND)

if (MINGW)
	add_definitions(-D__MINGW__)
	set( WINSOCK2_LIBRARY ws2_32 )
endif()

if(NOT WIN32)

	# VERIFICATION si Proj4 est installé
	#find_program(PROJ4_EXECUTABLE proj4 PATHS ${UnixBinPath} )
	if(PROJ4_EXECUTABLE_FOUND)

		message("Proj4 non trouvé!!")

	endif()

	# VERIFICATION si X11 est présent sous MacOS et Linux
	FIND_PACKAGE(X11)
	
	IF ( ${CMAKE_SYSTEM_NAME} MATCHES "Linux" )
		set( DYN_LIB_LIBRARY dl )
	endif()
	
	IF (X11_FOUND)
		#MESSAGE("X11 Found : ${X11_INCLUDE_DIRS} | ${X11_LIBRARIES} | ${X11_LIBRARY_DIRS}")
	ELSE()
		add_definitions(-DNO_X11)
		MESSAGE("X11 Not Found : pas de SaisieMasq")
	ENDIF()
ELSE()

	find_program(MAKE_EXECUTABLE make
				PATHS 	${BinAuxWinDir}
				DOC "Chemin vers la commande Make"
				NO_DEFAULT_PATH)

	if(MAKE_EXECUTABLE_NOTFOUND)

		message("Make non trouvé!!")

	endif()

ENDIF()

# DEFINITION : des preprocesseurs windows
IF (MSVC)
	add_definitions( "/W3 /D_CRT_SECURE_NO_WARNINGS /wd4005 /wd4996 /wd4661 /wd4267 /nologo" )

	if(MSVC80 OR MSVC90 OR MSVC10)
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
	endif()

ENDIF()

# DEFINITION : Dossier include
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)

# DEFINITION : des preprocesseurs G++
IF (CMAKE_COMPILER_IS_GNUCC)
	ADD_DEFINITIONS(-O2 -Wall)
ENDIF()

# DEFINITION : Windows : le projet debug est généré directement sous bin pour avoir les informations de Debug

IF (MSVC)
	 set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)
	 set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)
endif()

# DEFINITION : du dossier d'installation
set(Install_Dir ${PROJECT_SOURCE_DIR}/bin)
set(Install_Dir_lib ${PROJECT_SOURCE_DIR}/lib)

# DEFINITION : du dossier des sources d'Elise
set(ELISE_DIR src)

# CONSTRUCTION : Librairie ANN
include(${PROJECT_SOURCE_DIR}/CodeExterne/ANN/CMakeLists.txt)
# CONSTRUCTION : Librairie ELISE
set( code_genere_dir ${PROJECT_SOURCE_DIR}/CodeGenere )
include(${code_genere_dir}/Sources.cmake)
ADD_SUBDIRECTORY(${ELISE_DIR})

# INSTALLATION : Liste des exe et Dll nécessaires sous windows
if ( ${CMAKE_SYSTEM_NAME} STREQUAL "Windows" )
	configure_file( ${BinAuxWinDir}/convert.exe ${PROJECT_SOURCE_DIR}/binaire-aux/convert.exe COPYONLY )
	configure_file( ${BinAuxWinDir}/exiftool.exe ${PROJECT_SOURCE_DIR}/binaire-aux/exiftool.exe COPYONLY )
	configure_file( ${BinAuxWinDir}/exiv2.exe ${PROJECT_SOURCE_DIR}/binaire-aux/exiv2.exe COPYONLY )
	configure_file( ${BinAuxWinDir}/libexpat.dll ${PROJECT_SOURCE_DIR}/binaire-aux/libexpat.dll COPYONLY )
	configure_file( ${BinAuxWinDir}/libiconv2.dll ${PROJECT_SOURCE_DIR}/binaire-aux/libiconv2.dll COPYONLY )
	configure_file( ${BinAuxWinDir}/libintl3.dll ${PROJECT_SOURCE_DIR}/binaire-aux/libintl3.dll COPYONLY )
	configure_file( ${BinAuxWinDir}/make.exe ${PROJECT_SOURCE_DIR}/binaire-aux/make.exe COPYONLY )
endif()

# Doxygen
if(${WITH_DOXYGEN})
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
	  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    message("${CMAKE_CURRENT_SOURCE_DIR}")
	  add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMENT "Generating API documentation with Doxygen" VERBATIM)
    endif(DOXYGEN_FOUND)
endif()
