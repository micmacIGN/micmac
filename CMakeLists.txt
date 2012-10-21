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

# Utiliser Kakadu
OPTION(WITH_KAKADU  "Include KAKADU Support" OFF)

# Utiliser Matis orientation
OPTION(WITH_MATIS_ORI "Include Matis orrientation" OFF )

# Utiliser les en-tetes precompilees
OPTION(WITH_HEADER_PRECOMP "En-tetes precompilees" ON )

# Utiliser SIFTGPU
OPTION(WITH_SIFT_GPU "Utiliser SiftGpu" OFF )

##################################
## 	Trouver les EXES et Libs	##
##################################

# DEFINITION : Dossier des exe et Dll nécessaires sous windows 
set(BinAuxWinDir ${PROJECT_SOURCE_DIR}/binaire-aux/BIN_AUX_WIN32)

# SUBVERSION

FIND_PACKAGE(Subversion)
IF(NOT SUBVERSION_FOUND)
	MESSAGE("Impossible de trouver la version SVN")
	MESSAGE("Impossible d'effectuer les tests sans SVN")
ENDIF()

set(UnixBinPath	/bin
		/usr/bin
		/usr/local/bin)

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
	message("Derniere revision mercurial de MicMac : " ${HG_REVISION_DIST})
	
	set(HG_REV ${HG_REVISION_DIST} CACHE STRING "Revision" FORCE)
endif()

CONFIGURE_FILE (
  "include/general/MM_InstalDir.h.in"
  "${PROJECT_BINARY_DIR}/include/general/MM_InstalDir.h"
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

IF(${WITH_MATIS_ORI})
	FIND_PACKAGE( MatisOrientation )
	IF(NOT MatisOrientation_FOUND)
		MESSAGE("Warning! MatisOrientation not found. Micmac will be built without MatisOrientation support.")
	ELSE()
		OPTION(MICMAC_USE_PROJ4 "Use PROJ4" OFF)
		#-- recherche de boost
		SET(Boost_USE_MULTITHREADED ON)
		FIND_PACKAGE( Boost 1.46.0 COMPONENTS unit_test_framework system filesystem serialization thread  REQUIRED )
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

if(NOT WIN32)

	# VERIFICATION si Proj4 est installé
	#find_program(PROJ4_EXECUTABLE proj4 PATHS ${UnixBinPath} )
	if(PROJ4_EXECUTABLE_FOUND)

		message("Proj4 non trouvé!!")

	endif()
	
	# VERIFICATION si X11 est présent sous MacOS et Linux
	FIND_PACKAGE(X11)
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
	add_definitions( "/W3 /D_CRT_SECURE_NO_WARNINGS /wd4005 /wd4996 /wd4661 /nologo" )
ENDIF()

# DEFINITION : Dossier include
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)

# DEFINITION : des preprocesseurs G++
IF (CMAKE_COMPILER_IS_GNUCC)
	ADD_DEFINITIONS(-O2)
ENDIF()

# DEFINITION : Utilisation de SIFT GPU

IF (WITH_SIFT_GPU)
		ADD_DEFINITIONS(-DUSE_SIFT_GPU)
ENDIF()

# DEFINITION : Windows : le projet debug est généré directement sous bin pour avoir les informations de Debug

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)

# DEFINITION : du dossier d'installation
set(Install_Dir ${PROJECT_SOURCE_DIR}/bin)

# DEFINITION : du dossier des sources d'Elise
set(ELISE_DIR src)

# DEFINITION : du dossier des sources des applications
set(APPLIS_DIR applis)

# CONSTRUCTION : de la librairie Elise
ADD_SUBDIRECTORY(${ELISE_DIR}) 

# CONSTRUCTION : des applications
ADD_SUBDIRECTORY(${APPLIS_DIR})

# INSTALLATION : Liste des exe et Dll nécessaires sous windows
if (win32)
	set(BinAuxWin 	
		${BinAuxWinDir}/exiftool.exe
		${BinAuxWinDir}/exiv2.exe
		${BinAuxWinDir}/libexpat.dll
		${BinAuxWinDir}/libiconv2.dll
		${BinAuxWinDir}/libintl3.dll
		${BinAuxWinDir}/make.exe
	)
endif()

# INSTALLATION : des exe et Dll nécessaires sous windows 
install(FILES ${BinAuxWin} DESTINATION ${Install_Dir})
