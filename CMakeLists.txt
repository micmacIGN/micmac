# fichier contenant la fonction gerant les en-tetes precompilees
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

# Utiliser Ign orientation
OPTION(WITH_IGN_IMAGE "Include Ign image" OFF )

# Utiliser les en-tetes precompilees
OPTION(WITH_HEADER_PRECOMP "En-tetes precompilees" ON )

# Utiliser les en-tetes precompilees
OPTION(BUILD_ONLY_ELISE_MM3D "Projet Elise et MM3D uniquement" OFF )

# construire l'interface graphique
OPTION(WITH_ETALONPOLY  "build etalonnage polygone" ON)

# Utiliser Cuda
OPTION(CUDA_ENABLED "Utilisation de cuda" OFF )

# Multi thread Poisson avec OpenMP
OPTION(WITH_OPEN_MP "use OpenMP" OFF )

# print system calls
OPTION(TRACE_SYSTEM "print system calls" OFF )

# compile low-level test bench
OPTION(BUILD_BENCH "compile low-level test bench" OFF )

# Interfaces de saisie des paramètres
OPTION(WITH_QT5 "compile Qt interfaces" OFF )

# Interfaces de saisie des paramètres
OPTION(WITH_QT4 "compile Qt 4 interfaces" OFF )

#
OPTION(WITH_MM3dSat "Graphical interface for MNS creation" OFF)

if ( TRACE_SYSTEM )
        add_definitions(-D__TRACE_SYSTEM__)
endif()

if ( WITH_OPEN_MP )
        FIND_PACKAGE(OpenMP REQUIRED)
        if (${CUDA_ENABLED})
            message("using OpenMP for Poisson 3d shell computing and Cuda precompute")
        else()
            message("using OpenMP for Poisson 3d shell computing")
        endif()
        list( APPEND CMAKE_C_FLAGS ${OpenMP_C_FLAGS} )
        list( APPEND CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS} )
        add_definitions(-DUSE_OPEN_MP)
endif()

# Utiliser Doxygen
OPTION(WITH_DOXYGEN "Generation de documentation" OFF )


######################################
## 	Trouver les EXES et Libs	##
######################################
IF ( ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" )
    add_definitions(-DELISE_Darwin)
endif()

# DEFINITION : Dossier des exe et Dll necessaires sous windows
set(BinAuxWinDir ${PROJECT_SOURCE_DIR}/binaire-aux/BIN_AUX_WIN32)

# DEFINITIION : des chemins binaires sous UNIX
set(UnixBinPath
                /bin
                /usr/bin
                /usr/local/bin)

set(qt_version 0)

if (${WITH_QT4} OR ${WITH_QT5})
    find_package(OpenGL)
endif()

if ( OPENGL_FOUND AND OPENGL_GLU_FOUND )
   message( STATUS "-----> OpenGL found" )

    if ( MSVC )
                include("${PROJECT_SOURCE_DIR}/src/saisieQT/FindWindowsSDK.txt")
                # ajoute /lib a tous les répertoires contenant un SDK
                foreach( sdk_dir ${WINDOWSSDK_DIRS} )
                        list( APPEND SDK_LIB_DIRS "${sdk_dir}/lib" )
                endforeach()
                set( CMAKE_PREFIX_PATH "${SDK_LIB_DIRS}" )
                message (" WindowsSDKDir : " ${CMAKE_PREFIX_PATH} )
    endif()

    if(${WITH_QT4})
        message("Force Qt 4")
    else()
       find_package(Qt5Widgets QUIET)
       find_package(Qt5Core QUIET)
       find_package(Qt5Gui QUIET)
       find_package(Qt5Xml QUIET)
       find_package(Qt5Concurrent QUIET)
       find_package(Qt5OpenGL QUIET)
    endif()

   if ( Qt5Widgets_FOUND AND
        Qt5Core_FOUND AND
        Qt5Gui_FOUND AND
        Qt5Xml_FOUND AND
        Qt5Concurrent_FOUND AND
        Qt5OpenGL_FOUND )
      message( STATUS "-----> Qt5 found" )
      set(qt_version 5)
   else()
      FIND_PACKAGE(Qt4 QUIET COMPONENTS QtCore QtGui QtOpenGL QtXml)
      if ( QT4_FOUND )
         message( STATUS "-----> Qt4 found" )
         set(qt_version 4)
      else()
         message( "Qt not available, Qt graphical tools disabled" )
         message( "**************************************************************************************************************************************" )
         message( "If Qt4 is installed, please set QT_QMAKE_EXECUTABLE" )
         message( "If Qt5 is installed, please set CMAKE_PREFIX_PATH (ex: cmake ../ -DCMAKE_PREFIX_PATH=/opt/Qt5.1.1/5.1.1/gcc_64/lib/cmake/Qt5Widgets/ )" )
         message( "**************************************************************************************************************************************" )
      endif()
   endif()
else()
   message( "OpenGL or GLU not available, Qt graphical tools disabled" )
endif()

if ( NOT ${qt_version} EQUAL 0 )
      message( STATUS "QT tools enabled (Qt" ${qt_version} ")" )
endif()

if ( ${WITH_MM3dSat} )
   if ( NOT ${qt_version} EQUAL 0 )
      add_definitions(-DMM3dSat)
   else()
      set( WITH_MM3dSat false )
   endif()
endif()

# CUDA
if (${CUDA_ENABLED})

        FIND_PACKAGE(CUDA)
        INCLUDE(FindCUDA)
        INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/inc)
        #IFNOT(${WITH_IGN_ORI})

    # ne pas oublier boost_no_system_paths
        set(Boost_USE_STATIC_LIBS ON)
    set(Boost_NO_SYSTEM_PATHS ON)
        FIND_PACKAGE( Boost 1.48.0 COMPONENTS system serialization thread date_time chrono REQUIRED )
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

if(NOT MERCURIAL_EXECUTABLE)
        message( STATUS "Mercurial (hg) cannot be found" )
else()
        # Affiche la revision MICMAC
        EXECUTE_PROCESS(
                COMMAND
                hg identify --num
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                RESULT_VARIABLE hg_result
                OUTPUT_VARIABLE HG_REVISION_DIST
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        message( STATUS "Derniere revision mercurial de MicMac : " ${HG_REVISION_DIST})

        set(HG_REV ${HG_REVISION_DIST} CACHE STRING "Revision" FORCE)

        if ( NOT ${hg_result} )
           set( HG_REVISION_DIST \"${HG_REVISION_DIST}\" )
           CONFIGURE_FILE (
              "include/general/HG_defines.h.in"
              "${CMAKE_SOURCE_DIR}/include/general/HG_defines.h"
           )
        endif()
endif()

CONFIGURE_FILE (
        "include/general/CMake_defines.h.in"
        "${CMAKE_SOURCE_DIR}/include/general/CMake_defines.h"
)

INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR}/include)

# Utiliser le driver Kakadu pour gerer les images JP2OOO
IF(${WITH_KAKADU} AND NOT ${WITH_IGN_IMAGE})
#IF(${WITH_KAKADU})
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

# Utiliser le socle IGN pour gerer les orientations coniques ET/OU les images JP2000
IF(${WITH_IGN_ORI} OR ${WITH_IGN_IMAGE})
        SET(SOCLE_COMPONENTS "")
        IF(${WITH_IGN_ORI})
                SET(SOCLE_COMPONENTS core data orientation geodesy transform math numeric tools xml)
        ENDIF()
        IF(${WITH_IGN_IMAGE})
                LIST(APPEND SOCLE_COMPONENTS core data image transform math numeric tools xml)

                #pour le DMR sinon on peut l'enlever aisni que la dependeance a Geos
                LIST(APPEND SOCLE_COMPONENTS geometry)
        ENDIF()

        FIND_PACKAGE( IgnSocle  COMPONENTS "${SOCLE_COMPONENTS}")
        FIND_PACKAGE(TinyXML)
        FIND_PACKAGE(PROJ4)

        if (${WITH_IGN_IMAGE})
                FIND_PACKAGE(GDAL)
                FIND_PACKAGE( KAKADU )
                FIND_PACKAGE(ECWJ2)
                FIND_PACKAGE(Geos)
        endif()

        FIND_PACKAGE( Boost 1.48.0 COMPONENTS unit_test_framework date_time filesystem system serialization thread regex chrono REQUIRED)

        #add_definitions( "-DBOOST_THREAD_USE_LIB" )
        add_definitions( "-DBOOST_TYPEOF_COMPLIANT" )
        add_definitions( "-DBoost_USE_STATIC_LIBS" )
        SET(Boost_USE_MULTITHREADED ON)
        SET(Boost_USE_STATIC_LIBS ON)

        SET(DEPENDENCIES "")
        IF(IgnSocle_FOUND AND TinyXML_FOUND AND Boost_FOUND AND ${WITH_IGN_ORI})
                ADD_DEFINITIONS(-D__USE_ORIENTATIONIGN__ )

                LIST(APPEND DEPENDENCIES  IgnSocle TinyXML Boost)
                IF ( PROJ4_FOUND )
                        LIST(APPEND DEPENDENCIES PROJ4)
                ENDIF()
        ELSE()
                IF (${WITH_IGN_ORI})
                        MESSAGE("IgnSocle_FOUND : ${IgnSocle_FOUND}")
            MESSAGE("TinyXML_FOUND : ${TinyXML_FOUND}")
            MESSAGE("Boost_FOUND : ${Boost_FOUND}")
                        MESSAGE("Warning! Ign Orientation not found. Micmac will be built without Ign Orientation support.")
                ENDIF()
        ENDIF()

        IF(IgnSocle_FOUND AND TinyXML_FOUND AND Boost_FOUND AND GDAL_FOUND AND ${WITH_IGN_IMAGE})
                ADD_DEFINITIONS(-D__USE_IMAGEIGN__ )

                LIST(APPEND DEPENDENCIES IgnSocle TinyXML GDAL Boost)
                IF (KAKADU_FOUND)
                        LIST(APPEND DEPENDENCIES KAKADU)
                ENDIF()
                IF (ECWJ2_FOUND)
                        LIST(APPEND DEPENDENCIES ECWJ2)
                ENDIF()
                IF (Geos_FOUND)
                        LIST(APPEND DEPENDENCIES Geos)
                ENDIF()
        ELSE()
                IF (${WITH_IGN_IMAGE})
                        MESSAGE("IgnSocle_FOUND : ${IgnSocle_FOUND}")
                        MESSAGE("TinyXML_FOUND : ${TinyXML_FOUND}")
            MESSAGE("GDAL_FOUND : ${GDAL_FOUND}")
            MESSAGE("KAKADU_FOUND : ${KAKADU_FOUND}")
            MESSAGE("ECWJ2_FOUND : ${ECWJ2_FOUND}")
            MESSAGE("Geos_FOUND : ${Geos_FOUND}")
                        MESSAGE("Boost_FOUND : ${Boost_FOUND}")
                        MESSAGE("Warning! Ign Image not found. Micmac will be built without Ign Image support.")
                ENDIF()
        ENDIF()

        foreach( DEPENDENCY ${DEPENDENCIES} )
                LINK_DIRECTORIES   (${${DEPENDENCY}_LIBRARY_DIRS})
                INCLUDE_DIRECTORIES(${${DEPENDENCY}_INCLUDE_DIRS})
                #message("LINK_DIR ${DEPENDENCY} --> ${${DEPENDENCY}_LIBRARY_DIRS}")
                #message("INCLUDE_DIR ${DEPENDENCY} --> ${${DEPENDENCY}_INCLUDE_DIRS}")
        endforeach()

ENDIF()

# Utiliser la librairie du MATIS pour gerer les orientations coniques
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


# VERIFICATION si ImageMagick est installe

find_package(ImageMagick REQUIRED)

if(NOT ImageMagick_FOUND)
        MESSAGE("ATTENTION!!! Vous devez installer Image Magick.")
        MESSAGE("http://www.imagemagick.org/script/index.php")
endif(NOT ImageMagick_FOUND)

# VERIFICATION si exiftool est installe

find_program(EXIF_EXECUTABLE exiftool
                                PATHS 	${BinAuxWinDir} ${UnixBinPath}
                                DOC "Chemin vers la commande exiftool"
                                NO_DEFAULT_PATH)

if(EXIF_EXECUTABLE_NOTFOUND)

        message("Exiftool non trouve!!")

endif()

# VERIFICATION si exiv2 est installe
find_program(EXIV2_EXECUTABLE exiv2
                                PATHS 	${BinAuxWinDir} ${UnixBinPath}
                                DOC "Chemin vers la commande exiv2"
                                NO_DEFAULT_PATH)

if(EXIV2_EXECUTABLE_NOTFOUND)

        message("Exiv2 non trouve!!")

endif(EXIV2_EXECUTABLE_NOTFOUND)

if (MINGW)
        add_definitions(-D__MINGW__)
        set( WINSOCK2_LIBRARY ws2_32 )
endif()

if(NOT WIN32)

        # VERIFICATION si Proj4 est installe
        #find_program(PROJ4_EXECUTABLE proj4 PATHS ${UnixBinPath} )
        if(PROJ4_EXECUTABLE_FOUND)

                message("Proj4 non trouve!!")

        endif()

        # VERIFICATION si X11 est present sous MacOS et Linux
        OPTION(NO_X11  "ignoring X11 libraries even if existing" OFF)

        IF (NOT NO_X11)
                #message("Trying to find X11 packageÉ")
                FIND_PACKAGE(X11)
        ELSE()
                message("Ignoring X11 package...")
                set(X11_LIBRARY_DIR "") # au cas ou on ait fait une passe de cmake sans actier 'NO_X11'
        ENDIF()

        IF ( ${CMAKE_SYSTEM_NAME} MATCHES "Linux" )
                set( DYN_LIB_LIBRARY dl )
        endif()

        IF (X11_FOUND)
                #message("X11 Found : ")
                #message("   include dir   : " ${X11_INCLUDE_DIR} )
                #message("   libraries     : " ${X11_LIBRARIES} )
                #message("   libraries dir : " ${X11_LIBRARY_DIR} )
        INCLUDE_DIRECTORIES( ${X11_INCLUDE_DIR} )
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

                message("Make non trouve!!")

        endif()

ENDIF()

# DEFINITION : des preprocesseurs windows
IF (MSVC)

        # Zm131 -> Erreur de compilation sous windows 7 avec Cuda et Boost
        if (${CUDA_ENABLED})
            add_definitions( "/W3 /D_CRT_SECURE_NO_WARNINGS /wd4005 /wd4996 /wd4661 /wd4267 /wd4355 /nologo" )
        else()
            add_definitions( "/W3 /D_CRT_SECURE_NO_WARNINGS /wd4005 /wd4996 /wd4661 /wd4267 /nologo" )
            # la valeur par defaut Zm1000 genere une erreur sous windows XP
                        set(zm_value "/Zm200")
                        string( FIND "${CMAKE_CXX_FLAGS}" "/Zm1000" has_zm1000 )
                        if ( ${has_zm1000} LESS 0 )
                                # la valeur par defaut Zm1000 n'existe plus a partir de la version 2.8.11
                                set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${zm_value}")
                        else()
                                # REPLACE pour eviter le conflit Zm1000 vs ZmXXX
                                string(REPLACE "/Zm1000" ${zm_value} CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
                        endif()
        endif()

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

# DEFINITION : Windows : le projet debug est genere directement sous bin pour avoir les informations de Debug

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

# INSTALLATION : Liste des exe et Dll necessaires sous windows
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
          add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMENT "Generating API documentation with Doxygen" VERBATIM)
    endif(DOXYGEN_FOUND)
endif()

if ( ${BUILD_BENCH} )
   include(${PROJECT_SOURCE_DIR}/BenchElise/bench/CMakeLists.txt)
   include(${PROJECT_SOURCE_DIR}/BenchElise/bench_mm3d/CMakeLists.txt)
endif()
