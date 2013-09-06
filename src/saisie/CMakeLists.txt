cmake_minimum_required(VERSION 2.8)
project(saisie)

if ( MSVC )
    # for Qt5 to find glu32.lib
    get_filename_component(WindowsSDKDir
                           "[HKEY_CURRENT_USER\\Software\\Microsoft\\Microsoft SDKs\\Windows;CurrentInstallFolder]"
                           ABSOLUTE)
    set( CMAKE_PREFIX_PATH "${WindowsSDKDir}\\Lib" )
endif()

# Find the QtWidgets library
find_package(OpenGL REQUIRED)
message( "OPENGL_LIBRARIES   : " ${OPENGL_LIBRARIES} )
message( "OPENGL_gl_LIBRARY  : " ${OPENGL_gl_LIBRARY} )
message( "OPENGL_glu_LIBRARY : " ${OPENGL_glu_LIBRARY} )
find_package(Qt5Widgets REQUIRED)

if ( NOT ${OPENGL_GLU_FOUND} )
	message( FATAL_ERROR "GLU library could not be found " )
	return( -1 )
endif()

set(CMAKE_AUTOMOC ON)

set(saisie_SRCS main.cpp
	        GLWidget.cpp
	        mainwindow.cpp
	        Cloud.cpp
	        Data.cpp
	        Engine.cpp
	        ../poisson/plyfile.cpp)

set(saisie_HEADERS GLWidget.h
		   mainwindow.h
		   Cloud.h
		   Engine.h
		   Data.h
		   3DTools.h)

SET(Saisie_RCS ./icones/icones.qrc)

QT5_ADD_RESOURCES( Saisie_RC_SRCS ${Saisie_RCS} )

qt5_wrap_ui(saisie_ui mainwindow.ui)

add_definitions(-DTWEAK)
if ( WIN32 )
  add_definitions(-DELISE_windows)
ENDIF()

message( "opengl include : " ${OPENGL_INCLUDE_DIR} )

include_directories( ${Qt5_INCLUDES}
		     ${OPENGL_INCLUDE_DIR}
		     ${PROJECT_SOURCE_DIR}
		     ${PROJECT_BINARY_DIR} )

add_executable(saisie ${saisie_SRCS} ${Saisie_RC_SRCS} ${saisie_ui} ${saisie_HEADERS_MOC})

qt5_use_modules(saisie Widgets Core Gui Xml Concurrent OpenGL)

target_link_libraries(saisie  ${OPENGL_LIBRARIES}
			      elise
			      ${X11_LIBRARIES} )

install(TARGETS saisie DESTINATION ${Install_Dir})
