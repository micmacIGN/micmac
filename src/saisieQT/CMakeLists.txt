cmake_minimum_required(VERSION 2.8.9)
project(saisieQT)

if ( MSVC )
	include("FindWindowsSDK.txt")
    
    set( CMAKE_PREFIX_PATH "${WINDOWSSDK_DIRS}\\Lib" )
	
	message (" WindowsSDKDir : " ${CMAKE_PREFIX_PATH} )
endif()

if ( WIN32 )
	#cmake_policy(SET CMP0020 NEW)
ENDIF()

# Find the QtWidgets library
find_package(OpenGL REQUIRED)
#message( "OPENGL_LIBRARIES   : " ${OPENGL_LIBRARIES} )
#message( "OPENGL_gl_LIBRARY  : " ${OPENGL_gl_LIBRARY} )
#message( "OPENGL_glu_LIBRARY : " ${OPENGL_glu_LIBRARY} )
find_package(Qt5Widgets REQUIRED)

if ( NOT ${OPENGL_GLU_FOUND} )
	message( FATAL_ERROR "GLU library could not be found " )
	return( -1 )
endif()

set(CMAKE_AUTOMOC ON)

set(saisie_SRCS saisieQT_main.cpp
                saisieMasqQT_main.cpp
				saisieAppuisInitQT_main.cpp
	        GLWidget.cpp
	        mainwindow.cpp
	        Cloud.cpp
	        Data.cpp
	        Engine.cpp
            3DObject.cpp
			GLWidgetSet.cpp)

set(saisie_HEADERS saisieQT_main.h
           GLWidget.h
		   mainwindow.h
		   Cloud.h
		   Engine.h
		   Data.h
		   3DObject.h
		   3DTools.h
		   GLWidgetSet.h)

SET(Saisie_RCS ./icones/icones.qrc)

QT5_ADD_RESOURCES( Saisie_RC_SRCS ${Saisie_RCS} )

qt5_wrap_ui(saisie_ui mainwindow.ui)

add_definitions(-DTWEAK)
if ( WIN32 )
  add_definitions(-DELISE_windows)
ENDIF()

#message( "opengl include : " ${OPENGL_INCLUDE_DIR} )

include_directories( ${Qt5_INCLUDES}
		     ${OPENGL_INCLUDE_DIR}
		     ${PROJECT_SOURCE_DIR}
		     ${PROJECT_BINARY_DIR} )

IF(WIN32)
        add_executable(SaisieQT WIN32 ${saisie_SRCS} ${Saisie_RC_SRCS} ${saisie_ui} ${saisie_HEADERS_MOC})
else()
        add_executable(SaisieQT ${saisie_SRCS} ${Saisie_RC_SRCS} ${saisie_ui} ${saisie_HEADERS_MOC})
endif()

qt5_use_modules(SaisieQT Widgets Core Gui Xml Concurrent OpenGL)

target_link_libraries(SaisieQT  ${OPENGL_LIBRARIES}
			      elise
			      ${X11_LIBRARIES} )
	
install(TARGETS SaisieQT DESTINATION ${Install_Dir})
