cmake_minimum_required(VERSION 2.8)
project(saisie)

if ( NOT ${OPENGL_GLU_FOUND} )
	message( FATAL_ERROR "GLU library could not be found " )
	return( -1 )
endif()

set(CMAKE_AUTOMOC ON)

set(SRCS main.cpp
		 GLWidget.cpp
         mainwindow.cpp
         Cloud.cpp
		 Data.cpp
		 Engine.cpp
		 ../poisson/plyfile.cpp)

# Check if environment variable QTDIR is set.
if (DEFINED ENV{QTDIR})
    set(CMAKE_PREFIX_PATH $ENV{QTDIR} ${CMAKE_PREFIX_PATH})
else ()
    message (FATAL_ERROR "ERROR: Environment variable QTDIR is not set. Please locate your Qt folder.")
	return( -1 )
endif ()
		 
		 		 
if ( WIN32 )			
	set (CMAKE_PREFIX_PATH "C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v7.0A\\Lib")
ENDIF()



		 
set(qt_lib_dir "$ENV{QTDIR}\\lib\\cmake\\")

set (Qt5_DIR "${qt_lib_dir}Qt5")
set (Qt5Core_DIR "${qt_lib_dir}Qt5Core")
set (Qt5Gui_DIR "${qt_lib_dir}Qt5Gui")
set (Qt5Widgets_DIR "${qt_lib_dir}Qt5Widgets")
set (Qt5Xml_DIR "${qt_lib_dir}Qt5Xml")
set (Qt5Concurrent_DIR "${qt_lib_dir}Qt5Concurrent")
set (Qt5_INCLUDES "${qt_dir}include\\")
set (QT_QTXML_INCLUDE_DIR "${qt_dir}include\\Qtxml\\")
set (QT_QTCONCURRENT_INCLUDE_DIR "${qt_dir}include\\QtConcurrent\\")
set (QT_QTOPENGL_INCLUDE_DIR "${qt_dir}include\\QtOpenGL\\")
set (Qt5Core_LIBRARIES "${qt_lib_dir}Qt5Core")
set (Qt5Gui_LIBRARIES "${qt_lib_dir}Qt5Gui")
set (Qt5Widgets_LIBRARIES "${qt_lib_dir}Qt5Widgets")
set (Qt5XML_LIBRARIES "${qt_lib_dir}Qt5Xml")
set (Qt5CONCURRENT_LIBRARIES "${qt_lib_dir}Qt5Concurrent")

SET(Saisie_RCS ./icones/icones.qrc)

set( Install_Dir ${PROJECT_SOURCE_DIR}/bin )

find_package(Qt5Core REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Qt5Xml REQUIRED)

message("OpenGL libraries : "${OPENGL_LIBRARIES})

if ( NOT WIN32 )
    FIND_PACKAGE(X11 REQUIRED)
    message("X11 libraries : "${X11_LIBRARIES})
ENDIF()

IF(QT_WRAP_CPP)
     IF (QT_WRAP_UI)
        qt5_wrap_ui(saisie_HEADERS  mainwindow.ui)
        qt5_wrap_cpp(saisie_HEADERS_MOC ${saisie_HEADERS})
    ENDIF (QT_WRAP_UI)
ENDIF(QT_WRAP_CPP)

list( APPEND saisie_HEADERS GLWidget.h
                            mainwindow.h
                            Cloud.h
							Engine.h
							Data.h
							3DTools.h )

QT5_ADD_RESOURCES( Saisie_RC_SRCS ${Saisie_RCS} )

add_definitions(-DTWEAK)
if ( WIN32 )
  add_definitions(-DELISE_windows)
ENDIF()

#set(CMAKE_CXX_FLAGS "-Wno-dev")

SET(QT_USE_QTOPENGL TRUE)
include_directories(${Qt5_INCLUDES} ${QT_QTXML_INCLUDE_DIR} ${QT_QTCONCURRENT_INCLUDE_DIR} ${QT_QTOPENGL_INCLUDE_DIR} ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR} )
add_executable(saisie ${SRCS} ${Saisie_RC_SRCS} ${saisie_HEADERS_MOC} ${saisie_HEADERS})
target_link_libraries(saisie ${OPENGL_LIBRARIES} ${Qt5Core_LIBRARIES} ${Qt5Gui_LIBRARIES} ${Qt5Widgets_LIBRARIES} elise ${X11_LIBRARIES})
link_directories(${Qt5XML_LIBRARIES} ${Qt5CONCURRENT_LIBRARIES})
install(TARGETS saisie DESTINATION Install_Dir)