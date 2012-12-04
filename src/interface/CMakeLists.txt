cmake_minimum_required(VERSION 2.8)

PROJECT(interfaceMicmac)
FIND_PACKAGE(Qt4 REQUIRED)

if (MSVC)
	set(interfaceMicmac_LIBS OpenGL32 GLU32)
ENDIF()

if (MINGW)
	add_definitions(-D__CYGWIN__)
	add_definitions(-DNO_X11)
#	set( WINSOCK2_LIBRAIRIE ws2_32 )
endif()

IF (CMAKE_COMPILER_IS_GNUCC)
	ADD_DEFINITIONS(-O3)
ENDIF()

link_directories (D:/dev/culture3d/build_vc/src/Release)

CONFIGURE_FILE (
  "../../include/general/MM_InstalDir.h.in"
  "${PROJECT_BINARY_DIR}/include/general/MM_InstalDir.h"
  )

INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR}/include)
INCLUDE_DIRECTORIES(../../include)

SET(interfaceMicmac_SOURCES appliThread.cpp
                            assistant.cpp
                            drawMask.cpp
                            interface.cpp
                            interfApero.cpp
                            interfConversion.cpp
                            interfMicmac.cpp
                            interfPastis.cpp
                            main.cpp
                            progressBar.cpp
                            readwrite.cpp
                            vueChantier.cpp)
SET(interfaceMicmac_HEADERS appliThread.h
                            assistant.h
                            drawMask.h
                            interface.h
                            interfApero.h
                            interfConversion.h
                            interfMicmac.h
                            interfPastis.h
                            progressBar.h
                            vueChantier.h)

SET(QT_USE_QTOPENGL TRUE)

QT4_WRAP_CPP(interfaceMicmac_HEADERS_MOC ${interfaceMicmac_HEADERS})
#QT4_WRAP_UI(interfaceMicmac_FORMS_HEADERS interface.ui)
QT4_ADD_RESOURCES(interfaceMicmac_RESOURCES_RCC ressources.qrc)

message(qt use file = ${QT_USE_FILE})
INCLUDE(${QT_USE_FILE})
message(qt defintions = ${QT_DEFINITIONS})
ADD_DEFINITIONS(${QT_DEFINITIONS})

ADD_EXECUTABLE(interfaceMicmac ${interfaceMicmac_SOURCES}
                               ${interfaceMicmac_HEADERS_MOC})
TARGET_LINK_LIBRARIES(interfaceMicmac ${QT_LIBRARIES}
                                      elise
                                      ${interfaceMicmac_LIBS})
