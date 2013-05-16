cmake_minimum_required(VERSION 2.8)
project(saisie3D)

FIND_PACKAGE(Qt4 REQUIRED QtCore QtGui QtOpenGL)
message("librairies Qt : "${QT_LIBRARIES})
FIND_PACKAGE(OpenGL REQUIRED)
message("Libraries OpenGL : "${OPENGL_LIBRARIES})
if ( NOT WIN32 )
    FIND_PACKAGE(X11 REQUIRED)
    message("librairies X11 : "${X11_LIBRARIES})
ENDIF()

include_directories(${CMAKE_CURRENT_BINARY_DIR})
message("include directory for ui files: "${CMAKE_CURRENT_BINARY_DIR})
set( Install_Dir ${PROJECT_SOURCE_DIR}/bin )

set( saisie3D_SOURCES main.cpp
                    GLWidget.cpp
                    mainwindow.cpp
                    Cloud.cpp)
QT4_WRAP_UI(saisie3D_HEADERS mainwindow.ui )
message("ui generated header : "${saisie3D_HEADERS})
list( APPEND saisie3D_HEADERS GLWidget.h
                            mainwindow.h
                            Cloud.h)

SET(QT_USE_QTOPENGL TRUE)

QT4_WRAP_CPP(saisie3D_HEADERS_MOC ${saisie3D_HEADERS})
#QT4_ADD_RESOURCES(interfaceMicmac_RESOURCES_RCC ressources.qrc)

INCLUDE(${QT_USE_FILE})
ADD_DEFINITIONS(${QT_DEFINITIONS})

ADD_EXECUTABLE(saisie3D ${saisie3D_SOURCES}
                      ${saisie3D_HEADERS_MOC})
TARGET_LINK_LIBRARIES(saisie3D ${OPENGL_LIBRARIES}
                             ${X11_LIBRARIES}
                             ${QT_LIBRARIES})

INSTALL(TARGETS saisie3D RUNTIME DESTINATION ${Install_Dir})
