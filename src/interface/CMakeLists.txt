cmake_minimum_required(VERSION 2.8)

ADD_SUBDIRECTORY(test_ISA0)

FIND_PACKAGE(Qt4 REQUIRED)

set(interface_directory ${PROJECT_SOURCE_DIR}/src/interface)

if (MSVC)
	set(interfaceMicmac_LIBS OpenGL32 GLU32)
ENDIF()

SET(interfaceMicmac_SOURCES ${interface_directory}/appliThread.cpp
                            ${interface_directory}/assistant.cpp
                            ${interface_directory}/drawMask.cpp
                            ${interface_directory}/interface.cpp
                            ${interface_directory}/interfApero.cpp
                            ${interface_directory}/interfConversion.cpp
                            ${interface_directory}/interfMicmac.cpp
                            ${interface_directory}/interfPastis.cpp
                            ${interface_directory}/main.cpp
                            ${interface_directory}/progressBar.cpp
                            ${interface_directory}/readwrite.cpp
                            ${interface_directory}/vueChantier.cpp)
SET(interfaceMicmac_HEADERS ${interface_directory}/appliThread.h
                            ${interface_directory}/assistant.h
                            ${interface_directory}/drawMask.h
                            ${interface_directory}/interface.h
                            ${interface_directory}/interfApero.h
                            ${interface_directory}/interfConversion.h
                            ${interface_directory}/interfMicmac.h
                            ${interface_directory}/interfPastis.h
                            ${interface_directory}/progressBar.h
                            ${interface_directory}/vueChantier.h)

SET(QT_USE_QTOPENGL TRUE)

QT4_WRAP_CPP(interfaceMicmac_HEADERS_MOC ${interfaceMicmac_HEADERS})
QT4_ADD_RESOURCES(interfaceMicmac_RESOURCES_RCC ressources.qrc)

INCLUDE(${QT_USE_FILE})
ADD_DEFINITIONS(${QT_DEFINITIONS})

ADD_EXECUTABLE(interfaceMicmac ${interfaceMicmac_SOURCES}
                               ${interfaceMicmac_HEADERS_MOC})
TARGET_LINK_LIBRARIES(interfaceMicmac ${OPENGL_LIBRARIES} ${GLU_LIBRARY}
									  ${X11_LIBRARIES} X11
									  ${QT_LIBRARIES}
                                      elise
                                      ${interfaceMicmac_LIBS})
INSTALL(TARGETS interfaceMicmac RUNTIME DESTINATION ${Install_Dir})
