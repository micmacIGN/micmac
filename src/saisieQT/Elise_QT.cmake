# Interface Qt au sein d'Elise
include_directories(${SAISIE_DIR}/include_QT)
include_directories(${Qt5Widgets_INCLUDE_DIRS})
include_directories(${Qt5Core_INCLUDE_DIRS})
include_directories(${Qt5Concurrent_INCLUDE_DIRS})
include_directories(${Qt5OpenGL_INCLUDE_DIRS})
include_directories(${Qt5Xml_INCLUDE_DIRS})
include_directories(${Qt5Gui_INCLUDE_DIRS})

set(vmm_SRCS
	${SAISIE_DIR}/GLWidget.cpp
    ${SAISIE_DIR}/saisieQT_window.cpp
    ${SAISIE_DIR}/Cloud.cpp
    ${SAISIE_DIR}/Data.cpp
    ${SAISIE_DIR}/Engine.cpp
    ${SAISIE_DIR}/3DObject.cpp
    ${SAISIE_DIR}/cgldata.cpp
    ${SAISIE_DIR}/GLWidgetSet.cpp
    ${SAISIE_DIR}/MatrixManager.cpp
    ${SAISIE_DIR}/HistoryManager.cpp
    ${SAISIE_DIR}/ContextMenu.cpp
    ${SAISIE_DIR}/Settings.cpp
    ${SAISIE_DIR}/QT_interface_Elise.cpp
    ${SAISIE_DIR}/Tree.cpp
    ${SAISIE_DIR}/mmglu.cpp
    ${SAISIE_DIR}/WorkbenchWidget.cpp
    ${SAISIE_DIR}/MipmapHandler.cpp
	${SAISIE_DIR}/GlExtensions.cpp
	)

set(HEADERS_nomoc
    ${SAISIE_DIR}/include_QT/Elise_QT.h
    ${SAISIE_DIR}/include_QT/HistoryManager.h
    ${SAISIE_DIR}/include_QT/MatrixManager.h
    ${SAISIE_DIR}/include_QT/Cloud.h
    ${SAISIE_DIR}/include_QT/saisieQT_main.h
    ${SAISIE_DIR}/include_QT/Data.h
    ${SAISIE_DIR}/include_QT/SaisieGlsl.glsl
    ${SAISIE_DIR}/include_QT/Engine.h
    ${SAISIE_DIR}/include_QT/GLWidgetSet.h
    ${SAISIE_DIR}/include_QT/3DObject.h
    ${SAISIE_DIR}/include_QT/cgldata.h
	${SAISIE_DIR}/include_QT/mmglu.h
	)

set(HEADERS_tomoc
	${SAISIE_DIR}/include_QT/GLWidget.h
	${SAISIE_DIR}/include_QT/saisieQT_window.h
	${SAISIE_DIR}/include_QT/ContextMenu.h
	${SAISIE_DIR}/include_QT/Settings.h
	${SAISIE_DIR}/include_QT/QT_interface_Elise.h
	${SAISIE_DIR}/include_QT/Tree.h
	${SAISIE_DIR}/include_QT/WorkbenchWidget.h
	${Uti_Headers_ToMoc}
	)

set(ui_toWrap
	${SAISIE_DIR}/ui/saisieQT_window.ui
	${SAISIE_DIR}/ui/Settings.ui
	${SAISIE_DIR}/ui/Help.ui
    ${SAISIE_DIR}/ui/WorkbenchWidget.ui
)

set(FILES_TO_TRANSLATE ${FILES_TO_TRANSLATE} ${vmm_SRCS} ${ui_toWrap} ${HEADERS_nomoc} ${HEADERS_Tomoc})

qt5_wrap_cpp(HEADERS_moced ${HEADERS_tomoc})
set(qt_ressource_files "${SAISIE_DIR}/icones/icones.qrc")
qt5_add_resources(RC_SRCS ${qt_ressource_files})
qt5_wrap_ui(saisie_ui ${ui_toWrap})

if(WIN32)
	add_definitions(-DELISE_windows)
endif()

add_definitions(${QT_DEFINITIONS})

#~ for ui generated cpp files
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(QT_ALLFILES ${vmm_SRCS} ${RC_SRCS} ${HEADERS_moced} ${HEADERS_nomoc} ${saisie_ui})

source_group(QT\\ui FILES ${saisie_ui})
source_group(QT\\include FILES ${HEADERS_nomoc})
source_group(QT\\include FILES ${HEADERS_tomoc})
source_group(QT\\src FILES ${RC_SRCS})
source_group(QT\\src FILES ${vmm_SRCS})