# Interface Qt au sein d'Elise

if ( NOT ${qt_version} EQUAL 0 )

#    INCLUDE_DIRECTORIES(${SAISIE_DIR}/include_QT)


INCLUDE_DIRECTORIES(${SAISIE_DIR}/include_QT)
#INCLUDE_DIRECTORIES(${SAISIE_DIR})

    set(vmm_SRCS    ${SAISIE_DIR}/GLWidget.cpp
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

if ( ${qt_version} EQUAL 4)
    set(vmm_SRCS ${vmm_SRCS} ${SAISIE_DIR}/gl_core_2_1.c)
endif()

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
       ${SAISIE_DIR}/include_QT/mmglu.h)

	set(HEADERS_tomoc
		${SAISIE_DIR}/include_QT/GLWidget.h
		${SAISIE_DIR}/include_QT/saisieQT_window.h
		${SAISIE_DIR}/include_QT/ContextMenu.h
		${SAISIE_DIR}/include_QT/Settings.h
		${SAISIE_DIR}/include_QT/QT_interface_Elise.h
		${SAISIE_DIR}/include_QT/Tree.h
		${SAISIE_DIR}/include_QT/WorkbenchWidget.h
		${Uti_Headers_ToMoc})
	set(HEADERS_tomoc ${HEADERS_tomoc} PARENT_SCOPE)

	set(ui_toWrap
		${SAISIE_DIR}/ui/saisieQT_window.ui
		${SAISIE_DIR}/ui/Settings.ui
		${SAISIE_DIR}/ui/Help.ui
		${SAISIE_DIR}/ui/WorkbenchWidget.ui)
	set(ui_toWrap ${ui_toWrap} PARENT_SCOPE)

   set (FILES_TO_TRANSLATE ${FILES_TO_TRANSLATE} ${vmm_SRCS} ${ui_toWrap} ${HEADERS_nomoc} ${HEADERS_Tomoc})

	if ( ${qt_version} EQUAL 5 )
		QT5_WRAP_CPP(HEADERS_moced ${HEADERS_tomoc})
		set(qt_ressource_files "${SAISIE_DIR}/icones/icones.qrc")
		QT5_ADD_RESOURCES(RC_SRCS ${qt_ressource_files})
		qt5_wrap_ui(saisie_ui ${ui_toWrap})

		if ( WIN32 )
			add_definitions(-DELISE_windows)
		ENDIF()

		if( Qt5Core_FOUND )
			include_directories(${Qt5Widgets_INCLUDE_DIRS})
			include_directories(${Qt5Core_INCLUDE_DIRS})
			include_directories(${Qt5Concurrent_INCLUDE_DIRS})
			include_directories(${Qt5OpenGL_INCLUDE_DIRS})
			include_directories(${Qt5Xml_INCLUDE_DIRS})
			include_directories(${Qt5Gui_INCLUDE_DIRS})

			# Use the compile definitions defined in the Qt 5 Widgets module
			add_definitions(${Qt5Widgets_DEFINITIONS})
		endif()
	elseif ( ${qt_version} EQUAL 4 )
		QT4_ADD_RESOURCES( RC_SRCS ${SAISIE_DIR}/icones/icones.qrc )

		QT4_WRAP_UI(saisie_ui ${ui_toWrap})
		QT4_WRAP_CPP(HEADERS_moced ${HEADERS_tomoc})

		INCLUDE(${QT_USE_FILE})
	endif()

	#~ for ui generated cpp files
	INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

	set(QT_ALLFILES ${vmm_SRCS} ${RC_SRCS} ${HEADERS_moced} ${HEADERS_nomoc} ${saisie_ui})
	set(elise_qt_files ${vmm_SRCS} PARENT_SCOPE)
	set(HEADERS_tomoc ${HEADERS_tomoc} PARENT_SCOPE)
	set(saisie_ui ${saisie_ui} PARENT_SCOPE)
	set(qt_ressource_files ${qt_ressource_files} PARENT_SCOPE)
	set(saisieQT_SRCS ${saisieQT_SRCS} PARENT_SCOPE)

   SOURCE_GROUP(QT\\ui FILES ${saisie_ui})
   SOURCE_GROUP(QT\\include FILES ${HEADERS_nomoc})
   SOURCE_GROUP(QT\\include FILES ${HEADERS_tomoc})
   SOURCE_GROUP(QT\\src FILES ${RC_SRCS})
   SOURCE_GROUP(QT\\src FILES ${vmm_SRCS})
endif()
