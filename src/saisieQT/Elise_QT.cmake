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
                    ${SAISIE_DIR}/mmglu.cpp)

    set( HEADERS_nomoc
       ${SAISIE_DIR}/include_QT/HistoryManager.h
       ${SAISIE_DIR}/include_QT/MatrixManager.h
       ${SAISIE_DIR}/include_QT/Cloud.h
       ${SAISIE_DIR}/include_QT/saisieQT_main.h
       ${SAISIE_DIR}/include_QT/Data.h
       #SaisieGlsl.glsl
       ${SAISIE_DIR}/include_QT/Engine.h
       ${SAISIE_DIR}/include_QT/GLWidgetSet.h
       ${SAISIE_DIR}/include_QT/3DObject.h
       ${SAISIE_DIR}/include_QT/cgldata.h
       ${SAISIE_DIR}/include_QT/mmglu.h )

    set( HEADERS_tomoc
       ${SAISIE_DIR}/include_QT/GLWidget.h
       ${SAISIE_DIR}/include_QT/saisieQT_window.h
       ${SAISIE_DIR}/include_QT/ContextMenu.h
       ${SAISIE_DIR}/include_QT/Settings.h
       ${SAISIE_DIR}/include_QT/QT_interface_Elise.h
       ${SAISIE_DIR}/include_QT/Tree.h
    )

    set (ui_toWrap
     ${SAISIE_DIR}/ui/saisieQT_window.ui
     ${SAISIE_DIR}/ui/Settings.ui
     ${SAISIE_DIR}/ui/Help.ui
    )

   set (FILES_TO_TRANSLATE ${FILES_TO_TRANSLATE} ${vmm_SRCS} ${ui_toWrap} ${HEADERS_nomoc} ${HEADERS_Tomoc})

   if ( ${qt_version} EQUAL 5 )

      set(CMAKE_INCLUDE_CURRENT_DIR ON)

      QT5_WRAP_CPP(HEADERS_moced ${HEADERS_tomoc} )

      QT5_ADD_RESOURCES( RC_SRCS ${SAISIE_DIR}/icones/icones.qrc )

      qt5_wrap_ui(saisie_ui ${ui_toWrap})

      add_definitions(-DTWEAK)

      if ( WIN32 )
         add_definitions(-DELISE_windows)
      ENDIF()

      find_package(Qt5Core REQUIRED)

      if( Qt5Core_FOUND )

        include_directories(${Qt5Widgets_INCLUDE_DIRS})
        include_directories(${Qt5Core_INCLUDE_DIRS})
        include_directories(${Qt5Concurrent_INCLUDE_DIRS})
        include_directories(${Qt5OpenGL_INCLUDE_DIRS})
        include_directories(${Qt5Xml_INCLUDE_DIRS})
        include_directories(${Qt5Gui_INCLUDE_DIRS})

         # Use the compile definitions defined in the Qt 5 Widgets module
         add_definitions(${Qt5Widgets_DEFINITIONS})

        # Add compiler flags for building executables (-fPIE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")

      endif()

   elseif ( ${qt_version} EQUAL 4 )

      QT4_ADD_RESOURCES( RC_SRCS ${SAISIE_DIR}/icones/icones.qrc )

      QT4_WRAP_UI(saisie_ui ${ui_toWrap})
      QT4_WRAP_CPP(HEADERS_moced ${HEADERS_tomoc} )

      INCLUDE(${QT_USE_FILE})
      include_directories( ${PROJECT_BINARY_DIR} )
      INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
   endif()

endif()
