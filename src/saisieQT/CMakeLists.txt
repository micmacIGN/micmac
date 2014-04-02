cmake_minimum_required(VERSION 2.8.1)
project(saisieQT)

IF ( ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" )
    add_definitions(-DELISE_Darwin)
endif()

set(saisie_SRCS saisieQT_main.cpp
                saisieMasqQT_main.cpp
                saisieAppuisInitQT_main.cpp
                GLWidget.cpp
                mainwindow.cpp
                Cloud.cpp
                Data.cpp
                Engine.cpp
                3DObject.cpp
                GLWidgetSet.cpp
                MatrixManager.cpp
                HistoryManager.cpp
                ContextMenu.cpp
                Settings.cpp
                QT_interface_Elise.cpp
                Tree.cpp)

set( saisie_HEADERS_nomoc
   HistoryManager.h
   MatrixManager.h
   Cloud.h
   saisieQT_main.h
   Data.h
   3DTools.h
   #SaisieGlsl.glsl
   Engine.h
   GLWidgetSet.h
   3DObject.h )

set( saisie_HEADERS_tomoc
   GLWidget.h
   mainwindow.h
   ContextMenu.h
   Settings.h
   QT_interface_Elise.h
   Tree.h)

if ( NOT ${qt_version} EQUAL 0 )
   if ( ${qt_version} EQUAL 5 )
      if ( WIN32 )
              #cmake_policy(SET CMP0020 NEW)
      ENDIF()

      if ( NOT ${OPENGL_GLU_FOUND} )
              message( FATAL_ERROR "GLU library could not be found " )
              return( -1 )
      endif()

      set(CMAKE_AUTOMOC ON)

      SET(Saisie_RCS ./icones/icones.qrc)

      QT5_ADD_RESOURCES( Saisie_RC_SRCS ${Saisie_RCS} )

      qt5_wrap_ui(saisie_ui mainwindow.ui Settings.ui)

      add_definitions(-DTWEAK)
      if ( WIN32 )
         add_definitions(-DELISE_windows)
      ENDIF()


      include_directories( ${Qt5_INCLUDES}
                           ${OPENGL_INCLUDE_DIR}
                           ${PROJECT_SOURCE_DIR}
                           ${PROJECT_BINARY_DIR} )

      IF(WIN32)
              add_executable(SaisieQT WIN32 ${saisie_SRCS} ${Saisie_RC_SRCS} ${saisie_ui} )
      else()
              add_executable(SaisieQT ${saisie_SRCS} ${Saisie_RC_SRCS} ${saisie_ui} )
      endif()

      qt5_use_modules(SaisieQT Widgets Core Gui Xml Concurrent OpenGL)

      target_link_libraries(SaisieQT ${OPENGL_LIBRARIES}
                                     elise
                                     ${X11_LIBRARIES} )
   elseif ( ${qt_version} EQUAL 4 )
      QT4_WRAP_UI(saisie_ui mainwindow.ui Settings.ui)
      QT4_WRAP_CPP(saisie_HEADERS_tomoc_MOC ${saisie_HEADERS_tomoc})
      INCLUDE(${QT_USE_FILE})
      include_directories( ${PROJECT_BINARY_DIR} )
      add_executable(SaisieQT ${saisie_SRCS} ${Saisie_RC_SRCS} ${saisie_ui} ${saisie_HEADERS_tomoc_MOC} ${saisie_HEADERS_nomoc})
      target_link_libraries(SaisieQT ${OPENGL_LIBRARIES}
                                     ${QT_LIBRARIES}
                                     elise
                                     ${X11_LIBRARIES} )
   endif()

   install(TARGETS SaisieQT DESTINATION ${Install_Dir})
endif()
