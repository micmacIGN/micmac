cmake_minimum_required(VERSION 2.8.1)
project(mm3dSat)

IF ( ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" )
    add_definitions(-DELISE_Darwin)
endif()

set(interf_SRCS mm3dSat_main.cpp
				mm3dSat_mainwindow.cpp)
		
set(interf_HEADERS mm3dSat_mainwindow.h)
   
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

      #~ SET(interf_RCS ./icones/icones.qrc)

      #~ QT5_ADD_RESOURCES(interf_RC_SRCS ${interf_RCS} )

      qt5_wrap_ui(interf_ui mm3dSat_mainwindow.ui)

      add_definitions(-DTWEAK)
      if ( WIN32 )
	 add_definitions(-DELISE_windows)
      ENDIF()


      include_directories( ${Qt5_INCLUDES}
			   ${OPENGL_INCLUDE_DIR}
			   ${PROJECT_SOURCE_DIR}
			   ${PROJECT_BINARY_DIR} )

      IF(WIN32)
	      add_executable(MM3dSat WIN32 ${interf_SRCS} ${interf_ui})
      else()
	      add_executable(MM3dSat ${interf_SRCS} ${interf_ui})
      endif()
      
      qt5_use_modules(MM3dSat Widgets Core Gui Xml Concurrent OpenGL)

      target_link_libraries(MM3dSat #${OPENGL_LIBRARIES}
				     elise
				     ${X11_LIBRARIES} )
   elseif ( ${qt_version} EQUAL 4 )
      QT4_WRAP_UI(interf_ui mainwindow.ui)
      QT4_WRAP_CPP(interf_HEADERS ${interf_HEADERS})
      INCLUDE(${QT_USE_FILE})
      include_directories( ${PROJECT_BINARY_DIR} )
      add_executable(MM3dSat ${interf_SRCS}  ${interf_ui}  ${interf_HEADERS})
      target_link_libraries(MM3dSat #${OPENGL_LIBRARIES}
                                     ${QT_LIBRARIES}
				     elise
				     ${X11_LIBRARIES} )      
   endif()


   install(TARGETS MM3dSat DESTINATION ${Install_Dir}) 
endif()
