cmake_minimum_required(VERSION 2.8.1)
project(saisieQT)

IF ( ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" )
    add_definitions(-DELISE_Darwin)
endif()

set(saisieQT_SRCS saisieQT_main.cpp
                  saisieMasqQT_main.cpp
                  saisieAppuisInitQT_main.cpp
                  saisieAppuisPredicQT_main.cpp
                  saisieBascQT_main.cpp
                  saisieBoxQT_main.cpp)

#message ("saisie srcs: " ${saisieQT_SRCS} )

if ( NOT ${qt_version} EQUAL 0 )
   if ( ${qt_version} EQUAL 5 AND WIN32)
      add_executable(SaisieQT WIN32 ${saisieQT_SRCS})
   else()
      add_executable(SaisieQT ${saisieQT_SRCS})
   endif()

   target_link_libraries(SaisieQT elise ${X11_LIBRARIES})

   install(TARGETS SaisieQT DESTINATION ${Install_Dir})
endif()
