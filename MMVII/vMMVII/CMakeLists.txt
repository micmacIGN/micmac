cmake_minimum_required(VERSION 3.5)

project(vMMVII VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif(CCACHE_FOUND)


find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

set(PROJECT_SOURCES
    actionbox.h
    actionbox.cpp
    actionbox.ui
    cmdconfigurewidget.h
    cmdconfigurewidget.cpp
    cmdselectwidget.h
    cmdselectwidget.cpp
    cmdSelect.ui
    commandspec.h
    commandspec.cpp
    ellipsiscombobox.h
    ellipsiscombobox.cpp
    global.h
    global.cpp
    inputwidget.h
    inputwidget.cpp
    main.cpp
    mainwindow.h
    mainwindow.cpp
    processwidget.h
    processwidget.cpp
    processwidget.ui
    settings.h
    settings.cpp
    settings.ui
    spinboxdefault.h
    spinboxdefault.cpp
    workingdirwidget.h
    workingdirwidget.cpp
    workingdirwidget.ui
    TODO
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(vMMVII MANUAL_FINALIZATION ${PROJECT_SOURCES})
else()
    add_executable(vMMVII ${PROJECT_SOURCES})
endif()

target_link_libraries(vMMVII PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)

if(${QT_VERSION} VERSION_LESS 6.1.0)
    set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.example.vMMVII)
endif()
set_target_properties(vMMVII PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

# CM: Needed for compiling ui_* with promotted widgets
 target_include_directories(vMMVII PRIVATE ${PROJECT_SOURCE_DIR})
 
if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(vMMVII)
endif()
